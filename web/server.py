"""FastAPI server for the Stable Audio Remix web app.

Loads StableAudioModel once at startup and exposes a single /api/remix endpoint
that takes an input audio file + prompt + generation knobs and returns a WAV.

Static assets (the SPA, demucs WASM, ONNX model) are served from web/static/.
"""

import asyncio
import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from stable_audio_3 import StableAudioModel

STATIC_DIR = Path(__file__).parent / "static"
MODEL_NAME = os.environ.get("SAO_MODEL", "medium")

ANALYSIS_URL = os.environ.get(
    "ANALYSIS_URL", "https://ai.pipeline.corpus.music/analysis/analyze"
)
# Required: bearer token for the music-intelligence pipeline. The /api/analyze
# endpoint returns 503 if this isn't set, which the frontend treats as a soft
# failure (BPM/Key just stay empty — the rest of the remix flow still works).
ANALYSIS_TOKEN = os.environ.get("ANALYSIS_TOKEN", "").strip()

state: dict = {"model": None}

# Per-job progress queues for SSE. Created in /api/remix, consumed by
# /api/remix/events/{job_id}. asyncio.Queue is unbounded, so callbacks can
# push from a worker thread (via call_soon_threadsafe) without blocking.
JOBS: dict[str, dict] = {}

# Single-tenant GPU lock. We're serving from one GPU, so only one render at a
# time — second arrivals get 503 with a friendly retry message rather than
# silently queuing for unbounded wait.
GPU_LOCK = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[remix] Loading Stable Audio model '{MODEL_NAME}'…")
    model = StableAudioModel.from_pretrained(MODEL_NAME)
    state["model"] = model
    # Max output duration is set by the trained sample_size of the model
    # (medium: 16777216 / 44100 ≈ 380.4s; small-*: 5292032 / 44100 ≈ 120s).
    state["max_seconds"] = model.model_config["sample_size"] / model.model.sample_rate
    print(
        f"[remix] Model ready (sample_rate={model.model.sample_rate}, "
        f"max_seconds={state['max_seconds']:.1f})."
    )
    yield


class CrossOriginIsolationMiddleware(BaseHTTPMiddleware):
    """Required for SharedArrayBuffer / ORT-web threaded WASM in the browser."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Resource-Policy"] = "cross-origin"
        return response


app = FastAPI(lifespan=lifespan, title="Stable Audio Remix")
app.add_middleware(CrossOriginIsolationMiddleware)


@app.post("/api/analyze")
async def analyze(audio: UploadFile = File(...)):
    """Proxy to the music-intelligence pipeline. Returns just bpm + key."""
    if not ANALYSIS_TOKEN:
        raise HTTPException(
            503,
            "Analysis pipeline not configured — set the ANALYSIS_TOKEN env var "
            "on the server to enable BPM/Key auto-detection.",
        )
    raw = await audio.read()
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            r = await client.post(
                ANALYSIS_URL,
                headers={"Authorization": f"Bearer {ANALYSIS_TOKEN}"},
                files={
                    "file": (
                        audio.filename or "track",
                        raw,
                        audio.content_type or "application/octet-stream",
                    ),
                },
                data={"priority": "high"},
            )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"analysis pipeline unreachable: {e}")
    if r.status_code != 200:
        raise HTTPException(r.status_code, f"analysis failed: {r.text[:200]}")
    data = r.json()
    return {"bpm": data.get("bpm"), "key": data.get("key")}


@app.get("/api/health")
async def health():
    return {
        "ready": state["model"] is not None,
        "model": MODEL_NAME,
        "sample_rate": state["model"].model.sample_rate if state["model"] else None,
    }


@app.post("/api/remix")
async def remix(
    audio: UploadFile = File(...),
    prompt: str = Form(...),
    init_noise_level: float = Form(0.7),
    steps: int = Form(8),
    cfg_scale: float = Form(1.5),
    seed: int = Form(-1),
    job_id: str = Form(""),
):
    model = state["model"]
    if model is None:
        raise HTTPException(503, "Model still loading, try again in a moment")

    # Reject concurrent renders before reading the upload — this is cheap and
    # the user gets feedback immediately. Body header makes the reason machine-
    # readable on the client.
    if GPU_LOCK.locked():
        raise HTTPException(
            status_code=503,
            detail="Another user is currently generating. Please wait a few seconds and try again.",
            headers={"X-Busy-Reason": "gpu"},
        )

    raw = await audio.read()
    try:
        waveform, sr = torchaudio.load(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not decode audio: {e}")

    duration = min(waveform.shape[-1] / sr, state["max_seconds"])

    # If a job_id is supplied, expose step-by-step progress via the SSE endpoint.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue | None = None
    if job_id:
        queue = asyncio.Queue()
        JOBS[job_id] = {"queue": queue}

    total_steps = int(steps)

    def step_callback(info):
        if queue is None:
            return
        i = int(info.get("i", 0))
        msg = {"stage": "gpu", "step": i + 1, "total": total_steps}
        try:
            loop.call_soon_threadsafe(queue.put_nowait, msg)
        except RuntimeError:
            pass  # event loop closed mid-render

    # Re-check under the lock to avoid a thin TOCTOU window where two requests
    # both pass the locked() check above. The second would block here — return
    # 503 instead so the user can retry without hanging.
    if GPU_LOCK.locked():
        raise HTTPException(
            status_code=503,
            detail="Another user is currently generating. Please wait a few seconds and try again.",
            headers={"X-Busy-Reason": "gpu"},
        )
    async with GPU_LOCK:
        try:
            audio_out = await asyncio.to_thread(
                _generate,
                model, prompt, waveform, sr, duration,
                init_noise_level, steps, cfg_scale, seed, step_callback,
            )
        finally:
            if job_id:
                JOBS.pop(job_id, None)
                if queue is not None:
                    queue.put_nowait(None)  # sentinel: done

    np_out = audio_out.transpose(0, 1).contiguous().numpy()
    buf = io.BytesIO()
    sf.write(buf, np_out, model.model.sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="remix.wav"'},
    )


def _generate(model, prompt, waveform, sr, duration, noise, steps, cfg, seed, callback):
    out = model.generate(
        prompt=prompt,
        duration=duration,
        steps=int(steps),
        cfg_scale=float(cfg),
        seed=int(seed),
        init_audio=(int(sr), waveform),
        init_noise_level=float(noise),
        batch_size=1,
        callback=callback,
    )
    return out[0].cpu().float()


@app.get("/api/remix/events/{job_id}")
async def remix_events(job_id: str):
    # The POST handler creates the job entry early on; if the SSE hits before
    # that, give it up to ~5s to appear.
    for _ in range(50):
        if job_id in JOBS:
            break
        await asyncio.sleep(0.1)
    if job_id not in JOBS:
        raise HTTPException(404, "job not found")
    queue: asyncio.Queue = JOBS[job_id]["queue"]

    async def stream():
        # Tell client we're connected.
        yield "event: open\ndata: {}\n\n"
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if msg is None:
                yield "event: done\ndata: {}\n\n"
                return
            yield f"event: progress\ndata: {json.dumps(msg)}\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Static files (SPA + WASM + models) mounted last so /api/* routes win.
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
