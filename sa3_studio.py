#!/usr/bin/env python3
"""SA3 Studio — local drag-and-drop web UI for the variation harness.

Open in a browser, drop a WAV, get 5 variations to listen to. Calls
`sa3_variations.run_variations()` in-process and keeps a long-lived MLX
Pipeline resident across clicks (rebuilt only when the audio duration
changes, since the DiT's `_local_zeros_1` buffer is sized to T_lat).

Usage:
  optimized/mlx/.venv/bin/python sa3_studio.py
  # then open http://localhost:8765 (auto-opens by default)
"""
from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from email.parser import BytesParser
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SA3_ROOT = SCRIPT_DIR
RUNS_DIR = SCRIPT_DIR / "runs"

# Pipeline state shared across requests. The HTTP server is multi-threaded,
# but the MLX pipeline isn't thread-safe (single mx context), so we serialize
# generate() calls behind a lock. A generation crash takes down the studio —
# that's the tradeoff for not paying ~1.3s of model load on every click.
_pipeline_lock = threading.Lock()
_pipeline = None  # type: ignore[var-annotated]  # sa3_pipeline.Pipeline | None

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SA3 Studio</title>
  <style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      max-width: 760px; margin: 32px auto; padding: 0 20px;
      line-height: 1.5;
    }
    h1 { font-size: 24px; margin: 0 0 4px; }
    .sub { color: #888; font-size: 13px; margin-bottom: 24px; }
    .drop {
      border: 2px dashed #bbb; border-radius: 10px; padding: 40px 20px;
      text-align: center; transition: background 0.15s, border-color 0.15s;
      margin-bottom: 16px;
    }
    .drop.over { border-color: #4a90e2; background: rgba(74, 144, 226, 0.08); }
    .drop p { margin: 0; }
    .drop .filename { font-weight: 600; }
    .drop .meta { color: #888; font-size: 12px; margin-top: 4px; }
    .controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .controls label { display: inline-flex; align-items: center; gap: 6px; font-size: 13px; }
    .seg {
      display: inline-flex; border: 1px solid #bbb; border-radius: 6px; overflow: hidden;
      margin-right: 8px;
    }
    .seg-btn {
      padding: 6px 12px; background: transparent; color: inherit; border: none;
      font-size: 13px; font-weight: 500; cursor: pointer; border-radius: 0;
    }
    .seg-btn:not(:last-child) { border-right: 1px solid #bbb; }
    .seg-btn.active { background: #4a90e2; color: white; }
    .preset-hint { font-size: 12px; color: #888; margin: -4px 0 12px; }
    .controls label.app-only { display: none; }
    .controls.show-app label.app-only { display: inline-flex; }
    input[type="number"], input[type="text"] {
      padding: 6px 8px; border: 1px solid #ccc; border-radius: 6px;
      font-size: 13px; width: 90px;
    }
    input[type="text"] { width: 120px; }
    input[type="range"] { width: 110px; vertical-align: middle; }
    .noise-val { display: inline-block; width: 32px; font-variant-numeric: tabular-nums; color: #555; font-size: 12px; }
    button {
      padding: 8px 16px; border: none; border-radius: 6px;
      background: #4a90e2; color: white; font-weight: 600; cursor: pointer;
      font-size: 14px;
    }
    button:disabled { background: #aaa; cursor: not-allowed; }
    .status { color: #888; font-style: italic; font-size: 13px; min-height: 20px; margin: 8px 0; }
    .source { margin: 24px 0 12px; }
    .source h2, .results h2 { font-size: 14px; text-transform: uppercase; letter-spacing: 0.08em; color: #888; margin-bottom: 8px; }
    audio { width: 100%; }
    .variation {
      border-top: 1px solid rgba(128,128,128,0.25); padding: 12px 0;
    }
    .variation:last-child { border-bottom: 1px solid rgba(128,128,128,0.25); }
    .v-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 6px; }
    .v-title { font-size: 14px; }
    .v-num { display: inline-block; width: 22px; color: #888; font-variant-numeric: tabular-nums; }
    .v-steer { font-style: italic; color: #666; }
    .v-meta { font-size: 11px; color: #aaa; font-variant-numeric: tabular-nums; white-space: nowrap; }
    .badge {
      display: inline-block; padding: 1px 6px; border-radius: 3px;
      font-size: 10px; font-weight: 600; text-transform: uppercase;
      letter-spacing: 0.05em; vertical-align: middle;
    }
    .badge.a2a { background: #e8f0fe; color: #1a73e8; }
    .badge.inpaint { background: #fef3e6; color: #d97706; }
    @media (prefers-color-scheme: dark) {
      .badge.a2a { background: rgba(74, 144, 226, 0.2); color: #6ba8e8; }
      .badge.inpaint { background: rgba(217, 119, 6, 0.2); color: #e3a155; }
    }
    .err { color: #c0392b; white-space: pre-wrap; font-family: ui-monospace, monospace; font-size: 12px; }
  </style>
</head>
<body>
  <h1>SA3 Studio</h1>
  <div class="sub">Drop a WAV, get 5 variations. Apple-Silicon-native via MLX.</div>

  <div class="drop" id="drop">
    <p><strong>Drop a WAV here</strong>, or <label style="cursor:pointer; text-decoration: underline;"><input type="file" id="file" accept="audio/*" style="display:none">browse</label></p>
  </div>

  <div class="controls">
    <div class="seg" id="preset">
      <button type="button" class="seg-btn active" data-preset="free">Free variation</button>
      <button type="button" class="seg-btn" data-preset="app">Preserve sound</button>
    </div>
    <label class="app-only">BPM <input type="number" id="bpm" step="0.1" placeholder="optional"></label>
    <label class="app-only">Key <input type="text" id="key" placeholder="optional"></label>
    <label title="Noise level for a2a candidates. 0.45 = default. Lower preserves timbre, higher drifts further.">Noise <input type="range" id="noise" min="0.10" max="1.00" step="0.01" value="0.45"><span class="noise-val" id="noise-val">0.45</span></label>
    <button id="gen" disabled>Generate</button>
  </div>
  <div class="preset-hint" id="preset-hint">5 unconditional a2a samples at n=0.45. Preserves harmonic context, lets timbre/instrument drift.</div>

  <div class="status" id="status"></div>

  <div id="output"></div>

  <script>
    let selectedFile = null;
    let currentPreset = 'free';
    const drop = document.getElementById('drop');
    const fileInput = document.getElementById('file');
    const genBtn = document.getElementById('gen');
    const statusEl = document.getElementById('status');
    const output = document.getElementById('output');
    const presetEl = document.getElementById('preset');
    const presetHint = document.getElementById('preset-hint');
    const noiseEl = document.getElementById('noise');
    const noiseValEl = document.getElementById('noise-val');

    function currentNoise() { return Number(noiseEl.value).toFixed(2); }
    function updateHint() {
      const n = currentNoise();
      if (currentPreset === 'free') {
        presetHint.textContent = `5 unconditional a2a samples at n=${n}. Preserves harmonic context, lets timbre/instrument drift.`;
      } else {
        presetHint.textContent = `3 a2a (n=${n}, cfg=4) + 2 inpaint (n=0.85, cfg=4) with per-candidate steers. Keeps timbre when n is low.`;
      }
    }
    noiseEl.addEventListener('input', () => {
      noiseValEl.textContent = currentNoise();
      updateHint();
    });

    const controlsEl = document.querySelector('.controls');
    function applyPresetVisibility() {
      controlsEl.classList.toggle('show-app', currentPreset === 'app');
    }
    presetEl.addEventListener('click', e => {
      const btn = e.target.closest('.seg-btn');
      if (!btn) return;
      currentPreset = btn.dataset.preset;
      [...presetEl.querySelectorAll('.seg-btn')].forEach(b => b.classList.toggle('active', b === btn));
      applyPresetVisibility();
      updateHint();
    });
    applyPresetVisibility();
    updateHint();

    function setFile(f) {
      selectedFile = f;
      const parsed = parseBpmKeyFromFilename(f.name);
      const hints = [];
      if (parsed.bpm !== null) {
        document.getElementById('bpm').value = parsed.bpm;
        hints.push(`BPM ${parsed.bpm}`);
      }
      if (parsed.key !== null) {
        document.getElementById('key').value = parsed.key;
        hints.push(`Key ${parsed.key}`);
      }
      const hintTxt = hints.length ? ` · auto-detected ${hints.join(', ')}` : '';
      drop.innerHTML =
        `<p class="filename">${escapeHtml(f.name)}</p>` +
        `<p class="meta">${(f.size/1024).toFixed(0)} KB${hintTxt} — drop another to replace</p>`;
      genBtn.disabled = false;
    }

    // Mirror of parse_bpm_key_from_filename in sa3_variations.py. Keep in sync.
    function parseBpmKeyFromFilename(name) {
      const stem = name.replace(/\.[^.]+$/, '');
      const parts = stem.split(/[_\-\s.]+/).filter(p => p.length > 0);

      let bpm = null;
      let bpmIdx = -1;

      for (let i = 0; i < parts.length; i++) {
        const m = parts[i].match(/^(\d{2,3})bpm$/i) || parts[i].match(/^bpm(\d{2,3})$/i);
        if (m) {
          const v = parseInt(m[1], 10);
          if (v >= 40 && v <= 240) { bpm = v; bpmIdx = i; break; }
        }
      }
      if (bpm === null) {
        for (let i = 0; i < parts.length; i++) {
          if (/^\d+$/.test(parts[i])) {
            const v = parseInt(parts[i], 10);
            if (v >= 40 && v <= 240) { bpm = v; bpmIdx = i; break; }
          }
        }
      }

      const keyRe = /^([A-G])([#b]?)(m|min|minor|maj|major)?$/i;
      function matchKey(p) {
        const m = keyRe.exec(p);
        if (!m) return null;
        const note = m[1].toUpperCase() + (m[2] || '');
        const qual = (m[3] || '').toLowerCase();
        if (qual === 'm' || qual === 'min' || qual === 'minor') return `${note} minor`;
        if (qual === 'maj' || qual === 'major') return `${note} major`;
        return note;
      }

      let key = null;
      if (bpmIdx >= 0 && bpmIdx + 1 < parts.length) key = matchKey(parts[bpmIdx + 1]);
      if (key === null && bpmIdx >= 1) key = matchKey(parts[bpmIdx - 1]);
      if (key === null) {
        for (const p of parts) {
          key = matchKey(p);
          if (key) break;
        }
      }
      return { bpm, key };
    }

    function escapeHtml(s) {
      return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
    }

    fileInput.addEventListener('change', e => {
      if (e.target.files[0]) setFile(e.target.files[0]);
    });

    drop.addEventListener('dragover', e => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', e => {
      e.preventDefault();
      drop.classList.remove('over');
      const f = e.dataTransfer.files[0];
      if (f) setFile(f);
    });

    genBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      genBtn.disabled = true;
      output.innerHTML = '';
      statusEl.textContent = 'Generating 5 variations… (~30s on M4 Max)';
      const t0 = performance.now();
      const fd = new FormData();
      fd.append('audio', selectedFile);
      fd.append('bpm', document.getElementById('bpm').value);
      fd.append('key', document.getElementById('key').value);
      fd.append('preset', currentPreset);
      fd.append('noise', currentNoise());
      try {
        const res = await fetch('/generate', { method: 'POST', body: fd });
        const data = await res.json();
        if (data.error) {
          statusEl.innerHTML = '<span class="err">' + escapeHtml(data.error + (data.stderr ? '\\n\\n' + data.stderr : '')) + '</span>';
          return;
        }
        const wall = (performance.now() - t0) / 1000;
        statusEl.textContent = `Done in ${data.elapsed.toFixed(1)}s wall (${wall.toFixed(1)}s round-trip)`;
        render(data);
      } catch (err) {
        statusEl.innerHTML = '<span class="err">' + escapeHtml(err.message) + '</span>';
      } finally {
        genBtn.disabled = false;
      }
    });

    function render(data) {
      const parts = [];
      parts.push('<div class="source"><h2>Source</h2>');
      parts.push(`<audio controls src="/file?path=${encodeURIComponent(data.source)}"></audio>`);
      parts.push('</div>');
      parts.push('<div class="results"><h2>Variations</h2>');
      for (const v of data.variations) {
        const isInpaint = v.mode !== 'a2a';
        const badge = isInpaint
          ? `<span class="badge inpaint">inpaint</span>`
          : `<span class="badge a2a">a2a</span>`;
        const maskTxt = v.mask
          ? ` · mask ${v.mask[0].toFixed(2)}–${v.mask[1].toFixed(2)}s`
          : '';
        parts.push('<div class="variation">');
        parts.push('<div class="v-head">');
        parts.push(`<div class="v-title"><span class="v-num">${v.index + 1}.</span> ${badge} <span class="v-steer">${escapeHtml(v.steer)}</span></div>`);
        parts.push(`<div class="v-meta">n=${v.noise.toFixed(2)} · cfg=${v.cfg.toFixed(1)}${maskTxt}</div>`);
        parts.push('</div>');
        parts.push(`<audio controls src="/file?path=${encodeURIComponent(v.path)}"></audio>`);
        parts.push('</div>');
      }
      parts.push('</div>');
      output.innerHTML = parts.join('');
    }
  </script>
</body>
</html>
"""


def parse_multipart(body: bytes, content_type: str) -> dict:
    """Parse a multipart/form-data body using email.parser. Returns a dict
    mapping field name -> str for text fields, or (filename, bytes) for files."""
    framed = b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    msg = BytesParser().parsebytes(framed)
    fields: dict = {}
    for part in msg.walk():
        if part.is_multipart():
            continue
        cd = part.get("Content-Disposition", "")
        if "form-data" not in cd:
            continue
        params: dict[str, str] = {}
        for chunk in cd.split(";"):
            chunk = chunk.strip()
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                params[k.strip()] = v.strip().strip('"')
        name = params.get("name", "")
        if not name:
            continue
        filename = params.get("filename", "")
        payload = part.get_payload(decode=True) or b""
        if filename:
            fields[name] = (filename, payload)
        else:
            try:
                fields[name] = payload.decode("utf-8")
            except UnicodeDecodeError:
                fields[name] = ""
    return fields


class Handler(http.server.BaseHTTPRequestHandler):
    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, query: str) -> None:
        params = urllib.parse.parse_qs(query)
        path_str = params.get("path", [""])[0]
        if not path_str:
            self.send_error(400, "missing path")
            return
        try:
            path = Path(path_str).resolve()
            path.relative_to(RUNS_DIR.resolve())
        except (ValueError, OSError):
            self.send_error(403, "outside runs dir")
            return
        if not path.is_file():
            self.send_error(404, "not found")
            return
        size = path.stat().st_size
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(size))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        try:
            with path.open("rb") as fh:
                while chunk := fh.read(64 * 1024):
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            # Browser cancelled the <audio> stream (seek, navigate, replace src).
            # Harmless — silence the noisy stack trace.
            sys.stderr.write(f"[studio] client disconnected mid-stream: {path.name}\n")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            body = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/file":
            self._serve_file(parsed.query)
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/generate":
            self.send_error(404)
            return
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "expected multipart/form-data"}, status=400)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            fields = parse_multipart(body, ctype)
        except Exception as exc:
            self._send_json({"error": f"multipart parse failed: {exc}"}, status=400)
            return

        audio = fields.get("audio")
        if not audio or not isinstance(audio, tuple):
            self._send_json({"error": "missing audio file"}, status=400)
            return
        filename, audio_bytes = audio
        bpm_str = fields.get("bpm", "").strip() if isinstance(fields.get("bpm"), str) else ""
        key_str = fields.get("key", "").strip() if isinstance(fields.get("key"), str) else ""
        preset_str = fields.get("preset", "free").strip() if isinstance(fields.get("preset"), str) else "free"
        if preset_str not in ("app", "free"):
            preset_str = "free"

        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = RUNS_DIR / f"studio_{ts}_{preset_str}"
        outdir.mkdir(parents=True, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (filename or "input.wav"))
        if not safe_name.lower().endswith(".wav"):
            safe_name += ".wav"
        input_wav = outdir / ("upload_" + safe_name)
        input_wav.write_bytes(audio_bytes)

        overrides = dict(
            sa3_root=SA3_ROOT,
            input=input_wav,
            outdir=outdir,
            kind="melodic",
            preset=preset_str,
        )
        try:
            overrides["bpm"] = float(bpm_str)
        except ValueError:
            pass
        if key_str:
            overrides["key"] = key_str
        noise_str = fields.get("noise", "").strip() if isinstance(fields.get("noise"), str) else ""
        try:
            noise_val = float(noise_str)
            if 0.01 <= noise_val <= 2.0:
                overrides["noise_a2a"] = noise_val
        except ValueError:
            pass

        from sa3_variations import make_default_args, run_variations
        ns = make_default_args(**overrides)

        global _pipeline
        sys.stderr.write(
            f"[studio] generating preset={preset_str} input={input_wav.name} outdir={outdir.name}\n"
        )
        t0 = time.time()
        try:
            with _pipeline_lock:
                result = run_variations(ns, pipeline=_pipeline)
                _pipeline = result["pipeline"]
        except Exception as exc:
            tb = traceback.format_exc()
            sys.stderr.write(tb)
            self._send_json(
                {"error": f"run_variations failed: {exc}", "trace": tb[-2000:]},
                status=500,
            )
            return
        elapsed = time.time() - t0

        variations = []
        for row in result["rows"]:
            spec = row["spec"]
            variations.append({
                "index": spec["index"],
                "mode": spec["mode"],
                "noise": spec["init_noise_level"],
                "cfg": spec["cfg"],
                "steer": spec.get("steer", ""),
                "path": str(Path(spec["out_path"]).resolve()),
                "mask": list(spec["inpaint_range"]) if spec.get("inpaint_range") else None,
            })
        self._send_json({
            "source": str(result["prepared"].resolve()),
            "variations": variations,
            "elapsed": elapsed,
        })

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        sys.stderr.write(f"[studio] {self.address_string()} {format % args}\n")


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            # Client went away mid-response — common for <audio> seek / src swap.
            return
        super().handle_error(request, client_address)


def main() -> int:
    ap = argparse.ArgumentParser(description="Local web UI for SA3 sample variations")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = ap.parse_args()

    try:
        import sa3_variations  # noqa: F401
    except ImportError as exc:
        sys.exit(
            f"error: cannot import sa3_variations ({exc}). Run with the MLX venv:\n"
            f"  optimized/mlx/.venv/bin/python {Path(__file__).name}"
        )
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    url = f"http://localhost:{args.port}"
    print(f"SA3 Studio listening on {url}")
    print(f"  runs land in: {RUNS_DIR}")
    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    server = ThreadedServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
