// SA3 Morph — Hairline WebView front end.
//
// Loads a loop, then morphs its style in real time. The C++ MorphEngine owns
// the streaming stack (sm-music DiT + SAME-S, ring buffer, MLX worker) and
// plays continuously through Processor::processBlock. The UI:
//   - loads a source file (decoded native-side; peaks come back)
//   - drives live morph knobs (Amount/Character/Motion/Style/Mode) over the
//     bridge — each takes effect on the next tick (no regenerate)
//   - receives a "morphState" push (~30fps) from C++ for the live playhead +
//     telemetry meter.
// All audio I/O is JUCE-native; the WebView never touches AudioContext.

// ── JUCE bridge plumbing ─────────────────────────────────────────────
const promises = new Map();
let nextPromiseId = 0;

window.__JUCE__.backend.addEventListener("__juce__complete", ({ promiseId, result }) => {
    const handlers = promises.get(promiseId);
    if (!handlers) return;
    promises.delete(promiseId);
    handlers.resolve(result);
});

function getNativeFunction(name) {
    return (...args) => new Promise((resolve) => {
        const promiseId = nextPromiseId++;
        promises.set(promiseId, { resolve });
        window.__JUCE__.backend.emitEvent("__juce__invoke", {
            name, params: args, resultId: promiseId,
        });
    });
}

const getStatus      = getNativeFunction("getStatus");
const getUiState     = getNativeFunction("getUiState");
const setUiState     = getNativeFunction("setUiState");
const loadSource     = getNativeFunction("loadSource");
const startMorph     = getNativeFunction("startMorph");
const stopMorph      = getNativeFunction("stopMorph");
const setDenoise     = getNativeFunction("setDenoise");
const setSourceBlend = getNativeFunction("setSourceBlend");
const setVelocity    = getNativeFunction("setVelocity");
const setCfg         = getNativeFunction("setCfg");
const setEvolve      = getNativeFunction("setEvolve");
const setPrompt      = getNativeFunction("setPrompt");
const setQuality     = getNativeFunction("setQuality");
const setWindow      = getNativeFunction("setWindow");
const setSteps       = getNativeFunction("setSteps");

// ── Constants ────────────────────────────────────────────────────────
const SOURCE_PEAKS_N = 220;
const COLOR_FG = "oklch(0.96 0.005 60)";

// ── DOM refs ─────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const statusEl    = $("status");
const statusText  = $("status-text");
const errorBanner = $("error-banner");
const errorText   = $("error-text");
const errorDismiss = $("error-dismiss");

const sourceEl     = $("source");
const sourceEmpty  = $("source-empty");
const sourceLoaded = $("source-loaded");
const srcNameEl    = $("srcname-text");
const srcDurEl     = $("srcdur-text");
const srcWaveEl    = $("srcwave");
const filePicker   = $("file-picker");

const promptEl   = $("prompt");
const promptClear = $("prompt-clear");
const bpmEl      = $("bpm");
const bpmValEl   = $("bpm-value");
const keyEl      = $("key");
const denoiseEl  = $("denoise");
const denoiseVal = $("denoise-value");
const blendEl    = $("blend");
const blendVal   = $("blend-value");
const velocityEl = $("velocity");
const velocityVal = $("velocity-value");
const cfgEl      = $("cfg");
const cfgVal     = $("cfg-value");
const windowEl   = $("window");
const stepsEl    = $("steps");
const stepsVal   = $("steps-value");
const evolveToggle = $("evolve-toggle");
const qualityToggle = $("quality-toggle");
const resetBtn   = $("reset");
const playBtn    = $("play");
const playLabel  = $("play-label");

const meterEl    = $("meter");
const mTick      = $("m-tick");
const mDecode    = $("m-decode");
const mGens      = $("m-gens");

// ── State ────────────────────────────────────────────────────────────
const state = {
    fileName: null,
    hasSource: false,
    fileSeconds: null,
    sourcePeaks: null,
    playing: false,
    evolve: false,
    quality: true,     // default Quality (medium) — matches MorphEngine default
    lastB64: null,     // kept so a Model-mode switch can reload the same source
    phase: "idle",
};

// ── Status / error ───────────────────────────────────────────────────
function setStatus(text, cls = "") {
    statusText.textContent = text;
    statusEl.className = "status" + (cls ? " " + cls : "");
}
function showError(msg) { errorText.textContent = msg; errorBanner.hidden = false; }
function clearError() { errorBanner.hidden = true; }
errorDismiss.addEventListener("click", clearError);

// ── Waveform ─────────────────────────────────────────────────────────
function escapeAttr(s) { return String(s).replace(/"/g, "&quot;"); }
let waveformIdCounter = 0;
let srcWfHandle = null;
let playheadEl = null;

function mountWaveform(el, peaks, opts = {}) {
    const { height = 36, barWidth = 1, gap = 1, color = "currentColor" } = opts;
    const totalWidth = peaks.length * (barWidth + gap) - gap;
    const cy = height / 2;
    let d = `M 0 ${cy}`;
    for (let i = 0; i < peaks.length; i++) {
        const x = i * (barWidth + gap) + barWidth / 2;
        d += ` L ${x} ${cy - peaks[i] * (height * 0.46)}`;
    }
    for (let i = peaks.length - 1; i >= 0; i--) {
        const x = i * (barWidth + gap) + barWidth / 2;
        d += ` L ${x} ${cy + peaks[i] * (height * 0.46)}`;
    }
    d += " Z";
    el.innerHTML = `
    <svg viewBox="0 0 ${totalWidth} ${height}" preserveAspectRatio="none"
         width="100%" height="${height}" style="display:block">
      <path d="${d}" fill="${escapeAttr(color)}" fill-opacity="0.22" />
      <path d="${d}" fill="${escapeAttr(color)}" fill-opacity="0.7" />
    </svg>
    <div class="playhead" style="left:0%"></div>`;
    playheadEl = el.querySelector(".playhead");
}

function renderSource(pending = false) {
    if (!state.hasSource && !state.fileName) {
        sourceEmpty.hidden = false;
        sourceLoaded.hidden = true;
        sourceEl.classList.add("empty");
        return;
    }
    sourceEl.classList.remove("empty");
    sourceEmpty.hidden = true;
    sourceLoaded.hidden = false;
    srcNameEl.textContent = state.fileName || "";
    srcDurEl.textContent = state.fileSeconds
        ? ` ${state.fileSeconds.toFixed(2)}s · 44.1 kHz`
        : (pending ? " decoding..." : "");
    if (state.sourcePeaks && state.sourcePeaks.length > 0) {
        mountWaveform(srcWaveEl, state.sourcePeaks, { color: COLOR_FG });
    } else {
        srcWaveEl.innerHTML = "";
        playheadEl = null;
    }
}

// ── File → base64 (chunked) ──────────────────────────────────────────
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const buf = new Uint8Array(reader.result);
            const CHUNK = 0x8000;
            let binary = "";
            for (let i = 0; i < buf.length; i += CHUNK)
                binary += String.fromCharCode.apply(null, buf.subarray(i, i + CHUNK));
            resolve(btoa(binary));
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(file);
    });
}

async function loadFile(file) {
    const b64 = await fileToBase64(file);
    state.fileName = file.name;
    state.lastB64 = b64;     // keep so a Model-mode switch can reload it
    await loadFromB64(b64);
    saveUiSoon();
}

async function loadFromB64(b64) {
    clearError();
    state.hasSource = false;
    state.playing = false;
    updatePlayButton();
    renderSource(/*pending=*/true);
    setStatus(state.quality ? "Loading (Quality)..." : "Loading models...", "accent");

    let resp;
    try {
        resp = await loadSource({ audioBase64: b64, peaksN: SOURCE_PEAKS_N });
    } catch (e) {
        showError("Load bridge error: " + e);
        setStatus("Error", "error");
        return;
    }
    if (!resp || !resp.ok) {
        showError((resp && resp.error) || "load failed");
        setStatus("Error", "error");
        state.hasSource = false;
        renderSource();
        return;
    }
    state.hasSource = true;
    state.fileSeconds = Number(resp.duration) || null;
    state.sourcePeaks = (resp.peaks || []).map(Number);
    renderSource();
    // Loading completes already morphing + playing (engine starts playback on ready).
    state.playing = true;
    setStatus("Morphing", "accent");
    meterEl.hidden = false;
    // Push the current control values to the freshly-loaded engine.
    pushAllControls();
    updatePlayButton();
}

// ── Controls ─────────────────────────────────────────────────────────
denoiseEl.addEventListener("input", () => {
    denoiseVal.textContent = Number(denoiseEl.value).toFixed(2);
    setDenoise(Number(denoiseEl.value));
    saveUiSoon();
});
// Intensity rebuilds the styled target (a brief one-shot generation), so apply it
// on release (change) rather than every input frame — dragging only updates the
// label; letting go recomputes the Amount=1 endpoint once.
blendEl.addEventListener("input", () => {
    blendVal.textContent = Number(blendEl.value).toFixed(2);
    saveUiSoon();
});
blendEl.addEventListener("change", () => {
    setSourceBlend(Number(blendEl.value));
});
velocityEl.addEventListener("input", () => {
    velocityVal.textContent = Number(velocityEl.value).toFixed(2);
    setVelocity(Number(velocityEl.value));
    saveUiSoon();
});
// CFG rebuilds the styled target (brief one-shot) -> apply on release, label on input.
cfgEl.addEventListener("input", () => {
    cfgVal.textContent = Number(cfgEl.value).toFixed(1);
    saveUiSoon();
});
cfgEl.addEventListener("change", () => {
    setCfg(Number(cfgEl.value));
});
// Window: morph the last N seconds. Fixed when the generator is built -> reload.
windowEl.addEventListener("change", async () => {
    let s = Math.round(Number(windowEl.value));
    if (!isFinite(s)) s = 10;
    s = Math.max(2, Math.min(30, s));
    windowEl.value = s;
    const changed = await setWindow(s);
    if (changed && state.lastB64) await loadFromB64(state.lastB64);
    saveUiSoon();
});
// Steps: diffusion steps per window. Fixed when the generator is built -> reload.
stepsEl.addEventListener("input", () => { stepsVal.textContent = String(Math.round(Number(stepsEl.value))); });
stepsEl.addEventListener("change", async () => {
    const n = Math.max(2, Math.min(16, Math.round(Number(stepsEl.value))));
    stepsEl.value = n; stepsVal.textContent = String(n);
    const changed = await setSteps(n);
    if (changed && state.lastB64) await loadFromB64(state.lastB64);
    saveUiSoon();
});
// Build the prompt the engine actually sees: wrap the user's Style in SA3's trained
// caption format and append the manual BPM / Key anchors (the offline remix recipe).
// Empty Style -> empty prompt (free variation / morph back to source).
function buildPrompt() {
    let p = promptEl.value.trim();
    if (!p) return "";
    if (!/^TrackType:/i.test(p)) p = "TrackType: Music, VocalType: Instrumental. " + p;
    if (!/[.!?]\s*$/.test(p)) p += ".";
    const bpm = bpmValEl.value.trim();
    const key = keyEl.value.trim();
    if (bpm) p += ` ${bpm} BPM.`;
    if (key) p += ` Key: ${key}.`;
    return p;
}
function sendPrompt() { setPrompt(buildPrompt()); }

function updatePromptClear() { promptClear.hidden = !promptEl.value; }
promptEl.addEventListener("input", updatePromptClear);
promptEl.addEventListener("change", () => { sendPrompt(); saveUiSoon(); });

// BPM: slider <-> number field stay in sync; either re-sends the (templated) prompt.
bpmEl.addEventListener("input", () => { bpmValEl.value = bpmEl.value; });
bpmEl.addEventListener("change", () => { bpmValEl.value = bpmEl.value; sendPrompt(); saveUiSoon(); });
bpmValEl.addEventListener("change", () => {
    const v = parseInt(bpmValEl.value, 10);
    if (isFinite(v)) bpmEl.value = Math.max(40, Math.min(200, v));
    sendPrompt(); saveUiSoon();
});
keyEl.addEventListener("change", () => { sendPrompt(); saveUiSoon(); });

// Clear Style: empties the prompt so the output morphs back toward the dry source.
promptClear.addEventListener("click", () => {
    promptEl.value = "";
    sendPrompt();
    updatePromptClear();
    promptEl.focus();
    saveUiSoon();
});
evolveToggle.addEventListener("click", () => {
    state.evolve = !state.evolve;
    evolveToggle.textContent = state.evolve ? "Evolve" : "Coherent";
    evolveToggle.classList.toggle("one-shot", state.evolve);
    setEvolve(state.evolve);
    saveUiSoon();
});

qualityToggle.addEventListener("click", async () => {
    state.quality = !state.quality;
    qualityToggle.textContent = state.quality ? "Quality" : "Live";
    qualityToggle.classList.toggle("one-shot", state.quality);
    const changed = await setQuality(state.quality);
    // Family is fixed per generator -> reload the current source to apply.
    if (changed && state.lastB64) await loadFromB64(state.lastB64);
    saveUiSoon();
});

function pushAllControls() {
    setDenoise(Number(denoiseEl.value));
    setSourceBlend(Number(blendEl.value));
    setVelocity(Number(velocityEl.value));
    setCfg(Number(cfgEl.value));
    setEvolve(state.evolve);
    if (promptEl.value) sendPrompt();
}

// Reset the morph sliders + Mode to defaults (leaves Style / Window / Model and
// the loaded source untouched). Intensity+CFG share one styled-target rebuild.
const DEFAULTS = { denoise: 0.5, blend: 0.6, velocity: 1.0, cfg: 4.0 };
resetBtn.addEventListener("click", () => {
    denoiseEl.value = DEFAULTS.denoise;  denoiseVal.textContent = DEFAULTS.denoise.toFixed(2);  setDenoise(DEFAULTS.denoise);
    blendEl.value = DEFAULTS.blend;      blendVal.textContent = DEFAULTS.blend.toFixed(2);      setSourceBlend(DEFAULTS.blend);
    velocityEl.value = DEFAULTS.velocity; velocityVal.textContent = DEFAULTS.velocity.toFixed(2); setVelocity(DEFAULTS.velocity);
    cfgEl.value = DEFAULTS.cfg;          cfgVal.textContent = DEFAULTS.cfg.toFixed(1);          setCfg(DEFAULTS.cfg);
    if (state.evolve) {
        state.evolve = false;
        evolveToggle.textContent = "Coherent";
        evolveToggle.classList.remove("one-shot");
        setEvolve(false);
    }
    saveUiSoon();
});

// ── Transport ────────────────────────────────────────────────────────
function updatePlayButton() {
    playBtn.disabled = !state.hasSource;
    playLabel.textContent = state.playing ? "Stop" : "Play";
    playBtn.classList.toggle("busy", state.playing);
}
playBtn.addEventListener("click", async () => {
    if (!state.hasSource) return;
    if (state.playing) { await stopMorph(); state.playing = false; setStatus("Stopped"); }
    else               { await startMorph(); state.playing = true;  setStatus("Morphing", "accent"); }
    updatePlayButton();
});

// ── Live morph state push (C++ → JS, ~30fps) ─────────────────────────
window.__JUCE__.backend.addEventListener("morphState", (s) => {
    if (!s) return;
    if (playheadEl && s.loopSecs > 0) {
        playheadEl.style.left = (Math.max(0, Math.min(1, s.progress)) * 100).toFixed(2) + "%";
    }
    if (!meterEl.hidden) {
        mTick.textContent   = s.tickMs   ? s.tickMs.toFixed(0) + " ms" : "–";
        mDecode.textContent = s.decodeMs ? s.decodeMs.toFixed(0) + " ms" : "–";
        mGens.textContent   = s.completions != null ? String(s.completions) : "–";
    }
});

// ── UI state persistence ─────────────────────────────────────────────
let saveTimer = null;
function saveUiSoon() {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
        setUiState(JSON.stringify({
            prompt: promptEl.value,
            bpm: bpmValEl.value,
            key: keyEl.value,
            denoise: denoiseEl.value,
            blend: blendEl.value,
            velocity: velocityEl.value,
            cfg: cfgEl.value,
            window: windowEl.value,
            steps: stepsEl.value,
            evolve: state.evolve,
            quality: state.quality,
        }));
    }, 250);
}
async function restoreUiState() {
    let json;
    try { json = await getUiState(); } catch { return; }
    if (!json) return;
    let s; try { s = JSON.parse(json); } catch { return; }
    if (s.prompt != null) { promptEl.value = s.prompt; updatePromptClear(); }
    if (s.bpm != null) { bpmValEl.value = s.bpm; if (s.bpm) bpmEl.value = s.bpm; }
    if (s.key != null) { keyEl.value = s.key; }
    if (s.denoise != null) { denoiseEl.value = s.denoise; denoiseVal.textContent = Number(s.denoise).toFixed(2); }
    if (s.blend != null) { blendEl.value = s.blend; blendVal.textContent = Number(s.blend).toFixed(2); }
    if (s.velocity != null) { velocityEl.value = s.velocity; velocityVal.textContent = Number(s.velocity).toFixed(2); }
    if (s.cfg != null) { cfgEl.value = s.cfg; cfgVal.textContent = Number(s.cfg).toFixed(1); }
    if (s.window != null) { windowEl.value = s.window; setWindow(Number(s.window)); }
    if (s.steps != null) { stepsEl.value = s.steps; stepsVal.textContent = String(Math.round(Number(s.steps))); setSteps(Number(s.steps)); }
    if (s.evolve != null) {
        state.evolve = !!s.evolve;
        evolveToggle.textContent = state.evolve ? "Evolve" : "Coherent";
        evolveToggle.classList.toggle("one-shot", state.evolve);
    }
    if (s.quality != null) {
        state.quality = !!s.quality;
        qualityToggle.textContent = state.quality ? "Quality" : "Live";
        qualityToggle.classList.toggle("one-shot", state.quality);
        setQuality(state.quality);   // sync engine's desired family (applied on next load)
    }
}

// ── File input + drag/drop ───────────────────────────────────────────
filePicker.addEventListener("change", () => {
    const f = filePicker.files && filePicker.files[0];
    if (f) loadFile(f);
    filePicker.value = "";
});
sourceEmpty.addEventListener("click", () => filePicker.click());

let dragDepth = 0;
function acceptDrag(e) {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
}
document.addEventListener("dragenter", (e) => {
    acceptDrag(e); dragDepth++;
    if (dragDepth === 1) sourceEl.classList.add("drag");
});
document.addEventListener("dragover", acceptDrag);
document.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) sourceEl.classList.remove("drag");
});
document.addEventListener("drop", (e) => {
    e.preventDefault();
    dragDepth = 0;
    sourceEl.classList.remove("drag");
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) loadFile(file);
});

// ── Keyboard ─────────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") { e.preventDefault(); playBtn.click(); }
});

// ── Init ─────────────────────────────────────────────────────────────
denoiseVal.textContent = Number(denoiseEl.value).toFixed(2);
blendVal.textContent = Number(blendEl.value).toFixed(2);
velocityVal.textContent = Number(velocityEl.value).toFixed(2);
cfgVal.textContent = Number(cfgEl.value).toFixed(1);
updatePromptClear();
renderSource();
updatePlayButton();
restoreUiState();
setStatus("Drop a loop");
