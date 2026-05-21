/**
 * Remix — Tape Lab UI on top of /api/remix + /api/analyze + in-browser Demucs.
 *
 * Stages: empty → (drop) → ready → (remix) → remixing → compared
 * From compared, regenerate / add-variant push more entries onto the variants
 * array (up to 4). The A/B player crossfades the source against whichever
 * variant is selected.
 */

import { initDemucs, separate, isReady as demucsReady } from "/js/demucs.js";

const DEMUCS_MODEL_URL = "/models/htdemucs.onnx";
const DEMUCS_WASM_MODULE_URL = "/wasm/demucs_wasm.js";
const MAX_VARIANTS = 4;
const ETA_BASE_SEC = 12;

const STAGES = [
  { id: "separate", label: "Separating vocals",     short: "Vocals",  weight: 0.30 },
  { id: "upload",   label: "Uploading instrumental", short: "Upload",  weight: 0.10 },
  { id: "gpu",      label: "Generating audio",       short: "Generate", weight: 0.50 },
  { id: "mixdown",  label: "Mixing vocals back",    short: "Mixdown", weight: 0.10 },
];

// ─── DOM ────────────────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);

const hdRight        = $("hdRight");
const dropzone       = $("dropzone");
const browseBtn      = $("browseBtn");
const fileInput      = $("fileInput");
const helpBtn        = $("helpBtn");

// Reels (whole panels — click to switch A/B)
const reelA  = $("reelA");
const reelB  = $("reelB");

// Reel A
const aTitle   = $("aTitle");
const aStats   = $("aStats");
const aBody    = $("aBody");
const aFooter  = $("aFooter");
const aSpecs   = $("aSpecs");
const replaceBtn = $("replaceBtn");

// Reel B
const bTitleEmpty   = $("bTitleEmpty");
const bVersionWrap  = $("bVersionWrap");
const versionSelect = $("versionSelect");
const bRuntime      = $("bRuntime");
const bPrompt       = $("bPrompt");
const bBody         = $("bBody");
const bFooter       = $("bFooter");
const bMeta         = $("bMeta");
const regenerateBtn = $("regenerateBtn");
const downloadBtn   = $("downloadBtn");

// Variants strip
const variantsStrip  = $("variantsStrip");
const variantChips   = $("variantChips");
const addVariantBtn  = $("addVariantBtn");

// Transport
const transport   = $("transport");
const tpPlay      = $("tpPlay");
const tpPlayIcon  = $("tpPlayIcon");
const tpBar       = $("tpBar");
const tpFill      = $("tpFill");
const tpHead      = $("tpHead");
const tpNow       = $("tpNow");
const tpTotal     = $("tpTotal");
const tpListening = $("tpListening");
const tpRoleDot   = $("tpRoleDot");
const tpRoleLabel = $("tpRoleLabel");
const swapBtn     = $("swapBtn");

// Progress card
const progressCard = $("progressCard");
const pcTitle      = $("pcTitle");
const pcSubtitle   = $("pcSubtitle");
const pcElapsed    = $("pcElapsed");
const pcEta        = $("pcEta");
const pcFill       = $("pcFill");
const pcStageList  = $("pcStageList");

// Controls
const controls       = $("controls");
const promptEl       = $("prompt");
const bpmEl          = $("bpm");
const keyEl          = $("key");
const bpmStatus      = $("bpmStatus");
const keyStatus      = $("keyStatus");
const suffixBpm      = $("suffixBpm");
const suffixKey      = $("suffixKey");
const noiseEl        = $("noise");
const noiseVal       = $("noiseVal");
const cfgEl          = $("cfg");
const cfgVal         = $("cfgVal");
const presetChips    = $("presetChips");
const separateTog    = $("separateTog");
const readdTog       = $("readdTog");
const vocalGainEl    = $("vocalGain");
const vocalGainVal   = $("vocalGainVal");
const vocalGainRow   = $("vocalGainRow");
const remixBtn       = $("remixBtn");
const remixLabel     = $("remixLabel");
const remixArr       = $("remixArr");
const promptCharCount = $("promptCharCount");

// Empty guide
const emptyGuide = $("emptyGuide");

// Error
const errorBox = $("errorBox");

// ─── State ──────────────────────────────────────────────────────────────────
const state = {
  stage: "empty",      // empty | ready | remixing | compared
  file: null,
  inputBuffer: null,   // AudioBuffer, 44.1k stereo
  inputPeaks: null,    // Float32Array of length 420
  analysis: { bpm: null, key: null, analyzing: false },
  vocalsBuffer: null,
  instrumentalBuffer: null,
  variants: [],        // [{n, prompt, label, runtime, audioBuffer, peaks, wavUrl}]
  activeVariantIdx: 0,
  mode: "A",           // A=source, B=remix
  separateOn: true,
  readdOn: true,
};

let lastWavUrl = null;

// ─── Slider value bindings ─────────────────────────────────────────────────
noiseEl.addEventListener("input", () => {
  const v = parseFloat(noiseEl.value);
  noiseVal.textContent = v.toFixed(2);
  syncPresetChips(v);
});
cfgEl.addEventListener("input", () => { cfgVal.textContent = parseFloat(cfgEl.value).toFixed(1); });
vocalGainEl.addEventListener("input", () => {
  vocalGainVal.textContent = parseFloat(vocalGainEl.value).toFixed(2) + "×";
});
promptEl.addEventListener("input", updatePromptCharCount);
function updatePromptCharCount() { promptCharCount.textContent = String(promptEl.value.length); }
updatePromptCharCount();

// Preset chips
presetChips.querySelectorAll("button").forEach((b) => {
  b.addEventListener("click", () => {
    const v = parseFloat(b.dataset.val);
    noiseEl.value = String(v);
    noiseEl.dispatchEvent(new Event("input"));
  });
});
function syncPresetChips(v) {
  let closest = null, best = Infinity;
  presetChips.querySelectorAll("button").forEach((b) => {
    const d = Math.abs(parseFloat(b.dataset.val) - v);
    if (d < best) { best = d; closest = b; }
  });
  presetChips.querySelectorAll("button").forEach((b) =>
    b.classList.toggle("active", b === closest && best < 0.06)
  );
}

// Vocal toggles
separateTog.addEventListener("click", () => {
  state.separateOn = !state.separateOn;
  separateTog.classList.toggle("on", state.separateOn);
  separateTog.setAttribute("aria-checked", String(state.separateOn));
});
readdTog.addEventListener("click", () => {
  state.readdOn = !state.readdOn;
  readdTog.classList.toggle("on", state.readdOn);
  readdTog.setAttribute("aria-checked", String(state.readdOn));
  vocalGainRow.style.display = state.readdOn ? "" : "none";
});

helpBtn.addEventListener("click", () =>
  window.open("https://github.com/Stability-AI/stable-audio-3", "_blank")
);

$("brandHome").addEventListener("click", reset);

// ─── Drop / file pick ──────────────────────────────────────────────────────
dropzone.addEventListener("click", () => fileInput.click());
browseBtn.addEventListener("click", (e) => { e.stopPropagation(); fileInput.click(); });
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files?.[0]) handleFile(fileInput.files[0]);
});
replaceBtn.addEventListener("click", reset);

// ─── Reset / handle file ───────────────────────────────────────────────────
function reset() {
  state.stage = "empty";
  state.file = null;
  state.inputBuffer = null;
  state.inputPeaks = null;
  state.analysis = { bpm: null, key: null, analyzing: false };
  state.vocalsBuffer = null;
  state.instrumentalBuffer = null;
  state.variants = [];
  state.activeVariantIdx = 0;
  state.mode = "A";
  abReset();
  fileInput.value = "";
  bpmEl.value = "";
  keyEl.value = "";
  hideError();
  render();
}

async function handleFile(file) {
  hideError();
  state.file = file;
  try {
    const arrayBuffer = await file.arrayBuffer();
    const tmpCtx = new (window.AudioContext || window.webkitAudioContext)();
    let buf = await tmpCtx.decodeAudioData(arrayBuffer);
    await tmpCtx.close();
    if (buf.sampleRate !== 44100) buf = await resample(buf, 44100);
    if (buf.numberOfChannels === 1) buf = monoToStereo(buf);
    state.inputBuffer = buf;
    state.inputPeaks = getPeaks(buf, 420);
  } catch (e) {
    showError("Couldn't decode audio: " + e.message);
    return;
  }

  state.stage = "ready";
  render();

  // Fire analysis in the background.
  analyzeTrack(file);
}

async function analyzeTrack(file) {
  state.analysis = { bpm: null, key: null, analyzing: true };
  updateBpmKeyStatus();
  try {
    const fd = new FormData();
    fd.append("audio", file, file.name);
    const r = await fetch("/api/analyze", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`status ${r.status}`);
    const { bpm, key } = await r.json();
    state.analysis = { bpm, key, analyzing: false };
    if (bpm != null && bpmEl.value.trim() === "") bpmEl.value = String(Math.round(bpm));
    if (key && keyEl.value.trim() === "") keyEl.value = titleCaseKey(key);
  } catch (e) {
    console.warn("[analyze] failed:", e);
    state.analysis = { bpm: null, key: null, analyzing: false };
  }
  updateBpmKeyStatus();
  updateReelAStats();
  updatePromptSuffix();
}

function titleCaseKey(k) {
  return k.split(/\s+/).map((w) =>
    w.length ? w[0].toUpperCase() + w.slice(1).toLowerCase() : w
  ).join(" ");
}

function buildPrompt() {
  let p = promptEl.value.trim();
  const bpmV = bpmEl.value.trim();
  const keyV = keyEl.value.trim();
  if (p && !/[.!?]$/.test(p)) p += ".";
  if (bpmV) p += ` ${bpmV} BPM.`;
  if (keyV) p += ` Key: ${titleCaseKey(keyV)}.`;
  return p;
}

function updatePromptSuffix() {
  suffixBpm.textContent = state.analysis.analyzing ? "…" : (bpmEl.value || "—");
  suffixKey.textContent = state.analysis.analyzing ? "…" : (keyEl.value || "—");
}
bpmEl.addEventListener("input", updatePromptSuffix);
keyEl.addEventListener("input", updatePromptSuffix);

function updateBpmKeyStatus() {
  if (state.analysis.analyzing) {
    bpmStatus.innerHTML = '<span class="dot"></span>detecting…';
    keyStatus.innerHTML = '<span class="dot"></span>detecting…';
    bpmStatus.hidden = false;
    keyStatus.hidden = false;
  } else {
    bpmStatus.hidden = true;
    keyStatus.hidden = true;
  }
}

// ─── Render: top-level stage routing ───────────────────────────────────────
function render() {
  renderHeader();
  renderReelA();
  renderReelB();
  renderVariantsStrip();
  renderTransport();
  renderControlsAndProgress();
  renderEmptyGuide();
}

function renderHeader() {
  hdRight.innerHTML = "";
  let html = "";
  if (state.stage === "empty") {
    html = `<span class="meta">Stable Audio 3 · <b>medium</b></span>`;
  } else if (state.stage === "ready") {
    html = `<span class="meta"><b>~${ETA_BASE_SEC}s</b> render</span>`
         + `<span class="sep">·</span>`
         + `<span class="meta">Stable Audio 3 · <b>medium</b></span>`;
  } else if (state.stage === "remixing") {
    html = `<span class="meta live">Rendering variation ${state.variants.length + 1}…</span>`;
  } else {
    html = `<span class="meta"><b>${state.variants.length}</b> of ${MAX_VARIANTS} variations</span>`
         + `<span class="sep">·</span>`
         + `<span class="meta">Stable Audio 3 · <b>medium</b></span>`;
  }
  hdRight.innerHTML = html;
}

function renderReelA() {
  const empty = state.stage === "empty";
  aFooter.hidden = empty;
  aTitle.hidden = empty;
  aStats.hidden = empty;
  if (empty) {
    aBody.innerHTML = "";
    aBody.appendChild(dropzone); // re-mount the dropzone DOM
    return;
  }
  // Title
  const name = state.file.name;
  const m = name.match(/(\.[^.]+)$/);
  const base = m ? name.slice(0, -m[1].length) : name;
  const ext = m ? m[1] : "";
  aTitle.innerHTML = "";
  aTitle.appendChild(document.createTextNode(base));
  if (ext) {
    const e = document.createElement("span");
    e.className = "ext";
    e.textContent = ext;
    aTitle.appendChild(e);
  }
  aTitle.title = name;

  updateReelAStats();

  // Specs
  const sr = state.inputBuffer.sampleRate;
  const ch = state.inputBuffer.numberOfChannels === 2 ? "stereo" : "mono";
  const sizeMB = (state.file.size / (1024 * 1024)).toFixed(1);
  aSpecs.innerHTML = `<span>${ch} · ${(sr / 1000).toFixed(1)} kHz · 16-bit</span><span class="sep">·</span><span>${sizeMB} MB</span>`;

  // Body — render waveform with hover readout
  paintReel(aBody, "a", state.inputPeaks, state.inputBuffer.duration, getBpmInt());
}

function updateReelAStats() {
  if (state.stage === "empty" || !state.inputBuffer) return;
  const dur = formatTime(state.inputBuffer.duration);
  const bpm = bpmEl.value || (state.analysis.bpm != null ? Math.round(state.analysis.bpm) : null);
  const key = keyEl.value || state.analysis.key || null;
  const parts = [];
  if (bpm) parts.push(`<span class="stat"><span class="v">${bpm}</span><span class="lbl">bpm</span></span>`);
  if (key) parts.push(`<span class="stat"><span class="v">${key}</span></span>`);
  parts.push(`<span class="stat"><span class="v">${dur}</span></span>`);
  if (state.analysis.analyzing) parts.push(`<span class="pulse">analyzing…</span>`);
  aStats.innerHTML = parts.join(`<span class="sep">/</span>`);
}

function renderReelB() {
  const isCompared = state.stage === "compared" && state.variants.length > 0;
  const isRemixing = state.stage === "remixing";
  const isEmptyish = !isCompared && !isRemixing;

  bTitleEmpty.hidden = !isEmptyish;
  if (isEmptyish) {
    bTitleEmpty.textContent = "No remix yet";
    bTitleEmpty.classList.add("muted");
  }
  bVersionWrap.hidden = !isCompared;
  bFooter.hidden = !isCompared;

  if (isCompared) {
    const v = state.variants[state.activeVariantIdx];
    // Version select
    versionSelect.innerHTML = state.variants
      .map((x, i) => {
        const lbl = x.label.length > 28 ? x.label.slice(0, 28) + "…" : x.label;
        return `<option value="${i}"${i === state.activeVariantIdx ? " selected" : ""}>Variation ${x.n} · ${escapeHtml(lbl)}</option>`;
      })
      .join("");
    bRuntime.textContent = v.runtime;
    bPrompt.textContent = v.label;
    bPrompt.title = v.prompt;
    downloadBtn.href = v.wavUrl;
    downloadBtn.download = (state.file.name.replace(/\.[^.]+$/, "") || "track") + `-remix-v${v.n}.wav`;

    bMeta.innerHTML = ""; // diff metrics intentionally omitted (no real data)
    paintReel(bBody, "b", v.peaks, state.inputBuffer.duration, getBpmInt(), { variant: v });
  } else if (isRemixing) {
    // Show source peaks with the scan overlay
    paintReel(bBody, "b", state.inputPeaks, state.inputBuffer.duration, getBpmInt(), { remixing: true });
    bTitleEmpty.hidden = false;
    bTitleEmpty.textContent = `Variation ${state.variants.length + 1} in progress`;
    bTitleEmpty.classList.remove("muted");
  } else {
    // Empty await
    bBody.innerHTML = `
      <div class="reel-await">
        <div class="await-mark">B</div>
        <div class="await-title">No remix yet</div>
        <div class="await-sub">Press <b>Remix track</b> to render the first variation. It usually takes 10–20 seconds.</div>
      </div>
    `;
  }
}

versionSelect.addEventListener("change", () => {
  const i = parseInt(versionSelect.value, 10);
  pickVariant(i);
});
regenerateBtn.addEventListener("click", () => startRemix());

function renderVariantsStrip() {
  const show = state.variants.length > 0 && state.stage === "compared";
  variantsStrip.hidden = !show;
  if (!show) return;
  variantChips.innerHTML = "";
  state.variants.forEach((v, i) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "variant-chip" + (i === state.activeVariantIdx ? " active" : "");
    chip.title = v.prompt;
    chip.innerHTML = `
      <svg class="thumb" viewBox="0 0 56 26" preserveAspectRatio="none"></svg>
      <div class="meta">
        <div class="label">${escapeHtml(v.label)}</div>
        <div class="sub">v${v.n} · ${escapeHtml(v.runtime)}</div>
      </div>
      ${state.variants.length > 1 ? '<span class="x" title="Remove">×</span>' : ""}
    `;
    paintMiniWave(chip.querySelector(".thumb"), v.peaks);
    chip.addEventListener("click", (e) => {
      if (e.target.classList.contains("x")) { e.stopPropagation(); removeVariant(i); return; }
      pickVariant(i);
    });
    variantChips.appendChild(chip);
  });
  addVariantBtn.hidden = state.variants.length >= MAX_VARIANTS;
  addVariantBtn.disabled = state.stage === "remixing";
}
addVariantBtn.addEventListener("click", () => startRemix());

function renderTransport() {
  transport.hidden = state.stage === "empty";
  if (state.stage === "empty") return;
  // Playback stays available across all stages — users can keep listening
  // while a remix is being rendered.
  tpPlay.disabled = !state.inputBuffer;
  swapBtn.disabled = state.variants.length === 0;
  tpTotal.textContent = formatTime(state.inputBuffer?.duration || 0);
  updateTpListening();
  updateReelClickability();
}

function updateReelClickability() {
  const compared = state.stage === "compared" && state.variants.length > 0;
  reelA.dataset.clickable = compared && state.mode !== "A" ? "true" : "false";
  reelB.dataset.clickable = compared && state.mode !== "B" ? "true" : "false";
}

function updateTpListening() {
  tpListening.parentElement.classList.toggle("b", state.mode === "B");
  tpRoleLabel.textContent = state.mode === "A" ? "Source" : "Remix";
  tpRoleDot.style.background = state.mode === "A" ? "var(--a-tint)" : "var(--b-tint)";
  tpRoleDot.style.boxShadow = `0 0 6px ${state.mode === "A" ? "var(--a-tint)" : "var(--b-tint)"}`;
}

function renderControlsAndProgress() {
  const remixing = state.stage === "remixing";
  controls.hidden = state.stage === "empty" || remixing;
  progressCard.hidden = !remixing;
  if (state.stage !== "empty") updatePromptSuffix();
}

function renderEmptyGuide() {
  emptyGuide.hidden = state.stage !== "empty";
}

// ─── Waveform painting ─────────────────────────────────────────────────────
function paintReel(bodyEl, role, peaks, durationSec, bpm, opts = {}) {
  bodyEl.innerHTML = "";
  // Beat grid
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  bg.setAttribute("class", "beat-grid");
  bg.setAttribute("viewBox", "0 0 100 100");
  bg.setAttribute("preserveAspectRatio", "none");
  if (bpm && durationSec) {
    const secPerBeat = 60 / bpm;
    const beats = Math.floor(durationSec / secPerBeat);
    for (let i = 4; i < beats; i += 4) {
      const p = (i * secPerBeat) / durationSec;
      const ln = document.createElementNS("http://www.w3.org/2000/svg", "line");
      ln.setAttribute("x1", String(p * 100));
      ln.setAttribute("y1", "0");
      ln.setAttribute("x2", String(p * 100));
      ln.setAttribute("y2", "100");
      ln.setAttribute("stroke", "var(--line)");
      ln.setAttribute("stroke-width", "0.1");
      ln.setAttribute("stroke-dasharray", "0.5 1");
      bg.appendChild(ln);
    }
  }
  bodyEl.appendChild(bg);

  // Waveform SVG
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "wave");
  svg.setAttribute("viewBox", "0 0 800 220");
  svg.setAttribute("preserveAspectRatio", "none");
  bodyEl.appendChild(svg);
  paintWaveformInto(svg, peaks, role, 0);
  bodyEl._svg = svg;
  bodyEl._role = role;
  bodyEl._peaks = peaks;

  // Time ruler
  const ruler = document.createElement("div");
  ruler.className = "time-ruler";
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];
  ticks.forEach((p, i) => {
    if (i > 0 && i < ticks.length - 1) {
      const mark = document.createElement("span");
      mark.className = "tick-mark";
      mark.style.left = `${p * 100}%`;
      ruler.appendChild(mark);
    }
    const tk = document.createElement("span");
    tk.className = "tick" + (i === 0 ? " first" : i === ticks.length - 1 ? " last" : "");
    if (i !== 0 && i !== ticks.length - 1) tk.style.left = `${p * 100}%`;
    tk.textContent = formatTime(p * durationSec);
    ruler.appendChild(tk);
  });
  bodyEl.appendChild(ruler);

  // Remixing overlay — created hidden; only revealed once the actual GPU
  // diffusion step begins (see beginScanAnim).
  if (opts.remixing) {
    const ov = document.createElement("div");
    ov.className = "remix-overlay";
    ov.hidden = true;
    ov.innerHTML = `
      <div class="veil"></div>
      <div class="scan" style="left:6%"></div>
      <div class="readout"><span class="blip"></span><span id="ovReadout">generating · 0%</span></div>
    `;
    bodyEl.appendChild(ov);
    bodyEl._overlay = ov;
    bodyEl._scan = ov.querySelector(".scan");
    bodyEl._ovReadout = ov.querySelector("#ovReadout");
  }

  // Hover readout + click-to-seek
  if (!opts.remixing) {
    const hov = document.createElement("div");
    hov.className = "hover-readout";
    hov.hidden = true;
    hov.innerHTML = `<span class="lbl">t</span><span class="t">0:00.0</span><span class="lbl">bar</span><span class="b">1.1</span>`;
    bodyEl.appendChild(hov);
    bodyEl.onmousemove = (e) => {
      const rect = bodyEl.getBoundingClientRect();
      const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      const tSec = x * durationSec;
      const m = Math.floor(tSec / 60);
      const s = Math.floor(tSec % 60);
      const ms = Math.floor((tSec % 1) * 10);
      const bpmUse = bpm || 0;
      const beatN = bpmUse ? Math.floor(tSec / (60 / bpmUse)) : 0;
      const barN = Math.floor(beatN / 4) + 1;
      const beatInBar = (beatN % 4) + 1;
      hov.querySelector(".t").textContent = `${m}:${s.toString().padStart(2, "0")}.${ms}`;
      hov.querySelector(".b").textContent = `${barN}.${beatInBar}`;
      hov.hidden = false;
    };
    bodyEl.onmouseleave = () => { hov.hidden = true; };
    // Bind click on the wave SVG itself so the rect we measure is the exact
    // visual element the user is clicking — and use abDuration (the actual
    // playable length) instead of the source's duration for the seek target.
    const wave = bodyEl.querySelector(".wave");
    if (wave) {
      wave.style.cursor = "crosshair";
      wave.onclick = (e) => {
        if (!state.inputBuffer) return;
        e.stopPropagation();
        const targetMode = role === "a" ? "A" : "B";
        if (state.stage === "compared" && state.mode !== targetMode
            && !(targetMode === "B" && state.variants.length === 0)) {
          abSetMode(targetMode);
        }
        const rect = wave.getBoundingClientRect();
        const x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        abSeekTo(x * (abDuration || durationSec));
      };
    }
  }
}

function paintWaveformInto(svg, peaks, role, playPct) {
  // One-time build of the wave geometry. The playhead lives as a separate
  // group whose attributes we mutate in-place every frame — see
  // setPlayheadPosition / updatePlayhead. Crucial: never call innerHTML on
  // this SVG again after this initial build, otherwise the SVG's children
  // are detached/recreated every animation frame, which loses the
  // mousedown/mouseup pairing needed for click events to fire during
  // playback.
  if (!peaks) return;
  const W = 800, H = 220;
  const N = peaks.length;
  const step = W / (N - 1);
  let dTop = `M 0 ${H / 2}`;
  let dBot = `M 0 ${H / 2}`;
  for (let i = 0; i < N; i++) {
    const x = i * step;
    const yT = H / 2 - peaks[i] * (H / 2) * 0.7;
    const yB = H / 2 + peaks[i] * (H / 2) * 0.7;
    dTop += ` L ${x.toFixed(2)} ${yT.toFixed(2)}`;
    dBot += ` L ${x.toFixed(2)} ${yB.toFixed(2)}`;
  }
  const color = role === "a" ? "var(--a-tint)" : "var(--b-tint)";
  svg.innerHTML = `
    <defs>
      <clipPath id="clip-played-${role}"><rect data-clip="played" x="0" y="0" width="0" height="${H}"/></clipPath>
      <clipPath id="clip-future-${role}"><rect data-clip="future" x="0" y="0" width="${W}" height="${H}"/></clipPath>
    </defs>
    <g clip-path="url(#clip-future-${role})" opacity="0.42">
      <path d="${dTop} L ${W} ${H/2} Z" fill="${color}" opacity="0.18"/>
      <path d="${dBot} L ${W} ${H/2} Z" fill="${color}" opacity="0.18"/>
      <path d="${dTop}" stroke="${color}" stroke-width="1" fill="none"/>
      <path d="${dBot}" stroke="${color}" stroke-width="1" fill="none"/>
    </g>
    <g clip-path="url(#clip-played-${role})">
      <path d="${dTop} L ${W} ${H/2} Z" fill="${color}" opacity="0.35"/>
      <path d="${dBot} L ${W} ${H/2} Z" fill="${color}" opacity="0.35"/>
      <path d="${dTop}" stroke="${color}" stroke-width="1.2" fill="none"/>
      <path d="${dBot}" stroke="${color}" stroke-width="1.2" fill="none"/>
    </g>
    <line x1="0" y1="${H/2}" x2="${W}" y2="${H/2}" stroke="${color}" stroke-opacity="0.25" stroke-width="0.5"/>
    <g data-playhead style="display:none">
      <line data-ph="line" x1="0" y1="0" x2="0" y2="${H}" stroke="var(--accent)" stroke-width="1"/>
      <circle data-ph="top" cx="0" cy="0" r="3" fill="var(--accent)"/>
      <circle data-ph="bot" cx="0" cy="${H}" r="3" fill="var(--accent)"/>
    </g>
  `;
  setPlayheadPosition(svg, playPct);
}

// Move the playhead and update the played/future clip regions without
// touching innerHTML — the SVG's children stay stable, so click events
// during playback aren't lost to the mousedown/mouseup target getting
// detached mid-gesture.
function setPlayheadPosition(svg, playPct) {
  if (!svg) return;
  const W = 800;
  const playX = playPct * W;
  const playedClip = svg.querySelector('[data-clip="played"]');
  const futureClip = svg.querySelector('[data-clip="future"]');
  const ph        = svg.querySelector('[data-playhead]');
  const phLine    = svg.querySelector('[data-ph="line"]');
  const phTop     = svg.querySelector('[data-ph="top"]');
  const phBot     = svg.querySelector('[data-ph="bot"]');
  if (playedClip) playedClip.setAttribute("width", String(playX));
  if (futureClip) {
    futureClip.setAttribute("x", String(playX));
    futureClip.setAttribute("width", String(Math.max(0, W - playX)));
  }
  if (ph) ph.style.display = playPct > 0 ? "" : "none";
  if (phLine) {
    phLine.setAttribute("x1", String(playX));
    phLine.setAttribute("x2", String(playX));
  }
  if (phTop) phTop.setAttribute("cx", String(playX));
  if (phBot) phBot.setAttribute("cx", String(playX));
}

function paintMiniWave(svg, peaks) {
  if (!peaks) return;
  const W = 56, H = 26;
  const N = peaks.length;
  const step = W / (N - 1);
  let dT = `M 0 ${H/2}`, dB = `M 0 ${H/2}`;
  for (let i = 0; i < N; i++) {
    const x = i * step;
    dT += ` L ${x.toFixed(1)} ${(H/2 - peaks[i] * (H/2) * 0.9).toFixed(1)}`;
    dB += ` L ${x.toFixed(1)} ${(H/2 + peaks[i] * (H/2) * 0.9).toFixed(1)}`;
  }
  svg.innerHTML = `
    <line x1="0" y1="${H/2}" x2="${W}" y2="${H/2}" stroke="var(--b-tint)" stroke-opacity="0.3" stroke-width="0.5"/>
    <path d="${dT}" stroke="var(--b-tint)" stroke-width="1" fill="none" opacity="0.85"/>
    <path d="${dB}" stroke="var(--b-tint)" stroke-width="1" fill="none" opacity="0.85"/>
  `;
}

// Update only the playhead overlay (just mutates attributes — no innerHTML).
function updatePlayhead(bodyEl, role, playPct) {
  if (!bodyEl || !bodyEl._svg) return;
  setPlayheadPosition(bodyEl._svg, playPct);
}

// ─── Variants ───────────────────────────────────────────────────────────────
function pickVariant(i) {
  if (i < 0 || i >= state.variants.length) return;
  state.activeVariantIdx = i;
  state.mode = "B";
  abLoad(state.inputBuffer, state.variants[i].audioBuffer);
  abSetMode("B");
  render();
}

function removeVariant(i) {
  const removed = state.variants[i];
  if (removed?.wavUrl) URL.revokeObjectURL(removed.wavUrl);
  state.variants.splice(i, 1);
  if (state.variants.length === 0) {
    state.stage = "ready";
    state.activeVariantIdx = 0;
    abReset();
  } else {
    state.activeVariantIdx = Math.min(state.activeVariantIdx, state.variants.length - 1);
    abLoad(state.inputBuffer, state.variants[state.activeVariantIdx].audioBuffer);
  }
  render();
}

// ─── Remix flow ────────────────────────────────────────────────────────────
remixBtn.addEventListener("click", () => startRemix());

let progressTimer = null;
let stageStarts = {};

async function startRemix() {
  if (state.variants.length >= MAX_VARIANTS) return;
  if (!state.inputBuffer) return;
  hideError();
  state.stage = "remixing";
  remixBtn.disabled = true;
  remixArr.textContent = "◐";
  remixLabel.textContent = "Rendering…";
  render();

  const variantNumber = state.variants.length + 1;
  setupProgressCard(variantNumber);

  try {
    let bufferToSend = state.inputBuffer;
    const stripBefore = state.separateOn;
    const addAfter = state.readdOn;
    const needSep = stripBefore || addAfter;

    if (needSep) {
      enterStage("separate");
      if (!state.instrumentalBuffer || !state.vocalsBuffer) {
        try {
          if (!demucsReady()) {
            await initDemucs({
              modelUrl: DEMUCS_MODEL_URL,
              wasmModuleUrl: DEMUCS_WASM_MODULE_URL,
              onProgress: (msg) => (pcSubtitle.textContent = msg),
            });
          }
          const stems = await separate(state.inputBuffer, (msg, pct) => {
            if (pct != null) {
              pcSubtitle.textContent = `Separating vocals · ${Math.round(pct * 100)}%`;
              setStageProgress("separate", pct);
            } else {
              pcSubtitle.textContent = msg;
            }
          });
          state.vocalsBuffer = stems.vocals;
          state.instrumentalBuffer = sumBuffers([stems.drums, stems.bass, stems.other]);
        } catch (e) {
          throw new Error(demucsErrorMessage(e));
        }
      } else {
        pcSubtitle.textContent = "Reusing cached separation…";
      }
      if (stripBefore) bufferToSend = state.instrumentalBuffer;
    } else {
      markStageSkipped("separate");
    }

    enterStage("upload");
    pcSubtitle.textContent = "Encoding…";
    const wavBlob = encodeWav(bufferToSend);

    const jobId = crypto.randomUUID();
    const fd = new FormData();
    const inputBase = (state.file.name.replace(/\.[^.]+$/, "") || "track");
    fd.append("audio", wavBlob, `${inputBase}-input.wav`);
    fd.append("prompt", buildPrompt());
    fd.append("init_noise_level", noiseEl.value);
    fd.append("steps", "8");
    fd.append("cfg_scale", cfgEl.value);
    fd.append("job_id", jobId);

    // Subscribe to GPU step events before the POST so they're not missed.
    const sse = new EventSource(`/api/remix/events/${jobId}`);
    sse.addEventListener("progress", (e) => {
      const m = JSON.parse(e.data);
      if (m.stage === "gpu" && m.step != null && m.total) {
        const pct = m.step / m.total;
        pcSubtitle.textContent = `Generating audio · step ${m.step} / ${m.total}`;
        const stgEl = pcStageList.querySelector('.stg[data-id="gpu"] .stg-time');
        if (stgEl) stgEl.textContent = `step ${m.step} / ${m.total}`;
        setStageProgress("gpu", pct);
        updateScanFrac(pct);
      }
    });
    sse.addEventListener("done", () => sse.close());

    const t0 = performance.now();
    let remixArrayBuf;
    try {
      remixArrayBuf = await uploadWithProgress("/api/remix", fd, (loaded, total) => {
        pcSubtitle.textContent = `Uploading instrumental · ${formatBytes(loaded)} / ${formatBytes(total)}`;
        if (total > 0) setStageProgress("upload", Math.min(1, loaded / total));
        if (loaded >= total) {
          // Once upload finishes, we're waiting on the GPU. Server-side SSE will
          // begin emitting step events shortly. Reveal the scan animation now —
          // it should only appear during the GPU stage, not while we're still
          // separating or uploading.
          enterStage("gpu");
          pcSubtitle.textContent = "Generating audio…";
          beginScanAnim();
        }
      });
    } finally {
      sse.close();
    }
    const runtime = ((performance.now() - t0) / 1000).toFixed(1) + "s";

    enterStage("mixdown");
    endScanAnim();
    pcSubtitle.textContent = "Decoding remix…";
    const decodeCtx = new (window.AudioContext || window.webkitAudioContext)();
    let remixBuf = await decodeCtx.decodeAudioData(remixArrayBuf.slice(0));
    await decodeCtx.close();

    let finalBuffer = remixBuf;
    if (addAfter && state.vocalsBuffer) {
      pcSubtitle.textContent = "Mixing vocals back in…";
      finalBuffer = await mixVocalsBack(remixBuf, state.vocalsBuffer, parseFloat(vocalGainEl.value));
    } else {
      markStageSkipped("mixdown");
    }

    const finalBlob = encodeWav(finalBuffer);
    if (lastWavUrl) URL.revokeObjectURL(lastWavUrl);
    lastWavUrl = URL.createObjectURL(finalBlob);

    const promptForLabel = promptEl.value.trim() || "Remix";
    const variant = {
      n: variantNumber,
      prompt: buildPrompt(),
      label: shortenLabel(promptForLabel),
      runtime,
      audioBuffer: finalBuffer,
      peaks: getPeaks(finalBuffer, 420),
      wavUrl: lastWavUrl,
    };
    state.variants.push(variant);
    state.activeVariantIdx = state.variants.length - 1;
    state.stage = "compared";
    state.mode = "B";
    abLoad(state.inputBuffer, variant.audioBuffer);
    abSetMode("B");

    // Mark all stages done & finish progress
    finishAllStages();
  } catch (e) {
    console.error(e);
    if (e.status === 503 && e.busyReason === "gpu") {
      // Friendlier framing for the "another user is generating" case.
      showError("Another user is currently generating on this server. Please wait a few seconds and try again.");
    } else {
      showError(e.message);
    }
    state.stage = state.variants.length > 0 ? "compared" : "ready";
  } finally {
    endScanAnim();
    stopProgressTimer();
    remixBtn.disabled = false;
    remixArr.textContent = "▶";
    remixLabel.textContent = "Remix track";
    render();
  }
}

// Per-stage fraction (0–1). Live signals from demucs / XHR / SSE push into this.
const stageFracs = { separate: 0, upload: 0, gpu: 0, mixdown: 0 };
const stageWeights = Object.fromEntries(STAGES.map((s) => [s.id, s.weight]));

function setStageProgress(id, frac) {
  stageFracs[id] = Math.max(0, Math.min(1, frac));
  // Overall bar = sum of (weight × frac) across all stages.
  let total = 0;
  for (const s of STAGES) total += s.weight * stageFracs[s.id];
  pcFill.style.width = `${Math.min(100, total * 100)}%`;
}

function setupProgressCard(variantNum) {
  pcTitle.textContent = `Rendering variation ${variantNum}`;
  pcSubtitle.textContent = STAGES[0].label + "…";
  pcElapsed.textContent = "0.0";
  pcEta.textContent = "";
  pcFill.style.width = "0%";
  pcStageList.innerHTML = STAGES.map(
    (s, i) =>
      `<div class="stg pending" data-id="${s.id}">
        <div class="num">${String(i + 1).padStart(2, "0")}</div>
        <div class="stg-title">${s.label}</div>
        <div class="stg-time">—</div>
      </div>`
  ).join("");
  for (const id of Object.keys(stageFracs)) stageFracs[id] = 0;
  stageStarts = {};
  const t0 = performance.now();
  stopProgressTimer();
  progressTimer = setInterval(() => {
    const elapsed = (performance.now() - t0) / 1000;
    pcElapsed.textContent = elapsed.toFixed(1);
    // Estimate remaining from the real overall fraction once it's meaningful.
    let overall = 0;
    for (const s of STAGES) overall += s.weight * stageFracs[s.id];
    if (overall > 0.05) {
      const remain = Math.max(0, elapsed * (1 / overall - 1));
      pcEta.textContent = `~${remain.toFixed(0)}s remaining`;
    }
    // Tick the active stage's running clock if no other signal has set it.
    const activeEl = pcStageList.querySelector(".stg.active");
    if (activeEl) {
      const id = activeEl.dataset.id;
      const start = stageStarts[id];
      const cell = activeEl.querySelector(".stg-time");
      // Only overwrite the time cell if it's still a plain duration (no live
      // signal text like "step 3 / 8" or "1.2 MB / 4.0 MB").
      if (start != null && /^[\d.]+s$|^—$/.test(cell.textContent)) {
        cell.textContent = `${((performance.now() - start) / 1000).toFixed(1)}s`;
      }
    }
  }, 100);
}

function enterStage(id) {
  // Mark previous active as done, mark this one active.
  const prevActive = pcStageList.querySelector(".stg.active");
  if (prevActive) {
    prevActive.classList.remove("active");
    prevActive.classList.add("done");
    const pid = prevActive.dataset.id;
    if (stageStarts[pid] != null) {
      prevActive.querySelector(".stg-time").textContent = `${((performance.now() - stageStarts[pid]) / 1000).toFixed(1)}s`;
    }
    setStageProgress(pid, 1); // lock at 100% so the bar holds the gain
  }
  const el = pcStageList.querySelector(`.stg[data-id="${id}"]`);
  if (el) {
    el.classList.remove("pending");
    el.classList.add("active");
    stageStarts[id] = performance.now();
  }
}

function markStageSkipped(id) {
  const el = pcStageList.querySelector(`.stg[data-id="${id}"]`);
  if (el) {
    el.classList.remove("pending", "active");
    el.classList.add("done");
    el.querySelector(".stg-time").textContent = "skipped";
  }
  setStageProgress(id, 1);
}

function finishAllStages() {
  const active = pcStageList.querySelector(".stg.active");
  if (active) {
    active.classList.remove("active");
    active.classList.add("done");
    const id = active.dataset.id;
    if (stageStarts[id] != null) {
      active.querySelector(".stg-time").textContent = `${((performance.now() - stageStarts[id]) / 1000).toFixed(1)}s`;
    }
    setStageProgress(id, 1);
  }
  for (const s of STAGES) setStageProgress(s.id, 1);
  pcFill.style.width = "100%";
}

function stopProgressTimer() {
  if (progressTimer) { clearInterval(progressTimer); progressTimer = null; }
}

// Scan-bar animation. Driven by the live GPU step counter (server SSE),
// falling back to elapsed-time interpolation if no step has landed yet.
let scanRAF = 0;
let scanT0 = 0;
let scanGpuFrac = null; // 0–1 when the latest SSE step has arrived
function beginScanAnim() {
  if (bBody._overlay) bBody._overlay.hidden = false;
  scanT0 = performance.now();
  scanGpuFrac = null;
  const step = () => {
    const elapsed = (performance.now() - scanT0) / 1000;
    // Prefer the real progress from setStageProgress when available.
    const pct = scanGpuFrac != null
      ? scanGpuFrac
      : (elapsed / (ETA_BASE_SEC + 4)) % 1;
    if (bBody._scan) bBody._scan.style.left = `${6 + pct * 88}%`;
    if (bBody._ovReadout) bBody._ovReadout.textContent = `generating · ${Math.min(100, Math.floor(pct * 100))}%`;
    scanRAF = requestAnimationFrame(step);
  };
  scanRAF = requestAnimationFrame(step);
}
function endScanAnim() {
  if (scanRAF) { cancelAnimationFrame(scanRAF); scanRAF = 0; }
  if (bBody._overlay) bBody._overlay.hidden = true;
}
function updateScanFrac(frac) {
  scanGpuFrac = Math.max(0, Math.min(1, frac));
}

// ─── A/B player (Web Audio crossfade) ───────────────────────────────────────
let abCtx = null;
let abBufA = null, abBufB = null;
let abGainA = null, abGainB = null;
let abSrcA = null, abSrcB = null;
let abPlaying = false;
let abStartedAt = 0;
let abOffset = 0;
let abDuration = 0;
let abRAF = 0;
let abSeeking = false;
const RAMP_TAU = 0.004;

function abReset() {
  abTeardown();
  abBufA = abBufB = null;
  abOffset = 0;
  abDuration = 0;
  tpFill.style.width = "0%";
  tpHead.style.left = "0%";
  tpNow.textContent = "0:00";
  tpTotal.textContent = formatTime(0);
  setPlayIcon(false);
}

function abLoad(a, b) {
  abTeardown();
  if (!abCtx) abCtx = new (window.AudioContext || window.webkitAudioContext)();
  abBufA = a; abBufB = b;
  abDuration = Math.min(a.duration, b.duration);
  abOffset = 0;
  tpFill.style.width = "0%";
  tpHead.style.left = "0%";
  tpNow.textContent = "0:00";
  tpTotal.textContent = formatTime(abDuration);
  setPlayIcon(false);
}

function abTeardown() {
  if (abRAF) { cancelAnimationFrame(abRAF); abRAF = 0; }
  for (const s of [abSrcA, abSrcB]) {
    if (s) {
      // Clear onended FIRST — stop() fires it asynchronously, and a stale
      // callback firing after the next abStart would incorrectly reset offset
      // to 0 (the natural-end handler), which broke seek-during-playback.
      s.onended = null;
      try { s.stop(); } catch {}
      try { s.disconnect(); } catch {}
    }
  }
  for (const g of [abGainA, abGainB]) {
    if (g) { try { g.disconnect(); } catch {} }
  }
  abSrcA = abSrcB = abGainA = abGainB = null;
  abPlaying = false;
}

function abStart(offsetSec) {
  if (!abBufA || !abBufB) return;
  abTeardown();
  if (abCtx.state === "suspended") abCtx.resume();
  abGainA = abCtx.createGain();
  abGainB = abCtx.createGain();
  abGainA.gain.value = state.mode === "A" ? 1 : 0;
  abGainB.gain.value = state.mode === "B" ? 1 : 0;
  abGainA.connect(abCtx.destination);
  abGainB.connect(abCtx.destination);
  abSrcA = abCtx.createBufferSource(); abSrcA.buffer = abBufA; abSrcA.connect(abGainA);
  abSrcB = abCtx.createBufferSource(); abSrcB.buffer = abBufB; abSrcB.connect(abGainB);
  const off = Math.max(0, Math.min(offsetSec, abDuration));
  abOffset = off;
  abStartedAt = abCtx.currentTime;
  abSrcA.start(0, off);
  abSrcB.start(0, off);
  abPlaying = true;
  setPlayIcon(true);
  const ender = abBufA.duration <= abBufB.duration ? abSrcA : abSrcB;
  ender.onended = () => {
    if (!abPlaying) return;
    abPause();
    abOffset = 0;
    tpFill.style.width = "0%";
    tpHead.style.left = "0%";
    tpNow.textContent = "0:00";
    paintPlayheads(0);
  };
  abLoop();
}

function abPause() {
  if (!abPlaying) return;
  abOffset = abCurrentTime();
  abTeardown();
  setPlayIcon(false);
}

function abCurrentTime() {
  if (!abPlaying) return abOffset;
  return Math.min(abOffset + (abCtx.currentTime - abStartedAt), abDuration);
}

function abLoop() {
  abRAF = requestAnimationFrame(() => {
    if (!abSeeking) {
      const t = abCurrentTime();
      const pct = abDuration > 0 ? t / abDuration : 0;
      tpFill.style.width = `${pct * 100}%`;
      tpHead.style.left = `${pct * 100}%`;
      tpNow.textContent = formatTime(t);
      paintPlayheads(pct);
    }
    if (abPlaying) abLoop();
  });
}

function paintPlayheads(pct) {
  updatePlayhead(aBody, "a", state.mode === "A" ? pct : 0);
  updatePlayhead(bBody, "b", state.mode === "B" ? pct : 0);
}

function abSetMode(mode) {
  if (mode !== "A" && mode !== "B") return;
  if (mode === "B" && state.variants.length === 0) return;
  state.mode = mode;
  if (abGainA && abGainB && abCtx) {
    const t = abCtx.currentTime;
    abGainA.gain.setTargetAtTime(mode === "A" ? 1 : 0, t, RAMP_TAU);
    abGainB.gain.setTargetAtTime(mode === "B" ? 1 : 0, t, RAMP_TAU);
  }
  updateTpListening();
  updateReelClickability();
  paintPlayheads(abDuration ? abCurrentTime() / abDuration : 0);
}

function setPlayIcon(playing) {
  tpPlayIcon.innerHTML = playing
    ? '<rect x="2" y="1" width="3.5" height="12" rx="0.5"/><rect x="8.5" y="1" width="3.5" height="12" rx="0.5"/>'
    : '<path d="M3 1.5 L12 7 L3 12.5 Z"/>';
}

// Click anywhere in a reel (except on interactive descendants) to switch A/B.
function reelClickToSwitch(targetMode) {
  return (e) => {
    if (state.stage !== "compared") return;
    if (targetMode === "B" && state.variants.length === 0) return;
    if (e.target.closest("button, select, a, input, textarea, [role='switch']")) return;
    if (state.mode === targetMode) return;
    abSetMode(targetMode);
  };
}
reelA.addEventListener("click", reelClickToSwitch("A"));
reelB.addEventListener("click", reelClickToSwitch("B"));

tpPlay.addEventListener("click", () => {
  if (!abBufA) {
    // If no remix yet, allow play of source only by treating itself as both A and B.
    if (state.inputBuffer && state.stage !== "empty") {
      abLoad(state.inputBuffer, state.inputBuffer);
      state.mode = "A";
      updateTpListening();
    } else return;
  }
  if (abPlaying) abPause();
  else abStart(abOffset);
});
swapBtn.addEventListener("click", () => {
  abSetMode(state.mode === "A" ? "B" : "A");
});
tpBar.addEventListener("click", (e) => {
  const rect = tpBar.getBoundingClientRect();
  const pct = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  abSeekTo(pct * (abDuration || state.inputBuffer?.duration || 0));
});

function abSeekTo(t) {
  // Auto-load source-only buffer if user seeks before pressing play in ready state.
  if (!abBufA) {
    if (state.inputBuffer && state.stage === "ready") {
      abLoad(state.inputBuffer, state.inputBuffer);
      state.mode = "A";
      updateTpListening();
    } else return;
  }
  const tt = Math.max(0, Math.min(t, abDuration));
  const pct = abDuration > 0 ? tt / abDuration : 0;
  const wasPlaying = abPlaying;
  if (wasPlaying) abPause();
  abOffset = tt;
  tpFill.style.width = `${pct * 100}%`;
  tpHead.style.left = `${pct * 100}%`;
  tpNow.textContent = formatTime(tt);
  paintPlayheads(pct);
  if (wasPlaying) abStart(abOffset);
}

// Keyboard
document.addEventListener("keydown", (e) => {
  if (state.stage === "empty" || state.stage === "remixing") return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea") return;
  if (e.code === "Space") { e.preventDefault(); tpPlay.click(); }
  else if (e.key === "x" || e.key === "X") { e.preventDefault(); swapBtn.click(); }
});

// ─── Audio helpers ─────────────────────────────────────────────────────────
function getPeaks(buffer, n = 420) {
  const data = buffer.getChannelData(0);
  const samplesPerPeak = Math.max(1, Math.floor(data.length / n));
  const peaks = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let max = 0;
    const start = i * samplesPerPeak;
    const end = Math.min(start + samplesPerPeak, data.length);
    for (let j = start; j < end; j++) {
      const v = Math.abs(data[j]);
      if (v > max) max = v;
    }
    peaks[i] = Math.min(1, max);
  }
  return peaks;
}

async function resample(buffer, targetSr) {
  const offline = new OfflineAudioContext(
    buffer.numberOfChannels,
    Math.ceil(buffer.duration * targetSr),
    targetSr
  );
  const src = offline.createBufferSource();
  src.buffer = buffer;
  src.connect(offline.destination);
  src.start();
  return await offline.startRendering();
}

function monoToStereo(buffer) {
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: buffer.sampleRate });
  const out = ctx.createBuffer(2, buffer.length, buffer.sampleRate);
  const m = buffer.getChannelData(0);
  out.getChannelData(0).set(m);
  out.getChannelData(1).set(m);
  ctx.close();
  return out;
}

function sumBuffers(buffers) {
  const length = buffers[0].length;
  const sr = buffers[0].sampleRate;
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sr });
  const out = ctx.createBuffer(2, length, sr);
  const oL = out.getChannelData(0), oR = out.getChannelData(1);
  for (const b of buffers) {
    const l = b.getChannelData(0);
    const r = b.numberOfChannels > 1 ? b.getChannelData(1) : l;
    for (let i = 0; i < length; i++) { oL[i] += l[i]; oR[i] += r[i]; }
  }
  ctx.close();
  return out;
}

async function mixVocalsBack(remixBuf, vocalsBuf, gain) {
  const sr = remixBuf.sampleRate;
  const vocals = vocalsBuf.sampleRate === sr ? vocalsBuf : await resample(vocalsBuf, sr);
  const length = Math.min(remixBuf.length, vocals.length);
  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: sr });
  const out = ctx.createBuffer(2, length, sr);
  for (let ch = 0; ch < 2; ch++) {
    const r = remixBuf.getChannelData(Math.min(ch, remixBuf.numberOfChannels - 1));
    const v = vocals.getChannelData(Math.min(ch, vocals.numberOfChannels - 1));
    const o = out.getChannelData(ch);
    for (let i = 0; i < length; i++) o[i] = r[i] + v[i] * gain;
  }
  ctx.close();
  return out;
}

function encodeWav(buffer) {
  const channels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const length = buffer.length;
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const dataBytes = length * blockAlign;
  const ab = new ArrayBuffer(44 + dataBytes);
  const view = new DataView(ab);
  function ws(off, s) { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); }
  ws(0, "RIFF"); view.setUint32(4, 36 + dataBytes, true); ws(8, "WAVE");
  ws(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, channels, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true); view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); ws(36, "data"); view.setUint32(40, dataBytes, true);
  const chans = [];
  for (let c = 0; c < channels; c++) chans.push(buffer.getChannelData(c));
  let off = 44;
  for (let i = 0; i < length; i++) {
    for (let c = 0; c < channels; c++) {
      let s = Math.max(-1, Math.min(1, chans[c][i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
  }
  return new Blob([ab], { type: "audio/wav" });
}

// ─── Misc helpers ──────────────────────────────────────────────────────────
function uploadWithProgress(url, fd, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.responseType = "arraybuffer";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded, e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(xhr.response);
      } else {
        // Attach status + parsed detail so callers can react to specific codes
        // (e.g. 503 with X-Busy-Reason: gpu = another user is generating).
        let detail = "";
        try {
          const txt = new TextDecoder().decode(xhr.response);
          if (txt) {
            try {
              const j = JSON.parse(txt);
              detail = j.detail || txt;
            } catch { detail = txt; }
          }
        } catch {}
        const err = new Error(detail || `${xhr.status} ${xhr.statusText}`);
        err.status = xhr.status;
        err.busyReason = xhr.getResponseHeader("X-Busy-Reason");
        reject(err);
      }
    };
    xhr.onerror = () => reject(new Error("Network error"));
    xhr.send(fd);
  });
}

function formatBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(secs) {
  if (!isFinite(secs) || secs < 0) secs = 0;
  const m = Math.floor(secs / 60);
  const s = Math.floor(secs % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}
function shortenLabel(p) {
  const clean = p.replace(/^TrackType:[^,]+,?\s*/i, "").replace(/^VocalType:[^,.]+\.?\s*/i, "");
  const trimmed = clean.length > 60 ? clean.slice(0, 60) + "…" : clean;
  return trimmed || "Remix";
}
function getBpmInt() {
  const v = parseFloat(bpmEl.value);
  if (isFinite(v) && v > 0) return v;
  if (state.analysis.bpm != null) return Math.round(state.analysis.bpm);
  return null;
}
function demucsErrorMessage(e) {
  const raw = (e && (e.message || e.toString())) || "unknown error";
  // Heuristically classify: missing-asset (404 / fetch failure) vs runtime
  // incompatibility (ORT init, WebAssembly compile, etc.).
  const r = raw.toLowerCase();
  const ua = navigator.userAgent || "";
  const isIOS = /ipad|iphone|ipod/.test(ua.toLowerCase()) ||
                (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
  if (r.includes("404") || r.includes("model fetch") || r.includes("failed to fetch") || r.includes("network")) {
    return "Vocal separation assets aren't installed on the server — htdemucs.onnx and the demucs WASM build need to be placed in web/static/{models,wasm}/. You can still remix by turning off both vocal toggles.";
  }
  if (isIOS) {
    return "Vocal separation didn't initialize. iOS Safari is currently the most fragile path for in-browser Demucs. Try on desktop Chrome/Edge, or turn off both vocal toggles to remix without separation.";
  }
  return (
    "Vocal separation couldn't run on this browser. It works best in desktop Chrome or Edge (WebGPU), and falls back to WASM on Safari/Firefox — yours doesn't appear to support either path. " +
    "Turn off both vocal toggles to remix the full track directly. " +
    "(" + raw + ")"
  );
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}
function hideError() { errorBox.hidden = true; }

// ─── Boot ──────────────────────────────────────────────────────────────────
render();
