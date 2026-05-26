// SA3 plugin — WebView front end.
//
// JUCE 8 injects window.__JUCE__.backend automatically when the editor is built
// with WebBrowserComponent::Options::withNativeIntegrationEnabled(true). All
// native calls go through a single "__juce__invoke" event channel and replies
// land in "__juce__complete" — exactly what the JUCE-bundled JS helper does.
// Inlining the helper here avoids the import/MIME-type ceremony of pulling in
// juce_gui_extra/native/javascript/index.js.

const promises = new Map();
let nextPromiseId = 0;

window.__JUCE__.backend.addEventListener("__juce__complete", ({ promiseId, result }) => {
  const handlers = promises.get(promiseId);
  if (!handlers) return;
  promises.delete(promiseId);
  handlers.resolve(result);
});

function getNativeFunction(name) {
  return (...args) => new Promise((resolve, reject) => {
    const promiseId = nextPromiseId++;
    promises.set(promiseId, { resolve, reject });
    window.__JUCE__.backend.emitEvent("__juce__invoke", {
      name,
      params: args,
      resultId: promiseId,
    });
  });
}

const getStatus  = getNativeFunction("getStatus");
const generate   = getNativeFunction("generate");
const getUiState = getNativeFunction("getUiState");
const setUiState = getNativeFunction("setUiState");

// ── DOM refs ─────────────────────────────────────────────────────────
const statusEl     = document.getElementById("status");
const errorBanner  = document.getElementById("error-banner");
const errorText    = document.getElementById("error-text");
const errorDismiss = document.getElementById("error-dismiss");
const dropEl       = document.getElementById("dropzone");
const sourceAudio  = document.getElementById("source-audio");
const sourceMeta   = document.getElementById("source-meta");
const presetSel    = document.getElementById("preset");
const noiseInput   = document.getElementById("noise");
const noiseValue   = document.getElementById("noise-value");
const bpmInput     = document.getElementById("bpm");
const keyInput     = document.getElementById("key");
const userPrompt   = document.getElementById("user-prompt");
const generateBtn  = document.getElementById("generate");
const variationsEl = document.getElementById("variations");
const appOnlyRows  = document.querySelectorAll(".app-only");

// ── State ─────────────────────────────────────────────────────────────
const state = {
  audioBase64:   null,
  audioMime:     null,
  fileName:      null,
  fileSeconds:   null,
  busy:          false,
  pipelineReady: false,
  // Set true after we've finished applying any persisted state; before then
  // we suppress save-on-change so rehydration doesn't echo back as a write.
  rehydrated:    false,
};

// 5-slot fallback labels — used if the engine response doesn't include
// steers (defensive; the engine always sets them post-2.3d).
const slotLabelsFree = ["1", "2", "3", "4", "5"];
const slotLabelsApp  = ["A1", "A2", "A3", "I1", "I2"];

// ── Filename BPM/Key sniffing (mirrors parse_bpm_key_from_filename in
// sa3_variations.py — same regexes, same precedence rules) ────────────
const KEY_TOKEN_RE = /^([A-G])([#b]?)(m|min|minor|maj|major)?$/i;

function matchKeyToken(part) {
  const m = KEY_TOKEN_RE.exec(part);
  if (!m) return null;
  const note = m[1].toUpperCase() + (m[2] || "");
  const qual = (m[3] || "").toLowerCase();
  if (qual === "m" || qual === "min" || qual === "minor") return `${note} minor`;
  if (qual === "maj" || qual === "major") return `${note} major`;
  return note;
}

function parseBpmKeyFromFilename(name) {
  const stem  = name.replace(/\.[^.]+$/, "");
  const parts = stem.split(/[_\-\s.]+/).filter(Boolean);

  let bpm = null, bpmIdx = -1;
  for (let i = 0; i < parts.length; ++i) {
    const m = /^(\d{2,3})bpm$/i.exec(parts[i]) || /^bpm(\d{2,3})$/i.exec(parts[i]);
    if (m) {
      const v = parseInt(m[1], 10);
      if (v >= 40 && v <= 240) { bpm = v; bpmIdx = i; break; }
    }
  }
  if (bpm === null) {
    for (let i = 0; i < parts.length; ++i) {
      if (/^\d+$/.test(parts[i])) {
        const v = parseInt(parts[i], 10);
        if (v >= 40 && v <= 240) { bpm = v; bpmIdx = i; break; }
      }
    }
  }
  let key = null;
  if (bpmIdx >= 0 && bpmIdx + 1 < parts.length) key = matchKeyToken(parts[bpmIdx + 1]);
  if (key === null && bpmIdx >= 1)              key = matchKeyToken(parts[bpmIdx - 1]);
  if (key === null) {
    for (const p of parts) { key = matchKeyToken(p); if (key) break; }
  }
  return { bpm, key };
}

// ── Helpers ───────────────────────────────────────────────────────────
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload  = () => {
      const buf = new Uint8Array(reader.result);
      // btoa needs a binary string; chunked apply() avoids the call-stack
      // limit on multi-MB files.
      const CHUNK = 0x8000;
      let binary = "";
      for (let i = 0; i < buf.length; i += CHUNK) {
        binary += String.fromCharCode.apply(
          null, buf.subarray(i, i + CHUNK));
      }
      resolve(btoa(binary));
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsArrayBuffer(file);
  });
}

async function loadFile(file) {
  clearVariations();

  const b64 = await fileToBase64(file);
  state.audioBase64 = b64;
  state.audioMime   = file.type || "audio/wav";
  state.fileName    = file.name;

  sourceAudio.src    = `data:${state.audioMime};base64,${b64}`;
  sourceAudio.hidden = false;
  sourceAudio.onloadedmetadata = () => {
    state.fileSeconds = sourceAudio.duration || null;
    updateMeta();
    updateButton();
  };

  // Best-effort filename sniff — only fills fields the user hasn't touched.
  const { bpm, key } = parseBpmKeyFromFilename(file.name);
  if (bpm !== null && bpmInput.value === "") { bpmInput.value = String(bpm); saveUiSoon(); }
  if (key !== null && keyInput.value === "") { keyInput.value = key;         saveUiSoon(); }

  dropEl.classList.add("has-file");
  updateMeta();
  updateButton();
}

function updateMeta() {
  if (!state.fileName) {
    sourceMeta.textContent = "";
    return;
  }
  const d = state.fileSeconds
    ? ` · ${state.fileSeconds.toFixed(2)}s` : "";
  sourceMeta.textContent = state.fileName + d;
}

function updateButton() {
  const can = state.pipelineReady && state.audioBase64 && !state.busy;
  generateBtn.disabled = !can;
  generateBtn.textContent = state.busy
    ? "Generating..."
    : "Generate 5 variations";
}

function applyPresetVisibility() {
  const isApp = presetSel.value === "app";
  appOnlyRows.forEach((r) => { r.hidden = !isApp; });
}

function renderSlots(slots) {
  const fallback = presetSel.value === "app" ? slotLabelsApp : slotLabelsFree;
  variationsEl.innerHTML = "";
  for (let i = 0; i < slots.length && i < 5; i++) {
    const s = slots[i];
    const row = document.createElement("div");
    row.className = "variation";

    const label = document.createElement("span");
    label.className   = "variation-label";
    label.textContent = fallback[i] || String(i + 1);

    const steer = document.createElement("span");
    steer.className   = "variation-steer";
    steer.textContent = s.steer || "";
    steer.title       = s.steer || "";   // full text in tooltip when truncated

    const audio = document.createElement("audio");
    audio.controls = true;
    audio.preload  = "none";
    audio.src      = s.url;

    row.appendChild(label);
    row.appendChild(steer);
    row.appendChild(audio);
    variationsEl.appendChild(row);
  }
}

function clearVariations() {
  variationsEl.innerHTML = "";
}

// ── Error banner ──────────────────────────────────────────────────────
function showError(msg) {
  errorText.textContent = msg;
  errorBanner.hidden    = false;
}
function clearError() {
  errorBanner.hidden    = true;
  errorText.textContent = "";
}
errorDismiss.addEventListener("click", clearError);

function showStatusFromBackend(s) {
  if (!s || typeof s !== "object") {
    statusEl.textContent = String(s ?? "");
    statusEl.className   = "status status-loading";
    return;
  }
  const phase   = String(s.phase || "");
  const status  = String(s.status ?? "");
  const busy    = !!s.busy;

  state.busy          = busy;
  state.pipelineReady = phase === "loaded";
  updateButton();

  let cls = "status";
  if      (phase === "error") cls += " status-error";
  else if (! state.pipelineReady) cls += " status-loading";
  else if (busy)              cls += " status-busy";
  else if (status.startsWith("Error")) cls += " status-error";
  else                        cls += " status-ready";
  statusEl.textContent = status;
  statusEl.className   = cls;

  // Mirror persistent errors into the banner so the user can't miss them.
  // Recoverable transitions (Loading → Ready) auto-clear it.
  if (phase === "error" || status.startsWith("Error")) {
    showError(status);
  }
}

async function pollStatus() {
  try { showStatusFromBackend(await getStatus()); }
  catch (e) { /* swallow — next tick retries */ }
}

// ── UI state persistence ──────────────────────────────────────────────
// We round-trip a small JSON object through getStateInformation. The audio
// file isn't persisted (would bloat project files); the user re-drops on
// reload. Saves are debounced so dragging the noise slider doesn't fire a
// native call per pixel.
function snapshotUi() {
  return {
    preset:     presetSel.value,
    noise:      Number(noiseInput.value),
    bpm:        bpmInput.value,
    key:        keyInput.value,
    userPrompt: userPrompt.value,
  };
}

let saveTimer = null;
function saveUiSoon() {
  if (! state.rehydrated) return;
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try { await setUiState(JSON.stringify(snapshotUi())); }
    catch (e) { /* non-fatal; the DAW save just won't capture latest state */ }
  }, 200);
}

async function rehydrateUi() {
  try {
    const blob = await getUiState();
    const json = typeof blob === "string" ? blob : "";
    if (json) {
      const s = JSON.parse(json);
      if (s.preset === "free" || s.preset === "app") presetSel.value = s.preset;
      if (Number.isFinite(s.noise))                  noiseInput.value = s.noise;
      if (typeof s.bpm === "string")                 bpmInput.value   = s.bpm;
      if (typeof s.key === "string")                 keyInput.value   = s.key;
      if (typeof s.userPrompt === "string")          userPrompt.value = s.userPrompt;
    }
  }
  catch (e) { /* corrupt blob — fall back to defaults */ }
  noiseValue.textContent = Number(noiseInput.value).toFixed(2);
  applyPresetVisibility();
  state.rehydrated = true;
}

// ── Events ────────────────────────────────────────────────────────────
dropEl.addEventListener("click", () => {
  const input = document.createElement("input");
  input.type   = "file";
  input.accept = "audio/wav,audio/aiff,audio/flac,audio/ogg,.wav,.aiff,.aif,.flac,.ogg";
  input.onchange = () => input.files[0] && loadFile(input.files[0]);
  input.click();
});

["dragenter", "dragover"].forEach((ev) => {
  dropEl.addEventListener(ev, (e) => {
    e.preventDefault();
    dropEl.classList.add("drag-over");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropEl.addEventListener(ev, (e) => {
    e.preventDefault();
    dropEl.classList.remove("drag-over");
  });
});
dropEl.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) loadFile(file);
});

noiseInput.addEventListener("input", () => {
  noiseValue.textContent = Number(noiseInput.value).toFixed(2);
  saveUiSoon();
});

presetSel.addEventListener("change", () => {
  applyPresetVisibility();
  clearVariations();
  saveUiSoon();
});

bpmInput.addEventListener("input",   saveUiSoon);
keyInput.addEventListener("input",   saveUiSoon);
userPrompt.addEventListener("input", saveUiSoon);

generateBtn.addEventListener("click", async () => {
  if (! state.audioBase64) return;
  clearError();
  clearVariations();
  state.busy = true; updateButton();

  const req = {
    preset:       presetSel.value,
    seconds:      state.fileSeconds || 0,
    noise:        Number(noiseInput.value),
    bpm:          presetSel.value === "app" && bpmInput.value
                    ? Number(bpmInput.value) : null,
    key:          presetSel.value === "app" ? keyInput.value : "",
    userPrompt:   presetSel.value === "app" ? userPrompt.value : "",
    audioBase64:  state.audioBase64,
    audioMime:    state.audioMime,
    seed:         Math.floor(Math.random() * 1_000_000),
    steps:        8,
  };

  try {
    const res = await generate(req);
    if (! res || ! res.ok) {
      showError(res && res.error || "unknown error");
    } else {
      renderSlots(res.slots || []);
    }
  } catch (err) {
    showError("Bridge error: " + err);
  } finally {
    state.busy = false; updateButton();
  }
});

// ── Boot ──────────────────────────────────────────────────────────────
clearVariations();
rehydrateUi();
pollStatus();
setInterval(pollStatus, 250);
