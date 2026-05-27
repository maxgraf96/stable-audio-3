// SA3 plugin — Hairline WebView front end.
//
// All audio I/O is JUCE-native: the source file is decoded by the engine on
// upload (and peaks come back across the bridge), the variation WAVs live
// as juce::AudioBuffer<float> in the engine, and playback is driven by
// Processor::processBlock pulling from the engine. The WebView never touches
// AudioContext or <audio> — that's important for running in a DAW where the
// host owns the audio device.

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

const getStatus = getNativeFunction("getStatus");
const getUiState = getNativeFunction("getUiState");
const setUiState = getNativeFunction("setUiState");
const uploadSource = getNativeFunction("uploadSource");
const generate = getNativeFunction("generate");
const play = getNativeFunction("play");
const pause = getNativeFunction("pause");
const stop = getNativeFunction("stop");
const seek = getNativeFunction("seek");
const getPlayState = getNativeFunction("getPlayState");
const setOneShotMode           = getNativeFunction("setOneShotMode");
const copyVariationToClipboard = getNativeFunction("copyVariationToClipboard");

// Fire a JUCE event (vs a native-function call). emitEvent is
// fire-and-forget and dispatches the listener on the C++ message thread
// inside the same run-loop tick as the calling JS — which is what makes
// it usable for drag-out: AppKit still has the originating mouseDown in
// queue when JUCE calls performExternalDragDropOfFiles.
function emitJuceEvent(name, data) {
    if (window.__JUCE__ && window.__JUCE__.backend &&
        typeof window.__JUCE__.backend.emitEvent === "function") {
        window.__JUCE__.backend.emitEvent(name, data);
    }
}

// dd-mm-yy-hh-mm-ss in the user's local time. Stamped once per
// generation so all 5 variations share it and DAW imports stay
// referentially stable across re-generations.
function makeGenerationStamp() {
    const d = new Date();
    const p = n => String(n).padStart(2, "0");
    return `${p(d.getDate())}-${p(d.getMonth() + 1)}-${String(d.getFullYear()).slice(-2)}`
         + `-${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
}

// Strip extension from the source file name; fall back to "SA3" if no
// source has been uploaded yet.
function variationBaseName() {
    const n = (state.fileName || "SA3").replace(/\.[^.]+$/, "");
    return n || "SA3";
}

// ── Constants ────────────────────────────────────────────────────────
const PRESETS = [
    { value: "free", label: "Free variation", sub: "unconditional" },
    { value: "app", label: "Preserve sound", sub: "steered · bar-aware" },
];

const SOURCE_PEAKS_N = 220;
const SLOT_PEAKS_N = 140;

const COLOR_FG = "oklch(0.96 0.005 60)";
const COLOR_ACCENT = "oklch(0.84 0.05 60)";

// ── DOM refs ─────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const statusEl = $("status");
const statusText = $("status-text");
const errorBanner = $("error-banner");
const errorText = $("error-text");
const errorDismiss = $("error-dismiss");

const sourceEl = $("source");
const sourceEmpty = $("source-empty");
const sourceLoaded = $("source-loaded");
const srcNameEl = $("srcname-text");
const srcDurEl = $("srcdur-text");
const srcWaveEl = $("srcwave");
const srcPlayBtn = $("src-play");
const srcTimeCurEl = $("src-time-cur");
const srcTimeRemEl = $("src-time-rem");

const presetEl = $("preset");
const presetToggle = $("preset-toggle");
const presetLabelEl = $("preset-label");
const presetSubEl = $("preset-sub");
const presetMenuEl = $("preset-menu");

const bpmInput = $("bpm");
const keyInput = $("key");
const userPrompt = $("user-prompt");

const noiseInput = $("noise");
const noiseValueEl = $("noise-value");
const noiseHintEl = $("noise-hint");

const generateBtn = $("generate");
const genLabelEl = generateBtn.querySelector(".gen-label");
const genProgressEl = $("gen-progress");
const resultsEl = $("results");
const resMetaEl = $("res-meta");
const slotsEl = $("slots");
const appOnlyRows = document.querySelectorAll(".app-only");
const modeToggleEl = $("mode-toggle");
const kbdHintEl = $("kbd-hint");

// ── State ────────────────────────────────────────────────────────────
const state = {
    fileName: null,
    hasSource: false,
    fileSeconds: null,
    sourcePeaks: null,
    preset: "free",
    presetOpen: false,
    variations: [],          // [{ steer, mode, peaks, duration }]
    expected: 0,
    busy: false,
    pipelineReady: false,
    rehydrated: false,
    // Mirror of native play state (updated by poll).
    playingIdx: -2,          // -2 idle, -1 source, 0..4 slot
    playing: false,
    progress: 0,
    // JS-only reentry guard for the Generate handler. Distinct from `busy`
    // because `busy` gets overwritten by every status poll — if a poll lands
    // in the window between click time and the engine flipping its own busy
    // flag, `busy` snaps back to false and a second click slips through.
    inFlight: false,
    // Loop vs one-shot playback. Auto-detected on file upload; user can flip
    // via the mode toggle. When true: cycling variations restarts from 0;
    // engine stops at end of buffer instead of looping.
    oneShot: false,
};

// Heuristic: filename keywords first (strong signal), then duration (short
// samples are usually one-shots). User can always override the toggle.
const ONE_SHOT_KEYWORDS = [
    "oneshot", "one-shot", "one_shot",
    "kick", "snare", "hat", "hihat", "clap", "perc", "tom",
    "hit", "shot", "stab", "stinger",
];
const LOOP_KEYWORDS = ["loop", "bar"];

function detectOneShot(fileName, durationSec) {
    const n = (fileName || "").toLowerCase();
    if (LOOP_KEYWORDS.some((kw) => n.includes(kw))) return false;
    if (ONE_SHOT_KEYWORDS.some((kw) => n.includes(kw))) return true;
    return Number.isFinite(durationSec) && durationSec > 0 && durationSec < 1.5;
}

// ── Time formatting ──────────────────────────────────────────────────
function fmtTime(s) {
    if (!isFinite(s) || s < 0) s = 0;
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return `${m}:${String(r).padStart(2, "0")}`;
}

// ── Waveform renderer (one SVG mount per call; setProgress is cheap) ──
function escapeAttr(s) { return String(s).replace(/"/g, "&quot;"); }

let waveformIdCounter = 0;

function mountWaveform(el, peaks, opts = {}) {
    const {
        kind = "line",
        height = 28,
        barWidth = 1,
        gap = 1,
        color = "currentColor",
        onSeek = null,
    } = opts;

    const totalWidth = peaks.length * (barWidth + gap) - gap;
    const cy = height / 2;
    const clipId = `wf-${++waveformIdCounter}`;

    let bgMarkup, fgMarkup;
    if (kind === "filled") {
        let d = `M 0 ${cy}`;
        for (let i = 0; i < peaks.length; i++) {
            const x = i * (barWidth + gap) + barWidth / 2;
            const half = peaks[i] * (height * 0.46);
            d += ` L ${x} ${cy - half}`;
        }
        for (let i = peaks.length - 1; i >= 0; i--) {
            const x = i * (barWidth + gap) + barWidth / 2;
            const half = peaks[i] * (height * 0.46);
            d += ` L ${x} ${cy + half}`;
        }
        d += " Z";
        bgMarkup = `<path d="${d}" fill="${escapeAttr(color)}" fill-opacity="0.22" />`;
        fgMarkup = `<path d="${d}" fill="${escapeAttr(color)}" />`;
    } else {
        const bg = [], fg = [];
        for (let i = 0; i < peaks.length; i++) {
            const x = i * (barWidth + gap) + barWidth / 2;
            const half = peaks[i] * (height * 0.45);
            const line = `x1="${x}" y1="${cy - half}" x2="${x}" y2="${cy + half}" stroke-width="${barWidth}" stroke-linecap="round"`;
            bg.push(`<line ${line} stroke="${escapeAttr(color)}" stroke-opacity="0.22" />`);
            fg.push(`<line ${line} stroke="${escapeAttr(color)}" />`);
        }
        bgMarkup = bg.join("");
        fgMarkup = fg.join("");
    }

    el.innerHTML = `
    <svg viewBox="0 0 ${totalWidth} ${height}" preserveAspectRatio="none"
         width="100%" height="${height}"
         style="display:block; cursor:${onSeek ? "pointer" : "default"}">
      <defs>
        <clipPath id="${clipId}">
          <rect class="wf-progress" x="0" y="0" width="0" height="${height}" />
        </clipPath>
      </defs>
      ${bgMarkup}
      <g clip-path="url(#${clipId})">${fgMarkup}</g>
    </svg>
  `;

    const svg = el.querySelector("svg");
    const progressRect = el.querySelector(".wf-progress");

    if (onSeek) {
        svg.addEventListener("click", (e) => {
            const r = svg.getBoundingClientRect();
            const p = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
            onSeek(p);
        });
    }

    return {
        setProgress(p) {
            const clamped = Math.max(0, Math.min(1, p));
            progressRect.setAttribute("width", String(totalWidth * clamped));
        },
    };
}

// ── BPM/Key filename sniff (port of parse_bpm_key_from_filename) ─────
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
    const stem = name.replace(/\.[^.]+$/, "");
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
    if (key === null && bpmIdx >= 1) key = matchKeyToken(parts[bpmIdx - 1]);
    if (key === null) {
        for (const p of parts) { key = matchKeyToken(p); if (key) break; }
    }
    return { bpm, key };
}

// ── File → base64 (chunked to avoid btoa() call-stack limits) ────────
function fileToBase64(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const buf = new Uint8Array(reader.result);
            const CHUNK = 0x8000;
            let binary = "";
            for (let i = 0; i < buf.length; i += CHUNK) {
                binary += String.fromCharCode.apply(null, buf.subarray(i, i + CHUNK));
            }
            resolve(btoa(binary));
        };
        reader.onerror = () => reject(reader.error);
        reader.readAsArrayBuffer(file);
    });
}

async function loadFile(file) {
    clearVariations();
    clearError();
    await stop();   // wipe any in-flight playback

    state.fileName = file.name;
    renderSource(/*pending=*/true);

    const b64 = await fileToBase64(file);

    // Native side decodes (any format JUCE supports), resamples to 44.1k
    // stereo, stores the AudioBuffer, computes peaks, returns them here.
    let resp;
    try {
        resp = await uploadSource({
            audioBase64: b64,
            peaksN: SOURCE_PEAKS_N,
        });
    } catch (e) {
        showError("Source upload bridge error: " + e);
        return;
    }
    if (!resp || !resp.ok) {
        showError((resp && resp.error) || "uploadSource failed");
        state.hasSource = false;
        renderSource();
        return;
    }

    state.hasSource = true;
    state.fileSeconds = Number(resp.duration) || null;
    state.sourcePeaks = (resp.peaks || []).map(Number);

    // Best-effort filename sniff — only fills empty inputs.
    const { bpm, key } = parseBpmKeyFromFilename(file.name);
    if (bpm !== null && bpmInput.value === "") { bpmInput.value = String(bpm); saveUiSoon(); }
    if (key !== null && keyInput.value === "") { keyInput.value = key; saveUiSoon(); }

    // Auto-detect playback mode for the new source. Always applies on file
    // change — manual override is per-file (a stored preference would lie
    // for the next sample). The user can flip it after via the toggle.
    applyOneShot(detectOneShot(file.name, state.fileSeconds));

    renderSource();
    updateButton();
}

function applyOneShot(oneShot) {
    state.oneShot = !!oneShot;
    modeToggleEl.textContent = state.oneShot ? "One-shot" : "Loop";
    modeToggleEl.classList.toggle("one-shot", state.oneShot);
    setOneShotMode(state.oneShot);
    saveUiSoon();
}

modeToggleEl.addEventListener("click", () => applyOneShot(!state.oneShot));

// ── Render: source ───────────────────────────────────────────────────
let srcWfHandle = null;

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
        srcWfHandle = mountWaveform(srcWaveEl, state.sourcePeaks, {
            kind: "filled",
            height: 36,
            color: COLOR_FG,
            onSeek: (p) => seekIdx(-1, p),
        });
    } else {
        srcWaveEl.innerHTML = "";
        srcWfHandle = null;
    }
    updateSrcTimes();
    updateSrcPlayState();
}

// ── Render: results ──────────────────────────────────────────────────
const slotHandles = [];

function clearVariations() {
    state.variations = [];
    state.expected = 0;
    slotHandles.length = 0;
    slotsEl.innerHTML = "";
    resultsEl.hidden = true;
}

// Brief icon swap + accent flash on the copy button when the variation
// successfully landed on the clipboard.
function flashCopiedFeedback(btn, toast) {
    if (!btn) return;
    btn.classList.add("copied");
    const ic = btn.querySelector(".copy-ico");
    const ck = btn.querySelector(".check-ico");
    if (ic) ic.hidden = true;
    if (ck) ck.hidden = false;
    if (toast) toast.classList.add("show");
    setTimeout(() => {
        btn.classList.remove("copied");
        if (ic) ic.hidden = false;
        if (ck) ck.hidden = true;
        if (toast) toast.classList.remove("show");
    }, 1200);
}

function renderResults() {
    if (state.variations.length === 0 && state.expected === 0) {
        resultsEl.hidden = true;
        if (kbdHintEl) kbdHintEl.hidden = true;
        return;
    }
    resultsEl.hidden = false;
    if (kbdHintEl) kbdHintEl.hidden = false;
    const noiseDisplay = Number(noiseInput.value).toFixed(2);
    const count = Math.max(state.expected, state.variations.length);
    resMetaEl.textContent = `${count} · σ ${noiseDisplay}`;

    slotsEl.innerHTML = "";
    slotHandles.length = 0;

    for (let i = 0; i < count; i++) {
        const v = state.variations[i] || null;
        const ready = !!(v && v.peaks && v.peaks.length > 0);
        const active = state.playingIdx === i;

        const slot = document.createElement("div");
        slot.className = "slot"
            + (ready ? "" : " empty")
            + (active ? " active" : "");
        slot.innerHTML = `
      <span class="idx">${String(i + 1).padStart(2, "0")}</span>
      <div class="wf"></div>
      <span class="dur"></span>
      <button type="button" class="row-play" aria-label="Play variation ${i + 1}">
        <svg class="play-ico"  width="10" height="10" viewBox="0 0 10 10"><polygon points="2.5,2 8.5,5 2.5,8" fill="currentColor"/></svg>
        <svg class="pause-ico" width="10" height="10" viewBox="0 0 10 10"><rect x="2" y="2" width="2" height="6" fill="currentColor"/><rect x="6" y="2" width="2" height="6" fill="currentColor"/></svg>
      </button>
      <span class="steer"></span>
      <button type="button" class="row-copy" aria-label="Copy variation ${i + 1} to clipboard (paste with ⌘V)">
        <svg class="copy-ico"  width="11" height="11" viewBox="0 0 11 11"><rect x="2.2" y="2.6" width="6" height="7.2" rx="0.8" stroke="currentColor" stroke-width="0.7" fill="none"/><rect x="3.5" y="1.4" width="3.4" height="1.5" rx="0.4" fill="currentColor"/></svg>
        <svg class="check-ico" width="11" height="11" viewBox="0 0 11 11" hidden><polyline points="2,6 4.5,8.5 9,3.5" stroke="currentColor" stroke-width="1.1" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
      <span class="copy-toast" aria-hidden="true">Copied to clipboard</span>
      <div class="row-grip" aria-label="Drag variation ${i + 1} to your DAW" title="Drag to your DAW">
        <svg width="10" height="14" viewBox="0 0 10 14" aria-hidden="true">
          <circle cx="3" cy="3"  r="1.1" fill="currentColor"/>
          <circle cx="7" cy="3"  r="1.1" fill="currentColor"/>
          <circle cx="3" cy="7"  r="1.1" fill="currentColor"/>
          <circle cx="7" cy="7"  r="1.1" fill="currentColor"/>
          <circle cx="3" cy="11" r="1.1" fill="currentColor"/>
          <circle cx="7" cy="11" r="1.1" fill="currentColor"/>
        </svg>
      </div>
    `;
        slotsEl.appendChild(slot);

        const wfHost = slot.querySelector(".wf");
        const steerEl = slot.querySelector(".steer");
        const durEl = slot.querySelector(".dur");
        const playBtn = slot.querySelector(".row-play");
        const copyBtn = slot.querySelector(".row-copy");
        const copyToast = slot.querySelector(".copy-toast");

        steerEl.textContent = v ? (v.steer || "") : "";
        durEl.textContent = (v && v.duration) ? `${v.duration.toFixed(2)}s` : "——";

        let wfMount = null;
        if (ready) {
            wfMount = mountWaveform(wfHost, v.peaks, {
                kind: "line",
                height: 22,
                color: active ? COLOR_ACCENT : COLOR_FG,
                onSeek: (p) => seekIdx(i, p),
            });
        } else {
            wfHost.innerHTML = `<div class="skel"></div>`;
            playBtn.disabled = true;
        }

        playBtn.addEventListener("click", (e) => {
            e.stopPropagation();   // don't double-fire the slot click
            if (!ready) return;
            togglePlay(i);
        });

        // Copy to clipboard — writes the variation to a temp WAV and
        // puts it on the system clipboard as a POSIX file via osascript.
        // ⌘V into Ableton / Finder lands the file. Pasted from wavnav's
        // approach since OS drag-out via WKWebView doesn't work.
        if (!ready) copyBtn.disabled = true;
        copyBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            if (!ready) return;
            try {
                const ok = await copyVariationToClipboard(
                    i, variationBaseName(), state.generationStamp || "");
                if (ok) flashCopiedFeedback(copyBtn, copyToast);
            } catch (_) { /* swallow */ }
        });

        // Clicking anywhere else on the row switches to (and plays) that
        // variation — UX shortcut. Skip clicks on the waveform (own seek
        // handler) and the play button (handled above).
        slot.addEventListener("click", (e) => {
            if (!ready) return;
            if (e.target.closest(".row-play")) return;
            if (e.target.closest("svg")) return;
            // play(idx) on the engine resets position when switching buffers,
            // so this is a "switch to this variation and start" not a toggle.
            play(i).then(() => getPlayState()).then(applyPlayState);
        });

        // Drag-out is handled by native JUCE overlay components positioned
        // on top of the slot's idx column (see reportSlotBoundsToNative
        // below). HTML5 dragstart from a WKWebView can't reliably start
        // an OS-level file drag — wavnav's pattern, mirrored here, is to
        // have JUCE handle the mouseDrag from a real NSView.
        slot.style.cursor = ready ? "pointer" : "default";

        // Drag-out: mousedown on the grip emits a JUCE event that fires
        // synchronously on the C++ message thread, which calls
        // performExternalDragDropOfFiles before AppKit's mouseDown ages
        // out — the only WKWebView-compatible way to start a real OS
        // file drag from inside the WebView. (Cf. JUCE forum #64813.)
        const grip = slot.querySelector(".row-grip");
        if (grip) {
            grip.addEventListener("mousedown", (e) => {
                if (!ready) return;
                e.preventDefault();   // suppress text-select cursor
                e.stopPropagation();  // don't trigger the slot's play click
                // Tell the user visually which row they're about to drag
                // out. Removed on global mouseup (so even a same-spot
                // release clears it).
                slot.classList.add("dragging");
                outboundDragInFlight = true;
                clearTimeout(outboundDragTimer);
                outboundDragTimer = setTimeout(() => {
                    outboundDragInFlight = false;
                    // NSDraggingSession sometimes swallows mouseUp on the
                    // initiator, so this is the belt-and-braces cleanup.
                    document.querySelectorAll(".slot.dragging")
                        .forEach(s => s.classList.remove("dragging"));
                }, 3000);
                emitJuceEvent("dragVariation", {
                    idx:      i,
                    baseName: variationBaseName(),
                    stamp:    state.generationStamp || "",
                });
            });
        }

        slotHandles[i] = { wfMount, slotEl: slot, playBtn, copyBtn };
    }
    updatePlayStateUI();
}

// ── Status / play polling ────────────────────────────────────────────
function updateStatus(s) {
    if (!s || typeof s !== "object") {
        statusText.textContent = String(s ?? "");
        statusEl.className = "status";
        return;
    }
    const phase = String(s.phase || "");
    const status = String(s.status || "");
    const busy = !!s.busy;

    state.busy = busy;
    state.pipelineReady = phase === "loaded";

    let cls = "status";
    let text = status;
    if (phase === "error") {
        cls += " error";
        text = "error";
    } else if (!state.pipelineReady) {
        text = phase === "loading" ? "Loading" : (status || "Idle");
    } else if (busy) {
        cls += " accent";
        text = status.replace(/\.\.\.$/, "");
    } else if (status.startsWith("Error")) {
        cls += " error";
        text = "Error";
    } else {
        text = "Ready";
    }
    statusText.textContent = text;
    statusEl.className = cls;

    if (phase === "error" || status.startsWith("Error")) {
        showError(status);
    }
    updateButton();
}

function applyPlayState(ps) {
    if (!ps || typeof ps !== "object") return;
    const prevIdx = state.playingIdx;
    const prevPlaying = state.playing;
    state.playingIdx = Number.isFinite(ps.idx) ? ps.idx : -2;
    state.playing = !!ps.playing;
    state.progress = Number.isFinite(ps.progress) ? ps.progress : 0;
    // Re-render the slot grid (cheap, ≤5 rows) when the active row changes
    // OR play/pause flips — keeps the colour tint in sync. Inline progress
    // overlay updates happen every poll.
    if (prevIdx !== state.playingIdx || prevPlaying !== state.playing) {
        updatePlayStateUI();
    }
    updateProgressOverlays();
    updateSrcTimes();
}

function updateButton() {
    const busy = state.busy || state.inFlight;
    const can = state.pipelineReady && state.hasSource && !busy;
    generateBtn.disabled = !can;
    generateBtn.classList.toggle("busy", busy);
    const m = /(\d+)\/(\d+)/.exec(statusText.textContent || "");
    if (busy) {
        genLabelEl.textContent = m
            ? `Generating  ${m[1]} / ${m[2]}`
            : "Generating…";
        if (m) {
            const pct = (Number(m[1]) / Number(m[2])) * 100;
            genProgressEl.style.width = pct + "%";
        }
    } else {
        genLabelEl.textContent = "Generate 5 variations";
        // Keep width where it landed — opacity fade hides it, so a snap
        // back to 0 would be visible as the bar shrinks behind the fade.
    }
}

// ── Error banner ─────────────────────────────────────────────────────
function showError(msg) { errorText.textContent = msg; errorBanner.hidden = false; }
function clearError() { errorBanner.hidden = true; errorText.textContent = ""; }
errorDismiss.addEventListener("click", clearError);

// ── Preset dropdown ──────────────────────────────────────────────────
function renderPreset() {
    const meta = PRESETS.find((p) => p.value === state.preset) || PRESETS[0];
    presetLabelEl.textContent = meta.label;
    presetSubEl.textContent = meta.sub;
    presetMenuEl.hidden = !state.presetOpen;

    appOnlyRows.forEach((r) => { r.hidden = state.preset !== "app"; });

    if (state.presetOpen) {
        presetMenuEl.innerHTML = PRESETS.map((p) => `
      <button type="button" class="opt ${p.value === state.preset ? "sel" : ""}"
              data-value="${p.value}">
        <span>${p.label}</span>
        <span class="sub">${p.sub}</span>
      </button>
    `).join("");
        presetMenuEl.querySelectorAll(".opt").forEach((btn) => {
            btn.addEventListener("click", () => {
                state.preset = btn.dataset.value;
                state.presetOpen = false;
                renderPreset();
                clearVariations();
                saveUiSoon();
            });
        });
    }
}

presetToggle.addEventListener("click", (e) => {
    e.stopPropagation();
    state.presetOpen = !state.presetOpen;
    renderPreset();
});
document.addEventListener("click", (e) => {
    if (!state.presetOpen) return;
    if (presetEl.contains(e.target)) return;
    state.presetOpen = false;
    renderPreset();
});

// ── Playback (delegates to native) ───────────────────────────────────
async function togglePlay(idx) {
    if (state.playingIdx === idx && state.playing) {
        await pause();
    } else {
        await play(idx);
    }
    // Push an immediate poll so the UI doesn't lag the next 100ms tick.
    applyPlayState(await getPlayState());
}

async function seekIdx(idx, fraction) {
    if (state.playingIdx !== idx) {
        await play(idx);
    }
    await seek(fraction);
    applyPlayState(await getPlayState());
}

function updateProgressOverlays() {
    if (srcWfHandle) {
        srcWfHandle.setProgress(state.playingIdx === -1 ? state.progress : 0);
    }
    for (let i = 0; i < slotHandles.length; i++) {
        const h = slotHandles[i];
        if (!h || !h.wfMount) continue;
        h.wfMount.setProgress(state.playingIdx === i ? state.progress : 0);
    }
}

function updateSrcTimes() {
    const dur = state.fileSeconds || 0;
    if (!state.hasSource) {
        srcTimeCurEl.textContent = "0:00";
        srcTimeRemEl.textContent = "0:00";
        return;
    }
    const cur = (state.playingIdx === -1) ? state.progress * dur : 0;
    srcTimeCurEl.textContent = fmtTime(cur);
    srcTimeRemEl.textContent = "−" + fmtTime(Math.max(0, dur - cur));
}

function updateSrcPlayState() {
    const playing = state.playingIdx === -1 && state.playing;
    srcPlayBtn.classList.toggle("active", playing);
    srcPlayBtn.classList.toggle("playing", playing);   // controls icon swap
}

function updatePlayStateUI() {
    updateSrcPlayState();
    for (let i = 0; i < slotHandles.length; i++) {
        const h = slotHandles[i];
        if (!h) continue;
        const active = state.playingIdx === i;
        h.slotEl.classList.toggle("active", active);
        const playingThis = active && state.playing;
        h.playBtn.classList.toggle("playing", playingThis);   // CSS swaps icons
        // Re-tint the waveform colour when active state changes.
        const v = state.variations[i];
        if (v && v.peaks && v.peaks.length > 0) {
            const wfHost = h.slotEl.querySelector(".wf");
            h.wfMount = mountWaveform(wfHost, v.peaks, {
                kind: "line",
                height: 22,
                color: active ? COLOR_ACCENT : COLOR_FG,
                onSeek: (p) => seekIdx(i, p),
            });
            h.wfMount.setProgress(active ? state.progress : 0);
        }
    }
    updateProgressOverlays();
}

srcPlayBtn.addEventListener("click", () => togglePlay(-1));

// Keyboard shortcuts:
//   Space     — toggle play/pause on whatever's active. If nothing is, start
//               the source (or the first ready variation if no source).
//   ←         — seek the active item back to the start. Doesn't change play
//               state — paused stays paused, playing keeps playing from 0.
//   ↑ / ↓     — cycle through ready variations. play(idx) preserves the
//               playback fraction, so this gives A/B-style scrubbing through
//               the 5 candidates without losing your place.
function firstReadyVariation() {
    for (let i = 0; i < state.variations.length; i++) {
        const v = state.variations[i];
        if (v && v.peaks && v.peaks.length > 0) return i;
    }
    return -1;
}

document.addEventListener("keydown", (e) => {
    // Stay out of the way while the user is typing in BPM / Key / Prompt.
    const tag = (e.target && e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") return;

    // ── Space: toggle play/pause ─────────────────────────────────────
    if (e.key === " " || e.code === "Space") {
        e.preventDefault();
        if (state.playingIdx === -2) {
            // Nothing active → pick something sensible.
            let target = state.hasSource ? -1 : firstReadyVariation();
            if (target === -2 || target === -1 && !state.hasSource) return;
            play(target).then(() => getPlayState()).then(applyPlayState);
            return;
        }
        togglePlay(state.playingIdx);
        return;
    }

    // ── Left: rewind active item to start ────────────────────────────
    if (e.key === "ArrowLeft") {
        if (state.playingIdx === -2) return;
        e.preventDefault();
        seek(0).then(() => getPlayState()).then(applyPlayState);
        return;
    }

    // ── Up / Down: cycle ready variations ────────────────────────────
    if (e.key !== "ArrowUp" && e.key !== "ArrowDown") return;
    const ready = state.variations
        .map((v, i) => ({ v, i }))
        .filter(({ v }) => v && v.peaks && v.peaks.length > 0);
    if (ready.length === 0) return;
    e.preventDefault();
    let pos = ready.findIndex(({ i }) => i === state.playingIdx);
    if (pos === -1) {
        pos = e.key === "ArrowUp" ? ready.length - 1 : 0;
    } else {
        const delta = e.key === "ArrowUp" ? -1 : 1;
        pos = (pos + delta + ready.length) % ready.length;
    }
    const targetIdx = ready[pos].i;
    play(targetIdx).then(() => getPlayState()).then(applyPlayState);
});

// ── Drop / pick ──────────────────────────────────────────────────────
// Persistent hidden <input type="file">. WKWebView only fires `change`
// reliably on inputs that are in the DOM at click time, so we declared
// it in index.html and reuse it.
const filePicker = $("file-picker");
filePicker.addEventListener("change", () => {
    const f = filePicker.files && filePicker.files[0];
    if (f) loadFile(f);
    // Reset value so picking the same file twice in a row still fires change.
    filePicker.value = "";
});

sourceEl.addEventListener("click", () => {
    if (state.hasSource) return;
    filePicker.click();
});

// Whole-window drop target. The visual cue (drag-highlight on the source
// area) stays where the user's eye is, but the actual receive zone is the
// entire editor — so a hurried drop anywhere lands the same way.
// Counter pattern handles nested dragenter/dragleave noise: child element
// transitions fire pairs that cancel out, only true window enter/exit
// drives the counter to 1 or back to 0.
let dragDepth = 0;
// Setting dropEffect="copy" on every drag tick is what signals "we'll
// take this" back through WKWebView to the macOS drag service. Without
// it, the underlying NSView returns NSDragOperationNone and the host
// (Ableton, Logic) interprets the drop as falling through the plugin
// window to whatever's behind — usually a session-view track slot.
function acceptDrag(e) {
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "copy";
}
document.addEventListener("dragenter", (e) => {
    acceptDrag(e);
    dragDepth++;
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
    // Suppress drops that came from our own outbound drag — otherwise a
    // click-drag-release on the grip lands the variation right back into
    // the plugin and gets treated as a fresh source upload.
    if (outboundDragInFlight) {
        outboundDragInFlight = false;
        clearTimeout(outboundDragTimer);
        return;
    }
    const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) loadFile(file);
});

// Tracks whether we just initiated an outbound drag via the grip. Set in
// the grip's mousedown handler; cleared in the drop handler above (or
// after a fallback timeout if no drop happens).
let outboundDragInFlight = false;
let outboundDragTimer    = null;

// Pointer is back up — clear the row highlight no matter where the drag
// ended (over the plugin, the DAW, or off-screen). The OS NSDraggingSession
// suppresses the JS mouseup for the *initiator*, so we listen on window
// with `capture:true` to catch it before the suppression kicks in.
window.addEventListener("mouseup", () => {
    document.querySelectorAll(".slot.dragging")
        .forEach(s => s.classList.remove("dragging"));
}, true);

// ── Noise slider ─────────────────────────────────────────────────────
function updateNoiseHint() {
    const v = Number(noiseInput.value);
    let text = "";
    if (v <= 0.30) text = "Results likely very similar to original";
    else if (v >= 0.61) text = "Results likely deviate significantly from original";
    noiseHintEl.textContent = text;
    noiseHintEl.classList.toggle("visible", text !== "");
}

noiseInput.addEventListener("input", () => {
    noiseValueEl.textContent = Number(noiseInput.value).toFixed(2);
    updateNoiseHint();
    saveUiSoon();
});

// ── BPM / Key / Prompt ───────────────────────────────────────────────
bpmInput.addEventListener("input", saveUiSoon);
keyInput.addEventListener("input", saveUiSoon);
userPrompt.addEventListener("input", saveUiSoon);

// ── State persistence ────────────────────────────────────────────────
function snapshotUi() {
    // Noise σ is intentionally not persisted — each new session starts at
    // the HTML default (0.45) so the user re-decides per session.
    return {
        preset: state.preset,
        bpm: bpmInput.value,
        key: keyInput.value,
        userPrompt: userPrompt.value,
        oneShot: state.oneShot,
    };
}

let saveTimer = null;
function saveUiSoon() {
    if (!state.rehydrated) return;
    clearTimeout(saveTimer);
    saveTimer = setTimeout(async () => {
        try { await setUiState(JSON.stringify(snapshotUi())); } catch (e) { }
    }, 200);
}

async function rehydrateUi() {
    try {
        const blob = await getUiState();
        const json = typeof blob === "string" ? blob : "";
        if (json) {
            const s = JSON.parse(json);
            if (s.preset === "free" || s.preset === "app") state.preset = s.preset;
            if (typeof s.bpm === "string") bpmInput.value = s.bpm;
            if (typeof s.key === "string") keyInput.value = s.key;
            if (typeof s.userPrompt === "string") userPrompt.value = s.userPrompt;
            if (typeof s.oneShot === "boolean") state.oneShot = s.oneShot;
        }
    } catch (e) { }
    noiseValueEl.textContent = Number(noiseInput.value).toFixed(2);
    updateNoiseHint();
    renderPreset();
    // Push the restored mode into native and reflect in the toggle label —
    // even if no file is loaded yet (the engine remembers it for next upload).
    applyOneShot(state.oneShot);
    state.rehydrated = true;
}

// ── Generate ─────────────────────────────────────────────────────────
generateBtn.addEventListener("click", async () => {
    if (!state.hasSource) return;
    if (state.inFlight || state.busy) return;     // reentry guard
    state.inFlight = true;
    updateButton();                               // disables immediately

    try {
        await stop();
        clearError();

        // Snap the progress bar back to 0 *with no transition* so the fresh
        // run doesn't visibly retract from the previous run's 100%.
        genProgressEl.style.transition = "none";
        genProgressEl.style.width = "0%";
        // Force a reflow so the transition reset takes effect before we restore it.
        void genProgressEl.offsetWidth;
        genProgressEl.style.transition = "";

        state.expected = 5;
        state.variations = Array.from({ length: 5 }, () => ({
            steer: "", mode: "", peaks: null, duration: null,
        }));
        renderResults();

        const req = {
            preset: state.preset,
            seconds: state.fileSeconds || 0,
            noise: Number(noiseInput.value),
            bpm: state.preset === "app" && bpmInput.value
                ? Number(bpmInput.value) : null,
            key: state.preset === "app" ? keyInput.value : "",
            userPrompt: state.preset === "app" ? userPrompt.value : "",
            seed: Math.floor(Math.random() * 1_000_000),
            steps: 8,
            peaksN: SLOT_PEAKS_N,
        };

        const res = await generate(req);
        if (!res || !res.ok) {
            showError((res && res.error) || "unknown error");
            clearVariations();
            return;
        }
        state.variations = (res.slots || []).map((s) => ({
            steer: s.steer || "",
            mode: s.mode || "",
            peaks: (s.peaks || []).map(Number),
            duration: Number(s.duration) || 0,
        }));
        state.expected = state.variations.length;
        // Stamp the whole generation. Every variation dragged or copied
        // from this batch gets `<file>_var<N>_<stamp>.wav`, so DAW clips
        // that reference it can't collide with previous generations or
        // other plugin instances overwriting a shared temp filename.
        state.generationStamp = makeGenerationStamp();
        renderResults();
    } catch (err) {
        showError("Bridge error: " + err);
        clearVariations();
    } finally {
        // The engine's busy flag has already flipped back to false by the
        // time `generate()` resolves (it clears before completion fires).
        // We mirror that here so the button re-enables immediately —
        // otherwise a fast follow-up click can land in the ~100 ms window
        // before the next status poll updates `state.busy`, get silently
        // gated by the reentry guard, and look like a dead button.
        state.inFlight = false;
        state.busy     = false;
        updateButton();
    }
});

// ── Polling ──────────────────────────────────────────────────────────
async function pollStatus() {
    try { updateStatus(await getStatus()); } catch (e) { }
}
async function pollPlayState() {
    try { applyPlayState(await getPlayState()); } catch (e) { }
}

// ── Boot ─────────────────────────────────────────────────────────────
renderPreset();
renderSource();
renderResults();
rehydrateUi();
pollStatus();
setInterval(pollStatus, 250);
setInterval(pollPlayState, 60);   // ~16fps playhead — enough for 10s clips
