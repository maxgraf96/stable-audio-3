/**
 * In-browser Demucs (htdemucs) source separation.
 *
 * Ported from the reference demo at https://github.com/sevagh/demucs.onnx
 * (src_wasm/demo.js). Strips out: WebGPU readback boilerplate, mixer UI,
 * encryption, downloads, iOS debug overlay. Keeps the WebGPU fast path with a
 * WASM fallback.
 *
 * Usage:
 *   import { initDemucs, separate, isReady } from "./demucs.js";
 *   await initDemucs({ modelUrl, wasmModuleUrl, onProgress });
 *   const { drums, bass, other, vocals } = await separate(audioBuffer, onProgress);
 *
 * Returns AudioBuffer objects (44.1 kHz stereo, same length as input).
 */

const SAMPLE_RATE = 44100;
const SEGMENT_LEN_SECS = 2.0;
const OVERLAP = 0.25;
const TRANSITION_POWER = 1.0;
const SEGMENT_SAMPLES = Math.floor(SEGMENT_LEN_SECS * SAMPLE_RATE);
const STRIDE_SAMPLES = Math.floor((1.0 - OVERLAP) * SEGMENT_SAMPLES);
const NB_SOURCES = 4;
const SOURCE_NAMES = ["drums", "bass", "other", "vocals"];

let wasmModule = null;
let ortSession = null;
let useWebGPU = false;
let gpuDevice = null;

const MODEL_CACHE_NAME = "remix-demucs-v1";

export function isReady() {
  return wasmModule !== null && ortSession !== null;
}

/**
 * Fetch the model bytes, caching them in Cache Storage so subsequent page
 * loads skip the network entirely. Falls back to a plain fetch if Cache
 * Storage isn't available (private mode etc.).
 */
async function fetchCachedModel(url, onProgress) {
  if ("caches" in self) {
    try {
      const cache = await caches.open(MODEL_CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        onProgress("Loading model from cache…");
        return await hit.arrayBuffer();
      }
      onProgress("Downloading model (cached for next time)…");
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`model fetch ${resp.status}`);
      // Tee the response so we can both cache and read.
      const clone = resp.clone();
      try { await cache.put(url, clone); } catch (e) {
        console.warn("[demucs] cache.put failed:", e);
      }
      return await resp.arrayBuffer();
    } catch (e) {
      console.warn("[demucs] cache path failed, falling back:", e);
    }
  }
  onProgress("Downloading model…");
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`model fetch ${resp.status}`);
  return await resp.arrayBuffer();
}

export async function initDemucs({ modelUrl, wasmModuleUrl, onProgress = () => {} }) {
  if (isReady()) return;

  onProgress("Loading DSP module…");
  const dsp = await import(/* @vite-ignore */ wasmModuleUrl);
  wasmModule = await dsp.default();

  const modelBytes = new Uint8Array(await fetchCachedModel(modelUrl, onProgress));

  onProgress("Initializing inference session…");
  let success = false;
  if (navigator.gpu) {
    try {
      ortSession = await ort.InferenceSession.create(modelBytes, {
        executionProviders: ["webgpu"],
      });
      useWebGPU = true;
      gpuDevice = ort.env.webgpu.device;
      success = true;
      onProgress("Model ready (WebGPU)");
    } catch (e) {
      console.warn("[demucs] WebGPU failed, falling back to WASM:", e);
    }
  }
  if (!success) {
    try {
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = true;
      const hc = navigator.hardwareConcurrency || 2;
      ort.env.wasm.numThreads = Math.min(4, Math.max(1, hc));
    } catch {}
    ortSession = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
    });
    useWebGPU = false;
    onProgress("Model ready (WASM)");
  }
}

function byteSizeFromDims(dims) {
  return dims.reduce((a, b) => a * b, 1) * 4;
}

async function readGpuBuffer(srcBuffer, dims) {
  const size = byteSizeFromDims(dims);
  const readback = gpuDevice.createBuffer({
    size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = gpuDevice.createCommandEncoder();
  encoder.copyBufferToBuffer(srcBuffer, 0, readback, 0, size);
  gpuDevice.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const arr = new Float32Array(readback.getMappedRange().slice(0));
  readback.unmap();
  readback.destroy();
  return arr;
}

/**
 * Run Demucs on an AudioBuffer (must already be 44.1 kHz stereo).
 * Returns: { drums, bass, other, vocals } as AudioBuffers.
 */
export async function separate(audioBuffer, onProgress = () => {}) {
  if (!isReady()) throw new Error("Demucs not initialized");
  if (audioBuffer.sampleRate !== SAMPLE_RATE) {
    throw new Error(`Expected ${SAMPLE_RATE} Hz audio, got ${audioBuffer.sampleRate}`);
  }
  if (audioBuffer.numberOfChannels < 2) {
    throw new Error("Expected stereo audio");
  }

  const numSamples = audioBuffer.length;
  const left = audioBuffer.getChannelData(0);
  const right = audioBuffer.getChannelData(1);

  // Per-track normalization (using mono mix stats).
  const mono = new Float32Array(numSamples);
  let sum = 0;
  for (let i = 0; i < numSamples; i++) {
    const m = 0.5 * (left[i] + right[i]);
    mono[i] = m;
    sum += m;
  }
  const mean = sum / numSamples;
  let varSum = 0;
  for (let i = 0; i < numSamples; i++) {
    const d = mono[i] - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / Math.max(1, numSamples - 1)) || 1e-8;

  // Random time-shift for time-invariance.
  const MAX_SHIFT_SECS = 0.5;
  const maxShift = Math.floor(MAX_SHIFT_SECS * SAMPLE_RATE);
  const shiftOffset = Math.floor(Math.random() * maxShift);

  const paddedLength = numSamples + 2 * maxShift;
  const paddedMix = new Float32Array(paddedLength * 2);
  for (let ch = 0; ch < 2; ch++) {
    const src = ch === 0 ? left : right;
    for (let i = 0; i < numSamples; i++) {
      paddedMix[ch * paddedLength + maxShift + i] = (src[i] - mean) / std;
    }
  }

  const shiftedLength = numSamples + maxShift - shiftOffset;
  const audioData = new Float32Array(shiftedLength * 2);
  for (let ch = 0; ch < 2; ch++) {
    for (let i = 0; i < shiftedLength; i++) {
      audioData[ch * shiftedLength + i] = paddedMix[ch * paddedLength + shiftOffset + i];
    }
  }

  // Segment-processing constants (must match C++ reference).
  const HOP = 1024;
  const pad = Math.floor(HOP / 2.0) * 3;
  const le = Math.ceil(SEGMENT_SAMPLES / HOP);
  const padEnd = pad + le * HOP - SEGMENT_SAMPLES;
  const paddedSegmentSamples = SEGMENT_SAMPLES + pad + padEnd;
  const nbFrames = Math.floor(paddedSegmentSamples / HOP) + 1;
  const nbFreqFrames = nbFrames - 4;

  // Pre-allocate WebGPU tensors once if we're on that path.
  const timeBytes = 1 * 2 * SEGMENT_SAMPLES * 4;
  const freqBytes = 1 * 4 * 2048 * nbFreqFrames * 4;
  const freqOutDims = [1, 4, 4, 2048, nbFreqFrames];
  const timeOutDims = [1, NB_SOURCES, 2, SEGMENT_SAMPLES];
  let timeBuf, freqBuf, timeTensorGPU, freqTensorGPU;
  let freqOutBuf, timeOutBuf, freqOutTensorGPU, timeOutTensorGPU;
  if (useWebGPU) {
    const usage =
      GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE;
    timeBuf = gpuDevice.createBuffer({ usage, size: timeBytes });
    freqBuf = gpuDevice.createBuffer({ usage, size: freqBytes });
    timeTensorGPU = ort.Tensor.fromGpuBuffer(timeBuf, {
      dataType: "float32",
      dims: [1, 2, SEGMENT_SAMPLES],
    });
    freqTensorGPU = ort.Tensor.fromGpuBuffer(freqBuf, {
      dataType: "float32",
      dims: [1, 4, 2048, nbFreqFrames],
    });
    freqOutBuf = gpuDevice.createBuffer({
      size: byteSizeFromDims(freqOutDims),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    timeOutBuf = gpuDevice.createBuffer({
      size: byteSizeFromDims(timeOutDims),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    freqOutTensorGPU = ort.Tensor.fromGpuBuffer(freqOutBuf, {
      dataType: "float32",
      dims: freqOutDims,
    });
    timeOutTensorGPU = ort.Tensor.fromGpuBuffer(timeOutBuf, {
      dataType: "float32",
      dims: timeOutDims,
    });
  }

  const totalChunks = Math.ceil(shiftedLength / STRIDE_SAMPLES);
  const out = new Float32Array(NB_SOURCES * 2 * shiftedLength);
  const sumWeight = new Float32Array(shiftedLength);

  // Triangular overlap-add window.
  const weight = new Float32Array(SEGMENT_SAMPLES);
  const half = Math.floor(SEGMENT_SAMPLES / 2);
  for (let i = 0; i < half; i++) weight[i] = i + 1;
  for (let i = half; i < SEGMENT_SAMPLES; i++) weight[i] = SEGMENT_SAMPLES - i;
  const maxw = weight[half - 1];
  for (let i = 0; i < SEGMENT_SAMPLES; i++) {
    weight[i] = Math.pow(weight[i] / maxw, TRANSITION_POWER);
  }

  for (let chunkIdx = 0; chunkIdx < totalChunks; chunkIdx++) {
    const segmentOffset = chunkIdx * STRIDE_SAMPLES;
    const chunkLength = Math.min(SEGMENT_SAMPLES, shiftedLength - segmentOffset);
    onProgress(`Separating ${chunkIdx + 1} / ${totalChunks}…`, (chunkIdx + 1) / totalChunks);

    const segment = new Float32Array(SEGMENT_SAMPLES * 2);
    for (let ch = 0; ch < 2; ch++) {
      for (let i = 0; i < chunkLength; i++) {
        segment[ch * SEGMENT_SAMPLES + i] = audioData[ch * shiftedLength + segmentOffset + i];
      }
    }

    const symmetricPadding = paddedSegmentSamples - chunkLength;
    const symmetricPaddingStart = Math.floor(symmetricPadding / 2);
    const paddedSegment = new Float32Array(paddedSegmentSamples * 2);
    for (let ch = 0; ch < 2; ch++) {
      for (let i = 0; i < chunkLength; i++) {
        paddedSegment[ch * paddedSegmentSamples + pad + symmetricPaddingStart + i] =
          segment[ch * SEGMENT_SAMPLES + i];
      }
    }
    // Reflect padding on both sides.
    for (let ch = 0; ch < 2; ch++) {
      const off = ch * paddedSegmentSamples;
      for (let i = 0; i < pad; i++) {
        paddedSegment[off + pad - 1 - i] = paddedSegment[off + pad + i];
      }
      const lastElem = SEGMENT_SAMPLES + pad - 1;
      for (let i = 0; i < padEnd; i++) {
        paddedSegment[off + lastElem + i + 1] = paddedSegment[off + lastElem - i];
      }
    }

    const spec = wasmModule.wasm_stft(paddedSegment, paddedSegmentSamples);
    const freqInput = wasmModule.wasm_prepare_freq_input(spec, nbFrames);
    const timeInput = wasmModule.wasm_prepare_time_input(paddedSegment, SEGMENT_SAMPLES, pad);

    let freqOutput, timeOutput;
    if (useWebGPU) {
      gpuDevice.queue.writeBuffer(freqBuf, 0, freqInput.buffer, freqInput.byteOffset, freqInput.byteLength);
      gpuDevice.queue.writeBuffer(timeBuf, 0, timeInput.buffer, timeInput.byteOffset, timeInput.byteLength);
      const feeds = {
        [ortSession.inputNames[0]]: timeTensorGPU,
        [ortSession.inputNames[1]]: freqTensorGPU,
      };
      const fetches = {
        [ortSession.outputNames[0]]: freqOutTensorGPU,
        [ortSession.outputNames[1]]: timeOutTensorGPU,
      };
      await ortSession.run(feeds, fetches);
      freqOutput = await readGpuBuffer(freqOutBuf, freqOutDims);
      timeOutput = await readGpuBuffer(timeOutBuf, timeOutDims);
    } else {
      const freqTensor = new ort.Tensor("float32", freqInput, [1, 4, 2048, nbFreqFrames]);
      const timeTensor = new ort.Tensor("float32", timeInput, [1, 2, SEGMENT_SAMPLES]);
      const results = await ortSession.run({
        [ortSession.inputNames[0]]: timeTensor,
        [ortSession.inputNames[1]]: freqTensor,
      });
      freqOutput = results[ortSession.outputNames[0]].data;
      timeOutput = results[ortSession.outputNames[1]].data;
    }

    const freqSpec = wasmModule.wasm_unpack_freq_output(freqOutput, NB_SOURCES, nbFrames);

    const freqWaveforms = new Float32Array(NB_SOURCES * 2 * SEGMENT_SAMPLES);
    for (let src = 0; src < NB_SOURCES; src++) {
      const srcSpecOffset = src * 2 * 2049 * nbFrames * 2;
      const srcSpecSize = 2 * 2049 * nbFrames * 2;
      const srcSpec = freqSpec.subarray(srcSpecOffset, srcSpecOffset + srcSpecSize);
      const srcWave = wasmModule.wasm_istft(srcSpec, paddedSegmentSamples);
      for (let ch = 0; ch < 2; ch++) {
        for (let i = 0; i < SEGMENT_SAMPLES; i++) {
          freqWaveforms[src * 2 * SEGMENT_SAMPLES + ch * SEGMENT_SAMPLES + i] =
            srcWave[ch * paddedSegmentSamples + pad + i];
        }
      }
    }

    const merged = wasmModule.wasm_merge_branches(freqWaveforms, timeOutput, NB_SOURCES, SEGMENT_SAMPLES);

    // Trim merged segment back to chunkLength using the center-of-segment offset.
    const chunkOut = new Float32Array(NB_SOURCES * 2 * chunkLength);
    for (let src = 0; src < NB_SOURCES; src++) {
      for (let ch = 0; ch < 2; ch++) {
        for (let i = 0; i < chunkLength; i++) {
          const k = Math.min(i + symmetricPaddingStart, SEGMENT_SAMPLES - 1);
          chunkOut[src * 2 * chunkLength + ch * chunkLength + i] =
            merged[src * 2 * SEGMENT_SAMPLES + ch * SEGMENT_SAMPLES + k];
        }
      }
    }

    // Overlap-add into the accumulator.
    for (let src = 0; src < NB_SOURCES; src++) {
      for (let ch = 0; ch < 2; ch++) {
        const dstBase = src * 2 * shiftedLength + ch * shiftedLength + segmentOffset;
        const srcBase = src * 2 * chunkLength + ch * chunkLength;
        for (let i = 0; i < chunkLength; i++) {
          out[dstBase + i] += chunkOut[srcBase + i] * weight[i];
        }
      }
    }
    for (let i = 0; i < chunkLength; i++) {
      sumWeight[segmentOffset + i] += weight[i];
    }

    // Let the UI repaint between chunks.
    await new Promise((r) => setTimeout(r, 0));
  }

  // Normalize and denormalize back to original scale.
  for (let src = 0; src < NB_SOURCES; src++) {
    for (let ch = 0; ch < 2; ch++) {
      const base = src * 2 * shiftedLength + ch * shiftedLength;
      for (let i = 0; i < shiftedLength; i++) {
        out[base + i] = (out[base + i] / sumWeight[i]) * std + mean;
      }
    }
  }

  // Trim back the random-shift offset.
  const trimStart = maxShift - shiftOffset;
  const trimmed = new Float32Array(NB_SOURCES * 2 * numSamples);
  for (let src = 0; src < NB_SOURCES; src++) {
    for (let ch = 0; ch < 2; ch++) {
      const srcBase = src * 2 * shiftedLength + ch * shiftedLength + trimStart;
      const dstBase = src * 2 * numSamples + ch * numSamples;
      for (let i = 0; i < numSamples; i++) {
        trimmed[dstBase + i] = out[srcBase + i];
      }
    }
  }

  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  const result = {};
  for (let src = 0; src < NB_SOURCES; src++) {
    const buffer = ctx.createBuffer(2, numSamples, SAMPLE_RATE);
    const l = buffer.getChannelData(0);
    const r = buffer.getChannelData(1);
    const base = src * 2 * numSamples;
    for (let i = 0; i < numSamples; i++) {
      l[i] = trimmed[base + i];
      r[i] = trimmed[base + numSamples + i];
    }
    result[SOURCE_NAMES[src]] = buffer;
  }
  // Best-effort close of the temporary decode context.
  try { await ctx.close(); } catch {}
  return result;
}
