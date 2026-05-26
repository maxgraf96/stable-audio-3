#include "VariationsEngine.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "sa3_orchestrator.h"   // sa3::orch::{Pipeline, build_*_preset, run_variations, InitAudio, load_pipeline}

namespace sa3plugin {

namespace {

// Hardcoded for now — Phase 2 punts on path configuration UI. These paths
// are correct on the dev machine; the shipped plugin will resolve weights
// from ~/Library/Application Support/SA3/models/ (or download on first
// use, matching optimized/mlx).
constexpr const char* kT5GemmaPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/t5gemma_f16.safetensors";
constexpr const char* kDitPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/dit_medium_f16.safetensors";
constexpr const char* kEncoderPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/same_l_encoder_f32.safetensors";
constexpr const char* kDecoderPath =
    "/Users/max/Code/stable-audio-3/optimized/mlx/models/mlx/same_l_decoder_f32.safetensors";

constexpr int kSampleRate = sa3::orch::SAMPLE_RATE;   // 44100

// Decode the user-dropped audio bytes (WAV/AIFF/FLAC/Ogg via JUCE's
// AudioFormatManager) into planar fp32 stereo at 44.1 kHz. Mono input is
// duplicated to L+R. Returns {planar buffer, samples-per-channel}.
struct DecodedAudio {
    std::vector<float> planar;   // layout: planar[c * samples + t]
    int                samples = 0;
};

DecodedAudio decodeSourceAudio(const juce::MemoryBlock& bytes)
{
    if (bytes.getSize() == 0) {
        throw std::runtime_error("source audio is empty");
    }
    juce::AudioFormatManager fmt;
    fmt.registerBasicFormats();
    auto stream = std::make_unique<juce::MemoryInputStream>(bytes, false);
    std::unique_ptr<juce::AudioFormatReader> reader(
        fmt.createReaderFor(std::move(stream)));
    if (!reader) {
        throw std::runtime_error(
            "unsupported audio format (expected WAV/AIFF/FLAC/Ogg)");
    }

    const int    src_channels = static_cast<int>(reader->numChannels);
    const int    src_samples  = static_cast<int>(reader->lengthInSamples);
    const double src_sr       = reader->sampleRate;
    if (src_samples <= 0 || src_channels < 1) {
        throw std::runtime_error("source audio has no samples");
    }

    // Read into a stereo buffer (duplicate mono into both channels).
    juce::AudioBuffer<float> src(2, src_samples);
    src.clear();
    reader->read(&src, /*startSample=*/0, src_samples,
                 /*readerStartSample=*/0,
                 /*useLeftChan=*/true, /*useRightChan=*/true);
    if (src_channels == 1) {
        src.copyFrom(/*destChan=*/1, /*destOffset=*/0,
                     src, /*srcChan=*/0, /*srcOffset=*/0, src_samples);
    }

    // Resample to 44.1k via Lagrange (cheap, good enough — SAME-L is robust
    // to mild aliasing). LagrangeInterpolator::process takes speedRatio =
    // src_sr / dst_sr.
    const bool needs_resample = std::abs(src_sr - kSampleRate) > 0.5;
    const int  dst_samples = needs_resample
        ? static_cast<int>(std::ceil(src_samples * (kSampleRate / src_sr)))
        : src_samples;

    DecodedAudio out;
    out.planar.resize(static_cast<size_t>(2) * static_cast<size_t>(dst_samples), 0.0f);
    out.samples = dst_samples;

    if (! needs_resample) {
        for (int c = 0; c < 2; ++c) {
            std::copy(src.getReadPointer(c),
                      src.getReadPointer(c) + src_samples,
                      out.planar.data() + c * dst_samples);
        }
    } else {
        const double ratio = src_sr / static_cast<double>(kSampleRate);
        for (int c = 0; c < 2; ++c) {
            juce::LagrangeInterpolator interp;
            interp.reset();
            interp.process(ratio,
                           src.getReadPointer(c),
                           out.planar.data() + c * dst_samples,
                           dst_samples);
        }
    }
    return out;
}

// 16-bit little-endian PCM WAV encoder. Mirrors sa3::orch::save_wav_pcm16
// byte-for-byte but writes into a std::vector<std::byte> instead of a file.
// Input: planar fp32 (channels * samples) in [-1, 1] (clipped on write).
std::vector<std::byte> encodeWavPcm16(
    const float* planar, int channels, int samples, int sample_rate)
{
    if (channels != 1 && channels != 2) {
        throw std::runtime_error("encodeWavPcm16: only mono or stereo supported");
    }
    auto write_u32 = [](std::vector<std::byte>& b, uint32_t v) {
        b.push_back(std::byte(v        & 0xff));
        b.push_back(std::byte((v >> 8) & 0xff));
        b.push_back(std::byte((v >> 16)& 0xff));
        b.push_back(std::byte((v >> 24)& 0xff));
    };
    auto write_u16 = [](std::vector<std::byte>& b, uint16_t v) {
        b.push_back(std::byte(v        & 0xff));
        b.push_back(std::byte((v >> 8) & 0xff));
    };
    auto write_bytes = [](std::vector<std::byte>& b, const char* p, size_t n) {
        for (size_t i = 0; i < n; ++i) b.push_back(std::byte(p[i]));
    };

    // Build interleaved s16 PCM, clipped to [-1, 1].
    std::vector<int16_t> pcm(static_cast<size_t>(samples) *
                              static_cast<size_t>(channels));
    for (int t = 0; t < samples; ++t) {
        for (int c = 0; c < channels; ++c) {
            float s = planar[c * samples + t];
            if (!std::isfinite(s)) s = 0.0f;
            s = std::clamp(s, -1.0f, 1.0f);
            pcm[static_cast<size_t>(t) * channels + c] =
                static_cast<int16_t>(std::lrint(s * 32767.0f));
        }
    }

    const uint32_t data_bytes  = static_cast<uint32_t>(pcm.size() * sizeof(int16_t));
    const uint32_t byte_rate   = static_cast<uint32_t>(sample_rate) *
                                 static_cast<uint32_t>(channels) * 2u;
    const uint16_t block_align = static_cast<uint16_t>(channels * 2);

    std::vector<std::byte> buf;
    buf.reserve(44 + data_bytes);
    write_bytes(buf, "RIFF", 4);
    write_u32(buf, 36 + data_bytes);
    write_bytes(buf, "WAVE", 4);
    write_bytes(buf, "fmt ", 4);
    write_u32(buf, 16);
    write_u16(buf, 1);                                 // PCM
    write_u16(buf, static_cast<uint16_t>(channels));
    write_u32(buf, static_cast<uint32_t>(sample_rate));
    write_u32(buf, byte_rate);
    write_u16(buf, block_align);
    write_u16(buf, 16);                                // bits/sample
    write_bytes(buf, "data", 4);
    write_u32(buf, data_bytes);
    buf.insert(buf.end(),
               reinterpret_cast<const std::byte*>(pcm.data()),
               reinterpret_cast<const std::byte*>(pcm.data() + pcm.size()));
    return buf;
}

}  // namespace

// ── VariationsEngine ─────────────────────────────────────────────────

VariationsEngine::VariationsEngine()
    : juce::Thread("SA3 MLX Worker")
{
    startThread();
}

VariationsEngine::~VariationsEngine()
{
    shutdown_.store(true, std::memory_order_release);
    notify();
    // Loading the pipeline can take 1–2 seconds; allow 30s in case the
    // worker is mid-load when teardown hits.
    stopThread(30000);
}

void VariationsEngine::requestLoad()
{
    // Idempotent: only the first call transitions NotLoaded → Loading and
    // wakes the worker. Subsequent calls (or calls after a successful load)
    // are no-ops.
    LoadPhase expected = LoadPhase::NotLoaded;
    if (load_phase_.compare_exchange_strong(expected, LoadPhase::Loading,
                                            std::memory_order_acq_rel)) {
        load_pending_.store(true, std::memory_order_release);
        writeStatus("Loading models...");
        notify();
    }
}

bool VariationsEngine::requestGenerate(GenerateRequest req,
                                       CompletionFn completion)
{
    if (load_phase_.load(std::memory_order_acquire) != LoadPhase::Loaded) {
        return false;   // can't generate until pipeline is up
    }
    {
        std::lock_guard<std::mutex> lk(request_mutex_);
        if (busy_.load(std::memory_order_acquire) || pending_request_.has_value()) {
            return false;
        }
        pending_request_    = std::move(req);
        pending_completion_ = std::move(completion);
        busy_.store(true, std::memory_order_release);
    }
    notify();
    return true;
}

juce::String VariationsEngine::getStatus() const
{
    std::lock_guard<std::mutex> lk(store_mutex_);
    return status_;
}

int VariationsEngine::getVariationCount() const
{
    std::lock_guard<std::mutex> lk(store_mutex_);
    return static_cast<int>(store_.size());
}

std::vector<std::byte> VariationsEngine::getVariationWav(int idx) const
{
    std::lock_guard<std::mutex> lk(store_mutex_);
    if (idx < 0 || idx >= static_cast<int>(store_.size())) return {};
    return store_[static_cast<size_t>(idx)];   // copy (small — ~1MB at 5s)
}

void VariationsEngine::writeStatus(juce::String s)
{
    std::lock_guard<std::mutex> lk(store_mutex_);
    status_ = std::move(s);
}

void VariationsEngine::clearStore()
{
    std::lock_guard<std::mutex> lk(store_mutex_);
    store_.clear();
}

void VariationsEngine::doLoad()
{
    try {
        // fp16 — same default as optimized/cpp/sa3_cli.
        auto p = std::make_unique<sa3::orch::Pipeline>(
            sa3::orch::load_pipeline(kT5GemmaPath, kDitPath,
                                     kEncoderPath, kDecoderPath,
                                     mlx::core::float16));
        pipeline_ = std::move(p);
        load_phase_.store(LoadPhase::Loaded, std::memory_order_release);
        writeStatus("Ready");
    }
    catch (const std::exception& ex) {
        load_phase_.store(LoadPhase::Error, std::memory_order_release);
        writeStatus(juce::String("Load failed: ") + ex.what());
    }
}

juce::var VariationsEngine::doGenerate(GenerateRequest req)
{
    try {
        if (! pipeline_) {
            throw std::runtime_error("pipeline not loaded");
        }
        writeStatus("Decoding source...");
        clearStore();
        DecodedAudio src = decodeSourceAudio(req.audio_bytes);

        // Default seconds = the source duration; cap at [1, 30] s.
        const float src_seconds = static_cast<float>(src.samples) /
                                  static_cast<float>(kSampleRate);
        float seconds = req.seconds > 0.0f ? req.seconds : src_seconds;
        seconds = std::clamp(seconds, 1.0f, 30.0f);

        sa3::orch::InitAudio init_audio{
            src.planar.data(),
            /*channels=*/2,
            src.samples,
        };

        std::vector<sa3::orch::CandidateSpec> specs;
        if (req.preset == "free") {
            specs = sa3::orch::build_free_preset(
                seconds, req.seed, req.noise);
        } else if (req.preset == "app") {
            specs = sa3::orch::build_app_preset(
                seconds, req.seed, req.user_prompt, req.bpm,
                req.key, req.beats_per_bar, req.noise,
                req.cfg_a2a, req.cfg_inpaint, req.apg);
        } else {
            throw std::runtime_error(
                "unknown preset '" + req.preset + "' (expected free|app)");
        }

        // Run one-by-one so the UI sees progress between candidates.
        const int total = static_cast<int>(specs.size());
        juce::Array<juce::var> slots;
        for (int i = 0; i < total; ++i) {
            writeStatus(juce::String::formatted("Generating %d/%d...",
                                                 i + 1, total));
            auto outs = sa3::orch::run_variations(
                *pipeline_, init_audio, {specs[static_cast<size_t>(i)]},
                seconds, req.steps);
            if (outs.size() != 1) {
                throw std::runtime_error(
                    "run_variations returned unexpected count");
            }
            const auto& v = outs[0];
            auto wav = encodeWavPcm16(v.audio.data(), v.channels,
                                      v.samples, kSampleRate);
            {
                std::lock_guard<std::mutex> lk(store_mutex_);
                if (store_.size() <= static_cast<size_t>(i))
                    store_.resize(static_cast<size_t>(i) + 1);
                store_[static_cast<size_t>(i)] = std::move(wav);
            }
            // Each slot carries url + steer + mode so the UI can render
            // an informative hint per pad without a second round-trip.
            // Cache-busting query param keeps the WebView from reusing the
            // previous run's audio when the same index is replayed.
            juce::DynamicObject::Ptr slot = new juce::DynamicObject();
            slot->setProperty("url", juce::var(juce::String::formatted(
                "juce://sa3.local/variations/%d.wav?t=%lld",
                i, juce::Time::currentTimeMillis())));
            slot->setProperty("steer", juce::String(v.spec.steer));
            slot->setProperty("mode",  juce::String(v.spec.mode));
            slots.add(juce::var(slot.get()));
        }

        writeStatus("Ready");

        juce::DynamicObject::Ptr obj = new juce::DynamicObject();
        obj->setProperty("ok",    true);
        obj->setProperty("count", total);
        obj->setProperty("slots", juce::var(slots));
        return juce::var(obj.get());
    }
    catch (const std::exception& ex) {
        writeStatus(juce::String("Error: ") + ex.what());
        juce::DynamicObject::Ptr obj = new juce::DynamicObject();
        obj->setProperty("ok",    false);
        obj->setProperty("error", juce::String(ex.what()));
        return juce::var(obj.get());
    }
}

void VariationsEngine::run()
{
    while (! shutdown_.load(std::memory_order_acquire)) {
        wait(-1);
        if (shutdown_.load(std::memory_order_acquire)) break;

        // Load first, if requested. Single-shot — load_pending_ is consumed
        // exactly once; load_phase_ tracks the outcome.
        if (load_pending_.exchange(false, std::memory_order_acq_rel)) {
            doLoad();
            // After loading, fall through to also handle any generate
            // request that arrived in the meantime.
        }

        // Generate, if requested and a pipeline is up.
        std::optional<GenerateRequest> req;
        CompletionFn                   completion;
        {
            std::lock_guard<std::mutex> lk(request_mutex_);
            req        = std::move(pending_request_);
            completion = std::move(pending_completion_);
            pending_request_.reset();
        }
        if (! req.has_value()) {
            busy_.store(false, std::memory_order_release);
            continue;
        }
        const juce::var result = doGenerate(*req);
        busy_.store(false, std::memory_order_release);
        if (completion) completion(result);
    }
}

}  // namespace sa3plugin
