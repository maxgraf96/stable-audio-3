// Smoke test for the non-Apple InferenceBackend.
//
// Drives the *same* WorkerBackend.cpp the plugin links — process spawn, JSON
// handshake, source WAV encode, streaming candidate events, WAV decode back to
// planar floats — without needing the GUI. The plugin's own path adds only
// VariationsEngine's buffer/peak bookkeeping on top of this.
//
//   sa3_worker_smoke <source.wav> [medium|sm-music|sm-sfx]
//
// Exits non-zero on any failure, so it doubles as a CI check that the Python
// worker and the C++ side still agree on the protocol.
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <juce_audio_formats/juce_audio_formats.h>
#include <juce_core/juce_core.h>

#include "../Source/InferenceBackend.h"

using namespace sa3plugin;

namespace {

ModelKind parseKind(const juce::String& s)
{
    if (s == "sm-music") return ModelKind::SMALL_MUSIC;
    if (s == "sm-sfx")   return ModelKind::SMALL_SFX;
    return ModelKind::MEDIUM;
}

// Read any WAV/AIFF/FLAC into planar stereo fp32, as VariationsEngine would
// hand it to the backend.
std::vector<float> readPlanar(const juce::File& f, int& samplesOut)
{
    juce::AudioFormatManager fmt;
    fmt.registerBasicFormats();
    std::unique_ptr<juce::AudioFormatReader> reader(fmt.createReaderFor(f));
    if (reader == nullptr)
        throw std::runtime_error(("cannot read " + f.getFullPathName()).toStdString());

    const int samples = (int) reader->lengthInSamples;
    juce::AudioBuffer<float> buf(2, samples);
    buf.clear();
    reader->read(&buf, 0, samples, 0, true, true);
    if (reader->numChannels == 1) buf.copyFrom(1, 0, buf, 0, 0, samples);

    std::vector<float> planar((size_t) 2 * (size_t) samples);
    for (int c = 0; c < 2; ++c)
        std::copy(buf.getReadPointer(c), buf.getReadPointer(c) + samples,
                  planar.data() + (size_t) c * (size_t) samples);
    samplesOut = samples;
    return planar;
}

float rms(const std::vector<float>& v)
{
    if (v.empty()) return 0.0f;
    double acc = 0.0;
    for (float x : v) acc += (double) x * x;
    return (float) std::sqrt(acc / (double) v.size());
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::puts("usage: sa3_worker_smoke <source.wav> [medium|sm-music|sm-sfx]");
        return 2;
    }

    const juce::File src(juce::String::fromUTF8(argv[1]));
    const auto kind = parseKind(argc > 2 ? juce::String::fromUTF8(argv[2]) : "medium");

    try {
        auto backend = InferenceBackend::create();
        // Surfaces the same progress the plugin puts in its status line —
        // notably the pipeline rebuild a differently-sized loop triggers.
        backend->setStatusCallback(
            [](const std::string& text) { std::printf("status  : %s\n", text.c_str()); });

        const auto mem = backend->probeMemory();
        std::printf("probe   : %.2f GB | mediumSupported=%d comfortable=%d default=%s\n",
                    mem.totalGB, (int) mem.mediumSupported, (int) mem.mediumComfortable,
                    modelDisplayName(mem.defaultKind));

        const auto t0 = juce::Time::getMillisecondCounterHiRes();
        backend->load(kind, mem);
        std::printf("load    : %s in %.1fs\n", modelDisplayName(kind),
                    (juce::Time::getMillisecondCounterHiRes() - t0) / 1000.0);

        int samples = 0;
        auto planar = readPlanar(src, samples);
        std::printf("source  : %d samples (%.2fs)\n", samples, samples / (double) kSampleRate);

        GenerateRequest req;
        req.preset  = "free";
        req.seconds = (float) samples / (float) kSampleRate;
        req.noise   = 0.45f;
        req.seed    = 4242;
        req.steps   = 8;

        int count = 0;
        const auto tg = juce::Time::getMillisecondCounterHiRes();
        auto lastAt = tg;
        backend->runVariations(
            planar.data(), 2, samples, req, req.seconds,
            [&](const BackendVariation& v) {
                const auto now = juce::Time::getMillisecondCounterHiRes();
                std::printf("  cand %d: %d samples (%.2fs) rms=%.4f mode=%-8s +%.2fs  %s\n",
                            count, v.samples, v.samples / (double) kSampleRate,
                            rms(v.audio), v.mode.c_str(), (now - lastAt) / 1000.0,
                            v.steer.c_str());
                lastAt = now;
                if (v.samples <= 0 || rms(v.audio) <= 0.0f)
                    throw std::runtime_error("candidate decoded to silence");
                ++count;
            });
        std::printf("generate: %d candidates in %.1fs\n", count,
                    (juce::Time::getMillisecondCounterHiRes() - tg) / 1000.0);

        backend->unload();
        if (count != 5) {
            std::printf("FAIL: expected 5 candidates, got %d\n", count);
            return 1;
        }
        std::puts("OK");
        return 0;
    }
    catch (const std::exception& ex) {
        std::printf("FAIL: %s\n", ex.what());
        return 1;
    }
}
