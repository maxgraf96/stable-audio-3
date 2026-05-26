#include "PluginEditor.h"

#include <cstdlib>
#include <cstring>
#include <optional>

#include "BinaryData.h"
#include "VariationsEngine.h"

namespace {

// JUCE 8 registers a custom URL scheme handler for `juce://` on WKWebView —
// standard schemes (http/https) can't be intercepted in WKWebView, so the
// resource provider is only invoked for `juce://` URLs. The provider receives
// just the path portion of the URL (e.g. "/index.html"), not the full URL.
constexpr const char* kOrigin   = "juce://sa3.local";
constexpr const char* kIndexURL = "juce://sa3.local/index.html";

juce::String mimeFor(const juce::String& path) {
    if (path.endsWithIgnoreCase(".html")) return "text/html";
    if (path.endsWithIgnoreCase(".js"))   return "application/javascript";
    if (path.endsWithIgnoreCase(".css"))  return "text/css";
    if (path.endsWithIgnoreCase(".svg"))  return "image/svg+xml";
    if (path.endsWithIgnoreCase(".png"))  return "image/png";
    if (path.endsWithIgnoreCase(".json")) return "application/json";
    return "application/octet-stream";
}

// JUCE passes us just the URL's path component (e.g. "/index.html"). Strip
// the leading "/" and any query string for lookup.
juce::String normalisePath(const juce::String& path) {
    juce::String p = path;
    if (p.startsWith("/")) p = p.substring(1);
    if (p.isEmpty())       p = "index.html";
    const int q = p.indexOfChar('?');
    if (q >= 0) p = p.substring(0, q);
    return p;
}

juce::File devResourcesDir() {
    if (const char* env = std::getenv("SA3_PLUGIN_RESOURCES")) {
        juce::File dir(juce::String::fromUTF8(env));
        if (dir.isDirectory()) return dir;
    }
    return {};
}

std::optional<juce::WebBrowserComponent::Resource>
serveStaticResource(const juce::String& name)
{
    if (auto dir = devResourcesDir(); dir != juce::File{}) {
        juce::File f = dir.getChildFile(name);
        if (f.existsAsFile()) {
            juce::MemoryBlock mb;
            if (f.loadFileAsData(mb)) {
                std::vector<std::byte> bytes(mb.getSize());
                std::memcpy(bytes.data(), mb.getData(), mb.getSize());
                return juce::WebBrowserComponent::Resource{
                    std::move(bytes), mimeFor(name)};
            }
        }
    }
    for (int i = 0; i < BinaryData::namedResourceListSize; ++i) {
        if (name == juce::String(BinaryData::originalFilenames[i])) {
            int size = 0;
            if (const char* data = BinaryData::getNamedResource(
                    BinaryData::namedResourceList[i], size); data && size > 0) {
                std::vector<std::byte> bytes(static_cast<size_t>(size));
                std::memcpy(bytes.data(), data, static_cast<size_t>(size));
                return juce::WebBrowserComponent::Resource{
                    std::move(bytes), mimeFor(name)};
            }
        }
    }
    return std::nullopt;
}

juce::MemoryBlock base64Decode(const juce::String& b64) {
    juce::MemoryBlock out;
    juce::MemoryOutputStream stream(out, false);
    if (! juce::Base64::convertFromBase64(stream, b64)) out.reset();
    return out;
}

sa3plugin::GenerateRequest parseGenerateRequest(const juce::var& v) {
    if (! v.isObject()) throw std::runtime_error("request must be an object");
    sa3plugin::GenerateRequest req;
    req.preset      = v["preset"].toString().toStdString();
    req.seconds     = static_cast<float>(static_cast<double>(v["seconds"]));
    req.noise       = static_cast<float>(static_cast<double>(v["noise"]));
    req.user_prompt = v["userPrompt"].toString().toStdString();
    req.key         = v["key"].toString().toStdString();
    if (v.hasProperty("bpm")) {
        const auto& bpm = v["bpm"];
        if (! bpm.isVoid() && ! bpm.isUndefined()) {
            const double b = static_cast<double>(bpm);
            if (b > 0.0) req.bpm = static_cast<float>(b);
        }
    }
    if (v.hasProperty("beatsPerBar")) {
        const int bpb = static_cast<int>(v["beatsPerBar"]);
        if (bpb > 0) req.beats_per_bar = bpb;
    }
    if (v.hasProperty("cfgA2a"))     req.cfg_a2a     = static_cast<float>(static_cast<double>(v["cfgA2a"]));
    if (v.hasProperty("cfgInpaint")) req.cfg_inpaint = static_cast<float>(static_cast<double>(v["cfgInpaint"]));
    if (v.hasProperty("apg"))        req.apg         = static_cast<float>(static_cast<double>(v["apg"]));
    if (v.hasProperty("seed"))       req.seed        = static_cast<uint64_t>(static_cast<int64_t>(v["seed"]));
    if (v.hasProperty("steps"))      req.steps       = static_cast<int>(v["steps"]);
    return req;
}

juce::WebBrowserComponent::Options buildOptions(SA3AudioProcessor& processor)
{
    auto resourceProvider = [](const juce::String& url)
            -> std::optional<juce::WebBrowserComponent::Resource>
        {
            return serveStaticResource(normalisePath(url));
        };

    return juce::WebBrowserComponent::Options{}
        .withBackend(juce::WebBrowserComponent::Options::Backend::defaultBackend)
        .withNativeIntegrationEnabled(true)
        .withResourceProvider(resourceProvider, juce::String(kOrigin))
        // ── Status / persistence ──────────────────────────────────────
        .withNativeFunction(
            "getStatus",
            [&processor](const juce::Array<juce::var>& /*args*/,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                const auto& eng = processor.getVariationsEngine();
                const auto phase = eng.getLoadPhase();
                const char* phaseStr =
                    phase == sa3plugin::VariationsEngine::LoadPhase::NotLoaded ? "not_loaded" :
                    phase == sa3plugin::VariationsEngine::LoadPhase::Loading   ? "loading"    :
                    phase == sa3plugin::VariationsEngine::LoadPhase::Loaded    ? "loaded"     :
                                                                                 "error";
                juce::DynamicObject::Ptr obj = new juce::DynamicObject();
                obj->setProperty("status", eng.getStatus());
                obj->setProperty("phase",  juce::String(phaseStr));
                obj->setProperty("busy",   eng.isBusy());
                complete(juce::var(obj.get()));
            })
        .withNativeFunction(
            "getUiState",
            [&processor](const juce::Array<juce::var>& /*args*/,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                complete(juce::var(processor.getPersistedUiStateJson()));
            })
        .withNativeFunction(
            "setUiState",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (! args.isEmpty()) {
                    processor.setPersistedUiStateJson(args[0].toString());
                    processor.updateHostDisplay();
                }
                complete(juce::var());
            })
        // ── uploadSource(base64, peaks_n) → { ok, duration, peaks } ───
        .withNativeFunction(
            "uploadSource",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (args.isEmpty()) {
                    juce::DynamicObject::Ptr err = new juce::DynamicObject();
                    err->setProperty("ok",    false);
                    err->setProperty("error", "uploadSource(): no arguments");
                    complete(juce::var(err.get()));
                    return;
                }
                const juce::var& v = args[0];
                const juce::String b64 = v["audioBase64"].toString();
                const int peaks_n = v.hasProperty("peaksN")
                    ? static_cast<int>(v["peaksN"]) : 220;
                if (b64.isEmpty()) {
                    juce::DynamicObject::Ptr err = new juce::DynamicObject();
                    err->setProperty("ok",    false);
                    err->setProperty("error", "audioBase64 missing");
                    complete(juce::var(err.get()));
                    return;
                }
                juce::MemoryBlock bytes = base64Decode(b64);
                if (bytes.getSize() == 0) {
                    juce::DynamicObject::Ptr err = new juce::DynamicObject();
                    err->setProperty("ok",    false);
                    err->setProperty("error", "audioBase64 failed to decode");
                    complete(juce::var(err.get()));
                    return;
                }
                const bool accepted = processor.getVariationsEngine().requestUploadSource(
                    std::move(bytes), peaks_n,
                    [complete](juce::var result) { complete(result); });
                if (! accepted) {
                    juce::DynamicObject::Ptr busy = new juce::DynamicObject();
                    busy->setProperty("ok",    false);
                    busy->setProperty("error", "engine busy");
                    complete(juce::var(busy.get()));
                }
            })
        // ── generate(req) → { ok, slots: [...] } ──────────────────────
        .withNativeFunction(
            "generate",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (args.isEmpty()) {
                    juce::DynamicObject::Ptr err = new juce::DynamicObject();
                    err->setProperty("ok",    false);
                    err->setProperty("error", "generate(): no arguments");
                    complete(juce::var(err.get()));
                    return;
                }
                try {
                    auto req = parseGenerateRequest(args[0]);
                    const int peaks_n = args[0].hasProperty("peaksN")
                        ? static_cast<int>(args[0]["peaksN"]) : 140;
                    const bool accepted = processor.getVariationsEngine().requestGenerate(
                        std::move(req), peaks_n,
                        [complete](juce::var result) { complete(result); });
                    if (! accepted) {
                        juce::DynamicObject::Ptr busy = new juce::DynamicObject();
                        busy->setProperty("ok",    false);
                        busy->setProperty("error",
                            "engine busy or no source — upload audio first");
                        complete(juce::var(busy.get()));
                    }
                }
                catch (const std::exception& ex) {
                    juce::DynamicObject::Ptr err = new juce::DynamicObject();
                    err->setProperty("ok",    false);
                    err->setProperty("error", juce::String(ex.what()));
                    complete(juce::var(err.get()));
                }
            })
        // ── Playback transport ────────────────────────────────────────
        .withNativeFunction(
            "play",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                const int idx = args.isEmpty() ? -2 : static_cast<int>(args[0]);
                processor.getVariationsEngine().play(idx);
                complete(juce::var());
            })
        .withNativeFunction(
            "pause",
            [&processor](const juce::Array<juce::var>& /*args*/,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                processor.getVariationsEngine().pausePlayback();
                complete(juce::var());
            })
        .withNativeFunction(
            "stop",
            [&processor](const juce::Array<juce::var>& /*args*/,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                processor.getVariationsEngine().stopPlayback();
                complete(juce::var());
            })
        .withNativeFunction(
            "seek",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                const double frac = args.isEmpty()
                    ? 0.0 : static_cast<double>(args[0]);
                processor.getVariationsEngine().seek(frac);
                complete(juce::var());
            })
        .withNativeFunction(
            "getPlayState",
            [&processor](const juce::Array<juce::var>& /*args*/,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                const auto p = processor.getVariationsEngine().getPlayState();
                juce::DynamicObject::Ptr obj = new juce::DynamicObject();
                obj->setProperty("idx",      p.idx);
                obj->setProperty("playing",  p.playing);
                obj->setProperty("progress", p.progress);
                obj->setProperty("duration", p.duration);
                obj->setProperty("oneShot",  p.oneShot);
                complete(juce::var(obj.get()));
            })
        .withNativeFunction(
            "setOneShotMode",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                const bool v = ! args.isEmpty() && static_cast<bool>(args[0]);
                processor.getVariationsEngine().setOneShotMode(v);
                complete(juce::var());
            });
}

}  // namespace

SA3AudioProcessorEditor::SA3AudioProcessorEditor(SA3AudioProcessor& p)
    : AudioProcessorEditor(&p),
      webView(buildOptions(p))
{
    // Tight default size — all 5 audition pads + controls fit without a
    // scrollbar (.hl uses overflow:hidden, so anything that doesn't fit
    // gets clipped rather than introducing scroll).
    setSize(500, 720);
    addAndMakeVisible(webView);
    webView.goToURL(kIndexURL);
}

SA3AudioProcessorEditor::~SA3AudioProcessorEditor() = default;

void SA3AudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void SA3AudioProcessorEditor::resized()
{
    webView.setBounds(getLocalBounds());
}
