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
    if (path.endsWithIgnoreCase(".wav"))  return "audio/wav";
    if (path.endsWithIgnoreCase(".json")) return "application/json";
    return "application/octet-stream";
}

// JUCE passes us just the URL's path component (e.g. "/index.html"). Strip
// the leading "/" and any query string ("?t=…" cache-buster) for lookup.
juce::String normalisePath(const juce::String& path) {
    juce::String p = path;
    if (p.startsWith("/")) p = p.substring(1);
    if (p.isEmpty())       p = "index.html";
    const int q = p.indexOfChar('?');
    if (q >= 0) p = p.substring(0, q);
    return p;
}

// In dev, set SA3_PLUGIN_RESOURCES=/path/to/plugin/Resources to serve static
// files (HTML/CSS/JS) straight from disk — iterate on UI without rebuilding.
// Variation WAVs are always served from the engine.
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
    // Dev-mode disk override.
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

    // Production: read out of BinaryData. juce_add_binary_data mangles
    // filenames into C identifiers ("index.html" → "index_html") and
    // getNamedResource() keys off the mangled name — walk the parallel
    // originalFilenames[] / namedResourceList[] tables.
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

// Match "variations/N.wav" → N. Returns -1 if not a variation path.
int parseVariationIndex(const juce::String& name) {
    static const juce::String kPrefix = "variations/";
    static const juce::String kSuffix = ".wav";
    if (! name.startsWith(kPrefix) || ! name.endsWith(kSuffix)) return -1;
    juce::String inner = name.substring(kPrefix.length(),
                                        name.length() - kSuffix.length());
    if (inner.isEmpty() || ! inner.containsOnly("0123456789")) return -1;
    return inner.getIntValue();
}

// Decode a base64 string to a juce::MemoryBlock. JUCE's Base64::convertFromBase64
// writes into a MemoryOutputStream which writes through to the block.
juce::MemoryBlock base64Decode(const juce::String& b64) {
    juce::MemoryBlock out;
    juce::MemoryOutputStream stream(out, false);
    if (! juce::Base64::convertFromBase64(stream, b64)) {
        out.reset();
    }
    return out;
}

// Build a GenerateRequest from a juce::var (the JS-side request object).
// Throws a runtime_error on malformed input.
sa3plugin::GenerateRequest parseRequest(const juce::var& v) {
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
    if (v.hasProperty("cfgA2a"))      req.cfg_a2a     = static_cast<float>(static_cast<double>(v["cfgA2a"]));
    if (v.hasProperty("cfgInpaint"))  req.cfg_inpaint = static_cast<float>(static_cast<double>(v["cfgInpaint"]));
    if (v.hasProperty("apg"))         req.apg         = static_cast<float>(static_cast<double>(v["apg"]));
    if (v.hasProperty("seed"))        req.seed        = static_cast<uint64_t>(
                                                            static_cast<int64_t>(v["seed"]));
    if (v.hasProperty("steps"))       req.steps       = static_cast<int>(v["steps"]);

    const juce::String b64 = v["audioBase64"].toString();
    if (b64.isEmpty()) throw std::runtime_error("audioBase64 missing");
    req.audio_bytes = base64Decode(b64);
    if (req.audio_bytes.getSize() == 0) {
        throw std::runtime_error("audioBase64 failed to decode");
    }
    return req;
}

juce::WebBrowserComponent::Options buildOptions(SA3AudioProcessor& processor)
{
    auto resourceProvider =
        [&processor](const juce::String& url)
            -> std::optional<juce::WebBrowserComponent::Resource>
        {
            const juce::String name = normalisePath(url);

            // Variation WAVs live in-memory in the engine — serve those first
            // so a /variations/0.wav request doesn't accidentally hit the
            // static-resource lookup with the same name.
            const int varIdx = parseVariationIndex(name);
            if (varIdx >= 0) {
                auto bytes = processor.getVariationsEngine().getVariationWav(varIdx);
                if (! bytes.empty()) {
                    return juce::WebBrowserComponent::Resource{
                        std::move(bytes), "audio/wav"};
                }
                return std::nullopt;   // 404 — let JS retry / show error
            }
            return serveStaticResource(name);
        };

    return juce::WebBrowserComponent::Options{}
        .withBackend(juce::WebBrowserComponent::Options::Backend::defaultBackend)
        .withNativeIntegrationEnabled(true)
        .withResourceProvider(resourceProvider, juce::String(kOrigin))
        .withNativeFunction(
            "getStatus",
            [&processor](const juce::Array<juce::var>& /*args*/,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                // Single worker now — status is just the engine's view of the
                // world plus a load-phase tag for the UI to pick the colour.
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
                // Hand back whatever the host most recently restored via
                // setStateInformation (or the last setUiState write). Empty
                // string on fresh sessions; JS treats that as "use defaults".
                complete(juce::var(processor.getPersistedUiStateJson()));
            })
        .withNativeFunction(
            "setUiState",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (! args.isEmpty()) {
                    processor.setPersistedUiStateJson(args[0].toString());
                    // Tell the host the project is dirty so the new state
                    // actually makes it into the DAW save.
                    processor.updateHostDisplay();
                }
                complete(juce::var());
            })
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
                    auto req = parseRequest(args[0]);
                    const bool accepted = processor.getVariationsEngine().requestGenerate(
                        std::move(req),
                        // Engine fires this from the worker thread. JUCE
                        // serialises the var back to JS internally.
                        [complete](juce::var result) { complete(result); });
                    if (! accepted) {
                        juce::DynamicObject::Ptr busy = new juce::DynamicObject();
                        busy->setProperty("ok",    false);
                        busy->setProperty("error", "already running — wait for current job");
                        complete(juce::var(busy.get()));
                    }
                }
                catch (const std::exception& ex) {
                    juce::DynamicObject::Ptr err = new juce::DynamicObject();
                    err->setProperty("ok",    false);
                    err->setProperty("error", juce::String(ex.what()));
                    complete(juce::var(err.get()));
                }
            });
}

}  // namespace

SA3AudioProcessorEditor::SA3AudioProcessorEditor(SA3AudioProcessor& p)
    : AudioProcessorEditor(&p),
      webView(buildOptions(p))
{
    // Larger window for the full variation UI (drop zone + 5 audition pads).
    setSize(720, 540);
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
