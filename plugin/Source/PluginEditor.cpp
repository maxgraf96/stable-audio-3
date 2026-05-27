#include "PluginEditor.h"

#include <cstdlib>
#include <cstring>
#include <optional>

#include "BinaryData.h"
#include "VariationsEngine.h"

#if JUCE_MAC
#include "MacClipboard.h"
#endif

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

// Match `/variations/<N>.wav` (after normalisePath) → N. Returns -1 on
// anything that isn't a variation request.
int parseVariationIndex(const juce::String& name) {
    static const juce::String kPrefix = "variations/";
    static const juce::String kSuffix = ".wav";
    if (! name.startsWith(kPrefix) || ! name.endsWith(kSuffix)) return -1;
    juce::String inner = name.substring(kPrefix.length(),
                                        name.length() - kSuffix.length());
    if (inner.isEmpty() || ! inner.containsOnly("0123456789")) return -1;
    return inner.getIntValue();
}

using DragRequestCb = std::function<void(int idx,
                                         const juce::String& baseName,
                                         const juce::String& stamp)>;

juce::WebBrowserComponent::Options buildOptions(SA3AudioProcessor& processor,
                                                DragRequestCb      dragRequestCb)
{
    auto resourceProvider = [&processor](const juce::String& url)
            -> std::optional<juce::WebBrowserComponent::Resource>
        {
            const juce::String name = normalisePath(url);
            // Variation WAVs encoded on demand from the engine's
            // in-memory AudioBuffer. Used by the HTML5 drag-out path —
            // WebKit fetches this URL when materializing the file
            // promise on drop.
            const int varIdx = parseVariationIndex(name);
            if (varIdx >= 0) {
                auto bytes = processor.getVariationsEngine().getVariationWavBytes(varIdx);
                if (! bytes.empty()) {
                    return juce::WebBrowserComponent::Resource{
                        std::move(bytes), "audio/wav"};
                }
                return std::nullopt;
            }
            return serveStaticResource(name);
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
            })
        // dragVariation event — fired from JS `mousedown` on a row's
        // grip. The listener runs synchronously on the message thread
        // (same run-loop tick as the WKWebView mousedown), so AppKit
        // still sees an active mouse-down event when we call
        // performExternalDragDropOfFiles and starts a real NSDraggingSession
        // rooted on the WebView's own NSView. This is the canonical
        // WKWebView-compatible drag-out path; sibling NSView overlays
        // don't work because WKWebView is out-of-process IOSurface-backed
        // and owns hit-testing for its whole rect. (JUCE forum #64813.)
        .withEventListener(
            "dragVariation",
            [dragRequestCb](juce::var v) {
                const int          idx      = static_cast<int>(v.getProperty("idx", -1));
                const juce::String baseName = v.getProperty("baseName", "SA3").toString();
                const juce::String stamp    = v.getProperty("stamp",    "").toString();
                if (idx >= 0 && dragRequestCb) dragRequestCb(idx, baseName, stamp);
            })
        // copyVariationToClipboard(idx, baseName) — write the variation
        // to a temp WAV, then put a POSIX file reference on the system
        // clipboard via NSPasteboard. Works in Finder + Logic where ⌘V
        // honours the OS clipboard; Live's ⌘V uses an internal cache
        // that doesn't refresh on plugin-originated writes, so the
        // drag-out overlay is the recommended path inside Live.
        .withNativeFunction(
            "copyVariationToClipboard",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (args.isEmpty()) { complete(juce::var(false)); return; }
                const int          idx      = static_cast<int>(args[0]);
                const juce::String baseName = args.size() > 1
                    ? args[1].toString() : juce::String("SA3");
                const juce::String stamp    = args.size() > 2
                    ? args[2].toString() : juce::String();
                const auto file = processor.getVariationsEngine()
                                      .writeVariationToTempFile(idx, baseName, stamp);
                if (! file.existsAsFile()) { complete(juce::var(false)); return; }
               #if JUCE_MAC
                const bool ok = sa3plugin::copyFileToClipboard(
                    file.getFullPathName().toRawUTF8());
                complete(juce::var(ok));
               #else
                // Other platforms — fall back to copying the path as text.
                juce::SystemClipboard::copyTextToClipboard(file.getFullPathName());
                complete(juce::var(true));
               #endif
            });
}

}  // namespace

SA3AudioProcessorEditor::SA3AudioProcessorEditor(SA3AudioProcessor& p)
    : AudioProcessorEditor(&p),
      processor(p),
      webView(buildOptions(p,
              [this](int idx, const juce::String& baseName,
                     const juce::String& stamp) {
                  const auto file = processor.getVariationsEngine()
                                        .writeVariationToTempFile(idx, baseName, stamp);
                  if (! file.existsAsFile()) return;
                  // canMoveFiles=false: Live (and Logic) silently no-op
                  // the drop if we declare we'd move the file.
                  juce::DragAndDropContainer::performExternalDragDropOfFiles(
                      { file.getFullPathName() }, /*canMoveFiles=*/false,
                      &webView);
              }))
{
    // 820 leaves enough vertical room for the source row + sigma slider
    // + generate button + 5 variation rows + the bottom keybinding hint
    // bar at the default DAW DPI. 720 clipped the footer once the
    // variations rendered.
    setSize(500, 820);
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
