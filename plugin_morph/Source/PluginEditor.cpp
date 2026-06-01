#include "PluginEditor.h"

#include <cstdlib>
#include <cstring>
#include <optional>

#include "BinaryData.h"
#include "MorphEngine.h"

namespace {

// JUCE 8 custom scheme — only juce:// URLs hit the resource provider (which is
// passed just the path component, e.g. "/index.html").
constexpr const char* kOrigin   = "juce://sa3morph.local";
constexpr const char* kIndexURL = "juce://sa3morph.local/index.html";

juce::String mimeFor(const juce::String& path) {
    if (path.endsWithIgnoreCase(".html")) return "text/html";
    if (path.endsWithIgnoreCase(".js"))   return "application/javascript";
    if (path.endsWithIgnoreCase(".css"))  return "text/css";
    if (path.endsWithIgnoreCase(".svg"))  return "image/svg+xml";
    if (path.endsWithIgnoreCase(".png"))  return "image/png";
    if (path.endsWithIgnoreCase(".json")) return "application/json";
    return "application/octet-stream";
}

juce::String normalisePath(const juce::String& path) {
    juce::String p = path;
    if (p.startsWith("/")) p = p.substring(1);
    if (p.isEmpty())       p = "index.html";
    const int q = p.indexOfChar('?');
    if (q >= 0) p = p.substring(0, q);
    return p;
}

juce::File devResourcesDir() {
    if (const char* env = std::getenv("SA3_MORPH_RESOURCES")) {
        juce::File dir(juce::String::fromUTF8(env));
        if (dir.isDirectory()) return dir;
    }
    return {};
}

std::optional<juce::WebBrowserComponent::Resource>
serveStaticResource(const juce::String& name) {
    if (auto dir = devResourcesDir(); dir != juce::File{}) {
        juce::File f = dir.getChildFile(name);
        if (f.existsAsFile()) {
            juce::MemoryBlock mb;
            if (f.loadFileAsData(mb)) {
                std::vector<std::byte> bytes(mb.getSize());
                std::memcpy(bytes.data(), mb.getData(), mb.getSize());
                return juce::WebBrowserComponent::Resource{std::move(bytes), mimeFor(name)};
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
                return juce::WebBrowserComponent::Resource{std::move(bytes), mimeFor(name)};
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

juce::var peaksToVar(const std::vector<float>& peaks) {
    juce::Array<juce::var> arr;
    arr.ensureStorageAllocated(static_cast<int>(peaks.size()));
    for (float p : peaks) arr.add(juce::var(p));
    return juce::var(arr);
}

juce::WebBrowserComponent::Options buildOptions(SA3MorphProcessor& processor) {
    auto resourceProvider = [](const juce::String& url)
            -> std::optional<juce::WebBrowserComponent::Resource> {
        return serveStaticResource(normalisePath(url));
    };

    return juce::WebBrowserComponent::Options{}
        .withBackend(juce::WebBrowserComponent::Options::Backend::defaultBackend)
        .withNativeIntegrationEnabled(true)
        .withResourceProvider(resourceProvider, juce::String(kOrigin))
        // ── status / persistence ─────────────────────────────────────
        .withNativeFunction("getStatus",
            [&processor](const juce::Array<juce::var>&,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                auto& eng = processor.getEngine();
                const auto ph = eng.getPhase();
                const char* phaseStr =
                    ph == sa3morph::MorphEngine::Phase::Idle    ? "idle"    :
                    ph == sa3morph::MorphEngine::Phase::Loading ? "loading" :
                    ph == sa3morph::MorphEngine::Phase::Running ? "running" : "error";
                juce::DynamicObject::Ptr o = new juce::DynamicObject();
                o->setProperty("status", eng.getStatus());
                o->setProperty("phase",  juce::String(phaseStr));
                o->setProperty("playing", eng.isPlaying());
                complete(juce::var(o.get()));
            })
        .withNativeFunction("getUiState",
            [&processor](const juce::Array<juce::var>&,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                complete(juce::var(processor.getPersistedUiStateJson()));
            })
        .withNativeFunction("setUiState",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (! args.isEmpty()) {
                    processor.setPersistedUiStateJson(args[0].toString());
                    processor.updateHostDisplay();
                }
                complete(juce::var());
            })
        // ── loadSource(base64, peaksN) → { ok, duration, peaks } ──────
        .withNativeFunction("loadSource",
            [&processor](const juce::Array<juce::var>& args,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                auto err = [&complete](const juce::String& m) {
                    juce::DynamicObject::Ptr e = new juce::DynamicObject();
                    e->setProperty("ok", false); e->setProperty("error", m);
                    complete(juce::var(e.get()));
                };
                if (args.isEmpty()) { err("loadSource(): no arguments"); return; }
                const juce::var& v = args[0];
                const juce::String b64 = v["audioBase64"].toString();
                const int peaksN = v.hasProperty("peaksN") ? (int)v["peaksN"] : 220;
                if (b64.isEmpty()) { err("audioBase64 missing"); return; }
                juce::MemoryBlock bytes = base64Decode(b64);
                if (bytes.getSize() == 0) { err("audioBase64 failed to decode"); return; }
                const bool accepted = processor.getEngine().requestLoadSource(
                    std::move(bytes), peaksN,
                    [complete](juce::var result) { complete(result); });
                if (! accepted) err("a load is already in progress");
            })
        // ── transport ─────────────────────────────────────────────────
        .withNativeFunction("startMorph",
            [&processor](const juce::Array<juce::var>&,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                processor.getEngine().startPlayback();
                complete(juce::var());
            })
        .withNativeFunction("stopMorph",
            [&processor](const juce::Array<juce::var>&,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                processor.getEngine().stopPlayback();
                complete(juce::var());
            })
        // ── live morph controls ───────────────────────────────────────
        .withNativeFunction("setDenoise",
            [&processor](const juce::Array<juce::var>& a,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (! a.isEmpty()) processor.getEngine().setDenoise((float)(double)a[0]);
                complete(juce::var());
            })
        .withNativeFunction("setSourceBlend",
            [&processor](const juce::Array<juce::var>& a,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (! a.isEmpty()) processor.getEngine().setSourceBlend((float)(double)a[0]);
                complete(juce::var());
            })
        .withNativeFunction("setVelocity",
            [&processor](const juce::Array<juce::var>& a,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                if (! a.isEmpty()) processor.getEngine().setVelocity((float)(double)a[0]);
                complete(juce::var());
            })
        .withNativeFunction("setEvolve",
            [&processor](const juce::Array<juce::var>& a,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                processor.getEngine().setEvolve(! a.isEmpty() && (bool)a[0]);
                complete(juce::var());
            })
        .withNativeFunction("setPrompt",
            [&processor](const juce::Array<juce::var>& a,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                const juce::String t = a.isEmpty() ? juce::String() : a[0].toString();
                processor.getEngine().setPrompt(t.toStdString());
                complete(juce::var());
            })
        // setQuality(bool) -> changed:bool. Model family is fixed per generator,
        // so JS reloads the current source when this returns true.
        .withNativeFunction("setQuality",
            [&processor](const juce::Array<juce::var>& a,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                const bool on = ! a.isEmpty() && (bool)a[0];
                complete(juce::var(processor.getEngine().setQuality(on)));
            })
        .withNativeFunction("getSourcePeaks",
            [&processor](const juce::Array<juce::var>&,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                complete(peaksToVar(processor.getEngine().getSourcePeaks()));
            });
}

}  // namespace

SA3MorphProcessorEditor::SA3MorphProcessorEditor(SA3MorphProcessor& p)
    : AudioProcessorEditor(&p),
      processor(p),
      webView(buildOptions(p))
{
    setSize(854, 480);   // 16:9
    addAndMakeVisible(webView);
    webView.goToURL(kIndexURL);
    startTimerHz(30);   // push morphState to JS at ~30 fps
}

SA3MorphProcessorEditor::~SA3MorphProcessorEditor() { stopTimer(); }

void SA3MorphProcessorEditor::timerCallback()
{
    const auto s = processor.getEngine().getMorphState();
    juce::DynamicObject::Ptr o = new juce::DynamicObject();
    o->setProperty("phase",       s.phase);
    o->setProperty("playing",     s.playing);
    o->setProperty("progress",    s.progress);
    o->setProperty("loopSecs",    s.loopSecs);
    o->setProperty("denoise",     s.denoise);
    o->setProperty("sourceBlend", s.sourceBlend);
    o->setProperty("velocity",    s.velocity);
    o->setProperty("evolve",      s.evolve);
    o->setProperty("tickMs",      s.tickMs);
    o->setProperty("decodeMs",    s.decodeMs);
    o->setProperty("completions", (int)s.completions);
    webView.emitEventIfBrowserIsVisible("morphState", juce::var(o.get()));
}

void SA3MorphProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void SA3MorphProcessorEditor::resized()
{
    webView.setBounds(getLocalBounds());
}
