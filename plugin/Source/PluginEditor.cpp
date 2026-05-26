#include "PluginEditor.h"

#include <cstdlib>

#include "BinaryData.h"

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

// JUCE passes us just the URL's path component (e.g. "/index.html"). Map
// that to the original resource filename ("index.html") for BinaryData
// lookup or disk read.
juce::String urlToResourceName(const juce::String& path) {
    juce::String name = path;
    if (name.startsWith("/")) name = name.substring(1);
    if (name.isEmpty() || name == "/") name = "index.html";
    return name;
}

// In dev, set SA3_PLUGIN_RESOURCES=/path/to/plugin/Resources to serve files
// straight from disk — iterate on HTML/CSS/JS without rebuilding.
juce::File devResourcesDir() {
    if (const char* env = std::getenv("SA3_PLUGIN_RESOURCES")) {
        juce::File dir(juce::String::fromUTF8(env));
        if (dir.isDirectory()) return dir;
    }
    return {};
}

std::optional<juce::WebBrowserComponent::Resource>
serveResource(const juce::String& url)
{
    const juce::String name = urlToResourceName(url);

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
    // getNamedResource() keys off the mangled name, so go through the parallel
    // originalFilenames[] / namedResourceList[] tables to look up by the
    // request's original filename.
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

juce::WebBrowserComponent::Options buildOptions(
    SA3AudioProcessor& processor)
{
    return juce::WebBrowserComponent::Options{}
        .withBackend(juce::WebBrowserComponent::Options::Backend::defaultBackend)
        .withNativeIntegrationEnabled(true)
        .withResourceProvider(&serveResource, juce::String(kOrigin))
        .withNativeFunction(
            "getStatus",
            [&processor](const juce::Array<juce::var>& /*args*/,
                         juce::WebBrowserComponent::NativeFunctionCompletion complete) {
                // Status is a mutex-protected string read; safe from any
                // thread, which is where JUCE invokes us.
                complete(juce::var(processor.getPipelineLoader().getStatus()));
            });
}

}  // namespace

SA3AudioProcessorEditor::SA3AudioProcessorEditor(SA3AudioProcessor& p)
    : AudioProcessorEditor(&p),
      webView(buildOptions(p))
{
    setSize(480, 240);
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
