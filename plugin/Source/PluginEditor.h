// SA3 plugin editor.
//
// Hosts a juce::WebBrowserComponent that loads our baked HTML/CSS/JS from
// BinaryData via a resource provider. JS calls back into native code via
// JUCE 8's WebBrowserComponent::Options::withNativeFunction bridge and
// fires fire-and-forget messages via emitEvent + withEventListener.
//
// Drag-out: JS's `mousedown` on a row's grip emits a JUCE event; the
// listener immediately calls performExternalDragDropOfFiles with &webView
// as the source. This is the only WKWebView-compatible path — sibling
// NSView overlays don't intercept events because WKWebView is rendered
// out-of-process and owns hit-testing for its whole rect. Reference:
// JUCE forum thread 64813.
#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class SA3AudioProcessorEditor : public juce::AudioProcessorEditor,
                                public juce::DragAndDropContainer
{
public:
    explicit SA3AudioProcessorEditor(SA3AudioProcessor&);
    ~SA3AudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    // WebView2 reports spurious navigation failures for resource-provider
    // origins — most commonly status 9 — even when every request was served.
    // JUCE's default pageLoadHadNetworkError returns true, which navigates the
    // view to a `data:text/plain,Error code: 9` page and throws our whole UI
    // away. Swallowing it is what JUCE's own source recommends in that code
    // path. Returning false leaves the already-loaded page alone.
    class SilentWebView final : public juce::WebBrowserComponent
    {
    public:
        using juce::WebBrowserComponent::WebBrowserComponent;
        bool pageLoadHadNetworkError(const juce::String&) override { return false; }
    };

    SA3AudioProcessor&  processor;
    SilentWebView       webView;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SA3AudioProcessorEditor)
};
