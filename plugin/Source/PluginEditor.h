// SA3 plugin editor — Phase 2.3b: WebView shell.
//
// Hosts a juce::WebBrowserComponent that loads our baked HTML/CSS/JS from
// BinaryData via a resource provider. JS calls back into native code through
// JUCE 8's WebBrowserComponent::Options::withNativeFunction bridge — the only
// function exposed at this phase is `getStatus()` (returns the PipelineLoader
// state string). All other interactions (file drop, generate, audition) land
// in 2.3c.
#pragma once

#include <array>
#include <memory>

#include <JuceHeader.h>
#include "PluginProcessor.h"

class SA3AudioProcessorEditor : public juce::AudioProcessorEditor
{
public:
    explicit SA3AudioProcessorEditor(SA3AudioProcessor&);
    ~SA3AudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    // The processor reference is captured inside the WebView's native-function
    // lambda (see PluginEditor.cpp), so we don't need a separate member.
    juce::WebBrowserComponent webView;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SA3AudioProcessorEditor)
};
