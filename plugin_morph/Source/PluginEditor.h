// SA3 Morph editor.
//
// Hosts a juce::WebBrowserComponent loading baked HTML/CSS/JS from BinaryData
// via a resource provider. JS calls native functions (load source, transport,
// live morph knobs) through JUCE 8's withNativeFunction bridge; the editor
// PUSHES live morph state to JS each timer tick via emitEventIfBrowserIsVisible
// ("morphState") — the C++->JS direction the Variations plugin doesn't use.
#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class SA3MorphProcessorEditor : public juce::AudioProcessorEditor,
                                public juce::DragAndDropContainer,
                                private juce::Timer
{
public:
    explicit SA3MorphProcessorEditor(SA3MorphProcessor&);
    ~SA3MorphProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;   // pushes morphState to JS

    SA3MorphProcessor&        processor;
    juce::WebBrowserComponent webView;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SA3MorphProcessorEditor)
};
