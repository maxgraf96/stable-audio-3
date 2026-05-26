// SA3 plugin — Phase 1 editor.
//
// Placeholder UI: just a title and a status line so we can verify the plugin
// loads in a DAW and the window opens. Phase 2 will add the file drop zone,
// prompt input, Generate button, and audition pads.
#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

class SA3AudioProcessorEditor : public juce::AudioProcessorEditor,
                                private juce::Timer
{
public:
    explicit SA3AudioProcessorEditor(SA3AudioProcessor&);
    ~SA3AudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    void timerCallback() override;   // polls PipelineLoader status

    SA3AudioProcessor& processorRef;
    juce::Label titleLabel;
    juce::Label statusLabel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SA3AudioProcessorEditor)
};
