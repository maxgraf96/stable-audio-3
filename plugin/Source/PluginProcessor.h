// SA3 plugin — Phase 1 audio processor.
//
// Minimal AudioProcessor: passes silence on the audio thread, no parameters
// yet. Phase 2 will wire in the C++ MLX pipeline (sa3::orch::Pipeline) on a
// background thread and play back the generated WAV from the audio thread.
#pragma once

#include <JuceHeader.h>

#include "PipelineLoader.h"

class SA3AudioProcessor : public juce::AudioProcessor
{
public:
    sa3plugin::PipelineLoader& getPipelineLoader() { return pipelineLoader; }
    const sa3plugin::PipelineLoader& getPipelineLoader() const { return pipelineLoader; }

    SA3AudioProcessor();
    ~SA3AudioProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override;

    const juce::String getName() const override;
    bool   acceptsMidi() const override;
    bool   producesMidi() const override;
    bool   isMidiEffect() const override;
    double getTailLengthSeconds() const override;

    int  getNumPrograms() override;
    int  getCurrentProgram() override;
    void setCurrentProgram(int index) override;
    const juce::String getProgramName(int index) override;
    void changeProgramName(int index, const juce::String& newName) override;

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

private:
    // Background-thread loader. Constructed in NotLoaded state; load is
    // triggered on first createEditor() call (so plugin scan / auval don't
    // pay the cost). Lifetime tied to the processor — destructor joins the
    // worker on plugin unload.
    sa3plugin::PipelineLoader pipelineLoader;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SA3AudioProcessor)
};
