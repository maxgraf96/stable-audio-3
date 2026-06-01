// SA3 Morph — audio processor.
//
// Generator-style plugin: it does not process host input. processBlock pulls
// continuously-morphing audio from the MorphEngine (which owns the C++ streaming
// stack + its MLX worker thread) and writes it to the output bus. The engine is
// a long-lived member so morph/playback state survives editor open/close.
#pragma once

#include <mutex>

#include <JuceHeader.h>

#include "MorphEngine.h"

class SA3MorphProcessor : public juce::AudioProcessor
{
public:
    sa3morph::MorphEngine&       getEngine()       { return engine; }
    const sa3morph::MorphEngine& getEngine() const { return engine; }

    // Opaque JSON blob round-tripped through get/setStateInformation so the
    // WebView UI state (prompt, knob values, evolve) persists with the project.
    juce::String getPersistedUiStateJson() const;
    void         setPersistedUiStateJson(juce::String json);

    SA3MorphProcessor();
    ~SA3MorphProcessor() override;

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
    sa3morph::MorphEngine engine;

    mutable std::mutex uiStateMutex;
    juce::String       uiStateJson;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SA3MorphProcessor)
};
