// SA3 plugin — Phase 1 audio processor.
//
// Minimal AudioProcessor: passes silence on the audio thread, no parameters
// yet. Phase 2 will wire in the C++ MLX pipeline (sa3::orch::Pipeline) on a
// background thread and play back the generated WAV from the audio thread.
#pragma once

#include <mutex>

#include <JuceHeader.h>

#include "VariationsEngine.h"

class SA3AudioProcessor : public juce::AudioProcessor
{
public:
    sa3plugin::VariationsEngine& getVariationsEngine() { return variationsEngine; }
    const sa3plugin::VariationsEngine& getVariationsEngine() const { return variationsEngine; }

    // Opaque blob the editor uses to round-trip JS-side UI state through
    // getStateInformation / setStateInformation. We persist a JSON string
    // (preset, noise, bpm, key, prompt) — small, schema-flexible, no need
    // for a binary format. The audio file is intentionally not persisted:
    // re-base64ing it across save/load would bloat project files.
    juce::String getPersistedUiStateJson() const;
    void         setPersistedUiStateJson(juce::String json);

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
    // Single MLX-owning worker thread.  Handles both the one-shot pipeline
    // load (triggered on first createEditor() so plugin scan / auval don't
    // pay the cost) and every subsequent variation generate request from
    // the WebView editor.  MLX binds default streams to the thread that
    // first touches them — one worker means one stream context, so arrays
    // created at load time stay reachable from generate calls.  Destructor
    // joins the worker on plugin unload.
    sa3plugin::VariationsEngine variationsEngine;

    // JS-side UI state (preset, noise, bpm, key, prompt) round-tripped as a
    // JSON string.  Written by the editor's setUiState native fn; read by
    // get/setStateInformation.  Mutex protects access from the JUCE message
    // thread (writes) and DAW serialization thread (reads).
    mutable std::mutex uiStateMutex;
    juce::String       uiStateJson;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SA3AudioProcessor)
};
