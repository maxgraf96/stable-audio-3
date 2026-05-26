#include "PluginProcessor.h"
#include "PluginEditor.h"

SA3AudioProcessor::SA3AudioProcessor()
    : AudioProcessor(BusesProperties()
          .withInput("Input",  juce::AudioChannelSet::stereo(), true)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

SA3AudioProcessor::~SA3AudioProcessor() = default;

void SA3AudioProcessor::prepareToPlay(double /*sampleRate*/, int /*samplesPerBlock*/) {}
void SA3AudioProcessor::releaseResources() {}

bool SA3AudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& main = layouts.getMainOutputChannelSet();
    if (main != juce::AudioChannelSet::stereo() && main != juce::AudioChannelSet::mono())
        return false;
    // Mirror input on the main bus.
    return main == layouts.getMainInputChannelSet();
}

void SA3AudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midi*/)
{
    juce::ScopedNoDenormals noDenormals;
    // Phase 1: silent passthrough. Clear any output channels beyond the input
    // count so the host doesn't get uninitialized memory.
    const auto numIn  = getTotalNumInputChannels();
    const auto numOut = getTotalNumOutputChannels();
    for (int ch = numIn; ch < numOut; ++ch)
        buffer.clear(ch, 0, buffer.getNumSamples());
    // For now: zero out the output. Phase 2 will fill from the generated WAV.
    for (int ch = 0; ch < numOut; ++ch)
        buffer.clear(ch, 0, buffer.getNumSamples());
}

juce::AudioProcessorEditor* SA3AudioProcessor::createEditor()
{
    // First editor open triggers the background model load. Idempotent: no-op
    // on subsequent calls or once the load has finished. Plugin scan never
    // opens the editor, so AU validation / DAW scan stays fast.
    variationsEngine.requestLoad();
    return new SA3AudioProcessorEditor(*this);
}

bool SA3AudioProcessor::hasEditor() const { return true; }

const juce::String SA3AudioProcessor::getName() const { return JucePlugin_Name; }

bool   SA3AudioProcessor::acceptsMidi() const         { return false; }
bool   SA3AudioProcessor::producesMidi() const        { return false; }
bool   SA3AudioProcessor::isMidiEffect() const        { return false; }
double SA3AudioProcessor::getTailLengthSeconds() const { return 0.0; }

int  SA3AudioProcessor::getNumPrograms()                              { return 1; }
int  SA3AudioProcessor::getCurrentProgram()                           { return 0; }
void SA3AudioProcessor::setCurrentProgram(int /*index*/)              {}
const juce::String SA3AudioProcessor::getProgramName(int /*index*/)   { return {}; }
void SA3AudioProcessor::changeProgramName(int, const juce::String&)   {}

juce::String SA3AudioProcessor::getPersistedUiStateJson() const
{
    std::lock_guard<std::mutex> lk(uiStateMutex);
    return uiStateJson;
}

void SA3AudioProcessor::setPersistedUiStateJson(juce::String json)
{
    std::lock_guard<std::mutex> lk(uiStateMutex);
    uiStateJson = std::move(json);
}

void SA3AudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    const juce::String json = getPersistedUiStateJson();
    destData.append(json.toRawUTF8(), json.getNumBytesAsUTF8());
}

void SA3AudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (data == nullptr || sizeInBytes <= 0) return;
    setPersistedUiStateJson(juce::String::fromUTF8(
        static_cast<const char*>(data), sizeInBytes));
}

// JUCE entry point for the plugin factory.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SA3AudioProcessor();
}
