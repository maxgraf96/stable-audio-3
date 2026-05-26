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

void SA3AudioProcessor::getStateInformation(juce::MemoryBlock& /*destData*/)        {}
void SA3AudioProcessor::setStateInformation(const void* /*data*/, int /*sizeInBytes*/) {}

// JUCE entry point for the plugin factory.
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SA3AudioProcessor();
}
