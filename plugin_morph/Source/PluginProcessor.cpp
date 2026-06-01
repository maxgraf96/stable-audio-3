#include "PluginProcessor.h"
#include "PluginEditor.h"

SA3MorphProcessor::SA3MorphProcessor()
    // Disabled stereo input bus (see plugin/ rationale): satisfies Live's VST3
    // bus negotiation without enabling input on what is a generator.
    : AudioProcessor(BusesProperties()
          .withInput ("Input",  juce::AudioChannelSet::stereo(), false)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

SA3MorphProcessor::~SA3MorphProcessor() = default;

void SA3MorphProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    engine.prepareToPlay(samplesPerBlock, sampleRate);
}
void SA3MorphProcessor::releaseResources()
{
    engine.releaseResources();
}

bool SA3MorphProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    const auto& main = layouts.getMainOutputChannelSet();
    return main == juce::AudioChannelSet::stereo()
        || main == juce::AudioChannelSet::mono();
}

void SA3MorphProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midi*/)
{
    juce::ScopedNoDenormals noDenormals;
    // Generator: replace the host input with the continuously-morphing loop.
    juce::AudioSourceChannelInfo info(&buffer, 0, buffer.getNumSamples());
    engine.getNextAudioBlock(info);
}

juce::AudioProcessorEditor* SA3MorphProcessor::createEditor()
{
    return new SA3MorphProcessorEditor(*this);
}

bool SA3MorphProcessor::hasEditor() const { return true; }

const juce::String SA3MorphProcessor::getName() const { return JucePlugin_Name; }

bool   SA3MorphProcessor::acceptsMidi() const          { return false; }
bool   SA3MorphProcessor::producesMidi() const         { return false; }
bool   SA3MorphProcessor::isMidiEffect() const         { return false; }
double SA3MorphProcessor::getTailLengthSeconds() const { return 0.0; }

int  SA3MorphProcessor::getNumPrograms()                            { return 1; }
int  SA3MorphProcessor::getCurrentProgram()                         { return 0; }
void SA3MorphProcessor::setCurrentProgram(int /*index*/)            {}
const juce::String SA3MorphProcessor::getProgramName(int /*index*/) { return {}; }
void SA3MorphProcessor::changeProgramName(int, const juce::String&) {}

juce::String SA3MorphProcessor::getPersistedUiStateJson() const
{
    std::lock_guard<std::mutex> lk(uiStateMutex);
    return uiStateJson;
}
void SA3MorphProcessor::setPersistedUiStateJson(juce::String json)
{
    std::lock_guard<std::mutex> lk(uiStateMutex);
    uiStateJson = std::move(json);
}

void SA3MorphProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    const juce::String json = getPersistedUiStateJson();
    destData.append(json.toRawUTF8(), json.getNumBytesAsUTF8());
}
void SA3MorphProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    if (data == nullptr || sizeInBytes <= 0) return;
    setPersistedUiStateJson(juce::String::fromUTF8(
        static_cast<const char*>(data), sizeInBytes));
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new SA3MorphProcessor();
}
