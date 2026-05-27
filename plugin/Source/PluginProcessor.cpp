#include "PluginProcessor.h"
#include "PluginEditor.h"

SA3AudioProcessor::SA3AudioProcessor()
    // Declared stereo input bus, disabled by default. We never read it —
    // this plugin is a sample generator — but Ableton Live 12's VST3 host
    // refuses to instantiate the plugin if `setBusArrangements` can't
    // negotiate an input arrangement, and a default-disabled bus still
    // satisfies that handshake. The standalone's "Audio input is muted to
    // avoid feedback loop" banner stays suppressed because it only checks
    // *enabled* input buses.
    : AudioProcessor(BusesProperties()
          .withInput ("Input",  juce::AudioChannelSet::stereo(), false)
          .withOutput("Output", juce::AudioChannelSet::stereo(), true))
{
}

SA3AudioProcessor::~SA3AudioProcessor() = default;

void SA3AudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    // Forward to the engine — it acts as the audio source pulling stored
    // decoded source / variation buffers into the host's output block.
    variationsEngine.prepareToPlay(samplesPerBlock, sampleRate);
}
void SA3AudioProcessor::releaseResources()
{
    variationsEngine.releaseResources();
}

bool SA3AudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // No input bus to mirror — only the main output bus needs to be mono
    // or stereo. The DAW (and the standalone) will only ever query this
    // for the output.
    const auto& main = layouts.getMainOutputChannelSet();
    return main == juce::AudioChannelSet::stereo()
        || main == juce::AudioChannelSet::mono();
}

void SA3AudioProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& /*midi*/)
{
    juce::ScopedNoDenormals noDenormals;
    // Always replace the host's input — we're a generator, not an effect.
    // The engine writes the active source/variation buffer at its own play
    // position; if nothing's playing it fills with silence via
    // clearActiveBufferRegion inside getNextAudioBlock.
    juce::AudioSourceChannelInfo info(&buffer, 0, buffer.getNumSamples());
    variationsEngine.getNextAudioBlock(info);
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
