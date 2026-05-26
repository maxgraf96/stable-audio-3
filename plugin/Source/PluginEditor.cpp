#include "PluginEditor.h"

SA3AudioProcessorEditor::SA3AudioProcessorEditor(SA3AudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    setSize(480, 240);

    titleLabel.setText("SA3", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(juce::FontOptions(40.0f, juce::Font::bold)));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);

    // Explicit UTF-8 — juce::String from a raw const char* assumes Latin-1 by
    // default, so a multi-byte UTF-8 sequence like `·` (0xC2 0xB7) gets read
    // as two glyphs (Â + ·).  fromUTF8 takes the literal as UTF-8.
    statusLabel.setText(processorRef.getPipelineLoader().getStatus(),
                       juce::dontSendNotification);
    statusLabel.setFont(juce::Font(juce::FontOptions(14.0f)));
    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(statusLabel);

    // Poll the background loader's status 10×/s while the editor is open.
    // Cheap (mutex-protected string read); auto-stops when the editor closes.
    startTimerHz(10);
}

SA3AudioProcessorEditor::~SA3AudioProcessorEditor() = default;

void SA3AudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(getLookAndFeel().findColour(juce::ResizableWindow::backgroundColourId));
}

void SA3AudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(20);
    titleLabel.setBounds(bounds.removeFromTop(80));
    bounds.removeFromTop(8);
    statusLabel.setBounds(bounds.removeFromTop(24));
}

void SA3AudioProcessorEditor::timerCallback()
{
    statusLabel.setText(processorRef.getPipelineLoader().getStatus(),
                        juce::dontSendNotification);
}
