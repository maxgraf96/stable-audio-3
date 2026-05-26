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
    statusLabel.setText(juce::String::fromUTF8("Phase 2 skeleton  ·  sa3_orchestrator linked"),
                       juce::dontSendNotification);
    statusLabel.setFont(juce::Font(juce::FontOptions(14.0f)));
    statusLabel.setJustificationType(juce::Justification::centred);
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::grey);
    addAndMakeVisible(statusLabel);
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
