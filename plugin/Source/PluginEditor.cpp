#include "PluginEditor.h"

SA3AudioProcessorEditor::SA3AudioProcessorEditor(SA3AudioProcessor& p)
    : AudioProcessorEditor(&p), processorRef(p)
{
    setSize(480, 240);

    titleLabel.setText("SA3", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(juce::FontOptions(40.0f, juce::Font::bold)));
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);

    statusLabel.setText("Phase 1 skeleton  ·  no inference wired yet",
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
