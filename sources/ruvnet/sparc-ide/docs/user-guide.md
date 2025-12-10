# SPARC IDE User Guide

## Introduction

SPARC IDE is an AI-powered development environment built on VSCodium that integrates the SPARC methodology with Roo Code to enable prompt-driven development, autonomous agent workflows, and AI-native collaboration.

This guide will help you get started with SPARC IDE and make the most of its features.

## Table of Contents

1. [Installation](#installation)
2. [Getting Started](#getting-started)
3. [SPARC Methodology](#sparc-methodology)
4. [Roo Code Integration](#roo-code-integration)
5. [AI Model Configuration](#ai-model-configuration)
6. [Keyboard Shortcuts](#keyboard-shortcuts)
7. [MCP Server](#mcp-server)
8. [Customization](#customization)
9. [Troubleshooting](#troubleshooting)

## Installation

### Linux

1. Download the appropriate package for your distribution:
   - DEB package: `sparc-ide_1.0.0_amd64.deb`
   - RPM package: `sparc-ide-1.0.0.x86_64.rpm`

2. Install the package:
   - DEB: `sudo dpkg -i sparc-ide_1.0.0_amd64.deb`
   - RPM: `sudo rpm -i sparc-ide-1.0.0.x86_64.rpm`

3. Launch SPARC IDE from your applications menu or run `sparc-ide` in the terminal.

### Windows

1. Download the Windows installer: `sparc-ide-setup-1.0.0.exe`
2. Run the installer and follow the on-screen instructions.
3. Launch SPARC IDE from the Start menu.

### macOS

1. Download the macOS package: `sparc-ide-1.0.0.dmg`
2. Open the DMG file and drag SPARC IDE to the Applications folder.
3. Launch SPARC IDE from the Applications folder.

## Getting Started

When you first launch SPARC IDE, you'll be greeted with a welcome screen that introduces you to the SPARC methodology and Roo Code integration.

### Creating a New Project

1. Click on "File" > "New Project" or press `Ctrl+Shift+N`.
2. Select a project template or start from scratch.
3. Choose a location for your project.
4. SPARC IDE will automatically set up the project structure and initialize the SPARC workflow.

### Opening an Existing Project

1. Click on "File" > "Open Folder" or press `Ctrl+O`.
2. Navigate to your project folder and click "Open".
3. SPARC IDE will detect if the project is already set up for SPARC workflow. If not, you can initialize it by clicking on the SPARC icon in the activity bar and selecting "Initialize SPARC Workflow".

## SPARC Methodology

SPARC stands for Specification, Pseudocode, Architecture, Refinement, and Completion. It's a structured approach to software development that leverages AI to streamline the development process.

### Phases

#### 1. Specification

In this phase, you define the requirements, user stories, and acceptance criteria for your project. SPARC IDE provides templates and AI prompts to help you create comprehensive specifications.

To create a specification document:
1. Click on the SPARC icon in the activity bar.
2. Select "Specification" phase.
3. Click on "Create Specification Document" or use one of the provided templates.
4. Use the AI prompts to generate or refine your specifications.

#### 2. Pseudocode

In this phase, you create high-level pseudocode and logic flow diagrams to outline your implementation approach. SPARC IDE provides tools to create and visualize pseudocode.

To create pseudocode:
1. Click on the SPARC icon in the activity bar.
2. Select "Pseudocode" phase.
3. Click on "Create Pseudocode Document" or use one of the provided templates.
4. Use the AI prompts to generate or refine your pseudocode.

#### 3. Architecture

In this phase, you design the system architecture, component interactions, and data models. SPARC IDE provides tools to create architecture diagrams and define interfaces.

To create architecture documentation:
1. Click on the SPARC icon in the activity bar.
2. Select "Architecture" phase.
3. Click on "Create Architecture Document" or use one of the provided templates.
4. Use the AI prompts to generate or refine your architecture design.

#### 4. Refinement

In this phase, you implement, test, and refine your code. SPARC IDE provides tools for code generation, testing, and performance analysis.

To work on refinement:
1. Click on the SPARC icon in the activity bar.
2. Select "Refinement" phase.
3. Click on "Create Refinement Document" or use one of the provided templates.
4. Use the AI prompts to generate code, tests, or refactoring suggestions.

#### 5. Completion

In this phase, you finalize documentation, prepare for deployment, and establish maintenance procedures. SPARC IDE provides tools for documentation generation and deployment configuration.

To complete your project:
1. Click on the SPARC icon in the activity bar.
2. Select "Completion" phase.
3. Click on "Create Completion Document" or use one of the provided templates.
4. Use the AI prompts to generate documentation, deployment scripts, or maintenance procedures.

### Switching Between Phases

You can switch between SPARC phases using the SPARC panel in the activity bar or using keyboard shortcuts:
- `Ctrl+Alt+1`: Switch to Specification phase
- `Ctrl+Alt+2`: Switch to Pseudocode phase
- `Ctrl+Alt+3`: Switch to Architecture phase
- `Ctrl+Alt+4`: Switch to Refinement phase
- `Ctrl+Alt+5`: Switch to Completion phase

### Tracking Progress

SPARC IDE tracks your progress through the SPARC phases and provides a visual indicator of your current status. To view your progress:
1. Click on the SPARC icon in the activity bar.
2. Click on "Show Progress" or press `Ctrl+Alt+P`.

## Roo Code Integration

SPARC IDE integrates Roo Code to provide AI-powered coding assistance. Roo Code can help you with code generation, explanation, refactoring, documentation, and testing.

### Chat Interface

To open the Roo Code chat interface:
1. Press `Ctrl+Shift+A` or click on the Roo Code icon in the activity bar.
2. Type your question or request in the chat input.
3. Press Enter to send your message.
4. Roo Code will respond with suggestions, explanations, or code snippets.

### Code Generation

To generate code with Roo Code:
1. Open the Roo Code chat interface.
2. Describe the code you want to generate.
3. Roo Code will generate the code and provide an explanation.
4. Click on "Insert Code" or press `Ctrl+Shift+I` to insert the generated code at the cursor position.

### Code Explanation

To get an explanation of existing code:
1. Select the code you want to explain.
2. Press `Ctrl+Shift+E` or right-click and select "Explain Code".
3. Roo Code will provide an explanation of the selected code.

### Code Refactoring

To refactor code with Roo Code:
1. Select the code you want to refactor.
2. Press `Ctrl+Shift+R` or right-click and select "Refactor Code".
3. Describe how you want to refactor the code.
4. Roo Code will suggest refactored code.
5. Click on "Apply Refactoring" to apply the suggested changes.

### Code Documentation

To generate documentation for your code:
1. Select the code you want to document.
2. Press `Ctrl+Shift+D` or right-click and select "Document Code".
3. Roo Code will generate documentation comments for the selected code.
4. Click on "Apply Documentation" to insert the documentation.

### Test Generation

To generate tests for your code:
1. Select the code you want to test.
2. Press `Ctrl+Shift+T` or right-click and select "Generate Tests".
3. Roo Code will generate test cases for the selected code.
4. Click on "Create Test File" to create a new test file with the generated tests.

## AI Model Configuration

SPARC IDE supports multiple AI models for different tasks. You can configure which model to use for each task and set up API keys.

### Configuring API Keys

1. Open Settings (File > Preferences > Settings).
2. Search for "roo-code.apiKey".
3. Enter your OpenRouter API key.
4. Optionally, configure keys for other AI providers:
   - "roo-code.anthropicApiKey" for Claude
   - "roo-code.openaiApiKey" for GPT-4
   - "roo-code.googleApiKey" for Gemini

### Switching Between Models

You can switch between AI models using keyboard shortcuts:
- `Ctrl+Shift+1`: Switch to OpenRouter
- `Ctrl+Shift+2`: Switch to Claude
- `Ctrl+Shift+3`: Switch to GPT-4
- `Ctrl+Shift+4`: Switch to Gemini

Or through the Roo Code settings:
1. Open the Roo Code chat interface.
2. Click on the settings icon.
3. Select "Change Model".
4. Choose the model you want to use.

### Custom AI Modes

SPARC IDE provides custom AI modes for specific tasks:
- QA Engineer: Detect edge cases and write tests
- Architect: Design scalable and maintainable systems
- Code Reviewer: Identify issues and suggest improvements
- Documentation Writer: Create clear and comprehensive documentation

To switch between AI modes:
- `Ctrl+Shift+Q`: Switch to QA Engineer mode
- `Ctrl+Shift+S`: Switch to Architect mode
- `Ctrl+Shift+C`: Switch to Code Reviewer mode
- `Ctrl+Shift+W`: Switch to Documentation Writer mode

Or through the Roo Code settings:
1. Open the Roo Code chat interface.
2. Click on the settings icon.
3. Select "Change Mode".
4. Choose the mode you want to use.

## Keyboard Shortcuts

SPARC IDE provides a variety of keyboard shortcuts to help you work efficiently:

### General Shortcuts

- `Ctrl+P`: Quick Open
- `Ctrl+Shift+P`: Show Command Palette
- `Ctrl+O`: Open Folder
- `Ctrl+S`: Save
- `Ctrl+W`: Close Editor
- `Ctrl+Tab`: Switch between open editors
- `Ctrl+\`: Split Editor
- `Ctrl+B`: Toggle Sidebar
- `Ctrl+J`: Toggle Panel
- `Ctrl+K Ctrl+S`: Open Keyboard Shortcuts

### SPARC Workflow Shortcuts

- `Ctrl+Alt+1`: Switch to Specification phase
- `Ctrl+Alt+2`: Switch to Pseudocode phase
- `Ctrl+Alt+3`: Switch to Architecture phase
- `Ctrl+Alt+4`: Switch to Refinement phase
- `Ctrl+Alt+5`: Switch to Completion phase
- `Ctrl+Alt+T`: Create template for current SPARC phase
- `Ctrl+Alt+A`: Create artifact for current SPARC phase
- `Ctrl+Alt+P`: Show SPARC progress

### Roo Code Shortcuts

- `Ctrl+Shift+A`: Open AI chat
- `Ctrl+Shift+I`: Insert AI-generated code
- `Ctrl+Shift+E`: Explain selected code
- `Ctrl+Shift+R`: Refactor selected code
- `Ctrl+Shift+D`: Document selected code
- `Ctrl+Shift+T`: Generate tests for selected code

### AI Model Shortcuts

- `Ctrl+Shift+1`: Switch to OpenRouter
- `Ctrl+Shift+2`: Switch to Claude
- `Ctrl+Shift+3`: Switch to GPT-4
- `Ctrl+Shift+4`: Switch to Gemini

### AI Mode Shortcuts

- `Ctrl+Shift+Q`: Switch to QA Engineer mode
- `Ctrl+Shift+S`: Switch to Architect mode
- `Ctrl+Shift+C`: Switch to Code Reviewer mode
- `Ctrl+Shift+W`: Switch to Documentation Writer mode

### UI Shortcuts

- `Ctrl+Alt+M`: Toggle minimal mode
- `Ctrl+Alt+F`: Toggle focus mode
- `Ctrl+Alt+L`: Switch layout
- `Ctrl+Alt+D`: Switch theme

## MCP Server

SPARC IDE includes an MCP (Model Context Protocol) server that provides additional tools and resources for AI-powered development.

### Starting the MCP Server

1. Open a terminal.
2. Navigate to the SPARC IDE installation directory.
3. Run `src/mcp/start-mcp-server.sh`.

### Available MCP Tools

The MCP server provides the following tools:

#### Code Analysis

Analyzes code to provide insights and suggestions.

Usage:
1. Select the code you want to analyze.
2. Right-click and select "Analyze Code with MCP".
3. The analysis results will be displayed in the MCP panel.

#### Code Modification

Modifies code based on instructions.

Usage:
1. Select the code you want to modify.
2. Right-click and select "Modify Code with MCP".
3. Enter instructions for how to modify the code.
4. The modified code will be displayed in the MCP panel.

#### Code Search

Searches code for patterns or specific content.

Usage:
1. Open the MCP panel.
2. Click on "Search Code".
3. Enter a search pattern.
4. Select the files to search in.
5. The search results will be displayed in the MCP panel.

## Customization

SPARC IDE is highly customizable to suit your preferences and workflow.

### Themes

SPARC IDE comes with two custom themes:
- Dracula Pro: A dark theme with vibrant colors
- Material Theme: A modern, material design-inspired theme

To change the theme:
1. Open Settings (File > Preferences > Settings).
2. Search for "workbench.colorTheme".
3. Select your preferred theme from the dropdown.

Or use the keyboard shortcut `Ctrl+Alt+D` to switch between themes.

### Layout

SPARC IDE provides different layout options:
- AI-centric: Optimized for AI-powered development
- Minimal: A simplified layout with fewer distractions
- Default: The standard VSCode layout

To change the layout:
1. Open Settings (File > Preferences > Settings).
2. Search for "ui-ux.customLayout".
3. Select your preferred layout from the dropdown.

Or use the keyboard shortcut `Ctrl+Alt+L` to switch between layouts.

### Settings

You can customize various aspects of SPARC IDE through the Settings:
1. Open Settings (File > Preferences > Settings).
2. Browse or search for settings to customize.

Some notable settings include:
- `editor.fontFamily`: Change the font used in the editor
- `editor.fontSize`: Change the font size in the editor
- `workbench.iconTheme`: Change the icon theme
- `roo-code.defaultModel`: Set the default AI model
- `sparc-workflow.defaultPhase`: Set the default SPARC phase

## Troubleshooting

### Common Issues

#### API Key Issues

If you're experiencing issues with AI features:
1. Check that you've entered the correct API keys in Settings.
2. Verify that your API keys are valid and have sufficient credits.
3. Check your internet connection.

#### MCP Server Issues

If the MCP server is not working:
1. Make sure the MCP server is running.
2. Check the MCP server logs for errors.
3. Restart the MCP server.

#### Performance Issues

If SPARC IDE is running slowly:
1. Close unused editors and terminals.
2. Disable unused extensions.
3. Adjust performance settings in Settings.

### Getting Help

If you need help with SPARC IDE:
1. Check the documentation in the `docs` directory.
2. Visit the SPARC IDE website at [https://sparc-ide.github.io](https://sparc-ide.github.io).
3. Join the SPARC IDE community on [Discord](https://discord.gg/sparc-ide).
4. Open an issue on the [GitHub repository](https://github.com/sparc-ide/sparc-ide).

## Conclusion

SPARC IDE combines the power of the SPARC methodology with Roo Code's AI capabilities to provide a comprehensive development environment for modern software development. By following this guide, you'll be able to make the most of SPARC IDE's features and streamline your development workflow.

Happy coding with SPARC IDE!