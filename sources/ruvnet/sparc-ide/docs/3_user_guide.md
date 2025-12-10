# SPARC IDE User Guide

This comprehensive guide explains how to use SPARC IDE effectively, including customizing the UI/UX, working with the SPARC methodology, and leveraging Roo Code for AI-assisted development.

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [User Interface Overview](#user-interface-overview)
4. [Customizing the UI/UX](#customizing-the-uiux)
5. [SPARC Methodology Workflow](#sparc-methodology-workflow)
6. [Working with Roo Code](#working-with-roo-code)
7. [AI Models and Modes](#ai-models-and-modes)
8. [Keyboard Shortcuts](#keyboard-shortcuts)
9. [Advanced Features](#advanced-features)
10. [Tips and Best Practices](#tips-and-best-practices)

## Introduction

SPARC IDE is an AI-powered development environment built on VSCodium that integrates the SPARC methodology with Roo Code to enable prompt-driven development, autonomous agent workflows, and AI-native collaboration.

This guide will help you make the most of SPARC IDE's features and capabilities, whether you're a new user or an experienced developer looking to optimize your workflow.

## Getting Started

### First Launch Experience

When you first launch SPARC IDE, you'll be greeted with a welcome screen that introduces you to the key features and capabilities. This screen provides quick access to:

- Create a new project
- Open an existing project
- Learn about the SPARC methodology
- Configure Roo Code integration
- Explore sample projects

### Creating a New Project

To create a new project:

1. Click on "File" > "New Project" or press `Ctrl+Shift+N`
2. Select a project template or start from scratch
3. Choose a location for your project
4. SPARC IDE will automatically set up the project structure and initialize the SPARC workflow

### Opening an Existing Project

To open an existing project:

1. Click on "File" > "Open Folder" or press `Ctrl+O`
2. Navigate to your project folder and click "Open"
3. SPARC IDE will detect if the project is already set up for SPARC workflow
4. If not, you can initialize it by clicking on the SPARC icon in the activity bar and selecting "Initialize SPARC Workflow"

## User Interface Overview

SPARC IDE features an AI-centric layout designed to optimize your development workflow:

### Activity Bar

The activity bar on the left side of the window provides access to the main views:

- **Explorer**: Browse and manage your project files
- **Search**: Search across your project
- **Source Control**: Manage Git repositories
- **Run and Debug**: Run and debug your code
- **Extensions**: Manage extensions
- **SPARC**: Access SPARC workflow tools
- **Roo Code**: Open the AI chat interface

### Editor Area

The central area of the window is the editor area, where you can:

- Edit code and text files
- View and interact with AI-generated content
- Work with multiple files in tabs or grid layouts

### Panels

The bottom panel area provides access to:

- **Terminal**: Run commands and scripts
- **Output**: View output from tasks and extensions
- **Problems**: See errors and warnings
- **Debug Console**: Interact with the debugger
- **Roo Code Chat**: Interact with AI models

### Status Bar

The status bar at the bottom of the window shows:

- Current SPARC phase
- Active AI model and mode
- Git branch and status
- Editor information (line/column, encoding, etc.)
- Notifications and status messages

## Customizing the UI/UX

SPARC IDE offers extensive customization options to tailor the environment to your preferences:

### Themes

SPARC IDE comes with two custom themes:

- **Dracula Pro**: A dark theme with vibrant colors
- **Material Theme**: A modern, material design-inspired theme

To change the theme:

1. Open Settings (File > Preferences > Settings or `Ctrl+,`)
2. Search for "workbench.colorTheme"
3. Select your preferred theme from the dropdown

Or use the keyboard shortcut `Ctrl+Alt+D` to switch between themes.

### Layouts

SPARC IDE provides different layout options:

- **AI-centric**: Optimized for AI-powered development
- **Minimal**: A simplified layout with fewer distractions
- **Default**: The standard VSCode layout

To change the layout:

1. Open Settings (File > Preferences > Settings or `Ctrl+,`)
2. Search for "ui-ux.customLayout"
3. Select your preferred layout from the dropdown

Or use the keyboard shortcut `Ctrl+Alt+L` to switch between layouts.

### Focus and Minimal Modes

SPARC IDE includes special modes to help you concentrate:

- **Focus Mode**: Highlights only the current file and dims everything else
- **Minimal Mode**: Hides UI elements for a distraction-free experience

To toggle these modes:

- `Ctrl+Alt+F`: Toggle Focus Mode
- `Ctrl+Alt+M`: Toggle Minimal Mode

### Font and Editor Settings

Customize the editor appearance and behavior:

1. Open Settings (File > Preferences > Settings or `Ctrl+,`)
2. Adjust editor settings, such as:
   - Font family and size: "editor.fontFamily" and "editor.fontSize"
   - Line height: "editor.lineHeight"
   - Tab size: "editor.tabSize"
   - Word wrap: "editor.wordWrap"
   - Auto-save: "files.autoSave"

### Custom Keybindings

Customize keyboard shortcuts to match your preferences:

1. Open Keyboard Shortcuts (File > Preferences > Keyboard Shortcuts or `Ctrl+K Ctrl+S`)
2. Search for the command you want to customize
3. Click on the pencil icon to edit the keybinding
4. Press the desired key combination
5. Press Enter to save the new keybinding

## SPARC Methodology Workflow

SPARC IDE implements the SPARC methodology with five phases:

### 1. Specification Phase

In this phase, you define the requirements, user stories, and acceptance criteria for your project.

**Key Activities:**
- Define project scope and objectives
- Create user stories and requirements
- Establish acceptance criteria
- Define constraints and assumptions

**Tools and Templates:**
- Requirements template
- User stories template
- Acceptance criteria template
- AI prompts for generating specifications

**To create a specification document:**
1. Click on the SPARC icon in the activity bar
2. Select "Specification" phase
3. Click on "Create Specification Document" or use one of the provided templates
4. Use the AI prompts to generate or refine your specifications

### 2. Pseudocode Phase

In this phase, you create high-level pseudocode and logic flow diagrams to outline your implementation approach.

**Key Activities:**
- Develop algorithm pseudocode
- Create logic flow diagrams
- Define data structures
- Outline key functions and components

**Tools and Templates:**
- Pseudocode template
- Flow diagram template
- Data structures template
- AI prompts for generating pseudocode

**To create pseudocode:**
1. Click on the SPARC icon in the activity bar
2. Select "Pseudocode" phase
3. Click on "Create Pseudocode Document" or use one of the provided templates
4. Use the AI prompts to generate or refine your pseudocode

### 3. Architecture Phase

In this phase, you design the system architecture, component interactions, and data models.

**Key Activities:**
- Design system architecture
- Define component interactions
- Create data models
- Establish interfaces and APIs

**Tools and Templates:**
- Architecture template
- Components template
- Interfaces template
- AI prompts for designing architecture

**To create architecture documentation:**
1. Click on the SPARC icon in the activity bar
2. Select "Architecture" phase
3. Click on "Create Architecture Document" or use one of the provided templates
4. Use the AI prompts to generate or refine your architecture design

### 4. Refinement Phase

In this phase, you implement, test, and refine your code.

**Key Activities:**
- Implement code based on pseudocode and architecture
- Write tests for your implementation
- Refactor and optimize code
- Fix bugs and address issues

**Tools and Templates:**
- Implementation template
- Tests template
- Refactoring template
- AI prompts for generating code, tests, and refactoring suggestions

**To work on refinement:**
1. Click on the SPARC icon in the activity bar
2. Select "Refinement" phase
3. Click on "Create Refinement Document" or use one of the provided templates
4. Use the AI prompts to generate code, tests, or refactoring suggestions

### 5. Completion Phase

In this phase, you finalize documentation, prepare for deployment, and establish maintenance procedures.

**Key Activities:**
- Complete documentation
- Prepare deployment plans
- Establish maintenance procedures
- Conduct final reviews and tests

**Tools and Templates:**
- Documentation template
- Deployment template
- Maintenance template
- AI prompts for generating documentation, deployment scripts, and maintenance procedures

**To complete your project:**
1. Click on the SPARC icon in the activity bar
2. Select "Completion" phase
3. Click on "Create Completion Document" or use one of the provided templates
4. Use the AI prompts to generate documentation, deployment scripts, or maintenance procedures

### Switching Between Phases

You can switch between SPARC phases using the SPARC panel in the activity bar or using keyboard shortcuts:
- `Ctrl+Alt+1`: Switch to Specification phase
- `Ctrl+Alt+2`: Switch to Pseudocode phase
- `Ctrl+Alt+3`: Switch to Architecture phase
- `Ctrl+Alt+4`: Switch to Refinement phase
- `Ctrl+Alt+5`: Switch to Completion phase

### Tracking Progress

SPARC IDE tracks your progress through the SPARC phases and provides a visual indicator of your current status. To view your progress:
1. Click on the SPARC icon in the activity bar
2. Click on "Show Progress" or press `Ctrl+Alt+P`

## Working with Roo Code

SPARC IDE integrates Roo Code to provide AI-powered coding assistance:

### Chat Interface

To open the Roo Code chat interface:
1. Press `Ctrl+Shift+A` or click on the Roo Code icon in the activity bar
2. Type your question or request in the chat input
3. Press Enter to send your message
4. Roo Code will respond with suggestions, explanations, or code snippets

### Code Generation

To generate code with Roo Code:
1. Open the Roo Code chat interface
2. Describe the code you want to generate
3. Roo Code will generate the code and provide an explanation
4. Click on "Insert Code" or press `Ctrl+Shift+I` to insert the generated code at the cursor position

**Example Prompts:**
- "Create a function that sorts an array of objects by a specific property"
- "Generate a React component for a login form with validation"
- "Write a SQL query to find the top 10 customers by order value"

### Code Explanation

To get an explanation of existing code:
1. Select the code you want to explain
2. Press `Ctrl+Shift+E` or right-click and select "Explain Code"
3. Roo Code will provide an explanation of the selected code

### Code Refactoring

To refactor code with Roo Code:
1. Select the code you want to refactor
2. Press `Ctrl+Shift+R` or right-click and select "Refactor Code"
3. Describe how you want to refactor the code
4. Roo Code will suggest refactored code
5. Click on "Apply Refactoring" to apply the suggested changes

**Example Prompts:**
- "Refactor this code to use async/await instead of promises"
- "Optimize this function for better performance"
- "Convert this class-based component to a functional component with hooks"

### Code Documentation

To generate documentation for your code:
1. Select the code you want to document
2. Press `Ctrl+Shift+D` or right-click and select "Document Code"
3. Roo Code will generate documentation comments for the selected code
4. Click on "Apply Documentation" to insert the documentation

### Test Generation

To generate tests for your code:
1. Select the code you want to test
2. Press `Ctrl+Shift+T` or right-click and select "Generate Tests"
3. Roo Code will generate test cases for the selected code
4. Click on "Create Test File" to create a new test file with the generated tests

## AI Models and Modes

SPARC IDE supports multiple AI models and modes to provide specialized assistance for different tasks:

### AI Models

SPARC IDE supports the following AI models:
- **OpenRouter**: Provides access to multiple models through a single API
- **Claude**: Anthropic's Claude model, known for its reasoning capabilities
- **GPT-4**: OpenAI's GPT-4 model, known for its general capabilities
- **Gemini**: Google's Gemini model, known for its code generation capabilities

To switch between AI models:
- `Ctrl+Shift+1`: Switch to OpenRouter
- `Ctrl+Shift+2`: Switch to Claude
- `Ctrl+Shift+3`: Switch to GPT-4
- `Ctrl+Shift+4`: Switch to Gemini

Or through the Roo Code settings:
1. Open the Roo Code chat interface
2. Click on the settings icon
3. Select "Change Model"
4. Choose the model you want to use

### AI Modes

SPARC IDE provides custom AI modes for specific tasks:
- **QA Engineer**: Detect edge cases and write tests
- **Architect**: Design scalable and maintainable systems
- **Code Reviewer**: Identify issues and suggest improvements
- **Documentation Writer**: Create clear and comprehensive documentation

To switch between AI modes:
- `Ctrl+Shift+Q`: Switch to QA Engineer mode
- `Ctrl+Shift+S`: Switch to Architect mode
- `Ctrl+Shift+C`: Switch to Code Reviewer mode
- `Ctrl+Shift+W`: Switch to Documentation Writer mode

Or through the Roo Code settings:
1. Open the Roo Code chat interface
2. Click on the settings icon
3. Select "Change Mode"
4. Choose the mode you want to use

### Creating Custom AI Modes

You can create your own custom AI modes:

1. Open Settings (File > Preferences > Settings or `Ctrl+,`)
2. Search for "roo-code.customModes"
3. Click "Edit in settings.json"
4. Add a new custom mode with the following structure:
   ```json
   "roo-code.customModes": {
     "Your Custom Mode Name": {
       "prompt": "Your custom system prompt here",
       "tools": ["readFile", "writeFile", "runCommand", "searchFiles"]
     }
   }
   ```
5. Save the settings file

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

### UI/UX Shortcuts

- `Ctrl+Alt+M`: Toggle minimal mode
- `Ctrl+Alt+F`: Toggle focus mode
- `Ctrl+Alt+L`: Switch layout
- `Ctrl+Alt+D`: Switch theme

## Advanced Features

### MCP Server Integration

SPARC IDE includes an MCP (Model Context Protocol) server that provides additional tools and resources for AI-powered development:

#### Starting the MCP Server

1. Open a terminal
2. Navigate to the SPARC IDE installation directory
3. Run `src/mcp/start-mcp-server.sh`

#### Available MCP Tools

The MCP server provides the following tools:

##### Code Analysis

Analyzes code to provide insights and suggestions:

1. Select the code you want to analyze
2. Right-click and select "Analyze Code with MCP"
3. The analysis results will be displayed in the MCP panel

##### Code Modification

Modifies code based on instructions:

1. Select the code you want to modify
2. Right-click and select "Modify Code with MCP"
3. Enter instructions for how to modify the code
4. The modified code will be displayed in the MCP panel

##### Code Search

Searches code for patterns or specific content:

1. Open the MCP panel
2. Click on "Search Code"
3. Enter a search pattern
4. Select the files to search in
5. The search results will be displayed in the MCP panel

### Multi-Agent Workflows

SPARC IDE supports multi-agent workflows, where multiple AI agents collaborate on a task:

1. Press `Ctrl+Shift+M` or open the Command Palette and search for "Execute Multi-Agent Workflow"
2. Enter a task description
3. Select the agents to include in the workflow
4. The workflow will execute, with each agent performing its part of the task
5. The results will be displayed in the Multi-Agent Workflow panel

### Extension Integration

SPARC IDE integrates with various extensions to enhance your development experience:

#### GitLens Integration

GitLens provides enhanced Git capabilities:

- Hover over a line to see when it was last changed and by whom
- View blame annotations in the editor
- Explore Git history and compare changes

#### Prettier Integration

Prettier provides code formatting:

- Format code on save
- Format selected code
- Configure formatting rules in settings

## Tips and Best Practices

### Effective AI Prompting

To get the best results from Roo Code:

1. **Be Specific**: Clearly describe what you want, including language, framework, and requirements
2. **Provide Context**: Include relevant information about your project and constraints
3. **Use Examples**: Provide examples of the desired output or similar code
4. **Iterate**: Refine your prompts based on the responses you receive

### SPARC Workflow Efficiency

To make the most of the SPARC methodology:

1. **Complete Each Phase**: Fully complete each phase before moving to the next
2. **Use Templates**: Leverage the provided templates to ensure consistency
3. **Track Artifacts**: Keep track of artifacts created in each phase
4. **Review Progress**: Regularly review your progress and adjust as needed

### Performance Optimization

To optimize SPARC IDE performance:

1. **Close Unused Editors**: Close editors you're not actively using
2. **Disable Unused Extensions**: Disable extensions you don't need
3. **Use Workspace Trust**: Trust only the workspaces you need
4. **Adjust Settings**: Configure settings for optimal performance:
   - Reduce auto-save frequency
   - Disable minimap
   - Limit the number of open editors

### Collaborative Development

To collaborate effectively with SPARC IDE:

1. **Share SPARC Artifacts**: Share artifacts created in each phase with your team
2. **Use Version Control**: Commit and push your changes regularly
3. **Document AI Interactions**: Document important AI interactions for future reference
4. **Standardize Prompts**: Create standardized prompts for common tasks