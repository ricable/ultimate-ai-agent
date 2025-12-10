# SPARC IDE Installation and Setup Guide

This comprehensive guide provides detailed instructions for installing, configuring, and setting up SPARC IDE on different platforms.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Platform-Specific Installation](#platform-specific-installation)
4. [Post-Installation Setup](#post-installation-setup)
5. [Roo Code Integration](#roo-code-integration)
6. [AI Model Configuration](#ai-model-configuration)
7. [SPARC Workflow Setup](#sparc-workflow-setup)
8. [Building from Source](#building-from-source)
9. [Troubleshooting](#troubleshooting)

## System Requirements

Before installing SPARC IDE, ensure your system meets the following requirements:

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 2 cores | 4+ cores |
| RAM | 4GB | 8GB+ |
| Storage | 1GB free space | 5GB+ free space |
| Display | 1280x720 | 1920x1080 or higher |
| Internet | Required for AI features | Broadband connection |

### Operating System Requirements

#### Linux
- Ubuntu 20.04 LTS or newer
- Debian 11 or newer
- Fedora 34 or newer
- Other distributions with equivalent libraries

#### Windows
- Windows 10 (64-bit) version 1909 or newer
- Windows 11 (64-bit)

#### macOS
- macOS 11 (Big Sur) or newer
- Apple Silicon or Intel processor

## Installation Methods

SPARC IDE can be installed using pre-built packages or built from source:

1. **Pre-built Packages**: The simplest method, providing ready-to-use installers for each platform
2. **Build from Source**: Provides the most control and latest features, but requires development tools

## Platform-Specific Installation

### Linux Installation

#### Debian/Ubuntu (DEB package)

1. Download the DEB package from the [releases page](https://github.com/sparc-ide/sparc-ide/releases)
2. Install using the package manager:
   ```bash
   sudo dpkg -i sparc-ide_1.0.0_amd64.deb
   sudo apt-get install -f  # Install any missing dependencies
   ```
3. Launch SPARC IDE from the applications menu or run `sparc-ide` in the terminal

#### Fedora/RHEL (RPM package)

1. Download the RPM package from the [releases page](https://github.com/sparc-ide/sparc-ide/releases)
2. Install using the package manager:
   ```bash
   sudo rpm -i sparc-ide-1.0.0.x86_64.rpm
   ```
3. Launch SPARC IDE from the applications menu or run `sparc-ide` in the terminal

#### AppImage (Universal Linux Package)

1. Download the AppImage from the [releases page](https://github.com/sparc-ide/sparc-ide/releases)
2. Make the AppImage executable:
   ```bash
   chmod +x SPARC_IDE-1.0.0.AppImage
   ```
3. Run the AppImage:
   ```bash
   ./SPARC_IDE-1.0.0.AppImage
   ```

### Windows Installation

1. Download the Windows installer (`sparc-ide-setup-1.0.0.exe`) from the [releases page](https://github.com/sparc-ide/sparc-ide/releases)
2. Run the installer and follow the on-screen instructions
3. Choose the installation location (default is `C:\Program Files\SPARC IDE`)
4. Select additional options:
   - Create desktop shortcut
   - Add to PATH
   - Register as default editor for supported file types
5. Complete the installation
6. Launch SPARC IDE from the Start menu or desktop shortcut

### macOS Installation

1. Download the macOS package (`sparc-ide-1.0.0.dmg`) from the [releases page](https://github.com/sparc-ide/sparc-ide/releases)
2. Open the DMG file
3. Drag the SPARC IDE application to the Applications folder
4. Eject the DMG
5. Launch SPARC IDE from the Applications folder or Launchpad

#### First Launch on macOS

When launching SPARC IDE for the first time on macOS, you may see a security warning:

1. Right-click (or Control-click) the SPARC IDE app in the Applications folder
2. Select "Open" from the context menu
3. Click "Open" in the dialog that appears
4. The app will now be saved as an exception to your security settings

## Post-Installation Setup

After installing SPARC IDE, follow these steps to complete the setup:

### First-Time Setup Wizard

When you first launch SPARC IDE, a setup wizard will guide you through the initial configuration:

1. **Welcome Screen**: Introduction to SPARC IDE
2. **Theme Selection**: Choose between light and dark themes
3. **Layout Selection**: Select your preferred layout
4. **Extension Setup**: Configure essential extensions
5. **AI Configuration**: Set up AI model access (optional)

### Extension Marketplace

SPARC IDE includes a curated set of extensions, but you can install additional extensions from the marketplace:

1. Click on the Extensions icon in the activity bar (or press `Ctrl+Shift+X`)
2. Browse or search for extensions
3. Click "Install" to add an extension

### Settings Configuration

Customize SPARC IDE settings to match your preferences:

1. Open Settings (File > Preferences > Settings or `Ctrl+,`)
2. Adjust editor settings, such as:
   - Font family and size
   - Tab size and indentation
   - Word wrap
   - Auto-save
3. Configure UI settings:
   - Theme
   - Icon theme
   - Layout
4. Set up language-specific settings

## Roo Code Integration

SPARC IDE comes with Roo Code pre-installed, but you need to configure it to use AI features:

### API Key Configuration

1. Obtain API keys from one or more of the following providers:
   - [OpenRouter](https://openrouter.ai/)
   - [Anthropic (Claude)](https://www.anthropic.com/)
   - [OpenAI (GPT-4)](https://openai.com/)
   - [Google (Gemini)](https://ai.google.dev/)

2. Open Settings (File > Preferences > Settings or `Ctrl+,`)

3. Search for "roo-code.apiKey" and enter your OpenRouter API key

4. Optionally, configure keys for other AI providers:
   - "roo-code.anthropicApiKey" for Claude
   - "roo-code.openaiApiKey" for GPT-4
   - "roo-code.googleApiKey" for Gemini

### Verifying Roo Code Integration

To verify that Roo Code is properly integrated:

1. Open the Command Palette (`Ctrl+Shift+P`)
2. Type "Roo Code: Open Chat" and press Enter
3. The Roo Code chat interface should open
4. Type a simple question like "What is SPARC IDE?" and press Enter
5. You should receive a response from the AI

## AI Model Configuration

SPARC IDE supports multiple AI models, which can be configured and customized:

### Selecting the Default AI Model

1. Open Settings (File > Preferences > Settings or `Ctrl+,`)
2. Search for "roo-code.defaultModel"
3. Select your preferred default model from the dropdown

### Switching Between Models

You can switch between AI models using keyboard shortcuts:
- `Ctrl+Shift+1`: Switch to OpenRouter
- `Ctrl+Shift+2`: Switch to Claude
- `Ctrl+Shift+3`: Switch to GPT-4
- `Ctrl+Shift+4`: Switch to Gemini

Or through the Roo Code settings:
1. Open the Roo Code chat interface
2. Click on the settings icon
3. Select "Change Model"
4. Choose the model you want to use

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

## SPARC Workflow Setup

SPARC IDE implements the SPARC methodology with five phases. Here's how to set up and use the SPARC workflow:

### Initializing SPARC Workflow for a Project

1. Open a project folder in SPARC IDE
2. Click on the SPARC icon in the activity bar
3. Click on "Initialize SPARC Workflow"
4. Select the project type
5. SPARC IDE will create the necessary directories and templates for your project

### SPARC Directory Structure

The SPARC workflow creates the following directory structure in your project:

```
project/
├── .sparc/
│   ├── templates/
│   │   ├── specification/
│   │   ├── pseudocode/
│   │   ├── architecture/
│   │   ├── refinement/
│   │   └── completion/
│   └── artifacts/
│       ├── specification/
│       ├── pseudocode/
│       ├── architecture/
│       ├── refinement/
│       └── completion/
└── [your project files]
```

### Using SPARC Templates

Each SPARC phase includes templates to help you create the necessary artifacts:

1. Click on the SPARC icon in the activity bar
2. Select the current phase
3. Click on "Create from Template"
4. Select a template
5. Choose a location to save the artifact

### Tracking SPARC Progress

SPARC IDE tracks your progress through the SPARC phases:

1. Click on the SPARC icon in the activity bar
2. Click on "Show Progress" or press `Ctrl+Alt+P`
3. The progress view shows your current status in each phase

## Building from Source

If you prefer to build SPARC IDE from source, follow these instructions:

### Prerequisites

Before building SPARC IDE, ensure you have the following tools installed:

#### All Platforms
- **Git**: For cloning the repository
- **Node.js**: Version 18 or later
- **Yarn**: Version 1.22 or later

#### Linux-specific
Install the following packages:

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install build-essential libx11-dev libxkbfile-dev libsecret-1-dev fakeroot rpm
```

**Fedora**:
```bash
sudo dnf install make gcc gcc-c++ libX11-devel libxkbfile-devel libsecret-devel rpm-build
```

#### Windows-specific
- **Visual Studio Build Tools**: For compiling native modules
  - Install from [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Required components: "Desktop development with C++"

#### macOS-specific
- **Xcode Command Line Tools**: For compiling native modules
  ```bash
  xcode-select --install
  ```

### Build Process

1. Clone the repository:
   ```bash
   git clone https://github.com/sparc-ide/sparc-ide.git
   cd sparc-ide
   ```

2. Set up the build environment:
   ```bash
   chmod +x scripts/setup-sparc-ide.sh
   ./scripts/setup-sparc-ide.sh
   ```

3. Build SPARC IDE:
   ```bash
   chmod +x scripts/build-sparc-ide.sh
   ./scripts/build-sparc-ide.sh
   ```

4. The build artifacts will be available in the `dist/` directory.

### Running the Development Version

To run the development version of SPARC IDE:

```bash
./dist/sparc-ide
```

### Creating Installation Packages

To create installation packages for distribution:

```bash
chmod +x scripts/package-sparc-ide.sh
./scripts/package-sparc-ide.sh
```

The packages will be available in the `package/` directory, organized by platform.

## Troubleshooting

### Common Installation Issues

#### Linux: Dependency Issues

If you encounter dependency issues when installing the DEB package:

```bash
sudo apt-get install -f
```

#### Windows: Antivirus Blocking Installation

Some antivirus software may block the installation. Try temporarily disabling your antivirus software during installation.

#### macOS: "App Cannot Be Opened"

If you see a message that the app cannot be opened because it is from an unidentified developer:

1. Right-click (or Control-click) the app icon
2. Select "Open" from the context menu
3. Click "Open" in the dialog that appears

### Roo Code Integration Issues

#### API Key Not Working

If your API key is not working:

1. Verify that you've entered the correct API key
2. Check that your API key has sufficient credits
3. Ensure you have an active internet connection
4. Check the API provider's status page for service outages

#### Chat Interface Not Responding

If the Roo Code chat interface is not responding:

1. Restart SPARC IDE
2. Check your internet connection
3. Verify that your API key is valid
4. Check the console for error messages (Help > Toggle Developer Tools)

### Performance Issues

If SPARC IDE is running slowly:

1. Close unused editors and terminals
2. Disable unused extensions
3. Adjust performance settings in Settings:
   - Reduce the number of open editors
   - Disable minimap
   - Reduce auto-save frequency

### Getting Help

If you encounter issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/sparc-ide/sparc-ide/issues) for known problems and solutions
2. Join the [SPARC IDE Discord](https://discord.gg/sparc-ide) for community support
3. Create a new issue on [GitHub](https://github.com/sparc-ide/sparc-ide/issues/new) with detailed information about your problem