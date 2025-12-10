# SPARC IDE

SPARC IDE is a customizable, open-source distribution of VSCode built specifically for agentic software development. It integrates Roo Code to enable prompt-driven development, autonomous agent workflows, and AI-native collaboration.

![SPARC IDE Logo](branding/icons/README.md)

## Features

### VSCodium Base
- **Open-Source Foundation**: Built on VSCodium, a community-driven, telemetry-free fork of VS Code
- **Cross-Platform Support**: Runs on Linux, macOS, and Windows with native performance
- **Custom Branding**: Unique SPARC IDE visual identity and splash screens
- **MIT-Licensed**: Fully open-source under the MIT license

### Roo Code Integration
- **Pre-installed Roo Code Extension**: Ready to use out of the box
- **Multi-Model AI Support**: Connect to OpenRouter, Claude, GPT-4, and Gemini
- **Custom AI Modes**: Specialized AI personas for different development tasks
- **Secure API Key Management**: Encrypted storage for AI service credentials
- **Context-Aware Assistance**: AI that understands your project structure and code

### SPARC Methodology Support
- **Structured Workflow**: Guided development through all SPARC phases
- **Phase-Specific Templates**: Templates for each phase of the SPARC methodology
- **Artifact Tracking**: Track progress and artifacts across phases
- **AI-Assisted Phase Transitions**: AI helps you move between phases
- **Progress Visualization**: Visual indicators of project progress

### AI-Powered Development
- **Code Generation**: Generate code based on natural language descriptions
- **Code Explanation**: Get explanations of complex code
- **Code Refactoring**: AI-assisted code improvements
- **Documentation Generation**: Automatically generate documentation
- **Test Generation**: Create test cases based on implementation
- **Multi-Agent Workflows**: Coordinate multiple AI agents for complex tasks

### Enhanced UI/UX
- **AI-Centric Layout**: Optimized interface for AI-assisted development
- **Custom Themes**: Dracula Pro and Material Theme included
- **Minimal Mode**: Distraction-free coding environment
- **Focus Mode**: Highlight only the current file and context
- **Custom Keybindings**: Optimized shortcuts for AI interactions

### MCP Server Integration
- **Model Context Protocol**: Enhanced context management for AI models
- **Additional Tools**: Extended capabilities through MCP server
- **Secure Communication**: HTTPS and authentication for API endpoints
- **Resource Management**: Efficient handling of AI resources

### Security Features
- **Cryptographic Verification**: Verify extensions before installation
- **Secure Credential Management**: Environment variable support for sensitive data
- **Content Security Policy**: Restrict extension capabilities
- **Dependency Pinning**: Prevent unexpected updates
- **Regular Security Audits**: Continuous security improvements

## System Requirements

### Operating System
- **Linux**: Ubuntu 20.04+, Debian 11+, Fedora 34+
- **Windows**: Windows 10+
- **macOS**: macOS 11+

### Hardware
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 5GB+ free space for building, 1GB+ for installation

## Quick Start

### Installation

#### Linux
```bash
# Debian/Ubuntu
sudo dpkg -i sparc-ide_1.0.0_amd64.deb
sudo apt-get install -f  # Install any missing dependencies

# Fedora/RHEL
sudo rpm -i sparc-ide-1.0.0.x86_64.rpm
```

#### Windows
1. Run the installer: `sparc-ide-setup-1.0.0.exe`
2. Follow the on-screen instructions

#### macOS
1. Open the DMG file: `sparc-ide-1.0.0.dmg`
2. Drag SPARC IDE to the Applications folder

### API Key Configuration

1. Open SPARC IDE
2. Go to Settings (File > Preferences > Settings)
3. Search for "roo-code.apiKey"
4. Enter your OpenRouter API key
5. Optionally, configure keys for other AI providers

### Starting a SPARC Project

1. Click on the SPARC icon in the activity bar
2. Click "Initialize SPARC Workflow"
3. Select a project type
4. Begin with the Specification phase

## SPARC Methodology

SPARC IDE implements the SPARC methodology with five phases:

1. **Specification**: Define detailed requirements and acceptance criteria
2. **Pseudocode**: Create implementation pseudocode and logic flow
3. **Architecture**: Design system architecture and component interactions
4. **Refinement**: Implement iterative improvements and testing
5. **Completion**: Finalize documentation, deployment, and maintenance

Each phase has dedicated templates, AI prompts, and tools to help you work efficiently.

## Key Keyboard Shortcuts

### Roo Code Integration
- `Ctrl+Shift+A`: Open AI chat
- `Ctrl+Shift+I`: Insert AI-generated code
- `Ctrl+Shift+E`: Explain selected code
- `Ctrl+Shift+R`: Refactor selected code
- `Ctrl+Shift+D`: Document selected code
- `Ctrl+Shift+T`: Generate tests for selected code

### SPARC Workflow
- `Ctrl+Alt+1-5`: Switch between SPARC phases
- `Ctrl+Alt+T`: Create template for current SPARC phase
- `Ctrl+Alt+A`: Create artifact for current SPARC phase
- `Ctrl+Alt+P`: Show SPARC progress

### AI Models and Modes
- `Ctrl+Shift+1-4`: Switch between AI models
- `Ctrl+Shift+Q/S/C/W`: Switch between AI modes

### UI/UX
- `Ctrl+Alt+M`: Toggle minimal mode
- `Ctrl+Alt+F`: Toggle focus mode
- `Ctrl+Alt+L`: Switch layout
- `Ctrl+Alt+D`: Switch theme

## Building from Source

### Prerequisites
- Node.js 18+
- Yarn 1.22+
- Git
- Platform-specific build dependencies

### Build Instructions

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

For detailed build instructions, see [Installation Guide](docs/installation-guide.md).

## Documentation

- [User Guide](docs/user-guide.md): Comprehensive guide to using SPARC IDE
- [Installation Guide](docs/installation-guide.md): Detailed installation instructions
- [Contributing Guide](docs/CONTRIBUTING.md): How to contribute to SPARC IDE
- [Security Enhancements](docs/security-enhancements.md): Security features and improvements
- [Packaging Guide](docs/packaging-guide.md): How to package SPARC IDE for distribution

## Contributing

We welcome contributions to SPARC IDE! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for details on how to contribute.

## License

SPARC IDE is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [VSCodium](https://github.com/VSCodium/vscodium) for providing the base for SPARC IDE
- [Roo Code](https://github.com/RooVeterinaryInc/roo-cline) for the AI integration
- All the open-source projects that make SPARC IDE possible
