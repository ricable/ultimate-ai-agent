# SPARC IDE Installation Guide

This guide provides detailed instructions for building and installing SPARC IDE from source.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building from Source](#building-from-source)
3. [Installation](#installation)
4. [Post-Installation Setup](#post-installation-setup)
5. [Troubleshooting](#troubleshooting)

## Prerequisites

Before building SPARC IDE, ensure your system meets the following requirements:

### System Requirements

- **Operating System**:
  - Linux: Ubuntu 20.04+, Debian 11+, Fedora 34+
  - Windows: Windows 10+
  - macOS: macOS 11+
- **Hardware**:
  - CPU: 4+ cores recommended
  - RAM: 8GB+ recommended
  - Storage: 5GB+ free space for building

### Required Software

#### All Platforms

- **Git**: For cloning the repository
  - Installation: [https://git-scm.com/downloads](https://git-scm.com/downloads)
- **Node.js**: Version 16 or later
  - Installation: [https://nodejs.org/](https://nodejs.org/)
- **Yarn**: Version 1.22 or later
  - Installation: `npm install -g yarn`

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
  - Installation: [https://visualstudio.microsoft.com/visual-cpp-build-tools/](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  - Required components: "Desktop development with C++"

#### macOS-specific

- **Xcode Command Line Tools**: For compiling native modules
  - Installation: `xcode-select --install`

### Verifying Prerequisites

You can verify that you have the required software installed by running:

```bash
git --version
node --version
yarn --version
```

## Building from Source

### 1. Clone the Repository

```bash
git clone https://github.com/sparc-ide/sparc-ide.git
cd sparc-ide
```

### 2. Run the Setup Script

This script will set up the build environment, download dependencies, and configure the project:

```bash
# Make the script executable
chmod +x scripts/setup-sparc-ide.sh

# Run the setup script
./scripts/setup-sparc-ide.sh
```

The setup script performs the following tasks:
- Sets up the VSCodium build environment
- Applies SPARC IDE branding
- Downloads the Roo Code extension
- Configures the UI
- Sets up the MCP server
- Creates SPARC workflow directories

### 3. Build SPARC IDE

After the setup is complete, you can build SPARC IDE for your platform:

```bash
# Make the build script executable
chmod +x scripts/build-sparc-ide.sh

# Build for your platform (auto-detected)
./scripts/build-sparc-ide.sh

# Or specify a platform explicitly
./scripts/build-sparc-ide.sh --platform linux
./scripts/build-sparc-ide.sh --platform windows
./scripts/build-sparc-ide.sh --platform macos
```

The build process may take 30-60 minutes depending on your system's performance.

### 4. Locate Build Artifacts

After the build completes successfully, you'll find the installation packages in the `dist/` directory:

- **Linux**:
  - DEB package: `dist/sparc-ide_1.0.0_amd64.deb`
  - RPM package: `dist/sparc-ide-1.0.0.x86_64.rpm`
- **Windows**:
  - Installer: `dist/sparc-ide-setup-1.0.0.exe`
- **macOS**:
  - DMG package: `dist/sparc-ide-1.0.0.dmg`

## Installation

### Linux

#### Debian/Ubuntu (DEB package)

```bash
sudo dpkg -i dist/sparc-ide_1.0.0_amd64.deb
sudo apt-get install -f  # Install any missing dependencies
```

#### Fedora/RHEL (RPM package)

```bash
sudo rpm -i dist/sparc-ide-1.0.0.x86_64.rpm
```

### Windows

1. Run the installer: `dist/sparc-ide-setup-1.0.0.exe`
2. Follow the on-screen instructions to complete the installation

### macOS

1. Open the DMG file: `dist/sparc-ide-1.0.0.dmg`
2. Drag the SPARC IDE application to the Applications folder
3. Eject the DMG

## Post-Installation Setup

### 1. Start the MCP Server

The MCP server provides additional tools and resources for SPARC IDE. To start it:

```bash
# Navigate to the MCP server directory
cd src/mcp

# Make the start script executable
chmod +x start-mcp-server.sh

# Start the MCP server
./start-mcp-server.sh
```

You can verify that the MCP server is running by opening a web browser and navigating to [http://localhost:3001](http://localhost:3001).

### 2. Configure API Keys

To use AI features in SPARC IDE, you need to configure API keys:

1. Launch SPARC IDE
2. Open Settings (File > Preferences > Settings)
3. Search for "roo-code.apiKey"
4. Enter your OpenRouter API key
5. Optionally, configure keys for other AI providers:
   - "roo-code.anthropicApiKey" for Claude
   - "roo-code.openaiApiKey" for GPT-4
   - "roo-code.googleApiKey" for Gemini

### 3. Initialize SPARC Workflow

When you create or open a project, you can initialize the SPARC workflow:

1. Click on the SPARC icon in the activity bar
2. Click on "Initialize SPARC Workflow"
3. Select the project type
4. SPARC IDE will create the necessary directories and templates for your project

## Troubleshooting

### Build Issues

#### Missing Dependencies

If you encounter errors about missing dependencies during the build process:

**Linux**:
```bash
sudo apt update
sudo apt install build-essential libx11-dev libxkbfile-dev libsecret-1-dev fakeroot rpm
```

**Windows**:
Make sure you have installed Visual Studio Build Tools with the "Desktop development with C++" workload.

**macOS**:
```bash
xcode-select --install
```

#### Node.js Version Issues

If you encounter errors related to Node.js version:

1. Install Node Version Manager (nvm):
   - [https://github.com/nvm-sh/nvm](https://github.com/nvm-sh/nvm) (Linux/macOS)
   - [https://github.com/coreybutler/nvm-windows](https://github.com/coreybutler/nvm-windows) (Windows)

2. Install and use Node.js 16:
   ```bash
   nvm install 16
   nvm use 16
   ```

#### Build Script Permission Issues

If you encounter permission issues with the build scripts:

```bash
chmod +x scripts/*.sh
```

### Installation Issues

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

### MCP Server Issues

#### Port Already in Use

If port 3001 is already in use, you can change the port in the MCP server configuration:

1. Edit `src/mcp/.env`
2. Change `MCP_SERVER_PORT=3001` to another port, e.g., `MCP_SERVER_PORT=3002`
3. Restart the MCP server

#### Connection Refused

If you cannot connect to the MCP server:

1. Make sure the MCP server is running
2. Check the MCP server logs for errors
3. Verify that no firewall is blocking the connection

## Additional Resources

- [User Guide](user-guide.md): Comprehensive guide to using SPARC IDE
- [SPARC Methodology](sparc-methodology.md): Detailed explanation of the SPARC methodology
- [Roo Code Documentation](roo-code.md): Guide to using Roo Code in SPARC IDE
- [MCP Server Documentation](mcp-server.md): Detailed documentation for the MCP server

## Getting Help

If you encounter any issues not covered in this guide:

1. Check the [GitHub Issues](https://github.com/sparc-ide/sparc-ide/issues) for known problems and solutions
2. Join the [SPARC IDE Discord](https://discord.gg/sparc-ide) for community support
3. Create a new issue on [GitHub](https://github.com/sparc-ide/sparc-ide/issues/new) with detailed information about your problem