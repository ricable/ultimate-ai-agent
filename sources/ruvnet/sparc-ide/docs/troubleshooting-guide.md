# SPARC IDE Troubleshooting Guide

## Introduction

This guide provides solutions to common issues you might encounter while using SPARC IDE. If you're experiencing a problem not covered in this guide, please check our [GitHub Issues](https://github.com/sparc-ide/sparc-ide/issues) or create a new issue.

## Installation Issues

### Installation Fails on Linux

**Symptoms:**
- Installation package fails to install
- Error messages about missing dependencies

**Solutions:**
1. Ensure you have the required dependencies installed:
   ```bash
   # For Debian/Ubuntu
   sudo apt-get install libgtk-3-0 libx11-xcb1 libasound2 libxss1 libgbm1 libnss3
   
   # For Fedora
   sudo dnf install gtk3 libX11-xcb alsa-lib libxss mesa-libgbm nss
   ```

2. Check if you have sufficient permissions:
   ```bash
   # For DEB package
   sudo dpkg -i sparc-ide_1.0.0_amd64.deb
   
   # For RPM package
   sudo rpm -i sparc-ide-1.0.0-1.x86_64.rpm
   ```

### Installation Fails on Windows

**Symptoms:**
- Installer fails to complete
- Error messages about Windows features

**Solutions:**
1. Ensure you're running Windows 10 or later
2. Run the installer as administrator
3. Temporarily disable antivirus software during installation
4. Install the latest Visual C++ Redistributable packages

### Installation Fails on macOS

**Symptoms:**
- DMG file won't mount
- Application won't start after installation

**Solutions:**
1. Ensure you're running macOS 10.15 (Catalina) or later
2. Check Gatekeeper settings:
   ```bash
   sudo spctl --master-disable  # Temporarily disable Gatekeeper
   # After installation
   sudo spctl --master-enable   # Re-enable Gatekeeper
   ```
3. Allow the application in System Preferences > Security & Privacy

## Startup Issues

### SPARC IDE Won't Start

**Symptoms:**
- Application crashes immediately after launch
- Application hangs on splash screen

**Solutions:**
1. Reset user settings:
   ```bash
   # On Linux
   rm -rf ~/.config/sparc-ide/User
   
   # On Windows
   rmdir /s /q %APPDATA%\sparc-ide\User
   
   # On macOS
   rm -rf ~/Library/Application\ Support/sparc-ide/User
   ```

2. Start with verbose logging:
   ```bash
   # On Linux/macOS
   sparc-ide --verbose
   
   # On Windows
   sparc-ide.exe --verbose
   ```

3. Check system resources (memory, disk space)

### Slow Startup

**Symptoms:**
- Application takes a long time to start
- High CPU/memory usage during startup

**Solutions:**
1. Disable unnecessary extensions
2. Clear the extension cache:
   ```bash
   # On Linux
   rm -rf ~/.config/sparc-ide/CachedExtensions
   
   # On Windows
   rmdir /s /q %APPDATA%\sparc-ide\CachedExtensions
   
   # On macOS
   rm -rf ~/Library/Application\ Support/sparc-ide/CachedExtensions
   ```

3. Increase performance settings in `settings.json`:
   ```json
   "performance.lazyLoading": true,
   "performance.caching": true,
   "performance.resourceManagement": true,
   "performance.memoryLimit": 4096
   ```

## Roo Code Integration Issues

### Roo Code Extension Not Working

**Symptoms:**
- Roo Code commands not available
- Error messages when using Roo Code features

**Solutions:**
1. Verify the extension is installed:
   ```bash
   # Check installed extensions
   sparc-ide --list-extensions | grep roo-code
   ```

2. Reinstall the extension:
   ```bash
   # Uninstall
   sparc-ide --uninstall-extension sparc-ide.roo-code
   
   # Reinstall
   ./scripts/download-roo-code.sh
   ```

3. Check API configuration in `settings.json`

### AI Model Connection Issues

**Symptoms:**
- Cannot connect to AI models
- Timeout errors when using AI features

**Solutions:**
1. Check internet connection
2. Verify API keys are correctly configured
3. Try switching to a different AI model provider
4. Check firewall settings for outbound connections

## SPARC Workflow Issues

### Cannot Switch Workflow Phases

**Symptoms:**
- Phase switching commands don't work
- Error messages when trying to switch phases

**Solutions:**
1. Ensure SPARC workflow is enabled in settings:
   ```json
   "sparc-workflow.enabled": true
   ```

2. Check keybindings for phase switching:
   ```json
   {
     "key": "ctrl+alt+1",
     "command": "sparc-workflow.switchPhase.specification",
     "when": "sparc-workflow.enabled"
   }
   ```

3. Verify workspace configuration

### Template Creation Issues

**Symptoms:**
- Cannot create templates
- Templates are created with errors

**Solutions:**
1. Check template directory configuration:
   ```json
   "sparc-workflow.templateDirectory": "${workspaceFolder}/.sparc/templates"
   ```

2. Ensure template directory exists and is writable
3. Check template format and structure

## UI/UX Issues

### Theme Not Applied Correctly

**Symptoms:**
- Theme colors are inconsistent
- Some UI elements use default theme

**Solutions:**
1. Verify theme is installed:
   ```bash
   sparc-ide --list-extensions | grep theme
   ```

2. Reset theme settings:
   ```json
   "workbench.colorTheme": "Dracula Pro",
   "ui-ux.customTheme": "dracula-pro"
   ```

3. Reload window: `Ctrl+R` or `Cmd+R`

### Layout Issues

**Symptoms:**
- UI elements misaligned
- Panels not displaying correctly

**Solutions:**
1. Reset layout:
   ```json
   "ui-ux.customLayout": "ai-centric"
   ```

2. Try different layout options:
   ```json
   "ui-ux.customLayout": "default"
   ```

3. Check display resolution and scaling settings

## Performance Issues

### High CPU Usage

**Symptoms:**
- High CPU usage during idle
- Fan noise and heat

**Solutions:**
1. Disable unnecessary extensions
2. Check for runaway processes in Task Manager/Activity Monitor
3. Adjust performance settings:
   ```json
   "performance.lazyLoading": true,
   "performance.caching": true
   ```

### Memory Leaks

**Symptoms:**
- Increasing memory usage over time
- Application becomes slower over time

**Solutions:**
1. Restart SPARC IDE periodically
2. Increase memory limit:
   ```json
   "performance.memoryLimit": 8192
   ```

3. Update to the latest version

## Security Issues

### Extension Verification Failures

**Symptoms:**
- Extensions fail to load due to verification errors
- Security warnings about unsigned extensions

**Solutions:**
1. Only use verified extensions from trusted sources
2. Check extension signature verification settings
3. Update to the latest version of SPARC IDE

### Permission Issues

**Symptoms:**
- Cannot access certain files or directories
- Permission denied errors

**Solutions:**
1. Check file permissions
2. Run SPARC IDE with appropriate privileges
3. Configure workspace trust settings

## Logging and Diagnostics

### Enabling Verbose Logging

To help diagnose issues, you can enable verbose logging:

```bash
# On Linux/macOS
sparc-ide --verbose > sparc-ide.log 2>&1

# On Windows
sparc-ide.exe --verbose > sparc-ide.log 2>&1
```

### Collecting Diagnostic Information

To collect diagnostic information for support:

1. Open the Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`)
2. Run the command "Developer: Open Process Explorer"
3. Take screenshots of the process information
4. Collect logs from:
   ```
   # On Linux
   ~/.config/sparc-ide/logs/
   
   # On Windows
   %APPDATA%\sparc-ide\logs\
   
   # On macOS
   ~/Library/Application\ Support/sparc-ide/logs/
   ```

## Getting Help

If you're still experiencing issues after trying the solutions in this guide:

1. Check our [GitHub Issues](https://github.com/sparc-ide/sparc-ide/issues) for similar problems
2. Join our [Community Discord](https://discord.gg/sparc-ide) for real-time help
3. Create a new issue with detailed information about your problem
4. Contact our support team at support@sparc-ide.example.com

## Contributing to This Guide

If you've found a solution to a problem not covered in this guide, please consider contributing:

1. Fork the repository
2. Add your solution to this guide
3. Submit a pull request

Your contributions help make SPARC IDE better for everyone!