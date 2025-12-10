# SPARC IDE Troubleshooting Guide

This comprehensive guide provides solutions for common issues you might encounter when using SPARC IDE, including installation problems, Roo Code integration issues, and performance concerns.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Startup Problems](#startup-problems)
3. [Roo Code Integration Issues](#roo-code-integration-issues)
4. [AI Model Connection Problems](#ai-model-connection-problems)
5. [SPARC Workflow Issues](#sparc-workflow-issues)
6. [Performance Problems](#performance-problems)
7. [UI/UX Issues](#uiux-issues)
8. [Extension Problems](#extension-problems)
9. [Crash Recovery](#crash-recovery)
10. [Diagnostic Tools](#diagnostic-tools)

## Installation Issues

### Linux Installation Problems

#### DEB Package Installation Fails

**Symptoms:**
- Error messages when installing the DEB package
- Missing dependencies

**Solutions:**
1. Install missing dependencies:
   ```bash
   sudo apt-get install -f
   ```

2. Check for conflicting packages:
   ```bash
   sudo apt-get check
   ```

3. Try manual installation:
   ```bash
   sudo dpkg -i sparc-ide_1.0.0_amd64.deb
   sudo apt-get install -f
   ```

#### RPM Package Installation Fails

**Symptoms:**
- Error messages when installing the RPM package
- Dependency conflicts

**Solutions:**
1. Install with dependency resolution:
   ```bash
   sudo dnf install sparc-ide-1.0.0.x86_64.rpm
   ```

2. Check for conflicts:
   ```bash
   sudo rpm -V sparc-ide
   ```

#### AppImage Won't Run

**Symptoms:**
- AppImage fails to launch
- Permission errors

**Solutions:**
1. Make the AppImage executable:
   ```bash
   chmod +x SPARC_IDE-1.0.0.AppImage
   ```

2. Install required libraries:
   ```bash
   sudo apt-get install libfuse2
   ```

3. Run with debugging:
   ```bash
   ./SPARC_IDE-1.0.0.AppImage --appimage-extract-and-run
   ```

### Windows Installation Problems

#### Installer Fails to Run

**Symptoms:**
- Error message when running the installer
- Installation process doesn't start

**Solutions:**
1. Run as administrator:
   - Right-click the installer
   - Select "Run as administrator"

2. Check Windows Defender:
   - Temporarily disable Windows Defender
   - Add an exception for the installer

3. Verify the installer integrity:
   - Check the SHA-256 hash against the published value
   - Download the installer again if necessary

#### Installation Hangs

**Symptoms:**
- Installation process stops responding
- Progress bar freezes

**Solutions:**
1. End the installer process and restart:
   - Open Task Manager (Ctrl+Shift+Esc)
   - End the installer process
   - Run the installer again

2. Use clean boot:
   - Restart in clean boot mode
   - Run the installer
   - Restart normally after installation

#### Missing DLLs After Installation

**Symptoms:**
- Error messages about missing DLLs
- Application fails to start

**Solutions:**
1. Install Visual C++ Redistributable:
   - Download and install the latest Visual C++ Redistributable
   - Restart your computer

2. Repair the installation:
   - Open Control Panel > Programs > Programs and Features
   - Select SPARC IDE
   - Click "Repair"

### macOS Installation Problems

#### "App Cannot Be Opened" Error

**Symptoms:**
- Message that the app cannot be opened because it is from an unidentified developer
- App fails to launch

**Solutions:**
1. Open using context menu:
   - Right-click (or Control-click) the app icon
   - Select "Open" from the context menu
   - Click "Open" in the dialog that appears

2. Adjust security settings:
   - Open System Preferences > Security & Privacy
   - Click the lock to make changes
   - Select "Allow apps downloaded from: App Store and identified developers"
   - Click "Open Anyway" for SPARC IDE

#### DMG Won't Mount

**Symptoms:**
- DMG file doesn't mount when double-clicked
- Error message when trying to open the DMG

**Solutions:**
1. Use Disk Utility:
   - Open Disk Utility
   - Select File > Open Disk Image
   - Navigate to the DMG file and open it

2. Verify the DMG integrity:
   - Check the SHA-256 hash against the published value
   - Download the DMG again if necessary

## Startup Problems

### Application Won't Launch

**Symptoms:**
- SPARC IDE doesn't start when clicked
- No error message appears

**Solutions:**
1. Check system resources:
   - Ensure you have enough free memory
   - Close other resource-intensive applications

2. Launch from terminal to see error messages:
   - **Linux/macOS**: Open a terminal and run `sparc-ide --verbose`
   - **Windows**: Open Command Prompt and run `"C:\Program Files\SPARC IDE\sparc-ide.exe" --verbose`

3. Reset user data:
   - **Linux**: Remove `~/.config/sparc-ide`
   - **macOS**: Remove `~/Library/Application Support/sparc-ide`
   - **Windows**: Remove `%APPDATA%\sparc-ide`

### Crash on Startup

**Symptoms:**
- SPARC IDE starts to launch but crashes immediately
- Error dialog appears

**Solutions:**
1. Start with extensions disabled:
   ```bash
   sparc-ide --disable-extensions
   ```

2. Reset workspace storage:
   - **Linux**: Remove `~/.config/sparc-ide/workspaceStorage`
   - **macOS**: Remove `~/Library/Application Support/sparc-ide/workspaceStorage`
   - **Windows**: Remove `%APPDATA%\sparc-ide\workspaceStorage`

3. Check for corrupted settings:
   - **Linux**: Check `~/.config/sparc-ide/User/settings.json`
   - **macOS**: Check `~/Library/Application Support/sparc-ide/User/settings.json`
   - **Windows**: Check `%APPDATA%\sparc-ide\User\settings.json`

### Slow Startup

**Symptoms:**
- SPARC IDE takes a long time to start
- High CPU or disk usage during startup

**Solutions:**
1. Disable startup extensions:
   - Open Settings (File > Preferences > Settings)
   - Search for "extensions.autoStartPatterns"
   - Remove unnecessary extensions from the list

2. Clear the extension cache:
   - **Linux**: Remove `~/.config/sparc-ide/CachedExtensionVSIXs`
   - **macOS**: Remove `~/Library/Application Support/sparc-ide/CachedExtensionVSIXs`
   - **Windows**: Remove `%APPDATA%\sparc-ide\CachedExtensionVSIXs`

3. Increase startup performance:
   - Open Settings (File > Preferences > Settings)
   - Set "window.restoreWindows" to "none"
   - Set "workbench.startupEditor" to "none"

## Roo Code Integration Issues

### Roo Code Extension Not Loading

**Symptoms:**
- Roo Code icon missing from activity bar
- Roo Code commands not available

**Solutions:**
1. Check if the extension is installed:
   - Open Extensions view (Ctrl+Shift+X)
   - Search for "Roo Code"
   - Install if not present

2. Reinstall the extension:
   ```bash
   chmod +x scripts/download-roo-code.sh
   ./scripts/download-roo-code.sh
   ```

3. Check extension logs:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Show Logs"
   - Look for errors related to Roo Code

### API Key Configuration Issues

**Symptoms:**
- Error messages about missing or invalid API keys
- AI features not working

**Solutions:**
1. Verify API key configuration:
   - Open Settings (File > Preferences > Settings)
   - Search for "roo-code.apiKey"
   - Ensure your API key is correctly entered

2. Check API key validity:
   - Verify that your API key is active and has sufficient credits
   - Try using the API key in a different tool to confirm it works

3. Reset API key storage:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Roo Code: Reset API Key Storage"
   - Enter your API key again

### Chat Interface Not Responding

**Symptoms:**
- Chat interface doesn't respond to input
- Loading indicator spins indefinitely

**Solutions:**
1. Restart the chat interface:
   - Close the chat panel
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Roo Code: Open Chat"

2. Check network connectivity:
   - Verify that you have an active internet connection
   - Check if you can access the AI provider's website

3. Reset the chat session:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Roo Code: Reset Chat Session"

## AI Model Connection Problems

### Cannot Connect to AI Models

**Symptoms:**
- Error messages about connection failures
- Timeout errors when using AI features

**Solutions:**
1. Check network connectivity:
   - Verify that you have an active internet connection
   - Check if you can access the AI provider's website

2. Verify API endpoint configuration:
   - Open Settings (File > Preferences > Settings)
   - Search for "roo-code.apiEndpoint"
   - Ensure the endpoint URL is correct

3. Check for service outages:
   - Visit the AI provider's status page
   - Check for announced outages or maintenance

### Model Switching Fails

**Symptoms:**
- Error when trying to switch AI models
- Model doesn't change after selection

**Solutions:**
1. Verify API key for the selected model:
   - Open Settings (File > Preferences > Settings)
   - Check that you have configured an API key for the selected model

2. Restart the AI integration:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Roo Code: Restart AI Integration"

3. Check model availability:
   - Verify that the selected model is available in your region
   - Check if the model requires additional permissions

### Rate Limit Errors

**Symptoms:**
- Error messages about rate limits
- AI features stop working temporarily

**Solutions:**
1. Wait and retry:
   - Rate limits are usually temporary
   - Wait a few minutes and try again

2. Check your usage:
   - Visit the AI provider's dashboard
   - Check your current usage and limits

3. Optimize token usage:
   - Reduce the length of your prompts
   - Use more efficient prompting techniques

## SPARC Workflow Issues

### SPARC Workflow Not Initializing

**Symptoms:**
- SPARC icon missing from activity bar
- SPARC commands not available

**Solutions:**
1. Check if SPARC workflow is enabled:
   - Open Settings (File > Preferences > Settings)
   - Search for "sparc-workflow.enabled"
   - Set to "true"

2. Verify SPARC configuration:
   - Check `src/config/product.json` for SPARC configuration
   - Ensure the SPARC phases are correctly defined

3. Reinitialize SPARC workflow:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "SPARC: Initialize Workflow"

### Phase Switching Fails

**Symptoms:**
- Error when trying to switch SPARC phases
- Phase doesn't change after selection

**Solutions:**
1. Check phase transition requirements:
   - Some phases may require artifacts from previous phases
   - Create the necessary artifacts and try again

2. Verify phase configuration:
   - Check `src/config/product.json` for phase definitions
   - Ensure the phases are correctly configured

3. Reset phase state:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "SPARC: Reset Phase State"

### Template Creation Fails

**Symptoms:**
- Error when trying to create from template
- Template content not generated

**Solutions:**
1. Check template directory:
   - Verify that the template directory exists
   - Ensure the templates are correctly formatted

2. Check template permissions:
   - Ensure you have read permissions for the template files
   - Check file ownership and permissions

3. Create template manually:
   - Copy the template content from `src/sparc-workflow/templates/`
   - Create a new file and paste the content

## Performance Problems

### High CPU Usage

**Symptoms:**
- SPARC IDE uses excessive CPU resources
- Fan runs at high speed
- System becomes sluggish

**Solutions:**
1. Identify resource-intensive extensions:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Open Process Explorer"
   - Identify processes using high CPU

2. Disable unnecessary extensions:
   - Open Extensions view (Ctrl+Shift+X)
   - Disable extensions you don't need

3. Adjust performance settings:
   - Open Settings (File > Preferences > Settings)
   - Search for "performance"
   - Enable "performance.lazyLoading" and "performance.caching"

### Memory Leaks

**Symptoms:**
- Memory usage increases over time
- Performance degrades the longer SPARC IDE runs
- Eventually crashes due to out of memory

**Solutions:**
1. Restart SPARC IDE regularly:
   - Close and reopen SPARC IDE periodically
   - This will release accumulated memory

2. Limit the number of open editors:
   - Open Settings (File > Preferences > Settings)
   - Set "workbench.editor.limit.value" to a reasonable number (e.g., 10)
   - Enable "workbench.editor.limit.enabled"

3. Monitor memory usage:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Open Process Explorer"
   - Watch for processes with increasing memory usage

### Slow File Operations

**Symptoms:**
- Opening, saving, or searching files is slow
- High disk activity during file operations

**Solutions:**
1. Exclude large directories from search:
   - Open Settings (File > Preferences > Settings)
   - Add large directories to "search.exclude" and "files.watcherExclude"

2. Disable auto-save or increase delay:
   - Open Settings (File > Preferences > Settings)
   - Set "files.autoSave" to "off" or increase "files.autoSaveDelay"

3. Check disk health:
   - Run disk health checks using your operating system's tools
   - Consider moving to an SSD if you're using an HDD

## UI/UX Issues

### Theme Not Applied Correctly

**Symptoms:**
- Theme colors not displaying correctly
- Missing or incorrect UI elements

**Solutions:**
1. Reload the theme:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Reload Window"

2. Reset theme settings:
   - Open Settings (File > Preferences > Settings)
   - Search for "workbench.colorTheme"
   - Select a different theme, then switch back

3. Check for theme conflicts:
   - Disable other theme-related extensions
   - Check for custom CSS overrides

### Layout Problems

**Symptoms:**
- UI elements misaligned or overlapping
- Panels or sidebars not displaying correctly

**Solutions:**
1. Reset the layout:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Workbench: Reset Layout"

2. Check display scaling:
   - Adjust your operating system's display scaling
   - Restart SPARC IDE after changing scaling

3. Verify layout configuration:
   - Open Settings (File > Preferences > Settings)
   - Check "ui-ux.customLayout" setting

### Font Rendering Issues

**Symptoms:**
- Fonts appear blurry or pixelated
- Inconsistent font rendering

**Solutions:**
1. Adjust font settings:
   - Open Settings (File > Preferences > Settings)
   - Modify "editor.fontFamily" and "editor.fontSize"
   - Try different font ligature settings

2. Check display resolution:
   - Ensure your display is set to its native resolution
   - Adjust scaling settings if necessary

3. Install required fonts:
   - Install the Fira Code font family
   - Restart SPARC IDE after installing fonts

## Extension Problems

### Extension Installation Fails

**Symptoms:**
- Error message when installing extensions
- Extension appears to install but doesn't work

**Solutions:**
1. Check extension compatibility:
   - Verify that the extension is compatible with SPARC IDE
   - Check the extension's minimum required VS Code version

2. Install manually:
   - Download the VSIX file from the extension's repository
   - Install using "Install from VSIX" in the Extensions view

3. Check extension logs:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Show Logs"
   - Look for errors related to extension installation

### Extension Conflicts

**Symptoms:**
- Extensions interfere with each other
- Features stop working after installing a new extension

**Solutions:**
1. Identify conflicting extensions:
   - Disable extensions one by one to identify the conflict
   - Enable only essential extensions

2. Check for keybinding conflicts:
   - Open Keyboard Shortcuts (Ctrl+K Ctrl+S)
   - Look for duplicate keybindings

3. Update extensions:
   - Update all extensions to their latest versions
   - Conflicts are often resolved in updates

### Extension Security Warnings

**Symptoms:**
- Security warnings when installing or using extensions
- Extensions blocked by security settings

**Solutions:**
1. Verify extension source:
   - Only install extensions from trusted sources
   - Check the extension's publisher and reviews

2. Adjust security settings:
   - Open Settings (File > Preferences > Settings)
   - Search for "security.workspace.trust"
   - Configure trust settings appropriately

3. Review extension permissions:
   - Check what permissions the extension requires
   - Consider if the permissions are appropriate for the extension's functionality

## Crash Recovery

### Recovering from Crashes

**Symptoms:**
- SPARC IDE crashes unexpectedly
- Unsaved work may be lost

**Solutions:**
1. Check for crash reports:
   - **Linux**: Look in `~/.config/sparc-ide/logs`
   - **macOS**: Look in `~/Library/Application Support/sparc-ide/logs`
   - **Windows**: Look in `%APPDATA%\sparc-ide\logs`

2. Restore from backup:
   - SPARC IDE creates automatic backups
   - Look for files with `.backup` extension

3. Use crash recovery:
   - Restart SPARC IDE
   - It should prompt to recover unsaved changes

### Preventing Data Loss

**Symptoms:**
- Frequent crashes leading to data loss
- Concerns about losing work

**Solutions:**
1. Enable auto-save:
   - Open Settings (File > Preferences > Settings)
   - Set "files.autoSave" to "afterDelay"
   - Set "files.autoSaveDelay" to a low value (e.g., 1000ms)

2. Use version control:
   - Commit changes frequently
   - Push to a remote repository regularly

3. Create regular backups:
   - Set up automated backups of your workspace
   - Use cloud storage for important files

## Diagnostic Tools

### Built-in Diagnostics

SPARC IDE includes several built-in diagnostic tools:

1. **Process Explorer**:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Open Process Explorer"
   - Monitor CPU and memory usage of SPARC IDE processes

2. **Developer Tools**:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Toggle Developer Tools"
   - Access browser-like developer tools for debugging

3. **Extension Logs**:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "Developer: Show Logs"
   - View logs from extensions and SPARC IDE components

### Log Files

SPARC IDE generates log files that can help diagnose issues:

1. **Main Log**:
   - **Linux**: `~/.config/sparc-ide/logs/main.log`
   - **macOS**: `~/Library/Application Support/sparc-ide/logs/main.log`
   - **Windows**: `%APPDATA%\sparc-ide\logs\main.log`

2. **Renderer Log**:
   - **Linux**: `~/.config/sparc-ide/logs/renderer.log`
   - **macOS**: `~/Library/Application Support/sparc-ide/logs/renderer.log`
   - **Windows**: `%APPDATA%\sparc-ide\logs\renderer.log`

3. **Extension Host Log**:
   - **Linux**: `~/.config/sparc-ide/logs/exthost.log`
   - **macOS**: `~/Library/Application Support/sparc-ide/logs/exthost.log`
   - **Windows**: `%APPDATA%\sparc-ide\logs\exthost.log`

### Diagnostic Commands

SPARC IDE provides several diagnostic commands:

1. **Collect Diagnostics**:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "SPARC: Collect Diagnostics"
   - This creates a diagnostic report in the logs directory

2. **Check System Requirements**:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "SPARC: Check System Requirements"
   - This verifies if your system meets the requirements

3. **Verify Installation**:
   - Open Command Palette (Ctrl+Shift+P)
   - Run "SPARC: Verify Installation"
   - This checks if all components are correctly installed

### Getting Help

If you're unable to resolve an issue using this guide:

1. **Community Support**:
   - Join the [SPARC IDE Discord](https://discord.gg/sparc-ide)
   - Post on the [SPARC IDE Forum](https://forum.sparc-ide.org)

2. **Issue Reporting**:
   - Check existing issues on [GitHub](https://github.com/sparc-ide/sparc-ide/issues)
   - Create a new issue with detailed information about your problem

3. **Documentation**:
   - Check the [SPARC IDE Documentation](https://docs.sparc-ide.org)
   - Look for updated troubleshooting information