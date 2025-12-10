# SPARC IDE Windows Installer Build Guide

This guide explains how to build the Windows installer (.exe) for SPARC IDE.

## Prerequisites

Before building the Windows installer, ensure you have the following prerequisites installed:

- **Windows 10/11** (64-bit)
- **Git** - For cloning repositories
- **Visual Studio Build Tools** - For compiling native modules
- **Node.js** (v16.x recommended) - The script can install this if missing
- **Yarn** (v1.22.x recommended) - The script can install this if missing
- **NSIS** (Nullsoft Scriptable Install System) - The script can install this if missing
- **Administrator privileges** - Required for installing dependencies and creating the installer

## Build Process Overview

The Windows installer build process consists of the following steps:

1. **Setup Build Environment** - Install and configure all necessary dependencies
2. **Clone VSCodium** - Clone the VSCodium repository as the base for SPARC IDE
3. **Apply SPARC IDE Branding** - Customize the application with SPARC IDE branding
4. **Integrate Roo Code** - Download and integrate the Roo Code extension
5. **Build Application** - Compile the application for Windows
6. **Create Installer** - Package the application into an NSIS installer
7. **Verify Build** - Verify the installer and generate a verification report

## Building the Windows Installer

### Automated Build

The easiest way to build the Windows installer is to use the provided batch file:

1. Open Command Prompt as Administrator
2. Navigate to the SPARC IDE repository root directory
3. Run the following command:

```
scripts\create-windows-installer.bat
```

This will execute the PowerShell script with the necessary privileges and handle all steps automatically.

### Manual Build

If you prefer to run the PowerShell script directly:

1. Open PowerShell as Administrator
2. Navigate to the SPARC IDE repository root directory
3. Run the following command:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\create-windows-installer.ps1
```

## Build Configuration

The build process can be customized by modifying the following files:

- `scripts/create-windows-installer.ps1` - Main build script
- `scripts/windows/branding-config.json` - Windows branding configuration
- `scripts/windows/installer-config.nsh` - NSIS installer configuration
- `src/config/product.json` - Product information and configuration

## Build Output

After a successful build, the following files will be created:

- **Windows Installer (.exe)** - Located in `package/windows/`
- **Checksums (SHA-256)** - Located in `package/windows/checksums.sha256`
- **Security Report** - Located in `package/windows/security/final-security-report.txt`
- **Build Log** - Located in `logs/windows-installer-build_[timestamp].log`
- **Verification Report** - Located in `package/windows-verification-report_[timestamp].md`

## Installer Features

The SPARC IDE Windows installer includes the following features:

- **Silent Installation** - Support for silent installation with `/VERYSILENT /SUPPRESSMSGBOXES` flags
- **Custom Installation Directory** - Option to specify installation directory
- **Start Menu Shortcuts** - Creates shortcuts in the Start Menu
- **Desktop Shortcut** - Creates a desktop shortcut (optional)
- **File Associations** - Associates relevant file types with SPARC IDE
- **Registry Settings** - Configures necessary registry settings
- **Roo Code Integration** - Includes the Roo Code extension for AI-powered development

## Troubleshooting

### Common Issues

#### Script Execution Policy

If you encounter an error related to PowerShell execution policy, run PowerShell as Administrator and execute:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

#### Missing Dependencies

If the script fails due to missing dependencies, you can install them manually:

- **Node.js**: Download from https://nodejs.org/
- **Yarn**: Download from https://yarnpkg.com/
- **NSIS**: Download from https://nsis.sourceforge.io/Download
- **Visual Studio Build Tools**: Download from https://visualstudio.microsoft.com/downloads/

#### Build Failures

If the build fails, check the log file in the `logs` directory for detailed error messages. Common issues include:

- Network connectivity problems when downloading dependencies
- Insufficient disk space
- Missing or incompatible dependencies
- Permission issues

## Advanced Configuration

### Customizing the Installer

To customize the installer appearance and behavior:

1. Modify `scripts/windows/branding-config.json` to update product information, file associations, and installer options
2. Update `scripts/windows/installer-config.nsh` to customize NSIS installer behavior
3. Replace branding assets in `branding/windows/` with your custom assets

### Signing the Installer

For production builds, it's recommended to sign the installer with a code signing certificate:

1. Obtain a code signing certificate from a trusted certificate authority
2. Modify the `Create-WindowsInstaller` function in `scripts/create-windows-installer.ps1` to include signing steps
3. Use the `signtool.exe` utility from the Windows SDK to sign the installer

## Security Considerations

The build process includes several security measures:

- **Dependency Verification** - Verifies the integrity of downloaded dependencies
- **Extension Verification** - Verifies the Roo Code extension signature
- **Installer Verification** - Generates checksums and verification reports
- **Security Scanning** - Checks for potential security issues in the build

## Additional Resources

- [NSIS Documentation](https://nsis.sourceforge.io/Docs/)
- [VSCodium Build Documentation](https://github.com/VSCodium/vscodium/blob/master/DOCS.md)
- [Windows Installer Best Practices](https://docs.microsoft.com/en-us/windows/win32/msi/windows-installer-best-practices)