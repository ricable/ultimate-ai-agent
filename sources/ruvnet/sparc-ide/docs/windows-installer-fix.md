# SPARC IDE Windows Installer Fix

This document explains the issue with the Windows installer showing "it can't find a version to run on your pc" and the solution implemented.

## Problem Diagnosis

The Windows installer was showing the error "it can't find a version to run on your pc" when executed. After investigation, the following issues were identified:

1. The installer file (`package/windows/SPARC-IDE-1.0.0-windows-x64.exe`) was not a real Windows executable but a text file with an .exe extension
2. The mock installer was created by `scripts/mock-windows-installer.sh` which simply outputs text content rather than building a proper installer
3. Windows branding assets required for building a real installer were missing
4. The proper build process using VSCodium as a base was not being executed

We confirmed this by examining the installer file:
```
$ file package/windows/SPARC-IDE-1.0.0-windows-x64.exe
package/windows/SPARC-IDE-1.0.0-windows-x64.exe: ASCII text

$ cat package/windows/SPARC-IDE-1.0.0-windows-x64.exe
This is a mock SPARC IDE Windows installer file
```

## Solution Implemented

To fix these issues, the following changes were made:

1. Created `scripts/prepare-windows-branding.sh` to download and prepare Windows branding assets
2. Created `scripts/build-real-windows-installer.sh` to replace the mock script with a proper build process
3. Added verification steps to ensure compatibility with different Windows versions and architectures
4. Added detailed documentation on how to build and use the Windows installer

The new build process:
- Uses VSCodium as a base for SPARC IDE
- Properly builds a Windows executable using NSIS (Nullsoft Scriptable Install System)
- Adds SPARC IDE branding and customizations
- Creates installers that are compatible with Windows 10 and 11, both 32-bit and 64-bit architectures
- Includes all required dependencies bundled with the installer
- Provides proper verification steps to ensure installer functionality

## How to Build the Windows Installer

### Prerequisites

- Windows 10/11 or Windows Subsystem for Linux (WSL)
- Visual Studio Build Tools
- Node.js (v16+)
- Yarn
- NSIS (Nullsoft Scriptable Install System)
- Administrator privileges

### Building the Installer

1. Run the following command to build the Windows installer:

```bash
# Make the script executable
chmod +x scripts/build-real-windows-installer.sh

# Run the script
./scripts/build-real-windows-installer.sh
```

2. The script will:
   - Check if running on Windows or WSL
   - Prepare Windows branding assets if needed
   - Build the proper Windows installer
   - Verify the installer functionality
   - Generate documentation about the installer

3. The resulting installer will be created at:
   ```
   package/windows/SPARC-IDE-1.0.0-windows-x64.exe
   ```

### On Non-Windows Systems

If running on a non-Windows system, the script will:
1. Warn that a real Windows installer cannot be built
2. Create a mock installer as a fallback (with clear warnings)
3. Provide instructions on how to build a real installer on Windows or WSL

## Verification Steps

To verify the installer functionality:

1. Run the verification script on Windows:
   ```powershell
   powershell.exe -ExecutionPolicy Bypass -File tests/windows/verify-multi-arch.ps1
   ```

2. The verification script checks:
   - Windows version and architecture
   - Installer file format (confirms it's a real Windows executable)
   - Architecture compatibility (x86, x64, ARM)
   - Installer extraction and file integrity
   - Executable format validation

## Compatibility

The new Windows installer is compatible with:
- Windows 10 (x86/x64)
- Windows 11 (x86/x64)

## Silent Installation

For automated deployments, use the following command:
```
SPARC-IDE-1.0.0-windows-x64.exe /VERYSILENT /SUPPRESSMSGBOXES /NORESTART
```

## Troubleshooting

If you encounter issues with the Windows installer:

1. Check the build logs in the `logs` directory
2. Verify that all prerequisites are installed correctly
3. Ensure you have administrator privileges when building the installer
4. Check the verification logs in `test-reports` directory

For additional assistance, refer to `docs/windows-installer-guide.md` for more detailed information.