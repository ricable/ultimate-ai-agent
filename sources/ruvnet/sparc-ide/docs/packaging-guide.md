# SPARC IDE Packaging Guide

This guide explains how to build and package SPARC IDE for multiple platforms.

## Prerequisites

Before building SPARC IDE packages, ensure you have the following prerequisites installed:

- **Linux Build Requirements:**
  - Node.js 18.x or later
  - Yarn 1.22.x or later
  - build-essential package
  - libgtk-3-dev
  - libxss-dev
  - rpm (for RPM package creation)

- **Windows Build Requirements:**
  - Windows Subsystem for Linux (WSL) with Ubuntu
  - NSIS (Nullsoft Scriptable Install System)
  - Wine (for testing Windows builds on Linux)

- **macOS Build Requirements:**
  - macOS 10.15 or later
  - Xcode Command Line Tools
  - Node.js 18.x or later
  - Yarn 1.22.x or later

## Building Packages

### Building for All Platforms

To build packages for all supported platforms (Linux, Windows, macOS), run:

```bash
./scripts/package-sparc-ide.sh
```

This script will:

1. Create a package directory structure
2. Execute the build-sparc-ide.sh script for each platform
3. Copy the resulting build artifacts to the appropriate package subdirectories
4. Create a manifest file listing all available packages with their versions
5. Generate a build report

The packages will be available in the `package/` directory, organized by platform.

### Building for a Specific Platform

If you only need to build for a specific platform, you can run:

```bash
./scripts/build-sparc-ide.sh --platform [linux|windows|macos]
```

This will build SPARC IDE for the specified platform and place the artifacts in the `dist/` directory.

## Package Directory Structure

After running the packaging script, the following directory structure will be created:

```
package/
├── linux/                  # Linux packages
│   ├── *.deb               # Debian/Ubuntu package
│   ├── *.rpm               # Red Hat/Fedora package
│   ├── checksums.sha256    # SHA-256 checksums
│   └── security/           # Security reports
├── windows/                # Windows packages
│   ├── *.exe               # Windows installer
│   ├── checksums.sha256    # SHA-256 checksums
│   └── security/           # Security reports
├── macos/                  # macOS packages
│   ├── *.dmg               # macOS disk image
│   ├── checksums.sha256    # SHA-256 checksums
│   └── security/           # Security reports
├── manifest.json           # Package manifest
├── build-report.md         # Build report
└── README.md               # Package documentation
```

## Manifest File

The `manifest.json` file contains metadata about all available packages, including:

- Package name
- Version
- File size
- SHA-256 checksum
- Package type
- System requirements

This file can be used by automated tools to download and verify packages.

## Verifying Packages

To verify the integrity of a package, you can use the provided checksums:

```bash
cd package/[platform]
sha256sum -c checksums.sha256
```

## Customizing the Build

### Customizing Branding

To customize the branding of SPARC IDE, modify the files in the `branding/` directory before running the packaging script.

### Customizing UI Settings

To customize the default UI settings, modify the files in the `src/config/` directory:

- `product.json` - Product information
- `settings.json` - Default editor settings
- `keybindings.json` - Default keyboard shortcuts

## Troubleshooting

### Build Failures

If a build fails, check the following:

1. Ensure all prerequisites are installed
2. Check the build logs for specific errors
3. Verify that the build environment is properly set up

### Security Verification Failures

If security verification fails, check the security reports in the `security-reports/` directory for specific issues.

## Continuous Integration

The packaging script can be integrated into CI/CD pipelines to automate the build process. For example, you can use GitHub Actions to build packages for all platforms on each release.

## Distribution

After building the packages, you can distribute them through various channels:

- GitHub Releases
- Package repositories (apt, yum, etc.)
- Direct download from your website

Ensure that you include the checksums and installation instructions with the distributed packages.