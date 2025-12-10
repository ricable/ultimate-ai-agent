# SPARC IDE Distribution Packages

This directory contains distribution packages for SPARC IDE across multiple platforms.

## Directory Structure

- `linux/` - Contains packages for Linux (DEB and RPM formats)
- `windows/` - Contains packages for Windows (EXE installer)
- `macos/` - Contains packages for macOS (DMG format)
- `manifest.json` - JSON file containing metadata about all available packages
- `build-report.md` - Detailed report of the build process

## Package Types

### Linux Packages

- `.deb` - Debian/Ubuntu package format
- `.rpm` - Red Hat/Fedora package format

### Windows Packages

- `.exe` - Windows installer

### macOS Packages

- `.dmg` - macOS disk image

## Security Verification

All packages undergo rigorous security verification:

1. Extension signature verification
2. Hardcoded credentials check
3. File permissions check
4. Source integrity verification

Each platform directory contains a `security/` subdirectory with detailed security reports.

## Checksums

Each platform directory contains a `checksums.sha256` file with SHA-256 checksums for all packages in that directory. You can verify the integrity of a package by running:

```bash
cd package/[platform]
sha256sum -c checksums.sha256
```

## Installation

For detailed installation instructions, please refer to the [Installation Guide](../docs/installation-guide.md).

## Building Packages

To build packages for all platforms, run:

```bash
./scripts/package-sparc-ide.sh
```

To build for a specific platform, you can run the build script directly:

```bash
./scripts/build-sparc-ide.sh --platform [linux|windows|macos]
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

## Build Report

The `build-report.md` file contains a detailed report of the build process, including:

- Build status for each platform
- List of packages generated
- Package sizes
- Security verification status

## Version History

For a complete version history, please refer to the [Release Notes](https://github.com/sparc-ide/sparc-ide/releases).