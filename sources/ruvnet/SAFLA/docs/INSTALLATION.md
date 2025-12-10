# SAFLA Installation Guide

This guide provides comprehensive instructions for installing SAFLA (Self-Aware Feedback Loop Algorithm) with its modular installation structure and rich UI components.

## Table of Contents

1. [Quick Installation](#quick-installation)
2. [Interactive Installation](#interactive-installation)
3. [Development Installation](#development-installation)
4. [Package Structure](#package-structure)
5. [Configuration](#configuration)
6. [Validation](#validation)
7. [Troubleshooting](#troubleshooting)
8. [Building from Source](#building-from-source)

## Quick Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Operating System: Windows, macOS, or Linux
- Memory: Minimum 512MB RAM
- Disk Space: At least 100MB free space

### Install from PyPI

```bash
pip install safla
```

### Install from Source

```bash
pip install git+https://github.com/ruvnet/SAFLA.git
```

### Verify Installation

```bash
safla validate
```

## Interactive Installation

SAFLA provides an interactive installer with rich UI components for a guided setup experience.

### Step 1: Install the Package

```bash
pip install safla
```

### Step 2: Run Interactive Installer

```bash
safla-install
```

### Interactive Installer Features

The interactive installer provides:

- **System Requirements Validation**: Checks Python version, OS compatibility, and available resources
- **Dependency Checking**: Validates all required dependencies are properly installed
- **Configuration Setup**: Guides through initial configuration options
- **Progress Tracking**: Rich progress bars and status indicators
- **Installation Verification**: Comprehensive validation of the installation
- **Error Handling**: Clear error messages and resolution suggestions

### Installation Process

1. **Welcome Screen**: Introduction and overview of the installation process
2. **System Check**: Validation of system requirements and compatibility
3. **Dependency Installation**: Automatic installation of required dependencies
4. **Configuration**: Setup of initial configuration files and settings
5. **Validation**: Comprehensive testing of the installation
6. **Completion**: Summary of installation results and next steps

## Development Installation

For developers who want to contribute to SAFLA or modify the source code.

### Clone Repository

```bash
git clone https://github.com/ruvnet/SAFLA.git
cd SAFLA
```

### Install in Development Mode

```bash
# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Manual Dependency Installation

```bash
# Install dependencies manually
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt
```

### Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit configuration (use your preferred editor)
nano .env
```

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=safla tests/

# Run specific test files
python -m pytest tests/test_installation.py
```

## Package Structure

The SAFLA package follows modern Python packaging standards:

```
safla/
├── __init__.py          # Main package exports and metadata
├── cli.py              # Command-line interface with rich UI
├── installer.py        # Interactive installer with progress tracking
├── exceptions.py       # Custom exception hierarchy
└── utils/
    ├── __init__.py     # Utilities package
    ├── config.py       # Configuration management
    ├── logging.py      # Logging utilities with rich formatting
    └── validation.py   # System and installation validation

scripts/
├── install.py          # Standalone installation script
└── build.py           # Package build script

tests/
└── test_installation.py # Installation validation tests

Configuration Files:
├── pyproject.toml      # Primary packaging configuration (PEP 621)
├── setup.py           # Backward compatibility setup script
├── setup.cfg          # Additional packaging configuration
├── requirements.txt   # Package dependencies
├── MANIFEST.in        # Distribution file inclusion rules
└── py.typed          # Type hints marker file
```

## Configuration

SAFLA supports multiple configuration methods:

### 1. Configuration File

Create `~/.safla/config.yaml`:

```yaml
# SAFLA Configuration
log_level: "INFO"
enable_monitoring: true
max_iterations: 1000
convergence_threshold: 0.001

# Memory settings
memory:
  max_size: 1000000000  # 1GB
  cleanup_threshold: 0.8

# Safety settings
safety:
  enable_constraints: true
  emergency_stop_threshold: 0.95
```

### 2. Environment Variables

```bash
# Core settings
export SAFLA_LOG_LEVEL="INFO"
export SAFLA_ENABLE_MONITORING="true"
export SAFLA_MAX_ITERATIONS="1000"

# Memory settings
export SAFLA_MEMORY_MAX_SIZE="1000000000"
export SAFLA_MEMORY_CLEANUP_THRESHOLD="0.8"

# Safety settings
export SAFLA_SAFETY_ENABLE_CONSTRAINTS="true"
export SAFLA_SAFETY_EMERGENCY_STOP_THRESHOLD="0.95"
```

### 3. Programmatic Configuration

```python
from safla.utils.config import SAFLAConfig

config = SAFLAConfig(
    log_level="INFO",
    enable_monitoring=True,
    max_iterations=1000,
    convergence_threshold=0.001
)
```

### 4. Interactive Configuration

```bash
safla config interactive
```

## Validation

SAFLA includes comprehensive validation capabilities:

### System Validation

```bash
# Validate entire system
safla validate

# Validate specific components
safla validate --component memory
safla validate --component safety
safla validate --component config
```

### Programmatic Validation

```python
from safla.utils.validation import validate_installation, validate_system_requirements

# Validate installation
is_valid, issues = validate_installation()
if not is_valid:
    print("Installation issues:", issues)

# Validate system requirements
requirements_met, missing = validate_system_requirements()
if not requirements_met:
    print("Missing requirements:", missing)
```

### Validation Components

- **Python Version**: Ensures Python 3.8+ is installed
- **Dependencies**: Validates all required packages are available
- **System Resources**: Checks available memory and disk space
- **Configuration**: Validates configuration files and settings
- **Permissions**: Ensures proper file and directory permissions
- **Package Integrity**: Verifies package files and structure

## Troubleshooting

### Common Issues

#### 1. Python Version Compatibility

**Error**: `Python version 3.7 is not supported`

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.8+ or use pyenv
pyenv install 3.9.0
pyenv global 3.9.0
```

#### 2. Dependency Installation Failures

**Error**: `Failed to install required dependencies`

**Solution**:
```bash
# Update pip
pip install --upgrade pip

# Install dependencies manually
pip install -r requirements.txt

# Use conda if pip fails
conda install -c conda-forge rich click pydantic
```

#### 3. Permission Errors

**Error**: `Permission denied when creating configuration directory`

**Solution**:
```bash
# Create directory manually
mkdir -p ~/.safla

# Set proper permissions
chmod 755 ~/.safla

# Run installer with user permissions
safla-install --user
```

#### 4. Configuration Issues

**Error**: `Invalid configuration format`

**Solution**:
```bash
# Validate configuration
safla config validate

# Reset to defaults
safla config reset

# Use interactive configuration
safla config interactive
```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
# Set debug environment variable
export SAFLA_DEBUG=true

# Run with debug logging
safla --debug validate

# Run installer with debug output
safla-install --debug
```

### Getting Help

```bash
# Show help for main command
safla --help

# Show help for specific commands
safla validate --help
safla config --help

# Show version information
safla --version
```

## Building from Source

For advanced users who want to build the package from source:

### Prerequisites

```bash
# Install build tools
pip install build twine wheel
```

### Build Process

```bash
# Clone repository
git clone https://github.com/ruvnet/SAFLA.git
cd SAFLA

# Clean previous builds
python scripts/build.py

# Or build manually
python -m build
```

### Build Script Features

The build script (`scripts/build.py`) provides:

- **Artifact Cleaning**: Removes previous build artifacts
- **Source Distribution**: Creates `.tar.gz` source distribution
- **Wheel Distribution**: Creates `.whl` binary distribution
- **Package Validation**: Validates built packages with twine
- **Rich UI**: Progress indicators and formatted output

### Manual Build Steps

```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info/

# Build source distribution
python -m build --sdist

# Build wheel distribution
python -m build --wheel

# Validate packages
python -m twine check dist/*
```

### Upload to PyPI

```bash
# Test upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Upload to PyPI
python -m twine upload dist/*
```

## Advanced Installation Options

### Virtual Environment

```bash
# Create virtual environment
python -m venv safla-env

# Activate (Linux/macOS)
source safla-env/bin/activate

# Activate (Windows)
safla-env\Scripts\activate

# Install SAFLA
pip install safla
```

### Docker Installation

```dockerfile
FROM python:3.9-slim

# Install SAFLA
RUN pip install safla

# Set working directory
WORKDIR /app

# Run validation
RUN safla validate

CMD ["safla", "--help"]
```

### Conda Installation

```bash
# Create conda environment
conda create -n safla python=3.9

# Activate environment
conda activate safla

# Install SAFLA
pip install safla
```

## Next Steps

After successful installation:

1. **Validate Installation**: Run `safla validate` to ensure everything is working
2. **Configure System**: Use `safla config interactive` for initial setup
3. **Read Documentation**: Check the [main documentation](README.md) for usage examples
4. **Run Examples**: Try the examples in the `examples/` directory
5. **Join Community**: Participate in discussions and contribute to the project

## Support

- **Documentation**: [README.md](../README.md)
- **Issues**: [GitHub Issues](https://github.com/ruvnet/SAFLA/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ruvnet/SAFLA/discussions)
- **Email**: support@safla.dev (if available)

---

*This installation guide covers the modular installation structure for SAFLA with rich UI components and comprehensive validation.*