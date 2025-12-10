# Flow2 Scripts Directory

This directory contains shell scripts and automation tools for the Flow2 project.

## Scripts

### Environment Setup
- **`setup_environment.sh`** - Sets up the development environment, installs dependencies
- **`activate.sh`** - Activates the virtual environment and sets up paths

## Usage

### Initial Setup
```bash
# Set up the development environment
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh
```

### Activate Environment
```bash
# Activate the Flow2 environment
source scripts/activate.sh
```

## Script Guidelines

### Adding New Scripts

When adding new scripts to this directory:

1. **Make executable**: `chmod +x script_name.sh`
2. **Add shebang**: Start with `#!/bin/bash`
3. **Add documentation**: Include usage information
4. **Error handling**: Use `set -e` for strict error handling
5. **Path independence**: Don't assume current directory

### Script Template

```bash
#!/bin/bash
# Script Name - Brief description
#
# Usage: ./script_name.sh [options]
#
# Options:
#   -h, --help    Show this help message

set -e  # Exit on any error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
VERBOSE=false

# Function definitions
usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help     Show this help"
    echo "  -v, --verbose  Verbose output"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            usage
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Main script logic
main() {
    if [[ "$VERBOSE" == true ]]; then
        echo "Running in verbose mode..."
    fi
    
    # Script implementation here
}

# Run main function
main "$@"
```

## Available Scripts

### Environment Scripts
Scripts for setting up and managing the development environment.

### Build Scripts  
Scripts for building, testing, and packaging the project.

### Deployment Scripts
Scripts for deployment and production setup.

### Utility Scripts
Helper scripts for common development tasks.

## Best Practices

1. **Error Handling**: Always use `set -e` and proper error checking
2. **Path Safety**: Use absolute paths and proper quoting
3. **Documentation**: Include usage information and examples
4. **Idempotent**: Scripts should be safe to run multiple times
5. **Platform Compatibility**: Test on different systems when possible

## Dependencies

Scripts may require:
- bash 4.0+
- Common Unix utilities (grep, sed, awk, etc.)
- Python 3.8+
- git
- Platform-specific tools (brew on macOS, apt on Ubuntu, etc.)

## Troubleshooting

### Permission Issues
```bash
# Fix script permissions
chmod +x scripts/*.sh
```

### Path Issues
```bash
# Run from project root
cd /path/to/flow2
./scripts/script_name.sh
```

### Environment Issues
```bash
# Check environment setup
source scripts/activate.sh
which python
```