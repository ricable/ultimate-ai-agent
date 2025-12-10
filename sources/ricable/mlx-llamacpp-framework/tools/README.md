# Flow2 Tools Directory

This directory contains development tools, utilities, and setup scripts for the Flow2 project.

## Contents

### Setup and Installation
- **`setup.py`** - Python package setup script for development installation

## Usage

### Development Installation
```bash
# Install Flow2 in development mode
cd /path/to/flow2
python tools/setup.py develop

# Or use pip with editable install
pip install -e .
```

### Adding New Tools

When adding new development tools, place them in this directory and follow these guidelines:

1. **Scripts**: Executable scripts with proper shebang lines
2. **Utilities**: Helper utilities for development workflow
3. **Setup**: Installation and configuration tools
4. **Testing**: Development testing utilities

### Tool Categories

#### Build Tools
Tools for building, packaging, and distributing the project.

#### Development Utilities  
Tools for development workflow, code generation, and automation.

#### Testing Tools
Tools for testing, validation, and quality assurance.

#### Deployment Tools
Tools for deployment, configuration, and production setup.

## Best Practices

- Keep tools focused and single-purpose
- Include proper documentation and help text
- Make tools executable with `chmod +x`
- Use consistent naming conventions
- Add tools to `.gitignore` if they're user-specific

## Tool Template

```python
#!/usr/bin/env python3
"""
Tool Name - Brief description

Usage:
    python tools/tool_name.py [options]
"""

import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Tool description")
    parser.add_argument("--option", help="Option description")
    args = parser.parse_args()
    
    # Tool implementation
    pass

if __name__ == "__main__":
    main()
```