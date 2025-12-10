#!/usr/bin/env python3
"""
SAFLA CLI Main Entry Point

This is the main entry point for the SAFLA CLI management system.
It provides comprehensive system management capabilities through a
rich command-line interface.
"""

import sys
import os
from pathlib import Path

# Add the SAFLA package to Python path
safla_root = Path(__file__).parent.parent
sys.path.insert(0, str(safla_root))

# Import the main CLI
from safla.cli_manager import cli

if __name__ == "__main__":
    # Set up environment
    os.environ.setdefault('SAFLA_CLI_MODE', 'true')
    
    try:
        cli()
    except KeyboardInterrupt:
        print("\nCLI interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"CLI error: {e}")
        sys.exit(1)