"""
SAFLA CLI Entry Point

This module provides the main entry point for the SAFLA CLI when the package
is executed with `python -m safla`.
"""

import sys
from safla.cli import main

if __name__ == "__main__":
    sys.exit(main())