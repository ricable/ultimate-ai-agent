#!/usr/bin/env python3
"""Entry point script for Flow4 pipeline."""

import sys
from pathlib import Path

# Add project root to path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import from src package
from src.cli.main import main

if __name__ == "__main__":
    sys.exit(main())