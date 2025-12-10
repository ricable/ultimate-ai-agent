#!/usr/bin/env python3
"""
Script to fix redacted patch targets in integration tests.
"""

import re

def fix_patch_targets(file_path):
    """Fix redacted patch targets in the test file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace redacted patch targets with actual module paths
    replacements = [
        # Client FASTMCP_AVAILABLE constant
        ("patch('****', True)", "patch('safla.integrations.fastmcp_client.FASTMCP_AVAILABLE', True)"),
        # Server FASTMCP_AVAILABLE constant  
        ("patch('****', True)", "patch('safla.integrations.fastmcp_server.FASTMCP_AVAILABLE', True)"),
        # Adapter FASTMCP_AVAILABLE constant
        ("patch('****', True)", "patch('safla.integrations.fastmcp_adapter.FASTMCP_AVAILABLE', True)"),
    ]
    
    # Apply replacements in order
    for old, new in replacements:
        content = content.replace(old, new, 1)  # Replace only first occurrence each time
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"Fixed patch targets in {file_path}")

if __name__ == "__main__":
    fix_patch_targets("tests/test_fastmcp_integration_fixed.py")