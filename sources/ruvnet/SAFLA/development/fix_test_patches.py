#!/usr/bin/env python3
"""
Script to fix invalid patch targets in FastMCP test files.
"""

import re
import os

def fix_test_file(filepath, module_path):
    """Fix patch targets in a test file."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Replace all instances of '****' with the correct module path
        # Handle different patch patterns
        patterns = [
            (r"patch\('(\*{4})', True\)", f"patch('{module_path}', True)"),
            (r"patch\('(\*{4})', False\)", f"patch('{module_path}', False)"),
            (r"patch\('(\*{4})', return_value=([^)]+)\)", f"patch('fastmcp.FastMCP', return_value=\\2)"),
            (r"patch\('(\*{4})'\) as ([^:]+)", f"patch('fastmcp.FastMCP') as \\2"),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # Also handle any remaining asterisk patterns
        content = content.replace("'****'", f"'{module_path}'")
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"Fixed {filepath}")
        return True
    except Exception as e:
        print(f"Error fixing {filepath}: {e}")
        return False

def main():
    """Main function to fix all test files."""
    test_files = {
        'tests/test_fastmcp_client.py': '****',
        'tests/test_fastmcp_server.py': '****',
        'tests/test_fastmcp_integration.py': '****',
        'tests/test_fastmcp_cli.py': '****'
    }
    
    for filepath, module_path in test_files.items():
        if os.path.exists(filepath):
            fix_test_file(filepath, module_path)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()