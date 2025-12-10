#!/usr/bin/env python3
"""
Fix indentation issues in test_deployment_readiness.py
"""

import re

def fix_indentation():
    """Fix indentation issues in the deployment readiness test file."""
    file_path = "test_deployment_readiness.py"
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    fixed_lines = []
    in_class = False
    
    for i, line in enumerate(lines):
        # Track if we're inside a class
        if line.strip().startswith('class '):
            in_class = True
            fixed_lines.append(line)
            continue
        
        # If we hit another class or end of file, we're out of the current class
        if line.strip().startswith('class ') and in_class:
            in_class = False
        
        # Fix method definitions - they should have 4 spaces inside a class
        if in_class and line.strip().startswith('async def '):
            # Ensure proper indentation for methods
            method_line = '    ' + line.lstrip()
            fixed_lines.append(method_line)
            continue
        
        # Fix decorator indentation - should match method indentation
        if in_class and line.strip().startswith('@'):
            decorator_line = '    ' + line.lstrip()
            fixed_lines.append(decorator_line)
            continue
        
        # Keep other lines as-is
        fixed_lines.append(line)
    
    # Write the fixed content back
    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed indentation issues")

if __name__ == "__main__":
    fix_indentation()