#!/bin/bash
# Example demonstrating SPARC CLI research-only mode
# This script shows how to analyze a codebase without making changes

# Run SPARC in research-only mode to analyze a Python project
sparc -m "Analyze the security of my Flask application and identify potential vulnerabilities" \
    --research-only \
    --hil
