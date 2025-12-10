#!/bin/bash
# Example demonstrating SPARC CLI cowboy mode
# This script shows automated command execution without confirmation prompts

# Run SPARC in cowboy mode to automatically format Python files
sparc -m "Format all Python files in src/ directory using black" \
    --cowboy-mode
