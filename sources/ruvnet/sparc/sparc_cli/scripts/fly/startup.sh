#!/bin/bash

# Set environment variables
export TERM=xterm-256color
export PYTHONPATH=/opt/sparc

# Change to SPARC directory
cd /opt/sparc

# Function to show usage
show_usage() {
    echo "Usage: startup.sh [mode]"
    echo
    echo "Modes:"
    echo "  chat     : Start in chat mode"
    echo "  research : Start in research mode"
    echo "  cowboy   : Start in cowboy mode (no confirmations)"
    echo
}

# Parse mode argument
MODE=${1:-chat}

case $MODE in
    chat)
        python -m sparc_cli --chat
        ;;
    research)
        python -m sparc_cli --research-only
        ;;
    cowboy)
        python -m sparc_cli --cowboy-mode
        ;;
    help|--help|-h)
        show_usage
        exit 0
        ;;
    *)
        echo "Error: Unknown mode '$MODE'"
        show_usage
        exit 1
        ;;
esac
