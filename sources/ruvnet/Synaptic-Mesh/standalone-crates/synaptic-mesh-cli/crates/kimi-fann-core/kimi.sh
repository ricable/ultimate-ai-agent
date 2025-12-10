#!/bin/bash
#
# Kimi-FANN Core CLI Wrapper
# 
# This script makes it easier to run kimi without having to remember the -- separator
# 
# Usage:
#   ./kimi.sh "your question here"
#   ./kimi.sh --expert coding "write a function"
#   ./kimi.sh --consensus "complex question"
#

# Check if cargo is available
CARGO_PATH=""
if command -v cargo &> /dev/null; then
    CARGO_PATH="cargo"
elif [ -x "/home/codespace/.cargo/bin/cargo" ]; then
    CARGO_PATH="/home/codespace/.cargo/bin/cargo"
else
    echo "Error: cargo is not installed. Please install Rust first."
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "Cargo.toml" ]; then
    echo "Error: Please run this script from the kimi-fann-core directory"
    exit 1
fi

# Pass all arguments to cargo run with the -- separator
$CARGO_PATH run --bin kimi -- "$@"