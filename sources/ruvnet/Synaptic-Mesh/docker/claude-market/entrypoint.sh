#!/bin/sh

# Minimal entrypoint for secure Claude execution
# Reads JSON from stdin, executes Claude, outputs to stdout

set -e

# Validate environment
if [ -z "$CLAUDE_API_KEY" ]; then
    echo '{"error": "CLAUDE_API_KEY not provided"}' >&2
    exit 1
fi

# In production, this would execute the actual Claude binary
# with the provided stdin input and security restrictions
echo '{"status": "placeholder", "message": "Claude execution container ready"}'