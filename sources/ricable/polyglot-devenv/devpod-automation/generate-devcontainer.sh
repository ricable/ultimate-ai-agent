#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <language>"
    echo "Example: $0 go"
    exit 1
fi

LANGUAGE=$1
BASE_FILE="devpod-automation/templates/base/devcontainer.json"
LANG_FILE="devpod-automation/templates/$LANGUAGE/devcontainer.json"
OUTPUT_FILE="$LANGUAGE-env/.devcontainer.json"

if [ ! -f "$LANG_FILE" ]; then
    echo "Error: Language-specific devcontainer.json not found for '$LANGUAGE'"
    exit 1
fi

# Merge the base and language-specific devcontainer.json files
jq -s '.[0] * .[1]' "$BASE_FILE" "$LANG_FILE" > "$OUTPUT_FILE"

echo "✅ Generated .devcontainer.json for $LANGUAGE at $OUTPUT_FILE"