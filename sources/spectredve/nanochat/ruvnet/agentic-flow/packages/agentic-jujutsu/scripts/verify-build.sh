#!/bin/bash
# Verify WASM builds

set -e
echo "🔍 Verifying agentic-jujutsu WASM builds..."

ERRORS=0
TARGETS=("web" "node" "bundler" "deno")

for target in "${TARGETS[@]}"; do
    echo "📦 Checking $target..."
    DIR="pkg/$target"
    
    [ ! -d "$DIR" ] && { echo "  ❌ Missing: $DIR"; ((ERRORS++)); continue; }
    
    for file in agentic_jujutsu.js agentic_jujutsu.d.ts agentic_jujutsu_bg.wasm; do
        if [ -f "$DIR/$file" ]; then
            echo "  ✓ $file"
        else
            echo "  ❌ Missing: $file"
            ((ERRORS++))
        fi
    done
done

if [ $ERRORS -eq 0 ]; then
    echo "✅ All checks passed!"
    exit 0
else
    echo "❌ $ERRORS errors found"
    exit 1
fi
