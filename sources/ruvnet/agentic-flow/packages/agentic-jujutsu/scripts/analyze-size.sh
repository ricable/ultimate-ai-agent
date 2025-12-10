#!/bin/bash
# Analyze WASM bundle sizes

echo "📊 WASM Bundle Size Analysis"
echo "─────────────────────────────"

for target in web node bundler deno; do
    if [ -f "pkg/$target/agentic_jujutsu_bg.wasm" ]; then
        size=$(ls -lh "pkg/$target/agentic_jujutsu_bg.wasm" | awk '{print $5}')
        echo "$target: $size"
    fi
done

echo ""
echo "📈 Compression Analysis:"
if command -v gzip &> /dev/null; then
    for target in web node bundler deno; do
        if [ -f "pkg/$target/agentic_jujutsu_bg.wasm" ]; then
            gzipped=$(gzip -c "pkg/$target/agentic_jujutsu_bg.wasm" | wc -c | awk '{print int($1/1024)}')
            echo "$target (gzipped): ${gzipped}KB"
        fi
    done
fi
