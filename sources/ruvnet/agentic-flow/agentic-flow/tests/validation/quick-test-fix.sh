#!/bin/bash
# Quick test of the anthropicReq.system fix

echo "Testing OpenRouter with fixed system field handling..."
echo ""

timeout 20 node dist/cli-proxy.js \
  --agent coder \
  --task "def add(a,b): return a+b" \
  --provider openrouter \
  --model "openai/gpt-4o-mini" \
  --max-tokens 200 \
  2>&1 | grep -A5 -B5 "===.*REQUEST\|===.*RESPONSE\|tool\|INCOMING\|CONVERTED\|finish"

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 124 ]; then
  echo "❌ Still timing out"
else
  echo "✅ Exit code: $EXIT_CODE"
fi
