#!/bin/bash
# Test OpenRouter with verbose logging enabled

echo "==================================================="
echo "OpenRouter Proxy Test - Verbose Logging Enabled"
echo "==================================================="
echo ""

# Set verbose logging
export DEBUG=*
export LOG_LEVEL=debug

# Test 1: Simple code generation (should NOT use tools)
echo "Test 1: GPT-4o-mini - Simple Code Generation"
echo "---------------------------------------------------"
echo "Task: Write a Python function to add two numbers. Just show code, do not create any files."
echo ""

timeout 30 node dist/cli-proxy.js \
  --agent coder \
  --task "Write a Python function to add two numbers. Just show code, do not create any files." \
  --provider openrouter \
  --model "openai/gpt-4o-mini" \
  --max-tokens 300 \
  2>&1 | tee /tmp/openrouter-test-1.log

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"
echo ""

if [ $EXIT_CODE -eq 124 ]; then
  echo "❌ TIMEOUT - Command hung for 30 seconds"
  echo ""
  echo "Analyzing logs for timeout cause..."
  echo ""
  grep -E "(INCOMING|CONVERTED|OPENROUTER|CONVERTING|tool)" /tmp/openrouter-test-1.log | tail -20
elif [ $EXIT_CODE -eq 0 ]; then
  echo "✅ COMPLETED"
else
  echo "⚠️  Exit code: $EXIT_CODE"
fi

echo ""
echo "==================================================="
echo "Full logs saved to: /tmp/openrouter-test-1.log"
echo "==================================================="
