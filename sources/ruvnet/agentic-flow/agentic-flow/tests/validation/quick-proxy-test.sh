#!/bin/bash
# Quick OpenRouter Proxy Test
# Tests actual behavior vs expected behavior

echo "═══════════════════════════════════════════════════════════"
echo "Quick OpenRouter Proxy Diagnostic"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Test 1: Simple code generation (should NOT have tool calls)
echo "Test 1: GPT-4o-mini - Simple Code Generation"
echo "Task: Write a Python function to add two numbers. Just show code."
echo "Expected: Clean Python code, NO tool calls"
echo ""

OUTPUT=$(node dist/cli-proxy.js \
  --agent coder \
  --task "Write a Python function called add that takes two numbers and returns their sum. Just show me the code, do not create any files." \
  --provider openrouter \
  --model "openai/gpt-4o-mini" \
  --max-tokens 500 2>&1)

echo "$OUTPUT" | grep -E "(def |function|tool_use|<function|file_write)" | head -10

if echo "$OUTPUT" | grep -q "tool_use\|<function\|file_write"; then
  echo "❌ FAIL: Contains tool calls when it shouldn't"
else
  if echo "$OUTPUT" | grep -q "def \|function"; then
    echo "✅ PASS: Clean code without tool calls"
  else
    echo "❌ FAIL: No code found"
  fi
fi

echo ""
echo "───────────────────────────────────────────────────────────"
echo ""

# Test 2: File operation (SHOULD have tool calls)
echo "Test 2: GPT-4o-mini - File Write Operation"
echo "Task: Create a file /tmp/test.py with a function"
echo "Expected: Should use Write tool, file should be created"
echo ""

OUTPUT2=$(node dist/cli-proxy.js \
  --agent coder \
  --task "Create a Python file at /tmp/agentic-test.py with a function to subtract two numbers" \
  --provider openrouter \
  --model "openai/gpt-4o-mini" \
  --max-tokens 500 2>&1)

echo "$OUTPUT2" | grep -E "(Write|file_write|tool_use)" | head -5

if [ -f "/tmp/agentic-test.py" ]; then
  echo "✅ File created successfully"
  cat /tmp/agentic-test.py
  rm /tmp/agentic-test.py
else
  echo "❌ File NOT created"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
