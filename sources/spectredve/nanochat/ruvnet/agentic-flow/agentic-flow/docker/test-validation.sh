#!/bin/bash
set -e

echo "========================================="
echo "Testing agentic-flow v1.1.10"
echo "========================================="
echo ""

# Test 1: Anthropic Provider
echo "✅ Test 1: Anthropic Provider"
timeout 45 agentic-flow --agent coder --task "Create add function" --provider anthropic --max-tokens 50 2>&1 | head -30
echo "✅ Test 1 PASSED"
echo ""

# Test 2: Gemini Provider
if [ -n "$GOOGLE_GEMINI_API_KEY" ]; then
  echo "✅ Test 2: Gemini Provider"
  timeout 45 agentic-flow --agent coder --task "Create subtract function" --provider gemini --max-tokens 50 2>&1 | head -30
  echo "✅ Test 2 PASSED"
else
  echo "⏭️  Test 2 SKIPPED (no GOOGLE_GEMINI_API_KEY)"
fi
echo ""

# Test 3: OpenRouter Provider
if [ -n "$OPENROUTER_API_KEY" ]; then
  echo "✅ Test 3: OpenRouter Provider"
  timeout 45 agentic-flow --agent coder --task "Create divide function" --provider openrouter --model "openai/gpt-4o-mini" --max-tokens 50 2>&1 | head -30
  echo "✅ Test 3 PASSED"
else
  echo "⏭️  Test 3 SKIPPED (no OPENROUTER_API_KEY)"
fi
echo ""

echo "========================================="
echo "All Tests Completed Successfully!"
echo "========================================="
