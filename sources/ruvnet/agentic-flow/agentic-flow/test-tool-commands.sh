#!/bin/bash
# Test tool emulation with non-interactive commands

echo "==================================="
echo "Testing Tool Emulation System"
echo "==================================="
echo ""

echo "Test 1: Simple math (no tools needed)"
echo "--------------------------------------"
node dist/cli-proxy.js --agent coder --task "Calculate 12 + 8" --provider openrouter --model "mistralai/mistral-7b-instruct" --max-tokens 50 2>&1 | tail -10
echo ""

echo "Test 2: Native tool model (DeepSeek)"
echo "--------------------------------------"
node dist/cli-proxy.js --agent coder --task "Calculate 15 + 25" --provider openrouter --model "deepseek/deepseek-chat" --max-tokens 50 2>&1 | tail -10
echo ""

echo "Test 3: Check emulation messages appear"
echo "--------------------------------------"
node dist/cli-proxy.js --agent coder --task "9 * 7" --provider openrouter --model "mistralai/mistral-7b-instruct" --max-tokens 50 2>&1 | grep -E "Tool Emulation|REACT|emulation"
echo ""

echo "==================================="
echo "Tests Complete"
echo "==================================="
