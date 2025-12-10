#!/bin/bash
# Test if MCP tools are being sent to OpenRouter

export DEBUG=*
export LOG_LEVEL=debug

echo "Testing MCP tool forwarding to OpenRouter..."
echo ""

timeout 20 node dist/cli-proxy.js \
  --agent coder \
  --task "Create file /tmp/test3.txt with text: Hello" \
  --provider openrouter \
  --model "openai/gpt-4o-mini" \
  --max-tokens 200 \
  2>&1 | tee /tmp/mcp-test.log | grep -E "INCOMING|tool|MCP|Converting" | head -50

echo ""
echo "Full logs saved to /tmp/mcp-test.log"
echo ""
echo "Checking for tool-related logs..."
grep -i "tool" /tmp/mcp-test.log | head -20
