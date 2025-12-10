#!/bin/bash
# Test OpenRouter with MCP file operation tools

echo "==================================================="
echo "OpenRouter File Operations Test"
echo "==================================================="
echo ""

# Test 1: Create a file with Write tool
echo "Test 1: GPT-4o-mini - Create file with Write tool"
echo "---------------------------------------------------"

timeout 30 node dist/cli-proxy.js \
  --agent coder \
  --task "Create a file at /tmp/test-openrouter.py with a function that adds two numbers" \
  --provider openrouter \
  --model "openai/gpt-4o-mini" \
  --max-tokens 500

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 124 ]; then
  echo "❌ TIMEOUT"
elif [ $EXIT_CODE -eq 0 ]; then
  echo "✅ COMPLETED"
  if [ -f /tmp/test-openrouter.py ]; then
    echo "✅ File created successfully"
    echo "Content:"
    cat /tmp/test-openrouter.py
  else
    echo "❌ File was NOT created"
  fi
else
  echo "⚠️  Exit code: $EXIT_CODE"
fi

echo ""
echo "==================================================="

# Test 2: Read the file with Read tool
echo ""
echo "Test 2: Llama 3.3 - Read file with Read tool"
echo "---------------------------------------------------"

timeout 30 node dist/cli-proxy.js \
  --agent coder \
  --task "Read the file /tmp/test-openrouter.py and explain what it does" \
  --provider openrouter \
  --model "meta-llama/llama-3.3-70b-instruct" \
  --max-tokens 500

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 124 ]; then
  echo "❌ TIMEOUT"
elif [ $EXIT_CODE -eq 0 ]; then
  echo "✅ COMPLETED"
else
  echo "⚠️  Exit code: $EXIT_CODE"
fi

echo ""
echo "==================================================="

# Test 3: Execute bash command
echo ""
echo "Test 3: GPT-4o-mini - Run bash command"
echo "---------------------------------------------------"

timeout 30 node dist/cli-proxy.js \
  --agent coder \
  --task "Use bash to run: python /tmp/test-openrouter.py" \
  --provider openrouter \
  --model "openai/gpt-4o-mini" \
  --max-tokens 500

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 124 ]; then
  echo "❌ TIMEOUT"
elif [ $EXIT_CODE -eq 0 ]; then
  echo "✅ COMPLETED"
else
  echo "⚠️  Exit code: $EXIT_CODE"
fi

echo ""
echo "==================================================="
echo "File Operations Test Complete"
echo "==================================================="
