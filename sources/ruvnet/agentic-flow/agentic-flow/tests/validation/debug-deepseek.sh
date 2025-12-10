#!/bin/bash
# Debug DeepSeek timeout issue with verbose logging

export DEBUG=*
export LOG_LEVEL=debug

echo "==================================================="
echo "DeepSeek Timeout Debug - Verbose Logging"
echo "==================================================="
echo ""

echo "Testing DeepSeek with simple task..."
timeout 30 node dist/cli-proxy.js \
  --agent coder \
  --task "Python function: def multiply(a,b): return a*b" \
  --provider openrouter \
  --model "deepseek/deepseek-chat" \
  --max-tokens 500 \
  2>&1 | tee /tmp/deepseek-debug.log

EXIT_CODE=$?
echo ""
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 124 ]; then
  echo ""
  echo "❌ TIMEOUT OCCURRED"
  echo ""
  echo "==================================================="
  echo "Analyzing logs for timeout cause..."
  echo "==================================================="
  echo ""

  echo "1. Check if request was sent:"
  grep "CONVERTED OPENAI REQUEST" /tmp/deepseek-debug.log | head -3
  echo ""

  echo "2. Check if response was received:"
  grep "OPENROUTER RESPONSE RECEIVED\|RAW OPENAI RESPONSE" /tmp/deepseek-debug.log | head -5
  echo ""

  echo "3. Check for errors:"
  grep -i "error\|ERROR\|fail" /tmp/deepseek-debug.log | head -10
  echo ""

  echo "4. Check last logs before timeout:"
  tail -30 /tmp/deepseek-debug.log | grep -E "INFO|ERROR|WARN"
else
  echo "✅ Completed successfully!"
  echo ""
  echo "Output:"
  grep -A10 "Completed!" /tmp/deepseek-debug.log
fi

echo ""
echo "==================================================="
echo "Full logs saved to: /tmp/deepseek-debug.log"
echo "==================================================="
