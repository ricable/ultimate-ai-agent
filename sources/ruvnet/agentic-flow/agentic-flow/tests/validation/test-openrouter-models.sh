#!/bin/bash
# Test multiple OpenRouter models

echo "==================================================="
echo "OpenRouter Models Compatibility Test"
echo "==================================================="
echo ""

RESULTS_FILE="/tmp/openrouter-model-results.md"

echo "# OpenRouter Model Test Results" > $RESULTS_FILE
echo "**Date:** $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE
echo "| Model | Status | Time | Output Quality |" >> $RESULTS_FILE
echo "|-------|--------|------|----------------|" >> $RESULTS_FILE

test_model() {
  local MODEL=$1
  local NAME=$2

  echo "Testing: $NAME ($MODEL)"
  echo "---------------------------------------------------"

  START=$(date +%s)
  timeout 20 node dist/cli-proxy.js \
    --agent coder \
    --task "Write Python: def add(a,b): return a+b" \
    --provider openrouter \
    --model "$MODEL" \
    --max-tokens 200 \
    > /tmp/model-test-output.txt 2>&1

  EXIT_CODE=$?
  END=$(date +%s)
  DURATION=$((END - START))

  if [ $EXIT_CODE -eq 124 ]; then
    echo "❌ TIMEOUT ($DURATION"s")"
    echo "| $NAME | ❌ Timeout | ${DURATION}s | - |" >> $RESULTS_FILE
  elif [ $EXIT_CODE -eq 0 ]; then
    if grep -q "def add" /tmp/model-test-output.txt; then
      echo "✅ SUCCESS ($DURATION"s")"
      echo "| $NAME | ✅ Working | ${DURATION}s | Good |" >> $RESULTS_FILE
    else
      echo "⚠️  Completed but no code ($DURATION"s")"
      echo "| $NAME | ⚠️ Partial | ${DURATION}s | Poor |" >> $RESULTS_FILE
    fi
  else
    echo "❌ ERROR: Exit code $EXIT_CODE ($DURATION"s")"
    echo "| $NAME | ❌ Error | ${DURATION}s | - |" >> $RESULTS_FILE
  fi
  echo ""
}

# Test OpenAI models via OpenRouter
test_model "openai/gpt-4o-mini" "GPT-4o-mini"
test_model "openai/gpt-3.5-turbo" "GPT-3.5-turbo"

# Test Meta models
test_model "meta-llama/llama-3.3-70b-instruct" "Llama 3.3 70B"
test_model "meta-llama/llama-3.1-8b-instruct" "Llama 3.1 8B"

# Test Anthropic via OpenRouter
test_model "anthropic/claude-3.5-sonnet" "Claude 3.5 Sonnet"

# Test Mistral models
test_model "mistralai/mistral-7b-instruct" "Mistral 7B"

# Test Google models
test_model "google/gemini-2.0-flash-exp" "Gemini 2.0 Flash"

echo "==================================================="
echo "Test Complete!"
echo "==================================================="
echo ""
cat $RESULTS_FILE
echo ""
echo "Results saved to: $RESULTS_FILE"
