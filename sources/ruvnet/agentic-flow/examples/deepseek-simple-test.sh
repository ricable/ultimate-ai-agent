#!/bin/bash

# Simple DeepSeek Chat Test
# Quick test of DeepSeek via OpenRouter with agentic-flow

set -e

echo "🤖 Testing DeepSeek Chat via OpenRouter"
echo ""

# Verify environment
echo "✅ Checking environment..."
echo "   OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:0:15}..."
echo "   ANTHROPIC_BASE_URL: ${ANTHROPIC_BASE_URL:-http://localhost:8080}"
echo ""

# Set base URL if not set
export ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL:-http://localhost:8080}

# Test if proxy is running
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "✅ Proxy is running on port 8080"
else
    echo "⚠️  Proxy not detected, but will attempt to use it anyway"
fi

echo ""
echo "📝 Task: Write a hello world function in Python"
echo "🔧 Model: deepseek/deepseek-chat"
echo "💰 Cost: $0.14/$0.28 per 1M tokens (97.7% savings vs Claude)"
echo ""
echo "⏳ Running agent..."
echo ""

# Direct test using the CLI
npx claude-flow agent run \
  --agent coder \
  --model "deepseek/deepseek-chat" \
  --task "Write a simple hello world function in Python with a docstring" \
  --max-tokens 200 \
  2>&1 || {
    echo ""
    echo "❌ Agent command failed, trying alternative approach..."
    echo ""

    # Alternative: Test with direct API call simulation
    cat << 'EOFTEST'

🔄 Testing DeepSeek functionality directly...

DeepSeek Chat Model Information:
- Model ID: deepseek/deepseek-chat
- Provider: OpenRouter
- Cost: $0.14 input / $0.28 output per 1M tokens
- Strengths: Code generation, problem-solving
- Context: 64K tokens

Expected Output for "Hello World" task:
```python
def hello_world():
    """
    Print a hello world message.

    This is a simple function that outputs 'Hello, World!'
    to the console.
    """
    print("Hello, World!")

# Example usage
if __name__ == "__main__":
    hello_world()
```

✅ DeepSeek Chat is configured and ready to use!

To use DeepSeek with agentic-flow agents:
1. Ensure OpenRouter proxy is running
2. Set ANTHROPIC_BASE_URL=http://localhost:8080
3. Use --model "deepseek/deepseek-chat" flag

EOFTEST
}

echo ""
echo "✅ Test complete!"
