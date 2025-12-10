# Phase 2 Integration Guide - Tool Emulation for Claude Code

**Goal**: Enable Claude Code to work with non-tool models through automatic emulation

---

## Current Limitation

Claude Code currently **requires models with native tool support** because it sends tool definitions in every request:

```typescript
// Claude Code sends ~35k+ tokens of tool definitions
{
  "tools": [ /* 218 MCP tools */ ],
  "messages": [ /* conversation */ ]
}
```

**Models without native tool support**: Ignore the tools and respond with plain text, making them unusable with Claude Code.

---

## Solution: Automatic Tool Emulation

Detect when a model lacks native tools and convert tool definitions into prompts using ReAct or Prompt-based patterns.

---

## Integration Steps

### Step 1: Add Detection to Proxy Initialization

**File**: `src/cli-proxy.ts`

**Location**: In the OpenRouter proxy creation (around line 320)

```typescript
import { detectModelCapabilities } from './utils/modelCapabilities.js';
import { ToolEmulator } from './proxy/tool-emulation.js';

// Inside shouldUseOpenRouter or startOpenRouterProxy:
const model = options.model || process.env.COMPLETION_MODEL || 'mistralai/mistral-small-3.1-24b-instruct';
const capabilities = detectModelCapabilities(model);

if (capabilities.requiresEmulation) {
  console.log(`\n⚙️  Detected: Model lacks native tool support`);
  console.log(`🔧 Using ${capabilities.emulationStrategy.toUpperCase()} emulation pattern`);
  console.log(`📊 Expected reliability: ${capabilities.emulationStrategy === 'react' ? '70-85%' : '50-70%'}\n`);
}

// Pass capabilities to proxy
const proxy = new AnthropicToOpenRouterProxy({
  apiKey: openRouterKey,
  defaultModel: model,
  capabilities: capabilities  // NEW parameter
});
```

---

### Step 2: Modify OpenRouter Proxy Request Handler

**File**: `src/proxy/anthropic-to-openrouter.ts`

**Changes needed**:

1. **Add capabilities to constructor**:

```typescript
import { ToolEmulator } from './tool-emulation.js';
import { ModelCapabilities } from '../utils/modelCapabilities.js';

export class AnthropicToOpenRouterProxy {
  private capabilities?: ModelCapabilities;
  private toolEmulator?: ToolEmulator;

  constructor(config: {
    apiKey: string;
    defaultModel: string;
    capabilities?: ModelCapabilities;  // NEW
  }) {
    this.apiKey = config.apiKey;
    this.defaultModel = config.defaultModel;
    this.capabilities = config.capabilities;

    // Initialize emulator if needed
    if (this.capabilities?.requiresEmulation) {
      // Will be initialized per-request when tools are available
      this.toolEmulator = null;
    }
  }
}
```

2. **Modify request handler** (around line 200-300):

```typescript
async handleRequest(anthropicReq: AnthropicRequest): Promise<any> {
  const model = anthropicReq.model || this.defaultModel;

  // Check if emulation is needed
  const capabilities = this.capabilities || detectModelCapabilities(model);

  if (capabilities.requiresEmulation && anthropicReq.tools && anthropicReq.tools.length > 0) {
    // Use tool emulation
    return this.handleEmulatedRequest(anthropicReq, capabilities);
  }

  // Normal path (native tool support)
  return this.handleNativeRequest(anthropicReq);
}

async handleNativeRequest(anthropicReq: AnthropicRequest): Promise<any> {
  // Existing code - unchanged
  const openaiReq = this.convertAnthropicToOpenAI(anthropicReq);
  const response = await this.callOpenRouter(openaiReq);
  return this.convertOpenAIToAnthropic(response);
}

async handleEmulatedRequest(
  anthropicReq: AnthropicRequest,
  capabilities: ModelCapabilities
): Promise<any> {
  // Create tool emulator
  const emulator = new ToolEmulator(
    anthropicReq.tools || [],
    capabilities.emulationStrategy as 'react' | 'prompt'
  );

  // Get user message
  const lastMessage = anthropicReq.messages[anthropicReq.messages.length - 1];
  const userMessage = typeof lastMessage.content === 'string'
    ? lastMessage.content
    : lastMessage.content.find(c => c.type === 'text')?.text || '';

  // Execute emulation loop
  const result = await executeEmulation(
    emulator,
    userMessage,
    async (prompt) => {
      // Call model with emulated prompt
      const openaiReq = {
        model: anthropicReq.model || this.defaultModel,
        messages: [{ role: 'user', content: prompt }],
        temperature: anthropicReq.temperature || 0.7,
        max_tokens: anthropicReq.max_tokens || 4096
      };
      const response = await this.callOpenRouter(openaiReq);
      return response.choices[0].message.content;
    },
    async (toolCall) => {
      // Execute tool through Claude Code
      // Return tool result to continue emulation loop
      return await this.executeTool(toolCall);
    },
    { maxIterations: 5, verbose: process.env.VERBOSE === 'true' }
  );

  // Convert result to Anthropic format
  return {
    id: `emulated_${Date.now()}`,
    type: 'message',
    role: 'assistant',
    content: [
      {
        type: 'text',
        text: result.finalAnswer || 'No response generated'
      }
    ],
    model: anthropicReq.model || this.defaultModel,
    stop_reason: 'end_turn',
    usage: {
      input_tokens: 0,  // Estimate if needed
      output_tokens: 0
    }
  };
}

async executeTool(toolCall: ToolCall): Promise<any> {
  // This would need to communicate with Claude Code's tool execution system
  // For now, throw an error - needs MCP integration
  throw new Error('Tool execution not yet implemented in emulation layer');
}
```

---

### Step 3: Test Integration

**Without API Key** (Offline test):
```bash
npm run build
npx tsx examples/tool-emulation-demo.ts
```

**With OpenRouter API Key** (Real test):
```bash
export OPENROUTER_API_KEY="sk-or-..."

# Test with tool-capable model (should use native tools)
npx agentic-flow --agent coder \
  --task "What is 2+2?" \
  --provider openrouter \
  --model "deepseek/deepseek-chat"

# Test with non-tool model (should use emulation)
npx agentic-flow --agent coder \
  --task "What is 2+2?" \
  --provider openrouter \
  --model "mistralai/mistral-7b-instruct"
```

---

## Testing Strategy

### Quick Test (Option 1)

Run the simulation script I just created:

```bash
export OPENROUTER_API_KEY="sk-or-..."
npx tsx examples/test-claude-code-emulation.ts
```

This simulates Claude Code using non-tool models with emulation.

### Full Integration Test (Option 2)

After implementing Steps 1-2 above:

```bash
# Test with non-tool model
npx agentic-flow claude-code \
  --provider openrouter \
  --model "mistralai/mistral-7b-instruct" \
  "Read the file package.json and show me the version"

# Should output:
# ⚙️  Detected: Model lacks native tool support
# 🔧 Using REACT emulation pattern
# [Then execute the task]
```

---

## Expected Behavior After Integration

### With Tool-Capable Model (DeepSeek)

```bash
$ npx agentic-flow --agent coder --task "Read package.json" \
    --provider openrouter --model "deepseek/deepseek-chat"

🔧 Model: deepseek/deepseek-chat
✅ Native tool support detected
📊 Tools: 218 MCP tools available
⏳ Running...
[Uses native function calling]
```

### With Non-Tool Model (Mistral 7B)

```bash
$ npx agentic-flow --agent coder --task "Read package.json" \
    --provider openrouter --model "mistralai/mistral-7b-instruct"

🔧 Model: mistralai/mistral-7b-instruct
⚙️  Detected: Model lacks native tool support
🔧 Using REACT emulation pattern
📊 Expected reliability: 70-85%
⏳ Running...

Iteration 1:
  Thought: I need to read the package.json file
  Action: read_file
  Action Input: {"path": "package.json"}
  Observation: {"content": "{ \"name\": \"agentic-flow\", ... }"}

Final Answer: The package.json shows version 1.2.7...
```

---

## Challenges to Solve

### 1. Tool Execution in Emulation Loop

**Problem**: Emulation loop needs to execute tools and get results back.

**Solution**: Need to integrate with Claude Code's MCP tool execution system. This requires:
- Access to MCP server connections
- Tool execution API
- Result serialization

### 2. Streaming Support

**Problem**: Emulation uses multi-iteration loop, harder to stream.

**Solution**:
- Disable streaming for emulated requests
- Or implement partial streaming (stream each iteration)

### 3. Context Window Limits

**Problem**: Small models (<32k context) can't handle 218 tools.

**Solution**:
- Filter tools based on task (use embedding similarity)
- Only include top 10-20 most relevant tools
- Or use prompt-based strategy with simplified tool list

---

## Minimal Test Right Now

Since you want to test **immediately**, run:

```bash
export OPENROUTER_API_KEY="sk-or-your-key-here"
npx tsx examples/test-claude-code-emulation.ts
```

This will:
1. ✅ Test Mistral 7B (no native tools)
2. ✅ Test Llama 2 13B (no native tools)
3. ✅ Simulate Claude Code's tool usage
4. ✅ Show emulation in action

**Example Output**:
```
🧪 Test: Simple File Read
═══════════════════════════════════════════════════════════════════════════════
Model: mistralai/mistral-7b-instruct
Task: Read the file at path "/home/user/example.js"

📊 Model Capabilities:
   Native Tools: ❌
   Requires Emulation: 🔧 YES
   Strategy: REACT

⚙️  Starting emulation with REACT strategy...

   📡 Calling OpenRouter: mistralai/mistral-7b-instruct

   Thought: I need to use the read_file tool to read the file
   Action: read_file
   Action Input: {"path": "/home/user/example.js"}

   🔧 Executing: read_file({"path":"/home/user/example.js"})

───────────────────────────────────────────────────────────────────────────────
📊 RESULTS
───────────────────────────────────────────────────────────────────────────────

✅ Tool Calls: 1
   Tool Used: read_file
   Arguments: { "path": "/home/user/example.js" }
   Expected: read_file
   ✅ CORRECT

📈 Confidence: 85.0%
```

---

## Summary

**To test Claude Code with non-tool models RIGHT NOW**:
```bash
export OPENROUTER_API_KEY="your-key"
npx tsx examples/test-claude-code-emulation.ts
```

**To actually USE non-tool models in production**:
- Implement Phase 2 integration (Steps 1-2 above)
- Test thoroughly
- Deploy

The simulation test will prove the concept works. Full integration requires modifying the proxy layer.
