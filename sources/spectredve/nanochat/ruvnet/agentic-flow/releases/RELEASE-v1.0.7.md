# Release Notes - agentic-flow v1.0.7

**Published:** October 4, 2025
**Package:** https://www.npmjs.com/package/agentic-flow

---

## 🎉 Major Features

### 1. Comprehensive MCP Tool Configuration

The `agentic_flow_agent` MCP tool now exposes **all configuration options** available in the CLI, making it possible to fully control agent behavior through MCP clients like Claude Desktop.

**New Configuration Parameters:**

#### Provider Configuration
- ✅ **model** - Specify any model (Anthropic, OpenRouter, ONNX)
  - Examples: `claude-sonnet-4-5-20250929`, `meta-llama/llama-3.1-8b-instruct`, `Xenova/gpt2`
- ✅ **provider** - Choose provider (anthropic/openrouter/onnx)
  - `anthropic` - Highest quality (default)
  - `openrouter` - 99% cost savings
  - `onnx` - Free local inference

#### API Configuration
- ✅ **anthropicApiKey** - Override environment ANTHROPIC_API_KEY
- ✅ **openrouterApiKey** - Override environment OPENROUTER_API_KEY

#### Agent Behavior
- ✅ **stream** - Enable real-time streaming output
- ✅ **temperature** - Control randomness (0.0-1.0)
- ✅ **maxTokens** - Limit response length

#### Directory Configuration
- ✅ **agentsDir** - Use custom agent definitions

#### Output Options
- ✅ **outputFormat** - Choose text/json/markdown
- ✅ **verbose** - Enable debug logging

#### Execution Control
- ✅ **timeout** - Custom timeout in milliseconds
- ✅ **retryOnError** - Auto-retry on transient errors

**Example Usage in Claude Desktop:**

```typescript
// Use OpenRouter for cost savings
{
  "agent": "coder",
  "task": "Write a Python function to parse CSV",
  "provider": "openrouter",
  "model": "meta-llama/llama-3.1-8b-instruct"
}

// Use custom temperature and output
{
  "agent": "researcher",
  "task": "Research quantum computing",
  "temperature": 0.7,
  "outputFormat": "markdown",
  "maxTokens": 2000
}

// Override API key temporarily
{
  "agent": "analyst",
  "task": "Analyze sales data",
  "anthropicApiKey": "sk-ant-api03-temp-key"
}
```

---

### 2. Enhanced Tool Descriptions

All parameters now include comprehensive descriptions that explain:
- **What the parameter does**
- **Valid values and examples**
- **Cost/performance implications**
- **Default behavior**

This makes the MCP tool self-documenting and easy to use directly from Claude Desktop's tool interface.

---

## 🔧 Improvements

### MCP Tool Usability
- **Self-Documenting** - Every parameter has detailed descriptions
- **Provider Switching** - Easy to switch between Anthropic/OpenRouter/ONNX
- **API Key Flexibility** - Override keys per-request without changing .env
- **Output Control** - Choose format based on use case
- **Error Handling** - Auto-retry option for resilience

### Developer Experience
- **Comprehensive Configuration** - All CLI options available via MCP
- **Clear Documentation** - Parameter descriptions guide usage
- **Flexible Execution** - Fine-tune behavior per request

---

## 📦 What's Included

### MCP Tool Parameters (agentic_flow_agent)

```typescript
{
  // Core (Required)
  agent: string,        // Agent type - see agentic_flow_list_agents
  task: string,         // Task description

  // Provider Configuration (Optional)
  model: string,        // Model to use
  provider: 'anthropic' | 'openrouter' | 'onnx',

  // API Keys (Optional - override environment)
  anthropicApiKey: string,
  openrouterApiKey: string,

  // Agent Behavior (Optional)
  stream: boolean,      // Default: false
  temperature: number,  // 0.0-1.0
  maxTokens: number,    // Default: 4096

  // Directories (Optional)
  agentsDir: string,    // Custom agents directory

  // Output (Optional)
  outputFormat: 'text' | 'json' | 'markdown',
  verbose: boolean,     // Default: false

  // Execution (Optional)
  timeout: number,      // Milliseconds, default: 300000
  retryOnError: boolean // Default: false
}
```

### MCP Configuration in Claude Desktop

```json
{
  "mcpServers": {
    "agentic-flow": {
      "command": "npx",
      "args": ["-y", "agentic-flow", "mcp", "start"]
    }
  }
}
```

---

## 🧪 Testing

### Docker Validation

Create a Docker test to validate the new configuration options:

```dockerfile
# docker/Dockerfile.test-mcp-config
FROM node:22-slim
WORKDIR /app

COPY agentic-flow/package.json ./
COPY agentic-flow/dist ./dist
COPY agentic-flow/.claude ./.claude

RUN npm install --omit=dev && npm link

ENV ANTHROPIC_API_KEY=sk-ant-test
ENV OPENROUTER_API_KEY=sk-or-test

CMD ["sh", "-c", "\
echo '1️⃣ Testing provider switching...' && \
node -e \"const { execSync } = require('child_process'); \
const result = execSync('npx --yes agentic-flow --agent coder --task \"test\" --provider openrouter --model \"meta-llama/llama-3.1-8b-instruct\"', { encoding: 'utf-8' }); \
console.log('Provider switch test passed');\" && \
echo '2️⃣ Testing temperature parameter...' && \
node -e \"const { execSync } = require('child_process'); \
const result = execSync('npx --yes agentic-flow --agent coder --task \"test\" --temperature 0.5', { encoding: 'utf-8' }); \
console.log('Temperature parameter test passed');\" && \
echo '3️⃣ Testing output format...' && \
node -e \"const { execSync } = require('child_process'); \
const result = execSync('npx --yes agentic-flow --agent coder --task \"test\" --output json', { encoding: 'utf-8' }); \
console.log('Output format test passed');\" && \
echo '✅ All MCP configuration tests passed!' \
"]
```

**Run Tests:**
```bash
cd /workspaces/agentic-flow
docker build -f docker/Dockerfile.test-mcp-config -t agentic-flow-mcp-config .
docker run --rm agentic-flow-mcp-config
```

---

## 🚀 Upgrade Guide

### From v1.0.6:

```bash
# Update globally
npm install -g agentic-flow@latest

# Or use with npx (always latest)
npx agentic-flow@latest --agent coder --task "Your task"
```

### Breaking Changes
**None** - Fully backward compatible

All new parameters are optional and use sensible defaults.

### New Features Available
1. **Provider switching in MCP tools** - Switch between Anthropic/OpenRouter/ONNX
2. **API key overrides** - Temporarily use different keys per request
3. **Output format control** - Choose text/json/markdown output
4. **Temperature control** - Fine-tune creativity vs. consistency
5. **Execution control** - Custom timeouts and retry logic

---

## 📊 Package Stats

- **Version:** 1.0.7
- **MCP Tools:** 2 core tools with comprehensive configuration
  - `agentic_flow_agent` - Execute agents with full configuration
  - `agentic_flow_list_agents` - List 66+ available agents
- **Configuration Options:** 13 parameters for fine-grained control
- **Providers Supported:** 3 (Anthropic, OpenRouter, ONNX)

---

## 🐛 Bug Fixes

**None** - This is a feature enhancement release

---

## 📚 Documentation Updates

### MCP Tool Documentation

The `agentic_flow_agent` tool now includes comprehensive parameter documentation:

**Provider Configuration:**
- Detailed descriptions of each provider (cost, quality, requirements)
- Example model names for each provider
- API key format validation

**Agent Behavior:**
- Temperature explanation (deterministic vs. creative)
- Token limit guidance
- Streaming vs. non-streaming trade-offs

**Output Options:**
- When to use each format (text/json/markdown)
- Verbose logging for debugging

---

## 🔗 Links

- **npm Package:** https://www.npmjs.com/package/agentic-flow
- **GitHub:** https://github.com/ruvnet/agentic-flow
- **Documentation:** See `docs/` directory
- **Issues:** https://github.com/ruvnet/agentic-flow/issues

---

## 📝 Full Changelog

### Added
- **Provider switching** in MCP `agentic_flow_agent` tool
- **13 configuration parameters** with detailed descriptions:
  - model, provider (provider config)
  - anthropicApiKey, openrouterApiKey (API config)
  - stream, temperature, maxTokens (agent behavior)
  - agentsDir (directory config)
  - outputFormat, verbose (output options)
  - timeout, retryOnError (execution control)
- **Self-documenting tool** - All parameters include usage guidance
- **Environment variable overrides** - API keys can be specified per-request

### Changed
- MCP server version updated to 1.0.7
- Enhanced tool description with comprehensive configuration info

### Improved
- **Developer Experience** - Full control over agent behavior from MCP
- **Cost Optimization** - Easy provider switching (99% savings with OpenRouter)
- **Flexibility** - Override any setting per-request
- **Documentation** - Parameter descriptions explain all options

---

## 💡 Use Cases

### 1. Cost Optimization
Switch to OpenRouter for 99% cost savings on non-critical tasks:
```json
{
  "agent": "researcher",
  "task": "Research topic X",
  "provider": "openrouter",
  "model": "meta-llama/llama-3.1-8b-instruct"
}
```

### 2. Quality Control
Use high temperature for creative tasks, low for consistency:
```json
{
  "agent": "coder",
  "task": "Generate creative algorithm",
  "temperature": 0.9
}
```

### 3. Output Integration
Get JSON for programmatic processing:
```json
{
  "agent": "analyst",
  "task": "Analyze data",
  "outputFormat": "json"
}
```

### 4. Temporary API Keys
Test with different accounts without changing .env:
```json
{
  "agent": "coder",
  "task": "Test task",
  "anthropicApiKey": "sk-ant-temp-test-key"
}
```

### 5. Local Inference
Use ONNX for free local processing:
```json
{
  "agent": "researcher",
  "task": "Summarize document",
  "provider": "onnx",
  "model": "Xenova/gpt2"
}
```

---

**Upgrade today for complete MCP tool control!**

```bash
npm install -g agentic-flow@latest
```
