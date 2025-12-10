# Web Search Integration for Research-Swarm v1.2.0

**Version**: 1.2.0
**Status**: ✅ Supported
**Models**: Claude (MCP), Gemini (Grounding), OpenRouter (various)

---

## 🌐 Overview

Research-swarm v1.2.0 supports multiple web search methods through different AI providers. You're **not limited to Perplexity** - use Google Gemini grounding, Claude with web search tools, or any OpenRouter model with search capabilities.

---

## 🎯 Web Search Methods

### Method 1: Google Gemini with Grounding (Recommended)

**Best for**: Real-time web search with Google Search integration

```bash
# Set up Gemini API key
export GOOGLE_GEMINI_API_KEY="your-gemini-key"

# Use GOALIE + Gemini for goal decomposition with grounding
npx research-swarm goal-research \
  "Latest developments in quantum computing 2024" \
  --provider gemini \
  --depth 6

# Or standard research with Gemini
npx research-swarm research researcher \
  "Current AI regulations in EU" \
  --provider gemini \
  --swarm-size 5
```

**Gemini Grounding Features**:
- ✅ Real-time Google Search integration
- ✅ Automatic source verification
- ✅ Up-to-date information (not limited by training cutoff)
- ✅ Free tier available (60 requests/minute)

---

### Method 2: Claude with Web Search MCP Tools

**Best for**: High-quality analysis with verified web sources

**Step 1: Install Web Search MCP Server**

```bash
# Install Brave Search MCP (recommended)
npm install -g @modelcontextprotocol/server-brave-search

# Or install other search MCPs
npm install -g @modelcontextprotocol/server-fetch  # For direct URL fetch
```

**Step 2: Configure MCP in research-swarm**

Create `mcp-config.json`:

```json
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-api-key"
      }
    }
  }
}
```

**Step 3: Use with research-swarm**

```bash
# Set Claude API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Enable MCP tools for web search
export MCP_CONFIG_PATH="./mcp-config.json"

# Run GOALIE research with web search
npx research-swarm goal-research \
  "Analyze recent cybersecurity threats 2024" \
  --depth 7 \
  --provider anthropic
```

**Claude Web Search Features**:
- ✅ Brave Search integration (high quality results)
- ✅ Direct URL fetching
- ✅ Content extraction and analysis
- ✅ Citation tracking

---

### Method 3: OpenRouter with Search-Capable Models

**Best for**: Cost-effective with multiple model options

```bash
# Set OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."

# Use Perplexity models via OpenRouter
npx research-swarm goal-research \
  "Tech trends in 2024" \
  --provider openrouter \
  --model "perplexity/llama-3.1-sonar-large-128k-online"

# Or use other search-capable models
npx research-swarm goal-research \
  "Climate change policies" \
  --provider openrouter \
  --model "anthropic/claude-3.5-sonnet"  # Can be configured with search
```

**OpenRouter Search Models**:
- `perplexity/llama-3.1-sonar-large-128k-online` - Built-in web search
- `perplexity/llama-3.1-sonar-small-128k-online` - Faster, lower cost
- Any model + search MCP tools

---

## 🔧 Configuration Examples

### Example 1: Gemini Grounding for Current Events

```bash
# .env file
GOOGLE_GEMINI_API_KEY=your-key
DEFAULT_PROVIDER=gemini
ENABLE_GROUNDING=true

# Use GOALIE to decompose + Gemini to execute with grounding
npx research-swarm goal-research \
  "2024 US election analysis" \
  --depth 6 \
  --time 60
```

### Example 2: Claude + Brave Search for Deep Analysis

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-...
BRAVE_API_KEY=your-brave-key
MCP_CONFIG_PATH=./mcp-config.json

# GOALIE planning + Claude research with web search
npx research-swarm goal-research \
  "Comprehensive blockchain scalability solutions" \
  --depth 8 \
  --swarm-size 7
```

### Example 3: Mixed Providers (Gemini Planning + Claude Execution)

```bash
# Use Gemini for GOALIE goal decomposition
export GOOGLE_GEMINI_API_KEY=your-key
npx research-swarm goal-decompose \
  "AI safety comprehensive analysis" \
  > plan.json

# Then use Claude for deep research on each sub-goal
export ANTHROPIC_API_KEY=sk-ant-...
for goal in $(jq -r '.subGoals[].description' plan.json); do
  npx research-swarm research researcher "$goal" \
    --depth 7 \
    --swarm-size 5
done
```

---

## 🆚 Provider Comparison

| Provider | Web Search | Cost | Speed | Quality | Best For |
|----------|-----------|------|-------|---------|----------|
| **Gemini Grounding** | ✅ Built-in | 💰 Free tier | ⚡ Fast | ⭐⭐⭐⭐ | Current events, real-time data |
| **Claude + MCP** | ✅ Via Brave | 💰💰 Moderate | ⚡⚡ Medium | ⭐⭐⭐⭐⭐ | Deep analysis, verification |
| **OpenRouter Perplexity** | ✅ Built-in | 💰 Low | ⚡⚡⚡ Very fast | ⭐⭐⭐ | Cost-effective search |
| **OpenRouter + MCP** | ✅ Via tools | 💰 Varies | ⚡⚡ Medium | ⭐⭐⭐⭐ | Flexible model choice |

---

## 📊 Complete Workflow Examples

### Workflow 1: Current Event Research with Gemini

```bash
#!/bin/bash
# research-current-events.sh

export GOOGLE_GEMINI_API_KEY="your-key"

echo "🔍 Researching current events with Gemini grounding..."

npx research-swarm goal-research \
  "Latest developments in AI regulation (2024)" \
  --provider gemini \
  --depth 6 \
  --time 30 \
  --swarm-size 5

echo "✅ Research complete with real-time web sources!"
```

### Workflow 2: Deep Analysis with Claude + Web Search

```bash
#!/bin/bash
# research-deep-web.sh

export ANTHROPIC_API_KEY="sk-ant-..."
export BRAVE_API_KEY="your-brave-key"
export MCP_CONFIG_PATH="./mcp-config.json"

echo "🔬 Deep research with Claude + Brave Search..."

# Phase 1: GOALIE decomposes the goal
npx research-swarm goal-decompose \
  "Comprehensive analysis of edge computing security" \
  --verbose > decomposition.log

# Phase 2: Execute with 7-agent swarm + web search
npx research-swarm goal-research \
  "Comprehensive analysis of edge computing security" \
  --depth 8 \
  --swarm-size 7 \
  --time 120

echo "✅ Deep analysis complete with verified web sources!"
```

### Workflow 3: Hybrid Multi-Provider Strategy

```bash
#!/bin/bash
# hybrid-research.sh

echo "🎯 Hybrid Research Strategy"
echo "Planning: Gemini | Execution: Claude with web search"
echo ""

# Step 1: Fast planning with Gemini
export GOOGLE_GEMINI_API_KEY="your-gemini-key"
npx research-swarm goal-plan \
  "AI applications in healthcare diagnostics" \
  --time 120

# Step 2: Deep execution with Claude + web search
export ANTHROPIC_API_KEY="sk-ant-..."
export BRAVE_API_KEY="your-brave-key"
npx research-swarm goal-research \
  "AI applications in healthcare diagnostics" \
  --provider anthropic \
  --depth 8 \
  --time 120

echo "✅ Hybrid strategy complete!"
```

---

## 🚀 Quick Setup Guide

### Option 1: Gemini (Easiest, Free Tier)

```bash
# 1. Get API key from Google AI Studio
# Visit: https://aistudio.google.com/app/apikey

# 2. Set environment variable
export GOOGLE_GEMINI_API_KEY="your-key"

# 3. Test with grounding
npx research-swarm goal-research \
  "Latest AI news today" \
  --provider gemini \
  --depth 5

# ✅ Automatic Google Search integration!
```

### Option 2: Claude + Brave Search (High Quality)

```bash
# 1. Get API keys
# Claude: https://console.anthropic.com/
# Brave: https://brave.com/search/api/

# 2. Create MCP config
cat > mcp-config.json << EOF
{
  "mcpServers": {
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your-brave-key"
      }
    }
  }
}
EOF

# 3. Set environment
export ANTHROPIC_API_KEY="sk-ant-..."
export MCP_CONFIG_PATH="./mcp-config.json"

# 4. Test with web search
npx research-swarm goal-research \
  "Cybersecurity trends 2024" \
  --depth 7

# ✅ Claude with verified web sources!
```

### Option 3: OpenRouter + Perplexity (Cost-Effective)

```bash
# 1. Get OpenRouter key
# Visit: https://openrouter.ai/keys

# 2. Set environment
export OPENROUTER_API_KEY="sk-or-..."

# 3. Test with Perplexity model
npx research-swarm goal-research \
  "Tech industry analysis" \
  --provider openrouter \
  --model "perplexity/llama-3.1-sonar-large-128k-online" \
  --depth 6

# ✅ Built-in web search at low cost!
```

---

## 💡 Best Practices

### When to Use Each Provider

**Use Gemini Grounding when:**
- ✅ You need current/real-time information
- ✅ Budget is constrained (free tier available)
- ✅ Speed is more important than depth
- ✅ Researching news, events, trends

**Use Claude + Web Search when:**
- ✅ You need highest quality analysis
- ✅ Deep verification is critical
- ✅ Complex reasoning required
- ✅ Research papers, technical topics

**Use OpenRouter when:**
- ✅ You want model flexibility
- ✅ Cost optimization is key
- ✅ You want to try multiple models
- ✅ Simple search tasks

---

## 🔧 Advanced Configuration

### Custom Search MCP Servers

You can build custom MCP servers for specialized search:

```javascript
// custom-search-mcp.js
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';

const server = new Server({
  name: 'custom-search',
  version: '1.0.0'
}, {
  capabilities: {
    tools: {}
  }
});

server.setRequestHandler('tools/call', async (request) => {
  if (request.params.name === 'web_search') {
    // Your custom search logic
    const results = await yourSearchAPI(request.params.arguments.query);
    return { content: [{ type: 'text', text: JSON.stringify(results) }] };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

Then use it:

```json
{
  "mcpServers": {
    "custom-search": {
      "command": "node",
      "args": ["custom-search-mcp.js"]
    }
  }
}
```

---

## 📚 Additional Resources

- **Google Gemini API**: https://ai.google.dev/docs
- **Brave Search API**: https://brave.com/search/api/
- **MCP Servers**: https://github.com/modelcontextprotocol/servers
- **OpenRouter Models**: https://openrouter.ai/models
- **Research-Swarm Docs**: ./README.md
- **GOALIE Docs**: https://www.npmjs.com/package/goalie

---

## ✅ Summary

Research-swarm v1.2.0 is **not limited to Perplexity**. You can use:

1. ✅ **Google Gemini** with automatic grounding (real-time Google Search)
2. ✅ **Claude** with web search MCP tools (Brave Search, URL fetch)
3. ✅ **OpenRouter** with 200+ models (including Perplexity, but not limited to it)
4. ✅ **Custom MCP servers** for specialized search needs
5. ✅ **Hybrid strategies** (mix providers for optimal results)

The choice is yours! Pick the provider that best fits your use case, budget, and quality requirements.

---

**Need Help?**
- Issues: https://github.com/ruvnet/agentic-flow/issues
- Tag with: `[web-search]` or `[goalie]`
