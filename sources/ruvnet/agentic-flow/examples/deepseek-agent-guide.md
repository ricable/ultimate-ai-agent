# Using DeepSeek Chat with Agentic-Flow Agents

## Overview

DeepSeek Chat is an excellent model for code generation and problem-solving, accessible via OpenRouter with **97.7% cost savings** compared to Claude direct API.

## Quick Start

### Prerequisites
```bash
# 1. Get OpenRouter API key
# Visit: https://openrouter.ai/keys

# 2. Set environment variable
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# 3. Start OpenRouter proxy
npx claude-flow proxy start --daemon

# 4. Configure base URL
export ANTHROPIC_BASE_URL=http://localhost:8080
```

### Basic Usage

```bash
# Simple code generation
npx claude-flow agent run \
  --agent coder \
  --model "deepseek/deepseek-chat" \
  --task "Create a Python function for binary search" \
  --max-tokens 500
```

## Available Agent Types

### 1. Code Generation Agents

#### **coder** - General purpose coding
```bash
npx claude-flow agent run \
  --agent coder \
  --model "deepseek/deepseek-chat" \
  --task "Create a REST API with Express and authentication" \
  --max-tokens 1000
```

**Best for:**
- Function implementations
- Algorithm development
- Code snippets
- Quick prototypes

#### **backend-dev** - Backend development specialist
```bash
npx claude-flow agent run \
  --agent backend-dev \
  --model "deepseek/deepseek-chat" \
  --task "Build a GraphQL API with PostgreSQL integration" \
  --max-tokens 1500
```

**Best for:**
- REST/GraphQL APIs
- Database design
- Authentication systems
- Microservices

#### **sparc-coder** - TDD-focused coding
```bash
npx claude-flow agent run \
  --agent sparc-coder \
  --model "deepseek/deepseek-chat" \
  --task "Implement user authentication with tests" \
  --max-tokens 2000
```

**Best for:**
- Test-driven development
- Production-ready code
- Well-documented functions
- Complete implementations

### 2. Architecture & Design Agents

#### **system-architect** - System design
```bash
npx claude-flow agent run \
  --agent system-architect \
  --model "deepseek/deepseek-chat" \
  --task "Design a scalable microservices architecture for e-commerce" \
  --max-tokens 1500
```

**Best for:**
- Architecture design
- System planning
- Component design
- Technical decisions

#### **base-template-generator** - Boilerplate generation
```bash
npx claude-flow agent run \
  --agent base-template-generator \
  --model "deepseek/deepseek-chat" \
  --task "Create React component template with TypeScript and tests" \
  --max-tokens 800
```

**Best for:**
- Project scaffolding
- Template creation
- Boilerplate code
- Starter kits

### 3. Review & Testing Agents

#### **reviewer** - Code review specialist
```bash
npx claude-flow agent run \
  --agent reviewer \
  --model "deepseek/deepseek-chat" \
  --task "Review this code and suggest improvements: [paste code]" \
  --max-tokens 1000
```

**Best for:**
- Code reviews
- Best practices
- Performance optimization
- Security analysis

#### **tester** - Test generation
```bash
npx claude-flow agent run \
  --agent tester \
  --model "deepseek/deepseek-chat" \
  --task "Create comprehensive tests for user authentication module" \
  --max-tokens 1500
```

**Best for:**
- Unit tests
- Integration tests
- Test scenarios
- Test coverage

### 4. Specialized Agents

#### **ml-developer** - Machine learning
```bash
npx claude-flow agent run \
  --agent ml-developer \
  --model "deepseek/deepseek-chat" \
  --task "Build a sentiment analysis model with scikit-learn" \
  --max-tokens 1500
```

**Best for:**
- ML model development
- Data preprocessing
- Model training
- Feature engineering

#### **mobile-dev** - Mobile development
```bash
npx claude-flow agent run \
  --agent mobile-dev \
  --model "deepseek/deepseek-chat" \
  --task "Create React Native authentication screen" \
  --max-tokens 1200
```

**Best for:**
- React Native apps
- iOS/Android development
- Mobile UI components
- Cross-platform code

#### **api-docs** - API documentation
```bash
npx claude-flow agent run \
  --agent api-docs \
  --model "deepseek/deepseek-chat" \
  --task "Generate OpenAPI spec for user management API" \
  --max-tokens 1000
```

**Best for:**
- OpenAPI/Swagger docs
- API documentation
- Endpoint descriptions
- Schema definitions

## Cost Comparison

### DeepSeek Chat vs Claude Direct

| Task Type | Tokens (In/Out) | DeepSeek Cost | Claude Cost | Savings |
|-----------|----------------|---------------|-------------|---------|
| Simple function | 200/300 | $0.000112 | $0.0051 | **97.8%** |
| REST API | 500/1000 | $0.000350 | $0.0165 | **97.9%** |
| Full application | 1000/3000 | $0.000980 | $0.0480 | **97.9%** |
| Code review | 800/500 | $0.000252 | $0.0099 | **97.5%** |

### Monthly Cost Estimates (100 tasks)

| Task Profile | DeepSeek/Month | Claude/Month | Annual Savings |
|--------------|----------------|--------------|----------------|
| Small tasks | $0.11 | $5.10 | **$59.88** |
| Medium tasks | $0.35 | $16.50 | **$193.80** |
| Large tasks | $0.98 | $48.00 | **$564.24** |

## Advanced Usage

### 1. Multi-Agent Workflow

```bash
#!/bin/bash

# Step 1: Design architecture
npx claude-flow agent run \
  --agent system-architect \
  --model "deepseek/deepseek-chat" \
  --task "Design user authentication system" \
  --max-tokens 800 > architecture.md

# Step 2: Generate code
npx claude-flow agent run \
  --agent coder \
  --model "deepseek/deepseek-chat" \
  --task "Implement $(cat architecture.md)" \
  --max-tokens 1500 > auth_system.py

# Step 3: Generate tests
npx claude-flow agent run \
  --agent tester \
  --model "deepseek/deepseek-chat" \
  --task "Create tests for $(cat auth_system.py)" \
  --max-tokens 1000 > test_auth.py

# Step 4: Review code
npx claude-flow agent run \
  --agent reviewer \
  --model "deepseek/deepseek-chat" \
  --task "Review $(cat auth_system.py)" \
  --max-tokens 800 > review.md
```

### 2. Batch Processing

```bash
#!/bin/bash

# Process multiple files
for file in src/*.py; do
  npx claude-flow agent run \
    --agent reviewer \
    --model "deepseek/deepseek-chat" \
    --task "Review and improve: $(cat $file)" \
    --max-tokens 1000 > "reviews/$(basename $file).review.md"
done
```

### 3. With Configuration File

```json
{
  "agent": "coder",
  "model": "deepseek/deepseek-chat",
  "maxTokens": 1500,
  "temperature": 0.7,
  "provider": "openrouter",
  "proxyUrl": "http://localhost:8080"
}
```

```bash
npx claude-flow agent run \
  --config config.json \
  --task "Build user registration system"
```

## Best Practices

### 1. Token Management

```bash
# Small tasks (functions, snippets)
--max-tokens 500

# Medium tasks (modules, components)
--max-tokens 1000-1500

# Large tasks (full features, systems)
--max-tokens 2000-3000
```

### 2. Task Clarity

✅ **Good:**
```bash
--task "Create a Python function that validates email addresses using regex.
Include: input validation, error handling, unit tests, and docstring."
```

❌ **Poor:**
```bash
--task "make email validator"
```

### 3. Agent Selection

| Task Type | Recommended Agent | Alternative |
|-----------|------------------|-------------|
| Quick function | `coder` | `sparc-coder` |
| API endpoint | `backend-dev` | `coder` |
| Code review | `reviewer` | `code-analyzer` |
| Tests | `tester` | `sparc-coder` |
| Architecture | `system-architect` | `planner` |
| Mobile UI | `mobile-dev` | `coder` |
| ML model | `ml-developer` | `researcher` |

### 4. Error Handling

```bash
# With retry logic
npx claude-flow agent run \
  --agent coder \
  --model "deepseek/deepseek-chat" \
  --task "..." \
  --max-tokens 1000 \
  --retry 3 \
  --timeout 60000
```

## Performance Optimization

### 1. Use Appropriate Token Limits

```bash
# Don't over-allocate tokens
# Before: --max-tokens 4000 (expensive)
# After:  --max-tokens 1000 (sufficient for most tasks)
```

### 2. Leverage Proxy Caching

```bash
# Proxy caches frequently used prompts
# Similar requests are served faster
export ENABLE_PROXY_CACHE=true
```

### 3. Parallel Agent Execution

```bash
# Run multiple agents in parallel
npx claude-flow agent run --agent coder --model deepseek/deepseek-chat --task "Task 1" &
npx claude-flow agent run --agent tester --model deepseek/deepseek-chat --task "Task 2" &
wait
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM node:18-alpine

WORKDIR /app

# Install agentic-flow
RUN npm install -g agentic-flow

# Set environment
ENV OPENROUTER_API_KEY=your-key-here
ENV ANTHROPIC_BASE_URL=http://proxy:8080
ENV MODEL=deepseek/deepseek-chat

# Start proxy and agent
CMD ["sh", "-c", "npx claude-flow proxy start --daemon && \
     npx claude-flow agent run --agent coder --model $MODEL --task \"$TASK\""]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  proxy:
    image: agentic-flow:latest
    command: npx claude-flow proxy start
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    ports:
      - "8080:8080"

  agent:
    image: agentic-flow:latest
    depends_on:
      - proxy
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      - ANTHROPIC_BASE_URL=http://proxy:8080
    command: npx claude-flow agent run --agent coder --model deepseek/deepseek-chat
```

## Troubleshooting

### 1. Proxy Not Running

```bash
# Check proxy status
npx claude-flow proxy status

# Restart proxy
npx claude-flow proxy restart

# Check logs
npx claude-flow proxy logs --follow
```

### 2. API Key Issues

```bash
# Verify key is set
echo $OPENROUTER_API_KEY | cut -c1-20

# Test key directly
curl -H "Authorization: Bearer $OPENROUTER_API_KEY" \
  https://openrouter.ai/api/v1/models
```

### 3. Model Not Found

```bash
# List available models
npx claude-flow models --provider openrouter

# Verify model name
# Correct: deepseek/deepseek-chat
# Wrong:   deepseek-chat, deepseek/chat
```

## Examples Gallery

### Example 1: FastAPI Application
```bash
npx claude-flow agent run \
  --agent backend-dev \
  --model "deepseek/deepseek-chat" \
  --task "Create a FastAPI app with:
    - User CRUD endpoints
    - JWT authentication
    - PostgreSQL integration
    - Pydantic models
    - Error handling
    - OpenAPI docs" \
  --max-tokens 2000
```

### Example 2: React Component
```bash
npx claude-flow agent run \
  --agent coder \
  --model "deepseek/deepseek-chat" \
  --task "Create a React component for user profile:
    - TypeScript interfaces
    - State management with hooks
    - Form validation
    - API integration
    - Loading/error states
    - Responsive design" \
  --max-tokens 1500
```

### Example 3: Data Pipeline
```bash
npx claude-flow agent run \
  --agent ml-developer \
  --model "deepseek/deepseek-chat" \
  --task "Build data preprocessing pipeline:
    - CSV/JSON loading
    - Missing value handling
    - Feature scaling
    - Train/test split
    - Pipeline with sklearn
    - Saving/loading functionality" \
  --max-tokens 1500
```

## Comparison with Other Models

| Feature | DeepSeek Chat | Llama 3.1 8B | Gemini Flash | Claude 3.5 |
|---------|---------------|--------------|--------------|------------|
| Cost (1M tokens) | $0.42 | $0.09 | $0.38 | $18.00 |
| Code Quality | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Context | 64K | 128K | 32K | 200K |
| Best For | Code gen | Simple tasks | Fast responses | Complex tasks |

## Resources

- **OpenRouter Models**: https://openrouter.ai/models
- **DeepSeek Docs**: https://platform.deepseek.com/docs
- **Agentic-Flow Docs**: https://github.com/ruvnet/agentic-flow
- **Agent List**: `npx claude-flow agent list`
- **Model Pricing**: https://openrouter.ai/docs/pricing

## Support

For issues or questions:
1. Check logs: `npx claude-flow proxy logs`
2. Verify configuration: `npx claude-flow proxy config`
3. Test connectivity: `npx claude-flow proxy status`
4. GitHub Issues: https://github.com/ruvnet/agentic-flow/issues

---

**Last Updated**: 2025-10-20
**Version**: 1.6.4
**Cost Savings**: 97.7% vs Claude Direct API
