# Python Local AI Agent

A FastAPI-based AI agent with web search capabilities, compatible with Open WebUI and featuring OpenAI-compatible endpoints. This agent can work with any OpenAI-compatible LLM provider (OpenAI, Ollama, etc.) and includes SearXNG integration for web search.

**Note**: This Python implementation provides the same functionality as the n8n agent workflow (`n8n_local_ai_agent.json`). Both agents:
- Use the same database table (`n8n_chat_histories`) for conversation storage
- Provide the same API endpoint structure for Open WebUI compatibility
- Support web search through SearXNG
- Handle both regular chat messages and metadata requests

Choose this Python version if you prefer a standalone service or need more programmatic control. Use the n8n version if you prefer a visual workflow approach.

## Features

- 🤖 Pydantic AI agent with conversation history
- 🔍 Web search integration via SearXNG
- 💾 Conversation persistence with Supabase
- 🔐 Bearer token authentication
- 🐳 Docker support with local-ai network integration
- 🔄 OpenAI-compatible (works with Ollama, OpenAI, etc.)

## Prerequisites

- Python 3.11+
- Supabase project (for conversation history)
- SearXNG instance (for web search)
- Ollama or OpenAI API access

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/coleam00/ottomator-agents.git
cd ottomator-agents/python-local-ai-agent
```

### 2. Set up virtual environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your values:
# - LLM_BASE_URL: Your OpenAI-compatible endpoint (e.g., http://localhost:11434/v1 for Ollama)
# - LLM_API_KEY: API key (use "ollama" for Ollama)
# - LLM_CHOICE: Model name (e.g., qwen3:14b)
# - SUPABASE_URL: Your Supabase project URL
# - SUPABASE_SERVICE_KEY: Your Supabase service key
# - SEARXNG_BASE_URL: SearXNG endpoint (http://localhost:8081 or http://searxng:8080 in Docker)
# - BEARER_TOKEN: Your chosen bearer token for API authentication
```

### 4. Set up database

**Important**: Only run this SQL if you haven't already set up the n8n agent (from `n8n_local_ai_agent.json`), as they share the same database table.

```sql
-- Run this in your Supabase SQL editor
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE n8n_chat_histories (
    id uuid DEFAULT gen_random_uuid() PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    session_id TEXT NOT NULL,
    message JSONB NOT NULL
);

CREATE INDEX idx_messages_session_id ON n8n_chat_histories(session_id);
CREATE INDEX idx_messages_created_at ON n8n_chat_histories(created_at);
```

## Running the Agent

### Local Development

```bash
# Activate virtual environment if not already active
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Run the FastAPI server
python main.py
```

The API will be available at `http://localhost:8055`

### Docker with local AI Compose Stack

You can run this agent alongside your local AI  stack:

#### Option 1: Combine with the Local AI Package

You can take the contents of `docker-compose.yml` and put them within the Docker Compose stack for the local AI package.

#### Option 2: Run Separately but Connected

```bash
# From the python-local-ai-agent directory
# Make sure the localai network exists (created by your main local AI stack)
docker compose -p localai up -d --build python-local-ai-agent
```

This approach:
- Uses the external `localai` network to connect to your existing services
- Doesn't require any changes to your main docker-compose.yml
- Can be started/stopped independently

**Note**: 
- Ensure your main local-ai stack is running first (so the network exists)
- Set environment variables in this directory's `.env` file
- The agent will be accessible at `http://localhost:8055`
- It can communicate with all services in the localai network (Ollama at `http://ollama:11434`, SearXNG at `http://searxng:8080`, etc.)

## API Usage

### Endpoint: POST `/invoke-python-agent`

Send chat messages to the agent:

```bash
curl -X POST http://localhost:8055/invoke-python-agent \
  -H "Authorization: Bearer YOUR_BEARER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "chatInput": "What is the latest news about AI?",
    "sessionId": "user-123"
  }'
```

## OpenAI Compatible Demo

The project includes a demo script showing how to use OpenAI's Python client with both OpenAI and Ollama:

### Quick Start

```bash
# Make sure you're in the virtual environment
source venv/bin/activate  # Linux/Mac

# Run the demo
python openai_compatible_demo.py
```

The demo will:
1. Show available providers (OpenAI if API key is set, Ollama if configured)
2. Let you choose which provider to use
3. Demonstrate:
   - Basic completions
   - Streaming responses
   - Multi-turn conversations

### Configuration for Demo

In your `.env` file:
- For Ollama: Set `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_CHOICE`
- For OpenAI: Set `OPENAI_API_KEY`

## Integration with Open WebUI

This agent is designed to work with Open WebUI functions. Use the provided endpoint URL and bearer token in your Open WebUI function configuration.

## Troubleshooting

1. **Bearer token errors**: Ensure `BEARER_TOKEN` in `.env` has no quotes or extra spaces
2. **Database connection**: Verify Supabase credentials and that the table exists
3. **SearXNG connection**: Check that SearXNG is running and accessible at the configured URL
4. **Ollama connection**: Ensure Ollama is running (`ollama serve`) and the model is pulled

## Development

To modify the agent:
1. Edit `main.py` for core functionality
2. Adjust the system prompt in the agent definition
3. Add new tools by creating tool functions decorated with `@agent.tool`
4. Update dependencies in `requirements.txt` as needed