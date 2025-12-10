# Agentic-Flow Docker Test Instance

Complete Docker setup for testing agentic-flow in an isolated environment.

## Quick Start

### 1. Setup Environment Variables

```bash
cd agentic-flow/docker/test-instance
cp .env.example .env
# Edit .env and add your API keys
```

### 2. Build and Run

```bash
# Build the Docker image
docker-compose build

# Start the container (interactive mode)
docker-compose up -d

# Enter the container
docker-compose exec agentic-flow /bin/bash
```

### 3. Run Tests

Inside the container:
```bash
# Run automated test suite
/app/docker/test-instance/test-runner.sh

# Or test commands manually
node /app/dist/cli.js --help
node /app/dist/cli.js list-agents
node /app/dist/cli.js sparc modes
```

## Manual Testing Commands

### Basic CLI Commands

```bash
# Show help
node /app/dist/cli.js --help

# Check version
node /app/dist/cli.js --version

# List all available agents
node /app/dist/cli.js list-agents

# List SPARC modes
node /app/dist/cli.js sparc modes
```

### Agent Execution

```bash
# Run a simple agent task (requires API key)
node /app/dist/cli.js agent researcher \
  --task "Explain the benefits of TypeScript" \
  --max-tokens 200

# Use local ONNX model (no API key needed)
node /app/dist/cli.js agent coder \
  --provider onnx \
  --model "Xenova/gpt2" \
  --task "Write a hello world function"
```

### SPARC Workflow

```bash
# Run specification mode
node /app/dist/cli.js sparc run spec-pseudocode \
  "Create a REST API for user authentication"

# Run architecture mode
node /app/dist/cli.js sparc run architect \
  "Design a microservices architecture"

# Full TDD workflow
node /app/dist/cli.js sparc tdd \
  "Implement user registration with email verification"
```

### AgentDB and Memory

```bash
# View AgentDB statistics
node /app/dist/cli.js agentdb stats

# Store a pattern
node /app/dist/cli.js pattern store \
  --session test-session \
  --task "implement authentication" \
  --reward 0.95 \
  --success true

# Search for patterns
node /app/dist/cli.js pattern search \
  --task "authentication" \
  --k 5

# Memory operations
node /app/dist/cli.js memory store my-key "my-value"
node /app/dist/cli.js memory retrieve my-key
```

## Environment Variables

### Required (at least one API key)

- `ANTHROPIC_API_KEY` - For Claude models (recommended)
- `OPENROUTER_API_KEY` - For 99% cost savings
- `GOOGLE_GEMINI_API_KEY` - For free tier access

### Optional Configuration

- `DEFAULT_PROVIDER` - anthropic | openrouter | gemini | onnx (default: anthropic)
- `DEFAULT_MODEL` - Model to use (provider-specific)
- `MAX_TOKENS` - Maximum response tokens (default: 4096)
- `TEMPERATURE` - Sampling temperature (default: 0.7)
- `ENABLE_STREAMING` - Enable streaming responses (default: true)
- `ENABLE_HOOKS` - Enable hooks system (default: true)
- `VERBOSE` - Verbose logging (default: false)

## Data Persistence

Data is persisted in Docker volumes:

- `agentdb-data` - AgentDB vector database
- `memory-data` - Memory storage
- `session-data` - Session information

### Access persisted data

```bash
# View volume contents
docker volume inspect agentic-flow_agentdb-data
docker volume inspect agentic-flow_memory-data

# Backup volumes
docker run --rm -v agentic-flow_agentdb-data:/data -v $(pwd):/backup \
  alpine tar czf /backup/agentdb-backup.tar.gz /data
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs agentic-flow

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### API key not working

```bash
# Verify environment variables inside container
docker-compose exec agentic-flow env | grep API_KEY

# Check if .env file is loaded
docker-compose config
```

### WASM modules missing

```bash
# Verify WASM files exist
docker-compose exec agentic-flow ls -la /app/wasm/reasoningbank/

# Rebuild if missing
docker-compose build --no-cache
```

## Architecture

```
agentic-flow-docker/
├── Dockerfile              # Build configuration
├── docker-compose.yml      # Service orchestration
├── .env.example           # Environment template
├── test-runner.sh         # Automated test suite
├── README.md             # This file
└── workspace/            # Optional local workspace mount
```

## Advanced Usage

### Use ONNX (Local, No API Key Required)

```bash
node /app/dist/cli.js agent coder \
  --provider onnx \
  --model "Xenova/gpt2" \
  --task "Write a function to reverse a string"
```

### Custom Agent Creation

```bash
node /app/dist/cli.js create-agent \
  --name my-custom-agent \
  --description "Custom agent for specific tasks" \
  --system-prompt "You are a specialized agent..."
```

### Batch Processing

```bash
# Create tasks file
cat > /tmp/tasks.txt << EOF
Task 1: Analyze code quality
Task 2: Write unit tests
Task 3: Generate documentation
EOF

# Run batch processing
node /app/dist/cli.js sparc concurrent coder /tmp/tasks.txt
```

## Testing Without API Keys

You can test basic functionality without API keys:

```bash
# CLI commands (no API key needed)
node /app/dist/cli.js --help
node /app/dist/cli.js list-agents
node /app/dist/cli.js sparc modes
node /app/dist/cli.js agentdb stats

# Local ONNX execution (no API key needed)
node /app/dist/cli.js agent researcher \
  --provider onnx \
  --model "Xenova/gpt2" \
  --task "Hello world"
```

## Cleanup

```bash
# Stop and remove container
docker-compose down

# Remove with volumes (deletes all data)
docker-compose down -v

# Remove images
docker rmi agentic-flow-test
```

## Support

For issues or questions:
- GitHub: https://github.com/ruvnet/agentic-flow
- Issues: https://github.com/ruvnet/agentic-flow/issues
