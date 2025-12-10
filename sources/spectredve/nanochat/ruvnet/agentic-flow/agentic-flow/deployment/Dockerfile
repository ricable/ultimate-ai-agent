FROM node:20-slim

# Avoid interactive tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal deps if you later add git or build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY package.json package-lock.json* pnpm-lock.yaml* yarn.lock* ./
RUN npm ci --no-audit --no-fund

COPY config ./config
COPY src ./src

RUN npm run build

# Install Claude Code CLI globally (required by @anthropic-ai/claude-agent-sdk)
RUN npm install -g @anthropic-ai/claude-code && \
    ln -s /usr/local/lib/node_modules/@anthropic-ai/claude-code/cli.js /usr/local/bin/claude-code && \
    chmod +x /usr/local/bin/claude-code

# Initialize claude-flow (MCP server will start at runtime)
RUN npx claude-flow init --force || true

# Copy .claude/agents directory for agent definitions
# Note: Build context must be set to /workspaces/flow-cloud for this to work
# docker build -f docker/claude-agent-sdk/Dockerfile -t claude-agents:cli .
COPY .claude/agents /app/.claude/agents
COPY deployment/.claude-settings.json /app/.claude/settings.local.json

# Runtime env
ENV NODE_ENV=production
# Required by Anthropic
ENV ANTHROPIC_API_KEY=""
ENV OPENROUTER_API_KEY=""
# Optional task knobs
ENV TOPIC="payments mandates"
ENV DIFF="refactor: route validation"
ENV DATASET="transactions last 30 days"

# Create writable workspace with proper permissions
RUN mkdir -p /workspace && chmod 777 /workspace

# Support CLI arguments
# Use proxy CLI for OpenRouter support
WORKDIR /app
ENTRYPOINT ["node", "dist/cli-proxy.js"]
CMD []
