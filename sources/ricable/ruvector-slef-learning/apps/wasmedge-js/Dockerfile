# WasmEdge QuickJS Agent Container
# Multi-stage build for minimal runtime image

# Stage 1: Build environment
FROM wasmedge/wasmedge:latest AS builder

WORKDIR /build

# Install Node.js for npm package bundling
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci --only=production

# Copy source files
COPY src/ ./src/

# Bundle JavaScript for WasmEdge
RUN npm run build:bundle

# Stage 2: Runtime environment
FROM wasmedge/wasmedge:slim

LABEL org.opencontainers.image.title="WasmEdge QuickJS Agent"
LABEL org.opencontainers.image.description="AI Agent running ruvnet npm packages with WasmEdge"
LABEL org.opencontainers.image.vendor="ruvnet"
LABEL org.opencontainers.image.source="https://github.com/ruvnet/ruvector-slef-learning"

WORKDIR /app

# Copy wasmedge-quickjs modules
COPY --from=builder /build/node_modules/wasmedge-quickjs/modules /modules

# Copy bundled application
COPY --from=builder /build/dist/agent.js /app/agent.js

# Environment variables with defaults
ENV AGENT_ID="wasmedge-agent"
ENV AGENT_MODEL="claude-3-5-sonnet-20241022"
ENV LITELLM_ENDPOINT="http://litellm:4000"
ENV RUVECTOR_ENDPOINT="http://ruvector:8765"
ENV AGENTDB_ENDPOINT="http://agentdb:8766"
ENV PORT="8080"

# WasmEdge specific settings
ENV WASMEDGE_PLUGIN_PATH="/usr/local/lib/wasmedge"

# Expose HTTP port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8080/health || exit 1

# Run with WasmEdge QuickJS
ENTRYPOINT ["wasmedge", "--dir", ".:/app", "--dir", "modules:/modules"]
CMD ["/app/agent.js"]
