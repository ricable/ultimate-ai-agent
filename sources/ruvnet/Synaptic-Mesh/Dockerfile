# Production Multi-Stage Dockerfile for Synaptic Neural Mesh
# Optimized for size, security, and performance

# ==============================================================================
# Stage 1: Build Environment
# ==============================================================================
FROM rust:1.75-slim AS rust-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    cmake \
    g++ \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install wasm-pack for WebAssembly builds
RUN curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

WORKDIR /build

# Copy Rust workspace files
COPY src/rs/ ./rs/
COPY Cargo.toml Cargo.lock ./

# Build Rust components with optimizations
RUN cd rs/QuDAG/QuDAG-main && \
    cargo build --release --bin qudag --features "cli full exchange" && \
    cargo build --release --bin qudag-exchange --manifest-path ./qudag-exchange/cli/Cargo.toml

# Build WASM modules
RUN cd rs/ruv-FANN/ruv-swarm && \
    wasm-pack build --target web --release --out-dir ../../../target/wasm/ruv-swarm

# ==============================================================================
# Stage 2: Node.js Build Environment
# ==============================================================================
FROM node:20-alpine AS node-builder

# Install build dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    sqlite-dev

WORKDIR /build

# Copy Node.js components
COPY src/js/ ./js/
COPY src/mcp/ ./mcp/
COPY package*.json ./

# Install and build Node.js components
RUN cd js/ruv-swarm && npm ci --only=production
RUN cd js/claude-flow && npm ci --only=production
RUN cd mcp && npm ci --only=production

# Clean up dev dependencies and build artifacts
RUN find . -name "node_modules" -type d -exec rm -rf {} +
RUN cd js/ruv-swarm && npm ci --only=production --no-audit
RUN cd js/claude-flow && npm ci --only=production --no-audit
RUN cd mcp && npm ci --only=production --no-audit

# ==============================================================================
# Stage 3: Production Runtime Environment
# ==============================================================================
FROM node:20-alpine AS production

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    sqlite \
    curl \
    bash \
    dumb-init \
    && addgroup -g 1000 neural \
    && adduser -D -s /bin/bash -u 1000 -G neural neural

# Install npm global packages in production
RUN npm install -g pm2

# Create application directories
RUN mkdir -p /app/bin /app/src /app/wasm /app/data /app/config /app/logs \
    && chown -R neural:neural /app

WORKDIR /app

# Copy Rust binaries from build stage
COPY --from=rust-builder /build/rs/QuDAG/QuDAG-main/target/release/qudag /app/bin/
COPY --from=rust-builder /build/rs/QuDAG/QuDAG-main/target/release/qudag-exchange /app/bin/
COPY --from=rust-builder /build/target/wasm/ /app/wasm/

# Copy Node.js application from build stage
COPY --from=node-builder /build/js/ /app/src/js/
COPY --from=node-builder /build/mcp/ /app/src/mcp/

# Copy package files and install production dependencies
COPY package*.json /app/
RUN npm ci --only=production --no-audit && npm cache clean --force

# Copy configuration and scripts
COPY docker/production/ /app/docker/
COPY scripts/ /app/scripts/

# Set permissions
RUN chmod +x /app/bin/* /app/scripts/* /app/docker/*.sh \
    && chown -R neural:neural /app

# Switch to non-root user
USER neural

# Environment variables
ENV NODE_ENV=production
ENV RUST_LOG=info,qudag=debug
ENV QUDAG_DATA_DIR=/app/data
ENV QUDAG_CONFIG_DIR=/app/config
ENV NEURAL_MESH_LOG_DIR=/app/logs
ENV PATH="/app/bin:${PATH}"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /app/docker/healthcheck.sh

# Expose ports
EXPOSE 4001 8080 8081 9090 3000

# Volume mounts
VOLUME ["/app/data", "/app/config", "/app/logs"]

# Use dumb-init for proper signal handling
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Default command with PM2 ecosystem
CMD ["/app/docker/start.sh"]