# Development Dockerfile with mise for reproducible environments
# Use for local development on Mac (including Apple Silicon)
#
# Build: docker build -f config/docker/mise-dev.Dockerfile -t zgents-dev .
# Run:   docker run -it --rm -v $(pwd):/workspace zgents-dev

FROM debian:bookworm-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    build-essential \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install mise
RUN curl https://mise.run | sh
ENV PATH="/root/.local/bin:$PATH"

# Verify mise installation
RUN mise --version

# =============================================================================
# Development Stage
# =============================================================================
FROM base AS development

WORKDIR /workspace

# Copy mise configuration first (for layer caching)
COPY mise.toml .
COPY agents/wasm/mise.toml agents/wasm/
COPY agents/python/mise.toml agents/python/
COPY agents/rust/mise.toml agents/rust/
COPY agents/infra/mise.toml agents/infra/

# Install tools via mise
RUN mise trust /workspace/mise.toml && \
    mise install

# Setup mise shims for non-interactive use
RUN mise activate --shims bash >> ~/.bashrc
ENV PATH="/root/.local/share/mise/shims:$PATH"

# Copy package files for dependency caching
COPY package*.json ./
COPY bun.lockb* ./

# Install Node.js dependencies
RUN mise exec -- npm install || mise exec -- bun install

# Copy Python requirements
COPY apps/fastapi/requirements.txt apps/fastapi/

# Install Python dependencies
RUN mise exec -- python -m venv .venv && \
    mise exec -- .venv/bin/pip install -r apps/fastapi/requirements.txt

# Copy rest of application
COPY . .

# Default command
CMD ["mise", "run", "dev"]

# =============================================================================
# Shims Stage (for CI/CD)
# =============================================================================
FROM base AS shims

WORKDIR /workspace

COPY mise.toml .

# Install and setup shims
RUN mise trust /workspace/mise.toml && \
    mise install && \
    mise reshim

# Export shims path
ENV PATH="/root/.local/share/mise/shims:$PATH"

# Verify shims work
RUN node --version && python --version && rustc --version

CMD ["bash"]

# =============================================================================
# Production Stage (minimal)
# =============================================================================
FROM debian:bookworm-slim AS production

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy built artifacts from development stage
COPY --from=development /workspace/dist ./dist
COPY --from=development /workspace/node_modules ./node_modules

CMD ["node", "dist/index.js"]
