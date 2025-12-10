# Multi-stage build for Rust API Gateway
# Stage 1: Build
FROM rust:1.75-slim as builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Build dependencies only (cached layer)
RUN mkdir src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy source code
COPY src ./src

# Build application
RUN touch src/main.rs && \
    cargo build --release

# Stage 2: Runtime
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -u 1000 -m -s /bin/bash apiuser

# Copy binary from builder
COPY --from=builder /app/target/release/api-gateway /app/api-gateway

# Change ownership
RUN chown -R apiuser:apiuser /app

USER apiuser

EXPOSE 3000 9090

CMD ["/app/api-gateway"]
