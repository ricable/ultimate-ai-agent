# Multi-stage Dockerfile for Climate Prediction API
# Optimized for minimal image size (~50-100MB) and fast builds

# Stage 1: Builder
FROM rust:1.75-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    musl-tools \
    && rm -rf /var/lib/apt/lists/*

# Add musl target for static linking
RUN rustup target add x86_64-unknown-linux-musl

# Create app directory
WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY climate-api/Cargo.toml ./climate-api/
COPY climate-core/Cargo.toml ./climate-core/
COPY climate-models/Cargo.toml ./climate-models/

# Create dummy source files to cache dependencies
RUN mkdir -p climate-api/src climate-core/src climate-models/src && \
    echo "fn main() {}" > climate-api/src/main.rs && \
    echo "pub fn dummy() {}" > climate-core/src/lib.rs && \
    echo "pub fn dummy() {}" > climate-models/src/lib.rs

# Build dependencies (cached layer)
RUN cargo build --release --target x86_64-unknown-linux-musl && \
    rm -rf target/x86_64-unknown-linux-musl/release/deps/climate*

# Copy actual source code
COPY climate-api ./climate-api
COPY climate-core ./climate-core
COPY climate-models ./climate-models

# Build the application
RUN cargo build --release --target x86_64-unknown-linux-musl --workspace

# Strip debug symbols to reduce binary size
RUN strip target/x86_64-unknown-linux-musl/release/climate-api

# Stage 2: Runtime
FROM alpine:3.19 AS runtime

# Install runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    tzdata \
    && addgroup -g 1000 climate \
    && adduser -D -u 1000 -G climate climate

# Copy binary from builder
COPY --from=builder /app/target/x86_64-unknown-linux-musl/release/climate-api /usr/local/bin/climate-api

# Set ownership
RUN chown -R climate:climate /usr/local/bin/climate-api

# Switch to non-root user
USER climate

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Set environment variables
ENV RUST_LOG=info \
    RUST_BACKTRACE=0 \
    HOST=0.0.0.0 \
    PORT=8080

# Run the application
CMD ["climate-api"]
