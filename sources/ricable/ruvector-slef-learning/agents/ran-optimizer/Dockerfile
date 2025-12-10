# RAN Optimizer LLM Agent - Docker Image
# Uses WasmEdge with WASI-NN GGML backend for LLM inference

# Build stage: Compile Rust to WASM
FROM rust:1.83-slim AS builder

WORKDIR /app

# Install WASM target
RUN rustup target add wasm32-wasi

# Copy source
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Build release
RUN cargo build --release --target wasm32-wasi

# Runtime stage: WasmEdge with GGML plugin
FROM wasmedge/slim-runtime:0.14.1 AS runtime

# Install GGML plugin for LLM inference
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && curl -sSf https://raw.githubusercontent.com/WasmEdge/WasmEdge/master/utils/install.sh | \
       bash -s -- --plugins wasi_nn-ggml \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy WASM binary from builder
COPY --from=builder /app/target/wasm32-wasi/release/ran-optimizer.wasm ./

# Copy configuration
COPY config/llamaedge-config.yaml ./config/

# Create models directory
RUN mkdir -p /models

# Environment variables
ENV CTX_SIZE=8192
ENV N_PREDICT=2048
ENV N_GPU_LAYERS=35
ENV TEMPERATURE=0.3
ENV TOP_P=0.9
ENV REPEAT_PENALTY=1.1
ENV STREAM_STDOUT=true
ENV MODEL_PATH=/models/model.gguf

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default entrypoint: API server mode
ENTRYPOINT ["wasmedge", "--dir", ".:."]
CMD ["--nn-preload", "default:GGML:AUTO:/models/model.gguf", \
     "ran-optimizer.wasm", "default", "interactive"]
