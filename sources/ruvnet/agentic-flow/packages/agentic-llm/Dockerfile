# GPU-enabled Agentic LLM Training System for Phi-4 Optimization
# Optimized for Claude SDK and MCP Tools

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install core dependencies
RUN pip3 install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install ONNX Runtime GPU
RUN pip3 install onnxruntime-gpu

# Install training and optimization dependencies
RUN pip3 install \
    transformers>=4.38.0 \
    datasets>=2.16.0 \
    accelerate>=0.26.0 \
    bitsandbytes>=0.42.0 \
    peft>=0.8.0 \
    trl>=0.7.10 \
    wandb>=0.16.0 \
    tensorboard>=2.15.0 \
    sentencepiece>=0.1.99 \
    protobuf>=4.25.0 \
    onnx>=1.15.0 \
    optimum>=1.16.0 \
    auto-gptq>=0.6.0 \
    scikit-learn>=1.4.0 \
    numpy>=1.24.0 \
    pandas>=2.1.0 \
    matplotlib>=3.8.0 \
    seaborn>=0.13.0 \
    tqdm>=4.66.0 \
    einops>=0.7.0

# Install evaluation and benchmarking tools
RUN pip3 install \
    rouge-score>=0.1.2 \
    bert-score>=0.3.13 \
    evaluate>=0.4.1 \
    sacrebleu>=2.4.0

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app/

# Create necessary directories
RUN mkdir -p /app/models /app/checkpoints /app/logs /app/results

# Set permissions
RUN chmod +x /app/training/*.py /app/deployment/*.sh || true

# Copy and configure health server entrypoint
COPY cloudrun/entrypoint.sh /app/cloudrun/entrypoint.sh
COPY cloudrun/health_server.py /app/cloudrun/health_server.py
RUN chmod +x /app/cloudrun/entrypoint.sh

# Install Flask for health server
RUN pip3 install flask

# Start via entrypoint - health server first, then training in background
CMD ["/app/cloudrun/entrypoint.sh"]
