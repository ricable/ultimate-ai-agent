# SAFLA GPU-Optimized Docker Image for Fly.io
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Set Python aliases
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3
RUN ln -sf /usr/bin/pip3 /usr/bin/pip

# Install NVIDIA CUDA keyring and libraries
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && apt-get update \
    && apt-get install -y \
        cuda-runtime-12-2 \
        libcudnn8 \
        libcublas-12-2 \
    && rm -rf /var/lib/apt/lists/* \
    && rm cuda-keyring_1.0-1_all.deb

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml requirements.txt ./
COPY safla/ ./safla/
COPY config/ ./config/
COPY data/ ./data/
COPY examples/ ./examples/

# Install Python dependencies (GPU optimized)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install FAISS GPU version using conda approach
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:$PATH"
RUN conda install -c conda-forge faiss-gpu -y || pip install faiss-cpu

RUN pip install --no-cache-dir -e .

# Install additional GPU optimization packages
RUN pip install --no-cache-dir \
    flash-attn \
    bitsandbytes \
    accelerate \
    optimum \
    transformers[torch] \
    sentence-transformers \
    xformers

# Create data directories
RUN mkdir -p /data/models /data/checkpoints /data/logs /data/memory

# Set environment variables for GPU optimization
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV TORCH_CUDA_ARCH_LIST="8.0"
ENV SAFLA_GPU_ENABLED=true
ENV SAFLA_DATA_DIR=/data
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available()); exit(0 if torch.cuda.is_available() else 1)"

# Expose port
EXPOSE 8080

# Default command
CMD ["python", "-m", "safla.cli", "serve", "--host", "0.0.0.0", "--port", "8080", "--gpu"]