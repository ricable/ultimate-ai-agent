# Off-Target Prediction Service with PyTorch
FROM python:3.11-slim

LABEL maintainer="CRISPR-Cas13 Team"
LABEL description="Off-target prediction service with PyTorch and Biopython"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY off_target_worker.py /app/
COPY predictor.py /app/
COPY utils.py /app/

# Copy pre-trained model
COPY models/cas13_off_target.pt /app/models/

# Create non-root user
RUN useradd -u 1000 -m -s /bin/bash worker && \
    chown -R worker:worker /app

USER worker

EXPOSE 9090

CMD ["python", "off_target_worker.py"]
