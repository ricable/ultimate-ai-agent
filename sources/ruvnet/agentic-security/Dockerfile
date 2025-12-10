FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    docker.io \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install security tools
# Install OWASP ZAP
RUN docker pull owasp/zap2docker-stable

# Install latest Nuclei
RUN curl -s https://api.github.com/repos/projectdiscovery/nuclei/releases/latest \
    | grep "browser_download_url.*linux_amd64.zip" \
    | cut -d '"' -f 4 \
    | wget -i - && \
    unzip nuclei-*-linux_amd64.zip && \
    mv nuclei /usr/local/bin/ && \
    rm nuclei-*-linux_amd64.zip

# Install Dependency-Check
RUN wget https://github.com/jeremy-lin/dependency-check/releases/download/v6.5.3/dependency-check-6.5.3-release.zip && \
    unzip dependency-check-6.5.3-release.zip -d /usr/local/dependency-check && \
    rm dependency-check-6.5.3-release.zip

# Set up working directory
WORKDIR /app

# Create and configure cache directory
RUN mkdir -p .security_cache && chmod 755 .security_cache

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
ENTRYPOINT ["python", "-m", "src.agentic_security.security_cli"]
CMD ["--help"]
