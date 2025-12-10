FROM python:3.10-slim

# Environment variables will be provided at runtime
ENV PYTHONPATH=/opt/sparc
ENV PYTHONUNBUFFERED=1
ENV TERM=xterm

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    git \
    build-essential \
    python3-dev \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create SPARC directory
WORKDIR /opt/sparc

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir hatchling sympy numpy
RUN pip install --no-cache-dir langchain langchain-core
RUN pip install --no-cache-dir -r requirements-dev.txt
RUN pip install --no-cache-dir -e .

# Install Playwright browsers
RUN python scripts/install_playwright.py

# Create a non-root user for SSH access
RUN useradd -m -s /bin/bash sparc_user && \
    echo "sparc_user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/sparc_user

# Copy and set up startup script
COPY sparc_cli/scripts/fly/startup.sh /usr/local/bin/sparc-startup
RUN chmod +x /usr/local/bin/sparc-startup

# Switch to the non-root user
USER sparc_user
WORKDIR /home/sparc_user

# Keep container running
CMD ["tail", "-f", "/dev/null"]
