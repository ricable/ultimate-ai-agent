FROM python:3.10-slim

# Environment variables will be provided at runtime

# Set working directory
WORKDIR /app

# Install system dependencies required for Playwright and build tools
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    gnupg \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project configuration files
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY sparc_cli/__version__.py ./sparc_cli/__version__.py

# Create required directories
RUN mkdir -p scripts
COPY scripts/install_playwright.py ./scripts/install_playwright.py

# Install build dependencies and the package
RUN pip install --no-cache-dir hatchling
RUN pip install --no-cache-dir sympy numpy
RUN pip install --no-cache-dir ".[dev]"

# Install Playwright browsers
RUN playwright install

# Copy the rest of the application
COPY . .

# Set non-sensitive environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Create entrypoint
ENTRYPOINT ["sparc"]
