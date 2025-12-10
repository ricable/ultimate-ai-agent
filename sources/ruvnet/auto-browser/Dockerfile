# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright system dependencies
RUN npx playwright install-deps chromium

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install -e .

# Install Playwright browsers
RUN playwright install chromium

# Create output directory
RUN mkdir -p output

# Copy example env file and set environment variables
COPY .env.example .env

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV BROWSER_HEADLESS=true

# Expose port if needed
EXPOSE 3000

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ -z "$OPENAI_API_KEY" ]; then\n\
    echo "Error: OPENAI_API_KEY environment variable must be set"\n\
    exit 1\n\
fi\n\
exec "$@"' > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["auto-browser", "--help"]

# Usage instructions as comments
# Build:
#   docker build -t auto-browser .
#
# Run basic example:
#   docker run -e OPENAI_API_KEY=your_key auto-browser \
#     auto-browser easy "https://www.google.com/finance" "Get AAPL stock price"
#
# Run with volume for output:
#   docker run -v $(pwd)/output:/app/output -e OPENAI_API_KEY=your_key auto-browser \
#     auto-browser easy -v "https://www.google.com/finance" "Get AAPL stock price"
#
# Run interactive mode:
#   docker run -e OPENAI_API_KEY=your_key auto-browser \
#     auto-browser easy --interactive "https://example.com" "Fill out contact form"
