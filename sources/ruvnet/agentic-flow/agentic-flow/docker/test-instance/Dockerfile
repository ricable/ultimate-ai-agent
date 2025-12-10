# Agentic-Flow Docker Test Instance
FROM node:20-alpine

# Install dependencies for WASM and system tools
RUN apk add --no-cache \
    git \
    bash \
    curl \
    python3 \
    make \
    g++ \
    sqlite

# Set working directory
WORKDIR /app

# Copy package files
COPY agentic-flow/package.json agentic-flow/package-lock.json* ./

# Install dependencies
RUN npm install

# Copy the entire agentic-flow package
COPY agentic-flow/ ./

# Create directories for data persistence
RUN mkdir -p /app/data/agentdb /app/data/memory /app/data/sessions

# Set environment variables
ENV NODE_ENV=production
ENV AGENTDB_PATH=/app/data/agentdb
ENV MEMORY_PATH=/app/data/memory

# Expose any ports if needed (for future web interface)
EXPOSE 3000

# Create entrypoint script that allows running commands
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'if [ "$1" = "test" ]; then' >> /entrypoint.sh && \
    echo '  exec /app/docker/test-instance/test-runner.sh' >> /entrypoint.sh && \
    echo 'elif [ "$1" = "cli" ]; then' >> /entrypoint.sh && \
    echo '  shift' >> /entrypoint.sh && \
    echo '  exec node /app/dist/cli.js "$@"' >> /entrypoint.sh && \
    echo 'else' >> /entrypoint.sh && \
    echo '  exec "$@"' >> /entrypoint.sh && \
    echo 'fi' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command starts bash
CMD ["/bin/bash"]
