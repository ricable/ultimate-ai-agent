# Claude Max Market Docker Container
# Secure, isolated Claude execution environment

FROM node:18-alpine

# Install system dependencies
RUN apk add --no-cache \
    ca-certificates \
    curl \
    docker-cli \
    && rm -rf /var/cache/apk/*

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy application code
COPY src/ ./src/
COPY bin/ ./bin/

# Create security configurations
COPY security-config.json ./
COPY compliance-config.json ./

# Create non-root user
RUN addgroup -g 1001 -S claude && \
    adduser -S claude -u 1001 -G claude

# Set up permissions
RUN chown -R claude:claude /app && \
    chmod +x bin/claude-max-market.js

# Create volumes for persistent data
VOLUME ["/app/data", "/app/logs"]

# Switch to non-root user
USER claude

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "console.log('healthy')" || exit 1

# Environment variables
ENV NODE_ENV=production
ENV LOG_LEVEL=info

# Expose no ports (security by default)
# Use docker run -p if network access needed

# Default command
ENTRYPOINT ["node", "bin/claude-max-market.js"]
CMD ["--help"]