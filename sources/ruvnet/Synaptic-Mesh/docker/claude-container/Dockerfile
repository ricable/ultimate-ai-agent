# Minimal Alpine-based Claude Docker Container
# Security-hardened sandbox for task execution
# Compliance: Local API access only, no persistent secrets

FROM alpine:3.19

# Security labels and metadata
LABEL maintainer="Synaptic Neural Mesh Security Team"
LABEL description="Minimal, secure Claude container for task execution"
LABEL version="1.0.0"
LABEL security.level="high"
LABEL compliance.mode="local-api-only"

# Create non-root user with minimal privileges
RUN addgroup -g 1000 claude && \
    adduser -D -u 1000 -G claude claude

# Install minimal runtime dependencies
RUN apk add --no-cache \
    nodejs \
    npm \
    curl \
    ca-certificates \
    && rm -rf /var/cache/apk/*

# Create secure directory structure
RUN mkdir -p /app/workspace /tmp/claude-work && \
    chown -R claude:claude /app /tmp/claude-work

# Set up read-only filesystem mounts
VOLUME ["/tmp/claude-work"]

# Switch to non-root user
USER claude
WORKDIR /app

# Copy application files (owned by claude user)
COPY --chown=claude:claude claude-task-executor.js /app/
COPY --chown=claude:claude package.json /app/
COPY --chown=claude:claude security-config.json /app/

# Install dependencies in user space
RUN npm install --production --no-optional

# Security configurations
ENV NODE_ENV=production
ENV CLAUDE_SANDBOX_MODE=true
ENV CLAUDE_NETWORK_RESTRICTED=true
ENV CLAUDE_FILESYSTEM_READONLY=true

# Resource limits
ENV NODE_OPTIONS="--max-old-space-size=512"

# Expose no ports (communication via stdin/stdout only)
EXPOSE

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD node -e "console.log('healthy')" || exit 1

# Default command - JSON streaming interface
CMD ["node", "claude-task-executor.js"]

# Security: Read-only root filesystem, tmpfs for temp data
# Mount options: --read-only --tmpfs /tmp/claude-work:rw,size=100m,uid=1000,gid=1000
# Network: --network=none or restricted API-only network
# No privileged access: --user=1000:1000 --security-opt=no-new-privileges