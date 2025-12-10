# Multi-stage build for agentic-flow on GCP
FROM node:20-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++ git

WORKDIR /app

# Copy package files
COPY agentic-flow/package*.json ./
COPY agentic-flow/config/tsconfig.json ./config/

# Install dependencies
RUN npm ci --only=production && \
    npm cache clean --force

# Copy source code
COPY agentic-flow/src ./src
COPY agentic-flow/.claude ./.claude
COPY agentic-flow/docs ./docs

# Build TypeScript
RUN npm run build

# Production stage
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    curl \
    ca-certificates \
    tini

# Create app user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

WORKDIR /app

# Copy built application from builder
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=nodejs:nodejs /app/dist ./dist
COPY --from=builder --chown=nodejs:nodejs /app/.claude ./.claude
COPY --from=builder --chown=nodejs:nodejs /app/docs ./docs
COPY --chown=nodejs:nodejs agentic-flow/package*.json ./

# Environment variables (overridden by GCP Secret Manager)
ENV NODE_ENV=production \
    PORT=8080 \
    AGENTS_DIR=/app/.claude/agents \
    ENABLE_LOGGING=true \
    LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD node dist/health.js || exit 1

# Switch to non-root user
USER nodejs

# Expose port
EXPOSE 8080

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Start the application
CMD ["node", "dist/index.js"]
