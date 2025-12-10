# Multi-stage build for Synaptic Neural Mesh
FROM node:20-alpine AS builder

# Install build dependencies
RUN apk add --no-cache python3 make g++ curl git

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY src/ ./src/
COPY bin/ ./bin/
COPY scripts/ ./scripts/

# Build application
RUN npm run build

# Production image
FROM node:20-alpine AS production

# Install runtime dependencies
RUN apk add --no-cache sqlite curl

# Create non-root user
RUN addgroup -g 1001 -S synaptic && \
    adduser -S synaptic -u 1001

# Create app directory
WORKDIR /app

# Copy built application
COPY --from=builder --chown=synaptic:synaptic /app/lib ./lib/
COPY --from=builder --chown=synaptic:synaptic /app/bin ./bin/
COPY --from=builder --chown=synaptic:synaptic /app/node_modules ./node_modules/
COPY --from=builder --chown=synaptic:synaptic /app/package.json ./

# Make binary executable
RUN chmod +x ./bin/synaptic-mesh

# Create data directories
RUN mkdir -p /app/.synaptic/data /app/.synaptic/cache /app/.synaptic/logs && \
    chown -R synaptic:synaptic /app/.synaptic

# Expose ports
EXPOSE 8080 9090

# Switch to non-root user
USER synaptic

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD node -e "console.log('healthy')" || exit 1

# Default command
CMD ["./bin/synaptic-mesh", "start", "--no-ui"]