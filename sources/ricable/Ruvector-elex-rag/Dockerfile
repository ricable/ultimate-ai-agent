# RuVector Telecom RAG - Docker Configuration
# Cognitive Automation Platform for Ericsson RAN

# Build stage
FROM node:22-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache python3 make g++ git

# Copy package files
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm ci

# Copy source
COPY tsconfig.json ./
COPY src ./src

# Build TypeScript
RUN npm run build

# Production stage
FROM node:22-alpine AS production

WORKDIR /app

# Install runtime dependencies only
RUN apk add --no-cache tini

# Create non-root user
RUN addgroup -g 1001 ruvector && \
    adduser -D -u 1001 -G ruvector ruvector

# Copy built application
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY package.json ./

# Create data directories
RUN mkdir -p /app/data/elex /app/data/3gpp /app/data/processed /app/data/vector_store /app/data/graph_store /app/logs && \
    chown -R ruvector:ruvector /app

# Switch to non-root user
USER ruvector

# Environment variables
ENV NODE_ENV=production
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080
ENV RUVECTOR_STORAGE_PATH=/app/data/vector_store
ENV GRAPH_STORAGE_PATH=/app/data/graph_store
ENV LOG_LEVEL=info

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8080/health || exit 1

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Start server
CMD ["node", "dist/index.js"]
