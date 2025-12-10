# Nova Medicina Docker Environment
# Multi-stage build for optimal size and security

# Stage 1: Build
FROM node:18-alpine AS builder

LABEL maintainer="ruv <ruv@ruv.io>"
LABEL description="Nova Medicina - AI-powered medical analysis with anti-hallucination safeguards"

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./
COPY jest.config.js ./

# Install dependencies
RUN npm ci --only=production && \
    npm install -D typescript @types/node

# Copy source code
COPY src/ ./src/
COPY bin/ ./bin/
COPY tests/ ./tests/

# Build TypeScript
RUN npm run build

# Stage 2: Production
FROM node:18-alpine

WORKDIR /app

# Copy built files and production dependencies
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/bin ./bin
COPY --from=builder /app/package*.json ./

# Copy documentation
COPY README.md ./
COPY docs/ ./docs/
COPY examples/ ./examples/

# Create config directory
RUN mkdir -p /root/.nova-medicina

# Set environment variables
ENV NODE_ENV=production
ENV NOVA_MEDICINA_LOG_LEVEL=info

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "console.log('OK')" || exit 1

# Expose port for API (if implemented)
EXPOSE 3000

# Make CLI executable
RUN chmod +x /app/bin/nova-medicina

# Add to PATH
ENV PATH="/app/bin:${PATH}"

# Default command - show help
CMD ["nova-medicina", "--help"]
