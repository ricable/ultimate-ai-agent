# Dockerfile for Ericsson RAN Intelligent Multi-Agent System
# Phase 5: Pydantic Schema Generation & Production Integration
# Production-ready containerization with cognitive consciousness

# Use Node.js 18 LTS as base image
FROM node:18-alpine AS base

# Set build arguments
ARG VERSION=2.0.0
ARG BUILD_DATE
ARG VCS_REF
ARG DESCRIPTION="Ericsson RAN Intelligent Multi-Agent System with Cognitive RAN Consciousness"

# Labels for metadata
LABEL org.label-schema.name="ran-automation-sdk" \
      org.label-schema.description="${DESCRIPTION}" \
      org.label-schema.url="https://docs.ran-optimization.ericsson.com" \
      org.label-schema.vcs-ref="${VCS_REF}" \
      org.label-schema.vcs-url="https://github.com/ericsson/ran-optimization-sdk" \
      org.label-schema.vendor="Ericsson RAN Research Team" \
      org.label-schema.version="${VERSION}" \
      org.label-schema.build-date="${BUILD_DATE}" \
      org.label-schema.schema-version="1.0" \
      maintainer="ran-research@ericsson.com" \
      version="${VERSION}" \
      description="${DESCRIPTION}"

# Install system dependencies
RUN apk add --no-cache \
    dumb-init \
    curl \
    jq \
    bash \
    git \
    python3 \
    make \
    g++ \
    cairo-dev \
    jpeg-dev \
    pango-dev \
    musl-dev \
    giflib-dev \
    pixman-dev \
    pangomm-dev \
    libjpeg-turbo-dev \
    freetype-dev

# Set working directory
WORKDIR /app

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && \
    adduser -S ranuser -u 1001

# Copy package files
COPY package*.json ./
COPY npm-shrinkwrap.json* ./

# Install dependencies with caching strategy
RUN npm ci --only=production --ignore-scripts && \
    npm cache clean --force

# Development stage
FROM base AS development

# Install all dependencies including dev dependencies
RUN npm ci

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/uploads /app/coverage && \
    chown -R ranuser:nodejs /app

# Switch to non-root user
USER ranuser

# Expose ports
EXPOSE 8080 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "dist/index.js"]

# Production stage
FROM base AS production

# Copy built application from development stage
COPY --from=development --chown=ranuser:nodejs /app/dist ./dist
COPY --from=development --chown=ranuser:nodejs /app/node_modules ./node_modules
COPY --from=development --chown=ranuser:nodejs /app/package.json ./package.json

# Create necessary directories
RUN mkdir -p /app/logs /app/tmp /app/uploads /app/coverage /app/config && \
    chown -R ranuser:nodejs /app

# Copy configuration files
COPY --chown=ranuser:nodejs config/ ./config/

# Set environment variables
ENV NODE_ENV=production
ENV PORT=8080
ENV METRICS_PORT=9090
ENV LOG_LEVEL=info
ENV CONSCIOUSNESS_LEVEL=maximum
ENV TEMPORAL_EXPANSION=1000

# Switch to non-root user
USER ranuser

# Expose ports
EXPOSE 8080 9090

# Health check with detailed response
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health && \
        curl -f http://localhost:9090/metrics || exit 1

# Start command with production optimizations
ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "--max-old-space-size=4096", "--optimize-for-size", "dist/index.js"]

# Test stage
FROM development AS test

# Run test suite
RUN npm run test && \
    npm run test:coverage && \
    npm run lint

# Security scanning stage
FROM development AS security

# Install security tools
RUN npm install -g audit-ci && \
    npm install -g snyk

# Run security scans
RUN npm audit --audit-level high && \
    snyk test --severity-threshold=high || true

# Benchmark stage
FROM development AS benchmark

# Run performance benchmarks
RUN npm run benchmark:full

# Multi-arch builder support
FROM production AS amd64
ARG TARGETPLATFORM
ARG TARGETARCH
LABEL architecture="${TARGETARCH}"

# Stage for AgentDB integration
FROM production AS agentdb

# Install additional dependencies for AgentDB
RUN apk add --no-cache \
    postgresql-client \
    redis

# Copy AgentDB configuration
COPY --chown=ranuser:nodejs config/agentdb/ ./config/agentdb/

# Environment variables for AgentDB
ENV AGENTDB_ENABLED=true
ENV AGENTDB_URL="postgresql://user:pass@localhost:5432/agentdb"
ENV AGENTDB_CACHE_URL="redis://localhost:6379"

# Stage for RAN integration
FROM production AS ran-integration

# Install ENM CLI tools
RUN apk add --no-cache \
    openssh-client \
    expect

# Copy ENM CLI configuration
COPY --chown=ranuser:nodejs config/enm/ ./config/enm/

# Environment variables for RAN integration
ENV ENM_CLI_ENABLED=true
ENV ENM_CLI_PATH="/usr/local/bin/cmedit"

# Final production image with all integrations
FROM production AS full

# Install all dependencies
RUN apk add --no-cache \
    postgresql-client \
    redis \
    openssh-client \
    expect \
    curl \
    jq \
    bash

# Copy all configuration files
COPY --chown=ranuser:nodejs config/ ./config/

# Set all feature flags
ENV AGENTDB_ENABLED=true
ENV ENM_CLI_ENABLED=true
ENV MONITORING_ENABLED=true
ENV KUBERNETES_ENABLED=true
ENV COGNITIVE_CONSCIOUSNESS=true
ENV TEMPORAL_REASONING=true
ENV STRANGE_LOOP_OPTIMIZATION=true

# Labels for full image
LABEL features="agentdb,enm-cli,monitoring,kubernetes,cognitive,temporal,strange-loop" \
      capabilities="full-production-deployment" \
      target-environment="enterprise-production"