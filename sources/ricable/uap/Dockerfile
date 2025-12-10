# UAP Production Dockerfile
# Multi-stage build for optimized production deployment with security hardening

# === Frontend Build Stage ===
FROM node:20-alpine AS frontend-build
LABEL stage=frontend-build

# Create non-root user for security
RUN addgroup -g 1001 -S nodejs && adduser -S nextjs -u 1001

# Set working directory
WORKDIR /app

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm ci --only=production && npm cache clean --force

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# Verify build output
RUN ls -la dist/

# === Python Dependencies Stage ===
FROM python:3.11-slim AS python-deps

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY backend/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# === Production Runtime Stage ===
FROM python:3.11-slim AS production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/backend"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r uap && useradd -r -g uap -s /bin/bash uap

# Create application directories
RUN mkdir -p /app/logs /app/temp /app/uploads /app/models \
    && chown -R uap:uap /app

# Copy Python virtual environment from deps stage
COPY --from=python-deps /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy backend application
COPY --chown=uap:uap backend/ ./backend/

# Copy built frontend from frontend-build stage
COPY --from=frontend-build --chown=uap:uap /app/dist ./frontend/dist/

# Copy production scripts and configurations
COPY --chown=uap:uap scripts/ ./scripts/
COPY --chown=uap:uap .env.production.template ./.env.template

# Make scripts executable
RUN chmod +x ./scripts/*.sh

# Create logging configuration
RUN cat > /app/logging.conf << EOF
[loggers]
keys=root,uvicorn,uap

[handlers]
keys=console,file,json

[formatters]
keys=generic,json

[logger_root]
level=INFO
handlers=console,file

[logger_uvicorn]
level=INFO
handlers=console,file
qualname=uvicorn
propagate=0

[logger_uap]
level=INFO
handlers=console,json
qualname=uap
propagate=0

[handler_console]
class=StreamHandler
formatter=generic
args=(sys.stdout,)

[handler_file]
class=logging.handlers.RotatingFileHandler
formatter=generic
args=('/app/logs/app.log', 'a', 100*1024*1024, 10)

[handler_json]
class=logging.handlers.RotatingFileHandler
formatter=json
args=('/app/logs/app-json.log', 'a', 100*1024*1024, 10)

[formatter_generic]
format=%(asctime)s [%(process)d] [%(levelname)s] %(name)s: %(message)s
datefmt=%Y-%m-%d %H:%M:%S

[formatter_json]
class=pythonjsonlogger.jsonlogger.JsonFormatter
format=%(asctime)s %(name)s %(levelname)s %(message)s
EOF

# Create health check script
RUN cat > /app/healthcheck.sh << 'EOF'
#!/bin/bash
curl -f http://localhost:8000/health || exit 1
EOF
RUN chmod +x /app/healthcheck.sh

# Switch to non-root user
USER uap

# Expose ports
EXPOSE 8000 3000 9090

# Add health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["/app/healthcheck.sh"]

# Set labels for metadata
LABEL org.opencontainers.image.title="UAP Agent Orchestration Platform"
LABEL org.opencontainers.image.description="Multi-framework AI agent orchestration platform"
LABEL org.opencontainers.image.version="3.0.0"
LABEL org.opencontainers.image.vendor="UAP Team"
LABEL org.opencontainers.image.source="https://github.com/uap/uap-platform"

# Use tini as PID 1 for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - can be overridden
CMD ["/app/scripts/start-production.sh"]

# === Development Stage (for development builds) ===
FROM production AS development

# Switch back to root for installing dev dependencies
USER root

# Install development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Install development Python packages
RUN /opt/venv/bin/pip install \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy

# Switch back to uap user
USER uap

# Override for development
ENV UAP_ENV=development
ENV DEBUG=true
CMD ["/opt/venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# === Testing Stage ===
FROM development AS testing

USER root

# Install additional testing dependencies
RUN /opt/venv/bin/pip install \
    pytest-benchmark \
    pytest-mock \
    locust

USER uap

# Run tests by default
CMD ["/opt/venv/bin/pytest", "/app/backend/tests/", "-v", "--cov=/app/backend"]