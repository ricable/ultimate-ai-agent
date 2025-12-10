# Multi-stage Dockerfile for ruvector-postgres extension
# Builds the extension and creates a PostgreSQL image with it installed

# Build stage
FROM rust:1.75-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    pkg-config \
    postgresql-server-dev-16 \
    postgresql-16 \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

# Install cargo-pgrx
RUN cargo install cargo-pgrx --version 0.12.0 --locked

# Set up workspace
WORKDIR /build

# Copy only Cargo files first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY crates/ruvector-postgres/Cargo.toml ./crates/ruvector-postgres/

# Copy source code
COPY crates/ruvector-postgres ./crates/ruvector-postgres/

# Initialize pgrx
RUN cd crates/ruvector-postgres && \
    cargo pgrx init --pg16=/usr/lib/postgresql/16/bin/pg_config

# Build the extension with all features
RUN cd crates/ruvector-postgres && \
    cargo pgrx package --features pg16,index-all,quant-all --release

# Runtime stage
FROM postgres:16-bookworm

# Labels
LABEL maintainer="ruvector team"
LABEL description="PostgreSQL with ruvector extension - high-performance vector similarity search"
LABEL version="0.1.0"

# Copy the built extension from builder
COPY --from=builder /build/target/release/ruvector-postgres-pg16/usr/share/postgresql/16/extension/* \
    /usr/share/postgresql/16/extension/
COPY --from=builder /build/target/release/ruvector-postgres-pg16/usr/lib/postgresql/16/lib/* \
    /usr/lib/postgresql/16/lib/

# Copy SQL files and control file
COPY --from=builder /build/crates/ruvector-postgres/ruvector.control \
    /usr/share/postgresql/16/extension/
COPY --from=builder /build/crates/ruvector-postgres/sql/*.sql \
    /usr/share/postgresql/16/extension/

# Set environment variables
ENV POSTGRES_DB=postgres
ENV POSTGRES_USER=postgres
ENV POSTGRES_PASSWORD=postgres

# Add initialization script to create extension
RUN mkdir -p /docker-entrypoint-initdb.d
RUN echo "CREATE EXTENSION IF NOT EXISTS ruvector;" > /docker-entrypoint-initdb.d/01-ruvector.sql

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD pg_isready -U postgres || exit 1

# Expose PostgreSQL port
EXPOSE 5432

# Use the default PostgreSQL entrypoint
CMD ["postgres"]
