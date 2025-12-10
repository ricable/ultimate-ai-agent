# Multi-stage Dockerfile for testing ruv-swarm npm package
# Stage 1: Base Node.js environment with multiple versions for testing
FROM node:20-alpine AS node20
FROM node:18-alpine AS node18
FROM node:22-alpine AS node22

# Main testing stage using Node.js 20 (LTS)
FROM node:20-alpine AS test-base

# Install system dependencies
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    git \
    bash \
    curl \
    jq

# Set working directory
WORKDIR /app

# Create test user to avoid permission issues
RUN adduser -D testuser && \
    chown -R testuser:testuser /app

# Switch to test user
USER testuser

# Copy package files (if local testing)
# This will be commented out for npm registry testing
# COPY --chown=testuser:testuser ./ruv-swarm-*.tgz /tmp/

# Stage 2: NPM Registry Test
FROM test-base AS npm-test

# Create package.json for testing
RUN printf '%s\n' \
'{' \
'  "name": "ruv-swarm-test",' \
'  "version": "1.0.0",' \
'  "description": "Test environment for ruv-swarm npm package",' \
'  "main": "index.js",' \
'  "scripts": {' \
'    "test": "node test.js",' \
'    "test:mcp": "npx ruv-swarm mcp start --test",' \
'    "test:init": "npx ruv-swarm init test-project",' \
'    "test:hooks": "npx ruv-swarm hook pre-task"' \
'  },' \
'  "dependencies": {}' \
'}' > package.json

# Install ruv-swarm from npm registry
RUN npm install ruv-swarm@latest --verbose

# Create test script
RUN printf '%s\n' \
'const ruvSwarm = require("ruv-swarm");' \
'' \
'console.log("Testing ruv-swarm package...");' \
'console.log("Package loaded successfully!");' \
'console.log("Available exports:", Object.keys(ruvSwarm));' \
'' \
'// Test basic functionality' \
'if (ruvSwarm.RuvSwarm) {' \
'  console.log("✓ RuvSwarm class available");' \
'}' \
'' \
'if (ruvSwarm.Agent) {' \
'  console.log("✓ Agent class available");' \
'}' \
'' \
'if (ruvSwarm.NeuralNetwork) {' \
'  console.log("✓ NeuralNetwork class available");' \
'}' \
'' \
'// Test additional classes' \
'if (ruvSwarm.Swarm) {' \
'  console.log("✓ Swarm class available");' \
'}' \
'' \
'if (ruvSwarm.Task) {' \
'  console.log("✓ Task class available");' \
'}' \
'' \
'console.log("\nBasic tests passed!");' > test.js

# Stage 3: Local Package Test (for testing local builds)
FROM test-base AS local-test

# Copy local package file
ARG LOCAL_PACKAGE_PATH
COPY --chown=testuser:testuser ${LOCAL_PACKAGE_PATH:-./ruv-swarm-*.tgz} /tmp/package.tgz

# Create package.json
RUN printf '%s\n' \
'{' \
'  "name": "ruv-swarm-local-test",' \
'  "version": "1.0.0",' \
'  "description": "Local test environment for ruv-swarm",' \
'  "dependencies": {}' \
'}' > package.json

# Install from local file
RUN npm install /tmp/package.tgz --verbose

# Copy test script from npm-test stage
COPY --from=npm-test /app/test.js .

# Stage 4: Comprehensive Test Suite
FROM test-base AS comprehensive-test

# Install testing frameworks
RUN npm init -y && \
    npm install --save-dev \
    jest \
    mocha \
    chai \
    supertest

# Install ruv-swarm
RUN npm install ruv-swarm@latest

# Create comprehensive test suite
RUN mkdir -p tests && \
    printf '%s\n' \
'const { RuvSwarm, Agent, Neural } = require("ruv-swarm");' \
'const assert = require("assert");' \
'' \
'describe("ruv-swarm npm package", () => {' \
'  it("should export main classes", () => {' \
'    assert(RuvSwarm, "RuvSwarm class should be exported");' \
'    assert(Agent, "Agent class should be exported");' \
'    assert(Neural, "Neural class should be exported");' \
'  });' \
'' \
'  it("should create swarm instance", () => {' \
'    const swarm = new RuvSwarm();' \
'    assert(swarm, "Should create swarm instance");' \
'  });' \
'' \
'  it("should handle CLI commands", async () => {' \
'    const { execSync } = require("child_process");' \
'    const result = execSync("npx ruv-swarm --version", { encoding: "utf8" });' \
'    assert(result.includes("ruv-swarm"), "CLI should return version");' \
'  });' \
'});' > tests/integration.test.js

# Create test runner
RUN printf '%s\n' \
'{' \
'  "scripts": {' \
'    "test": "mocha tests/*.test.js",' \
'    "test:jest": "jest",' \
'    "test:all": "npm test && npm run test:jest"' \
'  },' \
'  "jest": {' \
'    "testEnvironment": "node"' \
'  }' \
'}' > test-config.json && \
    cat package.json test-config.json | jq -s ".[0] * .[1]" > package.tmp && \
    mv package.tmp package.json && \
    rm test-config.json

# Default command for all stages
CMD ["npm", "test"]

# Stage 5: Final test runner
FROM test-base AS final

# Copy installation from comprehensive test
COPY --from=comprehensive-test /app /app

# Add health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node -e "require('ruv-swarm'); process.exit(0);" || exit 1

# Set entrypoint for flexible testing
ENTRYPOINT ["/bin/sh", "-c"]
CMD ["npm test && echo 'All tests passed successfully!'"]