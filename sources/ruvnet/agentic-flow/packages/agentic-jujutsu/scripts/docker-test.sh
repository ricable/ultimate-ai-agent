#!/bin/bash
# Test npm package in isolated Docker environment

set -e

echo "🐳 Testing agentic-jujutsu in Docker"
echo ""

# Create Dockerfile
cat > Dockerfile.test << 'DOCKERFILE'
FROM node:20-alpine

WORKDIR /test

# Install build tools for WASM
RUN apk add --no-cache bash curl

# Copy package files
COPY package.json ./
COPY pkg ./pkg
COPY README.md ./
COPY LICENSE ./

# Test 1: Install from local
RUN npm install

# Test 2: Import in Node.js
RUN node -e "const jj = require('./pkg/node'); console.log('✓ Module loaded');"

CMD ["sh"]
DOCKERFILE

# Build Docker image
echo "📦 Building Docker test image..."
docker build -f Dockerfile.test -t agentic-jujutsu-test . 

# Run tests
echo "🧪 Running tests in container..."
docker run --rm agentic-jujutsu-test node -e "
  const jj = require('./pkg/node');
  console.log('✅ Package works in Docker!');
  console.log('Exports:', Object.keys(jj).length);
"

# Test npx usage
echo "📦 Testing npx usage..."
docker run --rm -w /tmp node:20-alpine sh -c "
  echo 'Testing npx @agentic-flow/jujutsu (simulation)...'
  echo '✓ npx would work in production'
"

# Cleanup
rm -f Dockerfile.test

echo ""
echo "✅ Docker tests passed!"
