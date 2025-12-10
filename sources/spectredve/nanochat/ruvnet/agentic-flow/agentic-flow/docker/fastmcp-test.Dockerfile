# FastMCP Test Dockerfile
FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY tsconfig.json ./

# Install dependencies
RUN npm ci

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Build TypeScript
RUN npm run build

# Expose HTTP port
EXPOSE 3000

# Default command (can be overridden)
CMD ["node", "dist/cli/mcp.js", "status"]
