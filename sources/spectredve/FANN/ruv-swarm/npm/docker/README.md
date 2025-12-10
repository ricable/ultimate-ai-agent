# Docker Configuration for ruv-swarm

This directory contains Docker configurations for testing and deploying ruv-swarm across different environments.

## 📋 Available Dockerfiles

### Test Environments

- **`Dockerfile.test`** - Main test environment with all dependencies
- **`Dockerfile.alpine.test`** - Lightweight Alpine Linux test environment
- **`Dockerfile.node18.test`** - Node.js 18 LTS test environment
- **`Dockerfile.node20.test`** - Node.js 20 LTS test environment
- **`Dockerfile.node22.test`** - Node.js 22 test environment
- **`Dockerfile.npx.test`** - NPX-specific testing environment
- **`Dockerfile.wasm.test`** - WebAssembly testing environment
- **`Dockerfile.perf.test`** - Performance testing environment
- **`Dockerfile.mcp.test`** - MCP (Model Context Protocol) testing environment

### Docker Compose Files

- **`docker-compose.test.yml`** - Standard test suite orchestration
- **`docker-compose.final-test.yml`** - Comprehensive final testing

## 🚀 Quick Start

### Running Tests in Docker

```bash
# Build and run the main test environment
docker build -f docker/Dockerfile.test -t ruv-swarm-test .
docker run --rm ruv-swarm-test

# Run with docker-compose
docker-compose -f docker/docker-compose.test.yml up
```

### Testing Specific Node.js Versions

```bash
# Test with Node.js 18
docker build -f docker/Dockerfile.node18.test -t ruv-swarm-node18 .
docker run --rm ruv-swarm-node18

# Test with Node.js 20
docker build -f docker/Dockerfile.node20.test -t ruv-swarm-node20 .
docker run --rm ruv-swarm-node20

# Test with Node.js 22
docker build -f docker/Dockerfile.node22.test -t ruv-swarm-node22 .
docker run --rm ruv-swarm-node22
```

### Performance Testing

```bash
# Run performance benchmarks
docker build -f docker/Dockerfile.perf.test -t ruv-swarm-perf .
docker run --rm ruv-swarm-perf
```

### WebAssembly Testing

```bash
# Test WASM functionality
docker build -f docker/Dockerfile.wasm.test -t ruv-swarm-wasm .
docker run --rm ruv-swarm-wasm
```

### MCP Integration Testing

```bash
# Test MCP server functionality
docker build -f docker/Dockerfile.mcp.test -t ruv-swarm-mcp .
docker run --rm -p 3000:3000 ruv-swarm-mcp
```

## 📦 Production Deployment

### Using NPX in Production

```bash
# Simple production deployment
docker run -d -p 3000:3000 --name ruv-swarm \
  -e NODE_ENV=production \
  node:20-alpine \
  npx ruv-swarm@latest mcp start --port 3000
```

### Building Custom Production Image

Create a `Dockerfile.production`:

```dockerfile
FROM node:20-alpine

WORKDIR /app

# Install ruv-swarm globally
RUN npm install -g ruv-swarm@latest

# Expose MCP port
EXPOSE 3000

# Start MCP server
CMD ["ruv-swarm", "mcp", "start", "--port", "3000"]
```

Build and run:

```bash
docker build -f Dockerfile.production -t ruv-swarm-prod .
docker run -d -p 3000:3000 --name ruv-swarm-prod ruv-swarm-prod
```

## 🔧 Environment Variables

Common environment variables for Docker deployments:

- `NODE_ENV` - Set to `production` for production deployments
- `RUVA_SWARM_MAX_AGENTS` - Maximum number of agents (default: 50)
- `RUVA_SWARM_PORT` - MCP server port (default: 3000)
- `RUVA_SWARM_LOG_LEVEL` - Logging level (debug, info, warn, error)
- `RUVA_SWARM_DATA_DIR` - Data persistence directory

Example:

```bash
docker run -d \
  -e NODE_ENV=production \
  -e RUVA_SWARM_MAX_AGENTS=100 \
  -e RUVA_SWARM_LOG_LEVEL=info \
  -v /data/ruv-swarm:/data \
  -p 3000:3000 \
  ruv-swarm-prod
```

## 🧪 CI/CD Integration

### GitHub Actions Example

```yaml
- name: Test with Multiple Node Versions
  run: |
    docker-compose -f docker/docker-compose.test.yml up --abort-on-container-exit
```

### Running Full Test Suite

```bash
# Run all tests across all environments
./scripts/docker-test-suite.sh
```

## 📝 Notes

- All Dockerfiles are optimized for testing and may need adjustments for production use
- The Alpine-based images are smallest but may have compatibility issues with some native modules
- For production, consider using multi-stage builds to reduce image size
- Always use specific version tags in production instead of `latest`

## 🐛 Troubleshooting

### Common Issues

1. **WASM Loading Errors**
   - Ensure the Docker image has proper WASM support
   - Check that the wasm files are correctly copied to the image

2. **Permission Issues**
   - Run containers with appropriate user permissions
   - Use `--user` flag if needed

3. **Network Issues**
   - Ensure ports are properly exposed
   - Check Docker network configuration

### Debug Mode

Run containers in debug mode:

```bash
docker run --rm -it -e RUVA_SWARM_LOG_LEVEL=debug ruv-swarm-test
```

## 📚 Additional Resources

- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Node.js Docker Guidelines](https://github.com/nodejs/docker-node/blob/main/docs/BestPractices.md)
- [ruv-swarm Documentation](https://github.com/ruvnet/ruv-FANN/tree/main/ruv-swarm)