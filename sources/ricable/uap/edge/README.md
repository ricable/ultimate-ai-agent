# UAP Edge Computing Platform

WebAssembly-based edge computing runtime for the UAP (Unified Agentic Platform) enabling distributed AI agent execution with low-latency processing.

## Overview

The UAP Edge Computing Platform provides:
- **WebAssembly Runtime**: Secure, sandboxed execution environment
- **Distributed Processing**: Scale agent workloads across edge nodes
- **Low Latency**: Sub-millisecond execution for time-critical operations
- **Security**: Isolated execution with fine-grained permissions
- **Scalability**: Auto-scaling based on demand
- **Integration**: Seamless integration with UAP backend and mobile apps

## Architecture

### Core Components

```
┌──────────────────────────────────────────────────┐
│                UAP Backend                    │
│  ┌────────────────────────────────────────────┐  │
│  │          EdgeManager             │  │
│  │    - Module Management         │  │
│  │    - Instance Lifecycle       │  │
│  │    - Execution Orchestration  │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
                        │
                        │ HTTP/WebSocket
                        │
┌──────────────────────────────────────────────────┐
│                Edge Runtime                   │
│  ┌────────────────────────────────────────────┐  │
│  │        WebAssembly Engine        │  │
│  │   ┌────────────────────────────────────┐   │  │
│  │   │      WASM Module 1       │   │  │
│  │   │   - AI Model Inference   │   │  │
│  │   │   - Data Processing      │   │  │
│  │   └────────────────────────────────────┘   │  │
│  │   ┌────────────────────────────────────┐   │  │
│  │   │      WASM Module 2       │   │  │
│  │   │   - Document Analysis    │   │  │
│  │   │   - Language Processing  │   │  │
│  │   └────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────┘  │
│  ┌────────────────────────────────────────────┐  │
│  │         Security Layer         │  │
│  │    - Sandboxed Execution     │  │
│  │    - Resource Limits         │  │
│  │    - Permission Control      │  │
│  └────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────┘
```

### Technology Stack
- **Runtime**: Node.js with WebAssembly support
- **WASM Engine**: Wasmer with WASI support
- **Security**: Capability-based security model
- **Storage**: Redis for caching, SQLite for local data
- **Communication**: WebSocket and HTTP APIs
- **Monitoring**: Prometheus metrics and structured logging

## Features

### WebAssembly Runtime
- **Secure Execution**: Sandboxed environment with capability-based security
- **Multi-language Support**: Run code compiled from Rust, C++, AssemblyScript, etc.
- **Memory Management**: Configurable memory limits and garbage collection
- **File System Access**: WASI-based file system with controlled access
- **Network Isolation**: Controlled network access with proxy support

### Module Management
- **Dynamic Loading**: Load and unload modules at runtime
- **Version Control**: Support for multiple module versions
- **Hot Reloading**: Update modules without downtime
- **Dependency Management**: Handle module dependencies and conflicts
- **Registry Integration**: Connect to module registries

### Instance Lifecycle
- **Pool Management**: Instance pooling for better performance
- **Auto-scaling**: Scale instances based on demand
- **Resource Monitoring**: Track memory and CPU usage
- **Health Checks**: Monitor instance health and restart if needed
- **Graceful Shutdown**: Clean instance termination

### Performance Optimization
- **JIT Compilation**: Just-in-time compilation for better performance
- **Caching**: Intelligent caching of compiled modules
- **Load Balancing**: Distribute load across multiple instances
- **Resource Pooling**: Reuse expensive resources
- **Profiling**: Built-in performance profiling tools

## Getting Started

### Prerequisites
- Node.js 18+ with WebAssembly support
- Docker (optional, for containerized deployment)
- Redis server
- Basic understanding of WebAssembly concepts

### Installation

1. **Install dependencies**:
   ```bash
   cd edge
   npm install
   ```

2. **Configure environment**:
   Create `.env` file:
   ```env
   NODE_ENV=development
   PORT=3001
   REDIS_URL=redis://localhost:6379
   BACKEND_URL=http://localhost:8000
   LOG_LEVEL=info
   WASM_MODULE_PATH=./modules
   MAX_INSTANCES_PER_MODULE=10
   ```

3. **Start the edge runtime**:
   ```bash
   npm run dev
   ```

### Development Server

The edge runtime provides several endpoints:
- **Health Check**: `GET /health`
- **Module Management**: `POST /modules`, `GET /modules`
- **Instance Management**: `POST /instances`, `DELETE /instances/:id`
- **Function Execution**: `POST /instances/:id/execute`
- **Metrics**: `GET /metrics`

## WebAssembly Module Development

### Supported Languages

**Rust** (Recommended):
```rust
// src/lib.rs
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn log(message: &str);
}

#[wasm_bindgen]
pub fn process_data(input: &str) -> String {
    log(&format!("Processing: {}", input));
    // Process the input data
    format!("Processed: {}", input)
}

#[wasm_bindgen]
pub fn ai_inference(model_data: &[u8], input_data: &[u8]) -> Vec<u8> {
    // Perform AI inference
    // Return inference results
    vec![1, 2, 3, 4] // Placeholder
}
```

**AssemblyScript**:
```typescript
// assembly/index.ts
export function processText(text: string): string {
  // Process text using AssemblyScript
  return text.toUpperCase();
}

export function analyzeDocument(content: ArrayBuffer): ArrayBuffer {
  // Analyze document content
  // Return analysis results
  return new ArrayBuffer(0);
}
```

### Building Modules

**Rust to WASM**:
```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build the module
wasm-pack build --target web --out-dir pkg

# The .wasm file will be in pkg/
```

**AssemblyScript to WASM**:
```bash
# Install AssemblyScript
npm install -g assemblyscript

# Initialize project
npx asinit .

# Build the module
npm run asbuild

# The .wasm file will be in build/
```

### Module Configuration

Each module requires a configuration file:

```json
// module.json
{
  "name": "ai-inference-module",
  "version": "1.0.0",
  "description": "AI inference module for edge computing",
  "wasmPath": "./ai_inference.wasm",
  "exports": [
    "process_data",
    "ai_inference"
  ],
  "permissions": {
    "fileSystem": false,
    "network": false,
    "compute": true
  },
  "resources": {
    "maxMemoryMB": 64,
    "maxCpuTimeMs": 5000
  },
  "dependencies": [],
  "metadata": {
    "author": "UAP Team",
    "license": "MIT",
    "tags": ["ai", "inference", "ml"]
  }
}
```

## API Reference

### Module Management

**Load Module**:
```http
POST /api/edge/modules
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "my-module",
  "wasmPath": "/path/to/module.wasm",
  "version": "1.0.0",
  "description": "My WebAssembly module",
  "permissions": {
    "fileSystem": false,
    "network": false,
    "compute": true
  }
}
```

**List Modules**:
```http
GET /api/edge/modules
Authorization: Bearer <token>
```

### Instance Management

**Create Instance**:
```http
POST /api/edge/instances
Content-Type: application/json
Authorization: Bearer <token>

{
  "moduleName": "my-module",
  "timeout": 30000,
  "maxMemory": 67108864,
  "priority": "normal"
}
```

**Execute Function**:
```http
POST /api/edge/instances/{instanceId}/execute
Content-Type: application/json
Authorization: Bearer <token>

{
  "functionName": "process_data",
  "args": ["input data"],
  "timeout": 5000
}
```

### Response Format

**Success Response**:
```json
{
  "success": true,
  "result": "processed data",
  "executionTimeMs": 123,
  "memoryUsed": 1048576,
  "cpuTimeMs": 45
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Function execution failed: timeout exceeded",
  "executionTimeMs": 5000,
  "memoryUsed": 2097152
}
```

## Mobile Integration

### Real-time Communication

The edge runtime provides WebSocket endpoints for real-time communication with mobile apps:

```javascript
// Mobile app WebSocket connection
const ws = new WebSocket('ws://edge-server:3001/mobile/ws?token=jwt_token');

ws.onopen = () => {
  console.log('Connected to edge runtime');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.type === 'execution_result') {
    // Handle execution result
    console.log('Result:', message.result);
  }
};

// Execute function on edge
ws.send(JSON.stringify({
  type: 'execute_function',
  instanceId: 'instance-123',
  functionName: 'process_data',
  args: ['mobile data']
}));
```

### Offline Support

The edge runtime supports offline operation with local caching:

```javascript
// Queue execution for when online
EdgeClient.queueExecution({
  moduleName: 'ai-module',
  functionName: 'analyze',
  data: localData,
  priority: 'high'
});

// Sync when connection restored
EdgeClient.on('online', () => {
  EdgeClient.syncQueuedExecutions();
});
```

## Deployment

### Docker Deployment

**Dockerfile**:
```dockerfile
FROM node:18-alpine

# Install system dependencies
RUN apk add --no-cache \
    build-base \
    python3

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci --only=production

# Copy source code
COPY . .

# Build application
RUN npm run build

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001
USER nextjs

EXPOSE 3001

CMD ["npm", "start"]
```

**Docker Compose**:
```yaml
version: '3.8'
services:
  edge-runtime:
    build: .
    ports:
      - "3001:3001"
    environment:
      - NODE_ENV=production
      - REDIS_URL=redis://redis:6379
      - BACKEND_URL=http://backend:8000
    depends_on:
      - redis
    volumes:
      - ./modules:/app/modules:ro
      - ./logs:/app/logs
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### Kubernetes Deployment

**Deployment**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: uap-edge-runtime
spec:
  replicas: 3
  selector:
    matchLabels:
      app: uap-edge-runtime
  template:
    metadata:
      labels:
        app: uap-edge-runtime
    spec:
      containers:
      - name: edge-runtime
        image: uap/edge-runtime:latest
        ports:
        - containerPort: 3001
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        - name: BACKEND_URL
          value: "http://backend-service:8000"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 3001
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Production Configuration

**Environment Variables**:
```env
NODE_ENV=production
PORT=3001
REDIS_URL=redis://redis-cluster:6379
BACKEND_URL=https://api.uap.com
LOG_LEVEL=warn
WASM_MODULE_PATH=/app/modules
MAX_INSTANCES_PER_MODULE=100
INSTANCE_TIMEOUT_MS=30000
MEMORY_LIMIT_MB=128
CPU_LIMIT_MS=10000
ENABLE_METRICS=true
METRICS_PORT=9090
```

## Monitoring and Observability

### Metrics

The edge runtime exposes Prometheus metrics:

```
# HELP edge_modules_loaded_total Total number of loaded modules
# TYPE edge_modules_loaded_total counter
edge_modules_loaded_total 5

# HELP edge_instances_active Current number of active instances
# TYPE edge_instances_active gauge
edge_instances_active 23

# HELP edge_function_executions_total Total number of function executions
# TYPE edge_function_executions_total counter
edge_function_executions_total{status="success"} 1542
edge_function_executions_total{status="error"} 8

# HELP edge_execution_duration_seconds Function execution duration
# TYPE edge_execution_duration_seconds histogram
edge_execution_duration_seconds_bucket{le="0.001"} 856
edge_execution_duration_seconds_bucket{le="0.01"} 1200
edge_execution_duration_seconds_bucket{le="0.1"} 1480
```

### Logging

Structured logging with configurable levels:

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "service": "edge-runtime",
  "module": "EdgeRuntime",
  "message": "Function executed successfully",
  "data": {
    "instanceId": "ai-module_req_1705315845_abc123",
    "functionName": "process_data",
    "executionTimeMs": 23,
    "memoryUsed": 1048576
  }
}
```

### Health Checks

**Endpoint**: `GET /health`

```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:45.123Z",
  "uptime": 3600,
  "modules": {
    "loaded": 5,
    "active": 5
  },
  "instances": {
    "active": 23,
    "idle": 7
  },
  "memory": {
    "used": 67108864,
    "available": 134217728
  },
  "redis": {
    "status": "connected",
    "latency": 2
  }
}
```

## Security

### Sandboxing
- WebAssembly provides memory safety and type safety
- WASI provides controlled system access
- Capability-based security model
- Resource limits enforce isolation

### Authentication
- JWT-based authentication for API access
- Module signature verification
- Secure module distribution
- User-based access control

### Network Security
- TLS encryption for all communications
- Network isolation for WASM modules
- Firewall rules for edge nodes
- VPN support for private networks

## Performance Tuning

### Memory Optimization
```javascript
// Configure memory limits
const config = {
  memory: {
    initial: 256,  // Initial pages (16MB)
    maximum: 1024, // Maximum pages (64MB)
    shared: false
  }
};
```

### CPU Optimization
```javascript
// Configure execution limits
const context = {
  timeout: 5000,      // 5 second timeout
  maxCpuTime: 3000,   // 3 second CPU time
  priority: 'high'    // Execution priority
};
```

### Caching Strategy
```javascript
// Module compilation caching
const cache = new ModuleCache({
  maxSize: 100,       // Cache up to 100 compiled modules
  ttl: 3600000       // 1 hour TTL
});
```

## Troubleshooting

### Common Issues

**Module Loading Failures**:
```bash
# Check module validity
wasm-validate module.wasm

# Check module exports
wasm-objdump -x module.wasm
```

**Performance Issues**:
```bash
# Profile module execution
node --prof --prof-process edge-runtime.js

# Monitor memory usage
top -p $(pgrep node)
```

**Network Connectivity**:
```bash
# Test WebSocket connection
wscat -c ws://localhost:3001/health

# Check Redis connection
redis-cli -h localhost ping
```

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=debug
DEBUG=edge:*
```

### Performance Profiling

```javascript
// Enable profiling
const profiler = new EdgeProfiler();
profiler.start();

// Execute functions
const result = await executeFunction(instanceId, 'process_data', args);

// Get profile report
const report = profiler.getReport();
console.log('Execution profile:', report);
```

## Contributing

### Development Setup
1. Fork the repository
2. Install dependencies: `npm install`
3. Start development server: `npm run dev`
4. Run tests: `npm test`
5. Build for production: `npm run build`

### Testing
```bash
# Unit tests
npm run test:unit

# Integration tests
npm run test:integration

# Load tests
npm run test:load

# WASM module tests
npm run test:wasm
```

### Code Style
- TypeScript with strict mode
- ESLint and Prettier
- Conventional commits
- 100% test coverage for core functions

## Roadmap

### Phase 1 (Current)
- ✅ Basic WebAssembly runtime
- ✅ Module loading and execution
- ✅ Security sandboxing
- ✅ REST API endpoints

### Phase 2 (Q2 2024)
- 🔄 GPU acceleration support
- 🔄 Advanced caching strategies
- 🔄 Multi-tenant isolation
- 🔄 Load balancing improvements

### Phase 3 (Q3 2024)
- 📋 Distributed edge clusters
- 📋 Auto-scaling based on demand
- 📋 Advanced debugging tools
- 📋 Visual module designer

### Phase 4 (Q4 2024)
- 📋 Edge AI model optimization
- 📋 Real-time streaming support
- 📋 Enhanced security features
- 📋 Enterprise management console

## License

MIT License - see LICENSE file for details.

## Support

- **Documentation**: [docs.uap.com/edge](https://docs.uap.com/edge)
- **Issues**: [GitHub Issues](https://github.com/uap/platform/issues)
- **Discord**: [UAP Community](https://discord.gg/uap)
- **Email**: edge-support@uap.com