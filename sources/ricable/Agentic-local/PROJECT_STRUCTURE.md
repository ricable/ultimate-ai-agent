# Project Structure

## Overview

This repository contains a complete implementation of the Sovereign Agentic Stack, organized into clear functional layers.

## Directory Structure

```
Agentic-local/
├── README.md                          # Main project documentation
├── LICENSE                            # MIT License
├── CONTRIBUTING.md                    # Contribution guidelines
├── PROJECT_STRUCTURE.md               # This file
├── package.json                       # Node.js dependencies (ruvnet ecosystem)
├── .env.example                       # Environment configuration template
├── .gitignore                         # Git ignore rules
├── mcp-config.json                    # Model Context Protocol configuration
│
├── scripts/                           # Setup and installation scripts
│   ├── setup-wasmedge-mlx.sh         # Build WasmEdge with MLX backend
│   ├── setup-llamaedge.sh            # Install LlamaEdge inference server
│   ├── download-qwen-coder.sh        # Download Qwen 2.5 Coder models
│   └── setup-gaianet.sh              # Configure GaiaNet node
│
├── src/                               # Source code
│   ├── orchestration/                 # Agent orchestration layer
│   │   ├── basic-agent.js            # Simple agent examples
│   │   └── swarm-agent.js            # Multi-agent swarm examples
│   │
│   ├── sandbox/                       # Code execution sandbox
│   │   ├── docker-sandbox.js         # Docker container manager
│   │   └── test-sandbox.js           # Security test suite
│   │
│   ├── inference/                     # Inference layer (reserved)
│   │   └── (future: model management)
│   │
│   └── utils/                         # Utilities (reserved)
│       └── (future: logging, monitoring)
│
└── docs/                              # Documentation
    ├── technical-analysis/            # Deep technical analysis
    │   └── sovereign-agentic-architectures.md  # 8000+ word analysis
    │
    ├── setup-guides/                  # Setup and configuration guides
    │   ├── gaianet-monetization.md   # How to earn crypto with GaiaNet
    │   └── sandbox-security.md       # Security model and best practices
    │
    └── examples/                      # Usage examples
        └── quickstart.md             # 10-minute quick start guide
```

## File Purposes

### Root Configuration

| File | Purpose |
|------|---------|
| `package.json` | Node.js dependencies (agentic-flow, claude-flow, ruv-swarm, etc.) |
| `.env.example` | Environment variable template (copy to `.env` and configure) |
| `mcp-config.json` | Model Context Protocol servers (Docker sandbox, filesystem) |
| `.gitignore` | Excludes node_modules, models, temporary files from git |

### Scripts (`scripts/`)

Automated setup scripts for each component:

1. **`setup-wasmedge-mlx.sh`** - Builds WasmEdge from source with MLX backend enabled
2. **`setup-llamaedge.sh`** - Installs LlamaEdge WASM binaries and creates launcher
3. **`download-qwen-coder.sh`** - Downloads Qwen 2.5 Coder models (7B/14B/32B)
4. **`setup-gaianet.sh`** - Initializes GaiaNet node with Qwen configuration

All scripts are executable (`chmod +x`) and include extensive error checking.

### Source Code (`src/`)

#### Orchestration Layer

- **`basic-agent.js`** - Demonstrates:
  - Simple code generation
  - Iterative development with feedback
  - Multi-step reasoning tasks
  - Integration with local GaiaNet node

- **`swarm-agent.js`** - Demonstrates:
  - Hierarchical swarms (Queen/Drone pattern)
  - Mesh networks (parallel processing)
  - Star networks (code review)
  - Multi-agent collaboration

#### Sandbox Layer

- **`docker-sandbox.js`** - Production-ready Docker container manager
  - JavaScript/Python/TypeScript execution
  - Network isolation
  - Resource limits
  - Security hardening

- **`test-sandbox.js`** - Comprehensive security test suite
  - 7 security tests
  - Network isolation verification
  - Resource limit validation
  - File system protection checks

### Documentation (`docs/`)

#### Technical Analysis

- **`sovereign-agentic-architectures.md`** - Complete technical deep-dive covering:
  - Ruvnet ecosystem analysis
  - WasmEdge + MLX integration
  - GaiaNet economic model
  - Security architecture
  - System integration

#### Setup Guides

- **`gaianet-monetization.md`** - Comprehensive monetization guide:
  - Economic model explanation
  - Setup instructions
  - Earning potential calculations
  - Dual-mode operation (local + earning)
  - Performance optimization

- **`sandbox-security.md`** - Security deep-dive:
  - Threat model
  - Defense-in-depth architecture
  - Configuration options
  - Testing procedures
  - Incident response

#### Examples

- **`quickstart.md`** - 10-minute getting started guide:
  - Prerequisites
  - Installation steps
  - First agent example
  - Common workflows
  - Troubleshooting

## Key Technologies

### Dependencies (from `package.json`)

**Ruvnet Ecosystem:**
- `agentic-flow` (1.7.7) - Swarm orchestration
- `claude-flow` (2.7.10) - Enterprise workflows
- `ruv-swarm` (1.0.20) - Neural swarm ops
- `strange-loops` - Emergent intelligence
- `@agentics.org/sparc2` (2.0.25) - SPARC methodology

**Supporting Libraries:**
- `dotenv` - Environment configuration
- `express` - Web server (for examples)
- `ws` - WebSocket support

**Dev Tools:**
- `npm-check-updates` - Dependency management
- `nodemon` - Development server

### External Dependencies

**Required (installed via scripts):**
- WasmEdge (built from source)
- LlamaEdge (WASM binaries)
- Qwen 2.5 Coder models (GGUF format)
- Docker Desktop

**Optional:**
- GaiaNet node (for monetization)

## Workflow

### Development Workflow

1. **Setup**: Run scripts in `scripts/` directory
2. **Configure**: Copy `.env.example` to `.env`, adjust settings
3. **Start Inference**: `gaianet start` or `llamaedge`
4. **Run Agents**: Execute files in `src/orchestration/`
5. **Test Security**: `npm run sandbox:test`

### Production Deployment

1. Ensure WasmEdge + MLX backend functional
2. Download production model (14B or 32B)
3. Configure `.env` for production settings
4. Register GaiaNet node for monetization
5. Set up monitoring (logs, uptime)
6. Run agents with production error handling

## Extensibility

The architecture is designed for easy extension:

### Adding New Languages to Sandbox

Edit `src/sandbox/docker-sandbox.js`:

```javascript
case 'rust':
  const rsFile = join(sessionDir, 'code.rs');
  writeFileSync(rsFile, code);
  return {
    filename: 'code.rs',
    command: ['rustc', 'code.rs', '-o', 'code', '&&', './code']
  };
```

### Adding New Agent Types

Create new file in `src/orchestration/`:

```javascript
import { AgenticFlow } from 'agentic-flow';

export class MyCustomAgent extends AgenticFlow {
  // Custom agent logic
}
```

### Adding New MCP Tools

Edit `mcp-config.json`:

```json
{
  "mcpServers": {
    "my-custom-tool": {
      "command": "npx",
      "args": ["-y", "@my-org/my-tool"]
    }
  }
}
```

## Maintenance

### Updating Dependencies

```bash
# Check for ruvnet updates
npm run check-updates

# Update to latest versions
npm run update-ruvnet

# Install updates
npm install
```

### Updating Models

```bash
# Re-run download script to get latest model
./scripts/download-qwen-coder.sh
```

### Updating WasmEdge

```bash
# Re-run build script (pulls latest from git)
./scripts/setup-wasmedge-mlx.sh
```

## Testing

### Manual Testing

```bash
# Test sandbox security
npm run sandbox:test

# Test basic agent
node src/orchestration/basic-agent.js

# Test swarm agent
node src/orchestration/swarm-agent.js
```

### Automated Testing

Currently manual. Future: CI/CD pipeline with automated tests.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - see [LICENSE](LICENSE) file.

---

**This structure provides a complete sovereign AI stack from infrastructure (WasmEdge) through orchestration (ruvnet) to monetization (GaiaNet).**
