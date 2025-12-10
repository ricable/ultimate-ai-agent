# DAA SDK Templates

Comprehensive project templates for building DAA (Distributed Agentic Architecture) applications.

## Overview

The DAA SDK provides three production-ready templates to help you get started quickly:

### 1. Basic Template (`basic/`)

**Best for:** Learning DAA fundamentals, simple crypto applications

A minimal template demonstrating quantum-resistant cryptography:

- **ML-KEM-768**: Post-quantum key encapsulation mechanism
- **ML-DSA**: Digital signature algorithm
- **BLAKE3**: High-performance hashing
- **Password Vault**: Secure credential storage
- **Quantum Fingerprinting**: Data integrity verification

**Difficulty:** Beginner
**Files:** 5 (package.json, tsconfig.json, src/index.ts, .gitignore, README.md)
**Lines of Code:** ~150

### 2. Full-Stack Template (`full-stack/`)

**Best for:** Building autonomous agent systems, workflow orchestration

Complete DAA ecosystem with orchestrator and networking:

- **MRAP Orchestrator**: Monitor-Reason-Act-Plan autonomy loop
- **Workflow Engine**: Sequential, parallel, conditional, event-driven patterns
- **QuDAG Network**: P2P networking with quantum-resistant encryption
- **Rules Engine**: Declarative business logic
- **Token Economy**: rUv token management with dynamic fees
- **Event Bus**: Pub/sub messaging

**Difficulty:** Intermediate
**Files:** 8 (main + 4 example files)
**Lines of Code:** ~800

### 3. ML Training Template (`ml-training/`)

**Best for:** Federated learning, privacy-preserving AI training

Federated machine learning with distributed training:

- **Federated Learning**: Coordinate training across multiple nodes
- **Privacy Mechanisms**: Differential privacy + secure aggregation
- **Multiple Architectures**: GPT-Mini, BERT-Tiny, ResNet-18
- **Aggregation Strategies**: FedAvg, FedProx, FedYogi
- **Training Utilities**: Data loading, preprocessing, augmentation
- **Distributed Inference**: Production deployment

**Difficulty:** Advanced
**Files:** 9 (main + 4 utility modules)
**Lines of Code:** ~1,200

## Quick Start

### Interactive Setup

```bash
npx daa-sdk init
```

Follow the interactive prompts to:
1. Choose project name
2. Select template
3. Configure options (TypeScript, Git, dependencies)

### Command-Line Setup

```bash
# Basic template
npx daa-sdk init my-agent --template basic

# Full-stack template
npx daa-sdk init my-system --template full-stack

# ML training template
npx daa-sdk init my-ml-project --template ml-training
```

### Options

```bash
npx daa-sdk init <name> [options]

Options:
  -t, --template <type>   Template (basic|full-stack|ml-training)
  --no-install           Skip dependency installation
  --no-git              Skip git initialization
  --typescript          Use TypeScript (default)
  --javascript          Use JavaScript
```

## Template Structure

### Basic Template

```
basic/
├── package.json          # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── .gitignore           # Git ignore rules
├── README.md            # Documentation
└── src/
    └── index.ts         # Main application with examples
```

### Full-Stack Template

```
full-stack/
├── package.json          # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── .gitignore           # Git ignore rules
├── README.md            # Documentation
└── src/
    ├── index.ts         # Main application
    ├── orchestrator.ts  # MRAP examples
    ├── qudag.ts        # Network examples
    └── workflows.ts    # Workflow patterns
```

### ML Training Template

```
ml-training/
├── package.json          # Dependencies and scripts
├── tsconfig.json         # TypeScript configuration
├── .gitignore           # Git ignore rules
├── README.md            # Documentation
└── src/
    ├── index.ts         # Main federated learning demo
    ├── training-node.ts # Training node implementation
    ├── model.ts        # Model definitions
    └── data-loader.ts  # Data utilities
```

## Feature Comparison

| Feature | Basic | Full-Stack | ML Training |
|---------|-------|------------|-------------|
| Quantum-Resistant Crypto | ✅ | ✅ | ✅ |
| Password Vault | ✅ | ✅ | ✅ |
| MRAP Orchestrator | ❌ | ✅ | ❌ |
| Workflow Engine | ❌ | ✅ | ❌ |
| P2P Networking | ❌ | ✅ | ✅ |
| Token Economy | ❌ | ✅ | ❌ |
| Federated Learning | ❌ | ❌ | ✅ |
| Differential Privacy | ❌ | ❌ | ✅ |
| Model Training | ❌ | ❌ | ✅ |

## Use Cases

### Basic Template

- Learning DAA SDK fundamentals
- Building secure communication apps
- Password management tools
- Data integrity verification
- Quantum-resistant authentication

### Full-Stack Template

- Autonomous agent systems
- Workflow automation
- Multi-agent coordination
- Distributed task execution
- Token-based economies
- Event-driven architectures

### ML Training Template

- Federated machine learning
- Privacy-preserving AI
- Healthcare data analysis (HIPAA-compliant)
- Financial fraud detection
- Edge device learning
- Collaborative training without data sharing

## CLI Commands

### List Templates

```bash
npx daa-sdk templates
```

Shows detailed information about all available templates.

### Show Examples

```bash
# General examples
npx daa-sdk examples

# Template-specific examples
npx daa-sdk examples --template basic
npx daa-sdk examples --template full-stack
npx daa-sdk examples --template ml-training
```

### Platform Info

```bash
npx daa-sdk info
```

Display platform capabilities, available bindings, and performance characteristics.

## Development Workflow

### After Creating a Project

```bash
# Navigate to project
cd my-project

# Install dependencies (if not auto-installed)
npm install

# Build project
npm run build

# Run in development mode
npm run dev

# Run production build
npm start

# Run tests
npm test

# Lint code
npm run lint

# Format code
npm run format
```

### Full-Stack Specific

```bash
# Run orchestrator examples
npm run test:orchestrator

# Run QuDAG network examples
npm run test:qudag

# Run workflow examples
npm run test:workflows
```

### ML Training Specific

```bash
# Run federated learning demo
npm start

# Run training node
npm run train
```

## Performance

### Native vs WASM

All templates support both native NAPI-rs bindings and WebAssembly:

- **Native (Node.js)**: 100% performance, optimal for production
- **WASM (Browser)**: ~40% of native speed, full cross-platform compatibility

Check your runtime:

```bash
npx daa-sdk info
```

### Template Performance

| Template | Startup Time | Memory Usage | Throughput |
|----------|-------------|--------------|------------|
| Basic | <100ms | ~50MB | High |
| Full-Stack | ~200ms | ~150MB | Very High |
| ML Training | ~500ms | ~500MB+ | Variable |

## Security

All templates include:

- **Post-Quantum Cryptography**: NIST-standardized algorithms
- **Memory Safety**: Rust-based implementation
- **Constant-Time Operations**: Side-channel resistant
- **Zero-Copy Optimization**: Minimal memory overhead

Additional security in specific templates:

- **Full-Stack**: Encrypted P2P communication, signature verification
- **ML Training**: Differential privacy, secure aggregation, local data protection

## Customization

### Modifying Templates

Templates are designed to be customized:

1. **Add Dependencies**: Update `package.json`
2. **Configure Build**: Modify `tsconfig.json`
3. **Add Features**: Extend source files
4. **Update Tests**: Add test files

### Creating Custom Templates

To create your own template:

1. Create directory in `templates/`
2. Add required files (package.json, tsconfig.json, src/, README.md)
3. Register in `cli/templates.ts`
4. Test with `npx daa-sdk init`

## Best Practices

1. **Start Simple**: Begin with the Basic template
2. **Learn Patterns**: Study the Full-Stack examples
3. **Scale Up**: Use ML Training for distributed systems
4. **Read Docs**: Each template has comprehensive README
5. **Run Examples**: Execute built-in examples to understand features
6. **Customize**: Adapt templates to your specific needs

## Troubleshooting

### Installation Issues

```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

### Build Errors

```bash
# Clean build
rm -rf dist/
npm run build
```

### Runtime Issues

```bash
# Check platform compatibility
npx daa-sdk info

# Force WASM fallback
DAA_FORCE_WASM=1 npm start
```

## Resources

- [DAA SDK Documentation](https://github.com/ruvnet/daa)
- [API Reference](https://github.com/ruvnet/daa/tree/main/packages/daa-sdk)
- [NAPI-rs Guide](https://github.com/ruvnet/daa/blob/main/docs/NAPI_INTEGRATION_PLAN.md)
- [Examples Repository](https://github.com/ruvnet/daa/tree/main/examples)

## Contributing

We welcome contributions! To add a new template:

1. Fork the repository
2. Create template in `packages/daa-sdk/templates/`
3. Update `cli/templates.ts`
4. Add tests and documentation
5. Submit pull request

## License

MIT License - see [LICENSE](../../../LICENSE) file for details.

## Support

- GitHub Issues: https://github.com/ruvnet/daa/issues
- Documentation: https://github.com/ruvnet/daa
- Community: Join our discussions

---

**Total Templates:** 3
**Total Files:** 22
**Total Lines of Code:** ~2,800
**Coverage:** Beginner to Advanced
**Last Updated:** 2025-11-11
