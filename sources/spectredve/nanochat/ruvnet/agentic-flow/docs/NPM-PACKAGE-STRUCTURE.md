# NPM Package Structure - Agentic Flow

## 📦 Package Overview

**Package Name**: `agentic-flow`
**Version**: 1.10.0
**Author**: ruv (@ruvnet)
**License**: MIT
**Repository**: https://github.com/ruvnet/agentic-flow

## 🏗️ Package Structure

```
agentic-flow/
├── package.json                      # Root package configuration
├── tsconfig.json                     # Root TypeScript config
├── .npmrc                            # NPM configuration
├── .npmignore                        # NPM publish exclusions
├── LICENSE                           # MIT license
├── README.md                         # Main documentation
├── CHANGELOG.md                      # Version history
│
├── agentic-flow/                     # Main package directory
│   ├── package.json                  # Main package metadata
│   ├── config/
│   │   └── tsconfig.json            # Build TypeScript config
│   ├── src/                         # Source code (excluded from npm)
│   │   ├── index.ts
│   │   ├── agentdb/
│   │   ├── reasoningbank/
│   │   ├── router/
│   │   ├── agent-booster/
│   │   ├── transport/quic.ts
│   │   └── cli-proxy.ts
│   ├── dist/                        # Built output (published)
│   │   ├── index.js
│   │   ├── index.d.ts
│   │   ├── cli-proxy.js
│   │   └── ...
│   ├── docs/                        # Documentation (published)
│   ├── .claude/                     # Claude Code config (published)
│   ├── wasm/                        # WASM binaries (published)
│   ├── certs/                       # TLS certificates (published)
│   └── scripts/
│       └── postinstall.js           # Post-install hook (published)
│
├── agent-booster/                    # Sub-package (published dist only)
│   ├── package.json
│   └── dist/
│
├── reasoningbank/                    # Sub-package (published dist only)
│   ├── package.json
│   └── dist/
│
├── scripts/                          # Build/publish scripts (not published)
│   ├── build-all.sh
│   ├── verify-package.sh
│   └── quick-publish.sh
│
└── docs/                             # Documentation (not published)
    ├── PUBLISHING.md
    └── NPM-PACKAGE-STRUCTURE.md
```

## 📋 Package Configuration

### Root package.json

```json
{
  "name": "agentic-flow",
  "version": "1.10.0",
  "description": "Production-ready AI agent orchestration platform...",
  "type": "module",
  "main": "agentic-flow/dist/index.js",
  "types": "agentic-flow/dist/index.d.ts",
  "bin": {
    "agentic-flow": "agentic-flow/dist/cli-proxy.js",
    "agentdb": "agentic-flow/dist/agentdb/cli/agentdb-cli.js"
  },
  "exports": {
    ".": "./agentic-flow/dist/index.js",
    "./reasoningbank": "./agentic-flow/dist/reasoningbank/index.js",
    "./router": "./agentic-flow/dist/router/index.js",
    "./agent-booster": "./agentic-flow/dist/agent-booster/index.js",
    "./transport/quic": "./agentic-flow/dist/transport/quic.js",
    "./agentdb": "./agentic-flow/dist/agentdb/index.js"
  },
  "files": [
    "agentic-flow/dist",
    "agentic-flow/docs",
    "agentic-flow/.claude",
    "agentic-flow/wasm",
    "agentic-flow/certs",
    "agentic-flow/scripts",
    "agent-booster/dist",
    "reasoningbank/dist",
    "README.md",
    "LICENSE",
    "CHANGELOG.md"
  ]
}
```

### Entry Points

#### CLI Commands

```bash
# Main CLI
agentic-flow --agent coder --task "Build API"

# AgentDB CLI
agentdb reflexion store "session-1" "task-1" 0.95 true "Success"
```

#### Programmatic Imports

```javascript
// Main package
import AgenticFlow from 'agentic-flow';

// ReasoningBank
import * as reasoningbank from 'agentic-flow/reasoningbank';

// Model Router
import { ModelRouter } from 'agentic-flow/router';

// Agent Booster
import { AgentBooster } from 'agentic-flow/agent-booster';

// QUIC Transport
import { QuicTransport } from 'agentic-flow/transport/quic';

// AgentDB
import { ReflexionMemory, SkillLibrary } from 'agentic-flow/agentdb';
```

## 🔧 Build System

### TypeScript Configuration

**Root tsconfig.json** (for development)
- Strict type checking
- Full source maps
- Declaration files

**agentic-flow/config/tsconfig.json** (for production build)
- Relaxed strict mode for compatibility
- Optimized output
- Bundler module resolution

### Build Scripts

```bash
# Build all packages
npm run build
# Internally runs:
#   - npm run build:main      (builds agentic-flow)
#   - npm run build:packages  (builds agent-booster, reasoningbank)

# Build main package only
npm run build:main

# Build with verification
bash scripts/build-all.sh
```

## 📦 Publishing Workflow

### 1. Automated Verification

```bash
# Verify package is ready
bash scripts/verify-package.sh

# Checks:
# ✓ package.json validity
# ✓ Version format (semver)
# ✓ Required fields (name, author, license)
# ✓ Built files exist
# ✓ CLI executables have shebangs
# ✓ No secrets in code
# ✓ Documentation present
# ✓ Package size < 10MB
```

### 2. Quick Publish

```bash
# Full publish workflow (lint, build, test, publish)
bash scripts/quick-publish.sh

# Dry run (preview without publishing)
bash scripts/quick-publish.sh --dry-run

# Skip tests (faster)
bash scripts/quick-publish.sh --skip-tests
```

### 3. Manual Publishing

```bash
# 1. Build
npm run build

# 2. Test
npm test

# 3. Preview what will be published
npm pack --dry-run

# 4. Publish
npm publish
```

## 🔍 Quality Assurance

### Pre-Publish Checks

- ✅ TypeScript compilation succeeds
- ✅ All tests pass
- ✅ No linting errors
- ✅ No secrets in code
- ✅ Package size optimized
- ✅ Dependencies audited
- ✅ Documentation updated

### Files Excluded from Package

Via `.npmignore`:
- Source code (`src/`, `*.ts`)
- Tests (`tests/`, `*.test.*`)
- Build artifacts (`*.tsbuildinfo`, `target/`)
- Docker files (`Dockerfile*`, `docker-compose*.yml`)
- CI/CD configs (`.github/`)
- Development tools (`benchmarks/`, `examples/`)
- Rust artifacts (`*.rs`, `Cargo.toml`, `*.rlib`)
- Database files (`*.db`, `*.sqlite`)
- Environment files (`.env*`)
- Large directories (`node_modules/`, `crates/`)

### Files Included in Package

Via `files` field in package.json:
- ✅ `agentic-flow/dist/` - Main compiled code
- ✅ `agentic-flow/docs/` - API documentation
- ✅ `agentic-flow/.claude/` - Claude Code configuration
- ✅ `agentic-flow/wasm/` - WebAssembly binaries
- ✅ `agentic-flow/certs/` - TLS certificates
- ✅ `agentic-flow/scripts/postinstall.js` - Post-install hook
- ✅ `agent-booster/dist/` - Agent Booster compiled code
- ✅ `reasoningbank/dist/` - ReasoningBank compiled code
- ✅ `README.md` - Main documentation
- ✅ `LICENSE` - MIT license
- ✅ `CHANGELOG.md` - Version history

## 🚀 Installation & Usage

### Installation

```bash
# Global installation
npm install -g agentic-flow

# Local installation
npm install agentic-flow

# With optional MCP servers
npm install agentic-flow claude-flow flow-nexus
```

### Verification

```bash
# Check installation
agentic-flow --version
agentdb --version

# Test CLI
agentic-flow --list
agentdb --help

# Test programmatic import
node -e "const af = require('agentic-flow'); console.log('✓ Loaded successfully');"
```

## 📊 Package Metrics

### Size Optimization

| Component | Size |
|-----------|------|
| Main package | ~2-3 MB |
| Agent Booster | ~500 KB |
| ReasoningBank | ~1-2 MB |
| **Total** | **~4-6 MB** |

### Dependencies

**Production Dependencies**: 24
- Core: @anthropic-ai/claude-agent-sdk, @anthropic-ai/sdk
- Memory: agentdb, better-sqlite3
- LLM: @google/genai, @xenova/transformers
- Web: express, axios, ws
- Utils: dotenv, tiktoken, zod, yaml

**Peer Dependencies** (optional): 2
- claude-flow (^2.7.0)
- flow-nexus (^1.0.0)

## 🔐 Security

### Secret Detection

The package includes automated secret scanning:
```bash
# Scan for API keys
grep -r "sk-ant-" . --exclude-dir=node_modules
grep -r "ANTHROPIC_API_KEY.*sk-ant" . --exclude-dir=node_modules
```

### Environment Variables

Never hardcoded in package:
- `ANTHROPIC_API_KEY`
- `OPENROUTER_API_KEY`
- `GOOGLE_API_KEY`
- Database credentials
- API endpoints with secrets

## 📚 Documentation

- **README.md**: Overview, quick start, features
- **CHANGELOG.md**: Version history and changes
- **LICENSE**: MIT license
- **docs/PUBLISHING.md**: Complete publishing guide
- **docs/NPM-PACKAGE-STRUCTURE.md**: This document
- **agentic-flow/docs/**: API documentation and examples

## 🔗 Links

- **NPM**: https://www.npmjs.com/package/agentic-flow
- **GitHub**: https://github.com/ruvnet/agentic-flow
- **Issues**: https://github.com/ruvnet/agentic-flow/issues
- **Author**: [@ruvnet](https://github.com/ruvnet)

## 🛠️ Maintenance

### Version Bumping

```bash
# Patch version (1.10.0 -> 1.10.1)
npm version patch

# Minor version (1.10.0 -> 1.11.0)
npm version minor

# Major version (1.10.0 -> 2.0.0)
npm version major
```

### Publishing Checklist

- [ ] Version bumped in all package.json files
- [ ] CHANGELOG.md updated
- [ ] All tests passing
- [ ] Build successful
- [ ] No secrets in code
- [ ] Dependencies audited
- [ ] Documentation updated
- [ ] Package verified (`bash scripts/verify-package.sh`)
- [ ] Dry run successful (`npm publish --dry-run`)
- [ ] Published (`npm publish`)
- [ ] GitHub release created
- [ ] Installation tested

---

**Maintained by**: ruv (@ruvnet)
**Last Updated**: 2025-01-08
