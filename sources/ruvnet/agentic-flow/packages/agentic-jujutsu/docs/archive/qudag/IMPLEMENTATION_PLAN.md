# QUDAG + Agentic-Jujutsu Implementation Plan
## NPM/NPX Distribution with WASM + TypeScript

**Version**: 1.0.0
**Date**: 2025-11-09
**Package Name**: `@agentic-flow/qudag-jujutsu`
**Distribution**: npm registry with WASM + TypeScript

---

## Executive Summary

This document provides a **detailed, actionable implementation plan** for building and distributing **QUDAG + Agentic-Jujutsu** as a production-ready **npm/npx package** with **WebAssembly (WASM)** core and **TypeScript** facade.

**Timeline**: 12 weeks to v1.0.0 on npm
**Effort**: 12-15 person-weeks
**Budget**: ~$8,000 infrastructure + tooling
**Distribution**: Zero-install via `npx`, global via `npm install -g`

---

## Table of Contents

1. [Package Specifications](#package-specifications)
2. [Build Pipeline Architecture](#build-pipeline-architecture)
3. [Phase-by-Phase Implementation](#phase-by-phase-implementation)
4. [WASM Optimization Strategy](#wasm-optimization-strategy)
5. [TypeScript Integration](#typescript-integration)
6. [Testing Strategy](#testing-strategy)
7. [CI/CD Pipeline](#cicd-pipeline)
8. [npm Publishing Process](#npm-publishing-process)
9. [Documentation Requirements](#documentation-requirements)
10. [Maintenance & Updates](#maintenance--updates)

---

## Package Specifications

### Package Identity

```json
{
  "name": "@agentic-flow/qudag-jujutsu",
  "version": "0.1.0",
  "description": "Quantum-resistant distributed multi-agent collaborative coding with WASM + TypeScript",
  "keywords": [
    "qudag", "jujutsu", "vcs", "quantum-resistant", "wasm", "typescript",
    "multi-agent", "dag", "consensus", "distributed", "ai-agents"
  ],
  "license": "MIT",
  "author": "Agentic Flow Team <hello@agentic-flow.io>",
  "homepage": "https://github.com/ruvnet/agentic-flow/tree/main/packages/qudag-jujutsu",
  "repository": {
    "type": "git",
    "url": "https://github.com/ruvnet/agentic-flow.git",
    "directory": "packages/qudag-jujutsu"
  },
  "bugs": {
    "url": "https://github.com/ruvnet/agentic-flow/issues"
  }
}
```

### Multi-Target Exports

```json
{
  "main": "./dist/node/index.js",
  "module": "./dist/node/index.mjs",
  "types": "./dist/node/index.d.ts",
  "browser": "./dist/web/index.js",
  "bin": {
    "qudag-jujutsu": "./dist/cli/qudag-jujutsu.js",
    "qj": "./dist/cli/qudag-jujutsu.js"
  },
  "exports": {
    ".": {
      "types": "./dist/node/index.d.ts",
      "import": "./dist/node/index.mjs",
      "require": "./dist/node/index.js",
      "browser": "./dist/web/index.js"
    },
    "./web": {
      "types": "./dist/web/index.d.ts",
      "import": "./dist/web/index.js",
      "default": "./dist/web/index.js"
    },
    "./node": {
      "types": "./dist/node/index.d.ts",
      "import": "./dist/node/index.mjs",
      "require": "./dist/node/index.js"
    },
    "./package.json": "./package.json"
  },
  "files": [
    "dist/",
    "README.md",
    "LICENSE",
    "CHANGELOG.md"
  ]
}
```

### Dependencies

```json
{
  "dependencies": {},
  "devDependencies": {
    "@types/node": "^20.10.0",
    "@typescript-eslint/eslint-plugin": "^6.13.0",
    "@typescript-eslint/parser": "^6.13.0",
    "eslint": "^8.55.0",
    "jest": "^29.7.0",
    "prettier": "^3.1.0",
    "ts-jest": "^29.1.0",
    "typedoc": "^0.25.0",
    "typescript": "^5.3.0",
    "wasm-pack": "^0.12.1",
    "wasm-opt": "^1.4.0"
  },
  "peerDependencies": {},
  "engines": {
    "node": ">=16.0.0",
    "npm": ">=8.0.0"
  }
}
```

---

## Build Pipeline Architecture

### Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Source Code                               │
├───────────────┬──────────────────┬─────────────────────────┤
│  Rust (core)  │  TypeScript (API)│  Shell (scripts)        │
│   src/*.rs    │  src/ts/*.ts     │  scripts/*.sh           │
└───────┬───────┴────────┬─────────┴──────────┬──────────────┘
        │                │                    │
        ▼                ▼                    ▼
┌───────────────┐ ┌──────────────┐  ┌─────────────────┐
│  wasm-pack    │ │     tsc      │  │   Build Utils   │
│  (Rust→WASM)  │ │ (TS→JS+DTS)  │  │  (Optimization) │
└───────┬───────┘ └──────┬───────┘  └────────┬────────┘
        │                │                    │
        ▼                ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                  Build Artifacts                             │
├──────────────┬──────────────┬──────────────┬───────────────┤
│  dist/web/   │ dist/node/   │ dist/bundler/│  dist/cli/    │
│  (browser)   │ (Node.js)    │ (webpack)    │  (executable) │
└──────────────┴──────────────┴──────────────┴───────────────┘
        │                │                │           │
        └────────────────┴────────────────┴───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   npm package        │
              │  (tar.gz archive)    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   npm registry       │
              │  (public/private)    │
              └──────────────────────┘
```

### Build Scripts

**`scripts/build.sh`** (Master Build Script):

```bash
#!/bin/bash
set -e

echo "🏗️  Starting build pipeline for @agentic-flow/qudag-jujutsu"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf dist/ pkg/

# Create dist directories
mkdir -p dist/{web,node,bundler,cli}

# ===========================
# Step 1: Build Rust → WASM
# ===========================

echo ""
echo "🦀 Building Rust core to WASM..."

# Web target (browser)
echo "  📦 Building for web (browser)..."
wasm-pack build \
  --target web \
  --out-dir pkg/web \
  --release \
  --scope agentic-flow \
  -- --features "wasm,qudag,agentdb"

# Node.js target
echo "  📦 Building for Node.js..."
wasm-pack build \
  --target nodejs \
  --out-dir pkg/node \
  --release \
  --scope agentic-flow \
  -- --features "wasm,qudag,agentdb"

# Bundler target (webpack, vite, rollup)
echo "  📦 Building for bundlers..."
wasm-pack build \
  --target bundler \
  --out-dir pkg/bundler \
  --release \
  --scope agentic-flow \
  -- --features "wasm,qudag,agentdb"

# ===========================
# Step 2: Optimize WASM
# ===========================

echo ""
echo "⚡ Optimizing WASM binaries with wasm-opt..."

# Optimize web build
wasm-opt -Oz --enable-simd \
  -o dist/web/qudag_jujutsu_bg.wasm \
  pkg/web/qudag_jujutsu_bg.wasm

# Optimize Node.js build
wasm-opt -Oz --enable-simd \
  -o dist/node/qudag_jujutsu_bg.wasm \
  pkg/node/qudag_jujutsu_bg.wasm

# Optimize bundler build
wasm-opt -Oz --enable-simd \
  -o dist/bundler/qudag_jujutsu_bg.wasm \
  pkg/bundler/qudag_jujutsu_bg.wasm

# ===========================
# Step 3: Copy JS shims
# ===========================

echo ""
echo "📄 Copying JavaScript shims..."

cp pkg/web/qudag_jujutsu.js dist/web/
cp pkg/node/qudag_jujutsu.js dist/node/
cp pkg/bundler/qudag_jujutsu.js dist/bundler/

# ===========================
# Step 4: Build TypeScript
# ===========================

echo ""
echo "🔷 Building TypeScript facade..."

# Compile TypeScript
tsc -p tsconfig.json

# Build CLI
echo "  🖥️  Building CLI..."
tsc -p tsconfig.cli.json

# Make CLI executable
chmod +x dist/cli/qudag-jujutsu.js

# ===========================
# Step 5: Generate types
# ===========================

echo ""
echo "📝 Generating TypeScript definitions..."

# Copy WASM-generated types
cp pkg/web/qudag_jujutsu.d.ts dist/web/
cp pkg/node/qudag_jujutsu.d.ts dist/node/
cp pkg/bundler/qudag_jujutsu.d.ts dist/bundler/

# Bundle with hand-written types
cat src/ts/types.d.ts >> dist/node/index.d.ts
cat src/ts/types.d.ts >> dist/web/index.d.ts

# ===========================
# Step 6: Verify builds
# ===========================

echo ""
echo "✅ Verifying builds..."

# Check file sizes
echo "📊 Package sizes:"
echo "  Web WASM:     $(du -h dist/web/qudag_jujutsu_bg.wasm | cut -f1)"
echo "  Node WASM:    $(du -h dist/node/qudag_jujutsu_bg.wasm | cut -f1)"
echo "  Bundler WASM: $(du -h dist/bundler/qudag_jujutsu_bg.wasm | cut -f1)"
echo "  Total:        $(du -sh dist/ | cut -f1)"

# Verify WASM validity
echo ""
echo "🔍 Validating WASM modules..."
wasm-validate dist/web/qudag_jujutsu_bg.wasm
wasm-validate dist/node/qudag_jujutsu_bg.wasm
wasm-validate dist/bundler/qudag_jujutsu_bg.wasm

echo ""
echo "✅ Build complete!"
echo "📦 Package ready for npm publish"
```

**`scripts/test.sh`** (Test Runner):

```bash
#!/bin/bash
set -e

echo "🧪 Running test suite for @agentic-flow/qudag-jujutsu"

# Rust unit tests
echo ""
echo "🦀 Running Rust unit tests..."
cargo test --features wasm --lib

# WASM tests (Node.js)
echo ""
echo "📦 Running WASM tests (Node.js)..."
wasm-pack test --node --features wasm

# WASM tests (headless browser)
echo ""
echo "🌐 Running WASM tests (headless Chrome)..."
wasm-pack test --headless --chrome --features wasm

# TypeScript tests
echo ""
echo "🔷 Running TypeScript tests..."
npm run test:ts

# Integration tests
echo ""
echo "🔗 Running integration tests..."
npm run test:integration

echo ""
echo "✅ All tests passed!"
```

---

## Phase-by-Phase Implementation

### Phase 1: Project Setup & Skeleton (Week 1)

**Objective**: Create buildable npm package skeleton with WASM + TypeScript.

**Tasks**:

1. **Initialize Package Structure** (Day 1)
   ```bash
   mkdir -p packages/qudag-jujutsu/{src,tests,scripts,docs,examples}
   cd packages/qudag-jujutsu
   npm init -y
   cargo init --lib
   ```

2. **Configure Cargo.toml** (Day 1)
   ```toml
   [package]
   name = "qudag-jujutsu"
   version = "0.1.0"
   edition = "2021"

   [lib]
   crate-type = ["cdylib", "rlib"]

   [dependencies]
   wasm-bindgen = "0.2"
   serde = { version = "1.0", features = ["derive"] }
   serde-wasm-bindgen = "0.6"

   [features]
   default = []
   wasm = ["wasm-bindgen"]
   qudag = []
   agentdb = []

   [profile.release]
   opt-level = "z"
   lto = true
   codegen-units = 1
   ```

3. **Configure TypeScript** (Day 2)
   ```json
   // tsconfig.json
   {
     "compilerOptions": {
       "target": "ES2020",
       "module": "ESNext",
       "lib": ["ES2020", "DOM"],
       "declaration": true,
       "declarationMap": true,
       "sourceMap": true,
       "outDir": "./dist",
       "rootDir": "./src/ts",
       "strict": true,
       "esModuleInterop": true,
       "skipLibCheck": true,
       "forceConsistentCasingInFileNames": true,
       "moduleResolution": "node"
     },
     "include": ["src/ts/**/*"],
     "exclude": ["node_modules", "dist", "tests"]
   }
   ```

4. **Create Hello World WASM** (Day 2-3)
   ```rust
   // src/lib.rs
   use wasm_bindgen::prelude::*;

   #[wasm_bindgen]
   pub fn greet(name: &str) -> String {
       format!("Hello from QUDAG+Jujutsu, {}!", name)
   }

   #[wasm_bindgen(start)]
   pub fn main() {
       #[cfg(feature = "wasm")]
       console_error_panic_hook::set_once();
   }
   ```

5. **Create TypeScript Facade** (Day 3)
   ```typescript
   // src/ts/index.ts
   import * as wasm from '../pkg/node/qudag_jujutsu';

   export async function init(): Promise<void> {
     await wasm.default();
   }

   export function greet(name: string): string {
     return wasm.greet(name);
   }
   ```

6. **Test Build** (Day 3)
   ```bash
   # Build WASM
   wasm-pack build --target nodejs

   # Build TypeScript
   tsc

   # Test
   node -e "import('./dist/index.js').then(m => console.log(m.greet('World')))"
   ```

**Deliverables**:
- ✅ Buildable package structure
- ✅ Hello World WASM module
- ✅ TypeScript facade compiling
- ✅ npm scripts for build/test

### Phase 2: QUDAG Core Integration (Weeks 2-4)

**Objective**: Integrate QUDAG network layer, consensus, and cryptography.

**Week 2: QUDAG Vendoring & Adaptation**

1. **Vendor QUDAG Source** (Day 1-2)
   ```bash
   git subtree add --prefix src/qudag \
     https://github.com/ruvnet/qudag.git main --squash
   ```

2. **WASM-ify QUDAG Modules** (Day 2-5)
   - Remove native-only dependencies (tokio → wasm-bindgen-futures)
   - Replace std::net with web-sys::WebSocket
   - Adapt file I/O to IndexedDB (browser) or fs (Node.js)

**Week 3: Cryptography Integration**

1. **ML-DSA-87 Signatures** (Day 1-3)
   ```rust
   use ml_dsa::*;

   #[wasm_bindgen]
   pub struct Signer {
       keypair: Keypair,
   }

   #[wasm_bindgen]
   impl Signer {
       #[wasm_bindgen(constructor)]
       pub fn new() -> Self {
           let keypair = Keypair::generate();
           Self { keypair }
       }

       pub fn sign(&self, message: &[u8]) -> Vec<u8> {
           self.keypair.sign(message)
       }

       pub fn verify(&self, message: &[u8], signature: &[u8]) -> bool {
           self.keypair.public_key().verify(message, signature)
       }
   }
   ```

2. **BLAKE3 Hashing** (Day 3-4)
   ```rust
   use blake3::Hasher;

   #[wasm_bindgen]
   pub fn hash_blake3(data: &[u8]) -> Vec<u8> {
       let hash = blake3::hash(data);
       hash.as_bytes().to_vec()
   }
   ```

**Week 4: DAG & Consensus**

1. **DAG Operations** (Day 1-3)
   ```rust
   #[wasm_bindgen]
   pub struct DAGNode {
       vertices: HashMap<String, Vertex>,
   }

   #[wasm_bindgen]
   impl DAGNode {
       pub fn add_vertex(&mut self, id: String, data: JsValue) -> Result<(), JsValue> {
           let vertex = serde_wasm_bindgen::from_value(data)?;
           self.vertices.insert(id, vertex);
           Ok(())
       }

       pub fn get_vertex(&self, id: &str) -> Result<JsValue, JsValue> {
           let vertex = self.vertices.get(id)
               .ok_or_else(|| JsValue::from_str("Vertex not found"))?;
           Ok(serde_wasm_bindgen::to_value(vertex)?)
       }
   }
   ```

2. **QR-Avalanche Consensus** (Day 3-5)
   - Implement voting protocol
   - Byzantine fault tolerance
   - Finality detection

**Deliverables**:
- ✅ QUDAG network layer in WASM
- ✅ Post-quantum cryptography working
- ✅ DAG and consensus operational
- ✅ Integration tests passing

### Phase 3: Jujutsu Integration (Weeks 5-6)

**Objective**: Integrate Jujutsu VCS operations with WASM wrapper.

**Week 5: Operation Log**

1. **JJ Command Execution** (Day 1-2)
   ```rust
   #[cfg(not(target_arch = "wasm32"))]
   use std::process::Command;

   #[wasm_bindgen]
   pub struct JJWrapper {
       repo_path: String,
   }

   #[wasm_bindgen]
   impl JJWrapper {
       #[wasm_bindgen(constructor)]
       pub fn new(repo_path: String) -> Self {
           Self { repo_path }
       }

       #[cfg(not(target_arch = "wasm32"))]
       pub async fn commit(&self, message: &str) -> Result<String, JsValue> {
           let output = Command::new("jj")
               .args(&["commit", "-m", message])
               .current_dir(&self.repo_path)
               .output()
               .map_err(|e| JsValue::from_str(&e.to_string()))?;

           Ok(String::from_utf8_lossy(&output.stdout).to_string())
       }

       #[cfg(target_arch = "wasm32")]
       pub async fn commit(&self, message: &str) -> Result<String, JsValue> {
           // WASM: use web API or mock
           Err(JsValue::from_str("JJ CLI not available in WASM, use REST API"))
       }
   }
   ```

2. **Operation Log Parsing** (Day 3-5)
   - Parse `jj op log` output
   - Create structured operation objects
   - Sync with QUDAG DAG

**Week 6: Conflict Resolution**

1. **Conflict Detection** (Day 1-3)
   - Detect conflicting changes
   - CRDT-style merge algorithms
   - Human escalation when necessary

2. **Integration with AgentDB** (Day 4-5)
   - Store conflict resolution patterns
   - Learn from successful resolutions
   - Recommend strategies

**Deliverables**:
- ✅ Jujutsu operations working
- ✅ Conflict resolution implemented
- ✅ AgentDB integration complete
- ✅ End-to-end tests passing

### Phase 4: CLI & Documentation (Weeks 7-8)

**Objective**: Create user-friendly CLI and comprehensive documentation.

**Week 7: CLI Implementation**

1. **Command Structure** (Day 1-2)
   ```typescript
   // src/cli/index.ts
   import { Command } from 'commander';
   import { QudagJujutsu } from '../ts/index.js';

   const program = new Command();

   program
     .name('qudag-jujutsu')
     .description('Quantum-resistant distributed VCS')
     .version('0.1.0');

   program
     .command('init')
     .description('Initialize repository')
     .action(async () => {
       const qj = await QudagJujutsu.init({
         /* config */
       });
       console.log('✅ Repository initialized!');
     });

   program
     .command('commit <message>')
     .option('-f, --files <files...>', 'Files to commit')
     .action(async (message, options) => {
       const qj = await QudagJujutsu.init(loadConfig());
       const result = await qj.commit(message, options.files || []);
       console.log(`✅ Committed: ${result.commitId}`);
     });

   program.parse();
   ```

2. **Configuration Management** (Day 3)
   - `.qudag-jujutsu.json` config file
   - Environment variables
   - Command-line overrides

**Week 8: Documentation**

1. **API Reference** (Day 1-2)
   - Generate with TypeDoc
   - Publish to docs.agentic-flow.io

2. **User Guides** (Day 3-5)
   - Getting started
   - Multi-agent workflows
   - Configuration reference
   - Troubleshooting

**Deliverables**:
- ✅ Full-featured CLI
- ✅ API reference documentation
- ✅ User guides and tutorials
- ✅ Example projects

### Phase 5: Testing & Optimization (Weeks 9-10)

**Objective**: Comprehensive testing, performance optimization, and bundle size reduction.

**Week 9: Testing**

1. **Unit Tests** (Day 1-2)
   - Rust: 100+ unit tests
   - TypeScript: 50+ unit tests

2. **Integration Tests** (Day 3-4)
   - Multi-agent workflows
   - Network partition scenarios
   - Conflict resolution

3. **E2E Tests** (Day 5)
   - CLI commands
   - Browser compatibility
   - Node.js compatibility

**Week 10: Optimization**

1. **WASM Size Reduction** (Day 1-2)
   - Strip debug symbols
   - Tree shaking unused code
   - Target: <500 KB gzipped

2. **Performance Tuning** (Day 3-4)
   - Profile with Chrome DevTools
   - Optimize hot paths
   - Lazy loading

**Deliverables**:
- ✅ 90%+ test coverage
- ✅ <500 KB gzipped WASM
- ✅ Benchmarks demonstrating performance

### Phase 6: Publication (Weeks 11-12)

**Objective**: Publish to npm registry and launch.

**Week 11: Pre-Launch**

1. **Security Audit** (Day 1-3)
   - `npm audit` with zero vulnerabilities
   - Snyk scan
   - Manual code review

2. **License & Legal** (Day 4-5)
   - Verify MIT license headers
   - Third-party license compliance
   - Export control review (cryptography)

**Week 12: Launch**

1. **npm Publish** (Day 1)
   ```bash
   npm version 1.0.0
   npm publish --access public
   ```

2. **Announcement** (Day 2-5)
   - Blog post
   - Demo video
   - Social media
   - Submit to Hacker News, Reddit

**Deliverables**:
- ✅ Published to npm
- ✅ Documentation live
- ✅ Launch announcement
- ✅ Community channels active

---

## WASM Optimization Strategy

### Size Optimization

**Target**: <500 KB gzipped for web bundle

**Techniques**:

1. **Cargo Profile**
   ```toml
   [profile.release]
   opt-level = "z"           # Optimize for size
   lto = true                # Link-time optimization
   codegen-units = 1         # Better optimization
   panic = "abort"           # Smaller panic handler
   strip = true              # Strip symbols
   ```

2. **wasm-opt**
   ```bash
   wasm-opt -Oz \
     --enable-simd \
     --enable-bulk-memory \
     --strip-debug \
     --strip-producers \
     -o output.wasm \
     input.wasm
   ```

3. **Code Splitting**
   - Core module: consensus, crypto, VCS
   - Optional modules: advanced features, debugging
   - Lazy loading for large dependencies

### Performance Optimization

**Target**: <20% overhead vs. native Rust

**Techniques**:

1. **WASM SIMD**
   ```rust
   #[cfg(target_feature = "simd128")]
   use std::simd::*;

   #[wasm_bindgen]
   pub fn vectorized_hash(data: &[u8]) -> Vec<u8> {
       // Use SIMD instructions for hashing
   }
   ```

2. **Memory Management**
   - Reuse allocations
   - Pool buffers
   - Avoid unnecessary clones

3. **Async Optimization**
   ```rust
   use wasm_bindgen_futures::spawn_local;

   #[wasm_bindgen]
   pub async fn parallel_commits(commits: JsValue) -> Result<JsValue, JsValue> {
       let commits: Vec<Commit> = serde_wasm_bindgen::from_value(commits)?;

       let futures: Vec<_> = commits.into_iter()
           .map(|c| spawn_local(async move { process_commit(c).await }))
           .collect();

       // Wait for all to complete
       let results = futures::future::join_all(futures).await;
       Ok(serde_wasm_bindgen::to_value(&results)?)
   }
   ```

---

## TypeScript Integration

### Type Generation

```typescript
// src/ts/types.ts

/**
 * Configuration for QUDAG network
 */
export interface QudagConfig {
  /** Network identifier (e.g., "mainnet", "testnet") */
  networkId: string;

  /** Bootstrap peer URLs for initial connection */
  bootstrapPeers: string[];

  /** Cryptographic configuration */
  cryptoConfig: {
    /** Signature algorithm (ML-DSA-87 recommended) */
    signatureAlgorithm: 'ML-DSA-87' | 'Ed25519';

    /** Hash algorithm (BLAKE3 recommended) */
    hashAlgorithm: 'BLAKE3' | 'SHA256';
  };
}

/**
 * Configuration for Jujutsu VCS
 */
export interface JujutsuConfig {
  /** Repository path */
  repoPath: string;

  /** Maximum operation log entries to keep in memory */
  maxLogEntries: number;

  /** Workspace isolation mode */
  workspaceMode: 'shared' | 'isolated';
}

/**
 * Configuration for AgentDB pattern learning
 */
export interface AgentDBConfig {
  /** Vector dimension for embeddings */
  vectorDimension: 384 | 768 | 1536;

  /** Quantization strategy for memory efficiency */
  quantization: 'none' | 'scalar' | 'product';

  /** Maximum patterns to store */
  maxPatterns: number;
}

/**
 * Complete configuration for QUDAG+Jujutsu
 */
export interface QudagJujutsuConfig {
  qudag: QudagConfig;
  jujutsu: JujutsuConfig;
  agentdb: AgentDBConfig;
}

/**
 * Result of a distributed commit
 */
export interface CommitResult {
  /** Local commit ID (Jujutsu) */
  commitId: string;

  /** DAG vertex ID (QUDAG) */
  vertexId: string;

  /** Unix timestamp */
  timestamp: number;

  /** Time to reach consensus (milliseconds) */
  consensusTimeMs: number;

  /** ML-DSA-87 signature (base64) */
  signature: string;

  /** Number of agents that voted */
  voters: number;
}
```

### Runtime Detection & Loading

```typescript
// src/ts/runtime.ts

export type Runtime = 'browser' | 'node' | 'deno' | 'bun';

export function detectRuntime(): Runtime {
  if (typeof window !== 'undefined' && typeof document !== 'undefined') {
    return 'browser';
  }
  if (typeof process !== 'undefined' && process.versions?.node) {
    return 'node';
  }
  if (typeof Deno !== 'undefined') {
    return 'deno';
  }
  if (typeof Bun !== 'undefined') {
    return 'bun';
  }
  throw new Error('Unsupported runtime environment');
}

export async function loadWASM(runtime: Runtime) {
  switch (runtime) {
    case 'browser':
      return await import('../dist/web/qudag_jujutsu.js');
    case 'node':
      return await import('../dist/node/qudag_jujutsu.js');
    case 'deno':
    case 'bun':
      return await import('../dist/web/qudag_jujutsu.js');
  }
}
```

---

## Testing Strategy

### Test Categories

1. **Rust Unit Tests** (200+ tests)
   ```rust
   #[cfg(test)]
   mod tests {
       use super::*;

       #[test]
       fn test_ml_dsa_signature() {
           let signer = Signer::new();
           let message = b"Hello, QUDAG!";
           let signature = signer.sign(message);
           assert!(signer.verify(message, &signature));
       }

       #[test]
       fn test_dag_vertex_creation() {
           let mut dag = DAGNode::new();
           dag.add_vertex("v1".to_string(), vertex_data);
           assert!(dag.get_vertex("v1").is_ok());
       }
   }
   ```

2. **WASM Tests** (50+ tests)
   ```bash
   wasm-pack test --node --headless
   ```

3. **TypeScript Tests** (100+ tests)
   ```typescript
   describe('QudagJujutsu', () => {
     it('should initialize with config', async () => {
       const qj = await QudagJujutsu.init(testConfig);
       expect(qj).toBeDefined();
     });

     it('should create quantum-signed commit', async () => {
       const result = await qj.commit('test commit', ['file.ts']);
       expect(result.signature).toBeTruthy();
       expect(result.consensusTimeMs).toBeLessThan(1000);
     });
   });
   ```

4. **Integration Tests** (30+ tests)
   - Multi-agent workflows
   - Network partition recovery
   - Conflict resolution scenarios

5. **E2E Tests** (20+ tests)
   - CLI command execution
   - Browser integration
   - npm package installation

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml

name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Test
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        rust: [stable, nightly]

    steps:
      - uses: actions/checkout@v3

      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: ${{ matrix.rust }}
          override: true

      - name: Install wasm-pack
        run: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

      - name: Install Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'

      - name: Install dependencies
        run: npm ci

      - name: Run Rust tests
        run: cargo test --features wasm

      - name: Run WASM tests
        run: wasm-pack test --node --headless

      - name: Build package
        run: npm run build

      - name: Run TypeScript tests
        run: npm test

  publish:
    name: Publish to npm
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'

    steps:
      - uses: actions/checkout@v3

      - name: Setup Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '20'
          registry-url: 'https://registry.npmjs.org'

      - name: Install dependencies
        run: npm ci

      - name: Build package
        run: npm run build

      - name: Publish to npm
        run: npm publish --access public
        env:
          NODE_AUTH_TOKEN: ${{ secrets.NPM_TOKEN }}
```

---

## npm Publishing Process

### Pre-Publish Checklist

- [ ] All tests passing
- [ ] Version bumped (`npm version patch|minor|major`)
- [ ] CHANGELOG.md updated
- [ ] README.md accurate
- [ ] LICENSE file present
- [ ] No sensitive data in package
- [ ] `npm audit` shows zero vulnerabilities
- [ ] Package size acceptable (<5 MB unpacked)

### Publishing Commands

```bash
# 1. Final build
npm run build

# 2. Test package locally
npm pack
npm install -g ./agentic-flow-qudag-jujutsu-0.1.0.tgz
qj --version

# 3. Publish to npm (dry run)
npm publish --dry-run

# 4. Publish for real
npm publish --access public

# 5. Verify
npm view @agentic-flow/qudag-jujutsu
```

### Post-Publish

1. **Tag Release**
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Create GitHub Release**
   - Upload build artifacts
   - Write release notes
   - Link to CHANGELOG

3. **Announce**
   - Blog post
   - Twitter/social media
   - Reddit (r/rust, r/programming)
   - Hacker News

---

## Documentation Requirements

### API Reference (TypeDoc)

```bash
npx typedoc --out docs/api src/ts/index.ts
```

### User Guides

1. **Getting Started** (`docs/getting-started.md`)
   - Installation
   - Quick start
   - First commit

2. **Configuration** (`docs/configuration.md`)
   - Config file format
   - Environment variables
   - Advanced options

3. **Multi-Agent Workflows** (`docs/multi-agent.md`)
   - Spawning agents
   - Workspace isolation
   - Conflict resolution

4. **Performance Tuning** (`docs/performance.md`)
   - WASM optimization
   - Network configuration
   - Scaling strategies

### Examples

1. **Browser Usage** (`examples/browser/`)
2. **Node.js Usage** (`examples/node/`)
3. **CLI Usage** (`examples/cli/`)
4. **Multi-Agent** (`examples/multi-agent/`)

---

## Maintenance & Updates

### Versioning Strategy (SemVer)

- **Major** (1.0.0 → 2.0.0): Breaking API changes
- **Minor** (1.0.0 → 1.1.0): New features, backward compatible
- **Patch** (1.0.0 → 1.0.1): Bug fixes, security patches

### Release Cadence

- **Patch**: As needed (bug fixes, security)
- **Minor**: Monthly (new features)
- **Major**: Yearly (breaking changes)

### Dependency Updates

- Monitor dependencies weekly (`npm outdated`)
- Update patch versions automatically (Dependabot)
- Test minor/major updates in separate branch
- Security patches within 24 hours

---

## Success Criteria

### Technical Metrics

- ✅ Builds successfully on Linux, macOS, Windows
- ✅ WASM bundle <500 KB gzipped
- ✅ 90%+ test coverage
- ✅ Zero critical/high vulnerabilities
- ✅ <20% WASM overhead vs. native

### Adoption Metrics

- ✅ 1,000+ npm downloads/week (6 months)
- ✅ 500+ GitHub stars (6 months)
- ✅ 10+ community contributions
- ✅ 100+ active repositories using package

---

## Conclusion

This implementation plan provides a **12-week roadmap** to deliver **@agentic-flow/qudag-jujutsu** as a production-ready **npm/npx package** with **WASM + TypeScript**.

**Key Milestones**:
- Week 1: Buildable skeleton
- Week 4: QUDAG integration complete
- Week 6: Jujutsu integration complete
- Week 8: CLI and docs complete
- Week 10: Optimized and tested
- Week 12: Published to npm 🚀

**Zero-Install Usage**:
```bash
npx @agentic-flow/qudag-jujutsu init
```

**Global Installation**:
```bash
npm install -g @agentic-flow/qudag-jujutsu
qj commit "quantum-resistant multi-agent commit"
```

The future of distributed coding is quantum-resistant, economically-incentivized, and AI-native. **Let's build it.** 🦀⚡🔐
