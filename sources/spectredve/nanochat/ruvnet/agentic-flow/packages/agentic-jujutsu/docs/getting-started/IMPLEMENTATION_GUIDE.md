# Implementation Guide: Agentic-Jujutsu
## Step-by-Step Setup, Integration, and Usage

**Version:** 0.1.0
**Last Updated:** 2025-11-09
**Audience:** Developers, DevOps Engineers, AI Researchers

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Basic Usage](#basic-usage)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Topics](#advanced-topics)

---

## Quick Start

### 5-Minute Getting Started

**IMPORTANT:** Before proceeding, note that there is currently a WASM build issue that needs to be fixed. Follow the troubleshooting section if you encounter build errors.

```bash
# Clone the repository
git clone https://github.com/ruvnet/agentic-flow.git
cd agentic-flow/packages/agentic-jujutsu

# Install dependencies
npm install

# Build WASM (currently broken - see troubleshooting)
npm run build

# Run tests (will fail until build is fixed)
npm test
```

**Expected Result:** Build fails with errno crate error. See [Troubleshooting](#troubleshooting-wasm-build-failure) for the fix.

---

## Prerequisites

### System Requirements

**Operating System:**
- Linux (Ubuntu 20.04+, Debian 11+)
- macOS (12.0+)
- Windows (WSL2 recommended)

**Software Dependencies:**

1. **Node.js (Required)**
```bash
# Version 16.0.0 or higher
node --version  # Should show v16+ or v20+

# If not installed:
# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# macOS (with Homebrew)
brew install node@20
```

2. **Rust (Required)**
```bash
# Version 1.75.0 or higher
rustc --version  # Should show 1.75+ or higher

# If not installed:
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Add WASM target
rustup target add wasm32-unknown-unknown
```

3. **wasm-pack (Required)**
```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Verify installation
wasm-pack --version
```

4. **Jujutsu (Optional - for native functionality)**
```bash
# Ubuntu/Debian
cargo install --git https://github.com/martinvonz/jj.git jj-cli

# macOS (with Homebrew)
brew install jj

# Verify installation
jj --version
```

5. **Docker (Optional - for benchmarks)**
```bash
# Ubuntu/Debian
sudo apt-get install docker.io docker-compose

# macOS (with Homebrew)
brew install --cask docker

# Verify installation
docker --version
docker-compose --version
```

---

## Installation

### Standard Installation

#### Step 1: Install from npm (Future)

```bash
# This will work once package is published
npm install @agentic-flow/jujutsu
```

#### Step 2: Install from Source (Current)

```bash
# Clone repository
git clone https://github.com/ruvnet/agentic-flow.git
cd agentic-flow/packages/agentic-jujutsu

# Install dependencies
npm install

# Build WASM (FIX REQUIRED - see troubleshooting)
npm run build
```

#### Step 3: Verify Installation

```bash
# Check package structure
ls -la pkg/

# Should see:
# pkg/node/        - Node.js target
# pkg/web/         - Browser target
# pkg/bundler/     - Bundler target
# pkg/deno/        - Deno target

# Check TypeScript definitions
ls -la typescript/

# Should see:
# typescript/index.d.ts
# typescript/hooks-integration.ts
```

---

### Development Installation

For contributors and developers:

```bash
# Clone with full history
git clone https://github.com/ruvnet/agentic-flow.git
cd agentic-flow/packages/agentic-jujutsu

# Install dev dependencies
npm install

# Build in development mode (faster builds, no optimization)
npm run build:dev

# Run tests
npm run test

# Run benchmarks (requires fix first)
npm run bench

# Analyze bundle size
npm run size

# Verify build artifacts
npm run verify
```

---

## Configuration

### Basic Configuration

#### Config File: `.jjconfig.json`

Create a configuration file in your project root:

```json
{
  "workingDir": "./",
  "enableHooks": true,
  "agentdb": {
    "enabled": true,
    "dbPath": "./.agentdb/patterns.sqlite",
    "sessionId": "my-project-session"
  },
  "hooks": {
    "preTask": true,
    "postEdit": true,
    "postTask": true,
    "sessionRestore": true
  },
  "logging": {
    "level": "info",
    "destination": "./logs/jujutsu.log"
  }
}
```

#### Environment Variables

```bash
# Create .env file
cat > .env << 'EOF'
# AgentDB Configuration
AGENTDB_PATH=./.agentdb/patterns.sqlite
AGENTDB_SESSION_ID=my-session

# Hooks Configuration
HOOKS_ENABLED=true
CLAUDE_FLOW_VERSION=alpha

# Logging
LOG_LEVEL=info
LOG_DESTINATION=./logs/jujutsu.log

# Jujutsu Configuration
JJ_WORKING_DIR=./
JJ_ENABLE_HOOKS=true
EOF

# Load environment variables
source .env
```

---

## Basic Usage

### Node.js Example

```javascript
// examples/basic-usage.js
const { JJWrapper, JJConfig } = require('@agentic-flow/jujutsu');

async function main() {
  // Create configuration
  const config = {
    workingDir: process.cwd(),
    enableHooks: true,
    agentdbPath: './.agentdb/patterns.sqlite'
  };

  // Initialize wrapper
  const jj = await JJWrapper.new(config);

  try {
    // Check repository status
    const status = await jj.status();
    console.log('Repository Status:');
    console.log(status.stdout);

    // View commit log
    const log = await jj.log(['-r', 'all()', '--limit', '10']);
    console.log('\nRecent Commits:');
    console.log(log.stdout);

    // Create a commit (example)
    const commit = await jj.commit('Update documentation', ['README.md']);
    console.log('\nCommit Created:');
    console.log(commit.stdout);

  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
```

**Run the example:**
```bash
node examples/basic-usage.js
```

---

### TypeScript Example

```typescript
// examples/typescript-usage.ts
import { JJWrapper, JJConfig, CommandOutput } from '@agentic-flow/jujutsu';

interface RepositoryInfo {
  status: string;
  branches: string[];
  latestCommit: string;
}

async function getRepositoryInfo(): Promise<RepositoryInfo> {
  const config: JJConfig = {
    workingDir: process.cwd(),
    enableHooks: true,
    agentdbPath: './.agentdb/patterns.sqlite'
  };

  const jj = await JJWrapper.new(config);

  // Get status
  const statusResult: CommandOutput = await jj.status();

  // Get branches
  const branchesResult: CommandOutput = await jj.execute(['branch', 'list']);
  const branches = branchesResult.stdout
    .split('\n')
    .filter(line => line.trim().length > 0);

  // Get latest commit
  const logResult: CommandOutput = await jj.log(['-r', '@', '--no-graph']);
  const latestCommit = logResult.stdout.split('\n')[0];

  return {
    status: statusResult.stdout,
    branches,
    latestCommit
  };
}

// Usage
getRepositoryInfo()
  .then(info => {
    console.log('Repository Information:');
    console.log(JSON.stringify(info, null, 2));
  })
  .catch(error => {
    console.error('Error:', error);
    process.exit(1);
  });
```

**Run the TypeScript example:**
```bash
npx tsx examples/typescript-usage.ts
```

---

### Browser Example

```html
<!DOCTYPE html>
<html>
<head>
  <title>Agentic-Jujutsu Browser Example</title>
</head>
<body>
  <h1>Jujutsu VCS in Browser</h1>
  <button id="status-btn">Get Status</button>
  <pre id="output"></pre>

  <script type="module">
    import { JJWrapper } from '@agentic-flow/jujutsu/web';

    const outputEl = document.getElementById('output');
    const statusBtn = document.getElementById('status-btn');

    async function initialize() {
      try {
        const jj = await JJWrapper.new({
          workingDir: '/',
          enableHooks: false
        });

        statusBtn.addEventListener('click', async () => {
          try {
            const result = await jj.status();
            outputEl.textContent = result.stdout;
          } catch (error) {
            outputEl.textContent = `Error: ${error.message}`;
          }
        });

        outputEl.textContent = 'Initialized successfully!\nClick button to get status.';
      } catch (error) {
        outputEl.textContent = `Initialization error: ${error.message}`;
      }
    }

    initialize();
  </script>
</body>
</html>
```

**Note:** Browser functionality is limited due to process execution constraints in WASM.

---

## Integration Examples

### Integration with AgentDB

```typescript
// examples/integration/agentdb-integration.ts
import { JJWrapper } from '@agentic-flow/jujutsu';
import { AgentDBSync } from '@agentic-flow/jujutsu/agentdb';

async function trackOperationWithLearning() {
  const jj = await JJWrapper.new({
    workingDir: process.cwd(),
    enableHooks: true,
    agentdbPath: './.agentdb/patterns.sqlite'
  });

  const agentdb = new AgentDBSync({
    dbPath: './.agentdb/patterns.sqlite',
    sessionId: 'commit-learning-session'
  });

  // Execute operation with timing
  const startTime = Date.now();
  const result = await jj.commit('Feature implementation', ['src/**/*.ts']);
  const executionTime = Date.now() - startTime;

  // Store pattern in AgentDB
  await agentdb.storePattern({
    sessionId: 'commit-learning-session',
    task: 'commit-operation',
    input: JSON.stringify({
      operation: 'commit',
      files: ['src/**/*.ts'],
      message: 'Feature implementation'
    }),
    output: JSON.stringify({
      success: result.success,
      executionTime
    }),
    reward: result.success ? (executionTime < 1000 ? 0.9 : 0.7) : 0.1,
    success: result.success,
    latencyMs: executionTime,
    tokensUsed: 0
  });

  // Query historical patterns
  const similarPatterns = await agentdb.searchPatterns('commit-operation', 10);
  console.log(`Found ${similarPatterns.length} similar operations`);
  console.log('Average execution time:',
    similarPatterns.reduce((sum, p) => sum + p.latencyMs, 0) / similarPatterns.length
  );
}

trackOperationWithLearning();
```

---

### Integration with Hooks

```typescript
// examples/integration/hooks-integration.ts
import { JJWrapper } from '@agentic-flow/jujutsu';
import { HooksClient } from '@agentic-flow/jujutsu/hooks';

async function workflowWithHooks() {
  const hooks = new HooksClient('feature-development-session');

  // Pre-task hook
  await hooks.preTask('Implement new feature with tests');

  try {
    const jj = await JJWrapper.new({
      workingDir: process.cwd(),
      enableHooks: true
    });

    // Create feature branch
    await jj.execute(['branch', 'create', 'feature/new-component']);
    await hooks.notify('Created feature branch');

    // Make changes (simulate)
    // ... edit files ...

    // Commit changes
    const commitResult = await jj.commit('Add new component', [
      'src/components/NewComponent.tsx',
      'tests/NewComponent.test.tsx'
    ]);

    // Post-edit hook
    await hooks.postEdit(
      'src/components/NewComponent.tsx',
      'swarm/feature-development/component-created'
    );

    // Post-task hook
    await hooks.postTask('feature-development-complete');

    console.log('Feature development complete!');

  } catch (error) {
    console.error('Workflow error:', error);
    throw error;
  } finally {
    // Session end hook
    await hooks.sessionEnd(true);
  }
}

workflowWithHooks();
```

---

### Integration with Claude Flow

```typescript
// examples/integration/claude-flow-integration.ts
import { JJWrapper } from '@agentic-flow/jujutsu';
import { SwarmCoordination } from '@agentic-flow/jujutsu/swarm';

async function distributedWorkflow() {
  const swarm = new SwarmCoordination();

  // Initialize swarm for parallel development
  const swarmId = await swarm.initializeBenchmarkSwarm({
    id: 'parallel-feature-dev',
    execution: {
      parallelism: 3,
      timeout: 300000
    }
  });

  // Spawn agents for different tasks
  const agents = [
    { type: 'coder', task: 'Implement backend API', files: ['src/api/**/*.ts'] },
    { type: 'coder', task: 'Implement frontend UI', files: ['src/ui/**/*.tsx'] },
    { type: 'tester', task: 'Write integration tests', files: ['tests/**/*.test.ts'] }
  ];

  // Execute in parallel
  const results = await Promise.all(
    agents.map(async (agent) => {
      const jj = await JJWrapper.new({ workingDir: process.cwd() });

      // Agent does work...
      const result = await jj.commit(agent.task, agent.files);

      return {
        agent: agent.type,
        task: agent.task,
        success: result.success
      };
    })
  );

  // Monitor swarm status
  const status = await swarm.monitorProgress(swarmId);
  console.log('Swarm Status:', status);

  console.log('Parallel workflow results:', results);
}

distributedWorkflow();
```

---

## Best Practices

### 1. Configuration Management

**DO:**
```typescript
// ✅ Use environment variables for sensitive data
const config = {
  workingDir: process.env.JJ_WORKING_DIR || process.cwd(),
  agentdbPath: process.env.AGENTDB_PATH || './.agentdb/patterns.sqlite'
};
```

**DON'T:**
```typescript
// ❌ Hardcode paths
const config = {
  workingDir: '/home/user/project',
  agentdbPath: '/var/db/agentdb.sqlite'
};
```

---

### 2. Error Handling

**DO:**
```typescript
// ✅ Comprehensive error handling
async function safeCommit(jj: JJWrapper, message: string, files: string[]) {
  try {
    const result = await jj.commit(message, files);

    if (!result.success) {
      console.error('Commit failed:', result.stderr);
      return null;
    }

    return result;
  } catch (error) {
    if (error instanceof JJError) {
      console.error('JJ Error:', error.message);
    } else {
      console.error('Unexpected error:', error);
    }
    return null;
  }
}
```

**DON'T:**
```typescript
// ❌ Silent failures
async function badCommit(jj: JJWrapper, message: string, files: string[]) {
  const result = await jj.commit(message, files);
  return result; // Doesn't check success or handle errors
}
```

---

### 3. Resource Management

**DO:**
```typescript
// ✅ Proper cleanup
class JJWorkflow {
  private jj: JJWrapper;
  private hooks: HooksClient;

  async initialize() {
    this.jj = await JJWrapper.new({ workingDir: process.cwd() });
    this.hooks = new HooksClient();
    await this.hooks.preTask('workflow-start');
  }

  async cleanup() {
    await this.hooks.sessionEnd(true);
    // Any other cleanup...
  }

  async execute() {
    try {
      await this.initialize();
      // ... do work ...
    } finally {
      await this.cleanup();
    }
  }
}
```

---

### 4. AgentDB Learning

**DO:**
```typescript
// ✅ Store meaningful patterns with context
await agentdb.storePattern({
  sessionId: `project-${Date.now()}`,
  task: 'large-file-commit',
  input: JSON.stringify({
    operation: 'commit',
    fileCount: 50,
    totalSize: 1024 * 1024 * 100, // 100MB
    fileTypes: ['.ts', '.tsx', '.json']
  }),
  output: JSON.stringify({
    executionTime: 2500,
    memoryUsage: 1024 * 1024 * 50,
    success: true
  }),
  reward: calculateReward(2500, 50 * 1024 * 1024),
  critique: 'Large file commit completed efficiently',
  success: true,
  latencyMs: 2500,
  tokensUsed: 0
});
```

**DON'T:**
```typescript
// ❌ Vague patterns without context
await agentdb.storePattern({
  sessionId: 'session',
  task: 'commit',
  input: '{}',
  output: '{}',
  reward: 0.5,
  success: true
});
```

---

### 5. Hooks Integration

**DO:**
```typescript
// ✅ Consistent hook usage throughout workflow
const hooks = new HooksClient('feature-session');

await hooks.preTask('feature-implementation');
// ... work ...
await hooks.postEdit('src/feature.ts', 'swarm/feature/completed');
// ... more work ...
await hooks.postTask('feature-complete');
await hooks.sessionEnd(true);
```

**DON'T:**
```typescript
// ❌ Inconsistent or missing hooks
const hooks = new HooksClient();
await hooks.preTask('start');
// ... work without tracking ...
// Missing post-task and session-end hooks
```

---

## Troubleshooting

### WASM Build Failure

**Problem:**
```
error: The target OS is "unknown" or "none", so it's unsupported by the errno crate.
```

**Solution:**

1. **Edit `Cargo.toml`:**
```toml
[dependencies]
# Make process execution optional
async-process = { version = "2.0", optional = true }

[features]
default = ["wasm"]
native = ["tokio", "async-process"]
wasm = []
```

2. **Edit `src/wrapper.rs`:**
```rust
#[cfg(not(target_arch = "wasm32"))]
use async_process::Command;

#[cfg(target_arch = "wasm32")]
impl JJWrapper {
    pub async fn execute(&self, args: &[&str]) -> Result<CommandOutput> {
        Err(JJError::UnsupportedPlatform(
            "Direct process execution not available in WASM".into()
        ))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl JJWrapper {
    pub async fn execute(&self, args: &[&str]) -> Result<CommandOutput> {
        let output = Command::new("jj")
            .args(args)
            .current_dir(&self.config.working_dir)
            .output()
            .await?;

        Ok(CommandOutput::from(output))
    }
}
```

3. **Rebuild:**
```bash
npm run build
```

---

### Test Failures

**Problem:**
```
Tests fail with "jj command not found"
```

**Solution:**

Install Jujutsu CLI:
```bash
# Rust installation
cargo install --git https://github.com/martinvonz/jj.git jj-cli

# macOS with Homebrew
brew install jj

# Verify
jj --version
```

---

### Docker Issues

**Problem:**
```
Cannot connect to Docker daemon
```

**Solution:**

```bash
# Start Docker daemon
sudo systemctl start docker  # Linux
open -a Docker               # macOS

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker ps
```

---

### Permission Errors

**Problem:**
```
EACCES: permission denied
```

**Solution:**

```bash
# Fix npm permissions
sudo chown -R $USER:$USER ~/.npm
sudo chown -R $USER:$USER node_modules

# Fix cargo permissions
sudo chown -R $USER:$USER ~/.cargo

# Rebuild
npm run clean
npm install
npm run build
```

---

## Advanced Topics

### Custom Operations

```rust
// src/custom_operations.rs
use crate::{JJWrapper, Result, CommandOutput};

impl JJWrapper {
    pub async fn custom_workflow(&self) -> Result<CommandOutput> {
        // Step 1: Create branch
        self.execute(&["branch", "create", "workflow/automated"]).await?;

        // Step 2: Make changes
        // ... file operations ...

        // Step 3: Commit
        let result = self.commit("Automated workflow commit", &["."]).await?;

        Ok(result)
    }
}
```

---

### Performance Optimization

```typescript
// Cache JJWrapper instances
class JJWrapperPool {
  private static instances = new Map<string, JJWrapper>();

  static async get(workingDir: string): Promise<JJWrapper> {
    if (!this.instances.has(workingDir)) {
      this.instances.set(
        workingDir,
        await JJWrapper.new({ workingDir })
      );
    }
    return this.instances.get(workingDir)!;
  }
}

// Usage
const jj = await JJWrapperPool.get(process.cwd());
```

---

### Monitoring and Metrics

```typescript
import { JJWrapper } from '@agentic-flow/jujutsu';

class MonitoredJJWrapper {
  private metrics = {
    totalOperations: 0,
    successfulOperations: 0,
    failedOperations: 0,
    totalExecutionTime: 0
  };

  async execute(operation: string, ...args: any[]): Promise<any> {
    const startTime = Date.now();
    this.metrics.totalOperations++;

    try {
      const result = await (this.jj as any)[operation](...args);

      if (result.success) {
        this.metrics.successfulOperations++;
      } else {
        this.metrics.failedOperations++;
      }

      this.metrics.totalExecutionTime += (Date.now() - startTime);

      return result;
    } catch (error) {
      this.metrics.failedOperations++;
      throw error;
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      averageExecutionTime: this.metrics.totalExecutionTime / this.metrics.totalOperations,
      successRate: this.metrics.successfulOperations / this.metrics.totalOperations
    };
  }
}
```

---

## Next Steps

1. **Fix WASM Build** - Follow troubleshooting guide
2. **Run Basic Examples** - Verify installation works
3. **Integrate with Your Project** - Use integration examples
4. **Set Up Benchmarks** - Follow benchmark guide (separate doc)
5. **Join Community** - Contribute and get support

---

**Document Status:** ✅ COMPLETE
**Last Updated:** 2025-11-09
**Maintainer:** Agentic Flow Team
**Support:** https://github.com/ruvnet/agentic-flow/issues
