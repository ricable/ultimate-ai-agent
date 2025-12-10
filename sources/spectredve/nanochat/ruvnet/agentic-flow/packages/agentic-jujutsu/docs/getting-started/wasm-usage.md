# WASM Usage Guide

Complete guide for using agentic-jujutsu WASM bindings in JavaScript/TypeScript environments.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Platform-Specific Usage](#platform-specific-usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Performance Optimization](#performance-optimization)
- [Troubleshooting](#troubleshooting)

## Installation

### npm/yarn/pnpm

```bash
npm install @agentic-flow/jujutsu
# or
yarn add @agentic-flow/jujutsu
# or
pnpm add @agentic-flow/jujutsu
```

### Prerequisites

- Jujutsu VCS installed: https://github.com/jj-vcs/jj
- Node.js 16+ (for Node.js usage)
- Modern browser with WASM support (for web usage)
- Deno 1.30+ (for Deno usage)

## Quick Start

### Node.js

```javascript
const { JJWrapper, JJConfig } = require('@agentic-flow/jujutsu');

async function main() {
    const config = JJConfig.default();
    const jj = await JJWrapper.new(config);

    const status = await jj.status();
    console.log(status.stdout);
}

main();
```

### Browser (ES Modules)

```html
<script type="module">
    import init, { JJWrapper } from './pkg/web/agentic_jujutsu.js';

    await init();
    const jj = await JJWrapper.new();
    const operations = await jj.getOperations(5);
    console.log(operations);
</script>
```

### Deno

```typescript
import { JJWrapper } from "https://unpkg.com/@agentic-flow/jujutsu/pkg/deno/index.js";

const jj = await JJWrapper.new();
const status = await jj.status();
console.log(status.stdout);
```

## Platform-Specific Usage

### Node.js (CommonJS)

```javascript
const { JJWrapper, JJConfig } = require('@agentic-flow/jujutsu');

// Create custom configuration
const config = new JJConfig()
    .with_jj_path('/usr/local/bin/jj')
    .with_repo_path('/path/to/repo')
    .with_timeout(60000)
    .with_verbose(true);

const jj = await JJWrapper.new(config);
```

### Node.js (ES Modules)

```javascript
import { JJWrapper, JJConfig } from '@agentic-flow/jujutsu';

const jj = await JJWrapper.new(JJConfig.default());
```

### Browser with Bundler (Webpack/Rollup/Vite)

```javascript
import { JJWrapper } from '@agentic-flow/jujutsu/bundler';

const jj = await JJWrapper.new();
const operations = await jj.getOperations(10);
```

### Browser without Bundler

```html
<script type="module">
    import init, { JJWrapper, JJConfig } from './node_modules/@agentic-flow/jujutsu/pkg/web/agentic_jujutsu.js';

    // Initialize WASM module
    await init();

    // Use the API
    const jj = await JJWrapper.new();
    const status = await jj.status();
</script>
```

### Deno

```typescript
import { JJWrapper, JJConfig } from "@agentic-flow/jujutsu/deno";

const config = new JJConfig()
    .with_verbose(true)
    .with_timeout(30000);

const jj = await JJWrapper.new(config);
```

## API Reference

### JJConfig

Configuration object for JJWrapper.

```typescript
class JJConfig {
    constructor();
    static default(): JJConfig;

    with_jj_path(path: string): JJConfig;
    with_repo_path(path: string): JJConfig;
    with_timeout(timeout_ms: number): JJConfig;
    with_verbose(verbose: boolean): JJConfig;
    with_max_log_entries(max: number): JJConfig;
    with_agentdb_sync(enable: boolean): JJConfig;
}
```

**Example:**

```javascript
const config = new JJConfig()
    .with_jj_path('/usr/local/bin/jj')
    .with_repo_path('.')
    .with_timeout(30000)
    .with_verbose(true)
    .with_max_log_entries(1000)
    .with_agentdb_sync(true);
```

### JJWrapper

Main wrapper for Jujutsu operations.

```typescript
class JJWrapper {
    static async new(config?: JJConfig): Promise<JJWrapper>;

    async execute(args: string[]): Promise<JJResult>;
    async status(): Promise<JJResult>;
    async getOperations(limit: number): Promise<JJOperation[]>;
    async getConflicts(commit?: string): Promise<JJConflict[]>;
    async describe(message: string): Promise<JJOperation>;
    async diff(from: string, to: string): Promise<JJDiff>;
    async commit(message: string): Promise<JJOperation>;
    async branch_create(name: string): Promise<JJResult>;
    async branch_list(): Promise<JJBranch[]>;
    async abandon(commit: string): Promise<JJOperation>;
    async restore(paths: string[], from?: string): Promise<JJResult>;
}
```

## Examples

### Get Repository Status

```javascript
const jj = await JJWrapper.new();
const status = await jj.status();

console.log('Exit code:', status.exit_code);
console.log('Output:', status.stdout);
console.log('Success:', status.success);
```

### List Recent Operations

```javascript
const operations = await jj.getOperations(10);

for (const op of operations) {
    console.log(`${op.id.slice(0, 12)} - ${op.operation_type}`);
    console.log(`  ${op.description}`);
    console.log(`  by ${op.user} at ${op.timestamp}`);
}
```

### Check for Conflicts

```javascript
const conflicts = await jj.getConflicts();

if (conflicts.length > 0) {
    console.log('Found conflicts:');
    for (const conflict of conflicts) {
        console.log(`  ${conflict.path}: ${conflict.num_hunks} hunks`);
        console.log(`    Resolved: ${conflict.resolved}`);
    }
} else {
    console.log('No conflicts');
}
```

### Create and Manage Branches

```javascript
// Create new branch
await jj.branch_create('feature-x');

// List all branches
const branches = await jj.branch_list();
for (const branch of branches) {
    const remote = branch.remote ? ` (remote: ${branch.remote_name})` : '';
    console.log(`${branch.name} → ${branch.target}${remote}`);
}
```

### Execute Custom Commands

```javascript
// Any jj command
const result = await jj.execute(['log', '-r', '@', '--no-graph']);
console.log(result.stdout);

// With multiple arguments
const diffResult = await jj.execute(['diff', '-r', '@~', '@']);
console.log(diffResult.stdout);
```

### Error Handling

```javascript
try {
    const jj = await JJWrapper.new(config);
    const status = await jj.status();
    console.log(status.stdout);
} catch (error) {
    console.error('Error:', error.message());

    if (error.is_recoverable && error.is_recoverable()) {
        console.log('This error is recoverable. Retrying...');
        // Retry logic
    } else {
        console.log('Fatal error. Exiting...');
        process.exit(1);
    }
}
```

## Performance Optimization

### Memory Usage

The WASM module is designed to be memory-efficient:

- Bundle size: <200KB compressed
- Minimal runtime overhead
- Efficient string handling between JS and Rust
- Zero-copy operations where possible

### Caching

```javascript
// Cache JJWrapper instance
let jjInstance = null;

async function getJJ() {
    if (!jjInstance) {
        jjInstance = await JJWrapper.new();
    }
    return jjInstance;
}

// Reuse instance
const jj = await getJJ();
const status1 = await jj.status();
const status2 = await jj.status(); // Reuses same instance
```

### Batch Operations

```javascript
// Batch multiple operations
const [status, operations, branches] = await Promise.all([
    jj.status(),
    jj.getOperations(10),
    jj.branch_list()
]);
```

### Operation Log Limits

```javascript
// Configure max log entries to reduce memory
const config = new JJConfig()
    .with_max_log_entries(100); // Only keep 100 most recent

const jj = await JJWrapper.new(config);
```

## Troubleshooting

### "jj command not found"

**Problem:** The jj executable is not in PATH.

**Solution:**
```javascript
const config = new JJConfig()
    .with_jj_path('/usr/local/bin/jj'); // Absolute path

const jj = await JJWrapper.new(config);
```

### WASM Module Not Loading (Browser)

**Problem:** MIME type issues or CORS errors.

**Solution:**
1. Ensure server serves `.wasm` files with `application/wasm` MIME type
2. For local development, use a local server (not `file://`)
3. Check CORS headers if loading from CDN

```javascript
// Verify WASM is loaded
import init from './pkg/web/agentic_jujutsu.js';

try {
    await init();
    console.log('WASM loaded successfully');
} catch (error) {
    console.error('Failed to load WASM:', error);
}
```

### TypeScript Errors

**Problem:** TypeScript doesn't find type definitions.

**Solution:**
```json
// tsconfig.json
{
  "compilerOptions": {
    "types": ["@agentic-flow/jujutsu"]
  }
}
```

### Timeout Errors

**Problem:** Operations taking too long.

**Solution:**
```javascript
const config = new JJConfig()
    .with_timeout(120000); // Increase to 2 minutes

const jj = await JJWrapper.new(config);
```

### Memory Leaks (Long-Running Applications)

**Problem:** Memory usage grows over time.

**Solution:**
```javascript
// Periodically recreate instance
async function recreateJJ() {
    if (jjInstance) {
        jjInstance = null; // Let GC clean up
    }
    jjInstance = await JJWrapper.new();
}

// Call every hour or after N operations
setInterval(recreateJJ, 3600000);
```

## Browser Compatibility

| Browser | Minimum Version | Notes |
|---------|----------------|-------|
| Chrome  | 57+            | Full support |
| Firefox | 52+            | Full support |
| Safari  | 11+            | Full support |
| Edge    | 16+            | Full support |
| Node.js | 8+             | Full support |
| Deno    | 1.0+           | Full support |

## Bundle Size

Optimized bundle sizes (gzipped):

- **web**: ~85KB
- **node**: ~85KB
- **bundler**: ~85KB
- **deno**: ~85KB

Target: <200KB uncompressed per target.

## Advanced Topics

### AgentDB Integration

```javascript
const config = new JJConfig()
    .with_agentdb_sync(true); // Enable AgentDB sync

const jj = await JJWrapper.new(config);

// Operations are now synced to AgentDB
await jj.describe('Update documentation');
```

### Custom Event Handling

```javascript
// Monitor operations
const jj = await JJWrapper.new();

const operations = await jj.getOperations(100);
for (const op of operations) {
    if (op.operation_type === 'conflict') {
        console.log('Conflict detected:', op.description);
        // Handle conflict
    }
}
```

### Multi-Repository Management

```javascript
// Manage multiple repositories
const repos = [
    '/path/to/repo1',
    '/path/to/repo2',
    '/path/to/repo3'
];

const instances = await Promise.all(
    repos.map(path =>
        JJWrapper.new(
            new JJConfig().with_repo_path(path)
        )
    )
);

// Operate on all repositories
const statuses = await Promise.all(
    instances.map(jj => jj.status())
);
```

## Contributing

See the main repository for contribution guidelines:
https://github.com/ruvnet/agentic-flow

## License

MIT License - see LICENSE file for details.
