# GitHub Issues for AgentDB v2.0.0-alpha.1 Critical Bugs

## Issue #1: Package.json exports block version access (CRITICAL)

**Title**: `[BUG] Cannot access package.json version via require('agentdb/package.json')`

**Labels**: `bug`, `critical`, `alpha`, `api`

**Body**:
```markdown
## Bug Description

The package.json exports configuration blocks access to the package version, preventing programmatic version checking.

## Error

```javascript
const {AgentDB} = require('agentdb');
console.log(require('agentdb/package.json').version);
```

```
Error [ERR_PACKAGE_PATH_NOT_EXPORTED]: Package subpath './package.json'
is not defined by "exports" in package.json
```

## Expected Behavior

Users should be able to access the package version programmatically.

## Root Cause

`package.json` exports field doesn't include `"./package.json"` entry.

## Fix

```json
{
  "exports": {
    ".": "./dist/index.js",
    "./package.json": "./package.json"
  }
}
```

## Impact

- Prevents version checking in programmatic usage
- Breaks tools that rely on version detection
- Forces users to use `npm ls agentdb` workaround

## Environment

- Package: agentdb@2.0.0-alpha.1
- Node.js: 20.x
- Platform: All

## Workaround

Use `npm ls agentdb` or access version from `AgentDB.VERSION` if exported.
```

---

## Issue #2: Simulate commands not accessible via main CLI (HIGH)

**Title**: `[BUG] Simulate commands return "Unknown command: simulate"`

**Labels**: `bug`, `high`, `cli`, `alpha`

**Body**:
```markdown
## Bug Description

The `simulate` commands are not accessible via the main `agentdb` CLI, despite being documented in the README.

## Reproduction

```bash
$ npx agentdb@alpha simulate list
❌ Unknown command: simulate
```

## Expected Behavior

```bash
$ npx agentdb@alpha simulate list
✅ Available scenarios:
  - HNSW Exploration (8.2x speedup)
  - Attention Analysis (12.4% improvement)
  ...
```

## Root Cause

Simulation CLI is a separate entry point (`simulation/cli.ts`) not integrated into main CLI.

## Possible Fixes

### Option 1: Integrate into main CLI
Add simulate commands to `src/cli/agentdb-cli.ts`:
```typescript
program
  .command('simulate')
  .description('Run latent space simulations')
  .action(async () => {
    const { runSimulation } = await import('../simulation/cli.js');
    await runSimulation();
  });
```

### Option 2: Document separate entry point
Update README to use:
```bash
npx agentdb-simulate@alpha list
```

### Option 3: Add alias in package.json
```json
{
  "bin": {
    "agentdb": "./dist/cli/agentdb-cli.js",
    "agentdb-simulate": "./simulation/cli.js"
  }
}
```

## Impact

- Users cannot access advertised simulation features
- README examples don't work
- Reduces package value proposition

## Environment

- Package: agentdb@2.0.0-alpha.1
- Platform: All
```

---

## Issue #3: Programmatic API undocumented and requires complex initialization (CRITICAL)

**Title**: `[DOCS] Programmatic API initialization undocumented, tables not auto-created`

**Labels**: `documentation`, `critical`, `api`, `alpha`

**Body**:
```markdown
## Problem

The programmatic API is undocumented and requires complex manual initialization that isn't explained anywhere.

## Current Situation

README shows:
```javascript
const db = await createDatabase('./db.db');
const reflexion = new ReflexionMemory(db, embedder, vectorBackend, learningBackend, graphBackend);
```

But this fails with:
```
Error: no such table: episodes
```

## What's Missing

1. **Schema auto-initialization**: Tables aren't created automatically
2. **Factory function**: No simple `AgentDB.create(config)` method
3. **Documented initialization**: No explanation of required steps
4. **Working examples**: No `/examples/` directory with functional code

## Expected API

### Simple Initialization
```javascript
import { AgentDB } from 'agentdb';

// Option 1: Factory with auto-setup
const db = await AgentDB.create({
  path: './my-db.db',
  dimensions: 384,
  backend: 'ruvector'
});

// Option 2: Manual but documented
import { createDatabase, initializeSchemas } from 'agentdb';
const db = await createDatabase('./db.db');
await initializeSchemas(db);
const agentdb = new AgentDB(db, config);
```

### Working Example Needed
```javascript
// examples/quickstart.js
import { AgentDB } from 'agentdb';

async function main() {
  // Initialize database with auto-setup
  const db = await AgentDB.create({
    path: './agent-memory.db',
    dimensions: 384,
    backend: 'ruvector'
  });

  // Store episode
  await db.reflexion.store({
    sessionId: 'session-1',
    task: 'Implement authentication',
    reward: 0.95,
    success: true,
    critique: 'Used JWT tokens effectively'
  });

  // Search similar episodes
  const similar = await db.reflexion.retrieve('authentication', 5);
  console.log('Similar episodes:', similar);
}

main().catch(console.error);
```

## Suggested Fixes

1. Create `AgentDB.create()` factory method that handles all initialization
2. Auto-run schema migrations when creating database
3. Add `/examples/` directory with working code:
   - `examples/quickstart.js`
   - `examples/reflexion-memory.js`
   - `examples/skill-library.js`
   - `examples/causal-reasoning.js`
4. Create `/docs/API.md` with complete API documentation
5. Add TypeScript examples

## Impact

- **Critical**: Developers cannot use the library programmatically
- Forces all users to CLI-only usage
- Defeats purpose of packaging as library
- Prevents integration into applications

## Priority

**CRITICAL** - This blocks beta release. The library is unusable for programmatic integration without this.

## Environment

- Package: agentdb@2.0.0-alpha.1
- All platforms
```

---

## Issue #4: Deprecated dependencies with memory leak warnings (MEDIUM)

**Title**: `[DEPS] Update 7 deprecated dependencies, including memory-leaking inflight`

**Labels**: `dependencies`, `medium`, `maintenance`

**Body**:
```markdown
## Deprecated Dependencies

The package has 7 deprecated transitive dependencies:

### Critical
- `inflight@1.0.6` - **⚠️ Memory leak warning**

### Others
- `are-we-there-yet@3.0.1`
- `@npmcli/move-file@1.1.2`
- `rimraf@3.0.2`
- `npmlog@6.0.2`
- `glob@7.2.3`
- `gauge@4.0.4`

## Impact

- Security warnings during installation
- Potential memory leaks in long-running processes
- Outdated dependency graph

## Recommended Actions

1. Run `npm update` to get latest compatible versions
2. Replace `rimraf` with native `fs.rm()`
3. Replace `glob@7` with `glob@9`
4. Update other npm CLI utilities to latest versions

## Priority

Medium - Should fix before beta release
```

---

## Issue #5: Transformers.js fails in Docker/offline environments (MEDIUM)

**Title**: `[BUG] Transformers.js always fails in Docker, unclear error message`

**Labels**: `bug`, `medium`, `docker`, `embeddings`

**Body**:
```markdown
## Bug Description

Transformers.js initialization always fails in Docker environments with unclear error message.

## Error Output

```
✅ Using sql.js (WASM SQLite, no build tools required)
⚠️  Transformers.js initialization failed: fetch failed
Falling back to mock embeddings for testing
```

## Root Cause

- Docker build has no network access
- Models can't be downloaded
- Error message suggests `HUGGINGFACE_API_KEY` (incorrect - that's for API, not transformers.js)

## Impact

- All Docker users get mock embeddings
- Confusing error message
- No documentation for offline usage

## Suggested Fixes

1. **Better error message**:
   ```
   ⚠️  Could not download Transformers.js model (offline environment)
   → Falling back to mock embeddings
   → For production use, pre-download models or configure custom embeddings
   ```

2. **Document offline usage**:
   - How to pre-download models
   - Where to cache models
   - Environment variables to configure cache path

3. **Docker support**:
   - Add Dockerfile example with model pre-download
   - Document model caching strategies

## Workaround

For Docker users:
```dockerfile
# Pre-download models during build
RUN npx transformers download Xenova/all-MiniLM-L6-v2
```

## Priority

Medium - Affects Docker users, but mock fallback works for testing
```

---

**Total Issues Created**: 5 (3 Critical, 1 High, 1 Medium)
**Estimated Fix Time**: 2-4 hours
**Target**: Publish alpha.2 with fixes within 24 hours
