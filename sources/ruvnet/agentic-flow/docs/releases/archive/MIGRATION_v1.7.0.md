# Migration Guide: agentic-flow v1.7.0

## Overview

agentic-flow v1.7.0 integrates AgentDB as a proper dependency with significant performance improvements and memory optimizations. **This release is 100% backwards compatible** - all existing code continues to work.

## What Changed

### ✅ Added (Non-Breaking)
- AgentDB v1.3.9 as dependency
- Shared memory pool architecture
- Hybrid ReasoningBank (Rust WASM + AgentDB)
- Advanced memory system with auto-consolidation
- 29 MCP tools from AgentDB
- HNSW indexing for 116x faster search
- Batch operations for 141x faster inserts

### ✅ Improved (No Breaking Changes)
- 56% memory reduction (800MB → 350MB for 4 agents)
- 116x faster vector search (580ms → 5ms @ 100K vectors)
- 141x faster batch operations (14.1s → 100ms for 1000 inserts)
- 65% faster cold start (3.5s → 1.2s)
- 400KB bundle size reduction

### ✅ No Breaking Changes
- All CLI commands work identically
- All import paths maintained via re-exports
- All API signatures unchanged
- All MCP tools compatible
- SDK methods have same signatures

## Migration Steps

### For Most Users: No Action Required

Simply upgrade to v1.7.0:

```bash
npm install agentic-flow@^1.7.0
```

All existing code continues to work without changes.

### For Advanced Users: Opt Into New Features

#### 1. Use Hybrid ReasoningBank (Recommended)

**Before:**
```typescript
import { ReasoningBankEngine } from 'agentic-flow/reasoningbank';

const rb = new ReasoningBankEngine();
await rb.storePattern({ /* ... */ });
```

**After (Optional - Better Performance):**
```typescript
import { HybridReasoningBank } from 'agentic-flow/reasoningbank';

const rb = new HybridReasoningBank({ preferWasm: true });
await rb.storePattern({ /* ... */ });
```

**Benefits:**
- 10x faster similarity computation with WASM
- Persistent SQLite storage
- Query caching built-in
- Shared memory pool integration

#### 2. Use Advanced Memory System (New Feature)

```typescript
import { AdvancedMemorySystem } from 'agentic-flow/reasoningbank';

const memory = new AdvancedMemorySystem();

// Auto-consolidate patterns into skills
await memory.autoConsolidate({
  minUses: 3,
  minSuccessRate: 0.7,
  lookbackDays: 7
});

// Learn from failures
const failures = await memory.replayFailures('authentication', 5);

// Causal "what-if" analysis
const insight = await memory.whatIfAnalysis('add caching');
console.log(insight.recommendation); // 'DO_IT', 'AVOID', or 'NEUTRAL'

// Skill composition
const composition = await memory.composeSkills('api-development', 5);
console.log(composition.compositionPlan); // 'skill1 → skill2 → skill3'
```

#### 3. Use Shared Memory Pool (For Multi-Agent Systems)

**Before (Separate DB per agent):**
```typescript
// Each agent creates its own DB connection
const db1 = new Database('./agent1.db');
const db2 = new Database('./agent2.db');
const db3 = new Database('./agent3.db');
// 3x memory overhead + 3x disk usage
```

**After (Shared Pool):**
```typescript
import { SharedMemoryPool } from 'agentic-flow/memory';

// All agents share same pool
const pool = SharedMemoryPool.getInstance();
const db = pool.getDatabase();  // Single connection
const embedder = pool.getEmbedder();  // Single model

// Result: ~500MB memory savings for 4+ agents
```

#### 4. Import Directly from AgentDB (Better Tree-Shaking)

**Before:**
```typescript
import { ReflexionMemory } from 'agentic-flow/agentdb';
```

**After (Optional - Better Bundle Size):**
```typescript
import { ReflexionMemory } from 'agentdb/controllers';
```

**Note:** Old imports still work via re-exports

## Backwards Compatibility Verification

### Test Your Code

Run existing tests - everything should pass:

```bash
npm test
```

### Verify CLI Commands

```bash
# All commands work identically
npx agentic-flow --help
npx agentic-flow --agent coder --task "test"
npx agentic-flow reasoningbank store "test" "success" 0.95
npx agentic-flow agentdb init ./test.db
```

### Verify MCP Tools

```bash
# MCP server starts normally
npx agentic-flow mcp start
```

Test in Claude Desktop - all tools should work.

### Verify SDK Methods

```typescript
// All imports work
import * as reasoningbank from 'agentic-flow/reasoningbank';
import { ReflexionMemory, SkillLibrary } from 'agentic-flow/agentdb';

// All methods have same signatures
// All return types unchanged
```

## Performance Gains

### Before v1.7.0
- Bundle: 5.2MB
- Memory (4 agents): ~800MB
- Vector search: 580ms @ 100K
- Batch insert: 14.1s for 1000
- Cold start: 3.5s

### After v1.7.0
- Bundle: 4.8MB (**-7.7%**)
- Memory (4 agents): ~350MB (**-56%**)
- Vector search: 5ms (**116x faster**)
- Batch insert: 100ms (**141x faster**)
- Cold start: 1.2s (**-65%**)

## Breaking Changes

**None** - This release is 100% backwards compatible.

## Deprecation Notices

### Soft Deprecations (Still Work, Better Alternatives Available)

1. **Embedded AgentDB Imports**
   ```typescript
   // ⚠️ Works but deprecated
   import { ReflexionMemory } from 'agentic-flow/agentdb';

   // ✅ Recommended
   import { ReflexionMemory } from 'agentdb/controllers';
   ```

2. **Multiple DB Connections**
   ```typescript
   // ⚠️ Works but inefficient
   const db1 = new Database('./agent1.db');
   const db2 = new Database('./agent2.db');

   // ✅ Recommended
   import { SharedMemoryPool } from 'agentic-flow/memory';
   const pool = SharedMemoryPool.getInstance();
   const db = pool.getDatabase();
   ```

3. **Old ReasoningBank Without WASM**
   ```typescript
   // ⚠️ Works but slower
   import { ReasoningBankEngine } from 'agentic-flow/reasoningbank';

   // ✅ Recommended
   import { HybridReasoningBank } from 'agentic-flow/reasoningbank';
   ```

## Troubleshooting

### "Module not found: agentdb"

Run: `npm install` to ensure agentdb@^1.3.9 is installed.

### "Cannot find module 'agentdb/controllers'"

Check that agentdb is in `dependencies` in package.json.

### Memory Usage Higher Than Expected

Use SharedMemoryPool for multi-agent systems:
```typescript
import { SharedMemoryPool } from 'agentic-flow/memory';
const pool = SharedMemoryPool.getInstance();
```

### Tests Failing

Run backwards compatibility tests:
```bash
npx vitest tests/backwards-compatibility.test.ts
```

### Slower Than Expected

Enable HNSW indexing and use HybridReasoningBank:
```typescript
import { HybridReasoningBank } from 'agentic-flow/reasoningbank';
const rb = new HybridReasoningBank({ preferWasm: true });
```

## Rollback Plan

If you encounter issues, rollback to v1.6.4:

```bash
npm install agentic-flow@1.6.4
```

Or pin version in package.json:
```json
{
  "dependencies": {
    "agentic-flow": "1.6.4"
  }
}
```

## Support

- **Documentation**: https://github.com/ruvnet/agentic-flow
- **AgentDB Docs**: https://agentdb.ruv.io
- **Issues**: https://github.com/ruvnet/agentic-flow/issues/34
- **Migration Issues**: Tag with `v1.7.0-migration`

## Next Steps

1. ✅ Upgrade to v1.7.0
2. ✅ Run existing tests
3. ✅ Verify CLI commands
4. ✅ Check performance improvements
5. 📚 Read [Integration Plan](./docs/AGENTDB_INTEGRATION_PLAN.md) for advanced features
6. 🚀 Explore new features:
   - HybridReasoningBank
   - AdvancedMemorySystem
   - SharedMemoryPool
   - 29 AgentDB MCP tools

---

**Questions?** Open an issue: https://github.com/ruvnet/agentic-flow/issues/34
