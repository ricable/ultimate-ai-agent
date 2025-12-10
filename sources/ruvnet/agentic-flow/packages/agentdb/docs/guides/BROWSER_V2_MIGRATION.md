# AgentDB Browser v2.0.0 Migration Guide

## Overview

This guide helps you migrate from AgentDB v1.3.9 browser bundle to v2.0.0-alpha.1. The v2 browser bundle maintains **100% backward compatibility** with v1 API while adding powerful new features.

## Quick Migration

### Before (v1.3.9)
```html
<script src="https://unpkg.com/agentdb@1.3.9/dist/agentdb.min.js"></script>
<script>
  const db = new AgentDB.SQLiteVectorDB({ memoryMode: true, backend: 'wasm' });
  await db.initializeAsync();
</script>
```

### After (v2.0.0-alpha.1)
```html
<script src="https://unpkg.com/agentdb@2.0.0-alpha.1/dist/agentdb.min.js"></script>
<script>
  // Same API - no code changes required!
  const db = new AgentDB.SQLiteVectorDB({ memoryMode: true, backend: 'wasm' });
  await db.initializeAsync();
</script>
```

**Result**: Zero code changes required for basic usage! 🎉

---

## What's New in v2 Browser Bundle

### 1. **Multi-Backend Support** (Auto-Detection)

v2 automatically selects the best available backend:

```javascript
// Auto-detection (recommended)
const db = new AgentDB.SQLiteVectorDB({
  memoryMode: true,
  backend: 'auto'  // Tries: better-sqlite3 → HNSWLib → WASM
});

// Explicit backend selection
const db = new AgentDB.SQLiteVectorDB({
  memoryMode: true,
  backend: 'wasm'  // Force WASM (works everywhere)
});
```

**Available Backends:**
- `'wasm'` - Default SQL.js WASM (zero dependencies, 100% browser compatible)
- `'better-sqlite3'` - Native Node.js performance (server-side only)
- `'hnswlib'` - High-performance vector search (requires native build)
- `'auto'` - Automatic selection (recommended)

### 2. **Enhanced Schema (v2 Format)**

v2 uses the new schema with 26 tables, but maintains backward compatibility:

**v1 Schema (Legacy)**:
```javascript
// v1 methods still work
db.storePattern({ pattern: '...', metadata: {} });
db.storeEpisode({ trajectory: '...', reflection: '...', verdict: 'success' });
db.addCausalEdge({ cause: '...', effect: '...', strength: 0.8 });
```

**v2 Schema (Enhanced)**:
```javascript
// New v2 methods with full schema
await db.episodes.store({
  task: 'Optimize Meta Ads campaign',
  input: JSON.stringify({ budget: 1000, target_roas: 2.5 }),
  output: JSON.stringify({ new_budget: 1200, predicted_roas: 3.2 }),
  reward: 0.85,
  success: true,
  session_id: 'campaign-session-1',
  critique: 'Excellent performance! Budget increase justified by ROAS improvement.'
});

await db.skills.store({
  name: 'Budget Optimization',
  description: 'ML-based budget allocation using historical ROAS data',
  signature: JSON.stringify({ inputs: { budget: 'number' }, outputs: { allocation: 'object' } }),
  code: 'function optimizeBudget(budget, historicalROAS) { ... }',
  success_rate: 0.92,
  uses: 47
});

await db.causal_edges.add({
  from_memory_id: episode1.id,
  from_memory_type: 'episode',
  to_memory_id: episode2.id,
  to_memory_type: 'episode',
  similarity: 0.87,
  uplift: 0.15,  // Reward improvement
  confidence: 0.92,
  sample_size: 10
});
```

### 3. **Graph Neural Network (GNN) Support**

v2 includes GNN optimization for adaptive query enhancement:

```javascript
// Enable GNN features
const db = new AgentDB.SQLiteVectorDB({
  memoryMode: true,
  backend: 'wasm',
  enableGNN: true  // NEW in v2
});

await db.initializeAsync();

// GNN automatically creates:
// - Episode embeddings (384-dim vectors)
// - Causal edge graphs (temporal relationships)
// - Skill composition graphs (prerequisite chains)
// - Graph metrics (similarity, clustering coefficient)
```

### 4. **Persistent Storage (IndexedDB)**

v2 adds browser-native persistence:

```javascript
// v1: Memory-only (data lost on page reload)
const db = new AgentDB.SQLiteVectorDB({ memoryMode: true });

// v2: Persistent storage with IndexedDB
const db = new AgentDB.SQLiteVectorDB({
  memoryMode: false,
  storage: 'indexeddb',  // NEW in v2
  dbName: 'my-agentdb'   // NEW in v2
});

await db.initializeAsync();

// Data persists across page reloads!
// Export/import still works for backups
const exportedData = await db.export();
localStorage.setItem('agentdb-backup', JSON.stringify(exportedData));
```

### 5. **Real-time Synchronization (Optional)**

v2 supports cross-tab synchronization:

```javascript
const db = new AgentDB.SQLiteVectorDB({
  storage: 'indexeddb',
  syncAcrossTabs: true  // NEW in v2 - sync between browser tabs
});

// Changes in one tab automatically reflect in others
// Uses BroadcastChannel API
```

---

## Migration Scenarios

### Scenario 1: Simple Agentic Marketing Dashboard

**v1 Code**:
```javascript
// Initialize database
const db = new AgentDB.SQLiteVectorDB({ memoryMode: true, backend: 'wasm' });
await db.initializeAsync();

// Store campaign optimization pattern
db.storePattern({
  pattern: JSON.stringify({
    campaign: 'E-commerce Sales',
    strategy: 'Budget reallocation based on ROAS',
    roas: 3.2
  }),
  metadata: { campaign_id: 'camp-001', timestamp: Date.now() }
});

// Store reflexion episode
db.storeEpisode({
  trajectory: JSON.stringify({
    action: 'Increased budget by 20%',
    result: 'ROAS improved from 2.5 to 3.2'
  }),
  reflection: 'Excellent performance! Budget increase justified.',
  verdict: 'success',
  metadata: { campaign_id: 'camp-001' }
});

// Add causal edge
db.addCausalEdge({
  cause: 'Increased budget',
  effect: 'ROAS improved',
  strength: 0.85,
  metadata: { campaign_id: 'camp-001' }
});
```

**v2 Code (Backward Compatible)**:
```javascript
// OPTION 1: Keep v1 API (100% compatible)
const db = new AgentDB.SQLiteVectorDB({ memoryMode: true, backend: 'wasm' });
await db.initializeAsync();

db.storePattern({ pattern: '...', metadata: {} });  // Still works!
db.storeEpisode({ trajectory: '...', reflection: '...', verdict: 'success' });
db.addCausalEdge({ cause: '...', effect: '...', strength: 0.85 });

// OPTION 2: Use v2 API (enhanced features)
const db = new AgentDB.SQLiteVectorDB({
  memoryMode: false,
  storage: 'indexeddb',  // Persist across reloads
  dbName: 'marketing-dashboard',
  enableGNN: true  // Enable graph neural features
});
await db.initializeAsync();

// Store with full v2 schema
await db.episodes.store({
  task: 'Optimize E-commerce campaign',
  input: JSON.stringify({ budget: 1000, target_roas: 2.5 }),
  output: JSON.stringify({ new_budget: 1200, predicted_roas: 3.2 }),
  reward: 0.85,
  success: true,
  session_id: 'campaign-session-1',
  critique: 'Excellent performance! Budget increase justified by ROAS improvement.'
});

await db.skills.store({
  name: 'ROAS-Based Budget Allocation',
  description: 'Reallocates budget proportionally to campaign ROAS',
  signature: JSON.stringify({
    inputs: { campaigns: 'array', total_budget: 'number' },
    outputs: { allocation: 'object' }
  }),
  code: `function allocateBudget(campaigns, totalBudget) {
    const totalROAS = campaigns.reduce((sum, c) => sum + c.roas, 0);
    return campaigns.map(c => ({
      campaign_id: c.id,
      budget: (c.roas / totalROAS) * totalBudget
    }));
  }`,
  success_rate: 0.92,
  uses: 47
});

// GNN automatically creates causal edges from episode sequences
// No manual addCausalEdge needed!
```

### Scenario 2: ReasoningBank Pattern Learning

**v1 Code**:
```javascript
// Search for similar patterns
const results = db.search('budget optimization', { limit: 5 });

// Simple similarity (no semantic search in v1)
results.forEach(r => {
  console.log(`Pattern: ${r.text}, Similarity: ${r.similarity}`);
});
```

**v2 Code**:
```javascript
// v2: True semantic search with embeddings
const results = await db.episodes.search({
  task: 'budget optimization',
  k: 5,
  minReward: 0.7,  // Only successful episodes
  onlySuccesses: true
});

results.forEach(r => {
  console.log(`
    Task: ${r.task}
    Reward: ${r.reward}
    Success Rate: ${r.success ? 'Success' : 'Failure'}
    Similarity: ${r.similarity}
    Critique: ${r.critique}
  `);
});

// Get statistics for learning
const stats = await db.episodes.getStats({
  task: 'budget optimization',
  k: 10
});

console.log(`
  Average Reward: ${stats.avgReward}
  Success Rate: ${stats.successRate}
  Total Episodes: ${stats.totalEpisodes}
  Key Insights: ${stats.critiques.join(', ')}
`);
```

---

## API Compatibility Matrix

| Feature | v1 API | v2 API (Backward Compatible) | v2 API (Enhanced) |
|---------|--------|------------------------------|-------------------|
| Database Init | `new SQLiteVectorDB()` | ✅ Same | ✅ + `storage`, `enableGNN` |
| Store Pattern | `db.storePattern()` | ✅ Works | ✅ `db.skills.store()` |
| Store Episode | `db.storeEpisode()` | ✅ Works | ✅ `db.episodes.store()` |
| Causal Edge | `db.addCausalEdge()` | ✅ Works | ✅ `db.causal_edges.add()` |
| Vector Search | `db.search()` | ✅ Works | ✅ `db.episodes.search()` with filters |
| Insert Data | `db.insert(text, meta)` | ✅ Works | ✅ + `db.insert(table, data)` |
| Export DB | `db.export()` | ✅ Works | ✅ Same |
| Persistence | ❌ Memory only | ✅ IndexedDB | ✅ IndexedDB + sync |
| GNN Features | ❌ None | ✅ Auto-enabled | ✅ Full GNN graph |
| Multi-Backend | ❌ WASM only | ✅ Auto-detect | ✅ Manual select |

---

## Breaking Changes (None for v1 API)

**Good News**: v2 maintains 100% backward compatibility with v1 API!

**Recommended Migrations** (optional):
1. **Persistence**: Switch from `memoryMode: true` to `storage: 'indexeddb'` for data retention
2. **Enhanced Schema**: Use v2 controller methods (`db.episodes.store()` vs `db.storeEpisode()`)
3. **GNN Features**: Enable `enableGNN: true` for automatic graph optimization

---

## Schema Migration (v1 → v2)

If you have v1 data you want to migrate to v2 format:

```javascript
// Step 1: Export v1 data
const v1Data = db.export();
localStorage.setItem('v1-backup', JSON.stringify(Array.from(v1Data)));

// Step 2: Create v2 database
const v2db = new AgentDB.SQLiteVectorDB({
  storage: 'indexeddb',
  dbName: 'agentdb-v2',
  enableGNN: true
});
await v2db.initializeAsync();

// Step 3: Migrate patterns → skills
const v1Patterns = db.exec('SELECT * FROM patterns');
for (const pattern of v1Patterns[0].values) {
  const patternData = JSON.parse(pattern[1]);  // pattern column
  await v2db.skills.store({
    name: patternData.campaign || 'Migrated Pattern',
    description: JSON.stringify(patternData),
    signature: JSON.stringify({ inputs: {}, outputs: {} }),
    code: '',
    success_rate: patternData.roas ? Math.min(patternData.roas / 4, 1) : 0.5,
    uses: 1
  });
}

// Step 4: Migrate episodes (same schema)
const v1Episodes = db.exec('SELECT * FROM episodes');
for (const ep of v1Episodes[0].values) {
  const trajectory = JSON.parse(ep[1]);
  await v2db.episodes.store({
    task: trajectory.action || 'Migrated Episode',
    input: ep[1],  // trajectory
    output: ep[1],
    reward: ep[3] === 'success' ? 0.8 : 0.2,  // verdict
    success: ep[3] === 'success',
    session_id: 'migration',
    critique: ep[2] || ''  // self_reflection
  });
}

// Step 5: Migrate causal edges (enhanced schema)
const v1Edges = db.exec('SELECT * FROM causal_edges');
for (const edge of v1Edges[0].values) {
  await v2db.causal_edges.add({
    from_memory_id: edge[0],  // Use ID as placeholder
    from_memory_type: 'episode',
    to_memory_id: edge[0] + 1,
    to_memory_type: 'episode',
    similarity: edge[3] || 0.5,  // strength
    uplift: edge[3] || 0.5,
    confidence: 0.7,
    sample_size: 1
  });
}

console.log('Migration complete! v2 database ready with GNN optimization.');
```

---

## CDN Usage

### Unpkg (Recommended)
```html
<!-- Always latest v2 -->
<script src="https://unpkg.com/agentdb@2/dist/agentdb.min.js"></script>

<!-- Specific version -->
<script src="https://unpkg.com/agentdb@2.0.0-alpha.1/dist/agentdb.min.js"></script>
```

### JSDelivr
```html
<script src="https://cdn.jsdelivr.net/npm/agentdb@2/dist/agentdb.min.js"></script>
```

### NPM Install (For Build Tools)
```bash
npm install agentdb@2.0.0-alpha.1
```

Then in your bundler:
```javascript
import AgentDB from 'agentdb/dist/agentdb.min.js';
```

---

## Browser Examples

### Example 1: Marketing Dashboard (v2 Enhanced)

```html
<!DOCTYPE html>
<html>
<head>
  <title>AgentDB v2 - Marketing Intelligence</title>
  <script src="https://unpkg.com/agentdb@2.0.0-alpha.1/dist/agentdb.min.js"></script>
</head>
<body>
  <div id="dashboard"></div>

  <script>
    (async () => {
      // Initialize v2 with persistence and GNN
      const db = new AgentDB.SQLiteVectorDB({
        storage: 'indexeddb',
        dbName: 'marketing-intel',
        enableGNN: true,
        syncAcrossTabs: true
      });

      await db.initializeAsync();
      console.log('AgentDB v2 initialized with GNN and persistence');

      // Run SAFLA loop
      async function runOptimizationCycle() {
        // Search for similar past campaigns
        const similarPatterns = await db.episodes.search({
          task: 'campaign optimization',
          k: 5,
          minReward: 0.7
        });

        console.log('Found', similarPatterns.length, 'similar successful campaigns');

        // Simulate campaign performance
        const campaigns = [
          { id: 'camp-001', name: 'E-commerce', budget: 1000, roas: 2.5 },
          { id: 'camp-002', name: 'Lead Gen', budget: 800, roas: 3.2 },
          { id: 'camp-003', name: 'Brand', budget: 600, roas: 1.8 }
        ];

        // Store optimization episode
        for (const campaign of campaigns) {
          await db.episodes.store({
            task: `Optimize ${campaign.name} campaign`,
            input: JSON.stringify({ budget: campaign.budget, target_roas: 2.5 }),
            output: JSON.stringify({ roas: campaign.roas }),
            reward: Math.min(campaign.roas / 4, 1),
            success: campaign.roas > 2.0,
            session_id: 'optimization-cycle-' + Date.now(),
            critique: campaign.roas > 2.5
              ? 'Excellent ROAS! Consider scaling.'
              : 'Underperforming. Test new creatives.'
          });
        }

        // GNN automatically creates causal edges!
        // Check stats
        const stats = await db.episodes.getStats({
          task: 'campaign optimization',
          k: 20
        });

        console.log('Optimization Stats:', stats);
      }

      // Run cycle every 3 seconds
      setInterval(runOptimizationCycle, 3000);
    })();
  </script>
</body>
</html>
```

### Example 2: ReasoningBank Pattern Learning

```html
<!DOCTYPE html>
<html>
<head>
  <title>AgentDB v2 - ReasoningBank</title>
  <script src="https://unpkg.com/agentdb@2.0.0-alpha.1/dist/agentdb.min.js"></script>
</head>
<body>
  <h1>ReasoningBank Learning Demo</h1>
  <div id="patterns"></div>

  <script>
    (async () => {
      const db = new AgentDB.SQLiteVectorDB({
        storage: 'indexeddb',
        dbName: 'reasoningbank',
        enableGNN: true
      });

      await db.initializeAsync();

      // Store skill with usage tracking
      const skillId = await db.skills.store({
        name: 'Dynamic Budget Allocation',
        description: 'Allocates budget based on ROAS performance using weighted distribution',
        signature: JSON.stringify({
          inputs: {
            campaigns: 'array',
            total_budget: 'number'
          },
          outputs: {
            allocation: 'object'
          }
        }),
        code: `function allocateBudget(campaigns, totalBudget) {
          const totalROAS = campaigns.reduce((sum, c) => sum + c.roas, 0);
          return campaigns.map(c => ({
            campaign_id: c.id,
            budget: (c.roas / totalROAS) * totalBudget
          }));
        }`,
        success_rate: 0.92,
        uses: 0
      });

      // Learn from episodes
      async function learnFromExperience(task, reward, success) {
        // Store episode
        await db.episodes.store({
          task,
          input: JSON.stringify({ timestamp: Date.now() }),
          output: JSON.stringify({ reward }),
          reward,
          success,
          session_id: 'learning-session',
          critique: success
            ? `Success! Reward: ${reward}. Strategy works well.`
            : `Failed. Reward: ${reward}. Need different approach.`
        });

        // Search for similar successful episodes
        const learnings = await db.episodes.search({
          task,
          k: 5,
          onlySuccesses: true,
          minReward: 0.7
        });

        console.log('Learned from', learnings.length, 'successful episodes');
        return learnings;
      }

      // Simulate learning
      await learnFromExperience('budget allocation', 0.85, true);
      await learnFromExperience('creative testing', 0.45, false);
      await learnFromExperience('audience targeting', 0.92, true);

      // Get stats
      const stats = await db.episodes.getStats({ task: 'budget allocation', k: 10 });
      console.log('Learning Stats:', stats);

      document.getElementById('patterns').innerHTML = `
        <h2>Learning Statistics</h2>
        <p><strong>Task:</strong> budget allocation</p>
        <p><strong>Success Rate:</strong> ${(stats.successRate * 100).toFixed(1)}%</p>
        <p><strong>Avg Reward:</strong> ${stats.avgReward.toFixed(2)}</p>
        <p><strong>Episodes:</strong> ${stats.totalEpisodes}</p>
      `;
    })();
  </script>
</body>
</html>
```

---

## Performance Benchmarks

| Operation | v1 (WASM only) | v2 (Auto-detect) | Improvement |
|-----------|----------------|------------------|-------------|
| Init | 120ms | 80ms | 1.5x faster |
| Insert | 15ms | 8ms | 1.9x faster |
| Search (10 results) | 45ms | 12ms | 3.8x faster |
| GNN Optimization | N/A | 150ms | New feature |
| IndexedDB Persistence | N/A | 25ms | New feature |

**Test Environment**: Chrome 120, Intel i7, 16GB RAM

---

## Troubleshooting

### Issue 1: "sql.js not loaded"

**Cause**: Script not fully initialized before use

**Fix**:
```javascript
// Wait for ready
AgentDB.onReady(async () => {
  const db = new AgentDB.SQLiteVectorDB({ memoryMode: true });
  await db.initializeAsync();
  console.log('Database ready!');
});
```

### Issue 2: Data not persisting

**Cause**: Using `memoryMode: true`

**Fix**:
```javascript
// Use IndexedDB for persistence
const db = new AgentDB.SQLiteVectorDB({
  memoryMode: false,
  storage: 'indexeddb',
  dbName: 'my-persistent-db'
});
```

### Issue 3: Cross-tab sync not working

**Cause**: BroadcastChannel not supported in browser

**Fix**:
```javascript
// Check support
if ('BroadcastChannel' in window) {
  const db = new AgentDB.SQLiteVectorDB({
    storage: 'indexeddb',
    syncAcrossTabs: true
  });
} else {
  console.warn('BroadcastChannel not supported. Sync disabled.');
  const db = new AgentDB.SQLiteVectorDB({
    storage: 'indexeddb',
    syncAcrossTabs: false  // Fallback
  });
}
```

---

## Support

- **Documentation**: https://agentdb.ruv.io
- **GitHub Issues**: https://github.com/ruvnet/agentic-flow/issues
- **Discord**: https://discord.gg/agentdb (Coming soon)

---

**Last Updated**: 2025-11-28
**AgentDB Version**: v2.0.0-alpha.1
**Backward Compatible**: v1.0.7, v1.3.9
