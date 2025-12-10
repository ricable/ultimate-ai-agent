# AgentDB Migration Guide v2.0.0

## Overview

AgentDB v2.0.0 introduces a powerful migration tool that automatically converts legacy databases to the new v2 format with GNN (Graph Neural Network) optimization.

**Supported Source Formats:**
- AgentDB v1.x databases
- Claude-Flow memory databases (`.swarm/memory.db`)
- Custom SQLite databases with compatible schemas

## Quick Start

### Basic Migration

```bash
# Migrate with default settings (includes GNN optimization)
agentdb migrate /path/to/legacy.db

# Specify target location
agentdb migrate /path/to/legacy.db --target /path/to/new-v2.db

# Dry run to analyze without migrating
agentdb migrate /path/to/legacy.db --dry-run
```

### Advanced Options

```bash
# Skip GNN optimization (faster, but no graph analysis)
agentdb migrate legacy.db --no-optimize

# Verbose output (detailed progress)
agentdb migrate legacy.db --verbose

# Full command with all options
agentdb migrate legacy.db \
  --target migrated-v2.db \
  --verbose \
  --dry-run
```

## Migration Process

### 1. **Automatic Detection**

The migration tool automatically detects your source database type:

- **Claude-Flow Memory**: Detects `memory_entries`, `patterns`, `task_trajectories` tables
- **AgentDB v1**: Detects `episodes`, `skills`, `facts`, `notes` tables
- **Unknown**: Reports unrecognized schema

### 2. **Data Transformation**

**From Claude-Flow Memory:**
- `memory_entries` → `episodes` (68,861 records migrated in test)
- `patterns` → `skills` (with success rates from confidence scores)
- `task_trajectories` → `events` (execution history)
- `pattern_embeddings` → Preserved and enhanced
- `pattern_links` → `skill_links` (relationship graph)

**From AgentDB v1:**
- Direct table-to-table migration for compatible schemas
- Column mapping for renamed fields
- Metadata preservation

### 3. **GNN Optimization** (Default: Enabled)

The migration automatically creates graph structures for RuVector GNN training:

#### **Episode Embeddings**
- Generates 384-dimensional embeddings for episodes
- Uses mock embeddings during migration (can be regenerated with real ML later)
- Creates `episode_embeddings` table with BLOB storage

**Example output:**
```
✅ Generated 1000 episode embeddings
```

#### **Causal Edge Analysis**
- Analyzes episode sequences within sessions
- Creates causal relationships based on:
  - Temporal order (step → next step)
  - Reward delta (uplift calculation)
  - Session grouping
- Populates `causal_edges` table

**Example output:**
```
✅ Created 68,841 causal edges
```

#### **Skill Link Creation**
- Links related skills based on success patterns
- Creates skill composition graphs
- Enables skill recommendation and chaining

**Example output:**
```
✅ Created 0 skill links (1 skill found)
```

#### **Graph Metrics**
- **Average Similarity**: Mean similarity score across edges
- **Clustering Coefficient**: Graph connectivity measure
- Both used by GNN for attention weight optimization

### 4. **Performance Stats**

Migration provides detailed performance metrics:

```
Performance Metrics:
  Migration time:        9.03s
  Optimization time:     7.91s
  Total time:            17.02s
  Records/second:        4045
```

## Migration Report

After completion, you'll receive a comprehensive report:

```
🎉 Migration Complete!

Migration Summary:
============================================================

Source Information:
  Type: claude-flow-memory
  Tables found: 9

Records Migrated:
  memoryEntries         68861
  patterns                  1
  trajectories              0
  ----------------------------
  Total                 68862

GNN Optimization Results:
  Episode embeddings:    1000
  Causal edges created:  68841
  Skill links created:   0
  Avg similarity score:  0.750
  Clustering coeff:      0.420

Performance Metrics:
  Migration time:        9.03s
  Optimization time:     7.91s
  Total time:            17.02s
  Records/second:        4045

✅ Database ready for RuVector GNN training
```

## Schema Mapping

### Claude-Flow Memory → AgentDB v2

| Source Table | Target Table | Transformation |
|--------------|--------------|----------------|
| `memory_entries` | `episodes` | key → task, value → output, namespace → session_id |
| `patterns` | `skills` | type → name, confidence → success_rate, usage_count → uses |
| `task_trajectories` | `events` | task_id → session_id, agent_id → role, trajectory_json → content |
| `pattern_embeddings` | `skill_embeddings` | Direct copy with model field mapping |
| `pattern_links` | `skill_links` | src_id/dst_id → parent_skill_id/child_skill_id |

### AgentDB v1 → AgentDB v2

| Source Table | Target Table | Changes |
|--------------|--------------|---------|
| `episodes` | `episodes` | Direct migration (compatible schema) |
| `skills` | `skills` | Added `signature` field (JSON), removed `domain` field |
| `facts` | `facts` | Direct migration |
| `notes` | `notes` | Direct migration |
| `events` | `events` | Column renaming: event_type → phase, payload → content |

## Post-Migration Steps

### 1. Verify Migration

```bash
# Check table counts
sqlite3 migrated-v2.db "
  SELECT 'episodes' as table_name, COUNT(*) FROM episodes
  UNION ALL
  SELECT 'skills', COUNT(*) FROM skills
  UNION ALL
  SELECT 'causal_edges', COUNT(*) FROM causal_edges
  UNION ALL
  SELECT 'episode_embeddings', COUNT(*) FROM episode_embeddings;
"
```

### 2. Generate Real Embeddings (Optional)

If you skipped embeddings during migration or want real ML embeddings:

```bash
# Install embedding dependencies
agentdb install-embeddings

# Regenerate embeddings for all episodes
agentdb regenerate-embeddings migrated-v2.db
```

### 3. Test with MCP Server

```bash
# Start MCP server with migrated database
AGENTDB_PATH=migrated-v2.db agentdb mcp start
```

### 4. Train GNN Models (RuVector)

```bash
# Install RuVector for GNN training
npm install @ruvector/core @ruvector/gnn

# Train GNN attention model
agentdb train migrated-v2.db --model gnn-attention
```

## Troubleshooting

### Migration Fails with "Table Not Found"

**Cause**: Source database has unexpected schema
**Solution**: Use `--dry-run` to inspect schema first

```bash
agentdb migrate legacy.db --dry-run
```

### Slow Migration Performance

**Cause**: Large database (>100K records)
**Solutions**:
1. Skip GNN optimization: `--no-optimize`
2. Use better-sqlite3 backend (faster):
   ```bash
   npm install better-sqlite3
   agentdb migrate legacy.db --backend better-sqlite3
   ```

### Memory Issues During Migration

**Cause**: Generating embeddings for millions of records
**Solutions**:
1. Migrate in batches
2. Skip optimization initially: `--no-optimize`
3. Generate embeddings later with `regenerate-embeddings`

### Column Mismatch Errors

**Cause**: Source schema incompatible
**Solution**: Manual migration required - use SQL:

```sql
-- Example: Migrate custom schema
INSERT INTO episodes (task, input, output, reward, success, session_id, created_at)
SELECT
  custom_task_field,
  custom_input_field,
  custom_output_field,
  0.5, -- default reward
  1, -- default success
  'migration' || id, -- generate session_id
  created_timestamp
FROM legacy_custom_table;
```

## Migration Checklist

Before production migration:

- [ ] **Backup original database**
- [ ] **Test with dry-run**: `agentdb migrate old.db --dry-run`
- [ ] **Verify record counts** match expectations
- [ ] **Check storage space** (v2 uses more space for embeddings)
- [ ] **Test MCP server** with migrated database
- [ ] **Regenerate embeddings** for production (optional)
- [ ] **Update application** database path to new v2 file

## Docker Migration

Run migration in Docker container:

```dockerfile
FROM agentdb:2.0.0-alpha.1

# Copy legacy database
COPY legacy.db /app/data/legacy.db

# Run migration
RUN agentdb migrate /app/data/legacy.db --target /app/data/production-v2.db

# Use migrated database
ENV AGENTDB_PATH=/app/data/production-v2.db
CMD ["agentdb", "mcp", "start"]
```

Or use docker-compose:

```yaml
services:
  agentdb-migration:
    image: agentdb:2.0.0-alpha.1
    volumes:
      - ./legacy.db:/data/legacy.db
      - ./migrated:/data/output
    command: agentdb migrate /data/legacy.db --target /data/output/v2.db --verbose
```

## Performance Benchmarks

Migration performance varies by database size:

| Database Size | Records | Migration Time | Records/sec | GNN Time |
|---------------|---------|----------------|-------------|----------|
| Small | 1K | <1s | ~5,000 | <1s |
| Medium | 10K | ~2s | ~5,000 | ~2s |
| Large | 100K | ~20s | ~5,000 | ~15s |
| Very Large | 1M+ | ~5min | ~3,500 | ~3min |

**Test environment**: Node.js 20, SQLite WASM, single-threaded

**Optimization**: Use `better-sqlite3` backend for 2-3x faster performance on Node.js

## API Usage

Programmatic migration:

```typescript
import { migrateCommand } from 'agentdb/cli/commands/migrate';

await migrateCommand({
  sourceDb: './legacy.db',
  targetDb: './v2.db',
  optimize: true,
  dryRun: false,
  verbose: true
});
```

## Support

- **Issues**: https://github.com/ruvnet/agentic-flow/issues
- **Discussions**: https://github.com/ruvnet/agentic-flow/discussions
- **Documentation**: https://agentdb.ruv.io

---

**Last Updated**: 2025-11-28
**AgentDB Version**: v2.0.0-alpha.1
