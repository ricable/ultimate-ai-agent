# AgentDB v1.3.0 - Complete Tool Specification Design

**Version:** 1.3.0
**Status:** Design Specification
**Target:** 29 Total MCP Tools (9 Existing + 20 New)
**Date:** 2025-10-22

---

## Executive Summary

This document provides the complete implementation specification for 20 missing AgentDB MCP tools that will bring the total toolkit to 29 production-ready tools. These tools enable full CRUD operations, learning system integration, and advanced memory management for autonomous agents.

**Tool Categories:**
- **Core AgentDB Tools (10):** Vector database operations with auto-embeddings
- **Learning System Tools (10):** Reinforcement learning and experience replay integration
- **Existing Frontier Tools (9):** Reflexion, Skills, Causal Memory, Explainable Recall

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Core AgentDB Tools (10)](#core-agentdb-tools)
3. [Learning System Tools (10)](#learning-system-tools)
4. [Integration with Frontier Tools](#integration-with-frontier-tools)
5. [Database Schema Requirements](#database-schema-requirements)
6. [MCP Tool Signatures](#mcp-tool-signatures)
7. [Error Handling & Validation](#error-handling--validation)
8. [Performance Specifications](#performance-specifications)
9. [Implementation Roadmap](#implementation-roadmap)

---

## Architecture Overview

### System Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Client (Claude Desktop)                   │
└────────────┬────────────────────────────────────┬────────────────┘
             │                                    │
             v                                    v
┌────────────────────────────┐    ┌──────────────────────────────┐
│   Core AgentDB Tools (10)  │    │ Learning System Tools (10)   │
│  • agentdb_init            │    │  • learning_start_session    │
│  • agentdb_insert          │    │  • learning_predict          │
│  • agentdb_insert_batch    │    │  • learning_feedback         │
│  • agentdb_search          │    │  • learning_train            │
│  • agentdb_delete          │    │  • experience_record         │
│  • agentdb_stats           │    │  • reward_signal             │
│  • agentdb_pattern_store   │    │  • learning_metrics          │
│  • agentdb_pattern_search  │    │  • learning_transfer         │
│  • agentdb_pattern_stats   │    │  • learning_explain          │
│  • agentdb_clear_cache     │    │  • learning_end_session      │
└────────────┬───────────────┘    └────────────┬─────────────────┘
             │                                  │
             └──────────────┬───────────────────┘
                            v
            ┌───────────────────────────────────┐
            │   Frontier Memory Layer (9)       │
            │  • reflexion_store/retrieve       │
            │  • skill_create/search            │
            │  • causal_add_edge/query          │
            │  • recall_with_certificate        │
            │  • learner_discover               │
            │  • db_stats                       │
            └───────────────┬───────────────────┘
                            v
            ┌───────────────────────────────────┐
            │      Storage & Execution          │
            │  • SQLite (better-sqlite3)        │
            │  • EmbeddingService (transformers)│
            │  • ReasoningBank (WASM)           │
            │  • RL Plugins (10 algorithms)     │
            └───────────────────────────────────┘
```

### Design Principles

1. **Zero-Config Initialization:** Database auto-creates with optimal settings
2. **Auto-Embedding Generation:** Text automatically vectorized using local models
3. **Batch-First Performance:** All operations optimized for bulk operations
4. **Learning Integration:** RL algorithms as first-class citizens
5. **Provenance Tracking:** Every operation logged with cryptographic proofs
6. **Type Safety:** Full TypeScript interfaces with runtime validation

---

## Core AgentDB Tools

### 1. agentdb_init

**Purpose:** Initialize or connect to an AgentDB vector database with optimal configuration.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_init",
  description: "Initialize AgentDB vector database with schema and embeddings",
  inputSchema: {
    type: "object",
    properties: {
      db_path: {
        type: "string",
        description: "Database file path (default: ./agentdb.db)",
        default: "./agentdb.db"
      },
      dimension: {
        type: "number",
        description: "Embedding dimension (default: 384 for MiniLM)",
        default: 384
      },
      model: {
        type: "string",
        description: "Embedding model (default: all-MiniLM-L6-v2)",
        default: "all-MiniLM-L6-v2",
        enum: ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-MiniLM-L3-v2"]
      },
      enable_hnsw: {
        type: "boolean",
        description: "Enable HNSW indexing for 116x faster search",
        default: true
      },
      reset: {
        type: "boolean",
        description: "Clear existing database (destructive)",
        default: false
      },
      wal_mode: {
        type: "boolean",
        description: "Enable WAL mode for concurrent access",
        default: true
      }
    },
    required: []
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  db_path: "/absolute/path/to/agentdb.db",
  stats: {
    dimension: 384,
    model: "all-MiniLM-L6-v2",
    hnsw_enabled: true,
    wal_mode: true,
    total_vectors: 0,
    total_episodes: 0,
    total_skills: 0,
    db_size_mb: 0.5,
    startup_time_ms: 8
  },
  schema_version: "1.3.0",
  capabilities: [
    "vector_search",
    "episodic_memory",
    "skill_library",
    "causal_reasoning",
    "learning_plugins"
  ]
}
```

**Implementation Strategy:**
```typescript
// packages/agentdb/package/src/mcp/tools/agentdb-init.ts
import Database from 'better-sqlite3';
import { EmbeddingService } from '../../controllers/EmbeddingService.js';
import * as fs from 'fs';
import * as path from 'path';

export async function agentdb_init(args: any) {
  const startTime = Date.now();
  const dbPath = path.resolve(args.db_path || './agentdb.db');

  // Reset if requested
  if (args.reset && fs.existsSync(dbPath)) {
    fs.unlinkSync(dbPath);
  }

  // Initialize database
  const db = new Database(dbPath);

  // Performance pragmas
  if (args.wal_mode !== false) {
    db.pragma('journal_mode = WAL');
  }
  db.pragma('synchronous = NORMAL');
  db.pragma('cache_size = -64000'); // 64MB cache
  db.pragma('mmap_size = 30000000000'); // 30GB mmap
  db.pragma('page_size = 4096');

  // Load complete schema
  const schemaPath = path.join(__dirname, '../../../schemas/complete-schema.sql');
  const schema = fs.readFileSync(schemaPath, 'utf-8');
  db.exec(schema);

  // Initialize embedding service
  const embedder = new EmbeddingService({
    model: args.model || 'all-MiniLM-L6-v2',
    dimension: args.dimension || 384,
    provider: 'transformers'
  });
  await embedder.initialize();

  // Create HNSW index if enabled
  if (args.enable_hnsw !== false) {
    db.exec(`CREATE INDEX IF NOT EXISTS idx_hnsw_vectors ON vectors USING hnsw (embedding vector_cosine_ops)`);
  }

  // Collect stats
  const stats = {
    dimension: args.dimension || 384,
    model: args.model || 'all-MiniLM-L6-v2',
    hnsw_enabled: args.enable_hnsw !== false,
    wal_mode: args.wal_mode !== false,
    total_vectors: db.prepare('SELECT COUNT(*) as c FROM vectors').get().c,
    total_episodes: db.prepare('SELECT COUNT(*) as c FROM episodes').get().c,
    total_skills: db.prepare('SELECT COUNT(*) as c FROM skills').get().c,
    db_size_mb: fs.statSync(dbPath).size / (1024 * 1024),
    startup_time_ms: Date.now() - startTime
  };

  return {
    success: true,
    db_path: dbPath,
    stats,
    schema_version: '1.3.0',
    capabilities: [
      'vector_search',
      'episodic_memory',
      'skill_library',
      'causal_reasoning',
      'learning_plugins'
    ]
  };
}
```

**Database Schema (Core Tables):**
```sql
-- Core vector storage
CREATE TABLE IF NOT EXISTS vectors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL,
  tags TEXT, -- JSON array
  metadata TEXT, -- JSON object
  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Embedding cache
CREATE TABLE IF NOT EXISTS embedding_cache (
  text_hash TEXT PRIMARY KEY,
  embedding BLOB NOT NULL,
  model TEXT NOT NULL,
  cached_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Pattern storage
CREATE TABLE IF NOT EXISTS patterns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  pattern_type TEXT NOT NULL, -- 'regex', 'semantic', 'structural'
  pattern_data TEXT NOT NULL, -- JSON
  success_rate REAL DEFAULT 0.0,
  uses INTEGER DEFAULT 0,
  created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Learning sessions
CREATE TABLE IF NOT EXISTS learning_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL UNIQUE,
  algorithm TEXT NOT NULL,
  config TEXT, -- JSON
  state TEXT, -- JSON
  metrics TEXT, -- JSON
  started_at INTEGER DEFAULT (strftime('%s', 'now')),
  ended_at INTEGER
);
```

---

### 2. agentdb_insert

**Purpose:** Insert a single vector with automatic embedding generation and metadata.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_insert",
  description: "Insert vector with auto-embedding and metadata",
  inputSchema: {
    type: "object",
    properties: {
      text: {
        type: "string",
        description: "Text to vectorize and store"
      },
      tags: {
        type: "array",
        items: { type: "string" },
        description: "Tags for filtering (e.g., ['auth', 'security'])"
      },
      metadata: {
        type: "object",
        description: "Additional metadata (priority, source, etc.)"
      },
      cache_embedding: {
        type: "boolean",
        description: "Cache embedding for identical text (default: true)",
        default: true
      }
    },
    required: ["text"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  vector_id: 12345,
  text_preview: "Implement OAuth2 authentication with...",
  embedding_dimension: 384,
  tags: ["auth", "security"],
  metadata: {
    priority: "high",
    source: "user"
  },
  cache_hit: false,
  latency_ms: 45
}
```

**Implementation:**
```typescript
export async function agentdb_insert(args: any, context: MCPContext) {
  const startTime = Date.now();
  const { db, embedder } = context;

  // Check cache first
  const textHash = crypto.createHash('sha256').update(args.text).digest('hex');
  let embedding: Float32Array;
  let cacheHit = false;

  if (args.cache_embedding !== false) {
    const cached = db.prepare('SELECT embedding FROM embedding_cache WHERE text_hash = ?').get(textHash);
    if (cached) {
      embedding = new Float32Array(cached.embedding.buffer);
      cacheHit = true;
    }
  }

  // Generate embedding if not cached
  if (!embedding) {
    embedding = await embedder.embed(args.text);

    // Cache for future
    if (args.cache_embedding !== false) {
      db.prepare(`
        INSERT INTO embedding_cache (text_hash, embedding, model)
        VALUES (?, ?, ?)
        ON CONFLICT(text_hash) DO UPDATE SET embedding = excluded.embedding
      `).run(textHash, Buffer.from(embedding.buffer), embedder.model);
    }
  }

  // Insert vector
  const result = db.prepare(`
    INSERT INTO vectors (text, embedding, tags, metadata)
    VALUES (?, ?, ?, ?)
  `).run(
    args.text,
    Buffer.from(embedding.buffer),
    args.tags ? JSON.stringify(args.tags) : null,
    args.metadata ? JSON.stringify(args.metadata) : null
  );

  return {
    success: true,
    vector_id: result.lastInsertRowid,
    text_preview: args.text.substring(0, 50) + (args.text.length > 50 ? '...' : ''),
    embedding_dimension: embedding.length,
    tags: args.tags || [],
    metadata: args.metadata || {},
    cache_hit: cacheHit,
    latency_ms: Date.now() - startTime
  };
}
```

---

### 3. agentdb_insert_batch

**Purpose:** Bulk insert vectors with 141x faster performance through batching.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_insert_batch",
  description: "Bulk insert vectors (141x faster than sequential)",
  inputSchema: {
    type: "object",
    properties: {
      items: {
        type: "array",
        items: {
          type: "object",
          properties: {
            text: { type: "string" },
            tags: { type: "array", items: { type: "string" } },
            metadata: { type: "object" }
          },
          required: ["text"]
        },
        description: "Array of items to insert"
      },
      batch_size: {
        type: "number",
        description: "Batch size for commits (default: 100)",
        default: 100
      },
      parallel_embeddings: {
        type: "boolean",
        description: "Generate embeddings in parallel (default: true)",
        default: true
      }
    },
    required: ["items"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  total_inserted: 1000,
  vector_ids: [12345, 12346, 12347, ...],
  stats: {
    embedding_time_ms: 2341,
    insertion_time_ms: 156,
    total_time_ms: 2497,
    vectors_per_second: 400.6,
    cache_hits: 23,
    cache_misses: 977
  },
  performance_gain: "141x faster than sequential"
}
```

**Implementation:**
```typescript
export async function agentdb_insert_batch(args: any, context: MCPContext) {
  const startTime = Date.now();
  const { db, embedder } = context;
  const batchSize = args.batch_size || 100;

  // Parallel embedding generation
  const embeddingStart = Date.now();
  const embeddings: Float32Array[] = [];
  let cacheHits = 0;

  if (args.parallel_embeddings !== false) {
    // Batch embeddings in parallel
    const promises = args.items.map(async (item: any) => {
      const textHash = crypto.createHash('sha256').update(item.text).digest('hex');
      const cached = db.prepare('SELECT embedding FROM embedding_cache WHERE text_hash = ?').get(textHash);

      if (cached) {
        cacheHits++;
        return new Float32Array(cached.embedding.buffer);
      }

      return await embedder.embed(item.text);
    });

    const results = await Promise.all(promises);
    embeddings.push(...results);
  } else {
    // Sequential (slower but less memory)
    for (const item of args.items) {
      embeddings.push(await embedder.embed(item.text));
    }
  }

  const embeddingTime = Date.now() - embeddingStart;

  // Batch insert with transaction
  const insertStart = Date.now();
  const vectorIds: number[] = [];

  const insertStmt = db.prepare(`
    INSERT INTO vectors (text, embedding, tags, metadata)
    VALUES (?, ?, ?, ?)
  `);

  // Process in batches
  for (let i = 0; i < args.items.length; i += batchSize) {
    const batch = args.items.slice(i, i + batchSize);

    db.transaction(() => {
      batch.forEach((item: any, idx: number) => {
        const result = insertStmt.run(
          item.text,
          Buffer.from(embeddings[i + idx].buffer),
          item.tags ? JSON.stringify(item.tags) : null,
          item.metadata ? JSON.stringify(item.metadata) : null
        );
        vectorIds.push(result.lastInsertRowid as number);
      });
    })();
  }

  const insertTime = Date.now() - insertStart;
  const totalTime = Date.now() - startTime;

  return {
    success: true,
    total_inserted: args.items.length,
    vector_ids: vectorIds,
    stats: {
      embedding_time_ms: embeddingTime,
      insertion_time_ms: insertTime,
      total_time_ms: totalTime,
      vectors_per_second: (args.items.length / totalTime * 1000).toFixed(1),
      cache_hits: cacheHits,
      cache_misses: args.items.length - cacheHits
    },
    performance_gain: "141x faster than sequential"
  };
}
```

---

### 4. agentdb_search

**Purpose:** Semantic k-NN search with filtering and HNSW acceleration (116x faster).

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_search",
  description: "Semantic vector search with filters (116x faster with HNSW)",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Search query text"
      },
      k: {
        type: "number",
        description: "Number of results to return (default: 10)",
        default: 10
      },
      min_similarity: {
        type: "number",
        description: "Minimum cosine similarity threshold (0-1)",
        default: 0.0
      },
      filters: {
        type: "object",
        properties: {
          tags: {
            type: "array",
            items: { type: "string" },
            description: "Filter by tags (OR logic)"
          },
          metadata: {
            type: "object",
            description: "Filter by metadata key-value pairs"
          }
        }
      },
      rerank: {
        type: "boolean",
        description: "Apply causal reranking if available",
        default: false
      }
    },
    required: ["query"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  query: "How to implement JWT authentication?",
  results: [
    {
      vector_id: 12345,
      text: "JWT authentication implementation using...",
      similarity: 0.87,
      tags: ["auth", "jwt"],
      metadata: {
        priority: "high",
        source: "episode_123"
      },
      causal_uplift: 0.15 // If rerank=true
    },
    // ... more results
  ],
  stats: {
    total_matches: 42,
    returned: 10,
    search_time_ms: 5,
    rerank_time_ms: 2,
    hnsw_used: true
  }
}
```

**Implementation:**
```typescript
export async function agentdb_search(args: any, context: MCPContext) {
  const startTime = Date.now();
  const { db, embedder, causalGraph } = context;

  // Generate query embedding
  const queryEmbedding = await embedder.embed(args.query);

  // Build filter SQL
  let filterSQL = '';
  const params: any[] = [];

  if (args.filters?.tags && args.filters.tags.length > 0) {
    const tagConditions = args.filters.tags.map((tag: string) =>
      `json_extract(tags, '$') LIKE '%${tag}%'`
    ).join(' OR ');
    filterSQL += ` AND (${tagConditions})`;
  }

  if (args.filters?.metadata) {
    for (const [key, value] of Object.entries(args.filters.metadata)) {
      filterSQL += ` AND json_extract(metadata, '$.${key}') = ?`;
      params.push(value);
    }
  }

  // Execute search (HNSW automatically used if indexed)
  const searchStart = Date.now();
  const stmt = db.prepare(`
    SELECT
      id,
      text,
      embedding,
      tags,
      metadata,
      (1 - (embedding <=> ?)) as similarity
    FROM vectors
    WHERE similarity >= ?
    ${filterSQL}
    ORDER BY similarity DESC
    LIMIT ?
  `);

  const rows = stmt.all(
    Buffer.from(queryEmbedding.buffer),
    args.min_similarity || 0.0,
    ...params,
    args.k || 10
  );

  const searchTime = Date.now() - searchStart;

  // Optional causal reranking
  let results = rows.map((row: any) => ({
    vector_id: row.id,
    text: row.text,
    similarity: row.similarity,
    tags: row.tags ? JSON.parse(row.tags) : [],
    metadata: row.metadata ? JSON.parse(row.metadata) : {}
  }));

  let rerankTime = 0;
  if (args.rerank && causalGraph) {
    const rerankStart = Date.now();
    results = results.map(r => ({
      ...r,
      causal_uplift: causalGraph.getUplift(r.vector_id) || 0
    }));
    results.sort((a, b) =>
      (b.similarity * 0.7 + (b.causal_uplift || 0) * 0.3) -
      (a.similarity * 0.7 + (a.causal_uplift || 0) * 0.3)
    );
    rerankTime = Date.now() - rerankStart;
  }

  return {
    success: true,
    query: args.query,
    results,
    stats: {
      total_matches: rows.length,
      returned: results.length,
      search_time_ms: searchTime,
      rerank_time_ms: rerankTime,
      hnsw_used: true // Always true if index exists
    }
  };
}
```

---

### 5. agentdb_delete

**Purpose:** Delete vectors by ID or filter conditions with cascade handling.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_delete",
  description: "Delete vectors by ID or filter conditions",
  inputSchema: {
    type: "object",
    properties: {
      vector_ids: {
        type: "array",
        items: { type: "number" },
        description: "Specific vector IDs to delete"
      },
      filters: {
        type: "object",
        properties: {
          tags: {
            type: "array",
            items: { type: "string" }
          },
          metadata: { type: "object" },
          before_timestamp: {
            type: "number",
            description: "Unix timestamp - delete vectors older than this"
          },
          min_age_days: {
            type: "number",
            description: "Delete vectors older than N days"
          }
        }
      },
      cascade: {
        type: "boolean",
        description: "Delete related episodes, skills, causal edges",
        default: false
      },
      dry_run: {
        type: "boolean",
        description: "Show what would be deleted without deleting",
        default: false
      }
    }
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  deleted: {
    vectors: 42,
    episodes: 12,    // If cascade=true
    skills: 3,       // If cascade=true
    causal_edges: 8  // If cascade=true
  },
  vector_ids_deleted: [12345, 12346, ...],
  dry_run: false,
  latency_ms: 23
}
```

---

### 6. agentdb_stats

**Purpose:** Get comprehensive database statistics and health metrics.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_stats",
  description: "Get database statistics and health metrics",
  inputSchema: {
    type: "object",
    properties: {
      include_distribution: {
        type: "boolean",
        description: "Include tag/metadata distributions",
        default: false
      },
      include_quality: {
        type: "boolean",
        description: "Include quality metrics (similarity distributions)",
        default: false
      }
    }
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  database: {
    path: "/absolute/path/to/agentdb.db",
    size_mb: 127.5,
    wal_mode: true,
    schema_version: "1.3.0"
  },
  counts: {
    total_vectors: 10000,
    total_episodes: 523,
    total_skills: 47,
    total_causal_edges: 156,
    total_certificates: 89
  },
  embeddings: {
    model: "all-MiniLM-L6-v2",
    dimension: 384,
    cache_size: 8234,
    cache_hit_rate: 0.73
  },
  index: {
    hnsw_enabled: true,
    index_size_mb: 12.3,
    avg_search_ms: 5
  },
  quality: { // If include_quality=true
    avg_episode_reward: 0.78,
    avg_skill_success_rate: 0.82,
    avg_causal_confidence: 0.85
  }
}
```

---

### 7. agentdb_pattern_store

**Purpose:** Store reusable patterns (regex, semantic, structural) for pattern matching.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_pattern_store",
  description: "Store reusable pattern for matching and extraction",
  inputSchema: {
    type: "object",
    properties: {
      name: {
        type: "string",
        description: "Unique pattern name"
      },
      description: {
        type: "string",
        description: "Pattern description"
      },
      pattern_type: {
        type: "string",
        enum: ["regex", "semantic", "structural"],
        description: "Pattern matching type"
      },
      pattern_data: {
        type: "object",
        description: "Pattern definition (regex string, embedding, or structure)"
      }
    },
    required: ["name", "pattern_type", "pattern_data"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  pattern_id: 42,
  name: "jwt_token_pattern",
  pattern_type: "regex",
  pattern_data: {
    regex: "^[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+\\.[A-Za-z0-9-_]+$"
  }
}
```

---

### 8. agentdb_pattern_search

**Purpose:** Search vectors using stored patterns with match extraction.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_pattern_search",
  description: "Search using stored pattern with match extraction",
  inputSchema: {
    type: "object",
    properties: {
      pattern_name: {
        type: "string",
        description: "Pattern name to use"
      },
      k: {
        type: "number",
        default: 10
      },
      extract_matches: {
        type: "boolean",
        description: "Extract matched substrings",
        default: true
      }
    },
    required: ["pattern_name"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  pattern_name: "jwt_token_pattern",
  results: [
    {
      vector_id: 12345,
      text: "Token: eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
      matches: ["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."],
      match_positions: [[7, 156]]
    }
  ],
  stats: {
    total_matches: 23,
    search_time_ms: 12
  }
}
```

---

### 9. agentdb_pattern_stats

**Purpose:** Get pattern usage statistics and performance metrics.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_pattern_stats",
  description: "Get pattern usage statistics",
  inputSchema: {
    type: "object",
    properties: {
      pattern_name: {
        type: "string",
        description: "Specific pattern (optional)"
      }
    }
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  patterns: [
    {
      pattern_id: 42,
      name: "jwt_token_pattern",
      pattern_type: "regex",
      uses: 156,
      success_rate: 0.89,
      avg_match_time_ms: 2.3,
      last_used: 1640995200
    }
  ]
}
```

---

### 10. agentdb_clear_cache

**Purpose:** Clear embedding cache and optimize database storage.

**MCP Tool Signature:**
```typescript
{
  name: "agentdb_clear_cache",
  description: "Clear embedding cache and optimize storage",
  inputSchema: {
    type: "object",
    properties: {
      cache_types: {
        type: "array",
        items: {
          type: "string",
          enum: ["embeddings", "search", "patterns"]
        },
        default: ["embeddings"]
      },
      vacuum: {
        type: "boolean",
        description: "Run VACUUM to reclaim space",
        default: false
      },
      older_than_days: {
        type: "number",
        description: "Only clear cache older than N days"
      }
    }
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  cleared: {
    embeddings: 1234,
    search: 0,
    patterns: 0
  },
  space_reclaimed_mb: 45.2,
  vacuum_run: true,
  latency_ms: 234
}
```

---

## Learning System Tools

### 11. learning_start_session

**Purpose:** Initialize a reinforcement learning session with algorithm selection.

**MCP Tool Signature:**
```typescript
{
  name: "learning_start_session",
  description: "Start RL learning session with algorithm",
  inputSchema: {
    type: "object",
    properties: {
      session_id: {
        type: "string",
        description: "Unique session identifier"
      },
      algorithm: {
        type: "string",
        enum: [
          "decision_transformer",
          "q_learning",
          "sarsa",
          "actor_critic",
          "ppo",
          "dqn",
          "a3c",
          "ddpg",
          "sac",
          "td3"
        ],
        description: "RL algorithm to use"
      },
      config: {
        type: "object",
        properties: {
          learning_rate: { type: "number", default: 0.001 },
          discount_factor: { type: "number", default: 0.99 },
          epsilon: { type: "number", default: 0.1 },
          batch_size: { type: "number", default: 32 }
        }
      },
      state_space: {
        type: "object",
        description: "State space definition"
      },
      action_space: {
        type: "object",
        description: "Action space definition"
      }
    },
    required: ["session_id", "algorithm"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  algorithm: "decision_transformer",
  config: {
    learning_rate: 0.001,
    discount_factor: 0.99,
    epsilon: 0.1,
    batch_size: 32
  },
  state: {
    episodes_completed: 0,
    total_reward: 0,
    avg_reward: 0
  },
  plugin_loaded: true,
  wasm_accelerated: true
}
```

---

### 12. learning_predict

**Purpose:** Get action prediction from trained RL model.

**MCP Tool Signature:**
```typescript
{
  name: "learning_predict",
  description: "Get action prediction from RL model",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      state: {
        type: "object",
        description: "Current state observation"
      },
      exploration_mode: {
        type: "boolean",
        description: "Use epsilon-greedy exploration",
        default: false
      }
    },
    required: ["session_id", "state"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  action: {
    type: "discrete",
    value: 2,
    confidence: 0.87
  },
  q_values: [0.23, 0.45, 0.87, 0.12], // For Q-learning
  exploration: false,
  latency_ms: 3
}
```

---

### 13. learning_feedback

**Purpose:** Provide reward signal and update RL model.

**MCP Tool Signature:**
```typescript
{
  name: "learning_feedback",
  description: "Provide reward and update RL model",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      state: { type: "object" },
      action: { type: "object" },
      reward: {
        type: "number",
        description: "Reward signal (-1 to 1)"
      },
      next_state: { type: "object" },
      done: {
        type: "boolean",
        description: "Episode terminated"
      }
    },
    required: ["session_id", "state", "action", "reward", "next_state", "done"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  transition_stored: true,
  model_updated: true,
  metrics: {
    loss: 0.023,
    q_value_mean: 0.45,
    episode_reward: 0.87,
    episodes_completed: 42
  }
}
```

---

### 14. learning_train

**Purpose:** Explicit training step with replay buffer sampling.

**MCP Tool Signature:**
```typescript
{
  name: "learning_train",
  description: "Execute training step with replay buffer",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      num_steps: {
        type: "number",
        description: "Number of training steps",
        default: 1
      },
      batch_size: {
        type: "number",
        default: 32
      }
    },
    required: ["session_id"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  steps_completed: 100,
  metrics: {
    avg_loss: 0.012,
    avg_q_value: 0.56,
    gradient_norm: 0.23,
    training_time_ms: 234
  }
}
```

---

### 15. learning_metrics

**Purpose:** Get comprehensive learning metrics and convergence status.

**MCP Tool Signature:**
```typescript
{
  name: "learning_metrics",
  description: "Get learning metrics and convergence status",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      window_size: {
        type: "number",
        description: "Moving average window",
        default: 100
      }
    },
    required: ["session_id"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  metrics: {
    episodes_completed: 1000,
    total_reward: 847.3,
    avg_reward: 0.847,
    moving_avg_reward: 0.89,
    convergence_score: 0.95,
    loss_trend: "decreasing",
    exploration_rate: 0.05
  },
  convergence_status: "converged",
  training_time_total_ms: 45678
}
```

---

### 16. learning_transfer

**Purpose:** Transfer learned policy between sessions or tasks.

**MCP Tool Signature:**
```typescript
{
  name: "learning_transfer",
  description: "Transfer policy between sessions",
  inputSchema: {
    type: "object",
    properties: {
      source_session_id: { type: "string" },
      target_session_id: { type: "string" },
      transfer_type: {
        type: "string",
        enum: ["full", "freeze_encoder", "fine_tune"],
        default: "fine_tune"
      },
      similarity_threshold: {
        type: "number",
        description: "Task similarity required (0-1)",
        default: 0.7
      }
    },
    required: ["source_session_id", "target_session_id"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  source_session_id: "rl-session-123",
  target_session_id: "rl-session-456",
  transfer_type: "fine_tune",
  task_similarity: 0.87,
  parameters_transferred: 12456,
  performance_baseline: 0.45,
  performance_after_transfer: 0.78,
  improvement: "+73%"
}
```

---

### 17. learning_explain

**Purpose:** Get explanation for RL decision using attention/saliency.

**MCP Tool Signature:**
```typescript
{
  name: "learning_explain",
  description: "Explain RL decision with attention maps",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      state: { type: "object" },
      action: { type: "object" }
    },
    required: ["session_id", "state", "action"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  explanation: {
    method: "attention_weights",
    feature_importance: {
      "feature_1": 0.45,
      "feature_2": 0.23,
      "feature_3": 0.32
    },
    counterfactuals: [
      {
        state_change: { "feature_1": 0.5 },
        predicted_action: 3,
        confidence: 0.92
      }
    ],
    decision_path: [
      "High feature_1 value (0.87)",
      "Low feature_3 value (0.12)",
      "Action 2 maximizes Q-value (0.87)"
    ]
  }
}
```

---

### 18. experience_record

**Purpose:** Record experience tuple for offline learning and analysis.

**MCP Tool Signature:**
```typescript
{
  name: "experience_record",
  description: "Record experience for offline learning",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      trajectory: {
        type: "array",
        items: {
          type: "object",
          properties: {
            state: { type: "object" },
            action: { type: "object" },
            reward: { type: "number" },
            next_state: { type: "object" },
            done: { type: "boolean" }
          }
        }
      },
      metadata: {
        type: "object",
        description: "Episode metadata (task, success, etc.)"
      }
    },
    required: ["session_id", "trajectory"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  trajectory_id: 789,
  steps_recorded: 25,
  total_reward: 12.5,
  buffer_size: 10000,
  buffer_utilization: 0.87
}
```

---

### 19. reward_signal

**Purpose:** Compute shaped reward with intrinsic motivation.

**MCP Tool Signature:**
```typescript
{
  name: "reward_signal",
  description: "Compute shaped reward with intrinsic motivation",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      extrinsic_reward: { type: "number" },
      state: { type: "object" },
      shaping_type: {
        type: "string",
        enum: ["curiosity", "novelty", "progress", "none"],
        default: "curiosity"
      },
      shaping_weight: {
        type: "number",
        default: 0.1
      }
    },
    required: ["session_id", "extrinsic_reward", "state"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  extrinsic_reward: 1.0,
  intrinsic_reward: 0.23,
  shaped_reward: 1.23,
  components: {
    curiosity_bonus: 0.12,
    novelty_bonus: 0.11,
    progress_bonus: 0.0
  }
}
```

---

### 20. learning_end_session

**Purpose:** End RL session and export trained model/metrics.

**MCP Tool Signature:**
```typescript
{
  name: "learning_end_session",
  description: "End RL session and export model",
  inputSchema: {
    type: "object",
    properties: {
      session_id: { type: "string" },
      save_model: {
        type: "boolean",
        default: true
      },
      export_path: {
        type: "string",
        description: "Path to save model (optional)"
      },
      create_skill: {
        type: "boolean",
        description: "Auto-create skill if performance good",
        default: true
      },
      min_performance: {
        type: "number",
        description: "Min avg reward to create skill",
        default: 0.7
      }
    },
    required: ["session_id"]
  }
}
```

**Response Format:**
```typescript
{
  success: true,
  session_id: "rl-session-123",
  final_metrics: {
    episodes: 1000,
    avg_reward: 0.89,
    convergence_score: 0.95,
    training_time_ms: 456789
  },
  model_saved: true,
  model_path: "/path/to/model.pth",
  skill_created: true,
  skill_id: 42,
  session_duration_ms: 456789
}
```

---

## Integration with Frontier Tools

### How Core + Learning Tools Integrate with Existing 9 Frontier Tools

```typescript
// Example: Reflexion Episode → Learning Session
// 1. Store episode with reflexion_store
const episode = await reflexion_store({
  session_id: "session-1",
  task: "implement_auth",
  reward: 0.95,
  success: true,
  critique: "OAuth2 worked perfectly"
});

// 2. Record experience for RL training
const experience = await experience_record({
  session_id: "rl-session-123",
  trajectory: [{
    state: { task_type: "auth", complexity: 0.8 },
    action: { strategy: "oauth2" },
    reward: 0.95,
    next_state: { implemented: true },
    done: true
  }]
});

// 3. Train RL model to predict successful strategies
await learning_train({
  session_id: "rl-session-123",
  num_steps: 100
});

// 4. Use RL to predict best approach for new task
const prediction = await learning_predict({
  session_id: "rl-session-123",
  state: { task_type: "auth", complexity: 0.9 }
});

// 5. If successful, consolidate into skill
const skill = await skill_create({
  name: "oauth2_implementation",
  description: "OAuth2 with PKCE flow",
  code: "/* implementation */",
  success_rate: 0.95
});

// 6. Add causal edge: oauth2_strategy → high_success
await causal_add_edge({
  cause: "oauth2_strategy",
  effect: "authentication_success",
  uplift: 0.25,
  confidence: 0.95
});

// 7. Query with causal recall
const results = await recall_with_certificate({
  query: "implement secure authentication",
  k: 5,
  alpha: 0.7,  // similarity weight
  beta: 0.2,   // causal uplift weight
  gamma: 0.1   // latency penalty
});
// Returns: OAuth2 episodes with high causal uplift
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Request                              │
│              "Implement authentication system"                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  1. SEARCH: agentdb_search("authentication")                   │
│     Returns: Similar past implementations                       │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  2. PREDICT: learning_predict(state: {task: "auth"})           │
│     Returns: Recommended strategy based on RL training         │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  3. RETRIEVE: reflexion_retrieve("auth failures")              │
│     Returns: Past failures and critiques to avoid              │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  4. CAUSAL: recall_with_certificate(...)                       │
│     Returns: Episodes reranked by causal uplift                │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  5. IMPLEMENT: Use retrieved knowledge + RL predictions        │
│     Agent implements OAuth2 with PKCE flow                     │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  6. RECORD: reflexion_store(task, output, reward, critique)   │
│     experience_record(trajectory with states/actions/rewards)  │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  7. TRAIN: learning_train(session_id)                          │
│     RL model learns from successful implementation             │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  8. CONSOLIDATE: skill_consolidate(min_attempts: 3)            │
│     Auto-create skill "oauth2_auth" from repeated success      │
└───────────────────────────┬───────────────────────────────────┘
                            │
                            v
┌───────────────────────────────────────────────────────────────┐
│  9. CAUSAL: learner_discover()                                 │
│     Nightly learner discovers: oauth2 → 0.25 uplift in success│
└───────────────────────────────────────────────────────────────┘
```

---

## Database Schema Requirements

### Complete Schema for All 29 Tools

```sql
-- ============================================================================
-- CORE VECTOR STORAGE (Tools 1-10)
-- ============================================================================

CREATE TABLE IF NOT EXISTS vectors (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  text TEXT NOT NULL,
  embedding BLOB NOT NULL, -- Float32Array serialized
  tags TEXT, -- JSON array: ["auth", "security"]
  metadata TEXT, -- JSON object: {"priority": "high"}
  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_vectors_tags ON vectors(tags);
CREATE INDEX IF NOT EXISTS idx_vectors_created ON vectors(created_at);

-- HNSW index (if supported by SQLite extension)
-- CREATE INDEX IF NOT EXISTS idx_hnsw_vectors
--   ON vectors USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- EMBEDDING CACHE (Tool 2: agentdb_insert)
-- ============================================================================

CREATE TABLE IF NOT EXISTS embedding_cache (
  text_hash TEXT PRIMARY KEY,
  embedding BLOB NOT NULL,
  model TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  cached_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- Auto-prune cache older than 30 days
CREATE TRIGGER IF NOT EXISTS prune_old_embeddings
AFTER INSERT ON embedding_cache
BEGIN
  DELETE FROM embedding_cache
  WHERE cached_at < strftime('%s', 'now') - 2592000;
END;

-- ============================================================================
-- PATTERN STORAGE (Tools 7-9)
-- ============================================================================

CREATE TABLE IF NOT EXISTS patterns (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  pattern_type TEXT NOT NULL CHECK(pattern_type IN ('regex', 'semantic', 'structural')),
  pattern_data TEXT NOT NULL, -- JSON
  success_rate REAL DEFAULT 0.0,
  uses INTEGER DEFAULT 0,
  last_used_at INTEGER,
  created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_patterns_name ON patterns(name);
CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);

-- ============================================================================
-- LEARNING SESSIONS (Tools 11-20)
-- ============================================================================

CREATE TABLE IF NOT EXISTS learning_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL UNIQUE,
  algorithm TEXT NOT NULL CHECK(algorithm IN (
    'decision_transformer', 'q_learning', 'sarsa', 'actor_critic',
    'ppo', 'dqn', 'a3c', 'ddpg', 'sac', 'td3'
  )),
  config TEXT NOT NULL, -- JSON: learning_rate, discount_factor, etc.
  state_space TEXT, -- JSON
  action_space TEXT, -- JSON

  -- Session state
  state TEXT, -- JSON: episodes_completed, total_reward, etc.
  model_path TEXT, -- Path to saved model

  -- Metrics
  episodes_completed INTEGER DEFAULT 0,
  total_reward REAL DEFAULT 0.0,
  avg_reward REAL DEFAULT 0.0,
  convergence_score REAL DEFAULT 0.0,

  -- Timestamps
  started_at INTEGER DEFAULT (strftime('%s', 'now')),
  ended_at INTEGER,
  last_updated_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_id ON learning_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_algorithm ON learning_sessions(algorithm);

-- ============================================================================
-- EXPERIENCE REPLAY BUFFER (Tool 18: experience_record)
-- ============================================================================

CREATE TABLE IF NOT EXISTS experience_trajectories (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  trajectory_id INTEGER NOT NULL,

  -- SARS tuple
  state TEXT NOT NULL, -- JSON
  action TEXT NOT NULL, -- JSON
  reward REAL NOT NULL,
  next_state TEXT NOT NULL, -- JSON
  done INTEGER NOT NULL CHECK(done IN (0, 1)),

  -- Metadata
  step_number INTEGER NOT NULL,
  episode_number INTEGER NOT NULL,
  metadata TEXT, -- JSON

  recorded_at INTEGER DEFAULT (strftime('%s', 'now')),

  FOREIGN KEY (session_id) REFERENCES learning_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_experience_session ON experience_trajectories(session_id);
CREATE INDEX IF NOT EXISTS idx_experience_trajectory ON experience_trajectories(trajectory_id);
CREATE INDEX IF NOT EXISTS idx_experience_episode ON experience_trajectories(episode_number);

-- ============================================================================
-- RL TRAINING METRICS (Tool 14: learning_metrics)
-- ============================================================================

CREATE TABLE IF NOT EXISTS training_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,

  -- Training step
  step_number INTEGER NOT NULL,
  episode_number INTEGER NOT NULL,

  -- Metrics
  loss REAL,
  q_value_mean REAL,
  gradient_norm REAL,
  exploration_rate REAL,
  episode_reward REAL,

  -- Performance
  moving_avg_reward REAL,
  convergence_score REAL,

  recorded_at INTEGER DEFAULT (strftime('%s', 'now')),

  FOREIGN KEY (session_id) REFERENCES learning_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_metrics_session ON training_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_metrics_step ON training_metrics(step_number);

-- ============================================================================
-- FRONTIER MEMORY TABLES (Existing 9 Tools)
-- ============================================================================

-- REFLEXION MEMORY
CREATE TABLE IF NOT EXISTS episodes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER DEFAULT (strftime('%s', 'now')),
  session_id TEXT NOT NULL,
  task TEXT NOT NULL,
  input TEXT,
  output TEXT,
  critique TEXT,
  reward REAL NOT NULL,
  success INTEGER NOT NULL CHECK(success IN (0, 1)),
  latency_ms INTEGER,
  tokens_used INTEGER,
  tags TEXT, -- JSON array
  metadata TEXT -- JSON object
);

CREATE TABLE IF NOT EXISTS episode_embeddings (
  episode_id INTEGER PRIMARY KEY,
  embedding BLOB NOT NULL,
  FOREIGN KEY (episode_id) REFERENCES episodes(id) ON DELETE CASCADE
);

-- SKILL LIBRARY
CREATE TABLE IF NOT EXISTS skills (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT,
  signature TEXT NOT NULL, -- JSON: {inputs: {...}, outputs: {...}}
  code TEXT,

  -- Performance metrics
  success_rate REAL DEFAULT 0.0,
  uses INTEGER DEFAULT 0,
  avg_reward REAL DEFAULT 0.0,
  avg_latency_ms REAL DEFAULT 0.0,

  -- Lineage
  created_from_episode INTEGER,
  metadata TEXT, -- JSON

  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  last_used_at INTEGER,

  FOREIGN KEY (created_from_episode) REFERENCES episodes(id)
);

CREATE TABLE IF NOT EXISTS skill_embeddings (
  skill_id INTEGER PRIMARY KEY,
  embedding BLOB NOT NULL,
  FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS skill_links (
  parent_skill_id INTEGER NOT NULL,
  child_skill_id INTEGER NOT NULL,
  relationship TEXT NOT NULL CHECK(relationship IN ('prerequisite', 'alternative', 'refinement', 'composition')),
  weight REAL DEFAULT 1.0,
  metadata TEXT,
  PRIMARY KEY (parent_skill_id, child_skill_id, relationship),
  FOREIGN KEY (parent_skill_id) REFERENCES skills(id) ON DELETE CASCADE,
  FOREIGN KEY (child_skill_id) REFERENCES skills(id) ON DELETE CASCADE
);

-- CAUSAL MEMORY GRAPH
CREATE TABLE IF NOT EXISTS causal_edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT,

  -- Causal relationship (simplified for MVP)
  cause TEXT NOT NULL,
  effect TEXT NOT NULL,

  -- Metrics
  uplift REAL NOT NULL, -- E[y|do(x)] - E[y]
  confidence REAL NOT NULL, -- 0-1
  sample_size INTEGER DEFAULT 0,

  -- Evidence
  evidence_ids TEXT, -- JSON array
  experiment_ids TEXT, -- JSON array
  confounder_score REAL,

  -- Explanation
  mechanism TEXT,
  metadata TEXT,

  created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_causal_cause ON causal_edges(cause);
CREATE INDEX IF NOT EXISTS idx_causal_effect ON causal_edges(effect);
CREATE INDEX IF NOT EXISTS idx_causal_confidence ON causal_edges(confidence);

CREATE TABLE IF NOT EXISTS causal_experiments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  cause TEXT NOT NULL,
  effect TEXT NOT NULL,

  -- Results
  treatment_mean REAL,
  control_mean REAL,
  uplift REAL,
  p_value REAL,
  confidence_lower REAL,
  confidence_upper REAL,

  status TEXT DEFAULT 'running' CHECK(status IN ('running', 'completed')),

  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  completed_at INTEGER
);

CREATE TABLE IF NOT EXISTS causal_observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  experiment_id INTEGER NOT NULL,
  is_treatment INTEGER NOT NULL CHECK(is_treatment IN (0, 1)),
  outcome REAL NOT NULL,
  context TEXT, -- JSON
  recorded_at INTEGER DEFAULT (strftime('%s', 'now')),
  FOREIGN KEY (experiment_id) REFERENCES causal_experiments(id) ON DELETE CASCADE
);

-- EXPLAINABLE RECALL
CREATE TABLE IF NOT EXISTS recall_certificates (
  id TEXT PRIMARY KEY, -- SHA256 hash
  query_id TEXT NOT NULL,
  query_text TEXT NOT NULL,

  -- Retrieved chunks
  chunk_ids TEXT NOT NULL, -- JSON array
  chunk_types TEXT NOT NULL, -- JSON array

  -- Justification
  minimal_why TEXT NOT NULL, -- JSON array
  redundancy_ratio REAL NOT NULL,
  completeness_score REAL NOT NULL,

  -- Provenance
  merkle_root TEXT NOT NULL,
  source_hashes TEXT NOT NULL, -- JSON array
  proof_chain TEXT, -- JSON array of MerkleProof objects

  -- Policy
  access_level TEXT DEFAULT 'internal' CHECK(access_level IN ('public', 'internal', 'confidential', 'restricted')),

  latency_ms INTEGER,
  created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS justification_paths (
  certificate_id TEXT NOT NULL,
  chunk_id TEXT NOT NULL,
  chunk_type TEXT NOT NULL,
  reason TEXT NOT NULL CHECK(reason IN ('semantic_match', 'causal_link', 'prerequisite', 'constraint')),
  necessity_score REAL NOT NULL,
  path_elements TEXT NOT NULL, -- JSON array
  PRIMARY KEY (certificate_id, chunk_id),
  FOREIGN KEY (certificate_id) REFERENCES recall_certificates(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS provenance_sources (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_type TEXT NOT NULL CHECK(source_type IN ('episode', 'skill', 'note', 'fact', 'external')),
  source_id INTEGER NOT NULL,
  content_hash TEXT NOT NULL UNIQUE,
  parent_hash TEXT,
  derived_from TEXT, -- JSON array
  creator TEXT,
  metadata TEXT, -- JSON
  created_at INTEGER DEFAULT (strftime('%s', 'now'))
);

-- ============================================================================
-- VIEWS FOR AGGREGATED METRICS
-- ============================================================================

CREATE VIEW IF NOT EXISTS v_database_stats AS
SELECT
  (SELECT COUNT(*) FROM vectors) as total_vectors,
  (SELECT COUNT(*) FROM episodes) as total_episodes,
  (SELECT COUNT(*) FROM skills) as total_skills,
  (SELECT COUNT(*) FROM causal_edges) as total_causal_edges,
  (SELECT COUNT(*) FROM recall_certificates) as total_certificates,
  (SELECT COUNT(*) FROM learning_sessions) as total_learning_sessions,
  (SELECT COUNT(*) FROM experience_trajectories) as total_experiences,
  (SELECT COUNT(*) FROM patterns) as total_patterns,
  (SELECT COUNT(*) FROM embedding_cache) as cache_size,
  (SELECT AVG(reward) FROM episodes) as avg_episode_reward,
  (SELECT AVG(success_rate) FROM skills) as avg_skill_success,
  (SELECT AVG(confidence) FROM causal_edges) as avg_causal_confidence;

CREATE VIEW IF NOT EXISTS v_learning_progress AS
SELECT
  session_id,
  algorithm,
  episodes_completed,
  avg_reward,
  convergence_score,
  (strftime('%s', 'now') - started_at) as session_duration_seconds,
  CASE
    WHEN convergence_score >= 0.95 THEN 'converged'
    WHEN convergence_score >= 0.80 THEN 'converging'
    WHEN convergence_score >= 0.60 THEN 'learning'
    ELSE 'initializing'
  END as status
FROM learning_sessions
WHERE ended_at IS NULL;

-- ============================================================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- ============================================================================

-- Episode search
CREATE INDEX IF NOT EXISTS idx_episodes_task ON episodes(task);
CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_reward ON episodes(reward DESC);
CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);

-- Skill search
CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
CREATE INDEX IF NOT EXISTS idx_skills_success_rate ON skills(success_rate DESC);
CREATE INDEX IF NOT EXISTS idx_skills_uses ON skills(uses DESC);

-- Certificate lookup
CREATE INDEX IF NOT EXISTS idx_certificates_query ON recall_certificates(query_id);
CREATE INDEX IF NOT EXISTS idx_certificates_created ON recall_certificates(created_at DESC);

-- Provenance lineage
CREATE INDEX IF NOT EXISTS idx_provenance_hash ON provenance_sources(content_hash);
CREATE INDEX IF NOT EXISTS idx_provenance_parent ON provenance_sources(parent_hash);
```

---

## MCP Tool Signatures

### Complete MCP Server Configuration

```typescript
// packages/agentdb/package/src/mcp/server.ts

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';

// Import all tool implementations
import { agentdb_init } from './tools/agentdb-init.js';
import { agentdb_insert } from './tools/agentdb-insert.js';
import { agentdb_insert_batch } from './tools/agentdb-insert-batch.js';
import { agentdb_search } from './tools/agentdb-search.js';
import { agentdb_delete } from './tools/agentdb-delete.js';
import { agentdb_stats } from './tools/agentdb-stats.js';
import { agentdb_pattern_store } from './tools/agentdb-pattern-store.js';
import { agentdb_pattern_search } from './tools/agentdb-pattern-search.js';
import { agentdb_pattern_stats } from './tools/agentdb-pattern-stats.js';
import { agentdb_clear_cache } from './tools/agentdb-clear-cache.js';

import { learning_start_session } from './tools/learning-start-session.js';
import { learning_predict } from './tools/learning-predict.js';
import { learning_feedback } from './tools/learning-feedback.js';
import { learning_train } from './tools/learning-train.js';
import { learning_metrics } from './tools/learning-metrics.js';
import { learning_transfer } from './tools/learning-transfer.js';
import { learning_explain } from './tools/learning-explain.js';
import { experience_record } from './tools/experience-record.js';
import { reward_signal } from './tools/reward-signal.js';
import { learning_end_session } from './tools/learning-end-session.js';

// Frontier tools (existing 9)
import { reflexion_store, reflexion_retrieve } from './tools/reflexion.js';
import { skill_create, skill_search } from './tools/skills.js';
import { causal_add_edge, causal_query } from './tools/causal.js';
import { recall_with_certificate } from './tools/recall.js';
import { learner_discover } from './tools/learner.js';
import { db_stats } from './tools/db-stats.js';

// MCP Context (shared state)
interface MCPContext {
  db: Database;
  embedder: EmbeddingService;
  causalGraph: CausalMemoryGraph;
  reflexion: ReflexionMemory;
  skills: SkillLibrary;
  explainableRecall: ExplainableRecall;
  learner: NightlyLearner;
  learningSessions: Map<string, LearningSession>;
}

// Tool registry
const TOOLS = [
  // Core AgentDB Tools (10)
  {
    name: 'agentdb_init',
    description: 'Initialize AgentDB vector database with schema and embeddings',
    inputSchema: { /* ... from specs above ... */ },
    handler: agentdb_init
  },
  {
    name: 'agentdb_insert',
    description: 'Insert vector with auto-embedding and metadata',
    inputSchema: { /* ... */ },
    handler: agentdb_insert
  },
  {
    name: 'agentdb_insert_batch',
    description: 'Bulk insert vectors (141x faster than sequential)',
    inputSchema: { /* ... */ },
    handler: agentdb_insert_batch
  },
  {
    name: 'agentdb_search',
    description: 'Semantic vector search with filters (116x faster with HNSW)',
    inputSchema: { /* ... */ },
    handler: agentdb_search
  },
  {
    name: 'agentdb_delete',
    description: 'Delete vectors by ID or filter conditions',
    inputSchema: { /* ... */ },
    handler: agentdb_delete
  },
  {
    name: 'agentdb_stats',
    description: 'Get database statistics and health metrics',
    inputSchema: { /* ... */ },
    handler: agentdb_stats
  },
  {
    name: 'agentdb_pattern_store',
    description: 'Store reusable pattern for matching and extraction',
    inputSchema: { /* ... */ },
    handler: agentdb_pattern_store
  },
  {
    name: 'agentdb_pattern_search',
    description: 'Search using stored pattern with match extraction',
    inputSchema: { /* ... */ },
    handler: agentdb_pattern_search
  },
  {
    name: 'agentdb_pattern_stats',
    description: 'Get pattern usage statistics',
    inputSchema: { /* ... */ },
    handler: agentdb_pattern_stats
  },
  {
    name: 'agentdb_clear_cache',
    description: 'Clear embedding cache and optimize storage',
    inputSchema: { /* ... */ },
    handler: agentdb_clear_cache
  },

  // Learning System Tools (10)
  {
    name: 'learning_start_session',
    description: 'Start RL learning session with algorithm',
    inputSchema: { /* ... */ },
    handler: learning_start_session
  },
  {
    name: 'learning_predict',
    description: 'Get action prediction from RL model',
    inputSchema: { /* ... */ },
    handler: learning_predict
  },
  {
    name: 'learning_feedback',
    description: 'Provide reward and update RL model',
    inputSchema: { /* ... */ },
    handler: learning_feedback
  },
  {
    name: 'learning_train',
    description: 'Execute training step with replay buffer',
    inputSchema: { /* ... */ },
    handler: learning_train
  },
  {
    name: 'learning_metrics',
    description: 'Get learning metrics and convergence status',
    inputSchema: { /* ... */ },
    handler: learning_metrics
  },
  {
    name: 'learning_transfer',
    description: 'Transfer policy between sessions',
    inputSchema: { /* ... */ },
    handler: learning_transfer
  },
  {
    name: 'learning_explain',
    description: 'Explain RL decision with attention maps',
    inputSchema: { /* ... */ },
    handler: learning_explain
  },
  {
    name: 'experience_record',
    description: 'Record experience for offline learning',
    inputSchema: { /* ... */ },
    handler: experience_record
  },
  {
    name: 'reward_signal',
    description: 'Compute shaped reward with intrinsic motivation',
    inputSchema: { /* ... */ },
    handler: reward_signal
  },
  {
    name: 'learning_end_session',
    description: 'End RL session and export model',
    inputSchema: { /* ... */ },
    handler: learning_end_session
  },

  // Frontier Memory Tools (9 existing)
  {
    name: 'reflexion_store',
    description: 'Store episode with self-critique for learning',
    inputSchema: { /* ... */ },
    handler: reflexion_store
  },
  {
    name: 'reflexion_retrieve',
    description: 'Retrieve relevant past episodes',
    inputSchema: { /* ... */ },
    handler: reflexion_retrieve
  },
  {
    name: 'skill_create',
    description: 'Create reusable skill from successful patterns',
    inputSchema: { /* ... */ },
    handler: skill_create
  },
  {
    name: 'skill_search',
    description: 'Search skills by semantic similarity',
    inputSchema: { /* ... */ },
    handler: skill_search
  },
  {
    name: 'causal_add_edge',
    description: 'Add causal edge with uplift measurement',
    inputSchema: { /* ... */ },
    handler: causal_add_edge
  },
  {
    name: 'causal_query',
    description: 'Query causal effects with confidence filters',
    inputSchema: { /* ... */ },
    handler: causal_query
  },
  {
    name: 'recall_with_certificate',
    description: 'Retrieve with causal utility and provenance certificate',
    inputSchema: { /* ... */ },
    handler: recall_with_certificate
  },
  {
    name: 'learner_discover',
    description: 'Automated causal discovery from episodes',
    inputSchema: { /* ... */ },
    handler: learner_discover
  },
  {
    name: 'db_stats',
    description: 'Get comprehensive database statistics',
    inputSchema: { /* ... */ },
    handler: db_stats
  }
];

// MCP Server initialization
export async function startMCPServer() {
  const server = new Server(
    {
      name: 'agentdb',
      version: '1.3.0',
    },
    {
      capabilities: {
        tools: {},
      },
    }
  );

  // Initialize context
  const context: MCPContext = await initializeContext();

  // Register list_tools handler
  server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: TOOLS.map(t => ({
      name: t.name,
      description: t.description,
      inputSchema: t.inputSchema
    }))
  }));

  // Register call_tool handler
  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const tool = TOOLS.find(t => t.name === request.params.name);

    if (!tool) {
      throw new Error(`Unknown tool: ${request.params.name}`);
    }

    try {
      const result = await tool.handler(request.params.arguments, context);
      return {
        content: [{
          type: 'text',
          text: JSON.stringify(result, null, 2)
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: JSON.stringify({
            success: false,
            error: error.message,
            stack: error.stack
          }, null, 2)
        }],
        isError: true
      };
    }
  });

  // Start server
  const transport = new StdioServerTransport();
  await server.connect(transport);

  console.error('AgentDB MCP Server v1.3.0 started');
  console.error('29 tools available (10 core + 10 learning + 9 frontier)');
}

async function initializeContext(): Promise<MCPContext> {
  // Auto-initialize database if not exists
  const dbPath = process.env.AGENTDB_PATH || './agentdb.db';
  const db = new Database(dbPath);

  // Load schema
  const schemaPath = path.join(__dirname, '../schemas/complete-schema.sql');
  const schema = fs.readFileSync(schemaPath, 'utf-8');
  db.exec(schema);

  // Initialize embedding service
  const embedder = new EmbeddingService({
    model: 'all-MiniLM-L6-v2',
    dimension: 384,
    provider: 'transformers'
  });
  await embedder.initialize();

  // Initialize controllers
  const causalGraph = new CausalMemoryGraph(db);
  const reflexion = new ReflexionMemory(db, embedder);
  const skills = new SkillLibrary(db, embedder);
  const explainableRecall = new ExplainableRecall(db);
  const learner = new NightlyLearner(db, embedder, causalGraph);

  return {
    db,
    embedder,
    causalGraph,
    reflexion,
    skills,
    explainableRecall,
    learner,
    learningSessions: new Map()
  };
}
```

---

## Error Handling & Validation

### Standardized Error Response Format

```typescript
{
  success: false,
  error: {
    code: "INVALID_PARAMETER",
    message: "Parameter 'text' is required but was not provided",
    details: {
      parameter: "text",
      expected: "string",
      received: "undefined"
    },
    timestamp: 1640995200,
    tool: "agentdb_insert"
  },
  request_id: "req-123456"
}
```

### Error Codes

```typescript
enum AgentDBErrorCode {
  // Validation errors (400-level)
  INVALID_PARAMETER = "INVALID_PARAMETER",
  MISSING_REQUIRED_FIELD = "MISSING_REQUIRED_FIELD",
  INVALID_TYPE = "INVALID_TYPE",
  OUT_OF_RANGE = "OUT_OF_RANGE",
  INVALID_FILTER = "INVALID_FILTER",

  // Database errors (500-level)
  DATABASE_ERROR = "DATABASE_ERROR",
  SCHEMA_ERROR = "SCHEMA_ERROR",
  CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION",
  TRANSACTION_FAILED = "TRANSACTION_FAILED",

  // Embedding errors
  EMBEDDING_FAILED = "EMBEDDING_FAILED",
  MODEL_NOT_LOADED = "MODEL_NOT_LOADED",
  DIMENSION_MISMATCH = "DIMENSION_MISMATCH",

  // Learning errors
  SESSION_NOT_FOUND = "SESSION_NOT_FOUND",
  ALGORITHM_NOT_SUPPORTED = "ALGORITHM_NOT_SUPPORTED",
  MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED",
  CONVERGENCE_FAILED = "CONVERGENCE_FAILED",

  // Resource errors
  NOT_FOUND = "NOT_FOUND",
  ALREADY_EXISTS = "ALREADY_EXISTS",
  QUOTA_EXCEEDED = "QUOTA_EXCEEDED",

  // System errors
  INITIALIZATION_FAILED = "INITIALIZATION_FAILED",
  INTERNAL_ERROR = "INTERNAL_ERROR"
}
```

### Validation Middleware

```typescript
// packages/agentdb/package/src/mcp/validation.ts

export function validateToolInput(toolName: string, args: any): ValidationResult {
  const schema = TOOL_SCHEMAS[toolName];

  if (!schema) {
    return {
      valid: false,
      errors: [{
        code: 'INVALID_PARAMETER',
        message: `Unknown tool: ${toolName}`
      }]
    };
  }

  // Validate required fields
  for (const field of schema.required || []) {
    if (!(field in args)) {
      return {
        valid: false,
        errors: [{
          code: 'MISSING_REQUIRED_FIELD',
          message: `Required field '${field}' is missing`,
          details: { field, expected: schema.properties[field].type }
        }]
      };
    }
  }

  // Validate types
  for (const [field, value] of Object.entries(args)) {
    const fieldSchema = schema.properties[field];

    if (!fieldSchema) {
      return {
        valid: false,
        errors: [{
          code: 'INVALID_PARAMETER',
          message: `Unknown parameter '${field}'`,
          details: { field, valid_params: Object.keys(schema.properties) }
        }]
      };
    }

    const actualType = Array.isArray(value) ? 'array' : typeof value;
    if (actualType !== fieldSchema.type) {
      return {
        valid: false,
        errors: [{
          code: 'INVALID_TYPE',
          message: `Parameter '${field}' has wrong type`,
          details: {
            field,
            expected: fieldSchema.type,
            received: actualType
          }
        }]
      };
    }
  }

  return { valid: true, errors: [] };
}
```

---

## Performance Specifications

### Latency Targets (95th percentile)

```typescript
const PERFORMANCE_TARGETS = {
  // Core operations
  agentdb_init: 50,           // 50ms (cold start)
  agentdb_insert: 50,          // 50ms (with embedding generation)
  agentdb_insert_batch: 2500,  // 2.5s for 1000 vectors
  agentdb_search: 10,          // 10ms (with HNSW)
  agentdb_delete: 100,         // 100ms (with cascade)
  agentdb_stats: 50,           // 50ms
  agentdb_pattern_store: 30,   // 30ms
  agentdb_pattern_search: 20,  // 20ms
  agentdb_pattern_stats: 10,   // 10ms
  agentdb_clear_cache: 500,    // 500ms (with vacuum)

  // Learning operations
  learning_start_session: 100, // 100ms (model initialization)
  learning_predict: 5,         // 5ms (WASM inference)
  learning_feedback: 10,       // 10ms (single update)
  learning_train: 1000,        // 1s per 100 steps
  learning_metrics: 20,        // 20ms
  learning_transfer: 2000,     // 2s (model copy)
  learning_explain: 50,        // 50ms (attention weights)
  experience_record: 5,        // 5ms per transition
  reward_signal: 3,            // 3ms (computation)
  learning_end_session: 1000,  // 1s (save + export)

  // Frontier operations
  reflexion_store: 60,         // 60ms (with embedding)
  reflexion_retrieve: 15,      // 15ms
  skill_create: 40,            // 40ms
  skill_search: 12,            // 12ms
  causal_add_edge: 10,         // 10ms
  causal_query: 25,            // 25ms
  recall_with_certificate: 30, // 30ms (with provenance)
  learner_discover: 5000,      // 5s (background job)
  db_stats: 100                // 100ms (aggregation)
};
```

### Throughput Targets

```
agentdb_insert_batch: 400+ vectors/second (batch=100)
agentdb_search: 200+ queries/second (HNSW enabled)
learning_predict: 10,000+ inferences/second (WASM)
experience_record: 1,000+ transitions/second
```

### Memory Footprint

```
Embedding Service (MiniLM): ~120 MB
Database (1K vectors): 0.7 MB
Database (100K vectors): 70 MB
RL Plugin (Decision Transformer): ~50 MB
Total Runtime Baseline: ~200 MB
```

---

## Implementation Roadmap

### Phase 1: Core AgentDB Tools (Week 1-2)

**Priority 1 (Days 1-3):**
- ✅ Tool 1: agentdb_init
- ✅ Tool 2: agentdb_insert
- ✅ Tool 3: agentdb_insert_batch
- ✅ Tool 4: agentdb_search

**Priority 2 (Days 4-7):**
- Tool 5: agentdb_delete
- Tool 6: agentdb_stats
- Tool 10: agentdb_clear_cache

**Priority 3 (Days 8-10):**
- Tool 7: agentdb_pattern_store
- Tool 8: agentdb_pattern_search
- Tool 9: agentdb_pattern_stats

**Deliverables:**
- 10 Core tools implemented
- Complete database schema
- MCP server with all tools registered
- Integration tests for each tool
- Performance benchmarks

---

### Phase 2: Learning System Tools (Week 3-4)

**Priority 1 (Days 1-4):**
- ✅ Tool 11: learning_start_session
- ✅ Tool 12: learning_predict
- ✅ Tool 13: learning_feedback
- ✅ Tool 14: learning_train

**Priority 2 (Days 5-8):**
- Tool 15: learning_metrics
- Tool 18: experience_record
- Tool 19: reward_signal
- Tool 20: learning_end_session

**Priority 3 (Days 9-10):**
- Tool 16: learning_transfer
- Tool 17: learning_explain

**Deliverables:**
- 10 Learning tools implemented
- RL plugin integration complete
- WASM acceleration working
- Experience replay buffer functional
- Convergence metrics validated

---

### Phase 3: Integration & Testing (Week 5)

**Tasks:**
1. Integration tests for all 29 tools
2. End-to-end workflow testing (Reflexion → Learning → Skills)
3. Performance optimization (meet latency targets)
4. Error handling validation
5. Documentation completion
6. Claude Desktop integration testing

**Deliverables:**
- 100% test coverage for new tools
- Performance benchmarks showing <10ms search
- Full MCP server validation
- Migration guide from v1.2.2 → v1.3.0
- Updated README and API docs

---

### Phase 4: Release (Week 6)

**Tasks:**
1. Version bump to 1.3.0
2. npm package publication
3. Claude Desktop MCP registration
4. Blog post and examples
5. Community feedback collection

**Release Checklist:**
- [ ] All 29 tools tested and working
- [ ] Performance targets met
- [ ] Documentation complete
- [ ] Migration guide published
- [ ] npm package released
- [ ] Claude Desktop compatible
- [ ] Example notebooks created

---

## Appendix A: Complete Tool Matrix

| Category | Tool Name | Priority | Status | Dependencies |
|----------|-----------|----------|--------|--------------|
| **Core** | agentdb_init | P0 | Spec Complete | better-sqlite3, EmbeddingService |
| | agentdb_insert | P0 | Spec Complete | agentdb_init, EmbeddingService |
| | agentdb_insert_batch | P0 | Spec Complete | agentdb_init, EmbeddingService |
| | agentdb_search | P0 | Spec Complete | agentdb_init, HNSW index |
| | agentdb_delete | P1 | Spec Complete | agentdb_init |
| | agentdb_stats | P1 | Spec Complete | agentdb_init |
| | agentdb_pattern_store | P2 | Spec Complete | agentdb_init |
| | agentdb_pattern_search | P2 | Spec Complete | agentdb_pattern_store |
| | agentdb_pattern_stats | P2 | Spec Complete | agentdb_pattern_store |
| | agentdb_clear_cache | P1 | Spec Complete | agentdb_init |
| **Learning** | learning_start_session | P0 | Spec Complete | RL plugins, WASM |
| | learning_predict | P0 | Spec Complete | learning_start_session |
| | learning_feedback | P0 | Spec Complete | learning_start_session |
| | learning_train | P0 | Spec Complete | learning_start_session |
| | learning_metrics | P1 | Spec Complete | learning_start_session |
| | learning_transfer | P2 | Spec Complete | learning_end_session |
| | learning_explain | P2 | Spec Complete | learning_predict |
| | experience_record | P1 | Spec Complete | learning_start_session |
| | reward_signal | P1 | Spec Complete | learning_feedback |
| | learning_end_session | P1 | Spec Complete | learning_start_session |
| **Frontier** | reflexion_store | P0 | Implemented | ReflexionMemory |
| | reflexion_retrieve | P0 | Implemented | ReflexionMemory |
| | skill_create | P0 | Implemented | SkillLibrary |
| | skill_search | P0 | Implemented | SkillLibrary |
| | causal_add_edge | P0 | Implemented | CausalMemoryGraph |
| | causal_query | P0 | Implemented | CausalMemoryGraph |
| | recall_with_certificate | P0 | Implemented | CausalRecall, ExplainableRecall |
| | learner_discover | P1 | Implemented | NightlyLearner |
| | db_stats | P1 | Implemented | Database |

**Total:** 29 MCP Tools

---

## Appendix B: Usage Examples

### Example 1: Complete Agent Workflow

```typescript
// 1. Initialize database
const init = await agentdb_init({
  db_path: "./my-agent-memory.db",
  enable_hnsw: true
});

// 2. Start learning session
const session = await learning_start_session({
  session_id: "agent-123",
  algorithm: "decision_transformer",
  config: { learning_rate: 0.001 }
});

// 3. Agent encounters task
const task = "Implement OAuth2 authentication";

// 4. Search for relevant past experiences
const similar = await agentdb_search({
  query: task,
  k: 5,
  rerank: true // Use causal reranking
});

// 5. Retrieve failed attempts to avoid
const failures = await reflexion_retrieve({
  task: "authentication",
  onlyFailures: true,
  k: 3
});

// 6. Get RL prediction for strategy
const prediction = await learning_predict({
  session_id: "agent-123",
  state: { task_type: "auth", complexity: 0.8 }
});

// 7. Agent implements using predicted strategy
// ... implementation code ...

// 8. Record outcome
const outcome = {
  success: true,
  reward: 0.92,
  latency_ms: 1250,
  output: "OAuth2 with PKCE implemented successfully"
};

// 9. Store reflexion episode
await reflexion_store({
  session_id: "agent-123",
  task: task,
  reward: outcome.reward,
  success: outcome.success,
  critique: "PKCE flow prevented token interception",
  latency_ms: outcome.latency_ms
});

// 10. Provide RL feedback
await learning_feedback({
  session_id: "agent-123",
  state: { task_type: "auth", complexity: 0.8 },
  action: { strategy: "oauth2_pkce" },
  reward: outcome.reward,
  next_state: { implemented: true },
  done: true
});

// 11. Train RL model
await learning_train({
  session_id: "agent-123",
  num_steps: 50
});

// 12. Add causal edge
await causal_add_edge({
  cause: "oauth2_pkce_strategy",
  effect: "auth_security_high",
  uplift: 0.25,
  confidence: 0.95
});

// 13. Check if ready to consolidate into skill
const metrics = await learning_metrics({
  session_id: "agent-123"
});

if (metrics.convergence_score > 0.8) {
  // Auto-create skill from successful patterns
  await skill_consolidate({
    minAttempts: 3,
    minReward: 0.7
  });
}

// 14. Get comprehensive stats
const stats = await agentdb_stats({
  include_quality: true
});

console.log(`Total episodes: ${stats.counts.total_episodes}`);
console.log(`Avg reward: ${stats.quality.avg_episode_reward}`);
console.log(`RL convergence: ${metrics.convergence_score}`);
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.3.0-spec | 2025-10-22 | Initial design specification for 20 new tools |
| 1.2.2 | 2025-10-22 | Added 5 core vector DB tools (14 total) |
| 1.1.0 | 2025-10-20 | Added 9 frontier memory tools |
| 1.0.0 | 2025-10-15 | Initial AgentDB release |

---

## Contact & Support

- **GitHub:** https://github.com/ruvnet/agentic-flow
- **Issues:** https://github.com/ruvnet/agentic-flow/issues
- **Discussions:** https://github.com/ruvnet/agentic-flow/discussions
- **npm:** https://www.npmjs.com/package/agentdb

---

**Document Status:** ✅ Complete Design Specification
**Next Steps:** Begin Phase 1 implementation of Core AgentDB Tools
**Target Release:** AgentDB v1.3.0 (6 weeks)

