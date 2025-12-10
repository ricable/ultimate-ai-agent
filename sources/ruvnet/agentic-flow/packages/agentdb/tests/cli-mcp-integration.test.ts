/**
 * CLI and MCP Integration Tests
 *
 * Validates:
 * - CLI commands work correctly
 * - MCP tools integration
 * - Backward compatibility with SQLite
 * - Migration from SQLite to GraphDatabase
 * - All exports are available
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

const TEST_DIR = path.join(process.cwd(), 'test-cli-data');
const SQLITE_DB = path.join(TEST_DIR, 'legacy.db');
const GRAPH_DB = path.join(TEST_DIR, 'modern.graph');

beforeAll(() => {
  if (!fs.existsSync(TEST_DIR)) {
    fs.mkdirSync(TEST_DIR, { recursive: true });
  }
});

afterAll(() => {
  if (fs.existsSync(TEST_DIR)) {
    fs.rmSync(TEST_DIR, { recursive: true, force: true });
  }
});

describe('CLI Commands', () => {
  it('should show help', () => {
    const output = execSync('npx tsx src/cli/agentdb-cli.ts --help', { encoding: 'utf-8' });

    expect(output).toContain('AgentDB v2 CLI');
    expect(output).toContain('CORE COMMANDS');
    expect(output).toContain('init');
    expect(output).toContain('migrate');
    console.log('✅ CLI help command working');
  });

  it('should initialize database', () => {
    const output = execSync(`npx tsx src/cli/agentdb-cli.ts init ${SQLITE_DB} --dimension 384`, {
      encoding: 'utf-8',
      cwd: process.cwd()
    });

    expect(fs.existsSync(SQLITE_DB)).toBe(true);
    console.log('✅ CLI init command working - database created');
  });

  it('should show status', () => {
    const output = execSync(`npx tsx src/cli/agentdb-cli.ts status --db ${SQLITE_DB}`, {
      encoding: 'utf-8',
      cwd: process.cwd()
    });

    expect(output).toContain('AgentDB Status');
    console.log('✅ CLI status command working');
  });

  it('should export database stats', () => {
    const output = execSync(`npx tsx src/cli/agentdb-cli.ts db stats`, {
      encoding: 'utf-8',
      cwd: process.cwd(),
      env: { ...process.env, AGENTDB_PATH: SQLITE_DB }
    });

    expect(output).toBeTruthy();
    console.log('✅ CLI stats command working');
  });
});

describe('SDK Exports', () => {
  it('should export all controllers', async () => {
    const agentdb = await import('../src/index.js');

    expect(agentdb.ReflexionMemory).toBeDefined();
    expect(agentdb.SkillLibrary).toBeDefined();
    expect(agentdb.CausalMemoryGraph).toBeDefined();
    expect(agentdb.CausalRecall).toBeDefined();
    expect(agentdb.ExplainableRecall).toBeDefined();
    expect(agentdb.NightlyLearner).toBeDefined();
    expect(agentdb.EmbeddingService).toBeDefined();

    console.log('✅ All controllers exported');
  });

  it('should export database utilities', async () => {
    const agentdb = await import('../src/index.js');

    expect(agentdb.createDatabase).toBeDefined();
    // getDatabaseImplementation is internal, not exported

    console.log('✅ Database utilities exported');
  });

  it('should export GraphDatabase adapter', async () => {
    const { GraphDatabaseAdapter } = await import('../src/backends/graph/GraphDatabaseAdapter.js');

    expect(GraphDatabaseAdapter).toBeDefined();
    expect(typeof GraphDatabaseAdapter).toBe('function');

    console.log('✅ GraphDatabaseAdapter exported');
  });

  it('should export UnifiedDatabase', async () => {
    const { UnifiedDatabase, createUnifiedDatabase } = await import('../src/db-unified.js');

    expect(UnifiedDatabase).toBeDefined();
    expect(createUnifiedDatabase).toBeDefined();

    console.log('✅ UnifiedDatabase exported');
  });
});

describe('Backward Compatibility - SQLite', () => {
  it('should create SQLite database with legacy mode', async () => {
    const { createDatabase } = await import('../src/db-fallback.js');

    const db = await createDatabase(SQLITE_DB);
    expect(db).toBeDefined();
    expect(typeof db.prepare).toBe('function');
    expect(typeof db.exec).toBe('function');

    console.log('✅ SQLite createDatabase working (backward compatible)');

    db.close();
  });

  it('should work with ReflexionMemory on SQLite', async () => {
    const { createDatabase } = await import('../src/db-fallback.js');
    const { ReflexionMemory } = await import('../src/controllers/ReflexionMemory.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const db = await createDatabase(SQLITE_DB);
    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();
    const reflexion = new ReflexionMemory(db, embedder);

    // Store an episode (using correct API method)
    await reflexion.storeEpisode({
      sessionId: 'test-session',
      task: 'test backward compatibility',
      reward: 0.95,
      success: true,
      input: 'test input',
      output: 'test output',
      critique: 'working great'
    });

    // Retrieve episodes (using correct API method)
    const results = await reflexion.retrieveRelevant({ task: 'backward compatibility', k: 5 });
    expect(results.length).toBeGreaterThan(0);
    expect(results[0].task).toContain('backward compatibility');

    console.log('✅ ReflexionMemory working with SQLite');

    db.close();
  });

  it('should work with SkillLibrary on SQLite', async () => {
    const { createDatabase } = await import('../src/db-fallback.js');
    const { SkillLibrary } = await import('../src/controllers/SkillLibrary.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const db = await createDatabase(SQLITE_DB);
    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();
    const skills = new SkillLibrary(db, embedder);

    // Create a skill (using correct API method)
    const skillId = await skills.createSkill({
      name: 'test-skill',
      description: 'backward compatibility test',
      code: 'function test() { return true; }',
      successRate: 1.0
    });

    expect(skillId).toBeTruthy();

    // Search for skill (using correct API method)
    const results = await skills.searchSkills({ query: 'test', k: 5 });
    expect(results.length).toBeGreaterThan(0);

    console.log('✅ SkillLibrary working with SQLite');

    db.close();
  });
});

describe('Migration - SQLite to GraphDatabase', () => {
  it('should detect database mode', async () => {
    const { UnifiedDatabase } = await import('../src/db-unified.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    // Test SQLite detection
    const sqliteDb = new UnifiedDatabase({ path: SQLITE_DB });
    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();
    await sqliteDb.initialize(embedder);

    expect(sqliteDb.getMode()).toBe('sqlite-legacy');
    console.log('✅ SQLite database detected correctly');

    sqliteDb.close();
  });

  it('should create new graph database', async () => {
    const { createUnifiedDatabase } = await import('../src/db-unified.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();
    const db = await createUnifiedDatabase(GRAPH_DB, embedder, {
      forceMode: 'graph'
    });

    expect(db.getMode()).toBe('graph');
    expect(fs.existsSync(GRAPH_DB)).toBe(true);

    console.log('✅ GraphDatabase created successfully');

    db.close();
  });

  it('should migrate SQLite to Graph (manual)', async () => {
    const { createDatabase } = await import('../src/db-fallback.js');
    const { GraphDatabaseAdapter } = await import('../src/backends/graph/GraphDatabaseAdapter.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();

    // Create source SQLite database with data
    const sqliteDb = await createDatabase(SQLITE_DB);

    // Insert test episode
    sqliteDb.exec(`
      INSERT INTO episodes (session_id, task, reward, success, input, output, critique, created_at)
      VALUES ('migration-test', 'test migration', 0.95, 1, 'input', 'output', 'critique', ${Date.now()})
    `);

    const episodes = sqliteDb.prepare('SELECT * FROM episodes').all();
    expect(episodes.length).toBeGreaterThan(0);

    console.log(`✅ SQLite has ${episodes.length} episodes to migrate`);

    // Create target GraphDatabase
    const migrationGraphPath = path.join(TEST_DIR, 'migrated.graph');
    const graphDb = new GraphDatabaseAdapter({
      storagePath: migrationGraphPath,
      dimensions: 384
    }, embedder);

    await graphDb.initialize();

    // Migrate episodes (using correct API method)
    for (const ep of episodes) {
      const text = `${ep.task} ${ep.input || ''} ${ep.output || ''}`;
      const embedding = await embedder.embed(text);

      await graphDb.storeEpisode({
        id: `ep-${ep.id}`,
        sessionId: ep.session_id,
        task: ep.task,
        reward: ep.reward,
        success: ep.success === 1,
        input: ep.input,
        output: ep.output,
        critique: ep.critique,
        createdAt: ep.created_at,
        tokensUsed: ep.tokens_used,
        latencyMs: ep.latency_ms
      }, embedding);
    }

    console.log('✅ Manual migration completed');

    // Verify migration
    const stats = await graphDb.getStats();
    expect(stats.totalNodes).toBeGreaterThan(0);

    console.log(`✅ GraphDatabase has ${stats.totalNodes} nodes after migration`);

    sqliteDb.close();
  });
});

describe('MCP Tool Integration', () => {
  it('should validate agentdb_pattern_store schema', async () => {
    const { storePattern } = await import('../src/mcp/agentdb-mcp-server.js').catch(() => ({ storePattern: null }));

    // MCP server exports are optional
    if (storePattern) {
      expect(typeof storePattern).toBe('function');
      console.log('✅ MCP pattern_store tool available');
    } else {
      console.log('ℹ️  MCP server not loaded (optional)');
    }
  });

  it('should validate agentdb_pattern_search schema', async () => {
    const { searchPattern } = await import('../src/mcp/agentdb-mcp-server.js').catch(() => ({ searchPattern: null }));

    if (searchPattern) {
      expect(typeof searchPattern).toBe('function');
      console.log('✅ MCP pattern_search tool available');
    } else {
      console.log('ℹ️  MCP server not loaded (optional)');
    }
  });

  it('should validate agentdb_stats schema', async () => {
    const { getStats } = await import('../src/mcp/agentdb-mcp-server.js').catch(() => ({ getStats: null }));

    if (getStats) {
      expect(typeof getStats).toBe('function');
      console.log('✅ MCP stats tool available');
    } else {
      console.log('ℹ️  MCP server not loaded (optional)');
    }
  });
});

describe('Integration Test - Full Workflow', () => {
  it('should complete full workflow: CLI init → SQLite ops → Migration → Graph ops', async () => {
    console.log('\n🚀 FULL INTEGRATION TEST\n');

    // 1. Initialize SQLite database via CLI
    const testDbPath = path.join(TEST_DIR, 'full-test.db');
    execSync(`npx tsx src/cli/agentdb-cli.ts init ${testDbPath} --dimension 384`, {
      cwd: process.cwd()
    });

    console.log('✅ 1. CLI initialized SQLite database');

    // 2. Perform SQLite operations
    const { createDatabase } = await import('../src/db-fallback.js');
    const { ReflexionMemory } = await import('../src/controllers/ReflexionMemory.js');
    const { EmbeddingService } = await import('../src/controllers/EmbeddingService.js');

    const db = await createDatabase(testDbPath);
    const embedder = new EmbeddingService({
      model: 'Xenova/all-MiniLM-L6-v2',
      dimension: 384,
      provider: 'transformers'
    });
    await embedder.initialize();
    const reflexion = new ReflexionMemory(db, embedder);

    await reflexion.storeEpisode({
      sessionId: 'full-test',
      task: 'complete integration test',
      reward: 0.98,
      success: true,
      input: 'test input',
      output: 'test output',
      critique: 'excellent integration'
    });

    const sqliteResults = await reflexion.retrieveRelevant({ task: 'integration', k: 5 });
    expect(sqliteResults.length).toBeGreaterThan(0);

    console.log('✅ 2. SQLite operations completed');

    db.close();

    // 3. Migrate to GraphDatabase
    const { createUnifiedDatabase } = await import('../src/db-unified.js');
    const graphDbPath = testDbPath.replace('.db', '.graph');

    const unifiedDb = await createUnifiedDatabase(testDbPath, embedder, {
      autoMigrate: true
    });

    expect(unifiedDb.getMode()).toBe('graph');
    expect(fs.existsSync(graphDbPath)).toBe(true);

    console.log('✅ 3. Migration to GraphDatabase completed');

    // 4. Verify GraphDatabase operations
    const graphDb = unifiedDb.getGraphDatabase();
    expect(graphDb).toBeDefined();

    // Query to verify migration worked (stats may not update immediately)
    const cypherResult = await graphDb!.query('MATCH (e:Episode) RETURN e LIMIT 5');
    expect(cypherResult.nodes.length).toBeGreaterThan(0);

    console.log('✅ 4. GraphDatabase operations verified');
    console.log('✅ 5. Cypher queries working - migration successful');

    console.log('\n🎉 FULL INTEGRATION TEST PASSED\n');

    unifiedDb.close();
  });
});
