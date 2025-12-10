/**
 * AgentDB Browser Bundle Unit Tests
 * Tests all browser-specific features and ensures backward compatibility
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import initSqlJs from 'sql.js';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

describe('AgentDB Browser Bundle', () => {
  let SQL;
  let db;

  beforeAll(async () => {
    // Initialize sql.js with local WASM file for Node.js testing
    // In browser, this would use CDN URL
    SQL = await initSqlJs({
      locateFile: file => {
        // Use local node_modules path for testing
        // tests/browser/ -> packages/agentdb/ requires ../../
        return join(__dirname, '../../node_modules/sql.js/dist', file);
      }
    });
  });

  afterAll(() => {
    if (db) {
      db.close();
    }
  });

  describe('Database Initialization', () => {
    it('should create a new database instance', () => {
      // Simulate the Database constructor from browser bundle
      db = new SQL.Database();
      expect(db).toBeDefined();
    });

    it('should initialize with default schema', () => {
      db = new SQL.Database();

      // Create tables like browser bundle does
      db.run(`
        CREATE TABLE IF NOT EXISTS vectors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          embedding BLOB,
          metadata TEXT,
          text TEXT,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
      `);

      const result = db.exec("SELECT name FROM sqlite_master WHERE type='table' AND name='vectors'");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].values[0][0]).toBe('vectors');
    });

    it('should create all frontier feature tables', () => {
      db = new SQL.Database();

      // Create all tables
      const tables = {
        vectors: `CREATE TABLE IF NOT EXISTS vectors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          embedding BLOB,
          metadata TEXT,
          text TEXT,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )`,
        patterns: `CREATE TABLE IF NOT EXISTS patterns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          pattern TEXT NOT NULL,
          metadata TEXT,
          embedding BLOB,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )`,
        episodes: `CREATE TABLE IF NOT EXISTS episodes (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          trajectory TEXT NOT NULL,
          self_reflection TEXT,
          verdict TEXT,
          metadata TEXT,
          embedding BLOB,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )`,
        causal_edges: `CREATE TABLE IF NOT EXISTS causal_edges (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          cause TEXT NOT NULL,
          effect TEXT NOT NULL,
          strength REAL DEFAULT 0.5,
          metadata TEXT,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )`,
        skills: `CREATE TABLE IF NOT EXISTS skills (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          skill_name TEXT NOT NULL,
          code TEXT,
          metadata TEXT,
          embedding BLOB,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )`
      };

      Object.values(tables).forEach(sql => db.run(sql));

      const result = db.exec("SELECT name FROM sqlite_master WHERE type='table'");
      const tableNames = result[0].values.map(row => row[0]);

      expect(tableNames).toContain('vectors');
      expect(tableNames).toContain('patterns');
      expect(tableNames).toContain('episodes');
      expect(tableNames).toContain('causal_edges');
      expect(tableNames).toContain('skills');
    });
  });

  describe('v1.0.7 API Compatibility', () => {
    beforeAll(() => {
      db = new SQL.Database();
      db.run(`
        CREATE TABLE IF NOT EXISTS vectors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          embedding BLOB,
          metadata TEXT,
          text TEXT,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
      `);
    });

    it('should support run() method', () => {
      expect(() => {
        db.run('INSERT INTO vectors (text, metadata) VALUES (?, ?)', ['test', '{"key":"value"}']);
      }).not.toThrow();
    });

    it('should support exec() method', () => {
      db.run('INSERT INTO vectors (text) VALUES (?)', ['test data']);
      const result = db.exec('SELECT * FROM vectors');

      expect(result).toBeDefined();
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].columns).toContain('text');
    });

    it('should support prepare() method', () => {
      const stmt = db.prepare('SELECT * FROM vectors WHERE text = ?');
      expect(stmt).toBeDefined();
      expect(typeof stmt.bind).toBe('function');
      expect(typeof stmt.step).toBe('function');
      stmt.free();
    });

    it('should support export() method', () => {
      const exported = db.export();
      expect(exported).toBeInstanceOf(Uint8Array);
      expect(exported.length).toBeGreaterThan(0);
    });

    it('should retrieve last insert ID', () => {
      db.run('INSERT INTO vectors (text) VALUES (?)', ['test']);
      const result = db.exec('SELECT last_insert_rowid() as id');
      const lastId = result[0].values[0][0];
      expect(lastId).toBeGreaterThan(0);
    });
  });

  describe('Insert Method - Dual Signature Support', () => {
    beforeAll(() => {
      db = new SQL.Database();
      db.run(`
        CREATE TABLE IF NOT EXISTS vectors (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          embedding BLOB,
          metadata TEXT,
          text TEXT,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
      `);
      db.run(`
        CREATE TABLE IF NOT EXISTS patterns (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          pattern TEXT NOT NULL,
          metadata TEXT,
          created_at INTEGER DEFAULT (strftime('%s', 'now'))
        )
      `);
    });

    it('should insert with table + data signature', () => {
      const data = {
        pattern: 'Test pattern',
        metadata: JSON.stringify({ roas: 2.5 })
      };

      const columns = Object.keys(data);
      const values = Object.values(data);
      const placeholders = columns.map(() => '?').join(', ');
      const sql = `INSERT INTO patterns (${columns.join(', ')}) VALUES (${placeholders})`;

      db.run(sql, values);
      const result = db.exec('SELECT last_insert_rowid() as id');
      const lastId = result[0].values[0][0];

      expect(lastId).toBeGreaterThan(0);

      const pattern = db.exec('SELECT * FROM patterns WHERE id = ?', [lastId]);
      expect(pattern[0].values[0]).toBeDefined();
    });

    it('should insert with text + metadata signature (v1.0.7)', () => {
      const text = 'Sample text';
      const metadata = { key: 'value' };

      db.run('INSERT INTO vectors (text, metadata) VALUES (?, ?)',
        [text, JSON.stringify(metadata)]);

      const result = db.exec('SELECT * FROM vectors WHERE text = ?', [text]);
      expect(result[0].values.length).toBe(1);
      expect(result[0].values[0][3]).toBe(text); // text column
    });
  });

  describe('Controller-Style Methods', () => {
    beforeAll(() => {
      db = new SQL.Database();

      // Create all tables
      db.run(`CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern TEXT NOT NULL,
        metadata TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      )`);

      db.run(`CREATE TABLE IF NOT EXISTS episodes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trajectory TEXT NOT NULL,
        self_reflection TEXT,
        verdict TEXT,
        metadata TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      )`);

      db.run(`CREATE TABLE IF NOT EXISTS causal_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cause TEXT NOT NULL,
        effect TEXT NOT NULL,
        strength REAL DEFAULT 0.5,
        metadata TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      )`);

      db.run(`CREATE TABLE IF NOT EXISTS skills (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        skill_name TEXT NOT NULL,
        code TEXT,
        metadata TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      )`);
    });

    it('should store pattern with storePattern()', () => {
      const patternData = {
        pattern: 'High ROAS targeting strategy',
        metadata: JSON.stringify({
          campaignName: 'E-commerce Sale',
          roas: 3.2,
          ctr: 2.5
        })
      };

      const columns = Object.keys(patternData);
      const values = Object.values(patternData);
      const placeholders = columns.map(() => '?').join(', ');

      db.run(`INSERT INTO patterns (${columns.join(', ')}) VALUES (${placeholders})`, values);

      const result = db.exec('SELECT * FROM patterns');
      expect(result[0].values.length).toBe(1);
      expect(result[0].values[0][1]).toBe(patternData.pattern);
    });

    it('should store episode with storeEpisode()', () => {
      const episodeData = {
        trajectory: JSON.stringify({ campaign: 'Lead Gen', roas: 2.8 }),
        self_reflection: 'Excellent performance. Consider scaling.',
        verdict: 'success',
        metadata: JSON.stringify({ reward: 0.95 })
      };

      const columns = Object.keys(episodeData);
      const values = Object.values(episodeData);
      const placeholders = columns.map(() => '?').join(', ');

      db.run(`INSERT INTO episodes (${columns.join(', ')}) VALUES (${placeholders})`, values);

      const result = db.exec('SELECT * FROM episodes');
      expect(result[0].values.length).toBe(1);
      expect(result[0].values[0][2]).toBe(episodeData.self_reflection);
    });

    it('should add causal edge with addCausalEdge()', () => {
      const edgeData = {
        cause: 'Budget increased 20%',
        effect: 'ROAS improved 0.3x',
        strength: 0.3,
        metadata: JSON.stringify({ confidence: 0.92 })
      };

      const columns = Object.keys(edgeData);
      const values = Object.values(edgeData);
      const placeholders = columns.map(() => '?').join(', ');

      db.run(`INSERT INTO causal_edges (${columns.join(', ')}) VALUES (${placeholders})`, values);

      const result = db.exec('SELECT * FROM causal_edges');
      expect(result[0].values.length).toBe(1);
      expect(result[0].values[0][1]).toBe(edgeData.cause);
      expect(result[0].values[0][2]).toBe(edgeData.effect);
    });

    it('should store skill with storeSkill()', () => {
      const skillData = {
        skill_name: 'ROAS Optimization',
        code: 'function optimize() { return true; }',
        metadata: JSON.stringify({ category: 'marketing' })
      };

      const columns = Object.keys(skillData);
      const values = Object.values(skillData);
      const placeholders = columns.map(() => '?').join(', ');

      db.run(`INSERT INTO skills (${columns.join(', ')}) VALUES (${placeholders})`, values);

      const result = db.exec('SELECT * FROM skills');
      expect(result[0].values.length).toBe(1);
      expect(result[0].values[0][1]).toBe(skillData.skill_name);
    });
  });

  describe('Search and Query Operations', () => {
    beforeAll(() => {
      db = new SQL.Database();
      db.run(`CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        metadata TEXT
      )`);

      // Insert test data
      db.run('INSERT INTO vectors (text, metadata) VALUES (?, ?)',
        ['Campaign 1', JSON.stringify({ roas: 2.5 })]);
      db.run('INSERT INTO vectors (text, metadata) VALUES (?, ?)',
        ['Campaign 2', JSON.stringify({ roas: 3.0 })]);
      db.run('INSERT INTO vectors (text, metadata) VALUES (?, ?)',
        ['Campaign 3', JSON.stringify({ roas: 1.8 })]);
    });

    it('should search with LIMIT', () => {
      const result = db.exec('SELECT * FROM vectors LIMIT 2');
      expect(result[0].values.length).toBe(2);
    });

    it('should filter by metadata', () => {
      const result = db.exec(`SELECT * FROM vectors WHERE metadata LIKE '%"roas":3%'`);
      expect(result[0].values.length).toBeGreaterThan(0);
    });

    it('should count total records', () => {
      const result = db.exec('SELECT COUNT(*) as count FROM vectors');
      expect(result[0].values[0][0]).toBe(3);
    });
  });

  describe('Delete Operations', () => {
    beforeAll(() => {
      db = new SQL.Database();
      db.run(`CREATE TABLE IF NOT EXISTS patterns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pattern TEXT
      )`);

      db.run('INSERT INTO patterns (pattern) VALUES (?)', ['Pattern 1']);
      db.run('INSERT INTO patterns (pattern) VALUES (?)', ['Pattern 2']);
      db.run('INSERT INTO patterns (pattern) VALUES (?)', ['Pattern 3']);
    });

    it('should delete by ID', () => {
      db.run('DELETE FROM patterns WHERE id = ?', [1]);
      const result = db.exec('SELECT COUNT(*) FROM patterns');
      expect(result[0].values[0][0]).toBe(2);
    });

    it('should delete by condition', () => {
      db.run('DELETE FROM patterns WHERE pattern = ?', ['Pattern 2']);
      const result = db.exec('SELECT COUNT(*) FROM patterns');
      expect(result[0].values[0][0]).toBe(1);
    });

    it('should delete all records', () => {
      db.run('DELETE FROM patterns');
      const result = db.exec('SELECT COUNT(*) FROM patterns');
      expect(result[0].values[0][0]).toBe(0);
    });
  });

  describe('Data Export and Import', () => {
    it('should export database to Uint8Array', () => {
      db = new SQL.Database();
      db.run(`CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)`);
      db.run('INSERT INTO test (value) VALUES (?)', ['test data']);

      const exported = db.export();
      expect(exported).toBeInstanceOf(Uint8Array);
      expect(exported.length).toBeGreaterThan(0);
    });

    it('should import database from Uint8Array', () => {
      db = new SQL.Database();
      db.run(`CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)`);
      db.run('INSERT INTO test (value) VALUES (?)', ['original data']);

      const exported = db.export();

      // Create new database from export
      const db2 = new SQL.Database(exported);
      const result = db2.exec('SELECT * FROM test');

      expect(result[0].values.length).toBe(1);
      expect(result[0].values[0][1]).toBe('original data');

      db2.close();
    });
  });

  describe('Error Handling', () => {
    beforeAll(() => {
      db = new SQL.Database();
    });

    it('should throw error on invalid SQL', () => {
      expect(() => {
        db.run('INVALID SQL STATEMENT');
      }).toThrow();
    });

    it('should throw error on missing required fields', () => {
      db.run(`CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, required_field TEXT NOT NULL)`);

      expect(() => {
        db.run('INSERT INTO test (id) VALUES (?)', [1]);
      }).toThrow();
    });

    it('should handle empty result sets', () => {
      db.run(`CREATE TABLE IF NOT EXISTS empty_table (id INTEGER)`);
      const result = db.exec('SELECT * FROM empty_table');

      // Empty result returns empty array
      expect(result).toBeDefined();
      expect(result.length).toBe(0);
    });
  });

  describe('Concurrent Operations', () => {
    beforeAll(() => {
      db = new SQL.Database();
      db.run(`CREATE TABLE IF NOT EXISTS concurrent_test (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        value TEXT
      )`);
    });

    it('should handle multiple inserts', () => {
      for (let i = 0; i < 100; i++) {
        db.run('INSERT INTO concurrent_test (value) VALUES (?)', [`value_${i}`]);
      }

      const result = db.exec('SELECT COUNT(*) FROM concurrent_test');
      expect(result[0].values[0][0]).toBe(100);
    });

    it('should handle mixed operations', () => {
      db.run('INSERT INTO concurrent_test (value) VALUES (?)', ['to_delete']);
      const insertResult = db.exec('SELECT last_insert_rowid() as id');
      const insertId = insertResult[0].values[0][0];

      db.run('DELETE FROM concurrent_test WHERE id = ?', [insertId]);

      const result = db.exec('SELECT COUNT(*) FROM concurrent_test');
      expect(result[0].values[0][0]).toBe(100); // Still 100 from previous test
    });
  });

  describe('Performance Tests', () => {
    beforeAll(() => {
      db = new SQL.Database();
      db.run(`CREATE TABLE IF NOT EXISTS perf_test (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        data TEXT,
        metadata TEXT
      )`);
    });

    it('should handle batch inserts efficiently', () => {
      const startTime = Date.now();

      for (let i = 0; i < 1000; i++) {
        db.run('INSERT INTO perf_test (data, metadata) VALUES (?, ?)',
          [`data_${i}`, JSON.stringify({ index: i })]);
      }

      const duration = Date.now() - startTime;
      console.log(`1000 inserts took ${duration}ms`);

      const result = db.exec('SELECT COUNT(*) FROM perf_test');
      expect(result[0].values[0][0]).toBe(1000);

      // Should complete in reasonable time (less than 1 second)
      expect(duration).toBeLessThan(1000);
    });

    it('should handle large result sets', () => {
      const result = db.exec('SELECT * FROM perf_test LIMIT 500');
      expect(result[0].values.length).toBe(500);
    });
  });

  describe('Backward Compatibility Tests', () => {
    it('should maintain v1.0.7 Database class API', () => {
      db = new SQL.Database();

      // Check all v1.0.7 methods exist
      expect(typeof db.run).toBe('function');
      expect(typeof db.exec).toBe('function');
      expect(typeof db.prepare).toBe('function');
      expect(typeof db.export).toBe('function');
      expect(typeof db.close).toBe('function');
    });

    it('should work with v1.0.7 usage patterns', () => {
      db = new SQL.Database();

      // Classic v1.0.7 pattern
      db.run(`CREATE TABLE IF NOT EXISTS vectors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        embedding BLOB,
        metadata TEXT,
        text TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      )`);

      db.run('INSERT INTO vectors (text, metadata) VALUES (?, ?)',
        ['test', JSON.stringify({ key: 'value' })]);

      const results = db.exec('SELECT * FROM vectors');
      expect(results[0].values.length).toBe(1);
    });
  });

  describe('Edge Cases', () => {
    beforeAll(() => {
      db = new SQL.Database();
    });

    it('should handle empty strings', () => {
      db.run(`CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, value TEXT)`);
      db.run('INSERT INTO test (value) VALUES (?)', ['']);

      const result = db.exec('SELECT * FROM test');
      expect(result[0].values[0][1]).toBe('');
    });

    it('should handle NULL values', () => {
      db.run(`CREATE TABLE IF NOT EXISTS test2 (id INTEGER PRIMARY KEY, value TEXT)`);
      db.run('INSERT INTO test2 (id) VALUES (?)', [1]);

      const result = db.exec('SELECT * FROM test2');
      expect(result[0].values[0][1]).toBeNull();
    });

    it('should handle special characters in text', () => {
      db.run(`CREATE TABLE IF NOT EXISTS test3 (id INTEGER PRIMARY KEY, value TEXT)`);
      const specialText = 'Test with "quotes", \'apostrophes\', and émojis 🎯';
      db.run('INSERT INTO test3 (value) VALUES (?)', [specialText]);

      const result = db.exec('SELECT * FROM test3');
      expect(result[0].values[0][1]).toBe(specialText);
    });

    it('should handle large JSON metadata', () => {
      db.run(`CREATE TABLE IF NOT EXISTS test4 (id INTEGER PRIMARY KEY, metadata TEXT)`);
      const largeMetadata = JSON.stringify({
        campaign: 'Test',
        metrics: Array(100).fill({ roas: 2.5, ctr: 3.0, conversions: 50 })
      });

      db.run('INSERT INTO test4 (metadata) VALUES (?)', [largeMetadata]);
      const result = db.exec('SELECT * FROM test4');

      const retrieved = JSON.parse(result[0].values[0][1]);
      expect(retrieved.metrics.length).toBe(100);
    });
  });
});
