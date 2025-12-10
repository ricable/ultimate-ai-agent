/**
 * SQLite Operations Tests
 * Tests database operations for the Hive Mind system
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Test database path
const testDbPath = path.join(__dirname, '../../../tmp/test-sqlite-ops.db');

// Database schema
const SCHEMA = `
CREATE TABLE IF NOT EXISTS agents (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  status TEXT DEFAULT 'active',
  capabilities TEXT,
  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  last_active INTEGER DEFAULT (strftime('%s', 'now'))
);

CREATE TABLE IF NOT EXISTS tasks (
  id TEXT PRIMARY KEY,
  description TEXT NOT NULL,
  status TEXT DEFAULT 'pending',
  priority TEXT DEFAULT 'medium',
  assigned_to TEXT,
  dependencies TEXT,
  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  completed_at INTEGER,
  FOREIGN KEY (assigned_to) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_agent TEXT NOT NULL,
  to_agent TEXT,
  type TEXT DEFAULT 'info',
  content TEXT NOT NULL,
  timestamp INTEGER DEFAULT (strftime('%s', 'now')),
  FOREIGN KEY (from_agent) REFERENCES agents(id),
  FOREIGN KEY (to_agent) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS consensus (
  id TEXT PRIMARY KEY,
  topic TEXT NOT NULL,
  status TEXT DEFAULT 'voting',
  initiator TEXT NOT NULL,
  participants TEXT NOT NULL,
  votes TEXT,
  result TEXT,
  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  closed_at INTEGER,
  FOREIGN KEY (initiator) REFERENCES agents(id)
);

CREATE TABLE IF NOT EXISTS memory (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  type TEXT DEFAULT 'general',
  access_count INTEGER DEFAULT 0,
  created_at INTEGER DEFAULT (strftime('%s', 'now')),
  updated_at INTEGER DEFAULT (strftime('%s', 'now')),
  expires_at INTEGER
);

CREATE INDEX idx_agents_type ON agents(type);
CREATE INDEX idx_agents_status ON agents(status);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_assigned ON tasks(assigned_to);
CREATE INDEX idx_messages_from ON messages(from_agent);
CREATE INDEX idx_messages_to ON messages(to_agent);
CREATE INDEX idx_memory_type ON memory(type);
`;

describe('SQLite Operations', () => {
  let db;

  beforeEach(() => {
    // Ensure tmp directory exists
    const tmpDir = path.dirname(testDbPath);
    if (!fs.existsSync(tmpDir)) {
      fs.mkdirSync(tmpDir, { recursive: true });
    }

    // Create and initialize database
    db = new Database(testDbPath);
    db.exec(SCHEMA);
  });

  afterEach(() => {
    // Close and clean up
    if (db) {
      db.close();
    }
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Agent Operations', () => {
    it('should insert an agent', () => {
      const stmt = db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)');
      const result = stmt.run('agent-001', 'ResearchBot', 'researcher');
      
      expect(result.changes).toBe(1);
      
      const agent = db.prepare('SELECT * FROM agents WHERE id = ?').get('agent-001');
      expect(agent.name).toBe('ResearchBot');
      expect(agent.type).toBe('researcher');
      expect(agent.status).toBe('active');
    });

    it('should update agent status', () => {
      // Insert agent
      db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)')
        .run('agent-001', 'TestBot', 'coder');
      
      // Update status
      const stmt = db.prepare('UPDATE agents SET status = ?, last_active = ? WHERE id = ?');
      const result = stmt.run('busy', Date.now(), 'agent-001');
      
      expect(result.changes).toBe(1);
      
      const agent = db.prepare('SELECT * FROM agents WHERE id = ?').get('agent-001');
      expect(agent.status).toBe('busy');
    });

    it('should list agents by type', () => {
      // Insert multiple agents
      const insert = db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)');
      insert.run('agent-001', 'R1', 'researcher');
      insert.run('agent-002', 'R2', 'researcher');
      insert.run('agent-003', 'C1', 'coder');
      
      const researchers = db.prepare('SELECT * FROM agents WHERE type = ?').all('researcher');
      expect(researchers).toHaveLength(2);
      expect(researchers.every(a => a.type === 'researcher')).toBe(true);
    });

    it('should handle agent capabilities as JSON', () => {
      const capabilities = JSON.stringify(['analysis', 'data-mining', 'reporting']);
      db.prepare('INSERT INTO agents (id, name, type, capabilities) VALUES (?, ?, ?, ?)')
        .run('agent-001', 'DataBot', 'analyst', capabilities);
      
      const agent = db.prepare('SELECT * FROM agents WHERE id = ?').get('agent-001');
      const parsed = JSON.parse(agent.capabilities);
      expect(parsed).toContain('analysis');
      expect(parsed).toContain('data-mining');
    });
  });

  describe('Task Operations', () => {
    beforeEach(() => {
      // Insert test agents
      db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)')
        .run('agent-001', 'TestBot', 'coder');
    });

    it('should create a task', () => {
      const stmt = db.prepare('INSERT INTO tasks (id, description) VALUES (?, ?)');
      const result = stmt.run('task-001', 'Implement authentication');
      
      expect(result.changes).toBe(1);
      
      const task = db.prepare('SELECT * FROM tasks WHERE id = ?').get('task-001');
      expect(task.description).toBe('Implement authentication');
      expect(task.status).toBe('pending');
      expect(task.priority).toBe('medium');
    });

    it('should assign task to agent', () => {
      // Create task
      db.prepare('INSERT INTO tasks (id, description) VALUES (?, ?)')
        .run('task-001', 'Write tests');
      
      // Assign to agent
      const stmt = db.prepare('UPDATE tasks SET assigned_to = ?, status = ? WHERE id = ?');
      const result = stmt.run('agent-001', 'in_progress', 'task-001');
      
      expect(result.changes).toBe(1);
      
      const task = db.prepare('SELECT * FROM tasks WHERE id = ?').get('task-001');
      expect(task.assigned_to).toBe('agent-001');
      expect(task.status).toBe('in_progress');
    });

    it('should handle task dependencies', () => {
      const deps = JSON.stringify(['task-001', 'task-002']);
      db.prepare('INSERT INTO tasks (id, description, dependencies) VALUES (?, ?, ?)')
        .run('task-003', 'Deploy application', deps);
      
      const task = db.prepare('SELECT * FROM tasks WHERE id = ?').get('task-003');
      const dependencies = JSON.parse(task.dependencies);
      expect(dependencies).toContain('task-001');
      expect(dependencies).toContain('task-002');
    });

    it('should query tasks by status', () => {
      // Insert tasks with different statuses
      const insert = db.prepare('INSERT INTO tasks (id, description, status) VALUES (?, ?, ?)');
      insert.run('task-001', 'Task 1', 'pending');
      insert.run('task-002', 'Task 2', 'in_progress');
      insert.run('task-003', 'Task 3', 'in_progress');
      insert.run('task-004', 'Task 4', 'completed');
      
      const inProgress = db.prepare('SELECT * FROM tasks WHERE status = ?').all('in_progress');
      expect(inProgress).toHaveLength(2);
    });
  });

  describe('Message Operations', () => {
    beforeEach(() => {
      // Insert test agents
      const insert = db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)');
      insert.run('agent-001', 'Bot1', 'researcher');
      insert.run('agent-002', 'Bot2', 'coder');
    });

    it('should send message between agents', () => {
      const stmt = db.prepare('INSERT INTO messages (from_agent, to_agent, content) VALUES (?, ?, ?)');
      const result = stmt.run('agent-001', 'agent-002', 'Found API documentation');
      
      expect(result.changes).toBe(1);
      expect(result.lastInsertRowid).toBeGreaterThan(0);
    });

    it('should broadcast message to all', () => {
      const stmt = db.prepare('INSERT INTO messages (from_agent, to_agent, type, content) VALUES (?, ?, ?, ?)');
      const result = stmt.run('agent-001', null, 'broadcast', 'Team update: Starting sprint');
      
      expect(result.changes).toBe(1);
      
      const msg = db.prepare('SELECT * FROM messages WHERE id = ?').get(result.lastInsertRowid);
      expect(msg.to_agent).toBeNull();
      expect(msg.type).toBe('broadcast');
    });

    it('should retrieve conversation history', () => {
      // Insert multiple messages
      const insert = db.prepare('INSERT INTO messages (from_agent, to_agent, content) VALUES (?, ?, ?)');
      insert.run('agent-001', 'agent-002', 'Message 1');
      insert.run('agent-002', 'agent-001', 'Message 2');
      insert.run('agent-001', 'agent-002', 'Message 3');
      
      // Get conversation between two agents
      const conversation = db.prepare(`
        SELECT * FROM messages 
        WHERE (from_agent = ? AND to_agent = ?) 
           OR (from_agent = ? AND to_agent = ?)
        ORDER BY timestamp
      `).all('agent-001', 'agent-002', 'agent-002', 'agent-001');
      
      expect(conversation).toHaveLength(3);
      expect(conversation[0].content).toBe('Message 1');
      expect(conversation[1].content).toBe('Message 2');
      expect(conversation[2].content).toBe('Message 3');
    });
  });

  describe('Consensus Operations', () => {
    beforeEach(() => {
      // Insert test agents
      const insert = db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)');
      insert.run('agent-001', 'Bot1', 'researcher');
      insert.run('agent-002', 'Bot2', 'analyst');
      insert.run('agent-003', 'Bot3', 'coder');
    });

    it('should create consensus vote', () => {
      const participants = JSON.stringify(['agent-001', 'agent-002', 'agent-003']);
      const stmt = db.prepare('INSERT INTO consensus (id, topic, initiator, participants) VALUES (?, ?, ?, ?)');
      const result = stmt.run('vote-001', 'Use TypeScript?', 'agent-001', participants);
      
      expect(result.changes).toBe(1);
      
      const vote = db.prepare('SELECT * FROM consensus WHERE id = ?').get('vote-001');
      expect(vote.topic).toBe('Use TypeScript?');
      expect(vote.status).toBe('voting');
    });

    it('should record votes', () => {
      // Create consensus
      const participants = JSON.stringify(['agent-001', 'agent-002', 'agent-003']);
      db.prepare('INSERT INTO consensus (id, topic, initiator, participants) VALUES (?, ?, ?, ?)')
        .run('vote-001', 'Tech decision', 'agent-001', participants);
      
      // Record votes
      const votes = JSON.stringify({
        'agent-001': { choice: 'yes', weight: 1.0, reason: 'Type safety' },
        'agent-002': { choice: 'yes', weight: 0.8, reason: 'Better tooling' },
        'agent-003': { choice: 'no', weight: 0.6, reason: 'Learning curve' }
      });
      
      const stmt = db.prepare('UPDATE consensus SET votes = ? WHERE id = ?');
      const result = stmt.run(votes, 'vote-001');
      
      expect(result.changes).toBe(1);
      
      const consensus = db.prepare('SELECT * FROM consensus WHERE id = ?').get('vote-001');
      const parsedVotes = JSON.parse(consensus.votes);
      expect(parsedVotes['agent-001'].choice).toBe('yes');
    });

    it('should close consensus with result', () => {
      // Create and setup consensus
      const participants = JSON.stringify(['agent-001', 'agent-002']);
      db.prepare('INSERT INTO consensus (id, topic, initiator, participants) VALUES (?, ?, ?, ?)')
        .run('vote-001', 'Decision', 'agent-001', participants);
      
      // Close with result
      const result = JSON.stringify({ decision: 'approved', confidence: 0.85 });
      const stmt = db.prepare('UPDATE consensus SET status = ?, result = ?, closed_at = ? WHERE id = ?');
      const updateResult = stmt.run('closed', result, Date.now(), 'vote-001');
      
      expect(updateResult.changes).toBe(1);
      
      const consensus = db.prepare('SELECT * FROM consensus WHERE id = ?').get('vote-001');
      expect(consensus.status).toBe('closed');
      expect(JSON.parse(consensus.result).decision).toBe('approved');
    });
  });

  describe('Memory Operations', () => {
    it('should store and retrieve memory', () => {
      const stmt = db.prepare('INSERT INTO memory (key, value) VALUES (?, ?)');
      const result = stmt.run('api_key', 'secret123');
      
      expect(result.changes).toBe(1);
      
      const memory = db.prepare('SELECT * FROM memory WHERE key = ?').get('api_key');
      expect(memory.value).toBe('secret123');
      expect(memory.access_count).toBe(0);
    });

    it('should update access count on retrieval', () => {
      // Store memory
      db.prepare('INSERT INTO memory (key, value) VALUES (?, ?)')
        .run('config', '{"port": 3000}');
      
      // Simulate retrieval with access count update
      db.prepare('UPDATE memory SET access_count = access_count + 1 WHERE key = ?')
        .run('config');
      
      const memory = db.prepare('SELECT * FROM memory WHERE key = ?').get('config');
      expect(memory.access_count).toBe(1);
    });

    it('should handle memory expiration', () => {
      // Store memory with expiration
      const expiresAt = Date.now() - 1000; // Expired 1 second ago
      db.prepare('INSERT INTO memory (key, value, expires_at) VALUES (?, ?, ?)')
        .run('temp_data', 'old_value', expiresAt);
      
      // Query non-expired memory
      const active = db.prepare('SELECT * FROM memory WHERE expires_at IS NULL OR expires_at > ?')
        .all(Date.now());
      
      expect(active).toHaveLength(0);
    });

    it('should categorize memory by type', () => {
      // Insert different types of memory
      const insert = db.prepare('INSERT INTO memory (key, value, type) VALUES (?, ?, ?)');
      insert.run('api_docs', 'https://...', 'reference');
      insert.run('user_pref', '{"theme": "dark"}', 'config');
      insert.run('cache_1', 'cached_data', 'cache');
      
      const references = db.prepare('SELECT * FROM memory WHERE type = ?').all('reference');
      expect(references).toHaveLength(1);
      expect(references[0].key).toBe('api_docs');
    });
  });

  describe('Performance and Indexing', () => {
    it('should efficiently query agents by type using index', () => {
      // Insert many agents
      const insert = db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)');
      for (let i = 0; i < 100; i++) {
        insert.run(`agent-${i}`, `Bot${i}`, i % 3 === 0 ? 'researcher' : 'coder');
      }
      
      // Query with EXPLAIN to verify index usage
      const explain = db.prepare('EXPLAIN QUERY PLAN SELECT * FROM agents WHERE type = ?').all('researcher');
      expect(explain[0].detail).toContain('idx_agents_type');
    });

    it('should handle concurrent operations with transactions', () => {
      const insertAgent = db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)');
      const insertTask = db.prepare('INSERT INTO tasks (id, description, assigned_to) VALUES (?, ?, ?)');
      
      // Use transaction for atomic operations
      const transaction = db.transaction((agents, tasks) => {
        for (const agent of agents) {
          insertAgent.run(agent.id, agent.name, agent.type);
        }
        for (const task of tasks) {
          insertTask.run(task.id, task.description, task.assignedTo);
        }
      });
      
      const agents = [
        { id: 'agent-001', name: 'Bot1', type: 'researcher' },
        { id: 'agent-002', name: 'Bot2', type: 'coder' }
      ];
      
      const tasks = [
        { id: 'task-001', description: 'Task 1', assignedTo: 'agent-001' },
        { id: 'task-002', description: 'Task 2', assignedTo: 'agent-002' }
      ];
      
      transaction(agents, tasks);
      
      const agentCount = db.prepare('SELECT COUNT(*) as count FROM agents').get().count;
      const taskCount = db.prepare('SELECT COUNT(*) as count FROM tasks').get().count;
      
      expect(agentCount).toBe(2);
      expect(taskCount).toBe(2);
    });
  });

  describe('Data Integrity', () => {
    it('should enforce foreign key constraints', () => {
      // Try to insert task with non-existent agent
      const stmt = db.prepare('INSERT INTO tasks (id, description, assigned_to) VALUES (?, ?, ?)');
      
      expect(() => {
        stmt.run('task-001', 'Invalid task', 'non-existent-agent');
      }).toThrow(/FOREIGN KEY constraint failed/);
    });

    it('should maintain referential integrity on agent deletion', () => {
      // Insert agent and task
      db.prepare('INSERT INTO agents (id, name, type) VALUES (?, ?, ?)')
        .run('agent-001', 'Bot1', 'coder');
      db.prepare('INSERT INTO tasks (id, description, assigned_to) VALUES (?, ?, ?)')
        .run('task-001', 'Task 1', 'agent-001');
      
      // Try to delete agent with assigned tasks
      expect(() => {
        db.prepare('DELETE FROM agents WHERE id = ?').run('agent-001');
      }).toThrow(/FOREIGN KEY constraint failed/);
    });
  });
});