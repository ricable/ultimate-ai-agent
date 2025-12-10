/**
 * Hive Mind CLI Command Tests
 * Tests all CLI commands and subcommands for the Hive Mind system
 */

import { describe, it, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { jest as jestMock } from '@jest/globals';
import { execSync } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Mock SQLite database for testing
const mockDb = {
  prepare: jest.fn(() => ({
    run: jest.fn(),
    get: jest.fn(),
    all: jest.fn()
  })),
  exec: jest.fn(),
  close: jest.fn()
};

// Test database path
const testDbPath = path.join(__dirname, '../../../tmp/test-hive-mind.db');

describe('Hive Mind CLI Commands', () => {
  beforeEach(() => {
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
    // Reset all mocks
    jest.clearAllMocks();
  });

  afterEach(() => {
    // Clean up
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('hive-mind command', () => {
    it('should display help when no arguments provided', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind --help', {
        encoding: 'utf8'
      });
      expect(result).toContain('Hive Mind - Distributed AI Agent Coordination');
      expect(result).toContain('Usage: hive-mind <objective>');
      expect(result).toContain('Options:');
      expect(result).toContain('--agents');
      expect(result).toContain('--strategy');
    });

    it('should initialize with wizard mode', () => {
      const mockInput = 'Test Objective\n5\nhierarchical\ncode-review\n';
      const result = execSync('echo "' + mockInput + '" | node src/cli/simple-cli.js hive-mind', {
        encoding: 'utf8'
      });
      expect(result).toContain('🧙 Hive Mind Setup Wizard');
      expect(result).toContain('Enter your objective');
      expect(result).toContain('How many agents');
      expect(result).toContain('Select topology');
    });

    it('should accept command line arguments', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind "Refactor authentication" --agents 5 --strategy hierarchical', {
        encoding: 'utf8'
      });
      expect(result).toContain('Initializing Hive Mind');
      expect(result).toContain('Objective: Refactor authentication');
      expect(result).toContain('Agents: 5');
      expect(result).toContain('Strategy: hierarchical');
    });

    it('should handle maximum agent limits', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind "Test" --agents 1001', {
        encoding: 'utf8'
      });
      expect(result).toContain('Warning: Maximum 1000 agents allowed');
    });
  });

  describe('hive-mind init', () => {
    it('should initialize SQLite database', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Initializing Hive Mind database');
      expect(result).toContain('Database created successfully');
      expect(fs.existsSync(testDbPath)).toBe(true);
    });

    it('should create required tables', () => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      
      // Verify tables exist
      const db = new Database(testDbPath);
      const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all();
      const tableNames = tables.map(t => t.name);
      
      expect(tableNames).toContain('agents');
      expect(tableNames).toContain('tasks');
      expect(tableNames).toContain('messages');
      expect(tableNames).toContain('consensus');
      expect(tableNames).toContain('memory');
      
      db.close();
    });

    it('should handle existing database gracefully', () => {
      // First init
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      
      // Second init
      const result = execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath + ' --force', {
        encoding: 'utf8'
      });
      expect(result).toContain('Reinitializing database');
    });
  });

  describe('hive-mind spawn', () => {
    beforeEach(() => {
      // Initialize database for spawn tests
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
    });

    it('should spawn a single agent', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind spawn researcher --name "DataBot" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Spawning agent: DataBot');
      expect(result).toContain('Type: researcher');
      expect(result).toContain('Agent spawned successfully');
    });

    it('should spawn multiple agents in batch', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind spawn --batch "researcher:3,coder:2,analyst:1" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Spawning 6 agents');
      expect(result).toContain('3 researcher agents');
      expect(result).toContain('2 coder agents');
      expect(result).toContain('1 analyst agents');
    });

    it('should validate agent types', () => {
      try {
        execSync('node src/cli/simple-cli.js hive-mind spawn invalid-type --db ' + testDbPath);
        fail('Should have thrown error');
      } catch (error) {
        expect(error.stderr.toString()).toContain('Invalid agent type');
        expect(error.stderr.toString()).toContain('Valid types:');
      }
    });

    it('should assign unique IDs to agents', () => {
      execSync('node src/cli/simple-cli.js hive-mind spawn researcher --name "Bot1" --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn researcher --name "Bot2" --db ' + testDbPath);
      
      const result = execSync('node src/cli/simple-cli.js hive-mind list --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toMatch(/agent-[a-z0-9]{8}/g);
    });
  });

  describe('hive-mind list', () => {
    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn researcher --name "R1" --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn coder --name "C1" --db ' + testDbPath);
    });

    it('should list all agents', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind list --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Active Agents:');
      expect(result).toContain('R1 (researcher)');
      expect(result).toContain('C1 (coder)');
      expect(result).toContain('Total: 2 agents');
    });

    it('should filter by agent type', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind list --type researcher --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('R1 (researcher)');
      expect(result).not.toContain('C1 (coder)');
    });

    it('should show detailed information with --verbose', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind list --verbose --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('ID:');
      expect(result).toContain('Status:');
      expect(result).toContain('Created:');
      expect(result).toContain('Tasks:');
    });
  });

  describe('hive-mind task', () => {
    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn researcher --name "R1" --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn coder --name "C1" --db ' + testDbPath);
    });

    it('should distribute task to agents', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind task "Implement user authentication" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Distributing task:');
      expect(result).toContain('Task distributed to');
      expect(result).toContain('agents');
    });

    it('should support task priorities', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind task "Critical bug fix" --priority critical --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Priority: critical');
    });

    it('should handle task dependencies', () => {
      execSync('node src/cli/simple-cli.js hive-mind task "Task A" --id task-a --db ' + testDbPath);
      const result = execSync('node src/cli/simple-cli.js hive-mind task "Task B" --depends-on task-a --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Dependencies: task-a');
    });
  });

  describe('hive-mind communicate', () => {
    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn researcher --name "R1" --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn coder --name "C1" --db ' + testDbPath);
    });

    it('should send message between agents', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind communicate --from R1 --to C1 --message "Found relevant API docs" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Message sent');
      expect(result).toContain('From: R1');
      expect(result).toContain('To: C1');
    });

    it('should support broadcast messages', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind communicate --from R1 --broadcast --message "Team update" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Broadcasting message');
      expect(result).toContain('to all agents');
    });

    it('should handle message types', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind communicate --from R1 --to C1 --message "Code review needed" --type request --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Type: request');
    });
  });

  describe('hive-mind consensus', () => {
    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn --batch "researcher:3,analyst:2" --db ' + testDbPath);
    });

    it('should initiate consensus voting', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind consensus "Should we use TypeScript?" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Initiating consensus');
      expect(result).toContain('Topic:');
      expect(result).toContain('Participants: 5');
    });

    it('should show voting results', () => {
      execSync('node src/cli/simple-cli.js hive-mind consensus "Tech stack decision" --id vote-1 --db ' + testDbPath);
      const result = execSync('node src/cli/simple-cli.js hive-mind consensus status vote-1 --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Consensus Status');
      expect(result).toContain('Votes:');
    });

    it('should support weighted voting', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind consensus "Architecture decision" --weighted --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Weighted voting enabled');
    });
  });

  describe('hive-mind memory', () => {
    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
    });

    it('should store shared memory', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind memory store --key "api-docs" --value "https://api.example.com/docs" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Memory stored');
      expect(result).toContain('Key: api-docs');
    });

    it('should retrieve shared memory', () => {
      execSync('node src/cli/simple-cli.js hive-mind memory store --key "config" --value "{"port":3000}" --db ' + testDbPath);
      const result = execSync('node src/cli/simple-cli.js hive-mind memory get --key "config" --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('"port":3000');
    });

    it('should list all memory entries', () => {
      execSync('node src/cli/simple-cli.js hive-mind memory store --key "k1" --value "v1" --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind memory store --key "k2" --value "v2" --db ' + testDbPath);
      const result = execSync('node src/cli/simple-cli.js hive-mind memory list --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('k1');
      expect(result).toContain('k2');
    });
  });

  describe('hive-mind status', () => {
    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn --batch "researcher:2,coder:3" --db ' + testDbPath);
    });

    it('should show swarm status', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind status --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Hive Mind Status');
      expect(result).toContain('Total Agents: 5');
      expect(result).toContain('Active Tasks:');
      expect(result).toContain('Messages:');
    });

    it('should show performance metrics', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind status --metrics --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Performance Metrics');
      expect(result).toContain('Avg Response Time:');
      expect(result).toContain('Task Completion Rate:');
    });
  });

  describe('hive-mind scale', () => {
    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn --batch "researcher:2" --db ' + testDbPath);
    });

    it('should scale up agents', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind scale up --agents 3 --type researcher --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Scaling up');
      expect(result).toContain('Adding 3 researcher agents');
    });

    it('should scale down agents', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind scale down --agents 1 --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Scaling down');
      expect(result).toContain('Removing 1 agents');
    });

    it('should auto-scale based on load', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind scale auto --min 2 --max 10 --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Auto-scaling enabled');
      expect(result).toContain('Min agents: 2');
      expect(result).toContain('Max agents: 10');
    });
  });

  describe('hive-mind export/import', () => {
    const exportPath = path.join(__dirname, '../../../tmp/hive-export.json');

    beforeEach(() => {
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + testDbPath);
      execSync('node src/cli/simple-cli.js hive-mind spawn researcher --name "R1" --db ' + testDbPath);
    });

    afterEach(() => {
      if (fs.existsSync(exportPath)) {
        fs.unlinkSync(exportPath);
      }
    });

    it('should export swarm configuration', () => {
      const result = execSync('node src/cli/simple-cli.js hive-mind export --output ' + exportPath + ' --db ' + testDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Exporting Hive Mind configuration');
      expect(fs.existsSync(exportPath)).toBe(true);
      
      const exported = JSON.parse(fs.readFileSync(exportPath, 'utf8'));
      expect(exported).toHaveProperty('agents');
      expect(exported).toHaveProperty('memory');
      expect(exported).toHaveProperty('configuration');
    });

    it('should import swarm configuration', () => {
      execSync('node src/cli/simple-cli.js hive-mind export --output ' + exportPath + ' --db ' + testDbPath);
      
      // Create new database
      const newDbPath = testDbPath.replace('.db', '-new.db');
      execSync('node src/cli/simple-cli.js hive-mind init --db ' + newDbPath);
      
      const result = execSync('node src/cli/simple-cli.js hive-mind import --input ' + exportPath + ' --db ' + newDbPath, {
        encoding: 'utf8'
      });
      expect(result).toContain('Importing Hive Mind configuration');
      expect(result).toContain('Import successful');
      
      // Clean up
      if (fs.existsSync(newDbPath)) {
        fs.unlinkSync(newDbPath);
      }
    });
  });
});