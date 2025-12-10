/**
 * Swarm Scaling Integration Tests
 * Tests agent scaling from 1 to 100+ agents
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';
import os from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Swarm orchestrator for scaling tests
class SwarmOrchestrator {
  constructor(dbPath) {
    this.dbPath = dbPath;
    this.db = new Database(dbPath);
    this.agents = new Map();
    this.tasks = new Map();
    this.metrics = {
      spawnTime: [],
      taskDistributionTime: [],
      communicationLatency: [],
      memoryUsage: [],
      cpuUsage: []
    };
    
    this.initializeDatabase();
  }

  initializeDatabase() {
    // Create tables
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS agents (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        status TEXT DEFAULT 'active',
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      );

      CREATE TABLE IF NOT EXISTS tasks (
        id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        assigned_to TEXT,
        created_at INTEGER DEFAULT (strftime('%s', 'now')),
        completed_at INTEGER,
        FOREIGN KEY (assigned_to) REFERENCES agents(id)
      );

      CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_agent TEXT NOT NULL,
        to_agent TEXT,
        content TEXT NOT NULL,
        timestamp INTEGER DEFAULT (strftime('%s', 'now')),
        FOREIGN KEY (from_agent) REFERENCES agents(id)
      );

      CREATE TABLE IF NOT EXISTS performance_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        metric_type TEXT NOT NULL,
        value REAL NOT NULL,
        agent_count INTEGER,
        timestamp INTEGER DEFAULT (strftime('%s', 'now'))
      );
    `);
  }

  async scaleAgents(targetCount, agentType = 'mixed') {
    const startTime = Date.now();
    const currentCount = this.agents.size;
    
    if (targetCount > currentCount) {
      // Scale up
      await this.spawnAgents(targetCount - currentCount, agentType);
    } else if (targetCount < currentCount) {
      // Scale down
      await this.removeAgents(currentCount - targetCount);
    }

    const scaleTime = Date.now() - startTime;
    this.recordMetric('scale_time', scaleTime, targetCount);
    
    return {
      previousCount: currentCount,
      currentCount: this.agents.size,
      scaleTime
    };
  }

  async spawnAgents(count, agentType = 'mixed') {
    const startTime = Date.now();
    const spawned = [];
    
    const agentTypes = ['researcher', 'coder', 'analyst', 'architect', 'tester', 'coordinator'];
    const batchSize = Math.min(count, 50); // Spawn in batches for performance
    
    for (let i = 0; i < count; i += batchSize) {
      const batchCount = Math.min(batchSize, count - i);
      const batch = [];
      
      for (let j = 0; j < batchCount; j++) {
        const type = agentType === 'mixed' 
          ? agentTypes[(i + j) % agentTypes.length]
          : agentType;
        
        const agent = {
          id: `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          name: `${type}-${this.agents.size + 1}`,
          type,
          status: 'active',
          createdAt: Date.now()
        };
        
        batch.push(agent);
        this.agents.set(agent.id, agent);
      }
      
      // Batch insert to database
      const insert = this.db.prepare(
        'INSERT INTO agents (id, name, type, status) VALUES (?, ?, ?, ?)'
      );
      
      const insertMany = this.db.transaction((agents) => {
        for (const agent of agents) {
          insert.run(agent.id, agent.name, agent.type, agent.status);
        }
      });
      
      insertMany(batch);
      spawned.push(...batch);
      
      // Simulate spawn delay
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    
    const spawnTime = Date.now() - startTime;
    this.metrics.spawnTime.push({ count, time: spawnTime });
    this.recordMetric('spawn_time', spawnTime, this.agents.size);
    
    return spawned;
  }

  async removeAgents(count) {
    const toRemove = [];
    const iterator = this.agents.entries();
    
    for (let i = 0; i < count && i < this.agents.size; i++) {
      const [id] = iterator.next().value;
      toRemove.push(id);
    }
    
    // Batch delete from database
    const deleteStmt = this.db.prepare('DELETE FROM agents WHERE id = ?');
    const deleteMany = this.db.transaction((ids) => {
      for (const id of ids) {
        deleteStmt.run(id);
        this.agents.delete(id);
      }
    });
    
    deleteMany(toRemove);
    
    return toRemove.length;
  }

  async distributeTasks(taskCount) {
    const startTime = Date.now();
    const tasks = [];
    const agents = Array.from(this.agents.values());
    
    if (agents.length === 0) {
      throw new Error('No agents available for task distribution');
    }
    
    // Create tasks
    for (let i = 0; i < taskCount; i++) {
      const task = {
        id: `task-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        description: `Task ${i + 1}: Process data batch ${i + 1}`,
        status: 'pending',
        assignedTo: null,
        createdAt: Date.now()
      };
      tasks.push(task);
      this.tasks.set(task.id, task);
    }
    
    // Distribute tasks among agents
    const distribution = this.getOptimalDistribution(tasks.length, agents.length);
    let taskIndex = 0;
    
    for (let i = 0; i < agents.length; i++) {
      const agentTaskCount = distribution[i];
      for (let j = 0; j < agentTaskCount; j++) {
        if (taskIndex < tasks.length) {
          tasks[taskIndex].assignedTo = agents[i].id;
          taskIndex++;
        }
      }
    }
    
    // Batch insert tasks
    const insert = this.db.prepare(
      'INSERT INTO tasks (id, description, status, assigned_to) VALUES (?, ?, ?, ?)'
    );
    
    const insertMany = this.db.transaction((tasks) => {
      for (const task of tasks) {
        insert.run(task.id, task.description, task.status, task.assignedTo);
      }
    });
    
    insertMany(tasks);
    
    const distributionTime = Date.now() - startTime;
    this.metrics.taskDistributionTime.push({
      taskCount,
      agentCount: agents.length,
      time: distributionTime
    });
    this.recordMetric('distribution_time', distributionTime, agents.length);
    
    return {
      tasksCreated: tasks.length,
      agentsUsed: new Set(tasks.map(t => t.assignedTo)).size,
      avgTasksPerAgent: tasks.length / agents.length,
      distributionTime
    };
  }

  getOptimalDistribution(taskCount, agentCount) {
    const baseTasksPerAgent = Math.floor(taskCount / agentCount);
    const remainder = taskCount % agentCount;
    const distribution = new Array(agentCount).fill(baseTasksPerAgent);
    
    // Distribute remainder
    for (let i = 0; i < remainder; i++) {
      distribution[i]++;
    }
    
    return distribution;
  }

  async simulateCommunication(messageCount) {
    const startTime = Date.now();
    const agents = Array.from(this.agents.values());
    
    if (agents.length < 2) {
      throw new Error('Need at least 2 agents for communication');
    }
    
    const messages = [];
    
    // Generate messages
    for (let i = 0; i < messageCount; i++) {
      const fromIndex = Math.floor(Math.random() * agents.length);
      let toIndex = Math.floor(Math.random() * agents.length);
      
      // Ensure different agents
      while (toIndex === fromIndex) {
        toIndex = Math.floor(Math.random() * agents.length);
      }
      
      const message = {
        from: agents[fromIndex].id,
        to: agents[toIndex].id,
        content: `Message ${i + 1}: Status update from ${agents[fromIndex].name}`,
        timestamp: Date.now()
      };
      
      messages.push(message);
    }
    
    // Batch insert messages
    const insert = this.db.prepare(
      'INSERT INTO messages (from_agent, to_agent, content) VALUES (?, ?, ?)'
    );
    
    const insertMany = this.db.transaction((messages) => {
      for (const msg of messages) {
        insert.run(msg.from, msg.to, msg.content);
      }
    });
    
    insertMany(messages);
    
    const communicationTime = Date.now() - startTime;
    const avgLatency = communicationTime / messageCount;
    
    this.metrics.communicationLatency.push({
      messageCount,
      agentCount: agents.length,
      totalTime: communicationTime,
      avgLatency
    });
    
    this.recordMetric('communication_latency', avgLatency, agents.length);
    
    return {
      messagesSent: messages.length,
      totalTime: communicationTime,
      avgLatency,
      messagesPerSecond: (messageCount / communicationTime) * 1000
    };
  }

  measureResourceUsage() {
    const usage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    const metrics = {
      memory: {
        heapUsed: usage.heapUsed / 1024 / 1024, // MB
        heapTotal: usage.heapTotal / 1024 / 1024,
        external: usage.external / 1024 / 1024,
        rss: usage.rss / 1024 / 1024
      },
      cpu: {
        user: cpuUsage.user / 1000000, // seconds
        system: cpuUsage.system / 1000000
      },
      agentCount: this.agents.size,
      taskCount: this.tasks.size,
      timestamp: Date.now()
    };
    
    this.metrics.memoryUsage.push(metrics.memory);
    this.metrics.cpuUsage.push(metrics.cpu);
    
    this.recordMetric('memory_heap', metrics.memory.heapUsed, this.agents.size);
    this.recordMetric('cpu_total', metrics.cpu.user + metrics.cpu.system, this.agents.size);
    
    return metrics;
  }

  recordMetric(type, value, agentCount) {
    const stmt = this.db.prepare(
      'INSERT INTO performance_metrics (metric_type, value, agent_count) VALUES (?, ?, ?)'
    );
    stmt.run(type, value, agentCount);
  }

  getPerformanceReport() {
    const metrics = this.db.prepare(`
      SELECT metric_type, AVG(value) as avg_value, MIN(value) as min_value, 
             MAX(value) as max_value, COUNT(*) as samples, agent_count
      FROM performance_metrics
      GROUP BY metric_type, agent_count
      ORDER BY agent_count, metric_type
    `).all();

    return {
      summary: {
        totalAgents: this.agents.size,
        totalTasks: this.tasks.size,
        totalMessages: this.db.prepare('SELECT COUNT(*) as count FROM messages').get().count
      },
      metrics,
      detailed: {
        spawnTime: this.metrics.spawnTime,
        taskDistribution: this.metrics.taskDistributionTime,
        communication: this.metrics.communicationLatency,
        memory: this.metrics.memoryUsage,
        cpu: this.metrics.cpuUsage
      }
    };
  }

  cleanup() {
    this.db.close();
  }
}

describe('Swarm Scaling Integration Tests', () => {
  const dbPath = path.join(__dirname, '../../../tmp/test-scaling.db');
  let orchestrator;

  beforeEach(() => {
    // Ensure tmp directory exists
    const tmpDir = path.dirname(dbPath);
    if (!fs.existsSync(tmpDir)) {
      fs.mkdirSync(tmpDir, { recursive: true });
    }

    // Clean up any existing database
    if (fs.existsSync(dbPath)) {
      fs.unlinkSync(dbPath);
    }

    orchestrator = new SwarmOrchestrator(dbPath);
  });

  afterEach(() => {
    if (orchestrator) {
      orchestrator.cleanup();
    }
    
    if (fs.existsSync(dbPath)) {
      fs.unlinkSync(dbPath);
    }
  });

  describe('Small Scale (1-10 agents)', () => {
    it('should handle single agent efficiently', async () => {
      const spawned = await orchestrator.spawnAgents(1);
      
      expect(spawned).toHaveLength(1);
      expect(spawned[0].type).toBeDefined();
      expect(orchestrator.agents.size).toBe(1);
      
      const metrics = orchestrator.measureResourceUsage();
      expect(metrics.memory.heapUsed).toBeLessThan(100); // Should use less than 100MB
    });

    it('should scale to 5 agents smoothly', async () => {
      const result = await orchestrator.scaleAgents(5);
      
      expect(result.currentCount).toBe(5);
      expect(result.scaleTime).toBeLessThan(1000); // Should take less than 1 second
      
      // Distribute tasks
      const distribution = await orchestrator.distributeTasks(10);
      expect(distribution.avgTasksPerAgent).toBe(2);
      expect(distribution.agentsUsed).toBe(5);
    });

    it('should handle communication between small group', async () => {
      await orchestrator.scaleAgents(5);
      
      const communication = await orchestrator.simulateCommunication(20);
      
      expect(communication.messagesSent).toBe(20);
      expect(communication.avgLatency).toBeLessThan(1); // Less than 1ms per message
      expect(communication.messagesPerSecond).toBeGreaterThan(1000);
    });
  });

  describe('Medium Scale (10-50 agents)', () => {
    it('should scale to 20 agents', async () => {
      const result = await orchestrator.scaleAgents(20);
      
      expect(result.currentCount).toBe(20);
      expect(result.scaleTime).toBeLessThan(2000); // Should take less than 2 seconds
      
      const metrics = orchestrator.measureResourceUsage();
      expect(metrics.agentCount).toBe(20);
    });

    it('should distribute 100 tasks among 20 agents', async () => {
      await orchestrator.scaleAgents(20);
      
      const distribution = await orchestrator.distributeTasks(100);
      
      expect(distribution.tasksCreated).toBe(100);
      expect(distribution.avgTasksPerAgent).toBe(5);
      expect(distribution.distributionTime).toBeLessThan(500);
    });

    it('should handle moderate communication load', async () => {
      await orchestrator.scaleAgents(30);
      
      const communication = await orchestrator.simulateCommunication(500);
      
      expect(communication.messagesSent).toBe(500);
      expect(communication.avgLatency).toBeLessThan(2);
      expect(communication.messagesPerSecond).toBeGreaterThan(500);
    });

    it('should scale down efficiently', async () => {
      // Scale up first
      await orchestrator.scaleAgents(30);
      expect(orchestrator.agents.size).toBe(30);
      
      // Scale down
      const result = await orchestrator.scaleAgents(15);
      
      expect(result.previousCount).toBe(30);
      expect(result.currentCount).toBe(15);
      expect(orchestrator.agents.size).toBe(15);
    });
  });

  describe('Large Scale (50-100 agents)', () => {
    it('should scale to 75 agents', async () => {
      const result = await orchestrator.scaleAgents(75);
      
      expect(result.currentCount).toBe(75);
      expect(result.scaleTime).toBeLessThan(5000); // Should take less than 5 seconds
      
      const metrics = orchestrator.measureResourceUsage();
      expect(metrics.memory.heapUsed).toBeLessThan(500); // Should use less than 500MB
    });

    it('should distribute 500 tasks among 75 agents', async () => {
      await orchestrator.scaleAgents(75);
      
      const distribution = await orchestrator.distributeTasks(500);
      
      expect(distribution.tasksCreated).toBe(500);
      expect(Math.round(distribution.avgTasksPerAgent)).toBeGreaterThanOrEqual(6);
      expect(distribution.agentsUsed).toBe(75);
    });

    it('should handle high-volume communication', async () => {
      await orchestrator.scaleAgents(50);
      
      const communication = await orchestrator.simulateCommunication(2000);
      
      expect(communication.messagesSent).toBe(2000);
      expect(communication.avgLatency).toBeLessThan(5);
      expect(communication.totalTime).toBeLessThan(10000);
    });

    it('should maintain performance at 100 agents', async () => {
      const result = await orchestrator.scaleAgents(100);
      
      expect(result.currentCount).toBe(100);
      
      // Test various operations at scale
      const taskDist = await orchestrator.distributeTasks(1000);
      expect(taskDist.avgTasksPerAgent).toBe(10);
      
      const comm = await orchestrator.simulateCommunication(1000);
      expect(comm.avgLatency).toBeLessThan(10);
      
      const metrics = orchestrator.measureResourceUsage();
      expect(metrics.memory.heapUsed).toBeLessThan(1000); // Less than 1GB
    });
  });

  describe('Stress Testing', () => {
    it('should handle rapid scaling up and down', async () => {
      const scaleSteps = [10, 30, 15, 50, 25, 75, 40, 100, 20, 5];
      const results = [];
      
      for (const target of scaleSteps) {
        const result = await orchestrator.scaleAgents(target);
        results.push(result);
        
        // Verify count
        expect(orchestrator.agents.size).toBe(target);
      }
      
      // All operations should complete
      expect(results).toHaveLength(scaleSteps.length);
    });

    it('should handle concurrent operations', async () => {
      await orchestrator.scaleAgents(50);
      
      // Run multiple operations concurrently
      const operations = await Promise.all([
        orchestrator.distributeTasks(200),
        orchestrator.simulateCommunication(500),
        orchestrator.measureResourceUsage()
      ]);
      
      expect(operations[0].tasksCreated).toBe(200);
      expect(operations[1].messagesSent).toBe(500);
      expect(operations[2].agentCount).toBe(50);
    });

    it('should recover from agent failures', async () => {
      await orchestrator.scaleAgents(50);
      
      // Simulate 10 agent failures
      const agents = Array.from(orchestrator.agents.keys()).slice(0, 10);
      for (const agentId of agents) {
        orchestrator.agents.delete(agentId);
      }
      
      expect(orchestrator.agents.size).toBe(40);
      
      // Should still function with remaining agents
      const dist = await orchestrator.distributeTasks(100);
      expect(dist.agentsUsed).toBeLessThanOrEqual(40);
    });
  });

  describe('Performance Metrics', () => {
    it('should track scaling performance', async () => {
      // Scale to different levels and measure
      const levels = [10, 25, 50, 75, 100];
      
      for (const level of levels) {
        await orchestrator.scaleAgents(level);
        await orchestrator.distributeTasks(level * 10);
        await orchestrator.simulateCommunication(level * 20);
        orchestrator.measureResourceUsage();
      }
      
      const report = orchestrator.getPerformanceReport();
      
      expect(report.summary.totalAgents).toBe(100);
      expect(report.metrics.length).toBeGreaterThan(0);
      
      // Verify metrics are tracked for each scale level
      const scaleMetrics = report.metrics.filter(m => m.metric_type === 'scale_time');
      expect(scaleMetrics.length).toBeGreaterThanOrEqual(levels.length);
      
      // Performance should degrade gracefully
      const memoryMetrics = report.metrics
        .filter(m => m.metric_type === 'memory_heap')
        .sort((a, b) => a.agent_count - b.agent_count);
      
      for (let i = 1; i < memoryMetrics.length; i++) {
        expect(memoryMetrics[i].avg_value).toBeGreaterThanOrEqual(memoryMetrics[i-1].avg_value);
      }
    });

    it('should identify bottlenecks', async () => {
      // Create bottleneck scenario
      await orchestrator.scaleAgents(100);
      
      // Heavy task load
      const taskPerf = await orchestrator.distributeTasks(5000);
      
      // Heavy communication
      const commPerf = await orchestrator.simulateCommunication(10000);
      
      const report = orchestrator.getPerformanceReport();
      
      // Distribution time should increase with load
      expect(taskPerf.distributionTime).toBeGreaterThan(100);
      
      // Communication latency should increase
      expect(commPerf.avgLatency).toBeGreaterThan(1);
      
      // Memory usage should be significant
      const finalMetrics = orchestrator.measureResourceUsage();
      expect(finalMetrics.memory.heapUsed).toBeGreaterThan(50);
    });

    it('should provide scaling recommendations', () => {
      const report = orchestrator.getPerformanceReport();
      
      // Analyze spawn time efficiency
      const spawnTimes = report.detailed.spawnTime;
      if (spawnTimes.length > 0) {
        const avgSpawnTimePerAgent = spawnTimes.reduce((sum, st) => 
          sum + (st.time / st.count), 0) / spawnTimes.length;
        
        // Should spawn agents efficiently
        expect(avgSpawnTimePerAgent).toBeLessThan(100); // Less than 100ms per agent
      }
    });
  });
});