/**
 * Performance Benchmark Scenarios
 * Tests various workload patterns and measures performance
 */

import { describe, it, expect, beforeAll, afterAll } from '@jest/globals';
import { performance } from 'perf_hooks';
import Database from 'better-sqlite3';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Performance benchmark harness
class BenchmarkHarness {
  constructor() {
    this.results = {
      scenarios: [],
      summary: {
        totalDuration: 0,
        avgAgentSpawnTime: 0,
        avgTaskCompletionTime: 0,
        avgMessageLatency: 0,
        peakMemoryUsage: 0,
        maxConcurrentAgents: 0
      }
    };
  }

  async runScenario(name, config, testFn) {
    console.log(`\n🚀 Running scenario: ${name}`);
    
    const scenario = {
      name,
      config,
      metrics: {},
      startTime: performance.now(),
      memoryStart: process.memoryUsage()
    };

    try {
      // Run the test function
      const result = await testFn();
      
      scenario.endTime = performance.now();
      scenario.duration = scenario.endTime - scenario.startTime;
      scenario.memoryEnd = process.memoryUsage();
      scenario.memoryDelta = {
        heapUsed: (scenario.memoryEnd.heapUsed - scenario.memoryStart.heapUsed) / 1024 / 1024,
        external: (scenario.memoryEnd.external - scenario.memoryStart.external) / 1024 / 1024
      };
      scenario.result = result;
      scenario.status = 'passed';
      
      // Update summary
      this.results.totalDuration += scenario.duration;
      this.results.summary.peakMemoryUsage = Math.max(
        this.results.summary.peakMemoryUsage,
        scenario.memoryEnd.heapUsed / 1024 / 1024
      );
      
    } catch (error) {
      scenario.status = 'failed';
      scenario.error = error.message;
    }

    this.results.scenarios.push(scenario);
    return scenario;
  }

  generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      environment: {
        nodeVersion: process.version,
        platform: process.platform,
        cpus: require('os').cpus().length,
        totalMemory: require('os').totalmem() / 1024 / 1024 / 1024 + ' GB'
      },
      results: this.results,
      recommendations: this.generateRecommendations()
    };

    return report;
  }

  generateRecommendations() {
    const recommendations = [];
    
    // Analyze spawn times
    const spawnScenarios = this.results.scenarios.filter(s => s.name.includes('spawn'));
    if (spawnScenarios.length > 0) {
      const avgSpawnTime = spawnScenarios.reduce((sum, s) => sum + s.duration, 0) / spawnScenarios.length;
      if (avgSpawnTime > 5000) {
        recommendations.push({
          type: 'performance',
          area: 'agent_spawning',
          message: 'Agent spawning is slow. Consider batch spawning or pre-warming agents.',
          severity: 'high'
        });
      }
    }

    // Analyze memory usage
    if (this.results.summary.peakMemoryUsage > 1024) {
      recommendations.push({
        type: 'resource',
        area: 'memory',
        message: 'High memory usage detected. Consider implementing agent pooling or memory limits.',
        severity: 'medium'
      });
    }

    return recommendations;
  }
}

// Test scenarios
describe('Hive Mind Performance Benchmarks', () => {
  const dbPath = path.join(__dirname, '../../../tmp/benchmark.db');
  let db;
  let harness;

  beforeAll(() => {
    // Setup
    const tmpDir = path.dirname(dbPath);
    if (!fs.existsSync(tmpDir)) {
      fs.mkdirSync(tmpDir, { recursive: true });
    }

    if (fs.existsSync(dbPath)) {
      fs.unlinkSync(dbPath);
    }

    db = new Database(dbPath);
    harness = new BenchmarkHarness();

    // Initialize schema
    db.exec(`
      CREATE TABLE agents (
        id TEXT PRIMARY KEY,
        type TEXT,
        status TEXT,
        created_at INTEGER
      );
      CREATE TABLE tasks (
        id TEXT PRIMARY KEY,
        status TEXT,
        assigned_to TEXT,
        created_at INTEGER,
        completed_at INTEGER
      );
      CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        from_agent TEXT,
        to_agent TEXT,
        timestamp INTEGER
      );
    `);
  });

  afterAll(() => {
    if (db) {
      db.close();
    }
    
    // Generate and save report
    const report = harness.generateReport();
    const reportPath = path.join(__dirname, '../../../tmp/performance-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    console.log('\n📊 Performance Report Summary:');
    console.log(`Total Duration: ${(report.results.totalDuration / 1000).toFixed(2)}s`);
    console.log(`Peak Memory: ${report.results.summary.peakMemoryUsage.toFixed(2)} MB`);
    console.log(`Scenarios Run: ${report.results.scenarios.length}`);
    console.log(`Report saved to: ${reportPath}`);

    if (fs.existsSync(dbPath)) {
      fs.unlinkSync(dbPath);
    }
  });

  describe('Agent Spawning Performance', () => {
    it('should spawn 10 agents quickly', async () => {
      await harness.runScenario('spawn_10_agents', {
        agentCount: 10,
        agentType: 'mixed'
      }, async () => {
        const agents = [];
        const insert = db.prepare('INSERT INTO agents (id, type, status, created_at) VALUES (?, ?, ?, ?)');
        
        const start = performance.now();
        
        for (let i = 0; i < 10; i++) {
          const agent = {
            id: `agent-${i}`,
            type: ['researcher', 'coder', 'analyst'][i % 3],
            status: 'active',
            created_at: Date.now()
          };
          insert.run(agent.id, agent.type, agent.status, agent.created_at);
          agents.push(agent);
        }
        
        const duration = performance.now() - start;
        
        expect(agents).toHaveLength(10);
        expect(duration).toBeLessThan(100); // Should take less than 100ms
        
        return { agents: agents.length, duration };
      });
    });

    it('should spawn 100 agents efficiently', async () => {
      await harness.runScenario('spawn_100_agents', {
        agentCount: 100,
        batchSize: 25
      }, async () => {
        const insert = db.prepare('INSERT INTO agents (id, type, status, created_at) VALUES (?, ?, ?, ?)');
        const batchInsert = db.transaction((agents) => {
          for (const agent of agents) {
            insert.run(agent.id, agent.type, agent.status, agent.created_at);
          }
        });
        
        const start = performance.now();
        const batches = 4;
        const batchSize = 25;
        
        for (let batch = 0; batch < batches; batch++) {
          const agents = [];
          for (let i = 0; i < batchSize; i++) {
            agents.push({
              id: `agent-${batch * batchSize + i}`,
              type: 'worker',
              status: 'active',
              created_at: Date.now()
            });
          }
          batchInsert(agents);
        }
        
        const duration = performance.now() - start;
        const count = db.prepare('SELECT COUNT(*) as count FROM agents').get().count;
        
        expect(count).toBeGreaterThanOrEqual(100);
        expect(duration).toBeLessThan(500); // Should take less than 500ms
        
        return { agents: count, duration, avgPerAgent: duration / 100 };
      });
    });

    it('should handle rapid agent spawning bursts', async () => {
      await harness.runScenario('spawn_burst', {
        burstSize: 50,
        burstCount: 5,
        delayBetweenBursts: 100
      }, async () => {
        const results = [];
        
        for (let burst = 0; burst < 5; burst++) {
          const burstStart = performance.now();
          
          // Spawn 50 agents rapidly
          const agents = [];
          for (let i = 0; i < 50; i++) {
            agents.push({
              id: `burst-${burst}-agent-${i}`,
              type: 'burst-worker',
              status: 'active',
              created_at: Date.now()
            });
          }
          
          // Batch insert
          const insert = db.prepare('INSERT INTO agents (id, type, status, created_at) VALUES (?, ?, ?, ?)');
          const transaction = db.transaction((agents) => {
            for (const agent of agents) {
              insert.run(agent.id, agent.type, agent.status, agent.created_at);
            }
          });
          
          transaction(agents);
          
          const burstDuration = performance.now() - burstStart;
          results.push({
            burst: burst + 1,
            duration: burstDuration,
            agentsPerSecond: (50 / burstDuration) * 1000
          });
          
          // Delay between bursts
          await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        const avgBurstTime = results.reduce((sum, r) => sum + r.duration, 0) / results.length;
        expect(avgBurstTime).toBeLessThan(200);
        
        return { bursts: results, avgBurstTime };
      });
    });
  });

  describe('Task Distribution Performance', () => {
    it('should distribute 1000 tasks among 50 agents', async () => {
      await harness.runScenario('distribute_1000_tasks', {
        taskCount: 1000,
        agentCount: 50
      }, async () => {
        // Ensure we have agents
        const agentCount = db.prepare('SELECT COUNT(*) as count FROM agents').get().count;
        if (agentCount < 50) {
          throw new Error('Not enough agents for test');
        }
        
        // Get agent IDs
        const agents = db.prepare('SELECT id FROM agents LIMIT 50').all();
        
        const start = performance.now();
        
        // Create and distribute tasks
        const insert = db.prepare('INSERT INTO tasks (id, status, assigned_to, created_at) VALUES (?, ?, ?, ?)');
        const batchInsert = db.transaction((tasks) => {
          for (const task of tasks) {
            insert.run(task.id, task.status, task.assigned_to, task.created_at);
          }
        });
        
        const tasks = [];
        for (let i = 0; i < 1000; i++) {
          tasks.push({
            id: `task-${i}`,
            status: 'pending',
            assigned_to: agents[i % agents.length].id,
            created_at: Date.now()
          });
        }
        
        batchInsert(tasks);
        
        const duration = performance.now() - start;
        
        // Verify distribution
        const distribution = db.prepare(`
          SELECT assigned_to, COUNT(*) as task_count 
          FROM tasks 
          GROUP BY assigned_to
        `).all();
        
        const avgTasksPerAgent = 1000 / 50;
        const maxDeviation = Math.max(...distribution.map(d => 
          Math.abs(d.task_count - avgTasksPerAgent)
        ));
        
        expect(maxDeviation).toBeLessThanOrEqual(1); // Fair distribution
        expect(duration).toBeLessThan(1000); // Less than 1 second
        
        return { 
          duration, 
          tasksPerSecond: (1000 / duration) * 1000,
          distribution: distribution.length 
        };
      });
    });

    it('should handle task completion updates efficiently', async () => {
      await harness.runScenario('task_completion', {
        taskCount: 500,
        completionRate: 0.8
      }, async () => {
        // Get pending tasks
        const pendingTasks = db.prepare('SELECT id FROM tasks WHERE status = ?').all('pending');
        const tasksToComplete = pendingTasks.slice(0, Math.floor(pendingTasks.length * 0.8));
        
        const start = performance.now();
        
        // Update tasks to completed
        const update = db.prepare('UPDATE tasks SET status = ?, completed_at = ? WHERE id = ?');
        const batchUpdate = db.transaction((tasks) => {
          const now = Date.now();
          for (const task of tasks) {
            update.run('completed', now, task.id);
          }
        });
        
        batchUpdate(tasksToComplete);
        
        const duration = performance.now() - start;
        const completedCount = db.prepare('SELECT COUNT(*) as count FROM tasks WHERE status = ?').get('completed').count;
        
        expect(duration).toBeLessThan(500);
        
        return {
          duration,
          completed: completedCount,
          updatesPerSecond: (tasksToComplete.length / duration) * 1000
        };
      });
    });
  });

  describe('Communication Performance', () => {
    it('should handle 10,000 messages efficiently', async () => {
      await harness.runScenario('message_flood', {
        messageCount: 10000,
        senderCount: 20
      }, async () => {
        const agents = db.prepare('SELECT id FROM agents LIMIT 20').all();
        if (agents.length < 20) {
          throw new Error('Not enough agents for test');
        }
        
        const start = performance.now();
        
        // Generate messages
        const insert = db.prepare('INSERT INTO messages (from_agent, to_agent, timestamp) VALUES (?, ?, ?)');
        const batchInsert = db.transaction((messages) => {
          for (const msg of messages) {
            insert.run(msg.from, msg.to, msg.timestamp);
          }
        });
        
        const messages = [];
        const timestamp = Date.now();
        
        for (let i = 0; i < 10000; i++) {
          const fromIdx = i % agents.length;
          const toIdx = (i + 1) % agents.length;
          
          messages.push({
            from: agents[fromIdx].id,
            to: agents[toIdx].id,
            timestamp: timestamp + i
          });
        }
        
        // Insert in batches of 1000
        for (let i = 0; i < messages.length; i += 1000) {
          batchInsert(messages.slice(i, i + 1000));
        }
        
        const duration = performance.now() - start;
        const messageCount = db.prepare('SELECT COUNT(*) as count FROM messages').get().count;
        
        expect(duration).toBeLessThan(2000); // Less than 2 seconds
        
        return {
          duration,
          messages: messageCount,
          messagesPerSecond: (10000 / duration) * 1000,
          avgLatency: duration / 10000
        };
      });
    });

    it('should query message history quickly', async () => {
      await harness.runScenario('message_query', {
        queryTypes: ['by_sender', 'by_conversation', 'recent']
      }, async () => {
        const queries = [
          {
            name: 'by_sender',
            sql: 'SELECT * FROM messages WHERE from_agent = ? LIMIT 100',
            params: ['agent-0']
          },
          {
            name: 'by_conversation',
            sql: 'SELECT * FROM messages WHERE (from_agent = ? AND to_agent = ?) OR (from_agent = ? AND to_agent = ?) ORDER BY timestamp LIMIT 100',
            params: ['agent-0', 'agent-1', 'agent-1', 'agent-0']
          },
          {
            name: 'recent',
            sql: 'SELECT * FROM messages ORDER BY timestamp DESC LIMIT 100',
            params: []
          }
        ];
        
        const results = [];
        
        for (const query of queries) {
          const stmt = db.prepare(query.sql);
          
          const start = performance.now();
          const rows = stmt.all(...query.params);
          const duration = performance.now() - start;
          
          results.push({
            query: query.name,
            duration,
            rowCount: rows.length
          });
          
          expect(duration).toBeLessThan(50); // Each query under 50ms
        }
        
        return { queries: results };
      });
    });
  });

  describe('Concurrent Operations', () => {
    it('should handle mixed workload efficiently', async () => {
      await harness.runScenario('mixed_workload', {
        duration: 5000,
        operations: ['spawn', 'task', 'message', 'query']
      }, async () => {
        const startTime = Date.now();
        const endTime = startTime + 5000;
        const operations = {
          spawns: 0,
          tasks: 0,
          messages: 0,
          queries: 0
        };
        
        // Prepare statements
        const spawnStmt = db.prepare('INSERT INTO agents (id, type, status, created_at) VALUES (?, ?, ?, ?)');
        const taskStmt = db.prepare('INSERT INTO tasks (id, status, assigned_to, created_at) VALUES (?, ?, ?, ?)');
        const messageStmt = db.prepare('INSERT INTO messages (from_agent, to_agent, timestamp) VALUES (?, ?, ?)');
        const queryStmt = db.prepare('SELECT COUNT(*) as count FROM agents WHERE status = ?');
        
        const agents = db.prepare('SELECT id FROM agents LIMIT 10').all();
        
        while (Date.now() < endTime) {
          const op = Math.floor(Math.random() * 4);
          
          switch (op) {
            case 0: // Spawn
              spawnStmt.run(
                `mixed-agent-${operations.spawns}`,
                'worker',
                'active',
                Date.now()
              );
              operations.spawns++;
              break;
              
            case 1: // Task
              if (agents.length > 0) {
                taskStmt.run(
                  `mixed-task-${operations.tasks}`,
                  'pending',
                  agents[operations.tasks % agents.length].id,
                  Date.now()
                );
                operations.tasks++;
              }
              break;
              
            case 2: // Message
              if (agents.length >= 2) {
                messageStmt.run(
                  agents[0].id,
                  agents[1].id,
                  Date.now()
                );
                operations.messages++;
              }
              break;
              
            case 3: // Query
              queryStmt.get('active');
              operations.queries++;
              break;
          }
          
          // Small delay to prevent CPU saturation
          if (operations.spawns + operations.tasks + operations.messages + operations.queries > 0 && 
              (operations.spawns + operations.tasks + operations.messages + operations.queries) % 100 === 0) {
            await new Promise(resolve => setTimeout(resolve, 1));
          }
        }
        
        const totalOps = Object.values(operations).reduce((sum, count) => sum + count, 0);
        const opsPerSecond = (totalOps / 5000) * 1000;
        
        expect(opsPerSecond).toBeGreaterThan(100); // At least 100 ops/sec
        
        return {
          operations,
          totalOps,
          opsPerSecond
        };
      });
    });
  });

  describe('Scalability Limits', () => {
    it('should identify maximum agent capacity', async () => {
      await harness.runScenario('max_capacity', {
        targetAgents: 1000,
        memoryLimit: 512 // MB
      }, async () => {
        let maxAgents = 0;
        const batchSize = 100;
        const insert = db.prepare('INSERT INTO agents (id, type, status, created_at) VALUES (?, ?, ?, ?)');
        
        try {
          for (let batch = 0; batch < 10; batch++) {
            const batchTransaction = db.transaction((agents) => {
              for (const agent of agents) {
                insert.run(agent.id, agent.type, agent.status, agent.created_at);
              }
            });
            
            const agents = [];
            for (let i = 0; i < batchSize; i++) {
              agents.push({
                id: `capacity-agent-${batch * batchSize + i}`,
                type: 'worker',
                status: 'active',
                created_at: Date.now()
              });
            }
            
            batchTransaction(agents);
            maxAgents += batchSize;
            
            // Check memory usage
            const memUsage = process.memoryUsage().heapUsed / 1024 / 1024;
            if (memUsage > 512) {
              break;
            }
          }
        } catch (error) {
          // Reached limit
        }
        
        const finalCount = db.prepare('SELECT COUNT(*) as count FROM agents').get().count;
        const memoryPerAgent = (process.memoryUsage().heapUsed / 1024 / 1024) / finalCount;
        
        return {
          maxAgents: finalCount,
          memoryPerAgent,
          theoreticalMax: Math.floor(512 / memoryPerAgent)
        };
      });
    });
  });
});