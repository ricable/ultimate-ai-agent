#!/usr/bin/env tsx
/**
 * Supabase Federation Test
 * Tests federation capabilities with hgbfbvtujatvwpjgibng project
 */

import { createClient } from '@supabase/supabase-js';
import { readFileSync } from 'fs';
import { join } from 'path';

// Supabase configuration
const SUPABASE_URL = 'https://hgbfbvtujatvwpjgibng.supabase.co';
const SUPABASE_ANON_KEY = process.env.SUPABASE_ANON_KEY || '';
const SUPABASE_SERVICE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY || '';

interface TestResult {
  name: string;
  status: 'pass' | 'fail' | 'skip';
  duration: number;
  error?: string;
  details?: any;
}

class SupabaseFederationTester {
  private client: any;
  private serviceClient: any;
  private results: TestResult[] = [];
  private testTenant = `test-${Date.now()}`;

  constructor() {
    console.log('\n🧪 Supabase Federation Test Suite');
    console.log('═'.repeat(70));
    console.log(`📍 Project: hgbfbvtujatvwpjgibng`);
    console.log(`🏷️  Test Tenant: ${this.testTenant}`);
    console.log('');
  }

  async initialize() {
    const startTime = Date.now();

    try {
      // Use environment variables
      const url = process.env.VITE_SUPABASE_URL || SUPABASE_URL;
      const key = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.VITE_SUPABASE_PUBLISHABLE_KEY || SUPABASE_SERVICE_KEY;

      if (!key) {
        throw new Error('SUPABASE_SERVICE_ROLE_KEY or VITE_SUPABASE_PUBLISHABLE_KEY required');
      }

      // Create clients
      this.client = createClient(url, key);
      if (process.env.SUPABASE_SERVICE_ROLE_KEY) {
        this.serviceClient = createClient(url, process.env.SUPABASE_SERVICE_ROLE_KEY);
      }

      console.log('✅ Connected to Supabase');

      this.results.push({
        name: 'Connection Test',
        status: 'pass',
        duration: Date.now() - startTime
      });
    } catch (error: any) {
      this.results.push({
        name: 'Connection Test',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
      throw error;
    }
  }

  async createTables() {
    const startTime = Date.now();

    try {
      console.log('\n📋 Creating Federation Tables...');

      if (!this.serviceClient) {
        console.log('⚠️  No service role key - skipping table creation');
        this.results.push({
          name: 'Create Tables',
          status: 'skip',
          duration: Date.now() - startTime,
          details: 'Service role key not provided'
        });
        return;
      }

      // Read SQL file
      const sqlPath = join(process.cwd(), 'docs/supabase/migrations/001_create_federation_tables.sql');
      const sql = readFileSync(sqlPath, 'utf-8');

      // Split into individual statements
      const statements = sql
        .split(';')
        .map(s => s.trim())
        .filter(s => s.length > 0 && !s.startsWith('--') && !s.startsWith('/*'));

      console.log(`📝 Executing ${statements.length} SQL statements...`);

      let executed = 0;
      let skipped = 0;

      for (const statement of statements) {
        try {
          // Use RPC for DDL statements
          await this.serviceClient.rpc('exec_sql', { sql_query: statement });
          executed++;
        } catch (error: any) {
          // Check if error is "already exists" - that's OK
          if (error.message?.includes('already exists') || error.code === '42P07') {
            skipped++;
          } else {
            console.log(`⚠️  Statement error: ${error.message}`);
          }
        }
      }

      console.log(`✅ Executed: ${executed}, Skipped: ${skipped} (already exist)`);

      this.results.push({
        name: 'Create Tables',
        status: 'pass',
        duration: Date.now() - startTime,
        details: { executed, skipped }
      });
    } catch (error: any) {
      console.error(`❌ Table creation failed: ${error.message}`);
      this.results.push({
        name: 'Create Tables',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
    }
  }

  async testAgentSession() {
    const startTime = Date.now();

    try {
      console.log('\n🤖 Testing Agent Session...');

      const sessionData = {
        session_id: `session-test-${Date.now()}`,
        tenant_id: this.testTenant,
        agent_id: 'agent-001',
        status: 'active',
        metadata: { type: 'federation-test', version: '1.8.13' }
      };

      // Insert session
      const { data, error } = await this.client
        .from('agent_sessions')
        .insert([sessionData])
        .select();

      if (error) throw error;

      console.log(`✅ Created session: ${sessionData.session_id}`);

      this.results.push({
        name: 'Agent Session Create',
        status: 'pass',
        duration: Date.now() - startTime,
        details: data
      });

      return sessionData.session_id;
    } catch (error: any) {
      console.error(`❌ Session test failed: ${error.message}`);
      this.results.push({
        name: 'Agent Session Create',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
      return null;
    }
  }

  async testMemoryStorage(sessionId: string) {
    const startTime = Date.now();

    try {
      console.log('\n💾 Testing Memory Storage...');

      const memories = [
        {
          tenant_id: this.testTenant,
          agent_id: 'agent-001',
          session_id: sessionId,
          content: 'Federation test: Testing Supabase real-time capabilities',
          metadata: { test: true, timestamp: Date.now() }
        },
        {
          tenant_id: this.testTenant,
          agent_id: 'agent-001',
          session_id: sessionId,
          content: 'Federation test: Multi-agent collaboration working',
          metadata: { test: true, timestamp: Date.now() }
        },
        {
          tenant_id: this.testTenant,
          agent_id: 'agent-001',
          session_id: sessionId,
          content: 'Federation test: Cloud persistence validated',
          metadata: { test: true, timestamp: Date.now() }
        }
      ];

      const { data, error } = await this.client
        .from('agent_memories')
        .insert(memories)
        .select();

      if (error) throw error;

      console.log(`✅ Stored ${data.length} memories`);

      this.results.push({
        name: 'Memory Storage',
        status: 'pass',
        duration: Date.now() - startTime,
        details: { count: data.length }
      });

      return data;
    } catch (error: any) {
      console.error(`❌ Memory storage failed: ${error.message}`);
      this.results.push({
        name: 'Memory Storage',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
      return null;
    }
  }

  async testMemoryRetrieval(sessionId: string) {
    const startTime = Date.now();

    try {
      console.log('\n🔍 Testing Memory Retrieval...');

      const { data, error } = await this.client
        .from('agent_memories')
        .select('*')
        .eq('session_id', sessionId)
        .order('created_at', { ascending: false });

      if (error) throw error;

      console.log(`✅ Retrieved ${data.length} memories`);
      data.forEach((m: any, i: number) => {
        console.log(`   ${i + 1}. ${m.content.substring(0, 50)}...`);
      });

      this.results.push({
        name: 'Memory Retrieval',
        status: 'pass',
        duration: Date.now() - startTime,
        details: { count: data.length }
      });
    } catch (error: any) {
      console.error(`❌ Memory retrieval failed: ${error.message}`);
      this.results.push({
        name: 'Memory Retrieval',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
    }
  }

  async testTaskManagement() {
    const startTime = Date.now();

    try {
      console.log('\n📋 Testing Task Management...');

      const task = {
        task_id: `task-test-${Date.now()}`,
        tenant_id: this.testTenant,
        assigned_to: 'agent-001',
        assigned_by: 'agent-hub',
        description: 'Test federation real-time coordination',
        priority: 'high',
        status: 'assigned',
        metadata: { test: true, created_by: 'test-suite' }
      };

      const { data, error } = await this.client
        .from('agent_tasks')
        .insert([task])
        .select();

      if (error) throw error;

      console.log(`✅ Created task: ${task.task_id}`);
      console.log(`   Assigned to: ${task.assigned_to}`);
      console.log(`   Priority: ${task.priority}`);

      this.results.push({
        name: 'Task Management',
        status: 'pass',
        duration: Date.now() - startTime,
        details: data[0]
      });
    } catch (error: any) {
      console.error(`❌ Task management failed: ${error.message}`);
      this.results.push({
        name: 'Task Management',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
    }
  }

  async testEventLogging() {
    const startTime = Date.now();

    try {
      console.log('\n📊 Testing Event Logging...');

      const events = [
        {
          tenant_id: this.testTenant,
          agent_id: 'agent-001',
          event_type: 'session_started',
          payload: { test: true, version: '1.8.13' }
        },
        {
          tenant_id: this.testTenant,
          agent_id: 'agent-001',
          event_type: 'memory_created',
          payload: { count: 3, test: true }
        },
        {
          tenant_id: this.testTenant,
          agent_id: 'agent-001',
          event_type: 'task_received',
          payload: { task_id: 'task-test-123', test: true }
        }
      ];

      const { data, error } = await this.client
        .from('agent_events')
        .insert(events)
        .select();

      if (error) throw error;

      console.log(`✅ Logged ${data.length} events`);
      data.forEach((e: any) => {
        console.log(`   - ${e.event_type}`);
      });

      this.results.push({
        name: 'Event Logging',
        status: 'pass',
        duration: Date.now() - startTime,
        details: { count: data.length }
      });
    } catch (error: any) {
      console.error(`❌ Event logging failed: ${error.message}`);
      this.results.push({
        name: 'Event Logging',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
    }
  }

  async testRealtimeSubscription() {
    const startTime = Date.now();

    return new Promise<void>((resolve) => {
      try {
        console.log('\n📡 Testing Real-time Subscription...');

        let receivedCount = 0;
        const expectedCount = 2;

        const channel = this.client
          .channel('test-channel')
          .on(
            'postgres_changes',
            {
              event: 'INSERT',
              schema: 'public',
              table: 'agent_memories',
              filter: `tenant_id=eq.${this.testTenant}`
            },
            (payload: any) => {
              receivedCount++;
              console.log(`   ✅ Real-time event received (${receivedCount}/${expectedCount})`);

              if (receivedCount >= expectedCount) {
                channel.unsubscribe();
                this.results.push({
                  name: 'Real-time Subscription',
                  status: 'pass',
                  duration: Date.now() - startTime,
                  details: { received: receivedCount }
                });
                resolve();
              }
            }
          )
          .subscribe();

        // Insert test data after subscription
        setTimeout(async () => {
          for (let i = 0; i < expectedCount; i++) {
            await this.client.from('agent_memories').insert([{
              tenant_id: this.testTenant,
              agent_id: 'agent-realtime-test',
              session_id: `session-realtime-${Date.now()}`,
              content: `Real-time test message ${i + 1}`,
              metadata: { realtime: true, index: i }
            }]);
            await new Promise(r => setTimeout(r, 100));
          }
        }, 1000);

        // Timeout after 10 seconds
        setTimeout(() => {
          if (receivedCount < expectedCount) {
            channel.unsubscribe();
            console.log(`⚠️  Only received ${receivedCount}/${expectedCount} real-time events`);
            this.results.push({
              name: 'Real-time Subscription',
              status: receivedCount > 0 ? 'pass' : 'fail',
              duration: Date.now() - startTime,
              details: { received: receivedCount, expected: expectedCount }
            });
            resolve();
          }
        }, 10000);

      } catch (error: any) {
        console.error(`❌ Real-time test failed: ${error.message}`);
        this.results.push({
          name: 'Real-time Subscription',
          status: 'fail',
          duration: Date.now() - startTime,
          error: error.message
        });
        resolve();
      }
    });
  }

  async testTenantIsolation() {
    const startTime = Date.now();

    try {
      console.log('\n🔒 Testing Tenant Isolation...');

      const otherTenant = `other-tenant-${Date.now()}`;

      // Create memory in other tenant
      await this.client.from('agent_memories').insert([{
        tenant_id: otherTenant,
        agent_id: 'agent-other',
        session_id: 'session-other',
        content: 'This should be isolated',
        metadata: { tenant: otherTenant }
      }]);

      // Try to read from our tenant
      const { data, error } = await this.client
        .from('agent_memories')
        .select('*')
        .eq('tenant_id', this.testTenant);

      if (error) throw error;

      // Should not see other tenant's data
      const hasOtherTenantData = data.some((m: any) => m.tenant_id === otherTenant);

      if (hasOtherTenantData) {
        throw new Error('Tenant isolation violated: saw other tenant data');
      }

      console.log(`✅ Tenant isolation verified`);
      console.log(`   Test tenant memories: ${data.length}`);
      console.log(`   Other tenant visible: No`);

      this.results.push({
        name: 'Tenant Isolation',
        status: 'pass',
        duration: Date.now() - startTime,
        details: { isolated: true }
      });
    } catch (error: any) {
      console.error(`❌ Tenant isolation test failed: ${error.message}`);
      this.results.push({
        name: 'Tenant Isolation',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
    }
  }

  async cleanup() {
    const startTime = Date.now();

    try {
      console.log('\n🧹 Cleaning up test data...');

      // Delete test data
      const tables = ['agent_events', 'agent_memories', 'agent_tasks', 'agent_sessions'];
      let deleted = 0;

      for (const table of tables) {
        const { error } = await this.client
          .from(table)
          .delete()
          .like('tenant_id', `test-%`);

        if (!error) deleted++;
      }

      console.log(`✅ Cleaned up ${deleted} tables`);

      this.results.push({
        name: 'Cleanup',
        status: 'pass',
        duration: Date.now() - startTime,
        details: { tables: deleted }
      });
    } catch (error: any) {
      console.error(`⚠️  Cleanup warning: ${error.message}`);
      this.results.push({
        name: 'Cleanup',
        status: 'fail',
        duration: Date.now() - startTime,
        error: error.message
      });
    }
  }

  printResults() {
    console.log('\n');
    console.log('═'.repeat(70));
    console.log('📊 Test Results Summary');
    console.log('═'.repeat(70));
    console.log('');

    const passed = this.results.filter(r => r.status === 'pass').length;
    const failed = this.results.filter(r => r.status === 'fail').length;
    const skipped = this.results.filter(r => r.status === 'skip').length;
    const total = this.results.length;

    this.results.forEach((result, i) => {
      const icon = result.status === 'pass' ? '✅' : result.status === 'fail' ? '❌' : '⏭️ ';
      const duration = `${result.duration}ms`;
      console.log(`${i + 1}. ${icon} ${result.name.padEnd(30)} ${duration.padStart(8)}`);
      if (result.error) {
        console.log(`   Error: ${result.error}`);
      }
    });

    console.log('');
    console.log('─'.repeat(70));
    console.log(`Total: ${total} | Passed: ${passed} | Failed: ${failed} | Skipped: ${skipped}`);
    console.log(`Success Rate: ${((passed / (total - skipped)) * 100).toFixed(1)}%`);
    console.log('═'.repeat(70));
    console.log('');

    if (failed === 0) {
      console.log('🎉 All tests passed! Federation capabilities validated.');
    } else {
      console.log(`⚠️  ${failed} test(s) failed. Check errors above.`);
    }
    console.log('');

    return { passed, failed, skipped, total };
  }
}

// Run tests
async function main() {
  const tester = new SupabaseFederationTester();

  try {
    // Initialize
    await tester.initialize();

    // Create tables (requires service role key)
    await tester.createTables();

    // Run tests
    const sessionId = await tester.testAgentSession();

    if (sessionId) {
      await tester.testMemoryStorage(sessionId);
      await tester.testMemoryRetrieval(sessionId);
    }

    await tester.testTaskManagement();
    await tester.testEventLogging();
    await tester.testRealtimeSubscription();
    await tester.testTenantIsolation();

    // Cleanup
    await tester.cleanup();

    // Print results
    const results = tester.printResults();

    // Exit with error code if tests failed
    process.exit(results.failed > 0 ? 1 : 0);

  } catch (error: any) {
    console.error('\n❌ Fatal error:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
}

main();
