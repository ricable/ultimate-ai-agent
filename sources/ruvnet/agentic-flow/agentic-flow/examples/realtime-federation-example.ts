/**
 * Real-Time Federation Example
 *
 * Demonstrates how to use Supabase real-time capabilities for:
 * - Live agent coordination
 * - Instant memory sharing
 * - Collaborative task execution
 * - Real-time presence tracking
 */

import { createRealtimeHub } from '../src/federation/integrations/realtime-federation.js';

/**
 * Example 1: Multi-Agent Research Team
 *
 * Three agents collaboratively research a topic:
 * - Researcher Agent: Gathers information
 * - Analyst Agent: Analyzes findings
 * - Writer Agent: Synthesizes into report
 */
async function multiAgentResearchExample() {
  console.log('\n📚 Example 1: Multi-Agent Research Team\n');

  // Create three agents in same tenant
  const researcher = createRealtimeHub('researcher-001', 'research-team');
  const analyst = createRealtimeHub('analyst-001', 'research-team');
  const writer = createRealtimeHub('writer-001', 'research-team');

  // Initialize all agents
  await Promise.all([
    researcher.initialize(),
    analyst.initialize(),
    writer.initialize(),
  ]);

  // Set up event handlers

  // Researcher shares findings with team
  researcher.on('message:task_assignment', async (message: any) => {
    console.log(`🔍 Researcher received task: ${message.payload.description}`);
    await researcher.updateStatus('busy', message.payload.description);

    // Simulate research
    setTimeout(async () => {
      await researcher.shareKnowledge('Found 5 relevant papers on AI safety', {
        papers: ['Paper1', 'Paper2', 'Paper3', 'Paper4', 'Paper5'],
      });
      await researcher.reportTaskComplete(message.payload.task_id, {
        papers_found: 5,
        confidence: 0.85,
      });
      await researcher.updateStatus('online');
    }, 2000);
  });

  // Analyst processes research findings
  analyst.on('message:share_knowledge', async (message: any) => {
    if (message.from_agent === 'researcher-001') {
      console.log(`📊 Analyst processing: ${message.payload.knowledge}`);
      await analyst.updateStatus('busy', 'Analyzing research data');

      setTimeout(async () => {
        await analyst.shareKnowledge(
          'Analysis complete: High confidence in safety frameworks',
          {
            insights: ['Insight 1', 'Insight 2', 'Insight 3'],
            recommendations: ['Rec 1', 'Rec 2'],
          }
        );
        await analyst.updateStatus('online');
      }, 1500);
    }
  });

  // Writer synthesizes into final report
  writer.on('message:share_knowledge', async (message: any) => {
    if (message.from_agent === 'analyst-001') {
      console.log(`✍️  Writer synthesizing: ${message.payload.knowledge}`);
      await writer.updateStatus('busy', 'Writing final report');

      setTimeout(async () => {
        await writer.broadcast('task_complete', {
          report: 'Final Research Report: AI Safety Frameworks',
          sections: 5,
          word_count: 2500,
        });
        await writer.updateStatus('online');
      }, 1000);
    }
  });

  // Track presence changes
  const presenceHandler = (event: any) => {
    console.log(`👥 Team update: ${event.agents?.length || 0} agents online`);
  };
  researcher.on('agents:sync', presenceHandler);
  analyst.on('agents:sync', presenceHandler);
  writer.on('agents:sync', presenceHandler);

  // Assign initial task to researcher
  await researcher.assignTask({
    task_id: 'research-001',
    assigned_to: 'researcher-001',
    description: 'Research AI safety frameworks',
    priority: 'high',
  });

  // Let the workflow run
  await new Promise((resolve) => setTimeout(resolve, 6000));

  // Print final stats
  const stats = await researcher.getStats();
  console.log('\n📊 Final Statistics:', JSON.stringify(stats, null, 2));

  // Cleanup
  await Promise.all([
    researcher.shutdown(),
    analyst.shutdown(),
    writer.shutdown(),
  ]);
}

/**
 * Example 2: Real-Time Memory Synchronization
 *
 * Demonstrates how memories are instantly shared across agents
 */
async function memorySync Example() {
  console.log('\n💾 Example 2: Real-Time Memory Synchronization\n');

  const agent1 = createRealtimeHub('agent-001', 'memory-test');
  const agent2 = createRealtimeHub('agent-002', 'memory-test');

  await Promise.all([agent1.initialize(), agent2.initialize()]);

  // Agent 2 listens for new memories
  agent2.on('memory:added', (memory: any) => {
    console.log(`📥 Agent2 received new memory from ${memory.agent_id}:`);
    console.log(`   Content: ${memory.content}`);
  });

  // Agent 1 creates a memory (would trigger database insert in real implementation)
  console.log('📤 Agent1 creating memory...');
  // In real usage, this would call SupabaseAdapter.storeMemory()
  // which triggers real-time event to agent2

  await new Promise((resolve) => setTimeout(resolve, 2000));

  await Promise.all([agent1.shutdown(), agent2.shutdown()]);
}

/**
 * Example 3: Collaborative Problem Solving
 *
 * Agents help each other solve complex problems
 */
async function collaborativeProblemSolving() {
  console.log('\n🤝 Example 3: Collaborative Problem Solving\n');

  const coder = createRealtimeHub('coder-001', 'dev-team');
  const debugger = createRealtimeHub('debugger-001', 'dev-team');
  const tester = createRealtimeHub('tester-001', 'dev-team');

  await Promise.all([
    coder.initialize(),
    debugger.initialize(),
    tester.initialize(),
  ]);

  // Coder encounters a problem and requests help
  coder.on('message:task_assignment', async (message: any) => {
    console.log(`💻 Coder working on: ${message.payload.description}`);
    await coder.updateStatus('busy', 'Implementing feature');

    setTimeout(async () => {
      await coder.requestHelp('Getting type errors in TypeScript', {
        file: 'api.ts',
        line: 42,
        error: 'Type "string" is not assignable to type "number"',
      });
    }, 1000);
  });

  // Debugger responds to help request
  debugger.on('message:request_help', async (message: any) => {
    console.log(`🔧 Debugger analyzing: ${message.payload.problem}`);
    await debugger.updateStatus('busy', 'Debugging type error');

    setTimeout(async () => {
      await debugger.sendMessage(
        message.payload.from,
        'share_knowledge',
        {
          solution: 'Change type definition to accept string | number',
          file_fix: 'type ApiResponse = string | number;',
        }
      );
      await debugger.updateStatus('online');
    }, 1500);
  });

  // Coder receives solution and applies fix
  coder.on('message:share_knowledge', async (message: any) => {
    if (message.payload.solution) {
      console.log(`✅ Coder applying fix: ${message.payload.solution}`);
      await coder.reportTaskComplete('impl-001', {
        status: 'fixed',
        applied_solution: message.payload.solution,
      });
      await coder.updateStatus('online');

      // Notify tester
      await coder.assignTask({
        task_id: 'test-001',
        assigned_to: 'tester-001',
        description: 'Test API type changes',
        priority: 'high',
      });
    }
  });

  // Tester validates the fix
  tester.on('message:task_assignment', async (message: any) => {
    console.log(`🧪 Tester running: ${message.payload.description}`);
    await tester.updateStatus('busy', 'Running test suite');

    setTimeout(async () => {
      await tester.reportTaskComplete(message.payload.task_id, {
        tests_passed: 42,
        tests_failed: 0,
        coverage: 0.95,
      });
      await tester.updateStatus('online');
    }, 2000);
  });

  // Start the workflow
  await coder.assignTask({
    task_id: 'impl-001',
    assigned_to: 'coder-001',
    description: 'Implement new API endpoint',
    priority: 'high',
  });

  await new Promise((resolve) => setTimeout(resolve, 8000));

  await Promise.all([coder.shutdown(), debugger.shutdown(), tester.shutdown()]);
}

/**
 * Example 4: Dynamic Team Scaling
 *
 * Monitor workload and request additional agents
 */
async function dynamicTeamScaling() {
  console.log('\n📈 Example 4: Dynamic Team Scaling\n');

  const coordinator = createRealtimeHub('coordinator-001', 'scaling-team');
  await coordinator.initialize();

  // Track team size
  coordinator.on('agents:sync', (event: any) => {
    const teamSize = event.agents.length;
    console.log(`👥 Current team size: ${teamSize} agents`);

    // If team is small but workload is high, request more agents
    if (teamSize < 3) {
      console.log('⚠️  Team understaffed, requesting additional agents...');
      coordinator.broadcast('status_update', {
        message: 'Need more agents for workload',
        current_team: teamSize,
        requested_team: 5,
      });
    }
  });

  // Simulate agents joining
  const worker1 = createRealtimeHub('worker-001', 'scaling-team');
  await worker1.initialize();
  await new Promise((resolve) => setTimeout(resolve, 1000));

  const worker2 = createRealtimeHub('worker-002', 'scaling-team');
  await worker2.initialize();
  await new Promise((resolve) => setTimeout(resolve, 1000));

  const worker3 = createRealtimeHub('worker-003', 'scaling-team');
  await worker3.initialize();
  await new Promise((resolve) => setTimeout(resolve, 1000));

  // Check final stats
  const stats = await coordinator.getStats();
  console.log('\n📊 Team Composition:', JSON.stringify(stats.agents, null, 2));

  await Promise.all([
    coordinator.shutdown(),
    worker1.shutdown(),
    worker2.shutdown(),
    worker3.shutdown(),
  ]);
}

/**
 * Run all examples
 */
async function main() {
  console.log('🌐 Real-Time Federation Examples\n');
  console.log('Prerequisites:');
  console.log('  - Set SUPABASE_URL environment variable');
  console.log('  - Set SUPABASE_ANON_KEY environment variable');
  console.log('  - Run Supabase migrations (see docs/supabase/migrations/)');
  console.log('\n' + '='.repeat(60) + '\n');

  try {
    await multiAgentResearchExample();
    console.log('\n' + '='.repeat(60) + '\n');

    await memorySyncExample();
    console.log('\n' + '='.repeat(60) + '\n');

    await collaborativeProblemSolving();
    console.log('\n' + '='.repeat(60) + '\n');

    await dynamicTeamScaling();
    console.log('\n' + '='.repeat(60) + '\n');

    console.log('✅ All examples completed successfully!');
  } catch (error) {
    console.error('❌ Error running examples:', error);
    process.exit(1);
  }
}

// Run if executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
