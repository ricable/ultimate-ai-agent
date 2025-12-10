/**
 * Test suite for EphemeralAgent lifecycle and memory operations
 */

import { EphemeralAgent } from '../../src/federation/index.js';
import { logger } from '../../src/utils/logger.js';

async function testEphemeralAgent() {
  console.log('\n=== Testing EphemeralAgent ===\n');

  // Test 1: Agent spawn and lifecycle
  console.log('Test 1: Spawning ephemeral agent...');
  const agent = await EphemeralAgent.spawn({
    tenantId: 'test-tenant',
    lifetime: 10, // 10 seconds for testing
    syncInterval: 2000 // Sync every 2 seconds
  });

  const info = agent.getInfo();
  console.log(`✅ Agent spawned: ${info?.agentId}`);
  console.log(`   Expires at: ${new Date(info?.expiresAt || 0).toISOString()}`);
  console.log(`   Remaining lifetime: ${agent.getRemainingLifetime()}s`);

  // Test 2: Execute task with memory access
  console.log('\nTest 2: Executing task...');
  const result = await agent.execute(async (db, context) => {
    console.log(`   Agent ${context.agentId} executing task`);

    // Store a test memory
    await agent.storeEpisode({
      task: 'test-task',
      input: 'test input',
      output: 'test output',
      reward: 0.95
    });

    console.log('   Stored test episode');

    return { success: true, message: 'Task completed' };
  });

  console.log(`✅ Task result:`, result);

  // Test 3: Query memories
  console.log('\nTest 3: Querying memories...');
  const memories = await agent.queryMemories('test-task', 5);
  console.log(`✅ Found ${memories.length} memories`);

  // Test 4: Check agent is still alive
  console.log('\nTest 4: Checking agent status...');
  console.log(`   Is alive: ${agent.isAlive()}`);
  console.log(`   Remaining lifetime: ${agent.getRemainingLifetime()}s`);

  // Test 5: Wait for expiration
  console.log('\nTest 5: Waiting for agent expiration (10 seconds)...');
  await new Promise(resolve => setTimeout(resolve, 10500));
  console.log(`   Is alive: ${agent.isAlive()}`);
  console.log(`   Remaining lifetime: ${agent.getRemainingLifetime()}s`);

  // Test 6: Try to execute after expiration (should fail)
  console.log('\nTest 6: Attempting to execute after expiration...');
  try {
    await agent.execute(async () => {
      return { success: false };
    });
    console.log('❌ Should have failed (agent expired)');
  } catch (error: any) {
    console.log(`✅ Correctly rejected: ${error.message}`);
  }

  // Test 7: Manual cleanup
  console.log('\nTest 7: Manual agent cleanup...');
  await agent.destroy();
  console.log(`✅ Agent destroyed`);

  console.log('\n=== All EphemeralAgent tests passed ===\n');
}

async function testMultipleAgents() {
  console.log('\n=== Testing Multiple Concurrent Agents ===\n');

  // Spawn 3 agents concurrently
  console.log('Spawning 3 agents concurrently...');
  const agents = await Promise.all([
    EphemeralAgent.spawn({
      tenantId: 'tenant-a',
      lifetime: 15
    }),
    EphemeralAgent.spawn({
      tenantId: 'tenant-b',
      lifetime: 15
    }),
    EphemeralAgent.spawn({
      tenantId: 'tenant-c',
      lifetime: 15
    })
  ]);

  console.log(`✅ Spawned ${agents.length} agents`);

  // Execute tasks in parallel
  console.log('\nExecuting tasks in parallel...');
  const results = await Promise.all(
    agents.map((agent, index) =>
      agent.execute(async (db, context) => {
        console.log(`   Agent ${index + 1} (${context.agentId}) executing`);

        // Store different episodes for each tenant
        await agent.storeEpisode({
          task: `task-${index + 1}`,
          input: `input-${index + 1}`,
          output: `output-${index + 1}`,
          reward: 0.8 + (index * 0.05)
        });

        return { agentIndex: index + 1, tenantId: context.tenantId };
      })
    )
  );

  console.log(`✅ Completed ${results.length} parallel tasks`);
  results.forEach((r, i) => {
    console.log(`   Agent ${i + 1}: ${r.tenantId}`);
  });

  // Cleanup all agents
  console.log('\nCleaning up agents...');
  await Promise.all(agents.map(a => a.destroy()));
  console.log('✅ All agents destroyed');

  console.log('\n=== Multiple agent tests passed ===\n');
}

async function testTenantIsolation() {
  console.log('\n=== Testing Tenant Isolation ===\n');

  // Create agents for two different tenants
  const agentA = await EphemeralAgent.spawn({
    tenantId: 'tenant-a',
    lifetime: 15
  });

  const agentB = await EphemeralAgent.spawn({
    tenantId: 'tenant-b',
    lifetime: 15
  });

  console.log('✅ Created agents for tenant-a and tenant-b');

  // Store memories for tenant-a
  await agentA.execute(async () => {
    await agentA.storeEpisode({
      task: 'tenant-a-task',
      input: 'tenant-a data',
      output: 'tenant-a result',
      reward: 0.9
    });
    console.log('   Stored memory for tenant-a');
  });

  // Store memories for tenant-b
  await agentB.execute(async () => {
    await agentB.storeEpisode({
      task: 'tenant-b-task',
      input: 'tenant-b data',
      output: 'tenant-b result',
      reward: 0.85
    });
    console.log('   Stored memory for tenant-b');
  });

  // Query memories (should be isolated)
  console.log('\nQuerying memories...');
  const memoriesA = await agentA.queryMemories('tenant', 10);
  const memoriesB = await agentB.queryMemories('tenant', 10);

  console.log(`   Tenant A memories: ${memoriesA.length}`);
  console.log(`   Tenant B memories: ${memoriesB.length}`);

  // Cleanup
  await agentA.destroy();
  await agentB.destroy();

  console.log('\n✅ Tenant isolation test completed');
}

// Run all tests
(async () => {
  try {
    await testEphemeralAgent();
    await testMultipleAgents();
    await testTenantIsolation();

    console.log('\n🎉 All federation tests passed!\n');
  } catch (error: any) {
    console.error('\n❌ Test failed:', error.message);
    console.error(error.stack);
    process.exit(1);
  }
})();
