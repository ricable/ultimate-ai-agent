#!/usr/bin/env node
/**
 * Example: Multi-Agent Coordination with QuantumDAG
 *
 * This example demonstrates how to use QuantumDAG integration for
 * quantum-resistant multi-agent coordination with conflict detection.
 */

const { JJWrapper } = require('../index.js');
const { createQuantumBridge } = require('../src/quantum_bridge.js');

async function main() {
  console.log('=== QuantumDAG Agent Coordination Example ===\n');

  // 1. Create JJWrapper and get coordination
  console.log('1. Initializing coordination system...');
  const wrapper = new JJWrapper('/tmp/test-repo'); // Replace with actual repo
  const coordination = wrapper.getCoordination();

  // 2. Create QuantumBridge
  console.log('2. Creating QuantumBridge...');
  const bridge = createQuantumBridge(coordination);
  await bridge.initialize();

  console.log('   ✓ QuantumDAG initialized with quantum-resistant security\n');

  // 3. Register multiple agents
  console.log('3. Registering agents...');
  const agents = [
    { id: 'coder-1', type: 'coder' },
    { id: 'reviewer-1', type: 'reviewer' },
    { id: 'tester-1', type: 'tester' },
    { id: 'architect-1', type: 'architect' },
  ];

  for (const agent of agents) {
    await coordination.registerAgent(agent.id, agent.type);
    console.log(`   ✓ Registered ${agent.type}: ${agent.id}`);
  }
  console.log('');

  // 4. Simulate agent operations
  console.log('4. Simulating agent operations...\n');

  // Coder edits main.js
  console.log('   [Coder] Editing main.js...');
  const coderOp = {
    operationType: 'edit',
    agentId: 'coder-1',
    metadata: { purpose: 'add-feature', lines: 50 },
  };

  const coderVertexId = await bridge.registerOperation(
    'op-coder-1',
    coderOp,
    ['src/main.js']
  );
  console.log(`   ✓ Registered as DAG vertex: ${coderVertexId}\n`);

  // Reviewer wants to review the same file
  console.log('   [Reviewer] Checking conflicts before reviewing main.js...');
  const reviewerConflicts = await bridge.checkConflicts(
    'op-reviewer-1',
    'review',
    ['src/main.js']
  );

  if (reviewerConflicts.length > 0) {
    console.log(`   ⚠ Found ${reviewerConflicts.length} conflict(s):`);
    for (const conflict of reviewerConflicts) {
      console.log(`     - Severity: ${conflict.severity}/3`);
      console.log(`     - Files: ${conflict.conflictingFiles.join(', ')}`);
      console.log(`     - Strategy: ${conflict.resolutionStrategy}`);
      console.log(`     - Quantum verified: ${conflict.quantumVerified}`);
      console.log(`     - DAG distance: ${conflict.dagDistance}`);
    }
  } else {
    console.log('   ✓ No conflicts detected');
  }
  console.log('');

  // Tester edits test file (no conflict)
  console.log('   [Tester] Editing test/main.test.js...');
  const testerOp = {
    operationType: 'edit',
    agentId: 'tester-1',
    metadata: { purpose: 'add-tests', coverage: 85 },
  };

  const testerVertexId = await bridge.registerOperation(
    'op-tester-1',
    testerOp,
    ['test/main.test.js']
  );
  console.log(`   ✓ Registered as DAG vertex: ${testerVertexId}`);
  console.log('   ✓ No conflicts (different file)\n');

  // Architect wants to modify main.js (conflict)
  console.log('   [Architect] Checking conflicts before refactoring main.js...');
  const architectConflicts = await bridge.checkConflicts(
    'op-architect-1',
    'refactor',
    ['src/main.js']
  );

  if (architectConflicts.length > 0) {
    console.log(`   ⚠ Found ${architectConflicts.length} conflict(s):`);
    for (const conflict of architectConflicts) {
      console.log(`     - ${conflict.description}`);
      console.log(`     - Recommended: ${conflict.resolutionStrategy}`);
    }
  }
  console.log('');

  // 5. Get coordination statistics
  console.log('5. Coordination statistics:');
  const dagStats = await bridge.getStats();
  console.log(`   - DAG vertices: ${dagStats.vertices}`);
  console.log(`   - DAG tips: ${dagStats.tips}`);
  console.log(`   - Quantum enabled: ${dagStats.quantumEnabled}`);
  console.log(`   - Total edges: ${dagStats.totalEdges || 0}`);

  const coordStats = await coordination.getStats();
  console.log(`   - Total agents: ${coordStats.totalAgents}`);
  console.log(`   - Active agents: ${coordStats.activeAgents}`);
  console.log(`   - Total operations: ${coordStats.totalOperations}`);
  console.log('');

  // 6. Get coordination tips
  console.log('6. Current DAG tips:');
  const tips = await bridge.getCoordinationTips();
  for (const tip of tips) {
    console.log(`   - ${tip}`);
  }
  console.log('');

  // 7. Verify quantum proof
  console.log('7. Verifying quantum-resistant proofs...');
  const coderProofValid = await bridge.verifyQuantumProof(coderVertexId);
  const testerProofValid = await bridge.verifyQuantumProof(testerVertexId);

  console.log(`   - Coder operation proof: ${coderProofValid ? '✓ Valid' : '✗ Invalid'}`);
  console.log(`   - Tester operation proof: ${testerProofValid ? '✓ Valid' : '✗ Invalid'}`);
  console.log('');

  // 8. Agent statistics
  console.log('8. Agent statistics:');
  const agentStats = await coordination.listAgents();
  for (const agent of agentStats) {
    console.log(`   - ${agent.agentType} (${agent.agentId}):`);
    console.log(`     Operations: ${agent.operationsCount}, Reputation: ${agent.reputation.toFixed(2)}`);
  }
  console.log('');

  // 9. Cleanup
  console.log('9. Cleaning up...');
  await bridge.cleanup();
  console.log('   ✓ Resources released\n');

  console.log('=== Example Complete ===');
  console.log('\nKey Features Demonstrated:');
  console.log('• Multi-agent coordination with QuantumDAG');
  console.log('• Quantum-resistant conflict detection');
  console.log('• DAG-based severity calculation');
  console.log('• Automatic resolution strategy suggestion');
  console.log('• Cryptographic proof verification');
  console.log('• Real-time coordination statistics');
}

// Run example
main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
