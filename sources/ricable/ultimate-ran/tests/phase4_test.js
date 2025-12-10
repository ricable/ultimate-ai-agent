
import { TitanOrchestrator } from '../src/racs/orchestrator.js';

// Mock dependencies
const mockConfig = {
    gnnModel: {},
    consensus: {}
};

const mockAgentDB = {
    embed: async (text) => [0.1, 0.2, 0.3],
    searchSimilar: async () => [],
    getWorldModel: async () => ({})
};

const mockRuVector = {};
const mockSparc = {};
const mockAgui = {
    emit: async () => { },
    start: async () => { }
};

async function runTest() {
    console.log('--- Phase 4: Production Autonomy Test ---');

    const orchestrator = new TitanOrchestrator({
        config: mockConfig,
        agentDB: mockAgentDB,
        ruvector: mockRuVector,
        sparcValidator: mockSparc,
        aguiServer: mockAgui
    });

    // 1. Scale to Production (100+ cells)
    const targetCells = 100;
    console.log(`\n[TEST] 1. Scaling Swarm to ${targetCells} cells...`);
    await orchestrator.scaleSwarm(targetCells);

    // Expected counts:
    // Clusters: ceil(100/20) = 5
    // Sentinels: ceil(100/10) = 10
    const expectedClusters = 5;
    const expectedSentinels = 10;

    verifyCounts(orchestrator, expectedClusters, expectedSentinels);

    // 2. Simulate Failure (Chaos Monkey)
    console.log('\n[TEST] 2. Simulating Agent Failure (Chaos Monkey)...');

    // Find a sentinel
    const sentinelId = Array.from(orchestrator.activeAgents.values()).find(a => a.type === 'sentinel').id;
    console.log(`Killing Sentinel: ${sentinelId}`);

    // "Kill" it by setting status to failed
    const victim = orchestrator.activeAgents.get(sentinelId);
    victim.status = 'failed';

    // 3. Verify Self-Healing
    console.log('\n[TEST] 3. Verifying Self-Healing...');

    // Manually trigger the Swarm Manager tick
    await orchestrator.swarmManager.healthCheckTick();

    // Verify victim is gone and new agent spawned
    if (orchestrator.activeAgents.has(sentinelId)) {
        throw new Error('Self-Healing Failed: Failed agent still exists in map.');
    }

    verifyCounts(orchestrator, expectedClusters, expectedSentinels);
    console.log('Self-Healing Successful.');

    // 4. Density Enforcement
    console.log('\n[TEST] 4. Verifying Density Enforcement...');
    // Delete an agent entirely (simulate crash/disappearance)
    const clusterId = Array.from(orchestrator.activeAgents.values()).find(a => a.type === 'cluster_orchestrator').id;
    console.log(`Deleting Cluster Orchestrator: ${clusterId}`);
    orchestrator.activeAgents.delete(clusterId);

    // Trigger tick
    await orchestrator.swarmManager.healthCheckTick();

    verifyCounts(orchestrator, expectedClusters, expectedSentinels);
    console.log('Density Enforcement Successful.');

    console.log('\n--- Phase 4 Test Complete ---');
    process.exit(0);
}

function verifyCounts(orchestrator, expectedClusters, expectedSentinels) {
    let clusters = 0;
    let sentinels = 0;

    for (const agent of orchestrator.activeAgents.values()) {
        if (agent.type === 'cluster_orchestrator') clusters++;
        if (agent.type === 'sentinel') sentinels++;
    }

    console.log(`Current State: ${clusters} Clusters, ${sentinels} Sentinels.`);

    if (clusters !== expectedClusters) throw new Error(`Cluster count mismatch. Expected ${expectedClusters}, got ${clusters}`);
    if (sentinels !== expectedSentinels) throw new Error(`Sentinel count mismatch. Expected ${expectedSentinels}, got ${sentinels}`);
}

runTest().catch(e => {
    console.error(e);
    process.exit(1);
});
