#!/usr/bin/env node
/**
 * Agent Coordination Test
 *
 * Tests the multi-agent coordination features of agentic-jujutsu
 */

const nativeAddon = require('../index.js');
console.log('Native addon loaded:', Object.keys(nativeAddon));
const JJWrapper = nativeAddon.JjWrapper; // Note: might be JjWrapper not JJWrapper
const assert = require('assert');

console.log('🧪 Testing Agent Coordination\n');

(async () => {
    try {
        const jj = new JJWrapper();

        // Test 1: Enable coordination
        console.log('1. Enable Agent Coordination...');
        await jj.enableAgentCoordination();
        console.log('   ✅ Coordination enabled\n');

        // Test 2: Register agents
        console.log('2. Register Agents...');
        await jj.registerAgent('coder-1', 'coder');
        await jj.registerAgent('reviewer-1', 'reviewer');
        await jj.registerAgent('tester-1', 'tester');
        console.log('   ✅ Registered 3 agents\n');

        // Test 3: List agents
        console.log('3. List Agents...');
        const agentsJson = await jj.listAgents();
        const agents = JSON.parse(agentsJson);
        assert.strictEqual(agents.length, 3, 'Should have 3 agents');
        console.log(`   Agents: ${agents.map(a => a.agent_id).join(', ')}`);
        console.log('   ✅ List agents working\n');

        // Test 4: Get agent stats
        console.log('4. Get Agent Stats...');
        const statsJson = await jj.getAgentStats('coder-1');
        const stats = JSON.parse(statsJson);
        assert.ok(stats, 'Stats should exist');
        assert.strictEqual(stats.agent_id, 'coder-1');
        assert.strictEqual(stats.agent_type, 'coder');
        console.log(`   Agent: ${stats.agent_id}, Type: ${stats.agent_type}`);
        console.log('   ✅ Agent stats working\n');

        // Test 5: Check coordination stats
        console.log('5. Coordination Statistics...');
        const coordStatsJson = await jj.getCoordinationStats();
        const coordStats = JSON.parse(coordStatsJson);
        assert.strictEqual(coordStats.total_agents, 3);
        console.log(`   Total agents: ${coordStats.total_agents}`);
        console.log(`   Active agents: ${coordStats.active_agents}`);
        console.log('   ✅ Coordination stats working\n');

        // Test 6: Check for conflicts (no operations yet)
        console.log('6. Check Conflicts (Empty)...');
        const conflictsJson = await jj.checkAgentConflicts(
            'op-1',
            'edit',
            ['file1.txt']
        );
        const conflicts = JSON.parse(conflictsJson);
        assert.strictEqual(conflicts.length, 0, 'Should have no conflicts initially');
        console.log('   ✅ No conflicts detected\n');

        // Test 7: Get coordination tips
        console.log('7. Get Coordination Tips...');
        const tips = await jj.getCoordinationTips();
        assert.ok(Array.isArray(tips), 'Tips should be an array');
        console.log(`   Tips count: ${tips.length}`);
        console.log('   ✅ Coordination tips working\n');

        console.log('═══════════════════════════════════════════════════');
        console.log('✅ ALL AGENT COORDINATION TESTS PASSED!');
        console.log('═══════════════════════════════════════════════════\n');

        console.log('Next Steps:');
        console.log('  1. ✅ AgentCoordination module implemented');
        console.log('  2. ✅ N-API bindings added');
        console.log('  3. ✅ TypeScript definitions updated');
        console.log('  4. ✅ All coordination tests passed');
        console.log('  5. ⬜ Integrate @qudag/napi-core QuantumDAG');
        console.log('  6. ⬜ Add quantum fingerprints');
        console.log('  7. ⬜ Implement ML-DSA signing');
        console.log('  8. ⬜ Release v2.2.0\n');

        process.exit(0);
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
})();
