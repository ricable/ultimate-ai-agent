#!/usr/bin/env node
/**
 * Test AgentDB functionality
 */

const { JjWrapper } = require('./index.js');

async function testAgentDB() {
    console.log('=== Testing AgentDB Functionality ===\n');

    // Test 1: Create wrapper and check initial stats
    console.log('Test 1: Initial Stats');
    const jj = new JjWrapper();
    const initialStats = JSON.parse(jj.getStats());
    console.log('Initial stats:', initialStats);
    console.log('Expected: total_operations=0');
    console.log('Result:', initialStats.total_operations === 0 ? '✅ PASS' : '❌ FAIL');
    console.log();

    // Test 2: Execute a command and check if it's logged
    console.log('Test 2: Execute Command and Check Logging');
    try {
        await jj.execute(['--version']);
        console.log('Command executed successfully');
    } catch (e) {
        console.log('Command error (expected if not in repo):', e.message.substring(0, 50));
    }

    const afterExecStats = JSON.parse(jj.getStats());
    console.log('Stats after execute:', afterExecStats);
    console.log('Expected: total_operations=1');
    console.log('Result:', afterExecStats.total_operations === 1 ? '✅ PASS' : '❌ FAIL');
    console.log();

    // Test 3: Check operations array
    console.log('Test 3: Get Operations Array');
    const ops = jj.getOperations(10);
    console.log('Operations count:', ops.length);
    console.log('Expected: 1 operation');
    console.log('Result:', ops.length === 1 ? '✅ PASS' : '❌ FAIL');

    if (ops.length > 0) {
        console.log('\nFirst operation:');
        console.log('  - ID:', ops[0].id || 'missing');
        console.log('  - Type:', ops[0].operationType || 'missing');
        console.log('  - Command:', ops[0].command || 'missing');
        console.log('  - User:', ops[0].user || 'missing');
        console.log('  - Success:', ops[0].success);
        console.log('  - Duration:', ops[0].durationMs, 'ms');
    }
    console.log();

    // Test 4: Execute multiple commands
    console.log('Test 4: Multiple Commands');
    for (let i = 0; i < 5; i++) {
        try {
            await jj.execute(['--version']);
        } catch (e) {
            // Ignore errors
        }
    }

    const multiStats = JSON.parse(jj.getStats());
    console.log('Stats after 5 more commands:', multiStats);
    console.log('Expected: total_operations=6');
    console.log('Result:', multiStats.total_operations === 6 ? '✅ PASS' : '❌ FAIL');
    console.log('Average duration:', multiStats.avg_duration_ms.toFixed(2), 'ms');
    console.log();

    // Test 5: Get user operations
    console.log('Test 5: Get User Operations');
    const userOps = jj.getUserOperations(10);
    console.log('User operations count:', userOps.length);
    console.log('Expected: Same as total (6)');
    console.log('Result:', userOps.length === 6 ? '✅ PASS' : '❌ FAIL');
    console.log();

    // Test 6: Clear log
    console.log('Test 6: Clear Log');
    jj.clearLog();
    const clearedStats = JSON.parse(jj.getStats());
    console.log('Stats after clear:', clearedStats);
    console.log('Expected: total_operations=0');
    console.log('Result:', clearedStats.total_operations === 0 ? '✅ PASS' : '❌ FAIL');
    console.log();

    // Summary
    console.log('=== Summary ===');
    const passed = [
        initialStats.total_operations === 0,
        afterExecStats.total_operations === 1,
        ops.length === 1,
        multiStats.total_operations === 6,
        userOps.length === 6,
        clearedStats.total_operations === 0
    ].filter(x => x).length;

    console.log(`Passed: ${passed}/6 tests`);
    console.log(passed === 6 ? '\n✅ ALL TESTS PASSED - AgentDB is working!' : '\n❌ SOME TESTS FAILED - AgentDB needs fixing');
}

testAgentDB().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
