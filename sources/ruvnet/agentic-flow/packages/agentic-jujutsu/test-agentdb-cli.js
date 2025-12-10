#!/usr/bin/env node
/**
 * Test AgentDB with actual CLI-like operations
 */

const { JjWrapper } = require('./index.js');
const fs = require('fs');
const path = require('path');

async function testWithRepo() {
    console.log('=== Testing AgentDB with Repository Operations ===\n');

    // Create temp directory
    const tmpDir = '/tmp/agentdb-test-' + Date.now();
    fs.mkdirSync(tmpDir, { recursive: true });
    process.chdir(tmpDir);

    console.log('Working directory:', tmpDir);
    console.log();

    // Initialize jj repo
    console.log('Initializing jj repository...');
    const { execSync } = require('child_process');
    try {
        execSync('~/.cache/agentic-jujutsu/jj git init', { stdio: 'ignore' });
        console.log('✅ Repository initialized\n');
    } catch (e) {
        console.log('❌ Failed to initialize repo:', e.message);
        return;
    }

    // Create JjWrapper
    const jj = new JjWrapper();

    // Create test file
    fs.writeFileSync('test.txt', 'Hello AgentDB!\n');
    fs.writeFileSync('another.txt', 'Another file\n');

    console.log('Test 1: Status Command');
    try {
        await jj.status();
        const stats1 = JSON.parse(jj.getStats());
        console.log('Operations logged:', stats1.total_operations);
        console.log('Result:', stats1.total_operations === 1 ? '✅ PASS' : '❌ FAIL');
        console.log();
    } catch (e) {
        console.log('Status error:', e.message);
    }

    console.log('Test 2: Log Command');
    try {
        await jj.log(5);
        const stats2 = JSON.parse(jj.getStats());
        console.log('Operations logged:', stats2.total_operations);
        console.log('Result:', stats2.total_operations === 2 ? '✅ PASS' : '❌ FAIL');
        console.log();
    } catch (e) {
        console.log('Log error:', e.message);
    }

    console.log('Test 3: New Commit');
    try {
        await jj.newCommit('Test commit from AgentDB test');
        const stats3 = JSON.parse(jj.getStats());
        console.log('Operations logged:', stats3.total_operations);
        console.log('Result:', stats3.total_operations === 3 ? '✅ PASS' : '❌ FAIL');
        console.log();
    } catch (e) {
        console.log('New commit error:', e.message);
    }

    console.log('Test 4: Describe');
    try {
        await jj.describe('Updated description');
        const stats4 = JSON.parse(jj.getStats());
        console.log('Operations logged:', stats4.total_operations);
        console.log('Result:', stats4.total_operations === 4 ? '✅ PASS' : '❌ FAIL');
        console.log();
    } catch (e) {
        console.log('Describe error:', e.message);
    }

    // Show all operations
    console.log('=== All Logged Operations ===');
    const ops = jj.getOperations(10);
    console.log(`Total: ${ops.length} operations\n`);

    ops.forEach((op, i) => {
        console.log(`${i + 1}. ${op.operationType} (${op.command})`);
        console.log(`   User: ${op.user}`);
        console.log(`   Success: ${op.success}`);
        console.log(`   Duration: ${op.durationMs}ms`);
        console.log(`   Timestamp: ${new Date(op.timestamp).toISOString()}`);
        console.log();
    });

    // Show stats
    const finalStats = JSON.parse(jj.getStats());
    console.log('=== Final Statistics ===');
    console.log('Total operations:', finalStats.total_operations);
    console.log('Success rate:', (finalStats.success_rate * 100).toFixed(1) + '%');
    console.log('Average duration:', finalStats.avg_duration_ms.toFixed(2) + 'ms');
    console.log();

    // Cleanup
    console.log('Cleaning up...');
    process.chdir('/tmp');
    try {
        fs.rmSync(tmpDir, { recursive: true, force: true });
        console.log('✅ Cleanup complete');
    } catch (e) {
        console.log('Cleanup warning:', e.message);
    }
}

testWithRepo().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
});
