#!/usr/bin/env node
const { JjWrapper } = require('./index.js');

async function testWithFailures() {
    console.log('=== Testing AgentDB with Failed Operations ===\n');

    const jj = new JjWrapper();

    console.log('Initial stats:');
    console.log(jj.getStats());

    console.log('\nExecuting commands that will fail (no repo)...');

    try {
        await jj.status();
    } catch (e) {
        console.log('✅ status() failed as expected');
    }

    try {
        await jj.log(10);
    } catch (e) {
        console.log('✅ log() failed as expected');
    }

    try {
        await jj.diff('@', '@-');
    } catch (e) {
        console.log('✅ diff() failed as expected');
    }

    console.log('\nStats after failed operations:');
    const stats = JSON.parse(jj.getStats());
    console.log('Stats:', stats);

    console.log('\nOperations:');
    const ops = jj.getOperations(10);
    console.log(`Count: ${ops.length}`);

    if (ops.length > 0) {
        console.log('\n✅ FIXED! Failed operations are now logged!\n');
        ops.forEach((op, i) => {
            console.log(`${i + 1}. ${op.operationType}: ${op.command}`);
            console.log(`   Success: ${op.success}`);
            console.log(`   Error: ${op.error || 'none'}`);
            console.log(`   Duration: ${op.durationMs}ms`);
        });
    } else {
        console.log('\n❌ NOT FIXED - operations still not logged');
    }
}

testWithFailures().catch(console.error);
