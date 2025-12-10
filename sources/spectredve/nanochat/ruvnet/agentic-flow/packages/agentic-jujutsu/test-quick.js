#!/usr/bin/env node
const { JjWrapper } = require('./index.js');

console.log('🧪 Quick ReasoningBank Verification Test\n');

try {
    const jj = new JjWrapper();

    // Test 1: startTrajectory
    console.log('1. Testing startTrajectory()...');
    const trajectoryId = jj.startTrajectory('Quick test task');
    console.log(`   ✅ Started trajectory: ${trajectoryId}\n`);

    // Test 2: addToTrajectory
    console.log('2. Testing addToTrajectory()...');
    jj.addToTrajectory();
    console.log('   ✅ Added to trajectory\n');

    // Test 3: finalizeTrajectory
    console.log('3. Testing finalizeTrajectory()...');
    jj.finalizeTrajectory(0.9, 'Test successful');
    console.log('   ✅ Finalized trajectory\n');

    // Test 4: getLearningStats
    console.log('4. Testing getLearningStats()...');
    const stats = JSON.parse(jj.getLearningStats());
    console.log(`   ✅ Total trajectories: ${stats.totalTrajectories}`);
    console.log(`   ✅ Total patterns: ${stats.totalPatterns}\n`);

    // Test 5: getPatterns
    console.log('5. Testing getPatterns()...');
    const patterns = JSON.parse(jj.getPatterns());
    console.log(`   ✅ Patterns returned: ${patterns.length}\n`);

    // Test 6: getSuggestion
    console.log('6. Testing getSuggestion()...');
    const suggestion = JSON.parse(jj.getSuggestion('Test task'));
    console.log(`   ✅ Confidence: ${suggestion.confidence}`);
    console.log(`   ✅ Expected success: ${suggestion.expectedSuccessRate}\n`);

    // Test 7: queryTrajectories
    console.log('7. Testing queryTrajectories()...');
    const trajectories = JSON.parse(jj.queryTrajectories('test', 10));
    console.log(`   ✅ Trajectories returned: ${trajectories.length}\n`);

    // Test 8: resetLearning
    console.log('8. Testing resetLearning()...');
    jj.resetLearning();
    const statsAfterReset = JSON.parse(jj.getLearningStats());
    console.log(`   ✅ Total trajectories after reset: ${statsAfterReset.totalTrajectories}\n`);

    console.log('═══════════════════════════════════════');
    console.log('✅ ALL 8 REASONINGBANK METHODS WORKING!');
    console.log('═══════════════════════════════════════\n');

    process.exit(0);
} catch (error) {
    console.error('❌ TEST FAILED:', error.message);
    console.error(error.stack);
    process.exit(1);
}
