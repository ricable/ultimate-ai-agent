#!/usr/bin/env node
/**
 * ReasoningBank Test Suite
 *
 * Tests advanced self-learning and pattern recognition features
 */

const { JjWrapper } = require('../index.js');
const assert = require('assert');

async function testBasicTrajectory() {
    console.log('\n=== Test 1: Basic Trajectory Creation ===');

    const jj = new JjWrapper();

    // Start a trajectory
    const trajectoryId = jj.startTrajectory('Create new feature');
    console.log(`✓ Started trajectory: ${trajectoryId}`);
    assert(trajectoryId, 'Trajectory ID should be returned');

    // Simulate some operations
    try {
        await jj.execute(['--version']);
        await jj.execute(['--help']);
    } catch (e) {
        // Expected if not in a repo
    }

    // Add operations to trajectory
    jj.addToTrajectory();
    console.log('✓ Added operations to trajectory');

    // Finalize with high success score
    jj.finalizeTrajectory(0.9, 'Successfully created feature with clean code');
    console.log('✓ Finalized trajectory with success score 0.9');

    return jj;
}

async function testLearningStats(jj) {
    console.log('\n=== Test 2: Learning Statistics ===');

    const statsJson = jj.getLearningStats();
    const stats = JSON.parse(statsJson);

    console.log('Learning Stats:', JSON.stringify(stats, null, 2));

    assert(stats.totalTrajectories >= 1, 'Should have at least 1 trajectory');
    assert(stats.avgSuccessRate >= 0, 'Average success rate should be >= 0');
    assert(stats.avgSuccessRate <= 1, 'Average success rate should be <= 1');

    console.log(`✓ Total trajectories: ${stats.totalTrajectories}`);
    console.log(`✓ Average success rate: ${(stats.avgSuccessRate * 100).toFixed(1)}%`);
    console.log(`✓ Total patterns: ${stats.totalPatterns}`);
}

async function testPatternDiscovery(jj) {
    console.log('\n=== Test 3: Pattern Discovery ===');

    // Create multiple trajectories with similar operations
    for (let i = 0; i < 3; i++) {
        jj.startTrajectory(`Feature development ${i + 1}`);

        try {
            await jj.execute(['--version']);
            await jj.execute(['--help']);
        } catch (e) {
            // Expected
        }

        jj.addToTrajectory();
        jj.finalizeTrajectory(0.85 + i * 0.05, `Iteration ${i + 1} completed successfully`);
    }

    const patternsJson = jj.getPatterns();
    const patterns = JSON.parse(patternsJson);

    console.log(`✓ Discovered ${patterns.length} pattern(s)`);

    if (patterns.length > 0) {
        const pattern = patterns[0];
        console.log(`  - Pattern: ${pattern.name}`);
        console.log(`  - Success rate: ${(pattern.successRate * 100).toFixed(1)}%`);
        console.log(`  - Observations: ${pattern.observationCount}`);
        console.log(`  - Confidence: ${(pattern.confidence * 100).toFixed(1)}%`);
        console.log(`  - Avg duration: ${pattern.avgDurationMs.toFixed(2)}ms`);

        assert(pattern.successRate >= 0.7, 'Pattern success rate should be >= 0.7');
        assert(pattern.observationCount >= 1, 'Pattern should have at least 1 observation');
    }
}

async function testDecisionSuggestion(jj) {
    console.log('\n=== Test 4: Decision Suggestions ===');

    const suggestionJson = jj.getSuggestion('Feature development task');
    const suggestion = JSON.parse(suggestionJson);

    console.log('Decision Suggestion:', JSON.stringify(suggestion, null, 2));

    assert(suggestion.confidence !== undefined, 'Should have confidence score');
    assert(suggestion.expectedSuccessRate !== undefined, 'Should have expected success rate');
    assert(Array.isArray(suggestion.recommendedOperations), 'Should have recommended operations');

    console.log(`✓ Confidence: ${(suggestion.confidence * 100).toFixed(1)}%`);
    console.log(`✓ Expected success: ${(suggestion.expectedSuccessRate * 100).toFixed(1)}%`);
    console.log(`✓ Reasoning: ${suggestion.reasoning}`);

    if (suggestion.recommendedOperations.length > 0) {
        console.log(`✓ Recommended operations: ${suggestion.recommendedOperations.join(', ')}`);
    }
}

async function testTrajectoryQuery(jj) {
    console.log('\n=== Test 5: Trajectory Query ===');

    const trajectoriesJson = jj.queryTrajectories('Feature', 5);
    const trajectories = JSON.parse(trajectoriesJson);

    console.log(`✓ Found ${trajectories.length} similar trajectory(ies)`);

    trajectories.forEach((traj, idx) => {
        console.log(`\n  Trajectory ${idx + 1}:`);
        console.log(`    - Task: ${traj.task}`);
        console.log(`    - Success score: ${(traj.successScore * 100).toFixed(1)}%`);
        console.log(`    - Operations: ${traj.operations.length}`);
        console.log(`    - Duration: ${((new Date(traj.completedAt) - new Date(traj.startedAt)) / 1000).toFixed(2)}s`);
        if (traj.critique) {
            console.log(`    - Critique: ${traj.critique}`);
        }
    });

    assert(Array.isArray(trajectories), 'Should return array of trajectories');
}

async function testLearningProgression() {
    console.log('\n=== Test 6: Learning Progression ===');

    const jj = new JjWrapper();

    // Simulate learning over multiple iterations
    const tasks = [
        { task: 'Bug fix', success: 0.6 },
        { task: 'Bug fix', success: 0.7 },
        { task: 'Bug fix', success: 0.8 },
        { task: 'Bug fix', success: 0.9 },
    ];

    console.log('Simulating learning progression...');

    for (const { task, success } of tasks) {
        jj.startTrajectory(task);

        try {
            await jj.execute(['--version']);
        } catch (e) {
            // Expected
        }

        jj.addToTrajectory();
        jj.finalizeTrajectory(success, `Improved to ${(success * 100).toFixed(0)}% success`);
    }

    const stats = JSON.parse(jj.getLearningStats());
    console.log(`✓ Average success rate: ${(stats.avgSuccessRate * 100).toFixed(1)}%`);
    console.log(`✓ Total trajectories: ${stats.totalTrajectories}`);

    // Check if patterns were discovered
    const patterns = JSON.parse(jj.getPatterns());
    if (patterns.length > 0) {
        console.log(`✓ Discovered ${patterns.length} pattern(s) from repeated task`);
        const pattern = patterns[0];
        console.log(`  - Pattern confidence: ${(pattern.confidence * 100).toFixed(1)}%`);
    }

    assert(stats.totalTrajectories >= 4, 'Should have all trajectories recorded');
}

async function testFailureTracking() {
    console.log('\n=== Test 7: Failure Tracking ===');

    const jj = new JjWrapper();

    // Record a failure
    jj.startTrajectory('Complex refactoring');

    try {
        await jj.execute(['invalid-command']);
    } catch (e) {
        // Expected failure
    }

    jj.addToTrajectory();
    jj.finalizeTrajectory(0.3, 'Failed due to complexity, need to break into smaller steps');

    const stats = JSON.parse(jj.getLearningStats());
    console.log(`✓ Recorded failure with success score 0.3`);
    console.log(`✓ Learning from failures enabled`);

    // Check that low-success trajectory was stored
    const trajectories = JSON.parse(jj.queryTrajectories('refactoring', 5));
    const failedTrajectory = trajectories.find(t => t.successScore < 0.5);

    if (failedTrajectory) {
        console.log(`✓ Failure trajectory found with critique: "${failedTrajectory.critique}"`);
        assert(failedTrajectory.critique, 'Failed trajectory should have critique');
    }
}

async function testReset() {
    console.log('\n=== Test 8: Reset Learning ===');

    const jj = new JjWrapper();

    // Add some data
    jj.startTrajectory('Test task');
    jj.finalizeTrajectory(0.8);

    let stats = JSON.parse(jj.getLearningStats());
    const beforeReset = stats.totalTrajectories;
    console.log(`Before reset: ${beforeReset} trajectory(ies)`);

    // Reset
    jj.resetLearning();

    stats = JSON.parse(jj.getLearningStats());
    console.log(`After reset: ${stats.totalTrajectories} trajectory(ies)`);

    assert(stats.totalTrajectories === 0, 'Should have 0 trajectories after reset');
    assert(stats.totalPatterns === 0, 'Should have 0 patterns after reset');

    console.log('✓ Learning successfully reset');
}

async function testConcurrentTrajectories() {
    console.log('\n=== Test 9: Sequential Trajectory Management ===');

    const jj = new JjWrapper();

    // Start first trajectory
    const id1 = jj.startTrajectory('Task 1');
    console.log(`✓ Started trajectory 1: ${id1}`);

    // Start second trajectory (should replace first if not finalized)
    const id2 = jj.startTrajectory('Task 2');
    console.log(`✓ Started trajectory 2: ${id2}`);

    // Finalize second
    jj.finalizeTrajectory(0.9);
    console.log('✓ Finalized trajectory 2');

    const stats = JSON.parse(jj.getLearningStats());
    console.log(`✓ Total trajectories stored: ${stats.totalTrajectories}`);

    assert(stats.totalTrajectories >= 1, 'Should have at least one trajectory');
}

// Main test runner
async function runAllTests() {
    console.log('╔═══════════════════════════════════════════════════════════╗');
    console.log('║         ReasoningBank Advanced Features Test Suite        ║');
    console.log('╚═══════════════════════════════════════════════════════════╝');

    try {
        const jj = await testBasicTrajectory();
        await testLearningStats(jj);
        await testPatternDiscovery(jj);
        await testDecisionSuggestion(jj);
        await testTrajectoryQuery(jj);
        await testLearningProgression();
        await testFailureTracking();
        await testReset();
        await testConcurrentTrajectories();

        console.log('\n╔═══════════════════════════════════════════════════════════╗');
        console.log('║                    ✅ ALL TESTS PASSED                     ║');
        console.log('╚═══════════════════════════════════════════════════════════╝');
        console.log('\nReasoningBank Features:');
        console.log('  ✓ Trajectory tracking and management');
        console.log('  ✓ Pattern discovery from successful operations');
        console.log('  ✓ Learning statistics and progress tracking');
        console.log('  ✓ Decision suggestions based on patterns');
        console.log('  ✓ Trajectory similarity search');
        console.log('  ✓ Learning progression over time');
        console.log('  ✓ Failure tracking with critiques');
        console.log('  ✓ Learning reset capability');
        console.log('  ✓ Sequential trajectory management');

        process.exit(0);
    } catch (error) {
        console.error('\n❌ Test failed:', error.message);
        console.error(error.stack);
        process.exit(1);
    }
}

// Run tests if called directly
if (require.main === module) {
    runAllTests();
}

module.exports = { runAllTests };
