/**
 * Example: Learning from jj operation history
 *
 * This example demonstrates how to use AgentDB to learn from
 * historical jj operations and improve future decision-making.
 */

import { JJWrapper, JJConfig } from '../../typescript';
import { createHooksIntegration, withHooks } from '../../typescript/hooks-integration';

async function main() {
    console.log('🧠 AgentDB Learning Example');
    console.log('===========================\n');

    // Create configuration
    const config: JJConfig = {
        jjPath: 'jj',
        repoPath: '.',
        timeoutMs: 30000,
        verbose: true,
        maxLogEntries: 1000,
        enableAgentdbSync: true,
    };

    // Create wrapper
    const jj = await JJWrapper.new(config);

    console.log('📊 Fetching historical operations...');
    const historicalOps = await jj.getOperations(100);
    console.log(`  Found ${historicalOps.length} historical operations\n`);

    // Simulate storing in AgentDB (would use MCP in real implementation)
    console.log('💾 Simulating AgentDB storage...');
    for (const op of historicalOps.slice(0, 10)) {
        const episode = {
            sessionId: 'history-import',
            task: op.description,
            agentId: 'history-importer',
            operation: op,
            success: true,
            reward: 1.0,
            timestamp: op.timestamp,
        };

        console.log(`  ✅ Stored: ${op.description.slice(0, 60)}...`);

        // In real implementation, would call:
        // await agentdb.storeEpisode(episode);
    }

    console.log('\n🔍 Simulating pattern search...');
    const searchQuery = 'refactor authentication';
    console.log(`  Query: "${searchQuery}"`);

    // In real implementation, would call:
    // const similarOps = await agentdb.searchPatterns(searchQuery, 5);
    const similarOps = historicalOps
        .filter(op => op.description.toLowerCase().includes('auth'))
        .slice(0, 5);

    console.log(`  Found ${similarOps.length} similar operations:`);
    for (const op of similarOps) {
        console.log(`    - ${op.description}`);
    }

    // Demonstrate using hooks integration with learning
    console.log('\n🚀 Demonstrating learned workflow...');

    const integration = await createHooksIntegration(
        config,
        'learning-session-001',
        'learning-agent',
        true
    );

    const { result, operations } = await withHooks(
        integration,
        'Implement learned authentication pattern',
        async (hooks) => {
            // Simulate work based on learned patterns
            await hooks.onPostEdit('src/auth.rs', 'Apply learned auth pattern');
            await hooks.onPostEdit('src/middleware.rs', 'Add auth middleware');
            await hooks.onPostEdit('tests/auth_test.rs', 'Add learned test cases');

            return { success: true, filesModified: 3 };
        }
    );

    console.log('\n✅ Workflow completed with learned patterns');
    console.log(`  Result: ${JSON.stringify(result)}`);
    console.log(`  Operations: ${operations.length}`);

    console.log('\n💡 Key Insights:');
    console.log('  1. Historical operations stored in AgentDB');
    console.log('  2. Semantic search finds similar past work');
    console.log('  3. Agents learn from successful patterns');
    console.log('  4. Future work is informed by past experience');
}

// Run example
main().catch(console.error);
