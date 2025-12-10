/**
 * Example: Concurrent agent execution with conflict detection
 *
 * This example shows how multiple agents can work concurrently
 * using jj's conflict-free merge capabilities.
 */

import { JJConfig } from '../../typescript';
import { createHooksIntegration, JJHooksIntegration } from '../../typescript/hooks-integration';

interface AgentTask {
    agentId: string;
    task: string;
    files: string[];
}

async function executeAgent(
    config: JJConfig,
    sessionId: string,
    agentTask: AgentTask
): Promise<void> {
    const { agentId, task, files } = agentTask;

    console.log(`\n🤖 [${agentId}] Starting task: ${task}`);

    const integration = await createHooksIntegration(
        config,
        sessionId,
        agentId,
        true
    );

    // Pre-task
    await integration.onPreTask(task);
    console.log(`  ✅ [${agentId}] Pre-task hook executed`);

    // Simulate work with delays
    for (const file of files) {
        // Simulate thinking/processing time
        await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));

        await integration.onPostEdit(file, `Updated ${file}`);
        console.log(`  📝 [${agentId}] Edited: ${file}`);
    }

    // Post-task
    const operations = await integration.onPostTask();
    console.log(`  ✅ [${agentId}] Completed with ${operations.length} operations`);
}

async function main() {
    console.log('⚡ Concurrent Agents Example');
    console.log('============================\n');

    const sessionId = `concurrent-${Date.now()}`;
    const config: JJConfig = {
        jjPath: 'jj',
        repoPath: '.',
        timeoutMs: 30000,
        verbose: false,
        maxLogEntries: 1000,
        enableAgentdbSync: true,
    };

    console.log(`📋 Session: ${sessionId}`);

    // Define concurrent agent tasks
    const agentTasks: AgentTask[] = [
        {
            agentId: 'backend-dev',
            task: 'Implement REST API endpoints',
            files: ['src/api/routes.rs', 'src/api/handlers.rs', 'src/api/middleware.rs'],
        },
        {
            agentId: 'frontend-dev',
            task: 'Create UI components',
            files: ['ui/src/App.tsx', 'ui/src/components/Auth.tsx', 'ui/src/api/client.ts'],
        },
        {
            agentId: 'db-architect',
            task: 'Design database schema',
            files: ['migrations/001_initial.sql', 'migrations/002_auth.sql', 'src/db/models.rs'],
        },
        {
            agentId: 'test-engineer',
            task: 'Write integration tests',
            files: ['tests/api_test.rs', 'tests/db_test.rs', 'tests/integration_test.rs'],
        },
    ];

    console.log(`👥 Agents: ${agentTasks.length}\n`);

    // Execute all agents concurrently
    const startTime = Date.now();

    await Promise.all(
        agentTasks.map(task => executeAgent(config, sessionId, task))
    );

    const duration = Date.now() - startTime;

    console.log('\n✨ All agents completed successfully!');
    console.log(`⏱️  Total time: ${(duration / 1000).toFixed(2)}s`);

    console.log('\n🔍 Conflict Detection...');

    // Check for conflicts (in real scenario, jj would handle this)
    const integration = await createHooksIntegration(
        config,
        sessionId,
        'conflict-detector',
        true
    );

    await integration.onPreTask('Detect conflicts');

    // Simulate conflict detection
    const conflicts: string[] = [];

    if (conflicts.length > 0) {
        await integration.onConflictDetected(conflicts);
        console.log('  ⚠️  Conflicts found - resolution required');
    } else {
        console.log('  ✅ No conflicts detected - clean merge!');
    }

    await integration.onPostTask();

    console.log('\n💡 Benefits of jj for concurrent agents:');
    console.log('  1. Conflict-free concurrent edits');
    console.log('  2. Full operation history for all agents');
    console.log('  3. Easy merge and synchronization');
    console.log('  4. Learn from concurrent collaboration patterns');
}

// Run example
main().catch(console.error);
