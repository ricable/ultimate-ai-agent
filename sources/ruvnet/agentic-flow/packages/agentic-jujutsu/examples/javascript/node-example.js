#!/usr/bin/env node
/**
 * Example: Using agentic-jujutsu in Node.js
 *
 * Prerequisites:
 *   - Install Jujutsu: https://github.com/jj-vcs/jj
 *   - npm install @agentic-flow/jujutsu
 */

const { JJWrapper, JJConfig } = require('@agentic-flow/jujutsu');

async function main() {
    console.log('🚀 Agentic-Jujutsu Node.js Example\n');

    try {
        // Create configuration
        const config = JJConfig.default()
            .with_repo_path('.')
            .with_verbose(true)
            .with_timeout(10000);

        console.log('📦 Initializing JJWrapper...');
        const jj = await JJWrapper.new(config);

        // Get repository status
        console.log('\n📊 Repository Status:');
        const status = await jj.status();
        console.log(status.stdout);

        // Get recent operations
        console.log('\n📜 Recent Operations (last 5):');
        const operations = await jj.getOperations(5);
        operations.forEach((op, idx) => {
            console.log(`\n${idx + 1}. Operation ${op.id.slice(0, 12)}`);
            console.log(`   Type: ${op.operation_type}`);
            console.log(`   User: ${op.user}`);
            console.log(`   Time: ${op.timestamp}`);
            console.log(`   Description: ${op.description}`);
        });

        // Check for conflicts
        console.log('\n⚠️  Checking for conflicts...');
        const conflicts = await jj.getConflicts();
        if (conflicts.length === 0) {
            console.log('✅ No conflicts found');
        } else {
            console.log(`❌ Found ${conflicts.length} conflicts:`);
            conflicts.forEach(conflict => {
                console.log(`   - ${conflict.path} (${conflict.num_hunks} hunks)`);
            });
        }

        // List branches
        console.log('\n🌿 Branches:');
        const branches = await jj.branch_list();
        branches.forEach(branch => {
            const remote = branch.remote ? ` (remote: ${branch.remote_name})` : '';
            console.log(`   - ${branch.name} → ${branch.target.slice(0, 12)}${remote}`);
        });

        // Execute custom command
        console.log('\n🔧 Custom command (jj log -r @):');
        const logResult = await jj.execute(['log', '-r', '@']);
        console.log(logResult.stdout);

        console.log('\n✅ All operations completed successfully!');

    } catch (error) {
        console.error('❌ Error:', error.message());
        if (error.is_recoverable && error.is_recoverable()) {
            console.log('💡 This error might be recoverable. Try checking your repository state.');
        }
        process.exit(1);
    }
}

// Run the example
if (require.main === module) {
    main().catch(err => {
        console.error('Fatal error:', err);
        process.exit(1);
    });
}

module.exports = { main };
