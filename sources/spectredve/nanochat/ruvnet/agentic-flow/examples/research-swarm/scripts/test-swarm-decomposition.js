#!/usr/bin/env node

/**
 * Test Swarm Decomposition Logic
 * Validates that swarm decomposition works correctly without making API calls
 */

import chalk from 'chalk';
import { decomposeTask, validateSwarmConfig, getSwarmRoles } from '../lib/swarm-decomposition.js';

console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
console.log(chalk.bold.cyan('║              SWARM DECOMPOSITION TEST                              ║'));
console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

const testCases = [
  {
    name: 'Simple Task (Depth 3)',
    task: 'What are REST APIs?',
    options: { depth: 3, timeMinutes: 15, swarmSize: 5 }
  },
  {
    name: 'Medium Task (Depth 5)',
    task: 'Compare microservices vs monolithic architecture',
    options: { depth: 5, timeMinutes: 30, swarmSize: 5 }
  },
  {
    name: 'Complex Task (Depth 8)',
    task: 'Analyze quantum computing impact on cryptography',
    options: { depth: 8, timeMinutes: 60, swarmSize: 5 }
  },
  {
    name: 'Custom Swarm Size (3 agents)',
    task: 'Quick research task',
    options: { depth: 5, timeMinutes: 20, swarmSize: 3 }
  }
];

console.log(chalk.yellow('🧪 Testing swarm decomposition logic...\n'));

let allTestsPassed = true;

for (const testCase of testCases) {
  console.log(chalk.bold.cyan(`\n${'─'.repeat(71)}`));
  console.log(chalk.bold.cyan(`TEST: ${testCase.name}`));
  console.log(chalk.bold.cyan(`${'─'.repeat(71)}\n`));

  console.log(chalk.dim(`Task: ${testCase.task}`));
  console.log(chalk.dim(`Options: depth=${testCase.options.depth}, time=${testCase.options.timeMinutes}min, swarmSize=${testCase.options.swarmSize}\n`));

  try {
    // Decompose task
    const agents = decomposeTask(testCase.task, testCase.options);

    console.log(chalk.green(`✅ Decomposition successful`));
    console.log(chalk.dim(`   Generated ${agents.length} agents\n`));

    // Display agents
    console.log(chalk.bold('🤖 Generated Agents:\n'));
    agents.forEach((agent, idx) => {
      const roleEmoji = getRoleEmoji(agent.config.role);
      console.log(chalk.cyan(`   ${idx + 1}. ${roleEmoji} ${agent.config.role.toUpperCase().replace('-', ' ')}`));
      console.log(chalk.dim(`      Priority: ${agent.config.priority}`));
      console.log(chalk.dim(`      Depth: ${agent.config.depth}`));
      console.log(chalk.dim(`      Time: ${agent.config.timeMinutes}min`));
      console.log(chalk.dim(`      Focus: ${agent.config.focus}`));
      console.log(chalk.dim(`      Task: ${agent.task.substring(0, 80)}...`));
      console.log();
    });

    // Validate configuration
    const validation = validateSwarmConfig(agents);

    if (validation.valid) {
      console.log(chalk.green('✅ Configuration validation passed'));
    } else {
      console.log(chalk.red('❌ Configuration validation failed:'));
      validation.errors.forEach(err => console.log(chalk.red(`   • ${err}`)));
      allTestsPassed = false;
    }

    if (validation.warnings.length > 0) {
      console.log(chalk.yellow('\n⚠️  Warnings:'));
      validation.warnings.forEach(warn => console.log(chalk.yellow(`   • ${warn}`)));
    }

    // Verify expected agent count
    const expectedCount = getExpectedAgentCount(testCase.options.depth, testCase.options.swarmSize);
    if (agents.length === expectedCount) {
      console.log(chalk.green(`\n✅ Agent count matches expected: ${expectedCount}`));
    } else {
      console.log(chalk.red(`\n❌ Agent count mismatch: expected ${expectedCount}, got ${agents.length}`));
      allTestsPassed = false;
    }

    // Verify priority distribution
    const priorities = agents.map(a => a.config.priority);
    const hasPriority1 = priorities.includes(1);
    const hasPriority3 = priorities.includes(3);

    if (hasPriority1 && hasPriority3) {
      console.log(chalk.green('✅ Priority distribution correct (has research and synthesis)'));
    } else {
      console.log(chalk.red('❌ Priority distribution incorrect'));
      allTestsPassed = false;
    }

    // Verify synthesizer exists
    const hasSynthesizer = agents.some(a => a.config.role === 'synthesizer');
    if (hasSynthesizer) {
      console.log(chalk.green('✅ Synthesizer agent present'));
    } else {
      console.log(chalk.red('❌ Missing synthesizer agent'));
      allTestsPassed = false;
    }

  } catch (error) {
    console.log(chalk.red(`❌ Test failed: ${error.message}`));
    console.log(chalk.dim(error.stack));
    allTestsPassed = false;
  }
}

// Display swarm roles
console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
console.log(chalk.bold.cyan('║                    AVAILABLE SWARM ROLES                           ║'));
console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

const roles = getSwarmRoles();
Object.entries(roles).forEach(([id, role]) => {
  const emoji = getRoleEmoji(id);
  console.log(chalk.cyan(`${emoji} ${role.name}`));
  console.log(chalk.dim(`   ${role.description}`));
  console.log(chalk.dim(`   Focus: ${role.focus} | Priority: ${role.priority}\n`));
});

// Final summary
console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
console.log(chalk.bold.cyan('║                    TEST SUMMARY                                    ║'));
console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

if (allTestsPassed) {
  console.log(chalk.bold.green('✅ ALL TESTS PASSED\n'));
  console.log(chalk.green('Swarm decomposition logic is working correctly!'));
  console.log(chalk.dim('\nReady to run full benchmark with actual API calls.\n'));
  process.exit(0);
} else {
  console.log(chalk.bold.red('❌ SOME TESTS FAILED\n'));
  console.log(chalk.red('Please fix the issues before running full benchmark.\n'));
  process.exit(1);
}

function getRoleEmoji(role) {
  const emojis = {
    'explorer': '🔍',
    'depth-analyst': '🔬',
    'verifier': '✅',
    'trend-analyst': '📈',
    'domain-expert': '🎓',
    'critic': '🔎',
    'synthesizer': '🧩'
  };
  return emojis[role] || '🤖';
}

function getExpectedAgentCount(depth, requestedSize) {
  // Adaptive sizing logic - synthesizer is already included in count
  if (requestedSize !== 5) {
    return Math.max(3, Math.min(7, requestedSize));
  }

  if (depth <= 3) return 3;
  if (depth <= 6) return 5;
  return 7;
}
