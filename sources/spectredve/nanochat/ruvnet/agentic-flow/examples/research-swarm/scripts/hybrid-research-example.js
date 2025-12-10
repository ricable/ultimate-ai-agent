#!/usr/bin/env node

/**
 * Hybrid Research Example: Research-Swarm + Agentic-Flow
 *
 * Demonstrates integration between research-swarm (for research) and
 * agentic-flow agents (for domain expertise and implementation).
 *
 * Usage:
 *   node scripts/hybrid-research-example.js "Build authentication system"
 */

import { decomposeTask } from '../lib/swarm-decomposition.js';
import { executeSwarm } from '../lib/swarm-executor.js';
import { spawn } from 'child_process';
import chalk from 'chalk';

async function hybridResearch(task, options = {}) {
  console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║          HYBRID RESEARCH: Research-Swarm + Agentic-Flow           ║'));
  console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.dim(`Task: ${task}\n`));

  // Phase 1: Research with research-swarm
  console.log(chalk.bold.yellow('🔬 PHASE 1: Multi-Perspective Research\n'));
  console.log(chalk.dim('Using research-swarm with 3-agent minimal swarm...\n'));

  const researchAgents = decomposeTask(task, {
    depth: options.depth || 5,
    timeMinutes: options.researchTime || 20,
    swarmSize: 3, // Minimal swarm for cost efficiency
    enableED2551: true,
    antiHallucination: 'high'
  });

  console.log(chalk.cyan(`Deploying ${researchAgents.length} research agents:\n`));
  researchAgents.forEach((agent, idx) => {
    console.log(chalk.dim(`   ${idx + 1}. ${agent.config.role} (${agent.config.timeMinutes}min)`));
  });
  console.log();

  const researchResults = await executeSwarm(researchAgents, {
    maxConcurrent: 3,
    verbose: options.verbose
  });

  if (!researchResults.success) {
    console.error(chalk.red('\n❌ Research phase failed. Aborting.\n'));
    process.exit(1);
  }

  console.log(chalk.green('\n✅ Research phase complete!\n'));

  // Phase 2: Domain expertise with agentic-flow (optional)
  if (options.addDomainExpert) {
    console.log(chalk.bold.yellow('🎓 PHASE 2: Domain Expert Analysis\n'));
    console.log(chalk.dim(`Using agentic-flow agent: ${options.expertAgent || 'backend-dev'}\n`));

    const expertTask = `Based on research findings, provide domain expert analysis for: ${task}`;
    await executeAgenticFlowAgent(options.expertAgent || 'backend-dev', expertTask);

    console.log(chalk.green('\n✅ Domain expert analysis complete!\n'));
  }

  // Phase 3: Implementation with agentic-flow (optional)
  if (options.generateCode) {
    console.log(chalk.bold.yellow('💻 PHASE 3: Code Generation\n'));
    console.log(chalk.dim('Using agentic-flow coder agent...\n'));

    const codeTask = `Based on research findings and best practices, generate implementation code for: ${task}`;
    await executeAgenticFlowAgent('coder', codeTask);

    console.log(chalk.green('\n✅ Code generation complete!\n'));
  }

  // Phase 4: Code review with agentic-flow (optional)
  if (options.reviewCode) {
    console.log(chalk.bold.yellow('🔍 PHASE 4: Code Review\n'));
    console.log(chalk.dim('Using agentic-flow reviewer agent...\n'));

    const reviewTask = `Review the generated code for: ${task}`;
    await executeAgenticFlowAgent('reviewer', reviewTask);

    console.log(chalk.green('\n✅ Code review complete!\n'));
  }

  // Final summary
  console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║                    HYBRID RESEARCH COMPLETE                        ║'));
  console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.bold('📊 Summary:\n'));
  console.log(chalk.dim(`   ✅ Research agents: ${researchResults.successful}/${researchResults.total}`));

  if (options.addDomainExpert) {
    console.log(chalk.dim('   ✅ Domain expert analysis: Complete'));
  }
  if (options.generateCode) {
    console.log(chalk.dim('   ✅ Code generation: Complete'));
  }
  if (options.reviewCode) {
    console.log(chalk.dim('   ✅ Code review: Complete'));
  }

  console.log(chalk.bold('\n🎯 Results available in database:\n'));
  console.log(chalk.cyan(`   npx research-swarm list\n`));
}

async function executeAgenticFlowAgent(agentType, task) {
  return new Promise((resolve, reject) => {
    const args = [
      'agentic-flow', 'agent', 'run',
      agentType,
      '--task', task,
      '--provider', process.env.PROVIDER || 'anthropic'
    ];

    console.log(chalk.dim(`   Executing: npx ${args.join(' ')}\n`));

    const child = spawn('npx', args, {
      stdio: 'inherit',
      env: process.env
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve();
      } else {
        console.log(chalk.yellow(`\n⚠️  Agent exited with code ${code}`));
        resolve(); // Don't fail entire workflow
      }
    });

    child.on('error', (error) => {
      console.error(chalk.red(`\n❌ Error executing agentic-flow agent: ${error.message}`));
      resolve(); // Don't fail entire workflow
    });
  });
}

// Parse command line arguments
const args = process.argv.slice(2);
const task = args.find(arg => !arg.startsWith('--')) || 'Build a REST API for user management';

const options = {
  depth: args.includes('--depth') ? parseInt(args[args.indexOf('--depth') + 1]) : 5,
  researchTime: args.includes('--research-time') ? parseInt(args[args.indexOf('--research-time') + 1]) : 20,
  addDomainExpert: args.includes('--add-expert'),
  expertAgent: args.includes('--expert-agent') ? args[args.indexOf('--expert-agent') + 1] : 'backend-dev',
  generateCode: args.includes('--generate-code'),
  reviewCode: args.includes('--review-code'),
  verbose: args.includes('--verbose')
};

// Display help
if (args.includes('--help') || args.includes('-h')) {
  console.log(chalk.bold.cyan('\nHybrid Research: Research-Swarm + Agentic-Flow\n'));
  console.log(chalk.bold('Usage:'));
  console.log(chalk.dim('  node scripts/hybrid-research-example.js "<task>" [options]\n'));
  console.log(chalk.bold('Options:'));
  console.log(chalk.dim('  --depth <number>              Research depth (default: 5)'));
  console.log(chalk.dim('  --research-time <minutes>     Research time budget (default: 20)'));
  console.log(chalk.dim('  --add-expert                  Add domain expert analysis (agentic-flow)'));
  console.log(chalk.dim('  --expert-agent <agent>        Expert agent type (default: backend-dev)'));
  console.log(chalk.dim('  --generate-code               Generate implementation code (agentic-flow)'));
  console.log(chalk.dim('  --review-code                 Review generated code (agentic-flow)'));
  console.log(chalk.dim('  --verbose                     Verbose output'));
  console.log(chalk.dim('  --help, -h                    Show this help\n'));
  console.log(chalk.bold('Examples:'));
  console.log(chalk.dim('  # Research only (research-swarm)'));
  console.log(chalk.cyan('  node scripts/hybrid-research-example.js "Authentication system"\n'));
  console.log(chalk.dim('  # Research + domain expert'));
  console.log(chalk.cyan('  node scripts/hybrid-research-example.js "ML pipeline" --add-expert --expert-agent ml-developer\n'));
  console.log(chalk.dim('  # Full workflow: research + code + review'));
  console.log(chalk.cyan('  node scripts/hybrid-research-example.js "REST API" --add-expert --generate-code --review-code\n'));
  console.log(chalk.bold('Available Agentic-Flow Agents:'));
  console.log(chalk.dim('  backend-dev, ml-developer, system-architect, code-analyzer,'));
  console.log(chalk.dim('  coder, reviewer, tester, planner, mobile-dev, cicd-engineer\n'));
  process.exit(0);
}

// Run hybrid research
console.log(chalk.dim('\nStarting hybrid research workflow...\n'));
hybridResearch(task, options).catch(error => {
  console.error(chalk.red(`\n❌ Hybrid research failed: ${error.message}\n`));
  console.error(chalk.dim(error.stack));
  process.exit(1);
});
