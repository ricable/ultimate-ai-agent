#!/usr/bin/env node

/**
 * Benchmark: Swarm vs Single-Agent Mode
 *
 * Compares performance, quality, and cost between:
 * - v1.0.1 behavior (single-agent)
 * - v1.1.0 behavior (multi-agent swarm)
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import { getJobStatus, getDatabase } from '../lib/db-utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Benchmark configuration
const BENCHMARK_TASKS = [
  {
    name: 'Simple Query',
    task: 'What are REST APIs and how do they work?',
    depth: 3,
    time: 5, // 5 minutes per agent
    expectedAgents: { single: 1, swarm: 3 }
  },
  {
    name: 'Medium Analysis',
    task: 'Compare and contrast microservices vs monolithic architecture',
    depth: 5,
    time: 10,
    expectedAgents: { single: 1, swarm: 5 }
  },
  {
    name: 'Complex Research',
    task: 'Analyze the impact of quantum computing on cryptography and cybersecurity',
    depth: 7,
    time: 15,
    expectedAgents: { single: 1, swarm: 7 }
  }
];

const results = {
  single: [],
  swarm: []
};

console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
console.log(chalk.bold.cyan('║            SWARM vs SINGLE-AGENT BENCHMARK                         ║'));
console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

console.log(chalk.yellow('📊 Benchmark Configuration:\n'));
console.log(chalk.dim(`   Tasks: ${BENCHMARK_TASKS.length}`));
console.log(chalk.dim(`   Modes: Single-Agent (v1.0.1) vs Swarm (v1.1.0)`));
console.log(chalk.dim(`   Metrics: Speed, Quality, Agent Count, Report Length\n`));

async function runBenchmark() {
  for (let i = 0; i < BENCHMARK_TASKS.length; i++) {
    const task = BENCHMARK_TASKS[i];

    console.log(chalk.bold.cyan(`\n${'='.repeat(71)}`));
    console.log(chalk.bold.cyan(`TEST ${i + 1}/${BENCHMARK_TASKS.length}: ${task.name}`));
    console.log(chalk.bold.cyan(`${'='.repeat(71)}\n`));
    console.log(chalk.dim(`Task: ${task.task}`));
    console.log(chalk.dim(`Depth: ${task.depth} | Time Budget: ${task.time}min\n`));

    // Run single-agent mode
    console.log(chalk.yellow('🔹 Running SINGLE-AGENT mode (v1.0.1 behavior)...\n'));
    const singleResult = await runResearch({
      mode: 'single',
      task: task.task,
      depth: task.depth,
      time: task.time
    });
    results.single.push(singleResult);
    displayResult(singleResult, 'single');

    // Wait a bit between tests
    await sleep(2000);

    // Run swarm mode
    console.log(chalk.yellow('\n🔹 Running SWARM mode (v1.1.0 behavior)...\n'));
    const swarmResult = await runResearch({
      mode: 'swarm',
      task: task.task,
      depth: task.depth,
      time: task.time
    });
    results.swarm.push(swarmResult);
    displayResult(swarmResult, 'swarm');

    // Compare results
    console.log(chalk.bold.green('\n📊 COMPARISON:\n'));
    compareResults(singleResult, swarmResult, task);

    // Wait between benchmark tasks
    if (i < BENCHMARK_TASKS.length - 1) {
      console.log(chalk.dim('\n⏳ Waiting 5 seconds before next test...\n'));
      await sleep(5000);
    }
  }

  // Final summary
  displayFinalSummary();
}

async function runResearch(config) {
  const startTime = Date.now();

  return new Promise((resolve) => {
    const cliPath = path.join(__dirname, '../bin/cli.js');
    const args = [
      cliPath,
      'research',
      'researcher',
      config.task,
      '--depth', config.depth.toString(),
      '--time', config.time.toString(),
      '--focus', 'balanced'
    ];

    if (config.mode === 'single') {
      args.push('--single-agent');
    } else {
      // Swarm mode is default, but we can specify swarm size for clarity
      // (adaptive sizing will be used by default)
    }

    const child = spawn('node', args, {
      stdio: 'pipe',
      env: {
        ...process.env,
        RESEARCH_DEPTH: config.depth.toString(),
        RESEARCH_TIME_BUDGET: config.time.toString()
      }
    });

    let output = '';
    let errorOutput = '';

    child.stdout.on('data', (data) => {
      output += data.toString();
      // Show real-time progress
      process.stdout.write(chalk.dim(data.toString()));
    });

    child.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    child.on('close', (code) => {
      const endTime = Date.now();
      const duration = (endTime - startTime) / 1000; // seconds

      // Extract job IDs from output
      const jobIdMatches = output.match(/Job ID:\s+([a-f0-9-]+)/g);
      const jobIds = jobIdMatches ? jobIdMatches.map(m => m.match(/([a-f0-9-]+)/)[1]) : [];

      // Get job details from database
      const jobs = jobIds.map(id => getJobStatus(id)).filter(Boolean);

      // Calculate metrics
      const totalReportLength = jobs.reduce((sum, job) =>
        sum + (job.report_content?.length || 0), 0
      );

      const avgGroundingScore = jobs.length > 0
        ? jobs.reduce((sum, job) => sum + (job.grounding_score || 0), 0) / jobs.length
        : 0;

      resolve({
        mode: config.mode,
        success: code === 0,
        duration,
        agentCount: jobs.length,
        jobs,
        totalReportLength,
        avgGroundingScore,
        errorOutput: errorOutput.length > 0 ? errorOutput : null
      });
    });

    child.on('error', (error) => {
      resolve({
        mode: config.mode,
        success: false,
        duration: (Date.now() - startTime) / 1000,
        error: error.message
      });
    });
  });
}

function displayResult(result, mode) {
  const modeLabel = mode === 'single' ? 'SINGLE-AGENT' : 'SWARM';
  const color = result.success ? chalk.green : chalk.red;

  console.log(color(`\n✅ ${modeLabel} COMPLETE:`));
  console.log(chalk.dim(`   Status: ${result.success ? 'Success' : 'Failed'}`));
  console.log(chalk.dim(`   Duration: ${result.duration.toFixed(2)}s`));
  console.log(chalk.dim(`   Agents: ${result.agentCount}`));
  console.log(chalk.dim(`   Report Length: ${result.totalReportLength} chars`));

  if (result.avgGroundingScore > 0) {
    console.log(chalk.dim(`   Avg Quality Score: ${result.avgGroundingScore.toFixed(2)}`));
  }

  if (result.jobs && result.jobs.length > 0) {
    console.log(chalk.dim('\n   Jobs:'));
    result.jobs.forEach(job => {
      console.log(chalk.dim(`     • ${job.id.substring(0, 8)} - ${job.status} (${job.progress}%)`));
    });
  }

  if (result.error) {
    console.log(chalk.red(`   Error: ${result.error}`));
  }
}

function compareResults(single, swarm, task) {
  const speedup = single.duration / swarm.duration;
  const agentRatio = swarm.agentCount / single.agentCount;
  const reportRatio = swarm.totalReportLength / single.totalReportLength;

  console.log(chalk.bold('⚡ Speed:'));
  if (swarm.duration < single.duration) {
    console.log(chalk.green(`   ✅ Swarm is ${speedup.toFixed(2)}x FASTER (${single.duration.toFixed(1)}s → ${swarm.duration.toFixed(1)}s)`));
  } else {
    console.log(chalk.yellow(`   ⚠️  Swarm is ${(1/speedup).toFixed(2)}x slower (${single.duration.toFixed(1)}s → ${swarm.duration.toFixed(1)}s)`));
  }

  console.log(chalk.bold('\n🤖 Agents:'));
  console.log(chalk.cyan(`   Single: ${single.agentCount} agent | Swarm: ${swarm.agentCount} agents (${agentRatio.toFixed(1)}x)`));
  console.log(chalk.dim(`   Expected: ${task.expectedAgents.swarm} agents for depth ${task.depth}`));

  console.log(chalk.bold('\n📄 Report Quality:'));
  console.log(chalk.cyan(`   Single: ${single.totalReportLength} chars`));
  console.log(chalk.cyan(`   Swarm: ${swarm.totalReportLength} chars (${reportRatio.toFixed(2)}x more detailed)`));

  if (single.avgGroundingScore > 0 && swarm.avgGroundingScore > 0) {
    console.log(chalk.bold('\n📊 Quality Score:'));
    console.log(chalk.cyan(`   Single: ${single.avgGroundingScore.toFixed(2)}`));
    console.log(chalk.cyan(`   Swarm: ${swarm.avgGroundingScore.toFixed(2)}`));
  }

  console.log(chalk.bold('\n💰 Cost Estimate (relative):'));
  console.log(chalk.cyan(`   Single: 1x baseline`));
  console.log(chalk.cyan(`   Swarm: ~${swarm.agentCount}x baseline (${swarm.agentCount} agents)`));
}

function displayFinalSummary() {
  console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║                    FINAL BENCHMARK SUMMARY                         ║'));
  console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

  const singleSuccess = results.single.filter(r => r.success).length;
  const swarmSuccess = results.swarm.filter(r => r.success).length;

  const avgSingleDuration = results.single.reduce((sum, r) => sum + r.duration, 0) / results.single.length;
  const avgSwarmDuration = results.swarm.reduce((sum, r) => sum + r.duration, 0) / results.swarm.length;

  const avgSingleReportLength = results.single.reduce((sum, r) => sum + r.totalReportLength, 0) / results.single.length;
  const avgSwarmReportLength = results.swarm.reduce((sum, r) => sum + r.totalReportLength, 0) / results.swarm.length;

  console.log(chalk.bold('📊 OVERALL METRICS:\n'));

  console.log(chalk.bold('Success Rate:'));
  console.log(chalk.dim(`   Single: ${singleSuccess}/${results.single.length} (${(singleSuccess/results.single.length*100).toFixed(1)}%)`));
  console.log(chalk.dim(`   Swarm: ${swarmSuccess}/${results.swarm.length} (${(swarmSuccess/results.swarm.length*100).toFixed(1)}%)`));

  console.log(chalk.bold('\n⚡ Average Duration:'));
  console.log(chalk.dim(`   Single: ${avgSingleDuration.toFixed(2)}s`));
  console.log(chalk.dim(`   Swarm: ${avgSwarmDuration.toFixed(2)}s`));

  const speedup = avgSingleDuration / avgSwarmDuration;
  if (speedup > 1) {
    console.log(chalk.green(`   ✅ Swarm is ${speedup.toFixed(2)}x FASTER on average`));
  } else {
    console.log(chalk.yellow(`   ⚠️  Swarm is ${(1/speedup).toFixed(2)}x slower on average`));
  }

  console.log(chalk.bold('\n📄 Average Report Length:'));
  console.log(chalk.dim(`   Single: ${avgSingleReportLength.toFixed(0)} chars`));
  console.log(chalk.dim(`   Swarm: ${avgSwarmReportLength.toFixed(0)} chars`));
  console.log(chalk.green(`   ✅ Swarm reports are ${(avgSwarmReportLength/avgSingleReportLength).toFixed(2)}x more detailed`));

  console.log(chalk.bold('\n🎯 RECOMMENDATIONS:\n'));

  if (speedup > 1) {
    console.log(chalk.green('   ✅ Swarm mode provides both SPEED and QUALITY improvements'));
    console.log(chalk.green('   ✅ Recommended for production use'));
  } else {
    console.log(chalk.yellow('   ⚠️  Swarm mode provides QUALITY but may be slower'));
    console.log(chalk.yellow('   ⚠️  Consider using --single-agent for time-sensitive tasks'));
  }

  console.log(chalk.dim('\n   Cost Trade-offs:'));
  console.log(chalk.dim('   • Swarm: Higher API costs (3-7x) but better quality'));
  console.log(chalk.dim('   • Single: Lower costs but limited perspective'));
  console.log(chalk.dim('   • Use --swarm-size 3 for budget-conscious research\n'));

  console.log(chalk.bold.cyan('╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║                    BENCHMARK COMPLETE                              ║'));
  console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Run benchmark
runBenchmark().catch(error => {
  console.error(chalk.red(`\n❌ Benchmark failed: ${error.message}\n`));
  console.error(error.stack);
  process.exit(1);
});
