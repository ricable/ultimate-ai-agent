#!/usr/bin/env node

/**
 * Self-Learning Benchmark Comparison
 *
 * Compares research performance WITH and WITHOUT AgentDB self-learning capabilities:
 *
 * Test 1: Without Self-Learning (ENABLE_REASONINGBANK=false)
 * Test 2: With Self-Learning (ENABLE_REASONINGBANK=true)
 *
 * Measures:
 * - Execution time
 * - Quality metrics (grounding score, citations)
 * - Token usage
 * - Memory usage
 * - Report completeness
 * - Pattern recognition improvements
 *
 * Usage:
 *   node scripts/self-learning-benchmark.js
 *   node scripts/self-learning-benchmark.js --tasks 5
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { getDatabase } from '../lib/db-utils.js';
import chalk from 'chalk';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const NUM_TASKS = parseInt(process.argv.find(arg => arg.startsWith('--tasks='))?.split('=')[1] || '5');

// Research tasks for benchmarking
const BENCHMARK_TASKS = [
  "Compare machine learning vs deep learning with 3 key differences",
  "List 5 cloud computing trends for 2024",
  "What are the benefits of TypeScript vs JavaScript?",
  "Explain REST API vs GraphQL in 3 points",
  "Compare NoSQL vs SQL databases with examples",
  "List 4 advantages of containerization with Docker",
  "What is serverless computing and its benefits?",
  "Compare React vs Vue.js framework differences"
];

console.log(chalk.bold.cyan(`
╔═══════════════════════════════════════════════════════════════════╗
║           SELF-LEARNING BENCHMARK COMPARISON                       ║
╚═══════════════════════════════════════════════════════════════════╝

🧪 Testing Impact of AgentDB Self-Learning

📊 Configuration:
   Tasks per phase: ${NUM_TASKS}
   Total tasks: ${NUM_TASKS * 2}

📝 Test Phases:
   Phase 1: WITHOUT self-learning (baseline)
   Phase 2: WITH self-learning (enhanced)

`));

const results = {
  withoutLearning: [],
  withLearning: []
};

/**
 * Run research task with specified configuration
 */
function runResearchTask(task, enableLearning) {
  return new Promise((resolve, reject) => {
    const env = {
      ...process.env,
      ENABLE_REASONINGBANK: enableLearning ? 'true' : 'false',
      RESEARCH_DEPTH: '5',
      RESEARCH_TIME_BUDGET: '60',
      RESEARCH_FOCUS: 'balanced',
      FORCE_COLOR: '0'
    };

    const startTime = Date.now();

    const child = spawn('node', [
      path.join(__dirname, '../run-researcher-local.js'),
      'researcher',
      task
    ], {
      env,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      const endTime = Date.now();
      const duration = (endTime - startTime) / 1000;

      // Extract job ID from output
      const jobIdMatch = stdout.match(/Job ID:\s+([a-f0-9-]+)/);
      const jobId = jobIdMatch ? jobIdMatch[1] : null;

      if (code === 0 && jobId) {
        resolve({ jobId, duration, success: true });
      } else {
        resolve({ jobId: null, duration, success: false, error: stderr });
      }
    });
  });
}

/**
 * Get job metrics from database
 */
function getJobMetrics(jobId) {
  const db = getDatabase();
  try {
    const job = db.prepare('SELECT * FROM research_jobs WHERE id = ?').get(jobId);

    if (!job) return null;

    // Count citations
    const citationCount = (job.report_content?.match(/\[.*?\]\(http.*?\)/g) || []).length;

    return {
      jobId,
      task: job.task,
      status: job.status,
      durationSeconds: job.duration_seconds,
      tokensUsed: job.tokens_used,
      memoryMb: job.memory_mb,
      groundingScore: job.grounding_score,
      reportLength: job.report_content?.length || 0,
      citationCount,
      quality: job.grounding_score * 100
    };
  } finally {
    db.close();
  }
}

/**
 * Phase 1: WITHOUT Self-Learning
 */
async function runPhase1() {
  console.log(chalk.bold.yellow('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.yellow('║   PHASE 1: WITHOUT SELF-LEARNING (Baseline)                       ║'));
  console.log(chalk.bold.yellow('╚═══════════════════════════════════════════════════════════════════╝\n'));

  const tasks = BENCHMARK_TASKS.slice(0, NUM_TASKS);

  for (let i = 0; i < tasks.length; i++) {
    console.log(chalk.cyan(`[${i + 1}/${tasks.length}] ${tasks[i].substring(0, 60)}...`));

    const result = await runResearchTask(tasks[i], false);

    if (result.success && result.jobId) {
      const metrics = getJobMetrics(result.jobId);
      results.withoutLearning.push(metrics);
      console.log(chalk.green(`   ✅ Completed in ${result.duration.toFixed(1)}s | Quality: ${metrics.quality.toFixed(0)}%\n`));
    } else {
      console.log(chalk.red(`   ❌ Failed\n`));
    }
  }
}

/**
 * Phase 2: WITH Self-Learning
 */
async function runPhase2() {
  console.log(chalk.bold.green('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.green('║   PHASE 2: WITH SELF-LEARNING (Enhanced)                          ║'));
  console.log(chalk.bold.green('╚═══════════════════════════════════════════════════════════════════╝\n'));

  const tasks = BENCHMARK_TASKS.slice(0, NUM_TASKS);

  for (let i = 0; i < tasks.length; i++) {
    console.log(chalk.cyan(`[${i + 1}/${tasks.length}] ${tasks[i].substring(0, 60)}...`));

    const result = await runResearchTask(tasks[i], true);

    if (result.success && result.jobId) {
      const metrics = getJobMetrics(result.jobId);
      results.withLearning.push(metrics);
      console.log(chalk.green(`   ✅ Completed in ${result.duration.toFixed(1)}s | Quality: ${metrics.quality.toFixed(0)}%\n`));
    } else {
      console.log(chalk.red(`   ❌ Failed\n`));
    }
  }
}

/**
 * Calculate statistics
 */
function calculateStats(data, metric) {
  const values = data.map(d => d[metric]).filter(v => v != null);
  if (values.length === 0) return { avg: 0, min: 0, max: 0 };

  return {
    avg: values.reduce((a, b) => a + b, 0) / values.length,
    min: Math.min(...values),
    max: Math.max(...values)
  };
}

/**
 * Display comparison results
 */
function displayResults() {
  console.log(chalk.bold.cyan(`
\n╔═══════════════════════════════════════════════════════════════════╗
║              BENCHMARK COMPARISON RESULTS                          ║
╚═══════════════════════════════════════════════════════════════════╝
`));

  const metrics = [
    { key: 'durationSeconds', label: 'Execution Time', unit: 's', lowerIsBetter: true },
    { key: 'quality', label: 'Quality Score', unit: '%', lowerIsBetter: false },
    { key: 'reportLength', label: 'Report Length', unit: ' chars', lowerIsBetter: false },
    { key: 'citationCount', label: 'Citations', unit: '', lowerIsBetter: false },
    { key: 'tokensUsed', label: 'Tokens Used', unit: '', lowerIsBetter: true },
    { key: 'memoryMb', label: 'Memory Usage', unit: 'MB', lowerIsBetter: true }
  ];

  console.log(chalk.bold('📊 Performance Comparison:\n'));

  metrics.forEach(metric => {
    const baseline = calculateStats(results.withoutLearning, metric.key);
    const enhanced = calculateStats(results.withLearning, metric.key);

    const improvement = ((enhanced.avg - baseline.avg) / baseline.avg) * 100;
    const isImprovement = metric.lowerIsBetter ? improvement < 0 : improvement > 0;

    console.log(chalk.bold(`${metric.label}:`));
    console.log(chalk.dim(`   Baseline (no learning):  ${baseline.avg.toFixed(2)}${metric.unit}`));
    console.log(chalk.dim(`   Enhanced (with learning): ${enhanced.avg.toFixed(2)}${metric.unit}`));

    const improvementStr = Math.abs(improvement).toFixed(1);
    if (isImprovement) {
      console.log(chalk.green(`   📈 Improvement: ${improvementStr}% ${metric.lowerIsBetter ? 'faster' : 'better'}\n`));
    } else if (Math.abs(improvement) < 1) {
      console.log(chalk.yellow(`   ➡️  No significant change\n`));
    } else {
      console.log(chalk.yellow(`   📉 Regression: ${improvementStr}%\n`));
    }
  });

  // Overall assessment
  const qualityImprovement = ((calculateStats(results.withLearning, 'quality').avg -
                               calculateStats(results.withoutLearning, 'quality').avg) /
                               calculateStats(results.withoutLearning, 'quality').avg) * 100;

  const timeImprovement = ((calculateStats(results.withoutLearning, 'durationSeconds').avg -
                            calculateStats(results.withLearning, 'durationSeconds').avg) /
                            calculateStats(results.withoutLearning, 'durationSeconds').avg) * 100;

  console.log(chalk.bold.cyan(`
╔═══════════════════════════════════════════════════════════════════╗
║              OVERALL IMPACT ASSESSMENT                             ║
╚═══════════════════════════════════════════════════════════════════╝
`));

  console.log(chalk.bold(`Self-Learning System Impact:`));
  console.log(`   Quality Improvement:     ${chalk.green(qualityImprovement > 0 ? '+' : '')}${chalk.green(qualityImprovement.toFixed(1) + '%')}`);
  console.log(`   Speed Improvement:       ${chalk.green(timeImprovement > 0 ? '+' : '')}${chalk.green(timeImprovement.toFixed(1) + '%')}`);

  const totalImprovement = (qualityImprovement + timeImprovement) / 2;
  console.log(`   Combined Score:          ${chalk.cyan(totalImprovement.toFixed(1) + '%')}\n`);

  if (totalImprovement > 10) {
    console.log(chalk.bold.green('✅ SIGNIFICANT IMPROVEMENT: Self-learning provides measurable benefits\n'));
  } else if (totalImprovement > 5) {
    console.log(chalk.bold.yellow('⚠️  MODERATE IMPROVEMENT: Benefits observable but not dramatic\n'));
  } else if (totalImprovement > 0) {
    console.log(chalk.bold.yellow('⚠️  MINOR IMPROVEMENT: Benefits present but minimal\n'));
  } else {
    console.log(chalk.bold.red('❌ NO IMPROVEMENT: Further optimization needed\n'));
  }

  // AgentDB learning statistics
  const db = getDatabase();
  const learningStats = db.prepare(`
    SELECT
      COUNT(*) as total_patterns,
      COUNT(DISTINCT task_category) as categories,
      AVG(reward) as avg_reward,
      COUNT(*) FILTER (WHERE success = 1) as successful
    FROM reasoningbank_patterns
  `).get();
  db.close();

  console.log(chalk.bold('🧠 AgentDB Learning Statistics:\n'));
  console.log(chalk.dim(`   Total Patterns:      ${learningStats.total_patterns}`));
  console.log(chalk.dim(`   Successful Patterns: ${learningStats.successful}`));
  console.log(chalk.dim(`   Task Categories:     ${learningStats.categories}`));
  console.log(chalk.dim(`   Average Reward:      ${learningStats.avg_reward?.toFixed(3) || 'N/A'}`));
  console.log('');
}

/**
 * Save results to JSON
 */
function saveResults() {
  const outputDir = path.join(__dirname, '../output/benchmarks');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `self-learning-benchmark-${timestamp}.json`;
  const filepath = path.join(outputDir, filename);

  const report = {
    timestamp: new Date().toISOString(),
    configuration: {
      tasksPerPhase: NUM_TASKS,
      totalTasks: NUM_TASKS * 2
    },
    results: {
      withoutLearning: results.withoutLearning,
      withLearning: results.withLearning
    },
    summary: {
      qualityImprovement: calculateStats(results.withLearning, 'quality').avg - calculateStats(results.withoutLearning, 'quality').avg,
      speedImprovement: calculateStats(results.withoutLearning, 'durationSeconds').avg - calculateStats(results.withLearning, 'durationSeconds').avg,
      tokenSavings: calculateStats(results.withoutLearning, 'tokensUsed').avg - calculateStats(results.withLearning, 'tokensUsed').avg
    }
  };

  fs.writeFileSync(filepath, JSON.stringify(report, null, 2));
  console.log(chalk.green(`📄 Benchmark results saved to: ${filepath}\n`));
}

/**
 * Main execution
 */
async function main() {
  try {
    await runPhase1();
    await runPhase2();
    displayResults();
    saveResults();

    console.log(chalk.green('✅ Benchmark completed successfully!\n'));
  } catch (error) {
    console.error(chalk.red(`\n❌ Benchmark failed: ${error.message}`));
    console.error(error.stack);
    process.exit(1);
  }
}

main();
