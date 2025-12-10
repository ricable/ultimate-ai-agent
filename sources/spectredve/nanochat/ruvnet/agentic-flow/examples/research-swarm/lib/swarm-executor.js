/**
 * Research Swarm Executor
 *
 * Orchestrates parallel execution of swarm agents with priority-based scheduling
 */

import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import chalk from 'chalk';
import ora from 'ora';
import { getJobStatus } from './db-utils.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

/**
 * Execute swarm agents with priority-based parallel execution
 *
 * @param {Array} agents - Swarm agent configurations from decomposition
 * @param {object} options - Execution options
 * @returns {Promise<object>} Execution results
 */
export async function executeSwarm(agents, options = {}) {
  const {
    maxConcurrent = 4,
    verbose = false
  } = options;

  console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║                    RESEARCH SWARM ACTIVATED                        ║'));
  console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.cyan(`🐝 Deploying ${agents.length} specialized research agents...\n`));

  // Display swarm composition
  agents.forEach((agent, idx) => {
    const roleEmoji = getRoleEmoji(agent.config.role);
    console.log(chalk.gray(`   ${roleEmoji} ${idx + 1}. ${agent.config.role.toUpperCase().replace('-', ' ')} (Priority ${agent.config.priority})`));
  });
  console.log();

  const results = {
    total: agents.length,
    successful: 0,
    failed: 0,
    agents: [],
    reports: []
  };

  try {
    // Group agents by priority
    const priorityGroups = groupByPriority(agents);
    const priorities = Object.keys(priorityGroups).sort((a, b) => parseInt(a) - parseInt(b));

    // Execute each priority group sequentially, but agents within group run in parallel
    for (const priority of priorities) {
      const groupAgents = priorityGroups[priority];
      console.log(chalk.bold.yellow(`\n🔄 Priority ${priority} (${groupAgents.length} agents):\n`));

      const groupResults = await executeAgentGroup(groupAgents, maxConcurrent, verbose);

      // Merge results
      results.agents.push(...groupResults);
      results.successful += groupResults.filter(r => r.success).length;
      results.failed += groupResults.filter(r => !r.success).length;

      // Collect reports
      groupResults.forEach(r => {
        if (r.success && r.jobId) {
          const job = getJobStatus(r.jobId);
          if (job && job.report_content) {
            results.reports.push({
              agent: r.agent,
              role: r.role,
              jobId: r.jobId,
              report: job.report_content,
              reportPath: job.report_path
            });
          }
        }
      });
    }

    // Summary
    console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
    console.log(chalk.bold.cyan('║                    SWARM COMPLETE                                  ║'));
    console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

    const successRate = ((results.successful / results.total) * 100).toFixed(1);
    console.log(chalk.bold(`📊 Results: ${chalk.green(results.successful)}/${results.total} successful (${successRate}%)\n`));

    // Trigger learning session if enough successful agents
    if (results.successful >= 2) {
      console.log(chalk.cyan('🧠 Running learning session to process swarm patterns...\n'));
      await runLearningSession();
    }

    // Display synthesis report if available
    const synthesisReport = results.reports.find(r => r.role === 'synthesizer');
    if (synthesisReport) {
      console.log(chalk.bold.cyan('📄 SYNTHESIS REPORT:\n'));
      console.log(chalk.dim('─'.repeat(71)));
      console.log(synthesisReport.report.substring(0, 1000));
      if (synthesisReport.report.length > 1000) {
        console.log(chalk.dim('\n... (truncated, see full report in database)'));
      }
      console.log(chalk.dim('─'.repeat(71)));
      console.log(chalk.dim(`\n💾 Full report: Job ID ${synthesisReport.jobId}\n`));
    }

  } catch (error) {
    console.error(chalk.red(`\n❌ Swarm execution failed: ${error.message}\n`));
    results.error = error.message;
  }

  return results;
}

/**
 * Execute a group of agents in parallel with concurrency limit
 *
 * @param {Array} agents - Agent configurations
 * @param {number} maxConcurrent - Maximum concurrent executions
 * @param {boolean} verbose - Verbose logging
 * @returns {Promise<Array>} Execution results
 */
async function executeAgentGroup(agents, maxConcurrent, verbose) {
  const running = new Set();
  const results = [];

  for (const agent of agents) {
    // Wait if at max concurrency
    while (running.size >= maxConcurrent) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    const agentPromise = executeAgent(agent, verbose)
      .then(result => {
        results.push(result);
        running.delete(agentPromise);
      })
      .catch(error => {
        results.push({
          agent: agent.agent,
          role: agent.config.role,
          success: false,
          error: error.message
        });
        running.delete(agentPromise);
      });

    running.add(agentPromise);
  }

  // Wait for all agents in group to complete
  await Promise.all([...running]);

  return results;
}

/**
 * Execute a single swarm agent
 *
 * @param {object} agent - Agent configuration
 * @param {boolean} verbose - Verbose logging
 * @returns {Promise<object>} Execution result
 */
function executeAgent(agent, verbose) {
  return new Promise((resolve) => {
    const roleEmoji = getRoleEmoji(agent.config.role);
    const spinner = ora({
      text: `${roleEmoji} ${agent.config.role}: ${agent.task.substring(0, 50)}...`,
      color: 'cyan'
    }).start();

    const env = {
      ...process.env,
      RESEARCH_DEPTH: agent.config.depth.toString(),
      RESEARCH_TIME_BUDGET: agent.config.timeMinutes.toString(),
      RESEARCH_FOCUS: agent.config.focus,
      ANTI_HALLUCINATION_LEVEL: agent.config.antiHallucination,
      ED2551_MODE: agent.config.enableED2551 ? 'true' : 'false',
      SWARM_ROLE: agent.config.role,
      SWARM_MODE: 'true'
    };

    const runnerPath = path.join(__dirname, '../run-researcher-local.js');

    const child = spawn('node', [runnerPath, agent.agent, agent.task], {
      env,
      stdio: verbose ? 'inherit' : 'pipe'
    });

    let output = '';
    if (!verbose) {
      child.stdout.on('data', (data) => output += data.toString());
      child.stderr.on('data', (data) => output += data.toString());
    }

    child.on('close', (code) => {
      const jobId = output.match(/Job ID:\s+([a-f0-9-]+)/)?.[1];

      if (code === 0) {
        spinner.succeed(chalk.green(`${roleEmoji} ${agent.config.role}: Complete (${jobId?.substring(0, 8)})`));
        resolve({
          agent: agent.agent,
          role: agent.config.role,
          jobId,
          success: true
        });
      } else {
        spinner.fail(chalk.red(`${roleEmoji} ${agent.config.role}: Failed`));
        resolve({
          agent: agent.agent,
          role: agent.config.role,
          success: false,
          exitCode: code
        });
      }
    });

    child.on('error', (error) => {
      spinner.fail(chalk.red(`${roleEmoji} ${agent.config.role}: Error - ${error.message}`));
      resolve({
        agent: agent.agent,
        role: agent.config.role,
        success: false,
        error: error.message
      });
    });
  });
}

/**
 * Group agents by priority level
 *
 * @param {Array} agents - Agent configurations
 * @returns {object} Agents grouped by priority
 */
function groupByPriority(agents) {
  return agents.reduce((groups, agent) => {
    const priority = agent.config.priority || 1;
    if (!groups[priority]) {
      groups[priority] = [];
    }
    groups[priority].push(agent);
    return groups;
  }, {});
}

/**
 * Get emoji for agent role
 *
 * @param {string} role - Agent role
 * @returns {string} Emoji
 */
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

/**
 * Run learning session after swarm completion
 *
 * @returns {Promise<void>}
 */
async function runLearningSession() {
  return new Promise((resolve) => {
    const learningScript = path.join(__dirname, '../scripts/learning-session.js');
    const child = spawn('node', [learningScript], {
      stdio: 'pipe'
    });

    let output = '';
    child.stdout.on('data', (data) => output += data.toString());

    child.on('close', (code) => {
      if (code === 0) {
        // Extract key metrics from learning session output
        const patternsMatch = output.match(/(\d+)\s+patterns/i);
        const episodesMatch = output.match(/(\d+)\s+episodes/i);

        if (patternsMatch || episodesMatch) {
          console.log(chalk.green(`✅ Learning session complete`));
          if (patternsMatch) {
            console.log(chalk.dim(`   Patterns stored: ${patternsMatch[1]}`));
          }
          if (episodesMatch) {
            console.log(chalk.dim(`   Episodes recorded: ${episodesMatch[1]}`));
          }
        }
      } else {
        console.log(chalk.yellow(`⚠️  Learning session exited with code ${code}`));
      }
      resolve();
    });

    child.on('error', (error) => {
      console.log(chalk.yellow(`⚠️  Learning session error: ${error.message}`));
      resolve();
    });
  });
}

export default {
  executeSwarm,
  executeAgent,
  groupByPriority,
  getRoleEmoji
};
