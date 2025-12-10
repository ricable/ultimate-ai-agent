/**
 * GOALIE SDK Integration for Research-Swarm
 *
 * Integrates Goal-Oriented Action Planning (GOAP) capabilities into research-swarm:
 * - Automatic goal decomposition
 * - Intelligent sub-task generation
 * - GOAP-based research planning
 * - Multi-agent swarm execution of GOAP plans
 *
 * Web Search Support (v1.2.0):
 * - Google Gemini with grounding (set GOOGLE_GEMINI_API_KEY)
 * - Claude with web search MCP tools (set ANTHROPIC_API_KEY + MCP_CONFIG_PATH)
 * - OpenRouter with search models (set OPENROUTER_API_KEY)
 * - Not limited to Perplexity - use any provider!
 */

import { spawn } from 'child_process';
import chalk from 'chalk';
import { decomposeTask } from './swarm-decomposition.js';
import { executeSwarm } from './swarm-executor.js';
import { createJob } from './db-utils.js';

/**
 * Decompose a complex research goal using GOALIE's GOAP algorithm
 *
 * @param {string} goal - High-level research goal
 * @param {object} options - Configuration options
 * @returns {Promise<Array>} Array of sub-goals with metadata
 */
export async function decomposeGoal(goal, options = {}) {
  const {
    useChainOfThought = true,
    useSelfConsistency = true,
    useAntiHallucination = true,
    maxSubGoals = 10,
    verbose = false
  } = options;

  console.log(chalk.bold.cyan('\n🎯 GOALIE Goal Decomposition\n'));
  console.log(chalk.dim(`Goal: ${goal}\n`));

  const args = ['goalie', 'search', goal];

  // Add plugins
  if (useChainOfThought) {
    args.push('--plugin', 'chain-of-thought');
  }
  if (useSelfConsistency) {
    args.push('--plugin', 'self-consistency');
  }
  if (useAntiHallucination) {
    args.push('--plugin', 'anti-hallucination');
  }

  return new Promise((resolve, reject) => {
    const child = spawn('npx', args, {
      stdio: verbose ? 'inherit' : 'pipe',
      env: process.env
    });

    let output = '';
    let errorOutput = '';

    if (!verbose) {
      child.stdout.on('data', (data) => output += data.toString());
      child.stderr.on('data', (data) => errorOutput += data.toString());
    }

    child.on('close', (code) => {
      if (code === 0) {
        const subGoals = parseGoalieOutput(output, maxSubGoals);
        console.log(chalk.green(`✅ Decomposed into ${subGoals.length} sub-goals\n`));
        resolve(subGoals);
      } else {
        console.warn(chalk.yellow('⚠️  GOALIE decomposition failed, using single goal'));
        // Fallback: treat entire goal as single sub-goal
        resolve([{
          id: 1,
          goal: goal,
          description: goal,
          priority: 1,
          complexity: 'high'
        }]);
      }
    });

    child.on('error', (error) => {
      console.warn(chalk.yellow(`⚠️  GOALIE error: ${error.message}, using fallback`));
      resolve([{
        id: 1,
        goal: goal,
        description: goal,
        priority: 1,
        complexity: 'high'
      }]);
    });
  });
}

/**
 * Execute a GOAP-planned research workflow
 *
 * @param {string} goal - High-level research goal
 * @param {object} options - Configuration options
 * @returns {Promise<object>} Execution results
 */
export async function executeGoalBasedResearch(goal, options = {}) {
  const {
    depth = 5,
    timeMinutes = 120,
    swarmSize = 5,
    enableED2551 = true,
    antiHallucination = 'high',
    maxConcurrent = 3,
    verbose = false
  } = options;

  console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║          GOAL-ORIENTED RESEARCH (GOALIE + SWARM)                   ║'));
  console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.dim(`Goal: ${goal}\n`));

  // Phase 1: GOALIE decomposition
  const subGoals = await decomposeGoal(goal, {
    useChainOfThought: true,
    useSelfConsistency: true,
    useAntiHallucination: true,
    verbose
  });

  // Display decomposition
  console.log(chalk.bold('📋 Research Plan:\n'));
  subGoals.forEach((subGoal, idx) => {
    console.log(chalk.cyan(`   ${idx + 1}. ${subGoal.description}`));
    console.log(chalk.dim(`      Complexity: ${subGoal.complexity} | Priority: ${subGoal.priority}\n`));
  });

  // Phase 2: Execute each sub-goal with research-swarm
  const results = {
    goal,
    subGoals: [],
    totalAgents: 0,
    successfulSubGoals: 0,
    failedSubGoals: 0,
    totalTime: 0
  };

  for (let i = 0; i < subGoals.length; i++) {
    const subGoal = subGoals[i];

    console.log(chalk.bold.yellow(`\n🔬 Executing Sub-Goal ${i + 1}/${subGoals.length}\n`));
    console.log(chalk.dim(`"${subGoal.description}"\n`));

    // Adjust swarm configuration based on sub-goal complexity
    const subGoalSwarmSize = getSwarmSizeForComplexity(subGoal.complexity, swarmSize);
    const subGoalDepth = getDepthForComplexity(subGoal.complexity, depth);
    const subGoalTime = Math.floor(timeMinutes / subGoals.length);

    // Decompose sub-goal into swarm agents
    const agents = decomposeTask(subGoal.description, {
      depth: subGoalDepth,
      timeMinutes: subGoalTime,
      swarmSize: subGoalSwarmSize,
      enableED2551,
      antiHallucination
    });

    console.log(chalk.cyan(`Deploying ${agents.length}-agent swarm...\n`));

    // Execute swarm
    const swarmResult = await executeSwarm(agents, {
      maxConcurrent,
      verbose
    });

    // Record results
    results.subGoals.push({
      id: subGoal.id,
      description: subGoal.description,
      complexity: subGoal.complexity,
      agentCount: agents.length,
      success: swarmResult.successful > 0,
      successful: swarmResult.successful,
      failed: swarmResult.failed
    });

    results.totalAgents += agents.length;

    if (swarmResult.successful > 0) {
      results.successfulSubGoals++;
      console.log(chalk.green(`✅ Sub-goal completed successfully\n`));
    } else {
      results.failedSubGoals++;
      console.log(chalk.red(`❌ Sub-goal failed\n`));
    }
  }

  // Final summary
  console.log(chalk.bold.cyan('\n╔═══════════════════════════════════════════════════════════════════╗'));
  console.log(chalk.bold.cyan('║                GOAL-ORIENTED RESEARCH COMPLETE                     ║'));
  console.log(chalk.bold.cyan('╚═══════════════════════════════════════════════════════════════════╝\n'));

  console.log(chalk.bold('📊 Summary:\n'));
  console.log(chalk.dim(`   Original goal: ${goal}`));
  console.log(chalk.dim(`   Sub-goals completed: ${results.successfulSubGoals}/${subGoals.length}`));
  console.log(chalk.dim(`   Total agents deployed: ${results.totalAgents}`));
  console.log(chalk.dim(`   Success rate: ${((results.successfulSubGoals / subGoals.length) * 100).toFixed(1)}%\n`));

  return results;
}

/**
 * Create a GOAP plan for a research goal (planning only, no execution)
 *
 * @param {string} goal - Research goal
 * @param {object} options - Options
 * @returns {Promise<object>} Research plan
 */
export async function planResearch(goal, options = {}) {
  const subGoals = await decomposeGoal(goal, options);

  const plan = {
    goal,
    subGoals,
    estimatedAgents: 0,
    estimatedTime: 0,
    estimatedCost: 0
  };

  // Calculate estimates
  for (const subGoal of subGoals) {
    const swarmSize = getSwarmSizeForComplexity(subGoal.complexity, options.swarmSize || 5);
    plan.estimatedAgents += swarmSize;
    plan.estimatedTime += Math.floor((options.timeMinutes || 120) / subGoals.length);
  }

  plan.estimatedCost = plan.estimatedAgents; // Cost multiplier

  return plan;
}

/**
 * Parse GOALIE output to extract sub-goals
 *
 * @param {string} output - GOALIE output text
 * @param {number} maxSubGoals - Maximum sub-goals to extract
 * @returns {Array} Array of sub-goal objects
 */
function parseGoalieOutput(output, maxSubGoals = 10) {
  const subGoals = [];

  // Look for numbered goals or bullet points
  const lines = output.split('\n');
  let goalId = 1;

  for (const line of lines) {
    // Match patterns like "1.", "1)", "- ", "• "
    const match = line.match(/^\s*(?:\d+[\.)]\s*|[-•]\s*)(.+?)$/);

    if (match && match[1].trim().length > 10) { // Ignore very short lines
      const description = match[1].trim();

      // Estimate complexity based on length and keywords
      const complexity = estimateComplexity(description);

      subGoals.push({
        id: goalId++,
        goal: description,
        description: description,
        priority: goalId <= 3 ? 1 : 2, // First 3 are high priority
        complexity
      });

      if (subGoals.length >= maxSubGoals) break;
    }
  }

  // If no sub-goals found, try simpler extraction
  if (subGoals.length === 0) {
    const sentences = output.split(/[.!?]+/).filter(s => s.trim().length > 20);
    sentences.slice(0, maxSubGoals).forEach((sentence, idx) => {
      subGoals.push({
        id: idx + 1,
        goal: sentence.trim(),
        description: sentence.trim(),
        priority: 1,
        complexity: 'medium'
      });
    });
  }

  return subGoals;
}

/**
 * Estimate complexity of a sub-goal
 *
 * @param {string} description - Sub-goal description
 * @returns {string} Complexity level: 'low', 'medium', 'high', 'very-high'
 */
function estimateComplexity(description) {
  const keywords = {
    high: ['comprehensive', 'analyze', 'evaluate', 'compare', 'complex', 'advanced', 'deep'],
    medium: ['research', 'investigate', 'explore', 'examine', 'study', 'review'],
    low: ['list', 'identify', 'define', 'describe', 'summarize', 'overview']
  };

  const lowerDesc = description.toLowerCase();

  if (keywords.high.some(kw => lowerDesc.includes(kw))) {
    return lowerDesc.length > 100 ? 'very-high' : 'high';
  } else if (keywords.medium.some(kw => lowerDesc.includes(kw))) {
    return 'medium';
  } else {
    return 'low';
  }
}

/**
 * Get swarm size based on complexity
 *
 * @param {string} complexity - Complexity level
 * @param {number} baseSize - Base swarm size
 * @returns {number} Adjusted swarm size
 */
function getSwarmSizeForComplexity(complexity, baseSize) {
  const sizeMap = {
    'low': Math.max(3, baseSize - 2),
    'medium': baseSize,
    'high': Math.min(7, baseSize + 1),
    'very-high': 7
  };

  return sizeMap[complexity] || baseSize;
}

/**
 * Get research depth based on complexity
 *
 * @param {string} complexity - Complexity level
 * @param {number} baseDepth - Base depth
 * @returns {number} Adjusted depth
 */
function getDepthForComplexity(complexity, baseDepth) {
  const depthMap = {
    'low': Math.max(3, baseDepth - 2),
    'medium': baseDepth,
    'high': Math.min(10, baseDepth + 1),
    'very-high': Math.min(10, baseDepth + 2)
  };

  return depthMap[complexity] || baseDepth;
}

/**
 * Explain GOAP planning for a goal (without executing)
 *
 * @param {string} goal - Research goal
 * @returns {Promise<string>} Explanation of the plan
 */
export async function explainGoalPlan(goal) {
  return new Promise((resolve, reject) => {
    const child = spawn('npx', [
      'goalie', 'explain', goal
    ], { stdio: 'pipe' });

    let output = '';
    child.stdout.on('data', (data) => output += data.toString());

    child.on('close', (code) => {
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error('GOALIE explanation failed'));
      }
    });
  });
}

export default {
  decomposeGoal,
  executeGoalBasedResearch,
  planResearch,
  explainGoalPlan
};
