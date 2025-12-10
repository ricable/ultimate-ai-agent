/**
 * Research Swarm Task Decomposition
 *
 * Decomposes a single research task into multiple specialized perspectives
 * for parallel swarm execution.
 */

/**
 * Decompose a research task into swarm agents with specialized roles
 *
 * @param {string} task - The research task to decompose
 * @param {object} options - Configuration options
 * @returns {Array} Array of swarm agent configurations
 */
export function decomposeTask(task, options = {}) {
  const {
    depth = 5,
    timeMinutes = 120,
    focus = 'balanced',
    swarmSize = 5,
    enableED2551 = true,
    antiHallucination = 'high'
  } = options;

  // Adaptive swarm sizing based on depth
  const actualSwarmSize = getAdaptiveSwarmSize(depth, swarmSize);

  // Calculate time budget per agent (agents run in parallel, so each gets full budget)
  const timePerAgent = Math.floor(timeMinutes / 1.2); // 20% overlap for synthesis

  const agents = [];

  // 1. Explorer Agent (20% focus, broad survey)
  if (actualSwarmSize >= 3) {
    agents.push({
      id: 'explorer',
      agent: 'researcher',
      task: `Conduct broad exploratory research on: ${task}. Focus on identifying key domains, sub-topics, and the overall landscape. Provide a comprehensive overview.`,
      config: {
        depth: Math.max(1, Math.floor(depth * 0.6)), // Shallower depth for breadth
        timeMinutes: Math.floor(timePerAgent * 0.20),
        focus: 'broad',
        enableED2551,
        antiHallucination,
        role: 'explorer',
        priority: 1 // Run first
      }
    });
  }

  // 2. Depth Analyst (30% focus, narrow deep-dive)
  if (actualSwarmSize >= 3) {
    agents.push({
      id: 'depth-analyst',
      agent: 'researcher',
      task: `Perform deep technical analysis of: ${task}. Focus on core concepts, mechanisms, and detailed understanding. Provide in-depth insights.`,
      config: {
        depth: Math.min(10, depth + 2), // Deeper depth for detail
        timeMinutes: Math.floor(timePerAgent * 0.30),
        focus: 'narrow',
        enableED2551,
        antiHallucination,
        role: 'depth-analyst',
        priority: 1 // Run first
      }
    });
  }

  // 3. Verification Agent (20% focus, fact-checking)
  if (actualSwarmSize >= 4) {
    agents.push({
      id: 'verifier',
      agent: 'researcher',
      task: `Verify and fact-check information about: ${task}. Cross-reference sources, validate claims, assess source quality. Identify contradictions or uncertainties.`,
      config: {
        depth: Math.floor(depth * 0.8),
        timeMinutes: Math.floor(timePerAgent * 0.20),
        focus: 'balanced',
        enableED2551: true, // Always enable for verification
        antiHallucination: 'high', // Maximum verification
        role: 'verifier',
        priority: 2 // Run after initial research
      }
    });
  }

  // 4. Trend Analyst (15% focus, temporal patterns)
  if (actualSwarmSize >= 5) {
    agents.push({
      id: 'trend-analyst',
      agent: 'researcher',
      task: `Analyze temporal trends and patterns for: ${task}. Track historical evolution, identify current state, project future directions. Focus on time-based insights.`,
      config: {
        depth: Math.floor(depth * 0.7),
        timeMinutes: Math.floor(timePerAgent * 0.15),
        focus: 'broad',
        enableED2551,
        antiHallucination,
        role: 'trend-analyst',
        priority: 1 // Run first
      }
    });
  }

  // 5. Domain Expert (Optional, for complex tasks)
  if (actualSwarmSize >= 6) {
    agents.push({
      id: 'domain-expert',
      agent: 'researcher',
      task: `Provide domain-specific expertise on: ${task}. Focus on specialized knowledge, best practices, and expert perspectives. Identify what practitioners should know.`,
      config: {
        depth: depth,
        timeMinutes: Math.floor(timePerAgent * 0.15),
        focus: 'balanced',
        enableED2551,
        antiHallucination,
        role: 'domain-expert',
        priority: 1
      }
    });
  }

  // 6. Critical Reviewer (Optional, for complex tasks)
  if (actualSwarmSize >= 7) {
    agents.push({
      id: 'critic',
      agent: 'researcher',
      task: `Critically review and challenge assumptions about: ${task}. Identify weaknesses, limitations, and potential biases. Provide constructive criticism.`,
      config: {
        depth: Math.floor(depth * 0.6),
        timeMinutes: Math.floor(timePerAgent * 0.10),
        focus: 'narrow',
        enableED2551,
        antiHallucination: 'high',
        role: 'critic',
        priority: 2
      }
    });
  }

  // 7. Synthesis Agent (Always last, 15% focus)
  agents.push({
    id: 'synthesizer',
    agent: 'researcher',
    task: `Synthesize all research findings about: ${task}. Combine insights from multiple perspectives, resolve contradictions, identify key themes. Generate unified, comprehensive report.`,
    config: {
      depth: depth,
      timeMinutes: Math.floor(timePerAgent * 0.15),
      focus: 'balanced',
      enableED2551,
      antiHallucination: 'high', // High verification for synthesis
      role: 'synthesizer',
      priority: 3, // Run last
      dependsOn: agents.map(a => a.id) // Waits for all other agents
    }
  });

  return agents;
}

/**
 * Determine adaptive swarm size based on task complexity
 *
 * @param {number} depth - Research depth (1-10)
 * @param {number} requestedSize - User-requested swarm size
 * @returns {number} Optimal swarm size
 */
function getAdaptiveSwarmSize(depth, requestedSize) {
  // If user explicitly set swarm size, respect it
  if (requestedSize !== 5) {
    return Math.max(3, Math.min(7, requestedSize));
  }

  // Adaptive sizing based on depth
  if (depth <= 3) {
    return 3; // Simple: explorer, depth, synthesis
  } else if (depth <= 6) {
    return 5; // Medium: + verifier, trend
  } else {
    return 7; // Complex: + domain expert, critic
  }
}

/**
 * Get swarm agent roles description
 *
 * @returns {object} Agent roles and descriptions
 */
export function getSwarmRoles() {
  return {
    explorer: {
      name: 'Explorer',
      description: 'Broad survey and topic mapping',
      focus: 'breadth',
      priority: 1
    },
    'depth-analyst': {
      name: 'Depth Analyst',
      description: 'Technical deep dive and detailed analysis',
      focus: 'depth',
      priority: 1
    },
    verifier: {
      name: 'Verifier',
      description: 'Fact-checking and source validation',
      focus: 'accuracy',
      priority: 2
    },
    'trend-analyst': {
      name: 'Trend Analyst',
      description: 'Temporal patterns and future projections',
      focus: 'trends',
      priority: 1
    },
    'domain-expert': {
      name: 'Domain Expert',
      description: 'Specialized knowledge and best practices',
      focus: 'expertise',
      priority: 1
    },
    critic: {
      name: 'Critical Reviewer',
      description: 'Challenge assumptions and identify limitations',
      focus: 'critique',
      priority: 2
    },
    synthesizer: {
      name: 'Synthesizer',
      description: 'Combine findings and generate unified report',
      focus: 'integration',
      priority: 3
    }
  };
}

/**
 * Validate swarm configuration
 *
 * @param {Array} agents - Swarm agent configurations
 * @returns {object} Validation result
 */
export function validateSwarmConfig(agents) {
  const errors = [];
  const warnings = [];

  // Check minimum agents
  if (agents.length < 3) {
    errors.push('Swarm requires at least 3 agents (explorer, depth, synthesizer)');
  }

  // Check for synthesizer
  const hasSynthesizer = agents.some(a => a.config.role === 'synthesizer');
  if (!hasSynthesizer) {
    errors.push('Swarm must include a synthesizer agent');
  }

  // Check time budgets
  const totalTime = agents.reduce((sum, a) => sum + a.config.timeMinutes, 0);
  if (totalTime > 600) {
    warnings.push(`Total time budget (${totalTime}min) is high. Consider reducing depth or swarm size.`);
  }

  // Check priorities
  const priorities = agents.map(a => a.config.priority);
  if (!priorities.includes(3)) {
    warnings.push('No agents with priority 3 (synthesis phase). Results may not be properly combined.');
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings
  };
}

export default {
  decomposeTask,
  getSwarmRoles,
  validateSwarmConfig,
  getAdaptiveSwarmSize
};
