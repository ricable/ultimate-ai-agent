// Claude Flow integration configuration
import { logger } from '../utils/logger.js';

export interface ClaudeFlowConfig {
  enableMemory: boolean;
  enableCoordination: boolean;
  enableSwarm: boolean;
  memoryNamespace?: string;
  coordinationTopology?: 'hierarchical' | 'mesh' | 'ring' | 'star';
}

export const defaultClaudeFlowConfig: ClaudeFlowConfig = {
  enableMemory: true,
  enableCoordination: true,
  enableSwarm: true,
  memoryNamespace: 'claude-agents',
  coordinationTopology: 'mesh'
};

/**
 * Initialize claude-flow MCP tools for agent use
 */
export function getClaudeFlowTools(config: ClaudeFlowConfig = defaultClaudeFlowConfig) {
  const tools: string[] = [];

  if (config.enableMemory) {
    tools.push(
      'mcp__claude-flow__memory_usage',
      'mcp__claude-flow__memory_search',
      'mcp__claude-flow__memory_persist',
      'mcp__claude-flow__memory_namespace'
    );
    logger.info('Claude Flow memory tools enabled', { namespace: config.memoryNamespace });
  }

  if (config.enableCoordination) {
    tools.push(
      'mcp__claude-flow__swarm_init',
      'mcp__claude-flow__agent_spawn',
      'mcp__claude-flow__task_orchestrate',
      'mcp__claude-flow__swarm_status',
      'mcp__claude-flow__coordination_sync'
    );
    logger.info('Claude Flow coordination tools enabled', { topology: config.coordinationTopology });
  }

  if (config.enableSwarm) {
    tools.push(
      'mcp__claude-flow__swarm_scale',
      'mcp__claude-flow__load_balance',
      'mcp__claude-flow__agent_metrics',
      'mcp__claude-flow__swarm_monitor'
    );
    logger.info('Claude Flow swarm tools enabled');
  }

  return tools;
}

/**
 * Check if claude-flow MCP is available
 */
export async function isClaudeFlowAvailable(): Promise<boolean> {
  try {
    // Check if claude-flow is installed
    const { execSync } = await import('child_process');
    execSync('npx claude-flow@alpha --version', { stdio: 'ignore' });
    logger.info('Claude Flow detected and available');
    return true;
  } catch (error) {
    logger.warn('Claude Flow not available', { error });
    return false;
  }
}

/**
 * Initialize memory namespace for agent session
 */
export function getMemoryConfig(agentName?: string) {
  const namespace = agentName
    ? `claude-agents:${agentName}`
    : 'claude-agents:default';

  return {
    namespace,
    ttl: 3600, // 1 hour default TTL
    action: 'store' as const
  };
}

/**
 * Initialize swarm coordination for multi-agent tasks
 */
export function getSwarmConfig(topology: ClaudeFlowConfig['coordinationTopology'] = 'mesh') {
  return {
    topology,
    maxAgents: 8,
    strategy: 'balanced' as const
  };
}
