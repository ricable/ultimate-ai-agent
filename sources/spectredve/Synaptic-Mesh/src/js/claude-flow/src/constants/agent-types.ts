/**
 * Central source of truth for agent types
 * This file ensures consistency across TypeScript types and runtime validation
 */

export const AGENT_TYPES = {
  COORDINATOR: 'coordinator',
  RESEARCHER: 'researcher',
  CODER: 'coder',
  ANALYST: 'analyst',
  ARCHITECT: 'architect',
  TESTER: 'tester',
  REVIEWER: 'reviewer',
  OPTIMIZER: 'optimizer',
  DOCUMENTER: 'documenter',
  MONITOR: 'monitor',
  SPECIALIST: 'specialist'
} as const;

export type AgentType = typeof AGENT_TYPES[keyof typeof AGENT_TYPES];

// Array of all valid agent types for runtime validation
export const VALID_AGENT_TYPES = Object.values(AGENT_TYPES);

// JSON Schema for agent type validation
export const AGENT_TYPE_SCHEMA = {
  type: 'string',
  enum: VALID_AGENT_TYPES,
  description: 'Type of AI agent'
};

// Helper function to validate agent type
export function isValidAgentType(type: string): type is AgentType {
  return VALID_AGENT_TYPES.includes(type as AgentType);
}

// Strategy types
export const SWARM_STRATEGIES = {
  AUTO: 'auto',
  RESEARCH: 'research',
  DEVELOPMENT: 'development',
  ANALYSIS: 'analysis',
  TESTING: 'testing',
  OPTIMIZATION: 'optimization',
  MAINTENANCE: 'maintenance',
  CUSTOM: 'custom'
} as const;

export type SwarmStrategy = typeof SWARM_STRATEGIES[keyof typeof SWARM_STRATEGIES];
export const VALID_SWARM_STRATEGIES = Object.values(SWARM_STRATEGIES);

// Task orchestration strategies (different from swarm strategies)
export const ORCHESTRATION_STRATEGIES = {
  PARALLEL: 'parallel',
  SEQUENTIAL: 'sequential',
  ADAPTIVE: 'adaptive',
  BALANCED: 'balanced'
} as const;

export type OrchestrationStrategy = typeof ORCHESTRATION_STRATEGIES[keyof typeof ORCHESTRATION_STRATEGIES];
export const VALID_ORCHESTRATION_STRATEGIES = Object.values(ORCHESTRATION_STRATEGIES);