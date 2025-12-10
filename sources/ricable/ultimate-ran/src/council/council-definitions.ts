/**
 * Council Member Definitions
 * Defines the three council members according to PRD specification
 */

import type { CouncilMember } from './debate-protocol-new';

/**
 * Council member configurations following PRD line 659-705
 */
export const COUNCIL_MEMBERS: Record<string, CouncilMember> = {
  'analyst-deepseek': {
    id: 'analyst-deepseek',
    role: 'analyst',
    model: 'deepseek-r1-distill',
    provider: 'deepseek',
    temperature: 0.3, // Low temperature for analytical precision
    maxTokens: 4096,
    enabled: true
  },

  'historian-gemini': {
    id: 'historian-gemini',
    role: 'historian',
    model: 'gemini-1.5-pro',
    provider: 'gemini',
    temperature: 0.5, // Medium temperature for contextual retrieval
    maxTokens: 8192,
    enabled: true
  },

  'strategist-claude': {
    id: 'strategist-claude',
    role: 'strategist',
    model: 'claude-3-7-sonnet',
    provider: 'claude',
    temperature: 0.7, // Higher temperature for strategic creativity
    maxTokens: 16384,
    enabled: true
  }
};

/**
 * Get all enabled council members
 */
export function getEnabledMembers(): CouncilMember[] {
  return Object.values(COUNCIL_MEMBERS).filter(m => m.enabled);
}

/**
 * Get council member by ID
 */
export function getMember(id: string): CouncilMember | undefined {
  return COUNCIL_MEMBERS[id];
}

/**
 * Get members by role
 */
export function getMembersByRole(role: 'analyst' | 'historian' | 'strategist'): CouncilMember[] {
  return Object.values(COUNCIL_MEMBERS).filter(m => m.role === role);
}
