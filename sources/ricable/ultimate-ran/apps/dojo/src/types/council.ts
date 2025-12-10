/**
 * Type definitions for Titan Council
 * Part of Ericsson Gen 7.0 Neuro-Symbolic Titan Platform
 */

export type CouncilRole = 'analyst' | 'historian' | 'strategist' | 'chairman';

export type ThinkingStepStatus = 'THINKING' | 'CRITIQUE' | 'PROPOSAL' | 'CONSENSUS' | 'DENIED';

export interface CouncilMember {
  id: string;
  role: CouncilRole;
  model_id: string;
  temperature: number;
  avatar?: string;
}

export interface ThinkingStepEvent {
  type: 'THINKING_STEP';
  agentId: string;
  agentName: string;
  role: CouncilRole;
  content: string;
  status: ThinkingStepStatus;
  timestamp: number;
  metadata?: Record<string, any>;
}

export interface InterferenceData {
  cellId: string;
  neighbors: Array<{
    cellId: string;
    interferenceLevel: number; // dBm
    distance: number; // meters
  }>;
  matrix: number[][]; // 2D interference matrix for heatmap
}

export interface ApprovalRequest {
  id: string;
  type: 'parameter_change' | 'reboot' | 'topology_change';
  description: string;
  proposedBy: CouncilMember;
  parameters: Record<string, any>;
  risk: 'low' | 'medium' | 'high' | 'critical';
  consensusScore: number; // 0-1, where >0.67 indicates 2/3+ consensus
  timestamp: number;
}

export interface AgentState {
  councilMembers: CouncilMember[];
  debateHistory: ThinkingStepEvent[];
  currentInterference: InterferenceData | null;
  pendingApprovals: ApprovalRequest[];
  isDebating: boolean;
}
