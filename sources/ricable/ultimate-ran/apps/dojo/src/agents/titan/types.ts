/**
 * TypeScript Type Definitions for Titan Council Agent
 *
 * Defines the core types for the Neuro-Symbolic Titan Council integration
 * in the AG-UI Dojo interface.
 */

/**
 * Council Member Roles
 */
export enum CouncilRole {
  ANALYST = 'analyst',
  HISTORIAN = 'historian',
  STRATEGIST = 'strategist',
  CHAIRMAN = 'chairman'
}

/**
 * Council Member Model Identifiers
 */
export enum CouncilModel {
  DEEPSEEK_R1 = 'deepseek-r1-distill',
  GEMINI_15_PRO = 'gemini-1.5-pro',
  CLAUDE_37_SONNET = 'claude-3-7-sonnet'
}

/**
 * Debate Status Types
 */
export enum DebateStatus {
  PROPOSAL = 'PROPOSAL',
  CRITIQUE = 'CRITIQUE',
  SYNTHESIS = 'SYNTHESIS',
  CONSENSUS = 'CONSENSUS',
  SPLIT_VOTE = 'SPLIT_VOTE'
}

/**
 * Avatar configuration for Council Members
 */
export interface CouncilAvatar {
  name: string;
  role: CouncilRole;
  model: CouncilModel;
  color: string;
  icon: string;
  description: string;
}

/**
 * Council Member Definition
 */
export interface CouncilMember {
  id: string;
  role: CouncilRole;
  model: CouncilModel;
  temperature: number;
  systemPrompt: string;
  tools: string[];
  avatar: CouncilAvatar;
}

/**
 * Thinking Step Event from Council Members
 */
export interface ThinkingStepEvent {
  type: 'THINKING_STEP';
  agentId: string;
  agentName: string;
  role: CouncilRole;
  content: string;
  status: DebateStatus;
  timestamp: number;
  metadata?: {
    lyapunovExponent?: number;
    interferenceLevel?: number;
    similarEpisodes?: string[];
    confidence?: number;
  };
}

/**
 * Generative UI Render Event
 */
export interface GenUIRenderEvent {
  type: 'gen_ui_render';
  componentType: 'InterferenceHeatmap' | 'DebateTimeline' | 'ConsensusCard';
  props: Record<string, any>;
  timestamp: number;
}

/**
 * Council Event Union Type
 */
export type CouncilEvent = ThinkingStepEvent | GenUIRenderEvent;

/**
 * Debate Timeline Entry
 */
export interface DebateTimelineEntry {
  id: string;
  event: ThinkingStepEvent;
  position: number;
}

/**
 * Consensus Result
 */
export interface ConsensusResult {
  decision: string;
  votes: Record<string, 'approve' | 'reject' | 'abstain'>;
  consensusLevel: number; // 0-1
  chairmanNotes: string;
  requiresHITL: boolean; // Human-in-the-Loop required
}

/**
 * Agent Integration Config for Dojo
 */
export interface AgentIntegrationConfig {
  name: string;
  agent: any; // TitanCouncilAgent class
  icon: string;
  description: string;
  version?: string;
  tags?: string[];
}

/**
 * Interference Data for Heatmap
 */
export interface InterferenceData {
  cellId: string;
  sector: number;
  interferenceLevel: number; // dBm
  timestamp: number;
  neighbors: Array<{
    cellId: string;
    distance: number;
    interferenceContribution: number;
  }>;
}

/**
 * RAN Parameter Change Proposal
 */
export interface ParameterProposal {
  proposedBy: string; // Agent ID
  cellId: string;
  parameters: {
    tx_power?: number;
    tilt?: number;
    bandwidth?: number;
    [key: string]: any;
  };
  reasoning: string;
  impact: {
    expected: string;
    risk: 'low' | 'medium' | 'high';
    rollbackPlan?: string;
  };
}

/**
 * War Room State
 */
export interface WarRoomState {
  activeDebate: string | null;
  timeline: DebateTimelineEntry[];
  currentConsensus: ConsensusResult | null;
  interferenceData: InterferenceData[];
  pendingApprovals: ParameterProposal[];
}
