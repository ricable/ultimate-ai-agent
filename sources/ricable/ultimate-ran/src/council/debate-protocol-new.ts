/**
 * LLM Council Debate Protocol Interfaces
 * Defines the structure for multi-model debate and consensus
 */

export interface CouncilMember {
  id: string;
  role: 'analyst' | 'historian' | 'strategist';
  model: string;
  provider: 'deepseek' | 'gemini' | 'claude';
  temperature: number;
  maxTokens?: number;
  enabled: boolean;
}

export interface DebateProposal {
  id: string;
  type: 'parameter_change' | 'optimization' | 'rollback' | 'investigation';
  description: string;
  context: {
    cellId?: string;
    parameters?: Record<string, any>;
    metrics?: Record<string, number>;
    urgency?: 'low' | 'medium' | 'high' | 'critical';
  };
  timestamp: number;
  proposer?: string;
}

export interface DebateResponse {
  memberId: string;
  role: string;
  model: string;
  content: string;
  confidence: number;
  vote: 'approve' | 'reject' | 'abstain';
  reasoning: string;
  timestamp: number;
  responseTime: number; // milliseconds
}

export interface CritiqueRound {
  roundNumber: number;
  responses: DebateResponse[];
  consensusReached: boolean;
  agreementRatio: number; // 0-1
  timestamp: number;
}

export interface ConsensusResult {
  decision: 'approved' | 'rejected' | 'needs_revision';
  confidence: number;
  agreementRatio: number;
  synthesis: string; // Chairman's synthesized conclusion
  participatingMembers: string[];
  totalRounds: number;
  duration: number; // milliseconds
  votes: {
    approve: number;
    reject: number;
    abstain: number;
  };
  byzantineFaultDetected: boolean;
  timestamp: number;
}

export interface DebateSession {
  proposalId: string;
  proposal: DebateProposal;
  rounds: CritiqueRound[];
  consensus?: ConsensusResult;
  status: 'pending' | 'in_progress' | 'completed' | 'timeout' | 'failed';
  startTime: number;
  endTime?: number;
}

/**
 * Consensus thresholds for Byzantine fault tolerance
 */
export const CONSENSUS_THRESHOLDS = {
  APPROVAL_RATIO: 2/3, // 66.7% agreement required (Byzantine 2f+1)
  MIN_PARTICIPANTS: 2, // Minimum 2 out of 3 must respond
  MAX_ROUNDS: 2, // Maximum critique rounds
  TIMEOUT_MS: 30000, // 30 second timeout per LLM call
  CONFIDENCE_THRESHOLD: 0.7 // Minimum confidence for approval
} as const;

/**
 * Debate protocol configuration
 */
export interface DebateConfig {
  maxRounds?: number;
  timeoutMs?: number;
  approvalRatio?: number;
  minParticipants?: number;
  confidenceThreshold?: number;
  enableByzantineDetection?: boolean;
}

/**
 * Byzantine fault detection result
 */
export interface ByzantineFaultCheck {
  detected: boolean;
  faultyMembers: string[];
  reasons: string[];
  timestamp: number;
}
