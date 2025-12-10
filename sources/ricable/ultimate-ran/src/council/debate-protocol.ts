/**
 * Debate Protocol - Council Engine Backend
 *
 * Implements the multi-phase debate protocol for the LLM Council Architecture:
 * 1. Fan-Out: Broadcast prompt to all council members
 * 2. Critique: Peer review rounds (typically 2 rounds)
 * 3. Synthesis: Chairman synthesizes consensus
 *
 * Part of the Ericsson Gen 7.0 Titan Platform
 */

import { EventEmitter } from 'events';

/**
 * Council Member definition
 */
export interface CouncilMember {
  id: string;
  role: 'analyst' | 'historian' | 'strategist' | 'chairman';
  modelId: string;
  temperature: number;
  systemPrompt: string;
  tools: string[];
}

/**
 * Proposal from a council member
 */
export interface Proposal {
  memberId: string;
  role: string;
  content: string;
  reasoning: string;
  confidence: number;
  timestamp: string;
}

/**
 * Critique of a proposal
 */
export interface Critique {
  reviewerId: string;
  targetProposalId: string;
  content: string;
  concerns: string[];
  suggestions: string[];
  approval: boolean;
  timestamp: string;
}

/**
 * Consensus result
 */
export interface Consensus {
  decision: string;
  approvedProposals: string[];
  rejectedProposals: string[];
  synthesisReasoning: string;
  confidence: number;
  votes: Record<string, boolean>;
  timestamp: string;
}

/**
 * Debate Round structure
 */
export interface DebateRound {
  roundNumber: number;
  proposals: Proposal[];
  critiques: Critique[];
  consensus: Consensus | null;
  status: 'in_progress' | 'completed' | 'failed';
  startedAt: string;
  completedAt?: string;
}

/**
 * Debate Session
 */
export interface DebateSession {
  id: string;
  prompt: string;
  context: any;
  rounds: DebateRound[];
  finalConsensus: Consensus | null;
  status: 'initialized' | 'fan_out' | 'critique' | 'synthesis' | 'completed' | 'failed';
  createdAt: string;
  completedAt?: string;
}

/**
 * DebateProtocol Configuration
 */
export interface DebateProtocolConfig {
  maxRounds?: number;
  minApprovalRate?: number;
  consensusThreshold?: number;
  timeoutMs?: number;
  enableParallelProcessing?: boolean;
}

/**
 * DebateProtocol class
 * Orchestrates the multi-phase debate process for the Council
 */
export class DebateProtocol extends EventEmitter {
  private config: Required<DebateProtocolConfig>;
  private councilMembers: Map<string, CouncilMember>;
  private activeSessions: Map<string, DebateSession>;

  constructor(config: DebateProtocolConfig = {}) {
    super();

    // Set defaults
    this.config = {
      maxRounds: config.maxRounds ?? 2,
      minApprovalRate: config.minApprovalRate ?? 0.67, // 2/3 majority
      consensusThreshold: config.consensusThreshold ?? 0.75,
      timeoutMs: config.timeoutMs ?? 30000, // 30 seconds
      enableParallelProcessing: config.enableParallelProcessing ?? true
    };

    this.councilMembers = new Map();
    this.activeSessions = new Map();

    console.log('[DEBATE_PROTOCOL] Initialized with config:', this.config);
  }

  /**
   * Register a council member
   */
  registerMember(member: CouncilMember): void {
    this.councilMembers.set(member.id, member);
    console.log(`[DEBATE_PROTOCOL] Registered council member: ${member.id} (${member.role})`);

    this.emit('member_registered', { memberId: member.id, role: member.role });
  }

  /**
   * Start a new debate session
   */
  async startDebate(prompt: string, context: any = {}): Promise<DebateSession> {
    const sessionId = `debate-${Date.now()}`;

    const session: DebateSession = {
      id: sessionId,
      prompt,
      context,
      rounds: [],
      finalConsensus: null,
      status: 'initialized',
      createdAt: new Date().toISOString()
    };

    this.activeSessions.set(sessionId, session);

    console.log(`[DEBATE_PROTOCOL] Started debate session: ${sessionId}`);
    console.log(`[DEBATE_PROTOCOL] Prompt: "${prompt}"`);

    this.emit('debate_started', { sessionId, prompt });

    try {
      // Phase 1: Fan-Out
      await this.fanOut(session);

      // Phase 2: Critique (multiple rounds)
      await this.critiquePhase(session);

      // Phase 3: Synthesis
      await this.synthesisPhase(session);

      session.status = 'completed';
      session.completedAt = new Date().toISOString();

      this.emit('debate_completed', { sessionId, consensus: session.finalConsensus });

    } catch (error) {
      session.status = 'failed';
      console.error(`[DEBATE_PROTOCOL] Debate failed: ${error}`);
      this.emit('debate_failed', { sessionId, error: (error as Error).message });
    }

    return session;
  }

  /**
   * Phase 1: Fan-Out
   * Multicast the prompt to all council members in parallel
   */
  private async fanOut(session: DebateSession): Promise<void> {
    console.log(`[DEBATE_PROTOCOL] Phase 1: Fan-Out - Broadcasting to ${this.councilMembers.size} members`);

    session.status = 'fan_out';
    this.emit('phase_started', { sessionId: session.id, phase: 'fan_out' });

    // Initialize first round
    const round: DebateRound = {
      roundNumber: 1,
      proposals: [],
      critiques: [],
      consensus: null,
      status: 'in_progress',
      startedAt: new Date().toISOString()
    };

    session.rounds.push(round);

    // Collect proposals from all members in parallel
    const proposals = await this.collectProposals(session, round);
    round.proposals = proposals;

    console.log(`[DEBATE_PROTOCOL] Fan-Out complete: Received ${proposals.length} proposals`);
    this.emit('fan_out_complete', { sessionId: session.id, proposalCount: proposals.length });
  }

  /**
   * Collect proposals from all council members
   */
  private async collectProposals(session: DebateSession, round: DebateRound): Promise<Proposal[]> {
    const members = Array.from(this.councilMembers.values())
      .filter(m => m.role !== 'chairman'); // Chairman doesn't propose in initial round

    if (this.config.enableParallelProcessing) {
      // Parallel execution using Promise.all (async gather pattern)
      const proposalPromises = members.map(member =>
        this.requestProposal(session, round, member)
      );

      return Promise.all(proposalPromises);
    } else {
      // Sequential execution
      const proposals: Proposal[] = [];
      for (const member of members) {
        const proposal = await this.requestProposal(session, round, member);
        proposals.push(proposal);
      }
      return proposals;
    }
  }

  /**
   * Request a proposal from a specific council member
   */
  private async requestProposal(
    session: DebateSession,
    round: DebateRound,
    member: CouncilMember
  ): Promise<Proposal> {
    console.log(`[DEBATE_PROTOCOL] Requesting proposal from ${member.id} (${member.role})`);

    this.emit('proposal_requested', {
      sessionId: session.id,
      roundNumber: round.roundNumber,
      memberId: member.id
    });

    // Simulate LLM call via agentic-flow
    // In production, this would call the actual model API
    const proposal: Proposal = {
      memberId: member.id,
      role: member.role,
      content: await this.simulateModelResponse(member, session.prompt, round),
      reasoning: `Analysis from ${member.role} perspective`,
      confidence: 0.7 + Math.random() * 0.3,
      timestamp: new Date().toISOString()
    };

    this.emit('proposal_received', {
      sessionId: session.id,
      memberId: member.id,
      proposal
    });

    return proposal;
  }

  /**
   * Simulate model response (placeholder for actual agentic-flow integration)
   */
  private async simulateModelResponse(
    member: CouncilMember,
    prompt: string,
    round: DebateRound
  ): Promise<string> {
    // In production, this would integrate with agentic-flow to call the actual model
    // For now, return role-specific simulation

    const responses: Record<string, string> = {
      'analyst': `[${member.modelId}] Mathematical analysis: Detecting potential chaos indicators. Lyapunov exponent analysis required.`,
      'historian': `[${member.modelId}] Historical context: Found 3 similar episodes. Episode #8492 suggests caution with power boost.`,
      'strategist': `[${member.modelId}] Strategic recommendation: Implement parameter changes with hysteresis: {tx_power: -3dB, scheduling: adaptive}`
    };

    return responses[member.role] || `[${member.modelId}] Proposal for: ${prompt}`;
  }

  /**
   * Phase 2: Critique
   * Multiple rounds of peer review
   */
  private async critiquePhase(session: DebateSession): Promise<void> {
    console.log(`[DEBATE_PROTOCOL] Phase 2: Critique - Starting ${this.config.maxRounds} rounds`);

    session.status = 'critique';
    this.emit('phase_started', { sessionId: session.id, phase: 'critique' });

    for (let i = 0; i < this.config.maxRounds; i++) {
      const currentRound = session.rounds[session.rounds.length - 1];

      console.log(`[DEBATE_PROTOCOL] Critique Round ${i + 1}/${this.config.maxRounds}`);

      // Collect critiques for current proposals
      const critiques = await this.collectCritiques(session, currentRound);
      currentRound.critiques = critiques;

      // Check if we need another round
      const needsAnotherRound = this.shouldContinueCritique(currentRound, i);

      if (!needsAnotherRound) {
        console.log(`[DEBATE_PROTOCOL] Critique converged after ${i + 1} rounds`);
        break;
      }

      // If not the last round, prepare next round with revised proposals
      if (i < this.config.maxRounds - 1) {
        const nextRound: DebateRound = {
          roundNumber: currentRound.roundNumber + 1,
          proposals: [],
          critiques: [],
          consensus: null,
          status: 'in_progress',
          startedAt: new Date().toISOString()
        };

        session.rounds.push(nextRound);

        // Request revised proposals based on critiques
        nextRound.proposals = await this.collectRevisedProposals(session, currentRound, nextRound);
      }
    }

    this.emit('critique_complete', {
      sessionId: session.id,
      totalRounds: session.rounds.length
    });
  }

  /**
   * Collect critiques from all members
   */
  private async collectCritiques(session: DebateSession, round: DebateRound): Promise<Critique[]> {
    const proposals = round.proposals;
    const critiques: Critique[] = [];

    // Each member critiques all other proposals
    const members = Array.from(this.councilMembers.values())
      .filter(m => m.role !== 'chairman');

    if (this.config.enableParallelProcessing) {
      // Parallel critique collection
      const critiquePromises: Promise<Critique[]>[] = [];

      for (const member of members) {
        for (const proposal of proposals) {
          if (proposal.memberId !== member.id) {
            critiquePromises.push(
              this.requestCritique(session, round, member, proposal)
                .then(critique => [critique])
            );
          }
        }
      }

      const allCritiques = await Promise.all(critiquePromises);
      return allCritiques.flat();

    } else {
      // Sequential critique collection
      for (const member of members) {
        for (const proposal of proposals) {
          if (proposal.memberId !== member.id) {
            const critique = await this.requestCritique(session, round, member, proposal);
            critiques.push(critique);
          }
        }
      }
      return critiques;
    }
  }

  /**
   * Request a critique from a member about a proposal
   */
  private async requestCritique(
    session: DebateSession,
    round: DebateRound,
    member: CouncilMember,
    proposal: Proposal
  ): Promise<Critique> {
    console.log(`[DEBATE_PROTOCOL] ${member.id} critiquing proposal from ${proposal.memberId}`);

    // Simulate critique generation
    const approval = Math.random() > 0.3; // 70% approval rate simulation

    const critique: Critique = {
      reviewerId: member.id,
      targetProposalId: proposal.memberId,
      content: approval
        ? `Proposal from ${proposal.role} is sound. ${member.role} perspective confirms validity.`
        : `Concerns about ${proposal.role}'s proposal. Requires adjustment.`,
      concerns: approval ? [] : ['Timing may be suboptimal', 'Missing hysteresis factor'],
      suggestions: approval ? [] : ['Add 20ms hysteresis', 'Verify neighbor interference'],
      approval,
      timestamp: new Date().toISOString()
    };

    this.emit('critique_generated', {
      sessionId: session.id,
      roundNumber: round.roundNumber,
      reviewerId: member.id,
      targetId: proposal.memberId,
      approval
    });

    return critique;
  }

  /**
   * Collect revised proposals based on critiques
   */
  private async collectRevisedProposals(
    session: DebateSession,
    previousRound: DebateRound,
    currentRound: DebateRound
  ): Promise<Proposal[]> {
    console.log(`[DEBATE_PROTOCOL] Collecting revised proposals for round ${currentRound.roundNumber}`);

    // Only members with rejected proposals need to revise
    const rejectedProposalIds = new Set(
      previousRound.critiques
        .filter(c => !c.approval)
        .map(c => c.targetProposalId)
    );

    const membersToRevise = Array.from(this.councilMembers.values())
      .filter(m => m.role !== 'chairman' && rejectedProposalIds.has(m.id));

    const revisedProposals: Proposal[] = [];

    for (const member of membersToRevise) {
      const proposal = await this.requestProposal(session, currentRound, member);
      revisedProposals.push(proposal);
    }

    // Carry forward approved proposals
    const approvedProposals = previousRound.proposals.filter(
      p => !rejectedProposalIds.has(p.memberId)
    );

    return [...approvedProposals, ...revisedProposals];
  }

  /**
   * Determine if critique should continue
   */
  private shouldContinueCritique(round: DebateRound, currentIteration: number): boolean {
    // Calculate approval rate
    const totalCritiques = round.critiques.length;
    if (totalCritiques === 0) return false;

    const approvedCount = round.critiques.filter(c => c.approval).length;
    const approvalRate = approvedCount / totalCritiques;

    console.log(`[DEBATE_PROTOCOL] Round ${round.roundNumber} approval rate: ${(approvalRate * 100).toFixed(1)}%`);

    // Continue if approval rate is below threshold and we haven't hit max rounds
    return approvalRate < this.config.minApprovalRate && currentIteration < this.config.maxRounds - 1;
  }

  /**
   * Phase 3: Synthesis
   * Chairman synthesizes final consensus
   */
  private async synthesisPhase(session: DebateSession): Promise<void> {
    console.log(`[DEBATE_PROTOCOL] Phase 3: Synthesis - Chairman synthesizing consensus`);

    session.status = 'synthesis';
    this.emit('phase_started', { sessionId: session.id, phase: 'synthesis' });

    const lastRound = session.rounds[session.rounds.length - 1];

    // Chairman synthesizes consensus from all critiques
    const consensus = await this.synthesizeConsensus(session, lastRound);

    lastRound.consensus = consensus;
    session.finalConsensus = consensus;
    lastRound.status = 'completed';
    lastRound.completedAt = new Date().toISOString();

    console.log(`[DEBATE_PROTOCOL] Consensus reached with ${(consensus.confidence * 100).toFixed(1)}% confidence`);

    this.emit('synthesis_complete', {
      sessionId: session.id,
      consensus
    });
  }

  /**
   * Synthesize consensus from critiques
   */
  private async synthesizeConsensus(
    session: DebateSession,
    round: DebateRound
  ): Promise<Consensus> {
    const proposals = round.proposals;
    const critiques = round.critiques;

    // Analyze critiques to determine approved proposals
    const proposalApprovals = new Map<string, { approvals: number; rejections: number }>();

    proposals.forEach(p => {
      proposalApprovals.set(p.memberId, { approvals: 0, rejections: 0 });
    });

    critiques.forEach(c => {
      const stats = proposalApprovals.get(c.targetProposalId);
      if (stats) {
        if (c.approval) {
          stats.approvals++;
        } else {
          stats.rejections++;
        }
      }
    });

    // Determine approved and rejected proposals
    const approvedProposals: string[] = [];
    const rejectedProposals: string[] = [];
    const votes: Record<string, boolean> = {};

    proposalApprovals.forEach((stats, memberId) => {
      const total = stats.approvals + stats.rejections;
      const approvalRate = total > 0 ? stats.approvals / total : 0;
      const approved = approvalRate >= this.config.minApprovalRate;

      votes[memberId] = approved;

      if (approved) {
        approvedProposals.push(memberId);
      } else {
        rejectedProposals.push(memberId);
      }
    });

    // Calculate overall confidence
    const totalVotes = Object.values(votes).length;
    const approvedCount = approvedProposals.length;
    const confidence = totalVotes > 0 ? approvedCount / totalVotes : 0;

    // Generate synthesis reasoning
    const synthesisReasoning = this.generateSynthesisReasoning(
      proposals,
      critiques,
      approvedProposals,
      rejectedProposals
    );

    // Generate final decision
    const decision = this.generateDecision(proposals, approvedProposals);

    const consensus: Consensus = {
      decision,
      approvedProposals,
      rejectedProposals,
      synthesisReasoning,
      confidence,
      votes,
      timestamp: new Date().toISOString()
    };

    return consensus;
  }

  /**
   * Generate synthesis reasoning text
   */
  private generateSynthesisReasoning(
    proposals: Proposal[],
    critiques: Critique[],
    approved: string[],
    rejected: string[]
  ): string {
    const parts: string[] = [
      `Analyzed ${proposals.length} proposals with ${critiques.length} peer critiques.`,
      `Consensus reached: ${approved.length} proposals approved, ${rejected.length} rejected.`
    ];

    if (approved.length > 0) {
      const approvedRoles = proposals
        .filter(p => approved.includes(p.memberId))
        .map(p => p.role)
        .join(', ');
      parts.push(`Approved proposals from: ${approvedRoles}.`);
    }

    if (rejected.length > 0) {
      const rejectedRoles = proposals
        .filter(p => rejected.includes(p.memberId))
        .map(p => p.role)
        .join(', ');
      parts.push(`Rejected proposals from: ${rejectedRoles} due to insufficient peer approval.`);
    }

    return parts.join(' ');
  }

  /**
   * Generate final decision
   */
  private generateDecision(proposals: Proposal[], approvedIds: string[]): string {
    if (approvedIds.length === 0) {
      return 'No consensus reached. Escalation to human-in-the-loop required.';
    }

    const approvedProposals = proposals.filter(p => approvedIds.includes(p.memberId));

    // Combine approved proposals into a unified decision
    const decisions = approvedProposals.map(p =>
      `${p.role}: ${p.content}`
    ).join(' | ');

    return `Consensus Decision: ${decisions}`;
  }

  /**
   * Get active session by ID
   */
  getSession(sessionId: string): DebateSession | undefined {
    return this.activeSessions.get(sessionId);
  }

  /**
   * Get all active sessions
   */
  getAllSessions(): DebateSession[] {
    return Array.from(this.activeSessions.values());
  }

  /**
   * Get debate history (all rounds)
   */
  getDebateHistory(sessionId: string): DebateRound[] {
    const session = this.activeSessions.get(sessionId);
    return session?.rounds || [];
  }

  /**
   * Get council members
   */
  getCouncilMembers(): CouncilMember[] {
    return Array.from(this.councilMembers.values());
  }

  /**
   * Clear completed sessions from memory
   */
  clearCompletedSessions(): number {
    let cleared = 0;
    this.activeSessions.forEach((session, id) => {
      if (session.status === 'completed' || session.status === 'failed') {
        this.activeSessions.delete(id);
        cleared++;
      }
    });

    console.log(`[DEBATE_PROTOCOL] Cleared ${cleared} completed sessions`);
    return cleared;
  }
}

/**
 * Factory function to create a standard council configuration
 */
export function createStandardCouncil(): CouncilMember[] {
  return [
    {
      id: 'analyst-deepseek',
      role: 'analyst',
      modelId: 'deepseek-r1-distill',
      temperature: 0.3,
      systemPrompt: 'You are the Analyst. Focus on mathematical analysis and chaos detection. Ignore history, analyze current telemetry.',
      tools: ['midstream_analyze_chaos', 'ruvector_query_topology']
    },
    {
      id: 'historian-gemini',
      role: 'historian',
      modelId: 'gemini-1.5-pro',
      temperature: 0.5,
      systemPrompt: 'You are the Historian. Query past episodes and similar contexts. Warn if strategies failed before.',
      tools: ['agentdb_query_episodes', 'agentdb_get_reflexion']
    },
    {
      id: 'strategist-claude',
      role: 'strategist',
      modelId: 'claude-3-7-sonnet',
      temperature: 0.7,
      systemPrompt: 'You are the Strategist. Synthesize inputs from Analyst and Historian. Propose concrete parameter changes.',
      tools: ['simulate_gnn_outcome', 'generate_parameter_set']
    },
    {
      id: 'chairman-claude',
      role: 'chairman',
      modelId: 'claude-3-7-sonnet',
      temperature: 0.4,
      systemPrompt: 'You are the Chairman. Listen to the Council. Synthesize consensus. Call for vote if split.',
      tools: ['synthesize_consensus', 'call_vote']
    }
  ];
}

export default DebateProtocol;
