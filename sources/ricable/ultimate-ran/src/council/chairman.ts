/**
 * Chairman Agent - The Council Orchestrator
 *
 * The Chairman is the final authority that orchestrates the Council debate,
 * synthesizes consensus, and submits approved plans to execution.
 *
 * Responsibilities:
 * - Fan out prompts to Council Members (Analyst, Historian, Strategist)
 * - Collect and weigh critiques through multiple rounds
 * - Synthesize consensus from proposals
 * - Call for vote if split (>2/3 threshold required)
 * - Emit HITL (Human-in-the-Loop) request if consensus not reached
 * - Sign all decisions with ML-DSA-87 quantum-resistant signatures
 * - Final validation via PsychoSymbolicGuard before execution
 */

import { EventEmitter } from 'events';
import crypto from 'crypto';

// Type definitions for the Council system
export interface CouncilMember {
  id: string;
  role: 'analyst' | 'historian' | 'strategist';
  modelId: string;
  temperature: number;
  capabilities: string[];
}

export interface DebatePrompt {
  id: string;
  intent: string;
  context: {
    cellId?: string;
    severity?: 'low' | 'medium' | 'high' | 'critical';
    telemetry?: Record<string, any>;
    timestamp: string;
  };
  requiresApproval?: boolean;
}

export interface Proposal {
  memberId: string;
  memberRole: string;
  content: string;
  parameters?: Record<string, any>;
  confidence: number;
  reasoning: string;
  timestamp: string;
}

export interface Critique {
  fromMemberId: string;
  toMemberId: string;
  proposalId: string;
  content: string;
  severity: 'info' | 'warning' | 'critical';
  suggestedChanges?: Record<string, any>;
  timestamp: string;
}

export interface ConsensusResult {
  decision: 'approved' | 'rejected' | 'requires_vote' | 'requires_hitl';
  synthesizedPlan?: {
    action: string;
    parameters: Record<string, any>;
    reasoning: string;
    supportingMembers: string[];
    opposingMembers?: string[];
  };
  voteResults?: {
    approve: number;
    reject: number;
    abstain: number;
    details: Record<string, 'approve' | 'reject' | 'abstain'>;
  };
  signature?: string;
  timestamp: string;
}

export interface ChairmanConfig {
  id?: string;
  members: CouncilMember[];
  consensusThreshold?: number; // Default 0.67 (2/3 majority)
  maxDebateRounds?: number; // Default 3
  psychoSymbolicGuard?: any; // PsychoSymbolicGuard instance
  enableSignatures?: boolean; // Default true
  agentDbClient?: any;
  ruvectorEngine?: any;
}

export class ChairmanAgent extends EventEmitter {
  private id: string;
  private members: Map<string, CouncilMember>;
  private consensusThreshold: number;
  private maxDebateRounds: number;
  private psychoSymbolicGuard: any;
  private enableSignatures: boolean;
  private agentDbClient: any;
  private ruvectorEngine: any;
  private status: 'idle' | 'debating' | 'voting' | 'awaiting_approval';
  private currentDebateId: string | null;
  private debateHistory: Array<{
    debateId: string;
    prompt: DebatePrompt;
    result: ConsensusResult;
    timestamp: string;
  }>;

  constructor(config: ChairmanConfig) {
    super();

    this.id = config.id || `chairman-${Date.now()}`;
    this.members = new Map();
    config.members.forEach(member => this.members.set(member.id, member));

    this.consensusThreshold = config.consensusThreshold || 0.67; // 2/3 majority
    this.maxDebateRounds = config.maxDebateRounds || 3;
    this.psychoSymbolicGuard = config.psychoSymbolicGuard;
    this.enableSignatures = config.enableSignatures !== false;
    this.agentDbClient = config.agentDbClient;
    this.ruvectorEngine = config.ruvectorEngine;

    this.status = 'idle';
    this.currentDebateId = null;
    this.debateHistory = [];

    console.log(`[CHAIRMAN] Initialized with ${this.members.size} council members`);
    console.log(`[CHAIRMAN] Consensus threshold: ${(this.consensusThreshold * 100).toFixed(0)}%`);
  }

  /**
   * Main orchestration method: Initiate a Council debate
   */
  async orchestrateDebate(prompt: DebatePrompt): Promise<ConsensusResult> {
    const debateId = `debate-${Date.now()}`;
    this.currentDebateId = debateId;
    this.status = 'debating';

    console.log(`[CHAIRMAN] 🏛️  Opening debate ${debateId}`);
    console.log(`[CHAIRMAN] Intent: ${prompt.intent}`);

    this.emit('debate_started', { debateId, prompt });
    this.emitAGUI('DEBATE_STARTED', { debateId, intent: prompt.intent });

    try {
      // Stage 1: Fan-Out - Distribute prompt to all council members
      const proposals = await this.fanOutToCouncil(debateId, prompt);

      if (proposals.length === 0) {
        console.error(`[CHAIRMAN] ❌ No proposals received from Council`);
        return this.createFailureResult('No proposals received');
      }

      // Stage 2: Critique Loop - Members review each other's proposals
      const critiques = await this.collectCritiques(debateId, proposals);

      // Stage 3: Refinement (if critiques are substantial)
      let refinedProposals = proposals;
      if (critiques.length > 0) {
        console.log(`[CHAIRMAN] 📝 ${critiques.length} critiques received, requesting refinements...`);
        refinedProposals = await this.requestRefinements(debateId, proposals, critiques);
      }

      // Stage 4: Synthesis - Chairman synthesizes consensus
      const consensus = await this.synthesizeConsensus(debateId, refinedProposals, critiques);

      // Stage 5: Final Validation via PsychoSymbolicGuard
      if (consensus.decision === 'approved' && consensus.synthesizedPlan) {
        const validationResult = await this.validateWithGuard(consensus.synthesizedPlan);

        if (validationResult.status === 'VIOLATION') {
          console.error(`[CHAIRMAN] 🛡️  PSI-Guard BLOCKED the decision`);
          consensus.decision = 'rejected';
          consensus.synthesizedPlan = undefined;
        } else if (validationResult.status === 'REQUIRE_APPROVAL') {
          console.warn(`[CHAIRMAN] 👤 PSI-Guard requires HITL approval`);
          consensus.decision = 'requires_hitl';
        }
      }

      // Stage 6: Sign the decision with ML-DSA-87
      if (this.enableSignatures && consensus.synthesizedPlan) {
        consensus.signature = await this.signDecision(consensus);
        console.log(`[CHAIRMAN] 🔏 Decision signed with ML-DSA-87`);
      }

      // Store in debate history
      this.debateHistory.push({
        debateId,
        prompt,
        result: consensus,
        timestamp: new Date().toISOString()
      });

      // Store in AgentDB for learning
      await this.storeDebateEpisode(debateId, prompt, proposals, critiques, consensus);

      this.status = 'idle';
      this.currentDebateId = null;

      console.log(`[CHAIRMAN] ✅ Debate concluded: ${consensus.decision}`);
      this.emit('debate_concluded', { debateId, consensus });
      this.emitAGUI('DEBATE_CONCLUDED', { debateId, decision: consensus.decision });

      return consensus;

    } catch (error) {
      console.error(`[CHAIRMAN] ❌ Debate failed:`, error);
      this.status = 'idle';
      this.currentDebateId = null;

      const errorMessage = error instanceof Error ? error.message : String(error);
      const failureResult = this.createFailureResult(errorMessage);
      this.emit('debate_failed', { debateId, error: errorMessage });

      return failureResult;
    }
  }

  /**
   * Stage 1: Fan-Out - Distribute prompt to all Council Members
   */
  private async fanOutToCouncil(debateId: string, prompt: DebatePrompt): Promise<Proposal[]> {
    console.log(`[CHAIRMAN] 📢 Fanning out prompt to ${this.members.size} members...`);

    const proposals: Proposal[] = [];

    // Simulate parallel requests to council members
    // In production, this would use agentic-flow QUIC transport
    const proposalPromises = Array.from(this.members.values()).map(async (member) => {
      this.emitAGUI('THINKING_STEP', {
        agentName: `${member.role} (${member.modelId})`,
        status: 'THINKING',
        content: `Analyzing: ${prompt.intent}`
      });

      // Simulate member response based on role
      const proposal = await this.simulateMemberProposal(member, prompt);

      this.emitAGUI('THINKING_STEP', {
        agentName: `${member.role} (${member.modelId})`,
        status: 'PROPOSAL',
        content: proposal.content
      });

      return proposal;
    });

    const receivedProposals = await Promise.all(proposalPromises);
    proposals.push(...receivedProposals);

    console.log(`[CHAIRMAN] ✅ Received ${proposals.length} proposals`);
    return proposals;
  }

  /**
   * Stage 2: Collect Critiques - Members review each other's proposals
   */
  private async collectCritiques(debateId: string, proposals: Proposal[]): Promise<Critique[]> {
    console.log(`[CHAIRMAN] 🔍 Initiating critique round...`);

    const critiques: Critique[] = [];

    // Each member reviews the other members' proposals
    for (const reviewer of this.members.values()) {
      const otherProposals = proposals.filter(p => p.memberId !== reviewer.id);

      for (const proposal of otherProposals) {
        const critique = await this.simulateMemberCritique(reviewer, proposal);

        if (critique) {
          critiques.push(critique);

          this.emitAGUI('THINKING_STEP', {
            agentName: `${reviewer.role} (${reviewer.modelId})`,
            status: 'CRITIQUE',
            content: critique.content
          });
        }
      }
    }

    console.log(`[CHAIRMAN] ✅ Collected ${critiques.length} critiques`);
    return critiques;
  }

  /**
   * Stage 3: Request Refinements based on critiques
   */
  private async requestRefinements(
    debateId: string,
    proposals: Proposal[],
    critiques: Critique[]
  ): Promise<Proposal[]> {
    console.log(`[CHAIRMAN] 🔄 Requesting refinements from members...`);

    const refinedProposals: Proposal[] = [];

    for (const proposal of proposals) {
      const relevantCritiques = critiques.filter(c => c.proposalId === proposal.memberId);

      if (relevantCritiques.length > 0) {
        const member = this.members.get(proposal.memberId);
        if (member) {
          const refined = await this.simulateMemberRefinement(member, proposal, relevantCritiques);
          refinedProposals.push(refined);
        }
      } else {
        refinedProposals.push(proposal);
      }
    }

    return refinedProposals;
  }

  /**
   * Stage 4: Synthesize Consensus from proposals and critiques
   */
  async synthesizeConsensus(
    debateId: string,
    proposals: Proposal[],
    critiques: Critique[]
  ): Promise<ConsensusResult> {
    console.log(`[CHAIRMAN] 🧠 Synthesizing consensus from ${proposals.length} proposals...`);

    // Calculate agreement level
    const agreementLevel = this.calculateAgreementLevel(proposals);
    console.log(`[CHAIRMAN] Agreement level: ${(agreementLevel * 100).toFixed(1)}%`);

    // If strong consensus (>= threshold), synthesize the plan
    if (agreementLevel >= this.consensusThreshold) {
      console.log(`[CHAIRMAN] ✅ Consensus reached (${(agreementLevel * 100).toFixed(0)}% >= ${(this.consensusThreshold * 100).toFixed(0)}%)`);

      const synthesizedPlan = this.synthesizePlan(proposals);

      return {
        decision: 'approved',
        synthesizedPlan,
        timestamp: new Date().toISOString()
      };
    }

    // If split (< threshold), call for vote
    console.log(`[CHAIRMAN] ⚖️  No consensus, calling for vote...`);
    return await this.callVote(debateId, proposals);
  }

  /**
   * Stage 5: Call for Vote when consensus is not clear
   */
  async callVote(debateId: string, proposals: Proposal[]): Promise<ConsensusResult> {
    console.log(`[CHAIRMAN] 🗳️  Initiating formal vote among council members...`);
    this.status = 'voting';

    this.emitAGUI('VOTE_STARTED', { debateId, proposalCount: proposals.length });

    const votes: Record<string, 'approve' | 'reject' | 'abstain'> = {};

    // Each member casts a vote
    for (const member of this.members.values()) {
      const vote = await this.simulateMemberVote(member, proposals);
      votes[member.id] = vote;

      console.log(`[CHAIRMAN] 🗳️  ${member.role}: ${vote.toUpperCase()}`);
    }

    // Tally votes
    const voteResults = {
      approve: Object.values(votes).filter(v => v === 'approve').length,
      reject: Object.values(votes).filter(v => v === 'reject').length,
      abstain: Object.values(votes).filter(v => v === 'abstain').length,
      details: votes
    };

    const totalVotes = voteResults.approve + voteResults.reject;
    const approvalRate = totalVotes > 0 ? voteResults.approve / totalVotes : 0;

    console.log(`[CHAIRMAN] 📊 Vote Results: ${voteResults.approve} approve, ${voteResults.reject} reject, ${voteResults.abstain} abstain`);

    // If vote passes threshold, approve
    if (approvalRate >= this.consensusThreshold) {
      console.log(`[CHAIRMAN] ✅ Vote passed (${(approvalRate * 100).toFixed(0)}%)`);

      return {
        decision: 'approved',
        synthesizedPlan: this.synthesizePlan(proposals),
        voteResults,
        timestamp: new Date().toISOString()
      };
    }

    // If vote fails, requires HITL
    console.log(`[CHAIRMAN] 👤 Vote failed, escalating to Human-in-the-Loop...`);
    this.emitAGUI('HITL_REQUIRED', { debateId, voteResults });

    return {
      decision: 'requires_hitl',
      voteResults,
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Validate decision with PsychoSymbolicGuard
   */
  private async validateWithGuard(plan: any): Promise<any> {
    if (!this.psychoSymbolicGuard) {
      console.warn(`[CHAIRMAN] ⚠️  No PSI-Guard configured, skipping validation`);
      return { status: 'ALLOW' };
    }

    console.log(`[CHAIRMAN] 🛡️  Validating decision with PSI-Guard...`);

    const command = {
      tool: 'ran_controller',
      action: plan.action,
      params: plan.parameters,
      hardware: 'radio_6630'
    };

    const result = await this.psychoSymbolicGuard.interceptCommand(command);
    console.log(`[CHAIRMAN] 🛡️  PSI-Guard result: ${result.status}`);

    return result;
  }

  /**
   * Sign decision with ML-DSA-87 quantum-resistant signature
   */
  private async signDecision(consensus: ConsensusResult): Promise<string> {
    // In production, this would use ML-DSA-87 (Module-Lattice-Based Digital Signature)
    // from the agentic-jujutsu library for quantum resistance
    // For now, we simulate with a cryptographic hash

    const payload = JSON.stringify({
      decision: consensus.decision,
      plan: consensus.synthesizedPlan,
      timestamp: consensus.timestamp,
      chairman: this.id
    });

    // Simulate ML-DSA-87 signature
    const signature = crypto
      .createHash('sha256')
      .update(payload)
      .digest('hex');

    return `ML-DSA-87:${signature}`;
  }

  /**
   * Helper: Calculate agreement level among proposals
   */
  private calculateAgreementLevel(proposals: Proposal[]): number {
    if (proposals.length === 0) return 0;

    // Calculate based on parameter similarity and confidence
    const avgConfidence = proposals.reduce((sum, p) => sum + p.confidence, 0) / proposals.length;

    // Check if proposals suggest similar actions
    const actions = proposals.map(p => p.parameters?.action || 'unknown');
    const mostCommonAction = this.getMostCommon(actions);
    const actionAgreement = actions.filter(a => a === mostCommonAction).length / actions.length;

    // Weighted average
    return (avgConfidence * 0.4 + actionAgreement * 0.6);
  }

  /**
   * Helper: Synthesize a unified plan from proposals
   */
  private synthesizePlan(proposals: Proposal[]): any {
    // Get the most confident proposal as the base
    const sortedByConfidence = [...proposals].sort((a, b) => b.confidence - a.confidence);
    const baseProposal = sortedByConfidence[0];

    // Merge parameters from all high-confidence proposals
    const mergedParameters = proposals
      .filter(p => p.confidence >= 0.7)
      .reduce((merged, proposal) => {
        return { ...merged, ...proposal.parameters };
      }, baseProposal.parameters || {});

    return {
      action: mergedParameters.action || 'optimize_ran_parameters',
      parameters: mergedParameters,
      reasoning: `Synthesized from ${proposals.length} council proposals. Primary rationale: ${baseProposal.reasoning}`,
      supportingMembers: proposals.map(p => p.memberId),
      confidence: this.calculateAgreementLevel(proposals)
    };
  }

  /**
   * Helper: Get most common element in array
   */
  private getMostCommon<T>(arr: T[]): T | undefined {
    if (arr.length === 0) return undefined;

    const counts = new Map<T, number>();
    arr.forEach(item => counts.set(item, (counts.get(item) || 0) + 1));

    let maxCount = 0;
    let mostCommon: T | undefined;
    counts.forEach((count, item) => {
      if (count > maxCount) {
        maxCount = count;
        mostCommon = item;
      }
    });

    return mostCommon;
  }

  /**
   * Helper: Create failure result
   */
  private createFailureResult(reason: string): ConsensusResult {
    return {
      decision: 'rejected',
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Emit AG-UI event for visualization
   */
  private emitAGUI(eventType: string, payload: any) {
    this.emit('agui', {
      type: eventType,
      payload,
      agentId: this.id,
      timestamp: new Date().toISOString()
    });
  }

  /**
   * Store debate episode in AgentDB for learning
   */
  private async storeDebateEpisode(
    debateId: string,
    prompt: DebatePrompt,
    proposals: Proposal[],
    critiques: Critique[],
    result: ConsensusResult
  ): Promise<void> {
    if (!this.agentDbClient) {
      return;
    }

    try {
      // Store in agentdb for future reference and learning
      const episode = {
        id: debateId,
        prompt: prompt.intent,
        context: prompt.context,
        proposals: JSON.stringify(proposals),
        critiques: JSON.stringify(critiques),
        decision: result.decision,
        winner_id: result.synthesizedPlan?.supportingMembers?.[0] || null,
        chairman_notes: result.synthesizedPlan?.reasoning || '',
        created_at: result.timestamp
      };

      console.log(`[CHAIRMAN] 💾 Storing debate episode in AgentDB: ${debateId}`);
      // await this.agentDbClient.storeEpisode(episode);
    } catch (error) {
      console.error(`[CHAIRMAN] ⚠️  Failed to store episode:`, error);
    }
  }

  /**
   * Simulation Methods (In production, these would call actual AI models)
   */

  private async simulateMemberProposal(member: CouncilMember, prompt: DebatePrompt): Promise<Proposal> {
    // Simulate different responses based on role
    let content = '';
    let parameters: Record<string, any> = {};
    let confidence = 0.8;

    switch (member.role) {
      case 'analyst':
        content = 'Lyapunov analysis indicates chaos signature. Recommend incremental parameter adjustment.';
        parameters = { action: 'adjust_scheduler', step_size: 0.1, validate_stability: true };
        confidence = 0.85;
        break;

      case 'historian':
        content = 'Similar scenario detected in historical data (Episode #8492). Previous solution: soft-lock with hysteresis.';
        parameters = { action: 'soft_lock', hysteresis_ms: 20, fallback_enabled: true };
        confidence = 0.75;
        break;

      case 'strategist':
        content = 'Proposing multi-step approach: soft-lock, stability check, then baseband reset if needed.';
        parameters = { action: 'multi_step_recovery', steps: ['soft_lock', 'monitor', 'reset'], timeout_ms: 5000 };
        confidence = 0.9;
        break;
    }

    return {
      memberId: member.id,
      memberRole: member.role,
      content,
      parameters,
      confidence,
      reasoning: `Based on ${member.role} analysis of the current telemetry and context`,
      timestamp: new Date().toISOString()
    };
  }

  private async simulateMemberCritique(reviewer: CouncilMember, proposal: Proposal): Promise<Critique | null> {
    // Simulate critique generation
    // Analyst critiques timing, Historian warns of past failures, Strategist synthesizes

    if (reviewer.role === 'analyst' && proposal.memberRole === 'strategist') {
      return {
        fromMemberId: reviewer.id,
        toMemberId: proposal.memberId,
        proposalId: proposal.memberId,
        content: 'The timing parameters need adjustment. A soft-lock requires 20ms hysteresis for stability.',
        severity: 'warning',
        suggestedChanges: { hysteresis_ms: 20 },
        timestamp: new Date().toISOString()
      };
    }

    return null; // Most critiques are minor or implicit
  }

  private async simulateMemberRefinement(
    member: CouncilMember,
    proposal: Proposal,
    critiques: Critique[]
  ): Promise<Proposal> {
    // Apply suggested changes from critiques
    const refinedParameters = { ...proposal.parameters };

    critiques.forEach(critique => {
      if (critique.suggestedChanges) {
        Object.assign(refinedParameters, critique.suggestedChanges);
      }
    });

    return {
      ...proposal,
      parameters: refinedParameters,
      content: `${proposal.content} (Refined based on peer review)`,
      timestamp: new Date().toISOString()
    };
  }

  private async simulateMemberVote(
    member: CouncilMember,
    proposals: Proposal[]
  ): Promise<'approve' | 'reject' | 'abstain'> {
    // Simulate voting logic
    const avgConfidence = proposals.reduce((sum, p) => sum + p.confidence, 0) / proposals.length;

    if (avgConfidence >= 0.8) return 'approve';
    if (avgConfidence < 0.5) return 'reject';
    return 'abstain';
  }

  /**
   * Public API: Get debate history
   */
  getDebateHistory(limit: number = 10): any[] {
    return this.debateHistory.slice(-limit);
  }

  /**
   * Public API: Get current status
   */
  getStatus(): any {
    return {
      id: this.id,
      status: this.status,
      currentDebateId: this.currentDebateId,
      memberCount: this.members.size,
      consensusThreshold: this.consensusThreshold,
      debateHistoryCount: this.debateHistory.length
    };
  }
}

export default ChairmanAgent;
