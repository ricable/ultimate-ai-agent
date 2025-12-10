/**
 * Council Orchestrator - Byzantine Fault Tolerant Debate Coordinator
 * Implements multi-round debate with 2/3+1 consensus threshold
 */

import type {
  DebateProposal,
  DebateResponse,
  DebateSession,
  CritiqueRound,
  ConsensusResult,
  DebateConfig
} from './debate-protocol-new';
import { CONSENSUS_THRESHOLDS } from './debate-protocol-new';
import type { LLMRouter } from './router-new';
import type { Chairman } from './chairman-new';
import { COUNCIL_MEMBERS } from './council-definitions';

/**
 * Council Orchestrator
 * Manages multi-model debate with Byzantine fault tolerance
 */
export class CouncilOrchestrator {
  constructor(
    private router: LLMRouter,
    private chairman: Chairman
  ) {}

  /**
   * Initiate a debate session
   * Implements fan-out → critique → synthesis workflow
   */
  async initiateDebate(
    proposal: DebateProposal,
    config?: DebateConfig
  ): Promise<DebateSession> {
    const startTime = Date.now();

    // Create session
    const session: DebateSession = {
      proposalId: proposal.id,
      proposal,
      rounds: [],
      status: 'in_progress',
      startTime
    };

    try {
      // Configure debate parameters
      const maxRounds = config?.maxRounds ?? CONSENSUS_THRESHOLDS.MAX_ROUNDS;
      const timeoutMs = config?.timeoutMs ?? CONSENSUS_THRESHOLDS.TIMEOUT_MS;
      const approvalRatio = config?.approvalRatio ?? CONSENSUS_THRESHOLDS.APPROVAL_RATIO;
      const minParticipants = config?.minParticipants ?? CONSENSUS_THRESHOLDS.MIN_PARTICIPANTS;

      // Round 1: Initial fan-out to all council members
      const round1 = await this.executeFanOut(proposal, timeoutMs);
      session.rounds.push(round1);

      // Check if we have minimum participants
      if (round1.responses.length < minParticipants) {
        session.status = 'failed';
        session.endTime = Date.now();
        return session;
      }

      // Check if consensus reached in round 1
      round1.consensusReached = round1.agreementRatio >= approvalRatio;

      // Additional critique rounds if needed (only if consensus NOT reached)
      if (!round1.consensusReached) {
        let currentRound = 1;
        while (currentRound < maxRounds) {
          currentRound++;

          const critiqueRound = await this.executeCritiqueRound(
            proposal,
            session.rounds[session.rounds.length - 1].responses,
            currentRound,
            timeoutMs
          );

          session.rounds.push(critiqueRound);

          // Check for consensus
          critiqueRound.consensusReached = critiqueRound.agreementRatio >= approvalRatio;

          if (critiqueRound.consensusReached) {
            break;
          }
        }
      }

      // Synthesize final consensus
      const lastRound = session.rounds[session.rounds.length - 1];
      const duration = Date.now() - startTime;

      session.consensus = this.chairman.synthesizeConsensus(
        lastRound.responses,
        session.rounds.length,
        duration
      );

      session.status = 'completed';
      session.endTime = Date.now();

      return session;

    } catch (error) {
      session.status = 'failed';
      session.endTime = Date.now();
      throw error;
    }
  }

  /**
   * Execute fan-out to all council members
   * Collects initial proposals in parallel
   */
  private async executeFanOut(
    proposal: DebateProposal,
    timeoutMs: number
  ): Promise<CritiqueRound> {
    const roundStartTime = Date.now();

    // Get all enabled members
    const members = Object.values(COUNCIL_MEMBERS).filter(m => m.enabled);

    // Fan out to all members in parallel
    const responsePromises = members.map(async (member) => {
      const memberStartTime = Date.now();

      try {
        // Route to appropriate LLM provider
        const llmResponse = await this.router.route(
          member.provider,
          this.createPrompt(proposal, member.role),
          { proposal },
          timeoutMs
        );

        const responseTime = Date.now() - memberStartTime;

        // Parse vote from response
        const vote = this.parseVote(llmResponse.content);

        const response: DebateResponse = {
          memberId: member.id,
          role: member.role,
          model: member.model,
          content: llmResponse.content,
          confidence: llmResponse.confidence ?? 0.8,
          vote,
          reasoning: llmResponse.content,
          timestamp: Date.now(),
          responseTime
        };

        return response;

      } catch (error) {
        // Byzantine fault: member failed to respond
        console.warn(`Member ${member.id} failed:`, error);
        return null;
      }
    });

    // Wait for all responses (or timeouts)
    const allResponses = await Promise.all(responsePromises);

    // Filter out failed responses
    const responses = allResponses.filter((r): r is DebateResponse => r !== null);

    // Calculate agreement ratio
    const approveCount = responses.filter(r => r.vote === 'approve').length;
    const agreementRatio = responses.length > 0 ? approveCount / responses.length : 0;

    return {
      roundNumber: 1,
      responses,
      consensusReached: false,
      agreementRatio,
      timestamp: roundStartTime
    };
  }

  /**
   * Execute a critique round
   * Members review previous responses
   */
  private async executeCritiqueRound(
    proposal: DebateProposal,
    previousResponses: DebateResponse[],
    roundNumber: number,
    timeoutMs: number
  ): Promise<CritiqueRound> {
    const roundStartTime = Date.now();

    const members = Object.values(COUNCIL_MEMBERS).filter(m => m.enabled);

    const responsePromises = members.map(async (member) => {
      const memberStartTime = Date.now();

      try {
        // Create critique prompt with previous round context
        const critiquePrompt = this.createCritiquePrompt(
          proposal,
          previousResponses,
          member.role
        );

        const llmResponse = await this.router.route(
          member.provider,
          critiquePrompt,
          { proposal, previousResponses },
          timeoutMs
        );

        const responseTime = Date.now() - memberStartTime;
        const vote = this.parseVote(llmResponse.content);

        const response: DebateResponse = {
          memberId: member.id,
          role: member.role,
          model: member.model,
          content: llmResponse.content,
          confidence: llmResponse.confidence ?? 0.8,
          vote,
          reasoning: llmResponse.content,
          timestamp: Date.now(),
          responseTime
        };

        return response;

      } catch (error) {
        console.warn(`Member ${member.id} failed in round ${roundNumber}:`, error);
        return null;
      }
    });

    const allResponses = await Promise.all(responsePromises);
    const responses = allResponses.filter((r): r is DebateResponse => r !== null);

    const approveCount = responses.filter(r => r.vote === 'approve').length;
    const agreementRatio = responses.length > 0 ? approveCount / responses.length : 0;

    return {
      roundNumber,
      responses,
      consensusReached: false,
      agreementRatio,
      timestamp: roundStartTime
    };
  }

  /**
   * Create prompt for council member
   */
  private createPrompt(proposal: DebateProposal, role: string): string {
    return `
You are the ${role} in the RAN optimization council.

Proposal: ${proposal.description}

Context:
${JSON.stringify(proposal.context, null, 2)}

Please analyze this proposal and provide your recommendation.
Include your vote: APPROVE, REJECT, or ABSTAIN.
    `.trim();
  }

  /**
   * Create critique prompt with previous round context
   */
  private createCritiquePrompt(
    proposal: DebateProposal,
    previousResponses: DebateResponse[],
    role: string
  ): string {
    const previousVotes = previousResponses.map(r =>
      `${r.role}: ${r.vote} (${r.reasoning.substring(0, 100)}...)`
    ).join('\n');

    return `
You are the ${role} in the RAN optimization council.

Proposal: ${proposal.description}

Previous round votes:
${previousVotes}

After considering your colleagues' perspectives, what is your updated recommendation?
Include your vote: APPROVE, REJECT, or ABSTAIN.
    `.trim();
  }

  /**
   * Parse vote from LLM response
   */
  private parseVote(content: string): 'approve' | 'reject' | 'abstain' {
    const lowerContent = content.toLowerCase();

    if (lowerContent.includes('approve') || lowerContent.includes('recommendation: approve')) {
      return 'approve';
    } else if (lowerContent.includes('reject')) {
      return 'reject';
    } else {
      return 'abstain';
    }
  }
}
