/**
 * Chairman - Consensus Synthesis Agent
 * Synthesizes final consensus from council member responses
 */

import type { DebateResponse, ConsensusResult } from './debate-protocol-new';

export class Chairman {
  /**
   * Synthesize consensus from debate responses
   * Implements weighted voting with confidence consideration
   */
  synthesizeConsensus(responses: DebateResponse[], totalRounds: number, duration: number): ConsensusResult {
    if (responses.length === 0) {
      throw new Error('Cannot synthesize consensus with no responses');
    }

    // Count votes
    const votes = {
      approve: responses.filter(r => r.vote === 'approve').length,
      reject: responses.filter(r => r.vote === 'reject').length,
      abstain: responses.filter(r => r.vote === 'abstain').length
    };

    const totalVotes = votes.approve + votes.reject + votes.abstain;
    const agreementRatio = totalVotes > 0 ? votes.approve / totalVotes : 0;

    // Determine decision based on 2/3 threshold
    let decision: 'approved' | 'rejected' | 'needs_revision';
    if (agreementRatio >= 2/3) {
      decision = 'approved';
    } else if (votes.reject / totalVotes >= 2/3) {
      decision = 'rejected';
    } else {
      decision = 'needs_revision';
    }

    // Calculate weighted confidence
    const confidenceSum = responses.reduce((sum, r) => sum + r.confidence, 0);
    const confidence = confidenceSum / responses.length;

    // Synthesize coherent conclusion
    const synthesis = this.generateSynthesis(responses, decision, agreementRatio);

    // Get participating members
    const participatingMembers = responses.map(r => r.memberId);

    // Check for Byzantine faults (timeouts, errors)
    const byzantineFaultDetected = responses.length < 3; // Less than all 3 members responded

    return {
      decision,
      confidence,
      agreementRatio,
      synthesis,
      participatingMembers,
      totalRounds,
      duration,
      votes,
      byzantineFaultDetected,
      timestamp: Date.now()
    };
  }

  /**
   * Generate synthesis text from responses
   */
  private generateSynthesis(
    responses: DebateResponse[],
    decision: string,
    agreementRatio: number
  ): string {
    const parts: string[] = [];

    // Add decision statement
    parts.push(`Council has reached a ${decision} decision with ${(agreementRatio * 100).toFixed(1)}% agreement.`);

    // Summarize each member's position
    responses.forEach(response => {
      parts.push(`${response.role}: ${response.vote} (confidence: ${(response.confidence * 100).toFixed(0)}%).`);
    });

    // Add recommendation
    if (decision === 'approved') {
      parts.push('Recommended for parameter deployment pending Guardian validation.');
    } else if (decision === 'rejected') {
      parts.push('Proposal rejected due to insufficient consensus. Escalation recommended.');
    } else {
      parts.push('Requires revision and additional debate round.');
    }

    return parts.join(' ');
  }

  /**
   * Extract key points from all member responses
   */
  extractKeyPoints(responses: DebateResponse[]): string[] {
    const keyPoints: string[] = [];

    responses.forEach(response => {
      // Extract meaningful sentences from reasoning
      const sentences = response.reasoning.split('.').filter(s => s.trim().length > 20);
      if (sentences.length > 0) {
        keyPoints.push(`${response.role}: ${sentences[0].trim()}.`);
      }
    });

    return keyPoints;
  }

  /**
   * Calculate confidence-weighted agreement
   */
  calculateWeightedAgreement(responses: DebateResponse[]): number {
    if (responses.length === 0) return 0;

    const approveResponses = responses.filter(r => r.vote === 'approve');
    const totalConfidence = responses.reduce((sum, r) => sum + r.confidence, 0);
    const approveConfidence = approveResponses.reduce((sum, r) => sum + r.confidence, 0);

    return totalConfidence > 0 ? approveConfidence / totalConfidence : 0;
  }
}
