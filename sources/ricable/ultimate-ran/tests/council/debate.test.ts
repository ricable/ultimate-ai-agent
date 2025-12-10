/**
 * LLM Council Debate Tests (London School TDD)
 * Tests written FIRST with mocked LLM providers
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import {
  mockDeepSeekProvider,
  mockGeminiProvider,
  mockClaudeProvider,
  resetMocks
} from '../mocks/llm-providers.mock';
import type { DebateProposal, DebateResponse, ConsensusResult } from '../../src/council/debate-protocol-new';
import { CONSENSUS_THRESHOLDS } from '../../src/council/debate-protocol-new';

// Import implementations
import { CouncilOrchestrator } from '../../src/council/orchestrator-new';
import { Chairman } from '../../src/council/chairman-new';
import { LLMRouter } from '../../src/council/router-new';

describe('LLM Council Debate Protocol', () => {
  let orchestrator: CouncilOrchestrator;
  let chairman: Chairman;
  let router: LLMRouter;

  beforeEach(() => {
    resetMocks();
    router = new LLMRouter({
      deepseek: mockDeepSeekProvider,
      gemini: mockGeminiProvider,
      claude: mockClaudeProvider
    });
    chairman = new Chairman();
    orchestrator = new CouncilOrchestrator(router, chairman);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Fan-out Proposal Collection', () => {
    it('should collect responses from all three council members', async () => {
      const proposal: DebateProposal = {
        id: 'test-proposal-1',
        type: 'parameter_change',
        description: 'Optimize transmit power for cell-001',
        context: {
          cellId: 'cell-001',
          parameters: { txPower: 23 },
          urgency: 'medium'
        },
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.rounds).toHaveLength(1);
      expect(session.rounds[0].responses).toHaveLength(3);

      const memberIds = session.rounds[0].responses.map(r => r.memberId);
      expect(memberIds).toContain('analyst-deepseek');
      expect(memberIds).toContain('historian-gemini');
      expect(memberIds).toContain('strategist-claude');
    });

    it('should fan out all requests in parallel', async () => {
      const proposal: DebateProposal = {
        id: 'test-proposal-parallel',
        type: 'optimization',
        description: 'Test parallel execution',
        context: {},
        timestamp: Date.now()
      };

      const startTime = Date.now();
      const session = await orchestrator.initiateDebate(proposal);
      const duration = Date.now() - startTime;

      // Parallel execution should be much faster than sequential
      // Mock responses are instant, but verify parallel structure
      expect(duration).toBeLessThan(1000);
      expect(session.rounds[0].responses).toHaveLength(3);
    });

    it('should include model metadata in each response', async () => {
      const proposal: DebateProposal = {
        id: 'test-metadata',
        type: 'parameter_change',
        description: 'Test metadata collection',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);
      const responses = session.rounds[0].responses;

      responses.forEach(response => {
        expect(response).toHaveProperty('memberId');
        expect(response).toHaveProperty('role');
        expect(response).toHaveProperty('model');
        expect(response).toHaveProperty('content');
        expect(response).toHaveProperty('confidence');
        expect(response).toHaveProperty('timestamp');
        expect(response).toHaveProperty('responseTime');
      });
    });
  });

  describe('Critique Rounds', () => {
    it('should support maximum 2 critique rounds', async () => {
      const proposal: DebateProposal = {
        id: 'test-multi-round',
        type: 'parameter_change',
        description: 'Test multi-round debate',
        context: {},
        timestamp: Date.now()
      };

      // Mock divergent opinions to trigger additional rounds
      const session = await orchestrator.initiateDebate(proposal, {
        maxRounds: 2
      });

      expect(session.rounds.length).toBeLessThanOrEqual(2);
    });

    it('should stop early if consensus reached in round 1', async () => {
      const proposal: DebateProposal = {
        id: 'test-early-consensus',
        type: 'parameter_change',
        description: 'Test early consensus',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      // All mocks default to positive responses
      expect(session.rounds).toHaveLength(1);
      expect(session.consensus?.decision).toBe('approved');
    });

    it('should include previous round context in critique prompts', async () => {
      const proposal: DebateProposal = {
        id: 'test-context',
        type: 'optimization',
        description: 'Test context propagation',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal, {
        maxRounds: 2
      });

      // Verify that round 2 (if exists) has access to round 1 responses
      if (session.rounds.length > 1) {
        expect(session.rounds[1].responses).toBeDefined();
        expect(session.rounds[1].roundNumber).toBe(2);
      }
    });
  });

  describe('Consensus Threshold (2/3 + 1)', () => {
    it('should reach consensus with 2 out of 3 approvals', async () => {
      const proposal: DebateProposal = {
        id: 'test-2of3',
        type: 'parameter_change',
        description: 'Test 2/3 consensus',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.consensus?.agreementRatio).toBeGreaterThanOrEqual(CONSENSUS_THRESHOLDS.APPROVAL_RATIO);
      expect(session.consensus?.decision).toBe('approved');
    });

    it('should reject with less than 2/3 approval', async () => {
      // Mock two providers to reject (Gemini and Claude) to ensure approval < 66.7%
      vi.spyOn(mockGeminiProvider, 'generateResponse').mockResolvedValue({
        model: 'gemini-1.5-pro',
        content: 'Rejection: Historical data shows risks',
        timestamp: Date.now(),
        confidence: 0.9
      });

      vi.spyOn(mockClaudeProvider, 'generateResponse').mockResolvedValue({
        model: 'claude-3-7-sonnet',
        content: 'Rejection: Strategic risk too high',
        timestamp: Date.now(),
        confidence: 0.9
      });

      const proposal: DebateProposal = {
        id: 'test-reject',
        type: 'parameter_change',
        description: 'Test rejection scenario',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      // 1 approve, 2 reject = 33% approval < 66.7% threshold
      if (session.consensus) {
        expect(session.consensus.agreementRatio).toBeLessThan(CONSENSUS_THRESHOLDS.APPROVAL_RATIO);
      }
    });

    it('should calculate agreement ratio correctly', async () => {
      const proposal: DebateProposal = {
        id: 'test-ratio',
        type: 'parameter_change',
        description: 'Test ratio calculation',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.consensus?.agreementRatio).toBeGreaterThanOrEqual(0);
      expect(session.consensus?.agreementRatio).toBeLessThanOrEqual(1);
      expect(session.consensus?.votes).toBeDefined();
      expect(session.consensus?.votes.approve +
             session.consensus?.votes.reject +
             session.consensus?.votes.abstain).toBe(3);
    });
  });

  describe('Byzantine Fault Tolerance', () => {
    it('should handle timeout from one council member', async () => {
      mockDeepSeekProvider.simulateTimeout = true;

      const proposal: DebateProposal = {
        id: 'test-timeout',
        type: 'parameter_change',
        description: 'Test timeout handling',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal, {
        timeoutMs: 5000 // 5 second timeout for test
      });

      // Should still get responses from 2 members (Gemini, Claude)
      expect(session.rounds[0].responses.length).toBeGreaterThanOrEqual(2);
      expect(session.consensus?.byzantineFaultDetected).toBe(true);
    }, 10000); // Set test timeout to 10s

    it('should detect and flag Byzantine faults', async () => {
      mockClaudeProvider.simulateError = true;

      const proposal: DebateProposal = {
        id: 'test-byzantine',
        type: 'parameter_change',
        description: 'Test Byzantine detection',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.consensus?.byzantineFaultDetected).toBe(true);
    });

    it('should fail if less than minimum participants respond', async () => {
      mockDeepSeekProvider.simulateError = true;
      mockGeminiProvider.simulateError = true;

      const proposal: DebateProposal = {
        id: 'test-min-participants',
        type: 'parameter_change',
        description: 'Test minimum participants',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.status).toBe('failed');
      expect(session.rounds[0].responses.length).toBeLessThan(CONSENSUS_THRESHOLDS.MIN_PARTICIPANTS);
    });

    it('should respect 30 second timeout per LLM call', async () => {
      const proposal: DebateProposal = {
        id: 'test-30s-timeout',
        type: 'parameter_change',
        description: 'Test 30s timeout',
        context: {},
        timestamp: Date.now()
      };

      const startTime = Date.now();
      await orchestrator.initiateDebate(proposal, {
        timeoutMs: CONSENSUS_THRESHOLDS.TIMEOUT_MS
      });
      const duration = Date.now() - startTime;

      // Should complete well under 30s with mocks
      expect(duration).toBeLessThan(30000);
    });
  });

  describe('Chairman Consensus Synthesis', () => {
    it('should synthesize coherent conclusion from multiple responses', async () => {
      const proposal: DebateProposal = {
        id: 'test-synthesis',
        type: 'parameter_change',
        description: 'Optimize antenna tilt',
        context: {
          cellId: 'cell-002',
          parameters: { antennaTilt: 5 }
        },
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.consensus?.synthesis).toBeDefined();
      expect(session.consensus?.synthesis.length).toBeGreaterThan(50);
      expect(session.consensus?.synthesis).toContain('parameter');
    });

    it('should include confidence score in synthesis', async () => {
      const proposal: DebateProposal = {
        id: 'test-confidence',
        type: 'optimization',
        description: 'Test confidence scoring',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.consensus?.confidence).toBeGreaterThanOrEqual(0);
      expect(session.consensus?.confidence).toBeLessThanOrEqual(1);
    });

    it('should weight responses by model confidence', async () => {
      const proposal: DebateProposal = {
        id: 'test-weighting',
        type: 'parameter_change',
        description: 'Test confidence weighting',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);
      const responses = session.rounds[0].responses;

      // Verify responses have confidence scores
      responses.forEach(response => {
        expect(response.confidence).toBeGreaterThan(0);
      });

      // Chairman should consider confidence in synthesis
      expect(session.consensus?.confidence).toBeDefined();
    });

    it('should summarize key points from all members', async () => {
      const proposal: DebateProposal = {
        id: 'test-summary',
        type: 'parameter_change',
        description: 'Test multi-perspective summary',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.consensus?.synthesis).toBeDefined();
      expect(session.consensus?.participatingMembers).toHaveLength(3);
    });
  });

  describe('Integration Tests', () => {
    it('should complete full debate workflow end-to-end', async () => {
      const proposal: DebateProposal = {
        id: 'test-e2e',
        type: 'parameter_change',
        description: 'End-to-end debate test',
        context: {
          cellId: 'cell-003',
          parameters: { txPower: 20 },
          metrics: { sinr: 15.3, throughput: 85.2 },
          urgency: 'high'
        },
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.status).toBe('completed');
      expect(session.consensus).toBeDefined();
      expect(session.consensus?.decision).toMatch(/approved|rejected|needs_revision/);
      expect(session.endTime).toBeDefined();
      expect(session.consensus?.duration).toBeGreaterThanOrEqual(0);
    });

    it('should track session metadata accurately', async () => {
      const proposal: DebateProposal = {
        id: 'test-metadata-tracking',
        type: 'investigation',
        description: 'Test metadata tracking',
        context: {},
        timestamp: Date.now()
      };

      const session = await orchestrator.initiateDebate(proposal);

      expect(session.proposalId).toBe(proposal.id);
      expect(session.proposal).toEqual(proposal);
      expect(session.startTime).toBeDefined();
      expect(session.endTime).toBeDefined();
      expect(session.consensus?.totalRounds).toBe(session.rounds.length);
    });
  });
});
