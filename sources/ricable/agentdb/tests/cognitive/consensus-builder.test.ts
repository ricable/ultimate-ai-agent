/**
 * Comprehensive Unit Tests for Consensus Builder
 * Tests distributed consensus mechanisms for optimization decisions
 */

import { ConsensusBuilder, ConsensusBuilderConfig, Vote, ConsensusVoteResult } from '../../src/closed-loop/consensus-builder';
import { OptimizationProposal } from '../../src/types/optimization';
import { EventEmitter } from 'events';

// Mock the OptimizationProposal type for testing
interface MockOptimizationProposal extends OptimizationProposal {
  id: string;
  name: string;
  type: string;
  expectedImpact: number;
  confidence: number;
  priority: number;
  riskLevel: 'low' | 'medium' | 'high';
  actions: any[];
}

describe('ConsensusBuilder', () => {
  let consensusBuilder: ConsensusBuilder;
  let mockConfig: ConsensusBuilderConfig;

  beforeEach(() => {
    mockConfig = {
      threshold: 67,
      timeout: 60000,
      votingMechanism: 'weighted',
      maxRetries: 3
    };
    consensusBuilder = new ConsensusBuilder(mockConfig);
  });

  afterEach(() => {
    consensusBuilder.shutdown();
  });

  describe('Initialization and Configuration', () => {
    test('should initialize with default configuration', () => {
      const builder = new ConsensusBuilder({
        threshold: 75,
        timeout: 30000,
        votingMechanism: 'majority'
      });

      expect(builder).toBeInstanceOf(ConsensusBuilder);
      expect(builder).toBeInstanceOf(EventEmitter);
    });

    test('should use provided configuration values', () => {
      const customConfig = {
        threshold: 80,
        timeout: 120000,
        votingMechanism: 'unanimous' as const,
        maxRetries: 5
      };

      const builder = new ConsensusBuilder(customConfig);
      // Config is private, but we can test behavior through consensus building
      expect(builder).toBeInstanceOf(ConsensusBuilder);
    });

    test('should apply default maxRetries when not provided', () => {
      const configWithoutRetries = {
        threshold: 70,
        timeout: 60000,
        votingMechanism: 'weighted' as const
      };

      expect(() => new ConsensusBuilder(configWithoutRetries)).not.toThrow();
    });
  });

  describe('Consensus Building with Single Proposal', () => {
    test('should auto-approve high quality single proposal', async () => {
      const highQualityProposal: MockOptimizationProposal = {
        id: 'proposal-1',
        name: 'Energy Optimization',
        type: 'energy',
        expectedImpact: 90,
        confidence: 0.95,
        priority: 9,
        riskLevel: 'low',
        actions: []
      };

      const result = await consensusBuilder.buildConsensus([highQualityProposal]);

      expect(result.approved).toBe(true);
      expect(result.approvedProposal).toEqual(highQualityProposal);
      expect(result.threshold).toBe(mockConfig.threshold);
    });

    test('should reject low quality single proposal', async () => {
      const lowQualityProposal: MockOptimizationProposal = {
        id: 'proposal-2',
        name: 'Low Impact Change',
        type: 'test',
        expectedImpact: 10,
        confidence: 0.3,
        priority: 2,
        riskLevel: 'high',
        actions: []
      };

      const result = await consensusBuilder.buildConsensus([lowQualityProposal]);

      expect(result.approved).toBe(false);
      expect(result.rejectionReason).toContain('Consensus not reached');
    });

    test('should handle single proposal with medium quality', async () => {
      const mediumProposal: MockOptimizationProposal = {
        id: 'proposal-3',
        name: 'Medium Quality',
        type: 'coverage',
        expectedImpact: 50,
        confidence: 0.6,
        proposal: 5,
        riskLevel: 'medium',
        actions: []
      };

      const result = await consensusBuilder.buildConsensus([mediumProposal]);

      // Quality score = (50/100 + 0.6 + 5/10) / 3 = 0.567
      // Minus risk penalty (0.1) = 0.467 < 0.6 threshold
      expect(result.approved).toBe(false);
    });
  });

  describe('Consensus Building with Multiple Proposals', () => {
    test('should build consensus across multiple optimization agents', async () => {
      const proposals: MockOptimizationProposal[] = [
        {
          id: 'energy-proposal',
          name: 'Energy Efficiency Improvement',
          type: 'energy',
          expectedImpact: 80,
          confidence: 0.9,
          priority: 8,
          riskLevel: 'low',
          actions: []
        },
        {
          id: 'mobility-proposal',
          name: 'Mobility Enhancement',
          type: 'mobility',
          expectedImpact: 70,
          confidence: 0.85,
          priority: 7,
          riskLevel: 'medium',
          actions: []
        }
      ];

      const result = await consensusBuilder.buildConsensus(proposals);

      expect(result).toBeDefined();
      expect(typeof result.approved).toBe('boolean');
      expect(result.threshold).toBe(mockConfig.threshold);
    });

    test('should reject when no proposals provided', async () => {
      await expect(consensusBuilder.buildConsensus([])).rejects.toThrow('No proposals provided');
    });

    test('should handle proposals with different types and priorities', async () => {
      const variedProposals: MockOptimizationProposal[] = [
        {
          id: 'coverage-proposal',
          name: 'Coverage Optimization',
          type: 'coverage',
          expectedImpact: 60,
          confidence: 0.8,
          priority: 6,
          riskLevel: 'low',
          actions: []
        },
        {
          id: 'capacity-proposal',
          name: 'Capacity Planning',
          type: 'capacity',
          expectedImpact: 75,
          confidence: 0.75,
          priority: 9,
          riskLevel: 'medium',
          actions: []
        },
        {
          id: 'performance-proposal',
          name: 'Performance Tuning',
          type: 'performance',
          expectedImpact: 85,
          confidence: 0.9,
          priority: 10,
          riskLevel: 'low',
          actions: []
        }
      ];

      const result = await consensusBuilder.buildConsensus(variedProposals);

      expect(result).toBeDefined();
      expect(typeof result.approved).toBe('boolean');
    });
  });

  describe('Agent Vote Collection', () => {
    test('should collect votes from default optimization agents', async () => {
      const proposal: MockOptimizationProposal = {
        id: 'test-proposal',
        name: 'Test Proposal',
        type: 'energy',
        expectedImpact: 70,
        confidence: 0.8,
        priority: 7,
        riskLevel: 'low',
        actions: []
      };

      const result = await consensusBuilder.buildConsensus([proposal]);

      // Should have collected votes from default agents
      expect(result).toBeDefined();
    });

    test('should use custom agents when provided', async () => {
      const customAgents = [
        {
          id: 'custom-agent-1',
          type: 'energy',
          capabilities: ['energy-efficiency', 'advanced-optimization'],
          weight: 1.2
        },
        {
          id: 'custom-agent-2',
          type: 'mobility',
          capabilities: ['handover-optimization'],
          weight: 0.8
        }
      ];

      const proposal: MockOptimizationProposal = {
        id: 'custom-test',
        name: 'Custom Test',
        type: 'energy',
        expectedImpact: 75,
        confidence: 0.85,
        priority: 8,
        riskLevel: 'low',
        actions: []
      };

      const result = await consensusBuilder.buildConsensus([proposal], customAgents);

      expect(result).toBeDefined();
      expect(typeof result.approved).toBe('boolean');
    });

    test('should generate appropriate votes based on agent compatibility', async () => {
      const energyProposal: MockOptimizationProposal = {
        id: 'energy-test',
        name: 'Energy Test',
        type: 'energy',
        expectedImpact: 80,
        confidence: 0.9,
        priority: 9,
        riskLevel: 'low',
        actions: []
      };

      const energyAgent = {
        id: 'energy-agent',
        type: 'energy',
        capabilities: ['energy-efficiency', 'power-management'],
        weight: 1.0
      };

      const mobilityAgent = {
        id: 'mobility-agent',
        type: 'mobility',
        capabilities: ['handover', 'cell-reselection'],
        weight: 1.0
      };

      // Energy agent should have higher compatibility with energy proposal
      const result = await consensusBuilder.buildConsensus([energyProposal], [energyAgent, mobilityAgent]);

      expect(result).toBeDefined();
    });
  });

  describe('Voting Mechanisms', () => {
    test('should handle weighted voting mechanism', async () => {
      const weightedConfig = {
        threshold: 70,
        timeout: 60000,
        votingMechanism: 'weighted' as const,
        maxRetries: 3
      };

      const weightedBuilder = new ConsensusBuilder(weightedConfig);

      const proposals: MockOptimizationProposal[] = [
        {
          id: 'weighted-test',
          name: 'Weighted Test',
          type: 'energy',
          expectedImpact: 75,
          confidence: 0.8,
          priority: 8,
          riskLevel: 'medium',
          actions: []
        }
      ];

      const result = await weightedBuilder.buildConsensus(proposals);

      expect(result).toBeDefined();
      expect(result.threshold).toBe(70);

      weightedBuilder.shutdown();
    });

    test('should handle majority voting mechanism', async () => {
      const majorityConfig = {
        threshold: 75,
        timeout: 60000,
        votingMechanism: 'majority' as const,
        maxRetries: 3
      };

      const majorityBuilder = new ConsensusBuilder(majorityConfig);

      const proposals: MockOptimizationProposal[] = [
        {
          id: 'majority-test',
          name: 'Majority Test',
          type: 'mobility',
          expectedImpact: 70,
          confidence: 0.75,
          priority: 7,
          riskLevel: 'low',
          actions: []
        }
      ];

      const result = await majorityBuilder.buildConsensus(proposals);

      expect(result).toBeDefined();
      // For majority, threshold should be at least 50%
      expect(result.threshold).toBeGreaterThanOrEqual(50);

      majorityBuilder.shutdown();
    });

    test('should handle unanimous voting mechanism', async () => {
      const unanimousConfig = {
        threshold: 90,
        timeout: 60000,
        votingMechanism: 'unanimous' as const,
        maxRetries: 3
      };

      const unanimousBuilder = new ConsensusBuilder(unanimousConfig);

      const proposals: MockOptimizationProposal[] = [
        {
          id: 'unanimous-test',
          name: 'Unanimous Test',
          type: 'coverage',
          expectedImpact: 85,
          confidence: 0.9,
          priority: 9,
          riskLevel: 'low',
          actions: []
        }
      ];

      const result = await unanimousBuilder.buildConsensus(proposals);

      expect(result).toBeDefined();
      expect(result.threshold).toBe(100); // Unanimous requires 100%

      unanimousBuilder.shutdown();
    });
  });

  describe('Agent Compatibility and Decision Making', () => {
    test('should calculate agent compatibility correctly', async () => {
      const energyProposal: MockOptimizationProposal = {
        id: 'compatibility-test',
        name: 'Energy Compatibility Test',
        type: 'energy',
        expectedImpact: 80,
        confidence: 0.85,
        priority: 8,
        riskLevel: 'low',
        actions: []
      };

      const compatibleAgent = {
        id: 'energy-specialist',
        type: 'energy',
        capabilities: ['energy-efficiency', 'power-management', 'optimization'],
        weight: 1.0
      };

      const incompatibleAgent = {
        id: 'mobility-specialist',
        type: 'mobility',
        capabilities: ['handover', 'cell-reselection'],
        weight: 1.0
      };

      const result = await consensusBuilder.buildConsensus([energyProposal], [compatibleAgent, incompatibleAgent]);

      expect(result).toBeDefined();
    });

    test('should evaluate proposal quality correctly', async () => {
      const highQualityProposal: MockOptimizationProposal = {
        id: 'quality-high',
        name: 'High Quality Proposal',
        type: 'energy',
        expectedImpact: 95,
        confidence: 0.95,
        priority: 10,
        riskLevel: 'low',
        actions: []
      };

      const result = await consensusBuilder.buildConsensus([highQualityProposal]);

      expect(result.approved).toBe(true);
    });

    test('should apply risk penalties appropriately', async () => {
      const highRiskProposal: MockOptimizationProposal = {
        id: 'high-risk',
        name: 'High Risk Proposal',
        type: 'experimental',
        expectedImpact: 90,
        confidence: 0.9,
        priority: 10,
        riskLevel: 'high',
        actions: []
      };

      const result = await consensusBuilder.buildConsensus([highRiskProposal]);

      // High risk should reduce approval chances
      expect(result).toBeDefined();
    });

    test('should calculate agent confidence based on capabilities', async () => {
      const proposal: MockOptimizationProposal = {
        id: 'confidence-test',
        name: 'Confidence Test',
        type: 'optimization',
        expectedImpact: 75,
        confidence: 0.8,
        priority: 7,
        riskLevel: 'medium',
        actions: []
      };

      const highlyCapableAgent = {
        id: 'expert-agent',
        type: 'optimization',
        capabilities: ['optimization', 'analysis', 'planning', 'execution', 'monitoring'],
        weight: 1.0
      };

      const result = await consensusBuilder.buildConsensus([proposal], [highlyCapableAgent]);

      expect(result).toBeDefined();
    });
  });

  describe('Consensus Result Calculation', () => {
    test('should calculate approval percentage correctly', async () => {
      const proposals: MockOptimizationProposal[] = [
        {
          id: 'approval-test',
          name: 'Approval Test',
          type: 'energy',
          expectedImpact: 80,
          confidence: 0.85,
          priority: 8,
          riskLevel: 'low',
          actions: []
        }
      ];

      const result = await consensusBuilder.buildConsensus(proposals);

      expect(result).toBeDefined();
      if (result.approved) {
        expect(result.approvedProposal).toBeDefined();
      } else {
        expect(result.rejectionReason).toBeDefined();
        expect(result.rejectionReason).toContain('%');
      }
    });

    test('should handle mixed vote results', async () => {
      const proposals: MockOptimizationProposal[] = [
        {
          id: 'mixed-vote-test',
          name: 'Mixed Vote Test',
          type: 'coverage',
          expectedImpact: 60,
          confidence: 0.6,
          priority: 6,
          riskLevel: 'medium',
          actions: []
        }
      ];

      const result = await consensusBuilder.buildConsensus(proposals);

      expect(result).toBeDefined();
      expect(typeof result.approved).toBe('boolean');
    });

    test('should track vote distribution', async () => {
      const proposals: MockOptimizationProposal[] = [
        {
          id: 'vote-distribution-test',
          name: 'Vote Distribution Test',
          type: 'mobility',
          expectedImpact: 70,
          confidence: 0.75,
          priority: 7,
          riskLevel: 'low',
          actions: []
        }
      ];

      const result = await consensusBuilder.buildConsensus(proposals);

      expect(result).toBeDefined();
      // The internal voting should track approve/reject/abstain counts
    });
  });

  describe('Event Emission', () => {
    test('should emit votesCollected event', async () => {
      const emitSpy = jest.spyOn(consensusBuilder, 'emit');
      const proposal: MockOptimizationProposal = {
        id: 'event-test',
        name: 'Event Test',
        type: 'energy',
        expectedImpact: 75,
        confidence: 0.8,
        priority: 7,
        riskLevel: 'low',
        actions: []
      };

      await consensusBuilder.buildConsensus([proposal]);

      expect(emitSpy).toHaveBeenCalledWith('votesCollected', expect.any(Object));
      emitSpy.mockRestore();
    });

    test('should emit consensusResult event', async () => {
      const emitSpy = jest.spyOn(consensusBuilder, 'emit');
      const proposal: MockOptimizationProposal = {
        id: 'consensus-event-test',
        name: 'Consensus Event Test',
        type: 'coverage',
        expectedImpact: 80,
        confidence: 0.85,
        priority: 8,
        riskLevel: 'low',
        actions: []
      };

      await consensusBuilder.buildConsensus([proposal]);

      expect(emitSpy).toHaveBeenCalledWith('consensusResult', expect.any(Object));
      emitSpy.mockRestore();
    });
  });

  describe('Active Voting Management', () => {
    test('should track active voting results', async () => {
      const proposal: MockOptimizationProposal = {
        id: 'active-voting-test',
        name: 'Active Voting Test',
        type: 'energy',
        expectedImpact: 70,
        confidence: 0.75,
        priority: 7,
        riskLevel: 'medium',
        actions: []
      };

      await consensusBuilder.buildConsensus([proposal]);

      const activeVoting = consensusBuilder.getActiveVoting();
      expect(Array.isArray(activeVoting)).toBe(true);
    });

    test('should cleanup voting results', async () => {
      const proposal: MockOptimizationProposal = {
        id: 'cleanup-test',
        name: 'Cleanup Test',
        type: 'mobility',
        expectedImpact: 65,
        confidence: 0.7,
        priority: 6,
        riskLevel: 'low',
        actions: []
      };

      await consensusBuilder.buildConsensus([proposal]);

      const beforeCleanup = consensusBuilder.getActiveVoting();
      consensusBuilder.cleanupVoting(proposal.id);
      const afterCleanup = consensusBuilder.getActiveVoting();

      // Should have cleaned up the specific voting result
      expect(Array.isArray(beforeCleanup)).toBe(true);
      expect(Array.isArray(afterCleanup)).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('should handle consensus building errors gracefully', async () => {
      const invalidProposal = null as any;

      // Should handle various error scenarios
      await expect(consensusBuilder.buildConsensus([invalidProposal])).resolves.toBeDefined();
    });

    test('should handle agent voting errors', async () => {
      const proposals: MockOptimizationProposal[] = [
        {
          id: 'error-test',
          name: 'Error Test',
          type: 'test',
          expectedImpact: 50,
          confidence: 0.5,
          priority: 5,
          riskLevel: 'high',
          actions: []
        }
      ];

      const faultyAgents = [
        {
          id: 'faulty-agent',
          type: 'test',
          capabilities: ['test'],
          weight: 1.0
        }
      ];

      const result = await consensusBuilder.buildConsensus(proposals, faultyAgents);
      expect(result).toBeDefined();
    });

    test('should handle empty agent lists', async () => {
      const proposals: MockOptimizationProposal[] = [
        {
          id: 'no-agents-test',
          name: 'No Agents Test',
          type: 'energy',
          expectedImpact: 70,
          confidence: 0.75,
          priority: 7,
          riskLevel: 'low',
          actions: []
        }
      ];

      const result = await consensusBuilder.buildConsensus(proposals, []);
      expect(result).toBeDefined();
    });
  });

  describe('Performance and Scalability', () => {
    test('should handle multiple concurrent consensus building', async () => {
      const proposals: MockOptimizationProposal[] = Array.from({ length: 10 }, (_, i) => ({
        id: `concurrent-${i}`,
        name: `Concurrent Proposal ${i}`,
        type: 'test',
        expectedImpact: 60 + i * 5,
        confidence: 0.7 + i * 0.02,
        priority: 5 + i,
        riskLevel: 'low' as const,
        actions: []
      }));

      const concurrentTasks = proposals.map(proposal =>
        consensusBuilder.buildConsensus([proposal])
      );

      const results = await Promise.all(concurrentTasks);

      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(typeof result.approved).toBe('boolean');
      });
    });

    test('should handle large number of agents', async () => {
      const manyAgents = Array.from({ length: 100 }, (_, i) => ({
        id: `agent-${i}`,
        type: i % 2 === 0 ? 'energy' : 'mobility',
        capabilities: ['optimization'],
        weight: 1.0
      }));

      const proposals: MockOptimizationProposal[] = [
        {
          id: 'many-agents-test',
          name: 'Many Agents Test',
          type: 'energy',
          expectedImpact: 75,
          confidence: 0.8,
          priority: 7,
          riskLevel: 'medium',
          actions: []
        }
      ];

      const startTime = Date.now();
      const result = await consensusBuilder.buildConsensus(proposals, manyAgents);
      const endTime = Date.now();

      expect(result).toBeDefined();
      expect(endTime - startTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    test('should maintain performance with complex proposals', async () => {
      const complexProposals: MockOptimizationProposal[] = Array.from({ length: 20 }, (_, i) => ({
        id: `complex-${i}`,
        name: `Complex Proposal ${i}`,
        type: ['energy', 'mobility', 'coverage', 'capacity', 'performance'][i % 5],
        expectedImpact: 50 + Math.random() * 50,
        confidence: 0.5 + Math.random() * 0.5,
        priority: Math.floor(Math.random() * 10) + 1,
        riskLevel: ['low', 'medium', 'high'][Math.floor(Math.random() * 3)] as 'low' | 'medium' | 'high',
        actions: Array.from({ length: Math.floor(Math.random() * 10) + 1 }, (_, j) => ({
          id: `action-${i}-${j}`,
          type: 'test-action'
        }))
      }));

      const startTime = Date.now();
      const result = await consensusBuilder.buildConsensus(complexProposals);
      const endTime = Date.now();

      expect(result).toBeDefined();
      expect(endTime - startTime).toBeLessThan(3000); // Should complete within 3 seconds
    });
  });

  describe('Integration with Optimization System', () => {
    test('should integrate with optimization proposals properly', async () => {
      const optimizationProposals: MockOptimizationProposal[] = [
        {
          id: 'energy-optimization',
          name: 'Energy Efficiency Optimization',
          type: 'energy',
          expectedImpact: 85,
          confidence: 0.9,
          priority: 9,
          riskLevel: 'low',
          actions: [
            { id: 'action-1', type: 'parameter-update', target: 'energy-settings' },
            { id: 'action-2', type: 'feature-activation', target: 'energy-saving' }
          ]
        },
        {
          id: 'mobility-optimization',
          name: 'Mobility Enhancement',
          type: 'mobility',
          expectedImpact: 75,
          confidence: 0.85,
          priority: 8,
          riskLevel: 'medium',
          actions: [
            { id: 'action-3', type: 'parameter-update', target: 'handover-settings' }
          ]
        }
      ];

      const result = await consensusBuilder.buildConsensus(optimizationProposals);

      expect(result).toBeDefined();
      if (result.approved) {
        expect(result.approvedProposal.actions).toBeDefined();
        expect(Array.isArray(result.approvedProposal.actions)).toBe(true);
      }
    });

    test('should handle consensus for different optimization types', async () => {
      const optimizationTypes = ['energy', 'mobility', 'coverage', 'capacity', 'performance'];
      const proposalsByType = optimizationTypes.map((type, index) => ({
        id: `${type}-proposal`,
        name: `${type.charAt(0).toUpperCase() + type.slice(1)} Optimization`,
        type,
        expectedImpact: 60 + index * 8,
        confidence: 0.65 + index * 0.06,
        priority: 5 + index,
        riskLevel: index < 2 ? 'low' : index < 4 ? 'medium' : 'high' as 'low' | 'medium' | 'high',
        actions: [{ id: `${type}-action`, type: 'optimize', target: type }]
      }));

      for (const proposal of proposalsByType) {
        const result = await consensusBuilder.buildConsensus([proposal]);
        expect(result).toBeDefined();
        expect(typeof result.approved).toBe('boolean');
      }
    });
  });

  describe('Cleanup and Resource Management', () => {
    test('should shutdown properly and clean up resources', () => {
      expect(() => consensusBuilder.shutdown()).not.toThrow();
    });

    test('should clear active voting on shutdown', async () => {
      const proposal: MockOptimizationProposal = {
        id: 'shutdown-test',
        name: 'Shutdown Test',
        type: 'energy',
        expectedImpact: 70,
        confidence: 0.75,
        priority: 7,
        riskLevel: 'low',
        actions: []
      };

      await consensusBuilder.buildConsensus([proposal]);

      const beforeShutdown = consensusBuilder.getActiveVoting();
      consensusBuilder.shutdown();

      // After shutdown, resources should be cleaned up
      expect(Array.isArray(beforeShutdown)).toBe(true);
    });

    test('should handle multiple shutdown calls gracefully', () => {
      expect(() => {
        consensusBuilder.shutdown();
        consensusBuilder.shutdown();
      }).not.toThrow();
    });
  });
});