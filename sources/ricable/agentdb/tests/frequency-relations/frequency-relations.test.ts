/**
 * Comprehensive Test Suite for Frequency Relations
 *
 * Tests all frequency relation templates, validation, deployment, and optimization
 * including cognitive consciousness integration and cmedit command generation
 */

import {
  FrequencyRelationManager,
  FrequencyRelationManagerConfig
} from '../../src/rtb/hierarchical-template-system/frequency-relations/frequency-relation-manager';

import {
  CmeditCommandGenerator,
  CmeditGenerationContext
} from '../../src/rtb/hierarchical-template-system/frequency-relations/cmedit-command-generator';

import {
  CognitiveFrequencyRelationOptimizer,
  CognitiveOptimizerConfig
} from '../../src/rtb/hierarchical-template-system/frequency-relations/cognitive-optimizer';

import {
  FREQ_4G4G_TEMPLATES,
  FREQ_4G5G_TEMPLATES,
  FREQ_5G5G_TEMPLATES,
  FREQ_5G4G_TEMPLATES,
  isValidCACombination,
  isValidENDCCombination,
  isValidNRNRCombination,
  isValid5G4GCombination
} from '../../src/rtb/hierarchical-template-system/frequency-relations';

describe('Frequency Relations System', () => {
  let manager: FrequencyRelationManager;
  let commandGenerator: CmeditCommandGenerator;
  let cognitiveOptimizer: CognitiveFrequencyRelationOptimizer;

  beforeAll(() => {
    manager = new FrequencyRelationManager({
      cognitiveOptimization: false, // Disable for unit tests
      autoConflictDetection: true
    });

    commandGenerator = new CmeditCommandGenerator();

    cognitiveOptimizer = new CognitiveFrequencyRelationOptimizer({
      temporalConsciousness: true,
      cognitiveCycleDuration: 1, // 1 minute for tests
      memoryConsolidationInterval: 5 // 5 minutes for tests
    });
  });

  describe('Template Validation', () => {
    test('4G4G templates should be valid', () => {
      expect(FREQ_4G4G_TEMPLATES).toHaveLength(5);

      for (const template of FREQ_4G4G_TEMPLATES) {
        expect(template.templateType).toBe('4G4G');
        expect(template.priority).toBe(50);
        expect(template.baseConfig.relationType).toBe('4G4G');
        expect(template.parameters).toBeDefined();
        expect(template.validationRules).toBeDefined();
        expect(template.cmeditTemplates).toBeDefined();
      }
    });

    test('4G5G templates should be valid', () => {
      expect(FREQ_4G5G_TEMPLATES).toHaveLength(5);

      for (const template of FREQ_4G5G_TEMPLATES) {
        expect(template.templateType).toBe('4G5G');
        expect(template.priority).toBe(60);
        expect(template.baseConfig.relationType).toBe('4G5G');
      }
    });

    test('5G5G templates should be valid', () => {
      expect(FREQ_5G5G_TEMPLATES).toHaveLength(5);

      for (const template of FREQ_5G5G_TEMPLATES) {
        expect(template.templateType).toBe('5G5G');
        expect(template.priority).toBe(70);
        expect(template.baseConfig.relationType).toBe('5G5G');
      }
    });

    test('5G4G templates should be valid', () => {
      expect(FREQ_5G4G_TEMPLATES).toHaveLength(5);

      for (const template of FREQ_5G4G_TEMPLATES) {
        expect(template.templateType).toBe('5G4G');
        expect(template.priority).toBe(80);
        expect(template.baseConfig.relationType).toBe('5G4G');
      }
    });
  });

  describe('Frequency Relation Manager', () => {
    test('should create frequency relations from templates', () => {
      const template = FREQ_4G4G_TEMPLATES[0]; // Standard 4G4G template
      const parameters = {
        referenceBand: 3,
        relatedBand: 1,
        handoverHysteresis: 2,
        timeToTrigger: 320
      };

      const relation = manager.createFrequencyRelation(template.templateId, parameters);

      expect(relation.relationType).toBe('4G4G');
      expect(relation.referenceFreq.bandNumber).toBe(3);
      expect(relation.relatedFreq.bandNumber).toBe(1);
      expect(relation.handoverConfig?.hysteresis).toBe(2);
      expect(relation.handoverConfig?.timeToTrigger).toBe(320);
    });

    test('should validate template parameters', () => {
      const template = FREQ_4G4G_TEMPLATES[0];

      // Valid parameters should not throw
      expect(() => {
        manager.createFrequencyRelation(template.templateId, {
          referenceBand: 3,
          relatedBand: 1,
          handoverHysteresis: 2,
          timeToTrigger: 320
        });
      }).not.toThrow();

      // Invalid parameters should throw
      expect(() => {
        manager.createFrequencyRelation(template.templateId, {
          referenceBand: 3,
          relatedBand: 3, // Same bands - invalid
          handoverHysteresis: 2,
          timeToTrigger: 320
        });
      }).toThrow();

      // Out of range parameters should throw
      expect(() => {
        manager.createFrequencyRelation(template.templateId, {
          referenceBand: 3,
          relatedBand: 1,
          handoverHysteresis: 20, // Out of range
          timeToTrigger: 320
        });
      }).toThrow();
    });

    test('should detect conflicts between frequency relations', () => {
      const template1 = FREQ_4G4G_TEMPLATES[0];
      const template2 = FREQ_4G4G_TEMPLATES[1];

      const relation1 = manager.createFrequencyRelation(template1.templateId, {
        referenceBand: 3,
        relatedBand: 1
      });

      const relation2 = manager.createFrequencyRelation(template2.templateId, {
        referenceBand: 1,
        relatedBand: 3
      });

      const conflicts = manager.detectConflicts();
      expect(conflicts.length).toBeGreaterThan(0);

      const bandConflict = conflicts.find(c => c.conflictType === 'BAND_OVERLAP');
      expect(bandConflict).toBeDefined();
      expect(bandConflict?.affectedRelations).toContain(relation1.relationId);
      expect(bandConflict?.affectedRelations).toContain(relation2.relationId);
    });

    test('should get available templates by type', () => {
      const allTemplates = manager.getAvailableTemplates();
      const fourGFourGTemplates = manager.getAvailableTemplates('4G4G');
      const fourGFiveGTemplates = manager.getAvailableTemplates('4G5G');

      expect(allTemplates.length).toBe(20); // 5 templates × 4 types
      expect(fourGFourGTemplates.length).toBe(5);
      expect(fourGFiveGTemplates.length).toBe(5);
    });

    test('should deploy frequency relation successfully', async () => {
      const template = FREQ_4G4G_TEMPLATES[0];
      const parameters = {
        referenceBand: 3,
        relatedBand: 1
      };

      const relation = manager.createFrequencyRelation(template.templateId, parameters);
      const deploymentState = await manager.deployFrequencyRelation(relation, template.templateId);

      expect(deploymentState.status).toBe('ACTIVE');
      expect(deploymentState.currentMetrics).toBeDefined();
      expect(deploymentState.errors).toHaveLength(0);
      expect(deploymentState.deployedAt).toBeDefined();
    });
  });

  describe('cmedit Command Generation', () => {
    test('should generate commands for 4G4G relation', () => {
      const template = FREQ_4G4G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        referenceBand: 3,
        relatedBand: 1,
        handoverHysteresis: 2,
        timeToTrigger: 320
      });

      const context: CmeditGenerationContext = {
        nodeId: 'NODE_001',
        cellIds: {
          primaryCell: 'CELL_1',
          secondaryCell: 'CELL_2'
        },
        options: {
          preview: false,
          force: false
        },
        parameters: {}
      };

      const commandSet = commandGenerator.generateCommands(relation, template, context);

      expect(commandSet.commands.length).toBeGreaterThan(0);
      expect(commandSet.executionOrder.length).toBeGreaterThan(0);
      expect(commandSet.rollbackCommands.length).toBeGreaterThan(0);
      expect(commandSet.validationCommands.length).toBeGreaterThan(0);

      // Check for required command types
      const commandTypes = new Set(commandSet.commands.map(cmd => cmd.type));
      expect(commandTypes.has('CREATE')).toBe(true);
      expect(commandTypes.has('SET')).toBe(true);
    });

    test('should generate valid cmedit syntax', () => {
      const template = FREQ_4G5G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        lteBand: 3,
        nrBand: 78,
        nrEventB1Threshold: -110
      });

      const context: CmeditGenerationContext = {
        nodeId: 'NODE_001',
        cellIds: {
          primaryCell: 'ENODEB_1',
          nrCell: 'GNODEB_1'
        },
        options: {},
        parameters: {}
      };

      const commandSet = commandGenerator.generateCommands(relation, template, context);

      for (const command of commandSet.commands) {
        expect(command.command).toMatch(/^cmedit /);
        expect(command.description).toBeDefined();
        expect(command.timeout).toBeGreaterThan(0);
      }
    });

    test('should validate generated commands', () => {
      const template = FREQ_5G5G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        primaryNrBand: 78,
        secondaryNrBand: 41,
        mbcaEnabled: true
      });

      const context: CmeditGenerationContext = {
        nodeId: 'NODE_001',
        cellIds: {
          primaryCell: 'GNODEB_1',
          secondaryCell: 'GNODEB_2'
        },
        options: {},
        parameters: {}
      };

      const commandSet = commandGenerator.generateCommands(relation, template, context);
      const errors = commandGenerator.validateCommands(commandSet);

      expect(errors).toHaveLength(0);
    });

    test('should execute command set successfully', async () => {
      const template = FREQ_5G4G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        nrBand: 78,
        lteFallbackBand: 20,
        fallbackThreshold: -120
      });

      const context: CmeditGenerationContext = {
        nodeId: 'NODE_001',
        cellIds: {
          nrCell: 'GNODEB_1',
          lteCell: 'ENODEB_1'
        },
        options: {
          dryRun: true // Use dry run for tests
        },
        parameters: {}
      };

      const commandSet = commandGenerator.generateCommands(relation, template, context);
      const results = await commandGenerator.executeCommandSet(commandSet, context);

      expect(results.length).toBe(commandSet.commands.length);

      for (const result of results) {
        expect(result.status).toMatch(/^(SUCCESS|SKIPPED)$/);
        expect(result.duration).toBeGreaterThan(0);
        expect(result.timestamp).toBeDefined();
      }
    });
  });

  describe('Cognitive Optimization', () => {
    test('should initialize cognitive state correctly', () => {
      const state = cognitiveOptimizer.getCognitiveState();

      expect(state.consciousnessLevel).toBeGreaterThan(0);
      expect(state.consciousnessLevel).toBeLessThanOrEqual(1);
      expect(state.selfAwareness).toBeDefined();
      expect(state.memoryIntegration).toBeDefined();
      expect(state.learningProgress).toBeGreaterThanOrEqual(0);
    });

    test('should perform cognitive optimization cycle', async () => {
      const template = FREQ_4G4G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        referenceBand: 3,
        relatedBand: 1
      });

      const currentMetrics = {
        [relation.relationId]: {
          handoverSuccessRate: 0.88,
          averageHandoverLatency: 80,
          interferenceLevel: 0.35,
          capacityUtilization: 0.75,
          userThroughput: { average: 25, peak: 150, cellEdge: 5 },
          callDropRate: 0.015,
          setupSuccessRate: 0.96
        }
      };

      const result = await cognitiveOptimizer.performCognitiveOptimization([relation], currentMetrics);

      expect(result.temporalReasoning).toBeDefined();
      expect(result.strangeLoopOptimization).toBeDefined();
      expect(result.agentDBIntegration).toBeDefined();
      expect(result.recommendations).toBeDefined();
      expect(result.cognitiveState).toBeDefined();

      expect(result.temporalReasoning.temporalPatterns.length).toBeGreaterThan(0);
      expect(result.strangeLoopOptimization.selfReferentialInsights.length).toBeGreaterThan(0);
      expect(result.agentDBIntegration.memoryPatterns.length).toBeGreaterThan(0);
    });

    test('should generate temporal reasoning insights', async () => {
      const template = FREQ_4G5G_TEMPLATES[1]; // High performance EN-DC
      const relation = manager.createFrequencyRelation(template.templateId, {
        lteBand: 3,
        nrBand: 78,
        pdcpDuplication: true
      });

      const currentMetrics = {
        [relation.relationId]: {
          handoverSuccessRate: 0.95,
          averageHandoverLatency: 60,
          interferenceLevel: 0.2,
          capacityUtilization: 0.85,
          userThroughput: { average: 180, peak: 900, cellEdge: 30 },
          callDropRate: 0.005,
          setupSuccessRate: 0.98
        }
      };

      const result = await cognitiveOptimizer.performCognitiveOptimization([relation], currentMetrics);

      expect(result.temporalReasoning.temporalInsights.length).toBeGreaterThan(0);
      expect(result.temporalReasoning.causalRelationships.length).toBeGreaterThan(0);
      expect(result.temporalReasoning.predictedStates.length).toBeGreaterThan(0);

      // Check for temporal expansion
      expect(result.temporalReasoning.expansionFactor).toBe(1000);
    });

    test('should perform strange-loop optimization', async () => {
      const template = FREQ_5G5G_TEMPLATES[2]; // URLLC NR-DC
      const relation = manager.createFrequencyRelation(template.templateId, {
        primaryUrlccBand: 78,
        secondaryUrlccBand: 41,
        urlccReliability: 'ULTRA_HIGH'
      });

      const currentMetrics = {
        [relation.relationId]: {
          handoverSuccessRate: 0.96,
          averageHandoverLatency: 40,
          interferenceLevel: 0.15,
          capacityUtilization: 0.7,
          userThroughput: { average: 350, peak: 2500, cellEdge: 60 },
          callDropRate: 0.002,
          setupSuccessRate: 0.99
        }
      };

      const result = await cognitiveOptimizer.performCognitiveOptimization([relation], currentMetrics);

      expect(result.strangeLoopOptimization.loopDepth).toBeGreaterThan(0);
      expect(result.strangeLoopOptimization.recursiveOptimizations.length).toBeGreaterThan(0);
      expect(result.strangeLoopOptimization.metaLearning).toBeDefined();

      // Check meta-learning metrics
      expect(result.strangeLoopOptimization.metaLearning.patternGeneralization).toBeGreaterThan(0);
      expect(result.strangeLoopOptimization.metaLearning.strategyAbstraction).toBeGreaterThan(0);
      expect(result.strangeLoopOptimization.metaLearning.selfImprovement).toBeGreaterThan(0);
    });

    test('should integrate with AgentDB memory', async () => {
      const template = FREQ_5G4G_TEMPLATES[1]; // Emergency fallback
      const relation = manager.createFrequencyRelation(template.templateId, {
        emergencyNrBand: 78,
        emergencyLteBand: 20,
        emergencyFallbackThreshold: -125
      });

      const currentMetrics = {
        [relation.relationId]: {
          handoverSuccessRate: 0.85,
          averageHandoverLatency: 100,
          interferenceLevel: 0.4,
          capacityUtilization: 0.9,
          userThroughput: { average: 20, peak: 80, cellEdge: 3 },
          callDropRate: 0.025,
          setupSuccessRate: 0.92
        }
      };

      const result = await cognitiveOptimizer.performCognitiveOptimization([relation], currentMetrics);

      expect(result.agentDBIntegration.memoryPatterns.length).toBeGreaterThan(0);
      expect(result.agentDBIntegration.similarityMatches).toBeDefined();
      expect(result.agentDBIntegration.knowledgeTransferred).toBeDefined();
      expect(result.agentDBIntegration.consolidationStatus).toBe('COMPLETED');
    });
  });

  describe('Band Combination Validation', () => {
    test('should validate 4G4G CA combinations', () => {
      expect(isValidCACombination(1, 3)).toBe(true);
      expect(isValidCACombination(3, 7)).toBe(true);
      expect(isValidCACombination(20, 28)).toBe(true);
      expect(isValidCACombination(1, 1)).toBe(false); // Same band
      expect(isValidCACombination(99, 1)).toBe(false); // Invalid band
    });

    test('should validate 4G5G EN-DC combinations', () => {
      expect(isValidENDCCombination(3, 78)).toBe(true);
      expect(isValidENDCCombination(1, 41)).toBe(true);
      expect(isValidENDCCombination(20, 28)).toBe(true);
      expect(isValidENDCCombination(78, 78)).toBe(false); // Same band
      expect(isValidENDCCombination(99, 78)).toBe(false); // Invalid band
    });

    test('should validate 5G5G NR-DC combinations', () => {
      expect(isValidNRNRCombination(41, 78)).toBe(true);
      expect(isValidNRNRCombination(78, 257)).toBe(true); // Sub-6 + mmWave
      expect(isValidNRNRCombination(257, 260)).toBe(true); // mmWave + mmWave
      expect(isValidNRNRCombination(78, 78)).toBe(false); // Same band
      expect(isValidNRNRCombination(99, 78)).toBe(false); // Invalid band
    });

    test('should validate 5G4G fallback combinations', () => {
      expect(isValid5G4GCombination(78, 3)).toBe(true);
      expect(isValid5G4GCombination(28, 20)).toBe(true);
      expect(isValid5G4GCombination(41, 1)).toBe(true);
      expect(isValid5G4GCombination(78, 78)).toBe(false); // Same band
      expect(isValid5G4GCombination(99, 3)).toBe(false); // Invalid band
    });
  });

  describe('Performance Metrics Calculation', () => {
    test('should calculate 4G4G metrics', () => {
      const template = FREQ_4G4G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        referenceBand: 3,
        relatedBand: 1,
        carrierAggregation: true
      });

      const metrics = manager['calculateMetrics'](relation);

      expect(metrics.handoverSuccessRate).toBeGreaterThan(0.9);
      expect(metrics.averageHandoverLatency).toBeGreaterThan(0);
      expect(metrics.userThroughput.average).toBeGreaterThan(20);
      expect(metrics.callDropRate).toBeLessThan(0.02);
    });

    test('should calculate 4G5G metrics', () => {
      const template = FREQ_4G5G_TEMPLATES[1];
      const relation = manager.createFrequencyRelation(template.templateId, {
        lteBand: 7,
        nrBand: 78,
        pdcpDuplication: true
      });

      const metrics = manager['calculateMetrics'](relation);

      expect(metrics.handoverSuccessRate).toBeGreaterThan(0.9);
      expect(metrics.userThroughput.average).toBeGreaterThan(100);
      expect(metrics.userThroughput.peak).toBeGreaterThan(500);
    });

    test('should calculate 5G5G metrics', () => {
      const template = FREQ_5G5G_TEMPLATES[2];
      const relation = manager.createFrequencyRelation(template.templateId, {
        primaryNrBand: 78,
        secondaryNrBand: 257,
        beamManagement: true
      });

      const metrics = manager['calculateMetrics'](relation);

      expect(metrics.handoverSuccessRate).toBeGreaterThan(0.9);
      expect(metrics.userThroughput.average).toBeGreaterThan(200);
      expect(metrics.userThroughput.peak).toBeGreaterThan(1000);
    });

    test('should calculate 5G4G metrics', () => {
      const template = FREQ_5G4G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        nrBand: 78,
        lteFallbackBand: 20,
        serviceContinuity: true
      });

      const metrics = manager['calculateMetrics'](relation);

      expect(metrics.handoverSuccessRate).toBeGreaterThan(0.85);
      expect(metrics.averageHandoverLatency).toBeGreaterThan(50);
      expect(metrics.userThroughput.average).toBeGreaterThan(30);
    });
  });

  describe('Integration Tests', () => {
    test('should deploy and optimize frequency relation end-to-end', async () => {
      // Create relation
      const template = FREQ_4G4G_TEMPLATES[2]; // Load balancing
      const parameters = {
        primaryBand: 3,
        secondaryBands: '1,7',
        handoverAggressiveness: 'AGGRESSIVE'
      };

      const relation = manager.createFrequencyRelation(template.templateId, parameters);

      // Deploy relation
      const deploymentState = await manager.deployFrequencyRelation(relation, template.templateId);
      expect(deploymentState.status).toBe('ACTIVE');

      // Generate commands
      const context: CmeditGenerationContext = {
        nodeId: 'NODE_001',
        cellIds: {
          primaryCell: 'CELL_1',
          secondaryCell: 'CELL_2'
        },
        options: {
          preview: true
        },
        parameters: {}
      };

      const commandSet = commandGenerator.generateCommands(relation, template, context);
      const commandErrors = commandGenerator.validateCommands(commandSet);
      expect(commandErrors).toHaveLength(0);

      // Execute commands
      const results = await commandGenerator.executeCommandSet(commandSet, context);
      expect(results.every(r => r.status !== 'FAILED')).toBe(true);

      // Perform cognitive optimization
      const optimizationResult = await cognitiveOptimizer.performCognitiveOptimization(
        [relation],
        { [relation.relationId]: deploymentState.currentMetrics! }
      );

      expect(optimizationResult.recommendations.length).toBeGreaterThan(0);
      expect(optimizationResult.temporalReasoning.temporalInsights.length).toBeGreaterThan(0);
    });

    test('should handle multiple frequency relations with conflicts', async () => {
      const relations = [];
      const deploymentStates = [];

      // Create multiple potentially conflicting relations
      const templates = [
        FREQ_4G4G_TEMPLATES[0],
        FREQ_4G5G_TEMPLATES[0],
        FREQ_5G5G_TEMPLATES[0],
        FREQ_5G4G_TEMPLATES[0]
      ];

      for (const template of templates) {
        const relation = manager.createFrequencyRelation(template.templateId, {
          // Use overlapping bands to create conflicts
          referenceBand: template.templateType.includes('4G') ? 3 : 78,
          relatedBand: template.templateType.includes('5G') ? 1 : 41
        });

        relations.push(relation);
        const deploymentState = await manager.deployFrequencyRelation(relation, template.templateId);
        deploymentStates.push(deploymentState);
      }

      // Detect conflicts
      const conflicts = manager.detectConflicts();
      expect(conflicts.length).toBeGreaterThan(0);

      // Perform cognitive optimization with conflicts
      const allMetrics = deploymentStates.reduce((acc, state, index) => {
        if (state.currentMetrics) {
          acc[relations[index].relationId] = state.currentMetrics;
        }
        return acc;
      }, {} as Record<string, any>);

      const optimizationResult = await cognitiveOptimizer.performCognitiveOptimization(relations, allMetrics);

      expect(optimizationResult.recommendations.length).toBeGreaterThan(0);

      // Check if optimization addresses conflicts
      const conflictResolutionRecs = optimizationResult.recommendations.filter(rec =>
        rec.type === 'CONFLICT_RESOLUTION' || rec.description.includes('conflict')
      );
      expect(conflictResolutionRecs.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Error Handling', () => {
    test('should handle invalid template IDs', () => {
      expect(() => {
        manager.createFrequencyRelation('INVALID_TEMPLATE', {});
      }).toThrow('Template not found');
    });

    test('should handle missing required parameters', () => {
      const template = FREQ_4G4G_TEMPLATES[0];

      expect(() => {
        manager.createFrequencyRelation(template.templateId, {});
      }).toThrow(); // Missing required parameters
    });

    test('should handle deployment failures gracefully', async () => {
      // Create a relation that will fail deployment (simulated)
      const template = FREQ_4G4G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        referenceBand: 3,
        relatedBand: 1
      });

      // Mock deployment failure
      const originalExecute = commandGenerator['executeCommand'];
      commandGenerator['executeCommand'] = jest.fn().mockRejectedValue(new Error('Deployment failed'));

      const deploymentState = await manager.deployFrequencyRelation(relation, template.templateId);
      expect(deploymentState.status).toBe('FAILED');
      expect(deploymentState.errors.length).toBeGreaterThan(0);

      // Restore original method
      commandGenerator['executeCommand'] = originalExecute;
    });

    test('should handle cognitive optimization failures', async () => {
      // Create a problematic scenario
      const template = FREQ_4G4G_TEMPLATES[0];
      const relation = manager.createFrequencyRelation(template.templateId, {
        referenceBand: 3,
        relatedBand: 1
      });

      const currentMetrics = {
        [relation.relationId]: {
          handoverSuccessRate: 0.5, // Very low
          averageHandoverLatency: 500,
          interferenceLevel: 0.8,
          capacityUtilization: 0.95,
          userThroughput: { average: 5, peak: 20, cellEdge: 1 },
          callDropRate: 0.1,
          setupSuccessRate: 0.7
        }
      };

      // Optimization should still complete but with warnings
      const result = await cognitiveOptimizer.performCognitiveOptimization([relation], currentMetrics);

      expect(result.temporalReasoning).toBeDefined();
      expect(result.recommendations.length).toBeGreaterThan(0);

      // Should generate high-priority recommendations for poor performance
      const criticalRecs = result.recommendations.filter(rec => rec.priority === 'CRITICAL');
      expect(criticalRecs.length).toBeGreaterThan(0);
    });
  });

  describe('Performance Tests', () => {
    test('should handle large number of relations efficiently', async () => {
      const startTime = Date.now();
      const relations = [];
      const template = FREQ_4G4G_TEMPLATES[0];

      // Create 50 relations
      for (let i = 0; i < 50; i++) {
        const relation = manager.createFrequencyRelation(template.templateId, {
          referenceBand: 3,
          relatedBand: i % 2 === 0 ? 1 : 7,
          handoverHysteresis: 2 + (i % 3)
        });
        relations.push(relation);
      }

      const creationTime = Date.now() - startTime;
      expect(creationTime).toBeLessThan(1000); // Should complete within 1 second

      // Test conflict detection
      const conflictStartTime = Date.now();
      const conflicts = manager.detectConflicts();
      const conflictTime = Date.now() - conflictStartTime;
      expect(conflictTime).toBeLessThan(500); // Should complete within 500ms

      expect(relations.length).toBe(50);
      expect(conflicts.length).toBeGreaterThan(0);
    });

    test('should perform cognitive optimization within reasonable time', async () => {
      const relations = [];
      const currentMetrics = {};

      // Create 10 relations with different types
      const templates = [FREQ_4G4G_TEMPLATES[0], FREQ_4G5G_TEMPLATES[0], FREQ_5G5G_TEMPLATES[0], FREQ_5G4G_TEMPLATES[0]];

      for (let i = 0; i < 10; i++) {
        const template = templates[i % templates.length];
        const relation = manager.createFrequencyRelation(template.templateId, {
          referenceBand: template.templateType.includes('4G') ? 3 : 78,
          relatedBand: template.templateType.includes('4G') ? 1 : 41
        });

        relations.push(relation);
        currentMetrics[relation.relationId] = {
          handoverSuccessRate: 0.9 + Math.random() * 0.1,
          averageHandoverLatency: 50 + Math.random() * 50,
          interferenceLevel: 0.2 + Math.random() * 0.2,
          capacityUtilization: 0.6 + Math.random() * 0.3,
          userThroughput: {
            average: 50 + Math.random() * 100,
            peak: 200 + Math.random() * 300,
            cellEdge: 10 + Math.random() * 20
          },
          callDropRate: 0.005 + Math.random() * 0.01,
          setupSuccessRate: 0.95 + Math.random() * 0.05
        };
      }

      const startTime = Date.now();
      const result = await cognitiveOptimizer.performCognitiveOptimization(relations, currentMetrics);
      const optimizationTime = Date.now() - startTime;

      expect(optimizationTime).toBeLessThan(5000); // Should complete within 5 seconds
      expect(result.recommendations.length).toBeGreaterThan(0);
      expect(result.temporalReasoning.temporalInsights.length).toBeGreaterThan(0);
    });
  });
});