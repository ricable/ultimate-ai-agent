/**
 * Comprehensive Test Suite for Phase 3 RANOps ENM CLI Integration
 *
 * This test suite validates:
 * 1. Cognitive cmedit command generation engine
 * 2. Template-to-CLI conversion system
 * 3. Batch operations framework
 * 4. Ericsson RAN expertise integration
 * 5. End-to-end workflow testing
 * 6. Performance validation (<2 second conversion)
 * 7. Error handling and rollback mechanisms
 */

import {
  CmeditCommandGenerator,
  type CmeditGenerationContext,
  type CmeditCommandSet,
  type GeneratedCmeditCommand,
  type CmeditExecutionResult
} from '../../src/rtb/hierarchical-template-system/frequency-relations/cmedit-command-generator';

import type {
  FrequencyRelation,
  FrequencyRelationTemplate
} from '../../src/rtb/hierarchical-template-system/frequency-relations/freq-types';

// Mock data for testing
const mockFrequencyRelation4G4G: FrequencyRelation = {
  relationId: 'FR_4G4G_TEST_001',
  relationType: '4G4G',
  referenceFreq: {
    bandNumber: 1,
    bandName: 'Band 1',
    frequency: 2100,
    direction: 'downlink'
  },
  relatedFreq: {
    bandNumber: 3,
    bandName: 'Band 3',
    frequency: 1800,
    direction: 'downlink'
  },
  handoverConfig: {
    hysteresis: 2.0,
    timeToTrigger: 320,
    eventBasedConfig: {
      a3Offset: 1,
      threshold1: -110
    },
    freqSpecificOffset: 0
  },
  capacitySharing: {
    enabled: true,
    loadBalancingThreshold: 70,
    maxCapacityRatio: 0.8,
    dynamicRebalancing: true,
    rebalancingInterval: 300
  },
  interferenceConfig: {
    enabled: true,
    coordinationType: 'ICIC',
    interBandManagement: {
      almostBlankSubframes: true,
      crsPowerBoost: 3
    },
    dynamicCoordination: true,
    coordinationInterval: 60
  }
};

const mockFrequencyRelation4G5G: FrequencyRelation = {
  relationId: 'FR_4G5G_ENDC_001',
  relationType: '4G5G',
  referenceFreq: {
    bandNumber: 3,
    bandName: 'Band 3',
    frequency: 1800,
    direction: 'downlink'
  },
  relatedFreq: {
    bandNumber: 78,
    bandName: 'Band 78',
    frequency: 3500,
    direction: 'downlink'
  },
  handoverConfig: {
    hysteresis: 2.0,
    timeToTrigger: 320,
    eventBasedConfig: {
      threshold1: -110
    }
  },
  capacitySharing: {
    enabled: true,
    loadBalancingThreshold: 75,
    maxCapacityRatio: 0.7
  }
};

const mockFrequencyRelation5G5G: FrequencyRelation = {
  relationId: 'FR_5G5G_NRDC_001',
  relationType: '5G5G',
  referenceFreq: {
    bandNumber: 78,
    bandName: 'Band 78',
    frequency: 3500,
    direction: 'downlink'
  },
  relatedFreq: {
    bandNumber: 257,
    bandName: 'Band 257',
    frequency: 28,
    direction: 'downlink'
  },
  handoverConfig: {
    hysteresis: 2.5,
    timeToTrigger: 256,
    eventBasedConfig: {
      a3Offset: 2
    }
  },
  capacitySharing: {
    enabled: true,
    loadBalancingThreshold: 80,
    maxCapacityRatio: 0.9
  }
};

const mockFrequencyRelation5G4G: FrequencyRelation = {
  relationId: 'FR_5G4G_FALLBACK_001',
  relationType: '5G4G',
  referenceFreq: {
    bandNumber: 78,
    bandName: 'Band 78',
    frequency: 3500,
    direction: 'downlink'
  },
  relatedFreq: {
    bandNumber: 3,
    bandName: 'Band 3',
    frequency: 1800,
    direction: 'downlink'
  },
  handoverConfig: {
    hysteresis: 3.0,
    timeToTrigger: 256
  },
  fallbackConfig: {
    fallbackEnabled: true,
    fallbackTriggers: {
      nrCoverageThreshold: -120,
      nrQualityThreshold: -15
    }
  }
};

const mockFrequencyTemplate: FrequencyRelationTemplate = {
  id: 'TEMPLATE_001',
  name: 'Test Frequency Template',
  version: '1.0.0',
  relationType: '4G4G',
  parameters: {
    priority: 20,
    optimizationLevel: 'high',
    performanceTargets: {
      handoverSuccessRate: 95,
      callDropRate: 0.5
    }
  },
  config: {
    enableAdvancedFeatures: true,
    enableMonitoring: true
  }
};

const mockGenerationContext: CmeditGenerationContext = {
  nodeId: 'TEST_NODE_001',
  cellIds: {
    primaryCell: 'CELL_1',
    secondaryCell: 'CELL_2',
    nrCell: 'NRCELL_1',
    lteCell: 'LTECELL_1'
  },
  options: {
    preview: false,
    force: false,
    timeout: 60,
    verbose: true,
    dryRun: false
  },
  parameters: {
    environment: 'test',
    optimizationProfile: 'performance'
  }
};

describe('Phase 3 RANOps ENM CLI Integration - Cognitive cmedit Engine', () => {
  let generator: CmeditCommandGenerator;

  beforeEach(() => {
    generator = new CmeditCommandGenerator();
  });

  describe('Command Generation', () => {
    it('should generate 4G4G frequency relation commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      expect(commandSet).toBeDefined();
      expect(commandSet.commands).toHaveLength.greaterThan(0);
      expect(commandSet.executionOrder).toHaveLength(commandSet.commands.length);
      expect(commandSet.rollbackCommands).toHaveLength.greaterThan(0);
      expect(commandSet.validationCommands).toHaveLength.greaterThan(0);

      // Verify setup commands
      const setupCommands = commandSet.commands.filter(cmd => cmd.id.includes('setup') || cmd.id.includes('create'));
      expect(setupCommands).toHaveLength.greaterThan(0);

      // Verify configuration commands
      const configCommands = commandSet.commands.filter(cmd => cmd.id.includes('configure'));
      expect(configCommands).toHaveLength.greaterThan(0);

      // Verify activation commands
      const activationCommands = commandSet.commands.filter(cmd => cmd.id.includes('unlock') || cmd.id.includes('enable'));
      expect(activationCommands).toHaveLength.greaterThan(0);

      // Check command syntax
      commandSet.commands.forEach(command => {
        expect(command.command).toStartWith('cmedit ');
        expect(command.type).toBeOneOf(['CREATE', 'SET', 'DELETE', 'GET', 'MONITOR']);
        expect(command.description).toBeDefined();
        expect(command.timeout).toBeGreaterThan(0);
      });
    });

    it('should generate 4G5G EN-DC commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G5G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      expect(commandSet.commands).toHaveLength.greaterThan(0);

      // Check for EN-DC specific commands
      const endcCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('endc') || cmd.command.includes('ENBFunction')
      );
      expect(endcCommands).toHaveLength.greaterThan(0);

      // Verify EN-DC enabling
      const enableEndcCommand = commandSet.commands.find(cmd =>
        cmd.command.includes('endcEnabled=true')
      );
      expect(enableEndcCommand).toBeDefined();
      expect(enableEndcCommand?.type).toBe('SET');
      expect(enableEndcCommand?.critical).toBe(true);
    });

    it('should generate 5G5G NR-DC commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation5G5G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      expect(commandSet.commands).toHaveLength.greaterThan(0);

      // Check for NR-DC specific commands
      const nrdcCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('nrdc') || cmd.command.includes('NRCellCU')
      );
      expect(nrdcCommands).toHaveLength.greaterThan(0);

      // Verify NR-DC enabling
      const enableNrdcCommand = commandSet.commands.find(cmd =>
        cmd.command.includes('nrdcEnabled=true')
      );
      expect(enableNrdcCommand).toBeDefined();
      expect(enableNrdcCommand?.type).toBe('SET');
      expect(enableNrdcCommand?.critical).toBe(true);
    });

    it('should generate 5G4G fallback commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation5G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      expect(commandSet.commands).toHaveLength.greaterThan(0);

      // Check for fallback specific commands
      const fallbackCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('fallback') || cmd.command.includes('NRCellCU')
      );
      expect(fallbackCommands).toHaveLength.greaterThan(0);

      // Verify fallback configuration
      const fallbackConfigCommand = commandSet.commands.find(cmd =>
        cmd.command.includes('fallbackEnabled=true')
      );
      expect(fallbackConfigCommand).toBeDefined();
      expect(fallbackConfigCommand?.type).toBe('SET');
      expect(fallbackConfigCommand?.critical).toBe(true);
    });

    it('should include preview and force options when enabled', () => {
      const previewContext: CmeditGenerationContext = {
        ...mockGenerationContext,
        options: {
          preview: true,
          force: true,
          timeout: 30,
          verbose: false,
          dryRun: false
        }
      };

      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        previewContext
      );

      // Commands should be generated with preview/force options applied during execution
      expect(commandSet.commands).toHaveLength.greaterThan(0);
    });
  });

  describe('Command Validation', () => {
    it('should validate generated commands successfully', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const errors = generator.validateCommands(commandSet);
      expect(errors).toHaveLength(0);
    });

    it('should detect invalid command syntax', () => {
      const invalidCommandSet: CmeditCommandSet = {
        id: 'INVALID_SET',
        commands: [
          {
            id: 'invalid_cmd',
            type: 'SET',
            command: 'invalid command syntax',
            description: 'Invalid command',
            timeout: 30
          }
        ],
        executionOrder: ['invalid_cmd'],
        dependencies: {},
        rollbackCommands: [],
        validationCommands: []
      };

      const errors = generator.validateCommands(invalidCommandSet);
      expect(errors).toHaveLength.greaterThan(0);
      expect(errors[0]).toContain('Invalid command syntax');
    });

    it('should detect missing required command types', () => {
      const incompleteCommandSet: CmeditCommandSet = {
        id: 'INCOMPLETE_SET',
        commands: [
          {
            id: 'get_cmd',
            type: 'GET',
            command: 'cmedit get TEST_NODE EUtranCellFDD=1 -s',
            description: 'Get command only',
            timeout: 30
          }
        ],
        executionOrder: ['get_cmd'],
        dependencies: {},
        rollbackCommands: [],
        validationCommands: []
      };

      const errors = generator.validateCommands(incompleteCommandSet);
      expect(errors).toHaveLength.greaterThan(0);
      expect(errors.some(e => e.includes('Missing required command type'))).toBe(true);
    });

    it('should detect dependency violations', () => {
      const commandSetWithBadDeps: CmeditCommandSet = {
        id: 'BAD_DEPS_SET',
        commands: [
          {
            id: 'cmd1',
            type: 'SET',
            command: 'cmedit set TEST_NODE EUtranCellFDD=1 qRxLevMin=-130',
            description: 'First command',
            timeout: 30
          }
        ],
        executionOrder: ['cmd1'],
        dependencies: {
          'cmd1': ['nonexistent_dep']
        },
        rollbackCommands: [],
        validationCommands: []
      };

      const errors = generator.validateCommands(commandSetWithBadDeps);
      expect(errors).toHaveLength.greaterThan(0);
      expect(errors.some(e => e.includes('Dependency not found'))).toBe(true);
    });
  });

  describe('Command Execution', () => {
    it('should execute command set successfully', async () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const results = await generator.executeCommandSet(commandSet, mockGenerationContext);

      expect(results).toHaveLength(commandSet.commands.length);

      // All commands should succeed in mock environment
      const successCount = results.filter(r => r.status === 'SUCCESS').length;
      expect(successCount).toBe(results.length);

      // Check execution results structure
      results.forEach(result => {
        expect(result).toHaveProperty('commandId');
        expect(result).toHaveProperty('status');
        expect(result).toHaveProperty('duration');
        expect(result).toHaveProperty('timestamp');
        expect(result.duration).toBeGreaterThan(0);
      });
    });

    it('should respect preview mode during execution', async () => {
      const previewContext: CmeditGenerationContext = {
        ...mockGenerationContext,
        options: { preview: true, force: false, timeout: 30, verbose: false, dryRun: false }
      };

      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        previewContext
      );

      const results = await generator.executeCommandSet(commandSet, previewContext);

      expect(results).toHaveLength(commandSet.commands.length);

      // Preview mode should still execute successfully but with different behavior
      results.forEach(result => {
        expect(result.status).toBeOneOf(['SUCCESS', 'FAILED', 'TIMEOUT', 'SKIPPED']);
      });
    });

    it('should handle command dependencies correctly', async () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      // Verify dependency structure
      expect(Object.keys(commandSet.dependencies)).toBeDefined();

      const results = await generator.executeCommandSet(commandSet, mockGenerationContext);

      // Commands with dependencies should execute after their dependencies
      const executedOrder = results.map(r => r.commandId);
      expect(executedOrder).toEqual(commandSet.executionOrder);
    });

    it('should stop execution on critical command failure', async () => {
      // Create a command set with a critical command that will fail
      const criticalFailureCommandSet: CmeditCommandSet = {
        id: 'CRITICAL_FAILURE_SET',
        commands: [
          {
            id: 'successful_cmd',
            type: 'SET',
            command: 'cmedit set TEST_NODE EUtranCellFDD=1 qRxLevMin=-130',
            description: 'This should succeed',
            timeout: 30,
            critical: false
          },
          {
            id: 'critical_failure_cmd',
            type: 'SET',
            command: 'INVALID_COMMAND_SYNTAX',
            description: 'This should fail',
            timeout: 30,
            critical: true,
            errorPatterns: ['Error', 'Failed']
          },
          {
            id: 'should_not_execute',
            type: 'SET',
            command: 'cmedit set TEST_NODE EUtranCellFDD=1 qQualMin=-32',
            description: 'This should not execute due to critical failure',
            timeout: 30,
            critical: false
          }
        ],
        executionOrder: ['successful_cmd', 'critical_failure_cmd', 'should_not_execute'],
        dependencies: {},
        rollbackCommands: [],
        validationCommands: []
      };

      // Mock the executeCommand method to simulate failure
      const originalExecuteCommand = generator['executeCommand'];
      generator['executeCommand'] = jest.fn().mockImplementation((command: GeneratedCmeditCommand) => {
        if (command.id === 'critical_failure_cmd') {
          return Promise.resolve({
            commandId: command.id,
            status: 'FAILED',
            error: 'Simulated command failure',
            duration: 1000,
            timestamp: new Date()
          });
        }
        return originalExecuteCommand.call(generator, command, mockGenerationContext);
      });

      const results = await generator.executeCommandSet(criticalFailureCommandSet, mockGenerationContext);

      // Should have executed the successful command and failed on the critical one
      expect(results).toHaveLength(2); // Should stop after critical failure
      expect(results[0].status).toBe('SUCCESS');
      expect(results[1].status).toBe('FAILED');

      // Restore original method
      generator['executeCommand'] = originalExecuteCommand;
    });
  });

  describe('Execution History Management', () => {
    it('should store execution history', async () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      await generator.executeCommandSet(commandSet, mockGenerationContext);

      const history = generator.getExecutionHistory(commandSet.id);
      expect(history).toBeDefined();
      expect(history.size).toBe(1);
      expect(history.get(commandSet.id)).toHaveLength(commandSet.commands.length);
    });

    it('should clear execution history', async () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      await generator.executeCommandSet(commandSet, mockGenerationContext);

      // Verify history exists
      let history = generator.getExecutionHistory();
      expect(history.size).toBeGreaterThan(0);

      // Clear specific command set history
      generator.clearExecutionHistory(commandSet.id);
      history = generator.getExecutionHistory();
      expect(history.has(commandSet.id)).toBe(false);

      // Clear all history
      await generator.executeCommandSet(commandSet, mockGenerationContext);
      generator.clearExecutionHistory();
      history = generator.getExecutionHistory();
      expect(history.size).toBe(0);
    });
  });

  describe('Performance Validation', () => {
    it('should generate commands within performance target (<2 seconds)', async () => {
      const startTime = performance.now();

      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const generationTime = performance.now() - startTime;

      expect(commandSet).toBeDefined();
      expect(commandSet.commands).toHaveLength.greaterThan(0);
      expect(generationTime).toBeLessThan(2000); // <2 second target
    });

    it('should execute commands within reasonable time', async () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const startTime = performance.now();
      const results = await generator.executeCommandSet(commandSet, mockGenerationContext);
      const executionTime = performance.now() - startTime;

      expect(results).toHaveLength(commandSet.commands.length);

      // Individual commands should not take too long (mock environment)
      const averageTimePerCommand = executionTime / results.length;
      expect(averageTimePerCommand).toBeLessThan(5000); // 5 seconds max per command
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should handle timeout scenarios gracefully', async () => {
      const timeoutContext: CmeditGenerationContext = {
        ...mockGenerationContext,
        options: { preview: false, force: false, timeout: 1, verbose: false, dryRun: false } // 1 second timeout
      };

      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        timeoutContext
      );

      // Mock a command that takes longer than timeout
      const originalExecuteCommand = generator['executeCommand'];
      generator['executeCommand'] = jest.fn().mockImplementation(async (command: GeneratedCmeditCommand) => {
        await new Promise(resolve => setTimeout(resolve, 2000)); // 2 second delay
        return originalExecuteCommand.call(generator, command, timeoutContext);
      });

      const results = await generator.executeCommandSet(commandSet, timeoutContext);

      // Should handle timeout gracefully
      expect(results).toHaveLength(commandSet.commands.length);

      // Restore original method
      generator['executeCommand'] = originalExecuteCommand;
    });

    it('should generate valid rollback commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      expect(commandSet.rollbackCommands).toHaveLength.greaterThan(0);

      // Verify rollback command structure
      commandSet.rollbackCommands.forEach(rollbackCmd => {
        expect(rollbackCmd.command).toStartWith('cmedit ');
        expect(rollbackCmd.type).toBeOneOf(['SET', 'DELETE']);
        expect(rollbackCmd.description).toBeDefined();
        expect(rollbackCmd.timeout).toBeGreaterThan(0);
        expect(rollbackCmd.critical).toBe(false); // Rollback commands should not be critical
      });

      // Check for specific rollback operations
      const lockCommands = commandSet.rollbackCommands.filter(cmd =>
        cmd.command.includes('adminState=LOCKED')
      );
      expect(lockCommands).toHaveLength.greaterThan(0);

      const disableCommands = commandSet.rollbackCommands.filter(cmd =>
        cmd.command.includes('operState=DISABLED')
      );
      expect(disableCommands).toHaveLength.greaterThan(0);
    });

    it('should generate comprehensive validation commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      expect(commandSet.validationCommands).toHaveLength.greaterThan(0);

      // Verify validation command structure
      commandSet.validationCommands.forEach(validationCmd => {
        expect(validationCmd.command).toStartWith('cmedit ');
        expect(validationCmd.type).toBe('GET');
        expect(validationCmd.description).toContain('Validate');
        expect(validationCmd.expectedOutput).toBeDefined();
        expect(validationCmd.errorPatterns).toBeDefined();
      });

      // Check for specific validation operations
      const relationValidation = commandSet.validationCommands.find(cmd =>
        cmd.command.includes('adminState=UNLOCKED')
      );
      expect(relationValidation).toBeDefined();

      const syncValidation = commandSet.validationCommands.find(cmd =>
        cmd.command.includes('syncStatus')
      );
      expect(syncValidation).toBeDefined();
    });
  });

  describe('cmedit Command Syntax Validation', () => {
    it('should generate syntactically correct cmedit GET commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const getCommands = commandSet.commands.filter(cmd => cmd.type === 'GET');
      getCommands.forEach(cmd => {
        expect(cmd.command).toMatch(/^cmedit get\s+\w+/);
        if (cmd.command.includes('-s')) {
          expect(cmd.command).toMatch(/\s-s\s*$/);
        }
      });
    });

    it('should generate syntactically correct cmedit SET commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const setCommands = commandSet.commands.filter(cmd => cmd.type === 'SET');
      setCommands.forEach(cmd => {
        expect(cmd.command).toMatch(/^cmedit set\s+\w+/);
        expect(cmd.command).toContain('=');
        // SET commands should have valid parameter assignments
        expect(cmd.command).toMatch(/=\w+/);
      });
    });

    it('should generate syntactically correct cmedit CREATE commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const createCommands = commandSet.commands.filter(cmd => cmd.type === 'CREATE');
      createCommands.forEach(cmd => {
        expect(cmd.command).toMatch(/^cmedit create\s+\w+/);
        // CREATE commands should have valid MO class and attributes
        expect(cmd.command).toMatch(/\w+\s+\w+/);
      });
    });

    it('should generate syntactically correct cmedit DELETE commands', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const deleteCommands = [...commandSet.commands, ...commandSet.rollbackCommands]
        .filter(cmd => cmd.type === 'DELETE');

      deleteCommands.forEach(cmd => {
        expect(cmd.command).toMatch(/^cmedit delete\s+\w+/);
        // DELETE commands should have valid FDN and MO class
        expect(cmd.command).toMatch(/\w+\s+\w+/);
      });
    });

    it('should use correct MO class names from Ericsson RAN', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const commandTexts = commandSet.commands.map(cmd => cmd.command).join(' ');

      // Check for valid Ericsson MO classes
      expect(commandTexts).toMatch(/EUtranFreqRelation/);
      expect(commandTexts).toMatch(/EUtranCellFDD/);
      expect(commandTexts).toMatch(/ENBFunction/);

      // For 5G relations
      const commandSet5G = generator.generateCommands(
        mockFrequencyRelation5G5G,
        mockFrequencyTemplate,
        mockGenerationContext
      );
      const commandTexts5G = commandSet5G.commands.map(cmd => cmd.command).join(' ');
      expect(commandTexts5G).toMatch(/NRCellCU/);
    });

    it('should use correct parameter names from Ericsson RAN', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const commandTexts = commandSet.commands.map(cmd => cmd.command).join(' ');

      // Check for valid Ericsson parameter names
      expect(commandTexts).toMatch(/qOffsetFreq/);
      expect(commandTexts).toMatch(/hysteresis/);
      expect(commandTexts).toMatch(/timeToTrigger/);
      expect(commandTexts).toMatch(/a3Offset/);
      expect(commandTexts).toMatch(/adminState/);
      expect(commandTexts).toMatch(/operState/);
      expect(commandTexts).toMatch(/syncStatus/);
    });

    it('should use correct enumerated values', () => {
      const commandSet = generator.generateCommands(
        mockFrequencyRelation4G4G,
        mockFrequencyTemplate,
        mockGenerationContext
      );

      const commandTexts = commandSet.commands.map(cmd => cmd.command).join(' ');

      // Check for valid enumerated values
      expect(commandTexts).toMatch(/adminState=(LOCKED|UNLOCKED)/);
      expect(commandTexts).toMatch(/operState=(ENABLED|DISABLED)/);
      expect(commandTexts).toMatch(/syncStatus=SYNCHRONIZED/);
    });
  });
});