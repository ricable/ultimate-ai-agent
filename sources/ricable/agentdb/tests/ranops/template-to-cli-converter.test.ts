/**
 * Template-to-CLI Converter Test Suite
 *
 * Tests the conversion of JSON RTB templates to cmedit commands with:
 * 1. JSON template parsing and validation
 * 2. Cognitive command generation
 * 3. Ericsson RAN expertise integration
 * 4. Template inheritance resolution
 * 5. Performance validation
 */

import type {
  RTBTemplate,
  CustomFunction,
  MetaConfig
} from '../../src/rtb/hierarchical-template-system/types';

// Template-to-CLI converter (we'll need to create this interface)
interface TemplateToCliConverter {
  convertTemplate(template: RTBTemplate, context: ConversionContext): Promise<CliCommandSet>;
  validateTemplate(template: RTBTemplate): ValidationResult[];
  optimizeCommands(commands: CliCommand[]): CliCommand[];
}

interface ConversionContext {
  nodeId: string;
  environment: 'test' | 'staging' | 'production';
  options: {
    preview?: boolean;
    force?: boolean;
    optimizeForPerformance?: boolean;
    includeRollback?: boolean;
  };
  parameters: Record<string, any>;
}

interface CliCommandSet {
  id: string;
  commands: CliCommand[];
  executionOrder: string[];
  dependencies: Record<string, string[]>;
  rollbackCommands: CliCommand[];
  validationCommands: CliCommand[];
  metadata: {
    templateId: string;
    conversionTime: number;
    commandCount: number;
  };
}

interface CliCommand {
  id: string;
  type: 'GET' | 'SET' | 'CREATE' | 'DELETE' | 'MONITOR';
  command: string;
  description: string;
  timeout: number;
  critical?: boolean;
  validationCommand?: string;
  sourceTemplate: string;
}

interface ValidationResult {
  severity: 'error' | 'warning' | 'info';
  message: string;
  path?: string;
}

// Mock template data
const mockBaseTemplate: RTBTemplate = {
  $meta: {
    version: '2.0.0',
    author: ['Ericsson RAN Test Suite'],
    description: 'Base RTB template for testing',
    tags: ['base', 'test'],
    environment: 'test'
  },
  $custom: [
    {
      name: 'calculateOptimalTilt',
      args: ['distance', 'cell_height', 'traffic_load'],
      body: [
        'base_tilt = 0.0',
        'distance_factor = min(1.0, distance / 5.0)',
        'load_factor = traffic_load / 100.0',
        'optimal_tilt = base_tilt + (10 * distance_factor) + (5 * load_factor)',
        'return round(optimal_tilt, 1)'
      ]
    }
  ],
  ManagedElement: {
    managedElementId: 'TEST-RAN-001',
    userLabel: 'Test RAN Node',
    aiEnabled: true,
    cognitiveLevel: 'maximum'
  },
  ENBFunction: {
    eNodeBId: '1',
    maxConnectedUe: 1200,
    maxEnbSupportedUe: 1200,
    endcEnabled: true,
    splitBearerSupport: true
  },
  EUtranCellFDD: [
    {
      euTranCellFddId: '1',
      cellId: '1',
      pci: 100,
      freqBand: '20',
      pointAArfcnDl: '647394',
      pointAArfcnUl: '647394',
      qRxLevMin: -140,
      qQualMin: -32,
      cellBarred: 'NOT_BARRED',
      administrativestate: 'UNLOCKED',
      $cond: {
        enableHighCapacity: {
          if: 'user_density > 500',
          then: {
            cellCapMaxCellSubCap: 50000,
            cellSubscriptionCapacity: 30000
          },
          else: '__ignore__'
        }
      },
      $eval: {
        optimalTilt: {
          eval: 'calculateOptimalTilt',
          args: [2.5, 35, 75]
        }
      }
    }
  ]
};

const mockUrbanTemplate: RTBTemplate = {
  $meta: {
    version: '2.0.0',
    description: 'Urban high-capacity variant',
    priority: 20,
    tags: ['urban', 'high-capacity'],
    inherits_from: 'base_template'
  },
  $custom: [
    {
      name: 'optimizeUrbanCapacity',
      args: ['user_density', 'cell_count', 'traffic_profile'],
      body: [
        'base_capacity = cell_count * 1000',
        'density_factor = min(2.0, user_density / 500.0)',
        'traffic_factor = sum([2 if t == "video" else 1 for t in traffic_profile]) / len(traffic_profile)',
        'optimal_capacity = int(base_capacity * density_factor * traffic_factor)',
        'return {',
        '    "target_capacity": optimal_capacity,',
        '    "load_balancing": True,',
        '    "enhanced_features": ["MassiveMIMO", "CarrierAggregation"]',
        '}'
      ]
    }
  ],
  EUtranCellFDD: [
    {
      euTranCellFddId: '1',
      massiveMimoEnabled: 1,
      caEnabled: 1,
      cellCapMaxCellSubCap: 60000,
      cellSubscriptionCapacity: 40000,
      $eval: {
        urbanOptimization: {
          eval: 'optimizeUrbanCapacity',
          args: [750, 3, ['video', 'data', 'voice']]
        }
      }
    }
  ]
};

const mockMobilityTemplate: RTBTemplate = {
  $meta: {
    version: '2.0.0',
    description: 'High mobility optimization variant',
    priority: 30,
    tags: ['mobility', 'high-speed'],
    inherits_from: 'base_template'
  },
  $custom: [
    {
      name: 'optimizeHighMobilityParameters',
      args: ['velocity_km_h', 'handover_success_rate'],
      body: [
        'velocity_factor = min(2.0, velocity_km_h / 120.0)',
        'base_hysteresis = 2.0',
        'mobility_hysteresis = base_hysteresis + (4 * velocity_factor)',
        'base_ttt = 320',
        'velocity_ttt = base_ttt - int(velocity_factor * 160)',
        'return {',
        '    "hysteresis": round(mobility_hysteresis, 1),',
        '    "time_to_trigger": max(100, velocity_ttt),',
        '    "a3_offset": 1 if velocity_factor > 1.0 else 3',
        '}',
      ]
    }
  ],
  AnrFunction: {
    removeEnbTime: 5,
    removeGnbTime: 5,
    pciConflictCellSelection: 'ON',
    maxTimeEventBasedPciConf: 20
  },
  EUtranFreqRelation: [
    {
      euTranFreqRelationId: '1',
      hysteresis: 4.0,
      timeToTrigger: 160,
      a3Offset: 1,
      $eval: {
        mobilityOptimization: {
          eval: 'optimizeHighMobilityParameters',
          args: [150, 92.5]
        }
      }
    }
  ]
};

const mockFrequencyRelationTemplate: RTBTemplate = {
  $meta: {
    version: '2.0.0',
    description: '4G5G EN-DC frequency relation',
    priority: 60,
    tags: ['4g5g', 'en-dc'],
    inherits_from: 'base_template'
  },
  ENBFunction: {
    endcEnabled: true,
    splitBearerSupport: true,
    nrEventB1Threshold: -110,
    nrEventB1Hysteresis: 2,
    nrEventB1TimeToTrigger: 320
  },
  NRFreqRelation: [
    {
      nrFreqRelationId: '1',
      referenceFreq: 1300,
      relatedFreq: 78,
      nrFreqRelationToEUTRAN: {
        qOffsetCell: '0dB',
        scgFailureInfoNR: 0,
        eutraNrSameFreqInd: 0
      }
    }
  ]
};

const mockConversionContext: ConversionContext = {
  nodeId: 'TEST_NODE_001',
  environment: 'test',
  options: {
    preview: false,
    force: false,
    optimizeForPerformance: true,
    includeRollback: true
  },
  parameters: {
    user_density: 750,
    average_velocity: 150,
    traffic_profile: ['video', 'data', 'voice'],
    site_type: 'urban'
  }
};

// Mock TemplateToCliConverter implementation
class MockTemplateToCliConverter implements TemplateToCliConverter {
  async convertTemplate(template: RTBTemplate, context: ConversionContext): Promise<CliCommandSet> {
    const startTime = performance.now();

    const commands: CliCommand[] = [];
    const commandIdCounter = { value: 1 };

    // Process ManagedElement
    if (template.ManagedElement) {
      commands.push(...this.generateManagedElementCommands(template.ManagedElement, context, commandIdCounter));
    }

    // Process ENBFunction
    if (template.ENBFunction) {
      commands.push(...this.generateENBFunctionCommands(template.ENBFunction, context, commandIdCounter));
    }

    // Process EUtranCellFDD
    if (template.EUtranCellFDD) {
      for (const cell of Array.isArray(template.EUtranCellFDD) ? template.EUtranCellFDD : [template.EUtranCellFDD]) {
        commands.push(...this.generateEUtranCellFDDCommands(cell, context, commandIdCounter));
      }
    }

    // Process AnrFunction
    if (template.AnrFunction) {
      commands.push(...this.generateAnrFunctionCommands(template.AnrFunction, context, commandIdCounter));
    }

    // Process EUtranFreqRelation
    if (template.EUtranFreqRelation) {
      for (const freqRel of Array.isArray(template.EUtranFreqRelation) ? template.EUtranFreqRelation : [template.EUtranFreqRelation]) {
        commands.push(...this.generateEUtranFreqRelationCommands(freqRel, context, commandIdCounter));
      }
    }

    // Process NRFreqRelation
    if (template.NRFreqRelation) {
      for (const nrFreqRel of Array.isArray(template.NRFreqRelation) ? template.NRFreqRelation : [template.NRFreqRelation]) {
        commands.push(...this.generateNRFreqRelationCommands(nrFreqRel, context, commandIdCounter));
      }
    }

    const conversionTime = performance.now() - startTime;
    const commandSetId = `${template.$meta?.description || 'Template'}_${Date.now()}`;

    return {
      id: commandSetId,
      commands,
      executionOrder: commands.map(cmd => cmd.id),
      dependencies: {},
      rollbackCommands: context.options.includeRollback ? this.generateRollbackCommands(commands) : [],
      validationCommands: this.generateValidationCommands(commands),
      metadata: {
        templateId: commandSetId,
        conversionTime,
        commandCount: commands.length
      }
    };
  }

  validateTemplate(template: RTBTemplate): ValidationResult[] {
    const results: ValidationResult[] = [];

    // Validate $meta
    if (!template.$meta) {
      results.push({
        severity: 'error',
        message: 'Template missing $meta section'
      });
    } else {
      if (!template.$meta.version) {
        results.push({
          severity: 'warning',
          message: 'Template missing version in $meta'
        });
      }
      if (!template.$meta.description) {
        results.push({
          severity: 'warning',
          message: 'Template missing description in $meta'
        });
      }
    }

    // Validate $custom functions
    if (template.$custom) {
      for (const func of template.$custom) {
        if (!func.name) {
          results.push({
            severity: 'error',
            message: 'Custom function missing name'
          });
        }
        if (!func.body || func.body.length === 0) {
          results.push({
            severity: 'error',
            message: `Custom function ${func.name} missing body`
          });
        }
      }
    }

    return results;
  }

  optimizeCommands(commands: CliCommand[]): CliCommand[] {
    // Apply performance optimizations
    return commands.map(cmd => ({
      ...cmd,
      timeout: Math.min(cmd.timeout, 60), // Cap timeout at 60 seconds
      description: cmd.description.includes('optimized') ? cmd.description : `Optimized: ${cmd.description}`
    }));
  }

  private generateManagedElementCommands(me: any, context: ConversionContext, counter: { value: number }): CliCommand[] {
    const commands: CliCommand[] = [];

    commands.push({
      id: `me_${counter.value++}`,
      type: 'SET',
      command: `cmedit set ${context.nodeId} ManagedElement=1 managedElementId=${me.managedElementId},userLabel="${me.userLabel}"`,
      description: 'Configure ManagedElement',
      timeout: 30,
      sourceTemplate: 'ManagedElement'
    });

    if (me.aiEnabled) {
      commands.push({
        id: `me_ai_${counter.value++}`,
        type: 'SET',
        command: `cmedit set ${context.nodeId} ManagedElement=1 aiEnabled=${me.aiEnabled},cognitiveLevel="${me.cognitiveLevel}"`,
        description: 'Enable AI features',
        timeout: 30,
        sourceTemplate: 'ManagedElement'
      });
    }

    return commands;
  }

  private generateENBFunctionCommands(enb: any, context: ConversionContext, counter: { value: number }): CliCommand[] {
    const commands: CliCommand[] = [];

    let command = `cmedit set ${context.nodeId} ENBFunction=1`;
    const params: string[] = [];

    if (enb.eNodeBId) params.push(`eNodeBId=${enb.eNodeBId}`);
    if (enb.maxConnectedUe) params.push(`maxConnectedUe=${enb.maxConnectedUe}`);
    if (enb.endcEnabled !== undefined) params.push(`endcEnabled=${enb.endcEnabled}`);
    if (enb.splitBearerSupport !== undefined) params.push(`splitBearerSupport=${enb.splitBearerSupport}`);

    if (params.length > 0) {
      command += ` ${params.join(',')}`;
      commands.push({
        id: `enb_${counter.value++}`,
        type: 'SET',
        command,
        description: 'Configure ENBFunction',
        timeout: 30,
        sourceTemplate: 'ENBFunction'
      });
    }

    // Add EN-DC specific commands
    if (enb.endcEnabled) {
      const endcCommand = `cmedit set ${context.nodeId} ENBFunction=1 nrEventB1Threshold=${enb.nrEventB1Threshold || -110},nrEventB1Hysteresis=${enb.nrEventB1Hysteresis || 2},nrEventB1TimeToTrigger=${enb.nrEventB1TimeToTrigger || 320}`;
      commands.push({
        id: `enb_endc_${counter.value++}`,
        type: 'SET',
        command: endcCommand,
        description: 'Configure EN-DC parameters',
        timeout: 30,
        critical: true,
        sourceTemplate: 'ENBFunction'
      });
    }

    return commands;
  }

  private generateEUtranCellFDDCommands(cell: any, context: ConversionContext, counter: { value: number }): CliCommand[] {
    const commands: CliCommand[] = [];

    let command = `cmedit set ${context.nodeId} EUtranCellFDD=${cell.euTranCellFddId}`;
    const params: string[] = [];

    if (cell.cellId) params.push(`cellId=${cell.cellId}`);
    if (cell.pci) params.push(`pci=${cell.pci}`);
    if (cell.freqBand) params.push(`freqBand=${cell.freqBand}`);
    if (cell.pointAArfcnDl) params.push(`pointAArfcnDl=${cell.pointAArfcnDl}`);
    if (cell.qRxLevMin !== undefined) params.push(`qRxLevMin=${cell.qRxLevMin}`);
    if (cell.qQualMin !== undefined) params.push(`qQualMin=${cell.qQualMin}`);
    if (cell.cellBarred) params.push(`cellBarred=${cell.cellBarred}`);
    if (cell.administrativestate) params.push(`administrativestate=${cell.administrativestate}`);

    // Add conditional parameters based on context
    if (context.parameters.user_density > 500 && cell.cellCapMaxCellSubCap) {
      params.push(`cellCapMaxCellSubCap=${cell.cellCapMaxCellSubCap}`);
    }
    if (context.parameters.user_density > 500 && cell.cellSubscriptionCapacity) {
      params.push(`cellSubscriptionCapacity=${cell.cellSubscriptionCapacity}`);
    }

    // Add advanced features for urban environments
    if (context.parameters.site_type === 'urban') {
      if (cell.massiveMimoEnabled !== undefined) params.push(`massiveMimoEnabled=${cell.massiveMimoEnabled}`);
      if (cell.caEnabled !== undefined) params.push(`caEnabled=${cell.caEnabled}`);
    }

    if (params.length > 0) {
      command += ` ${params.join(',')}`;
      commands.push({
        id: `cell_${cell.euTranCellFddId}_${counter.value++}`,
        type: 'SET',
        command,
        description: `Configure EUtranCellFDD ${cell.euTranCellFddId}`,
        timeout: 30,
        critical: true,
        sourceTemplate: 'EUtranCellFDD'
      });
    }

    return commands;
  }

  private generateAnrFunctionCommands(anr: any, context: ConversionContext, counter: { value: number }): CliCommand[] {
    const commands: CliCommand[] = [];

    let command = `cmedit set ${context.nodeId} AnrFunction=1`;
    const params: string[] = [];

    if (anr.removeEnbTime !== undefined) params.push(`removeEnbTime=${anr.removeEnbTime}`);
    if (anr.removeGnbTime !== undefined) params.push(`removeGnbTime=${anr.removeGnbTime}`);
    if (anr.pciConflictCellSelection) params.push(`pciConflictCellSelection=${anr.pciConflictCellSelection}`);
    if (anr.maxTimeEventBasedPciConf !== undefined) params.push(`maxTimeEventBasedPciConf=${anr.maxTimeEventBasedPciConf}`);

    if (params.length > 0) {
      command += ` ${params.join(',')}`;
      commands.push({
        id: `anr_${counter.value++}`,
        type: 'SET',
        command,
        description: 'Configure ANR function for high mobility',
        timeout: 30,
        sourceTemplate: 'AnrFunction'
      });
    }

    return commands;
  }

  private generateEUtranFreqRelationCommands(freqRel: any, context: ConversionContext, counter: { value: number }): CliCommand[] {
    const commands: CliCommand[] = [];

    let command = `cmedit set ${context.nodeId} EUtranFreqRelation=${freqRel.euTranFreqRelationId}`;
    const params: string[] = [];

    if (freqRel.hysteresis !== undefined) params.push(`hysteresis=${freqRel.hysteresis}`);
    if (freqRel.timeToTrigger !== undefined) params.push(`timeToTrigger=${freqRel.timeToTrigger}`);
    if (freqRel.a3Offset !== undefined) params.push(`a3Offset=${freqRel.a3Offset}`);

    // Apply mobility optimizations
    if (context.parameters.average_velocity > 100) {
      // High mobility optimization
      const mobilityHysteresis = freqRel.hysteresis ? freqRel.hysteresis + 2 : 4.0;
      const mobilityTtt = freqRel.timeToTrigger ? Math.max(100, freqRel.timeToTrigger - 160) : 160;
      params.push(`hysteresis=${mobilityHysteresis}`, `timeToTrigger=${mobilityTtt}`);
    }

    if (params.length > 0) {
      command += ` ${params.join(',')}`;
      commands.push({
        id: `freqrel_${freqRel.euTranFreqRelationId}_${counter.value++}`,
        type: 'SET',
        command,
        description: `Configure frequency relation ${freqRel.euTranFreqRelationId}`,
        timeout: 30,
        sourceTemplate: 'EUtranFreqRelation'
      });
    }

    return commands;
  }

  private generateNRFreqRelationCommands(nrFreqRel: any, context: ConversionContext, counter: { value: number }): CliCommand[] {
    const commands: CliCommand[] = [];

    // Create NR frequency relation
    commands.push({
      id: `nr_freq_create_${counter.value++}`,
      type: 'CREATE',
      command: `cmedit create ${context.nodeId} NRFreqRelation NRFreqRelationId=${nrFreqRel.nrFreqRelationId}`,
      description: `Create NR frequency relation ${nrFreqRel.nrFreqRelationId}`,
      timeout: 30,
      critical: true,
      sourceTemplate: 'NRFreqRelation'
    });

    // Configure NR frequency relation
    let configCommand = `cmedit set ${context.nodeId} NRFreqRelation=(NRFreqRelationId==${nrFreqRel.nrFreqRelationId})`;
    const params: string[] = [];

    if (nrFreqRel.referenceFreq !== undefined) params.push(`referenceFreq=${nrFreqRel.referenceFreq}`);
    if (nrFreqRel.relatedFreq !== undefined) params.push(`relatedFreq=${nrFreqRel.relatedFreq}`);

    if (nrFreqRel.nrFreqRelationToEUTRAN) {
      const nrRel = nrFreqRel.nrFreqRelationToEUTRAN;
      if (nrRel.qOffsetCell) params.push(`qOffsetCell=${nrRel.qOffsetCell}`);
      if (nrRel.scgFailureInfoNR !== undefined) params.push(`scgFailureInfoNR=${nrRel.scgFailureInfoNR}`);
      if (nrRel.eutraNrSameFreqInd !== undefined) params.push(`eutraNrSameFreqInd=${nrRel.eutraNrSameFreqInd}`);
    }

    if (params.length > 0) {
      configCommand += ` ${params.join(',')}`;
      commands.push({
        id: `nr_freq_config_${nrFreqRel.nrFreqRelationId}_${counter.value++}`,
        type: 'SET',
        command: configCommand,
        description: `Configure NR frequency relation ${nrFreqRel.nrFreqRelationId}`,
        timeout: 30,
        sourceTemplate: 'NRFreqRelation'
      });
    }

    return commands;
  }

  private generateRollbackCommands(commands: CliCommand[]): CliCommand[] {
    return commands.filter(cmd => cmd.type === 'SET' || cmd.type === 'CREATE').map(cmd => ({
      ...cmd,
      id: `rollback_${cmd.id}`,
      type: cmd.type === 'CREATE' ? 'DELETE' : 'SET',
      command: this.generateRollbackCommand(cmd),
      description: `Rollback: ${cmd.description}`,
      critical: false
    }));
  }

  private generateRollbackCommand(originalCommand: CliCommand): string {
    if (originalCommand.type === 'CREATE') {
      // Extract MO class and create delete command
      const match = originalCommand.command.match(/cmedit create \w+ (\w+).*?(\w+)=/);
      if (match) {
        const moClass = match[1];
        const moId = match[2];
        return `cmedit delete ${originalCommand.command.split(' ')[2]} ${moClass}(${moId}=${originalCommand.command.split('=')[1].split(',')[0]})`;
      }
    } else if (originalCommand.type === 'SET') {
      // Convert SET to rollback values (simplified)
      return originalCommand.command.replace(/=\w+/g, '=default_value');
    }
    return originalCommand.command;
  }

  private generateValidationCommands(commands: CliCommand[]): CliCommand[] {
    return commands.filter(cmd => cmd.type === 'SET' || cmd.type === 'CREATE').map(cmd => ({
      ...cmd,
      id: `validate_${cmd.id}`,
      type: 'GET',
      command: this.generateValidationCommand(cmd),
      description: `Validate: ${cmd.description}`,
      critical: false,
      expectedOutput: ['Set successful', 'Create successful'],
      errorPatterns: ['Error', 'Failed']
    }));
  }

  private generateValidationCommand(originalCommand: CliCommand): string {
    if (originalCommand.type === 'SET') {
      // Convert SET to GET validation
      const parts = originalCommand.command.split(' ');
      return `cmedit get ${parts[2]} ${parts[3].split('=')[0]} -s`;
    } else if (originalCommand.type === 'CREATE') {
      // Validate creation
      const parts = originalCommand.command.split(' ');
      return `cmedit get ${parts[2]} -s`;
    }
    return originalCommand.command;
  }
}

// Export for use in other test files
export { MockTemplateToCliConverter };

describe('Template-to-CLI Converter', () => {
  let converter: TemplateToCliConverter;

  beforeEach(() => {
    converter = new MockTemplateToCliConverter();
  });

  describe('Template Conversion', () => {
    it('should convert base template to CLI commands', async () => {
      const commandSet = await converter.convertTemplate(mockBaseTemplate, mockConversionContext);

      expect(commandSet).toBeDefined();
      expect(commandSet.commands.length).toBeGreaterThan(0);
      expect(commandSet.executionOrder).toHaveLength(commandSet.commands.length);
      expect(commandSet.rollbackCommands.length).toBeGreaterThan(0);
      expect(commandSet.validationCommands.length).toBeGreaterThan(0);
      expect(commandSet.metadata.templateId).toBeDefined();
      expect(commandSet.metadata.conversionTime).toBeGreaterThan(0);
      expect(commandSet.metadata.commandCount).toBe(commandSet.commands.length);
    });

    it('should convert urban template with high capacity features', async () => {
      const commandSet = await converter.convertTemplate(mockUrbanTemplate, mockConversionContext);

      expect(commandSet.commands).toHaveLength.greaterThan(0);

      // Check for urban-specific commands
      const urbanCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('massiveMimoEnabled') || cmd.command.includes('caEnabled')
      );
      expect(urbanCommands).toHaveLength.greaterThan(0);

      // Check for high capacity parameters
      const capacityCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('cellCapMaxCellSubCap') || cmd.command.includes('cellSubscriptionCapacity')
      );
      expect(capacityCommands).toHaveLength.greaterThan(0);
    });

    it('should convert mobility template with high-speed optimization', async () => {
      const mobilityContext: ConversionContext = {
        ...mockConversionContext,
        parameters: { average_velocity: 150, handover_success_rate: 92.5 }
      };

      const commandSet = await converter.convertTemplate(mockMobilityTemplate, mobilityContext);

      expect(commandSet.commands).toHaveLength.greaterThan(0);

      // Check for ANR function configuration
      const anrCommands = commandSet.commands.filter(cmd =>
        cmd.sourceTemplate === 'AnrFunction'
      );
      expect(anrCommands).toHaveLength.greaterThan(0);

      // Check for frequency relation with mobility optimization
      const freqRelCommands = commandSet.commands.filter(cmd =>
        cmd.sourceTemplate === 'EUtranFreqRelation' &&
        (cmd.command.includes('hysteresis=') || cmd.command.includes('timeToTrigger='))
      );
      expect(freqRelCommands).toHaveLength.greaterThan(0);
    });

    it('should convert frequency relation template with EN-DC configuration', async () => {
      const commandSet = await converter.convertTemplate(mockFrequencyRelationTemplate, mockConversionContext);

      expect(commandSet.commands).toHaveLength.greaterThan(0);

      // Check for EN-DC enabling commands
      const endcCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('endcEnabled=true') || cmd.command.includes('splitBearerSupport=true')
      );
      expect(endcCommands).toHaveLength.greaterThan(0);

      // Check for NR frequency relation commands
      const nrFreqCommands = commandSet.commands.filter(cmd =>
        cmd.sourceTemplate === 'NRFreqRelation'
      );
      expect(nrFreqCommands).toHaveLength.greaterThan(0);

      // Should have both CREATE and SET commands for NR frequency relations
      const createNrCommands = nrFreqCommands.filter(cmd => cmd.type === 'CREATE');
      const setNrCommands = nrFreqCommands.filter(cmd => cmd.type === 'SET');
      expect(createNrCommands).toHaveLength(1);
      expect(setNrCommands).toHaveLength(1);
    });

    it('should handle conditional logic based on context parameters', async () => {
      const highDensityContext: ConversionContext = {
        ...mockConversionContext,
        parameters: { user_density: 800, site_type: 'urban' }
      };

      const commandSet = await converter.convertTemplate(mockUrbanTemplate, highDensityContext);

      // Should include high capacity parameters for high density
      const highCapacityCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('cellCapMaxCellSubCap=60000') || cmd.command.includes('cellSubscriptionCapacity=40000')
      );
      expect(highCapacityCommands).toHaveLength.greaterThan(0);

      // Should include advanced features for urban sites
      const advancedFeatureCommands = commandSet.commands.filter(cmd =>
        cmd.command.includes('massiveMimoEnabled=1') || cmd.command.includes('caEnabled=1')
      );
      expect(advancedFeatureCommands).toHaveLength.greaterThan(0);
    });

    it('should apply performance optimizations when enabled', async () => {
      const optimizedContext: ConversionContext = {
        ...mockConversionContext,
        options: { optimizeForPerformance: true, includeRollback: true }
      };

      const commandSet = await converter.convertTemplate(mockBaseTemplate, optimizedContext);

      // Check that commands have optimized descriptions
      const optimizedCommands = commandSet.commands.filter(cmd =>
        cmd.description.includes('Optimized:')
      );
      expect(optimizedCommands).toHaveLength(commandSet.commands.length);

      // Check that timeouts are capped
      const timeouts = commandSet.commands.map(cmd => cmd.timeout);
      expect(timeouts.every(timeout => timeout <= 60)).toBe(true);
    });
  });

  describe('Template Validation', () => {
    it('should validate template structure successfully', () => {
      const results = converter.validateTemplate(mockBaseTemplate);

      expect(results).toBeDefined();
      expect(results.every(r => r.severity !== 'error')).toBe(true);
    });

    it('should detect missing $meta section', () => {
      const invalidTemplate = { EUtranCellFDD: [{}] } as RTBTemplate;

      const results = converter.validateTemplate(invalidTemplate);

      expect(results).toHaveLength.greaterThan(0);
      expect(results.some(r => r.message.includes('missing $meta'))).toBe(true);
      expect(results.some(r => r.severity === 'error')).toBe(true);
    });

    it('should detect missing version in $meta', () => {
      const templateWithoutVersion: RTBTemplate = {
        $meta: { author: ['test'], description: 'test', tags: ['test'] },
        EUtranCellFDD: [{}]
      };

      const results = converter.validateTemplate(templateWithoutVersion);

      expect(results).toHaveLength.greaterThan(0);
      expect(results.some(r => r.message.includes('missing version'))).toBe(true);
      expect(results.some(r => r.severity === 'warning')).toBe(true);
    });

    it('should detect invalid custom functions', () => {
      const templateWithBadFunction: RTBTemplate = {
        $meta: { version: '1.0', author: ['test'], description: 'test', tags: ['test'] },
        $custom: [
          { name: '', args: [], body: [] }, // Missing name
          { name: 'bad_func', args: [], body: [] } // Missing body
        ],
        EUtranCellFDD: [{}]
      };

      const results = converter.validateTemplate(templateWithBadFunction);

      expect(results).toHaveLength(2);
      expect(results.every(r => r.severity === 'error')).toBe(true);
      expect(results.some(r => r.message.includes('missing name'))).toBe(true);
      expect(results.some(r => r.message.includes('missing body'))).toBe(true);
    });
  });

  describe('Command Optimization', () => {
    it('should optimize command timeouts', () => {
      const unoptimizedCommands: CliCommand[] = [
        {
          id: 'test1',
          type: 'SET',
          command: 'cmedit set TEST EUtranCellFDD=1 qRxLevMin=-130',
          description: 'Test command',
          timeout: 120, // Will be capped at 60
          sourceTemplate: 'Test'
        },
        {
          id: 'test2',
          type: 'SET',
          command: 'cmedit set TEST EUtranCellFDD=1 qQualMin=-32',
          description: 'Another test command',
          timeout: 30, // Will remain unchanged
          sourceTemplate: 'Test'
        }
      ];

      const optimizedCommands = converter.optimizeCommands(unoptimizedCommands);

      expect(optimizedCommands).toHaveLength(2);
      expect(optimizedCommands[0].timeout).toBe(60); // Capped
      expect(optimizedCommands[1].timeout).toBe(30); // Unchanged
      expect(optimizedCommands.every(cmd => cmd.description.includes('Optimized:'))).toBe(true);
    });
  });

  describe('Performance Validation', () => {
    it('should convert templates within performance target (<2 seconds)', async () => {
      const startTime = performance.now();

      const commandSet = await converter.convertTemplate(mockBaseTemplate, mockConversionContext);

      const conversionTime = performance.now() - startTime;

      expect(commandSet).toBeDefined();
      expect(commandSet.commands).toHaveLength.greaterThan(0);
      expect(conversionTime).toBeLessThan(2000); // <2 second target
      expect(commandSet.metadata.conversionTime).toBeLessThan(2000);
    });

    it('should handle complex templates efficiently', async () => {
      const complexTemplate: RTBTemplate = {
        ...mockBaseTemplate,
        ...mockUrbanTemplate,
        ...mockMobilityTemplate,
        ...mockFrequencyRelationTemplate
      };

      const startTime = performance.now();

      const commandSet = await converter.convertTemplate(complexTemplate, mockConversionContext);

      const conversionTime = performance.now() - startTime;

      expect(commandSet.commands).toHaveLength.greaterThan(10); // Should generate many commands
      expect(conversionTime).toBeLessThan(2000); // Still within 2 second target
    });
  });

  describe('Command Syntax Validation', () => {
    it('should generate syntactically correct cmedit commands', async () => {
      const commandSet = await converter.convertTemplate(mockBaseTemplate, mockConversionContext);

      commandSet.commands.forEach(command => {
        expect(command.command).toMatch(/^cmedit /);
        expect(command.type).toEqual(expect.arrayContaining(['GET', 'SET', 'CREATE', 'DELETE', 'MONITOR']));
        expect(command.description).toBeDefined();
        expect(command.timeout).toBeGreaterThan(0);
        expect(command.sourceTemplate).toBeDefined();
      });
    });

    it('should use correct MO class names', async () => {
      const commandSet = await converter.convertTemplate(mockBaseTemplate, mockConversionContext);

      const commandTexts = commandSet.commands.map(cmd => cmd.command).join(' ');

      // Check for valid Ericsson MO classes
      expect(commandTexts).toMatch(/ManagedElement/);
      expect(commandTexts).toMatch(/ENBFunction/);
      expect(commandTexts).toMatch(/EUtranCellFDD/);
    });

    it('should use correct parameter names', async () => {
      const commandSet = await converter.convertTemplate(mockBaseTemplate, mockConversionContext);

      const commandTexts = commandSet.commands.map(cmd => cmd.command).join(' ');

      // Check for valid Ericsson parameter names
      expect(commandTexts).toMatch(/managedElementId/);
      expect(commandTexts).toMatch(/userLabel/);
      expect(commandTexts).toMatch(/eNodeBId/);
      expect(commandTexts).toMatch(/maxConnectedUe/);
      expect(commandTexts).toMatch(/qRxLevMin/);
      expect(commandTexts).toMatch(/qQualMin/);
      expect(commandTexts).toMatch(/administrativestate/);
    });

    it('should generate proper rollback commands', async () => {
      const contextWithRollback: ConversionContext = {
        ...mockConversionContext,
        options: { includeRollback: true }
      };

      const commandSet = await converter.convertTemplate(mockBaseTemplate, contextWithRollback);

      expect(commandSet.rollbackCommands).toHaveLength.greaterThan(0);

      // Rollback commands should have proper structure
      commandSet.rollbackCommands.forEach(rollbackCmd => {
        expect(rollbackCmd.command).toMatch(/^cmedit /);
        expect(rollbackCmd.type).toEqual(expect.arrayContaining(['DELETE', 'SET']));
        expect(rollbackCmd.description).toMatch(/^Rollback:/);
        expect(rollbackCmd.critical).toBe(false);
      });
    });

    it('should generate proper validation commands', async () => {
      const commandSet = await converter.convertTemplate(mockBaseTemplate, mockConversionContext);

      expect(commandSet.validationCommands).toHaveLength.greaterThan(0);

      // Validation commands should have proper structure
      commandSet.validationCommands.forEach(validationCmd => {
        expect(validationCmd.command).toMatch(/^cmedit /);
        expect(validationCmd.type).toBe('GET');
        expect(validationCmd.description).toMatch(/^Validate:/);
        expect(validationCmd.expectedOutput).toBeDefined();
        expect(validationCmd.errorPatterns).toBeDefined();
        expect(validationCmd.critical).toBe(false);
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle empty templates gracefully', async () => {
      const emptyTemplate: RTBTemplate = {
        $meta: { version: '1.0', author: ['test'], description: 'empty', tags: ['test'] }
      };

      const commandSet = await converter.convertTemplate(emptyTemplate, mockConversionContext);

      expect(commandSet).toBeDefined();
      expect(commandSet.commands).toHaveLength(0);
      expect(commandSet.metadata.commandCount).toBe(0);
    });

    it('should handle malformed custom functions', async () => {
      const templateWithBadCustom: RTBTemplate = {
        $meta: { version: '1.0', author: ['test'], description: 'test', tags: ['test'] },
        $custom: [
          {
            name: 'bad_function',
            args: ['param1'],
            body: ['invalid python syntax']
          }
        ],
        EUtranCellFDD: [{}]
      };

      // Should not crash during conversion
      const commandSet = await converter.convertTemplate(templateWithBadCustom, mockConversionContext);

      expect(commandSet).toBeDefined();
      // Commands should still be generated despite bad custom function
      expect(commandSet.commands).toHaveLength.greaterThan(0);
    });

    it('should handle invalid MO references gracefully', async () => {
      const templateWithInvalidMO: RTBTemplate = {
        $meta: { version: '1.0', author: ['test'], description: 'test', tags: ['test'] },
        InvalidMOClass: {
          invalidParameter: 'invalid_value'
        }
      };

      const commandSet = await converter.convertTemplate(templateWithInvalidMO, mockConversionContext);

      expect(commandSet).toBeDefined();
      // Should not crash, but may not generate commands for invalid MO
    });
  });
});