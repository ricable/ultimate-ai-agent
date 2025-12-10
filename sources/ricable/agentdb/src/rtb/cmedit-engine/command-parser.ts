/**
 * cmedit Command Syntax Parser
 *
 * Provides comprehensive parsing of cmedit command syntax with tokenization,
 * validation, and intelligent command structure analysis.
 */

import {
  CmeditCommand,
  CmeditCommandType,
  CmeditCommandOptions,
  CommandContext,
  CommandValidation,
  SyntaxValidation,
  CommandStructure,
  ValidationError,
  FDNPath,
  FDNComponent,
  FDNValidation,
  CardinalityInfo,
  PathComplexity
} from './types';
import { RTBTemplate, TemplateMeta } from '../../types/rtb-types';

export class CmeditCommandParser {
  private readonly COMMAND_PATTERNS: Map<CmeditCommandType, RegExp> = new Map([
    ['get', /^get\s+(.+?)(?:\s+(?:--attribute|--attr)\s+(.+?))?(?:\s+(-[a-z]+))?\s*$/i],
    ['set', /^set\s+(.+?)\s+(.+?)(?:\s+(--?[a-z-]+(?:\s+[^-]+)?))*\s*$/i],
    ['create', /^create\s+(.+?)\s+(.+?)(?:\s+(--?[a-z-]+(?:\s+[^-]+)?))*\s*$/i],
    ['delete', /^delete\s+(.+?)(?:\s+(--?[a-z-]+(?:\s+[^-]+)?))*\s*$/i],
    ['mon', /^mon\s+(.+?)(?:\s+(--?[a-z-]+(?:\s+[^-]+)?))*\s*$/i],
    ['unmon', /^unmon\s+(.+?)(?:\s+(--?[a-z-]+(?:\s+[^-]+)?))*\s*$/i],
    ['preview', /^.*\s+--preview\s*$/i]
  ]);

  private readonly OPTION_PATTERNS: Map<string, RegExp> = new Map([
    ['preview', /--preview/],
    ['force', /--force/],
    ['table', /-t|--table/],
    ['detailed', /-d|--detailed/],
    ['collection', /--collection\s+(\w+)/],
    ['scopeFilter', /--scopefilter\s+\(([^)]+)\)/],
    ['attributes', /--attribute|--attr\s+([^-]+)/],
    ['all', /--all/],
    ['dryRun', /--dry-run/],
    ['recursive', /-r|--recursive/]
  ]);

  private readonly FDN_COMPONENT_PATTERN = /([^=,\(\)]+)(?:=([^=,\(\)]+))?(?:\(([^)]+)\))?/g;
  private readonly WILDCARD_PATTERN = /[*?]/;
  private readonly ATTRIBUTE_PATTERN = /([^.]+(?:\.[^.]+)*)(?:\.\[(\d+)\])?/;

  constructor(private readonly moHierarchy: any) {}

  /**
   * Parse a cmedit command string into structured command object
   */
  parseCommand(commandString: string, context?: Partial<CommandContext>): CmeditCommand {
    const trimmedCommand = commandString.trim();
    if (!trimmedCommand) {
      throw new Error('Empty command string');
    }

    const commandType = this.detectCommandType(trimmedCommand);
    if (!commandType) {
      throw new Error(`Unable to detect command type: ${trimmedCommand}`);
    }

    const parseResult = this.parseCommandStructure(trimmedCommand, commandType);
    const options = this.parseCommandOptions(trimmedCommand);
    const fdnPath = this.parseFDNPath(parseResult.target);

    const command: CmeditCommand = {
      type: commandType,
      target: parseResult.target,
      parameters: parseResult.parameters,
      options,
      command: trimmedCommand,
      context: this.buildCommandContext(context, commandType, fdnPath),
      validation: this.validateCommandSyntax(trimmedCommand, commandType)
    };

    return command;
  }

  /**
   * Parse multiple commands from batch input
   */
  parseBatchCommands(commandStrings: string[], context?: Partial<CommandContext>): CmeditCommand[] {
    const commands: CmeditCommand[] = [];
    const errors: string[] = [];

    for (let i = 0; i < commandStrings.length; i++) {
      const commandString = commandStrings[i].trim();
      if (!commandString || commandString.startsWith('#')) {
        continue; // Skip empty lines and comments
      }

      try {
        const command = this.parseCommand(commandString, {
          ...context,
          priority: context.priority || 'medium'
        });
        commands.push(command);
      } catch (error) {
        errors.push(`Line ${i + 1}: ${error instanceof Error ? error.message : String(error)}`);
      }
    }

    if (errors.length > 0) {
      throw new Error(`Batch parsing errors:\n${errors.join('\n')}`);
    }

    return commands;
  }

  /**
   * Generate cmedit command from template
   */
  generateFromTemplate(
    template: RTBTemplate,
    commandType: CmeditCommandType,
    targetFDN: string,
    options?: CmeditCommandOptions
  ): CmeditCommand {
    const parameters = this.extractParametersFromTemplate(template);
    const commandString = this.buildCommandString(commandType, targetFDN, parameters, options);

    return this.parseCommand(commandString, {
      sourceTemplate: template.meta?.version || 'unknown',
      purpose: this.inferPurposeFromTemplate(template),
      cognitiveLevel: 'enhanced'
    });
  }

  /**
   * Detect command type from command string
   */
  private detectCommandType(command: string): CmeditCommandType | null {
    for (const [type, pattern] of this.COMMAND_PATTERNS) {
      if (pattern.test(command)) {
        return type;
      }
    }

    // Check for preview commands
    if (command.includes('--preview')) {
      const baseCommand = command.replace(/\s+--preview.*$/, '').trim();
      return this.detectCommandType(baseCommand);
    }

    return null;
  }

  /**
   * Parse command structure based on command type
   */
  private parseCommandStructure(command: string, type: CmeditCommandType): {
    target: string;
    parameters?: Record<string, any>;
  } {
    const pattern = this.COMMAND_PATTERNS.get(type);
    if (!pattern) {
      throw new Error(`No pattern found for command type: ${type}`);
    }

    const match = command.match(pattern);
    if (!match) {
      throw new Error(`Invalid command structure for ${type}: ${command}`);
    }

    switch (type) {
      case 'get':
        return {
          target: match[1]?.trim() || '',
          parameters: match[2] ? { attributes: match[2].split(',').map(a => a.trim()) } : undefined
        };

      case 'set':
        return this.parseSetCommand(command);

      case 'create':
        return this.parseCreateCommand(command);

      case 'delete':
        return {
          target: match[1]?.trim() || ''
        };

      case 'mon':
      case 'unmon':
        return {
          target: match[1]?.trim() || ''
        };

      default:
        return {
          target: match[1]?.trim() || ''
        };
    }
  }

  /**
   * Parse set command with parameters
   */
  private parseSetCommand(command: string): { target: string; parameters: Record<string, any> } {
    const parts = command.split(/\s+/);
    if (parts.length < 3) {
      throw new Error(`Invalid set command structure: ${command}`);
    }

    const target = parts[1];
    const parameters: Record<string, any> = {};

    // Handle parameter assignments
    let currentParam = '';
    for (let i = 2; i < parts.length; i++) {
      const part = parts[i];

      if (part.startsWith('--')) {
        // Skip options
        continue;
      } else if (part.includes('=')) {
        // Parameter with value
        const [param, value] = part.split('=', 2);
        parameters[param] = this.parseParameterValue(value);
        currentParam = '';
      } else if (currentParam) {
        // Value for previous parameter
        parameters[currentParam] = this.parseParameterValue(part);
        currentParam = '';
      } else {
        // Parameter expecting value
        currentParam = part;
      }
    }

    return { target, parameters };
  }

  /**
   * Parse create command with MO specifications
   */
  private parseCreateCommand(command: string): { target: string; parameters: Record<string, any> } {
    const parts = command.split(/\s+/);
    if (parts.length < 3) {
      throw new Error(`Invalid create command structure: ${command}`);
    }

    const target = parts[1];
    const moSpec = parts[2];
    const parameters: Record<string, any> = {};

    // Parse MO specification
    if (moSpec.includes('=')) {
      const [moClass, moValue] = moSpec.split('=', 2);
      parameters[moClass] = moValue;
    }

    // Parse additional parameters
    for (let i = 3; i < parts.length; i++) {
      const part = parts[i];
      if (part.includes('=')) {
        const [param, value] = part.split('=', 2);
        parameters[param] = this.parseParameterValue(value);
      }
    }

    return { target, parameters };
  }

  /**
   * Parse parameter value with type conversion
   */
  private parseParameterValue(value: string): any {
    // Remove quotes if present
    if ((value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))) {
      return value.slice(1, -1);
    }

    // Try to parse as number
    const numValue = Number(value);
    if (!isNaN(numValue)) {
      return numValue;
    }

    // Try to parse as boolean
    if (value.toLowerCase() === 'true') return true;
    if (value.toLowerCase() === 'false') return false;

    // Return as string
    return value;
  }

  /**
   * Parse command options
   */
  private parseCommandOptions(command: string): CmeditCommandOptions {
    const options: CmeditCommandOptions = {};

    for (const [option, pattern] of this.OPTION_PATTERNS) {
      const match = command.match(pattern);
      if (match) {
        switch (option) {
          case 'preview':
            options.preview = true;
            break;
          case 'force':
            options.force = true;
            break;
          case 'table':
            options.table = true;
            break;
          case 'detailed':
            options.detailed = true;
            break;
          case 'collection':
            options.collection = match[1];
            break;
          case 'scopeFilter':
            options.scopeFilter = match[1];
            break;
          case 'attributes':
            options.attributes = match[1].split(',').map(a => a.trim());
            break;
          case 'all':
            options.recursive = true;
            break;
          case 'dryRun':
            options.dryRun = true;
            break;
          case 'recursive':
            options.recursive = true;
            break;
        }
      }
    }

    return options;
  }

  /**
   * Parse FDN path and extract components
   */
  private parseFDNPath(fdnString: string): FDNPath {
    const components: FDNComponent[] = [];
    const moHierarchy: string[] = [];
    let componentMatch: RegExpExecArray | null;

    // Reset regex lastIndex
    this.FDN_COMPONENT_PATTERN.lastIndex = 0;

    while ((componentMatch = this.FDN_COMPONENT_PATTERN.exec(fdnString)) !== null) {
      const [fullMatch, componentName, componentValue, componentIndex] = componentMatch;

      const component: FDNComponent = {
        name: componentName,
        moClass: this.identifyMOClass(componentName),
        value: componentValue || componentName,
        type: this.determineComponentType(componentName, componentValue),
        optional: !componentValue,
        cardinality: this.analyzeCardinality(componentName, componentIndex)
      };

      components.push(component);
      moHierarchy.push(component.moClass);
    }

    const validation = this.validateFDNPath(fdnString, components);
    const complexity = this.calculatePathComplexity(components);

    return {
      path: fdnString,
      components,
      moHierarchy,
      validation,
      alternatives: this.generateAlternativePaths(components),
      complexity
    };
  }

  /**
   * Identify MO class from component name
   */
  private identifyMOClass(componentName: string): string {
    // Common MO class mappings
    const moClassMappings: Record<string, string> = {
      'MeContext': 'MeContext',
      'ManagedElement': 'ManagedElement',
      'ENodeBFunction': 'ENodeBFunction',
      'EUtranCellFDD': 'EUtranCellFDD',
      'EUtranCellTDD': 'EUtranCellTDD',
      'EUtranFrequencyRelation': 'EUtranFrequencyRelation',
      'EUtranCellRelation': 'EUtranCellRelation',
      'ExternalEUtranCellFDD': 'ExternalEUtranCellFDD',
      'ExternalEUtranCellTDD': 'ExternalEUtranCellTDD',
      'FeatureState': 'FeatureState',
      'OptionalFeatureLicense': 'OptionalFeatureLicense',
      'CmFunction': 'CmFunction',
      'BscFunction': 'BscFunction',
      'GeranCell': 'GeranCell',
      'UtranFreqRelation': 'UtranFreqRelation',
      'ExternalEUtranCellFDD': 'ExternalEUtranCellFDD'
    };

    return moClassMappings[componentName] || componentName;
  }

  /**
   * Determine component type
   */
  private determineComponentType(name: string, value?: string): FDNComponent['type'] {
    if (this.WILDCARD_PATTERN.test(name) || this.WILDCARD_PATTERN.test(value || '')) {
      return 'wildcard';
    }
    if (value && value.includes('=')) {
      return 'index';
    }
    if (name.includes('.')) {
      return 'attribute';
    }
    return 'class';
  }

  /**
   * Analyze component cardinality
   */
  private analyzeCardinality(componentName: string, index?: string): CardinalityInfo {
    // Default cardinality - would be enhanced with actual MO hierarchy data
    const baseCardinality: CardinalityInfo = {
      minimum: 1,
      maximum: 1,
      current: 1,
      type: 'single'
    };

    if (index) {
      const indexNum = parseInt(index);
      if (!isNaN(indexNum)) {
        baseCardinality.current = indexNum;
        baseCardinality.maximum = Math.max(baseCardinality.maximum, indexNum);
        baseCardinality.type = indexNum > 1 ? 'multiple' : 'single';
      }
    }

    return baseCardinality;
  }

  /**
   * Validate FDN path
   */
  private validateFDNPath(path: string, components: FDNComponent[]): FDNValidation {
    const errors: ValidationError[] = [];
    const warnings: ValidationError[] = [];

    // Check for empty path
    if (!path || components.length === 0) {
      errors.push({
        message: 'Empty FDN path',
        component: 'root',
        type: 'syntax',
        severity: 'error'
      });
    }

    // Check component sequence
    for (let i = 0; i < components.length; i++) {
      const component = components[i];

      // Validate component format
      if (!component.name || component.name.length === 0) {
        errors.push({
          message: `Invalid component format at position ${i}`,
          component: component.name,
          type: 'syntax',
          severity: 'error'
        });
      }

      // Check for required components
      if (i === 0 && !['MeContext', 'ManagedElement'].includes(component.moClass)) {
        warnings.push({
          message: 'FDN should typically start with MeContext or ManagedElement',
          component: component.name,
          type: 'semantic',
          severity: 'warning'
        });
      }
    }

    // Check for wildcards in inappropriate positions
    const wildcardComponents = components.filter(c => c.type === 'wildcard');
    if (wildcardComponents.length > 1) {
      warnings.push({
        message: 'Multiple wildcards may impact performance',
        component: 'wildcard',
        type: 'semantic',
        severity: 'warning'
      });
    }

    const isValid = errors.length === 0;
    const complianceLevel = isValid ? (warnings.length === 0 ? 'full' : 'partial') : 'minimal';

    return {
      isValid,
      errors,
      warnings,
      complianceLevel,
      ldnPatternMatch: this.matchLDNPattern(components)
    };
  }

  /**
   * Match against LDN patterns (simplified implementation)
   */
  private matchLDNPattern(components: FDNComponent[]): string | undefined {
    // This would be enhanced with actual LDN pattern matching
    const pathString = components.map(c => c.moClass).join('/');
    return `LDN_${pathString.replace(/\//g, '_')}`;
  }

  /**
   * Calculate path complexity
   */
  private calculatePathComplexity(components: FDNComponent[]): PathComplexity {
    const depth = components.length;
    const wildcardCount = components.filter(c => c.type === 'wildcard').length;

    // Base complexity score
    let score = depth * 10;

    // Add complexity for wildcards
    score += wildcardCount * 20;

    // Add complexity for indexed components
    const indexedComponents = components.filter(c => c.cardinality.current > 1);
    score += indexedComponents.length * 5;

    // Estimated execution time (very rough approximation)
    const estimatedTime = score * 10; // milliseconds

    let difficulty: PathComplexity['difficulty'] = 'simple';
    if (score > 80) difficulty = 'very_complex';
    else if (score > 60) difficulty = 'complex';
    else if (score > 40) difficulty = 'moderate';
    else if (score > 20) difficulty = 'simple';

    return {
      score: Math.min(100, score),
      depth,
      componentCount: components.length,
      wildcardCount,
      estimatedTime,
      difficulty
    };
  }

  /**
   * Generate alternative paths
   */
  private generateAlternativePaths(components: FDNComponent[]): string[] {
    const alternatives: string[] = [];

    // Generate wildcard variations
    if (components.length > 1) {
      const wildcardPath = components.map((c, i) => {
        if (i === components.length - 1) {
          return c.type === 'wildcard' ? c.name : '*';
        }
        return c.name;
      }).join(',');
      alternatives.push(wildcardPath);
    }

    // Generate shortened paths
    if (components.length > 2) {
      const shortenedPath = components.slice(-2).map(c => c.name).join(',');
      alternatives.push(shortenedPath);
    }

    return alternatives;
  }

  /**
   * Validate command syntax
   */
  private validateCommandSyntax(command: string, type: CmeditCommandType): CommandValidation {
    const syntaxValidation: SyntaxValidation = {
      isCorrect: false,
      errors: [],
      structure: {
        parts: command.split(/\s+/),
        argCount: command.split(/\s+/).length,
        expectedPattern: this.getExpectedPattern(type),
        actualPattern: this.getActualPattern(command)
      }
    };

    const pattern = this.COMMAND_PATTERNS.get(type);
    syntaxValidation.isCorrect = pattern ? pattern.test(command) : false;

    if (!syntaxValidation.isCorrect) {
      syntaxValidation.errors.push(`Invalid syntax for ${type} command`);
    }

    return {
      isValid: syntaxValidation.isCorrect,
      syntax: syntaxValidation,
      semantic: {
        isCorrect: true,
        moClasses: [],
        operation: {
          isSupported: true,
          constraints: [],
          permissions: []
        }
      },
      parameters: {
        isValid: true,
        errors: [],
        warnings: [],
        conversions: []
      },
      dependencies: {
        isSatisfied: true,
        unresolved: [],
        circular: [],
        graph: {
          nodes: [],
          edges: [],
          components: [],
          hasCycles: false
        }
      },
      score: syntaxValidation.isCorrect ? 100 : 0,
      recommendations: syntaxValidation.isCorrect ? [] : ['Check command syntax and structure']
    };
  }

  /**
   * Get expected pattern for command type
   */
  private getExpectedPattern(type: CmeditCommandType): string {
    const patterns: Record<CmeditCommandType, string> = {
      'get': 'get <fdn_path> [options]',
      'set': 'set <fdn_path> <parameters> [options]',
      'create': 'create <fdn_path> <mo_spec> [options]',
      'delete': 'delete <fdn_path> [options]',
      'mon': 'mon <fdn_path> [options]',
      'unmon': 'unmon <fdn_path> [options]',
      'preview': '<command> --preview'
    };
    return patterns[type] || 'unknown';
  }

  /**
   * Get actual pattern from command
   */
  private getActualPattern(command: string): string {
    const parts = command.split(/\s+/);
    return parts.map((part, i) => {
      if (part.startsWith('--')) return '[option]';
      if (i === 0) return part;
      if (part.includes('=')) return '[param=value]';
      if (part.includes(',')) return '[fdn_component]';
      return '[component]';
    }).join(' ');
  }

  /**
   * Build command context
   */
  private buildCommandContext(
    partialContext: Partial<CommandContext> | undefined,
    commandType: CmeditCommandType,
    fdnPath: FDNPath
  ): CommandContext {
    return {
      sourceTemplate: partialContext?.sourceTemplate,
      moClasses: fdnPath.moHierarchy,
      purpose: partialContext?.purpose || 'configuration_management',
      networkContext: partialContext?.networkContext || {
        technology: '4G',
        environment: 'urban_medium',
        vendor: {
          primary: 'ericsson',
          multiVendor: false,
          compatibilityMode: false
        },
        topology: {
          cellCount: 1,
          siteCount: 1,
          frequencyBands: [],
          carrierAggregation: false,
          networkSharing: false
        }
      },
      cognitiveLevel: partialContext?.cognitiveLevel || 'basic',
      expertisePatterns: [],
      generatedAt: new Date(),
      priority: partialContext?.priority || 'medium'
    };
  }

  /**
   * Extract parameters from template
   */
  private extractParametersFromTemplate(template: RTBTemplate): Record<string, any> {
    return template.configuration || {};
  }

  /**
   * Infer purpose from template
   */
  private inferPurposeFromTemplate(template: RTBTemplate): any {
    const tags = template.meta?.tags || [];
    const description = template.meta?.description || '';

    if (tags.includes('optimization') || description.includes('optimize')) {
      return 'cell_optimization';
    }
    if (tags.includes('mobility') || description.includes('handover')) {
      return 'mobility_management';
    }
    if (tags.includes('capacity') || description.includes('capacity')) {
      return 'capacity_expansion';
    }
    if (tags.includes('feature') || description.includes('feature')) {
      return 'feature_activation';
    }

    return 'configuration_management';
  }

  /**
   * Build command string from components
   */
  private buildCommandString(
    type: CmeditCommandType,
    target: string,
    parameters: Record<string, any>,
    options?: CmeditCommandOptions
  ): string {
    let command = `${type} ${target}`;

    // Add parameters for set commands
    if (type === 'set' && Object.keys(parameters).length > 0) {
      const paramStr = Object.entries(parameters)
        .map(([key, value]) => `${key}=${value}`)
        .join(',');
      command += ` ${paramStr}`;
    }

    // Add options
    if (options) {
      if (options.preview) command += ' --preview';
      if (options.force) command += ' --force';
      if (options.table) command += ' -t';
      if (options.detailed) command += ' -d';
      if (options.collection) command += ` --collection ${options.collection}`;
      if (options.scopeFilter) command += ` --scopefilter (${options.scopeFilter})`;
      if (options.attributes && options.attributes.length > 0) {
        command += ` --attribute ${options.attributes.join(',')}`;
      }
      if (options.dryRun) command += ' --dry-run';
      if (options.recursive) command += ' -r';
    }

    return command;
  }
}