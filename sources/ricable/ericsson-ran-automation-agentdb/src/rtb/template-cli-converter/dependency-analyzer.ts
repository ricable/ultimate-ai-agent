/**
 * Dependency Analyzer
 *
 * Analyzes command dependencies, creates execution graphs,
 * identifies critical paths, and optimizes execution order.
 */

import {
  GeneratedCliCommand,
  CliCommandType,
  DependencyAnalysisResult,
  DependencyGraph,
  DependencyNode,
  DependencyEdge,
  DependencyType,
  ExecutionLevel,
  CircularDependency,
  DependencyOptimization,
  TemplateToCliContext
} from './types';

/**
 * Dependency Analyzer Configuration
 */
export interface DependencyAnalyzerConfig {
  /** Enable circular dependency detection */
  enableCircularDependencyDetection: boolean;
  /** Enable critical path analysis */
  enableCriticalPathAnalysis: boolean;
  /** Enable optimization suggestions */
  enableOptimizationSuggestions: boolean;
  /** Maximum dependency depth */
  maxDependencyDepth: number;
  /** Dependency resolution strategy */
  resolutionStrategy: 'conservative' | 'balanced' | 'aggressive';
}

/**
 * Dependency rule
 */
interface DependencyRule {
  name: string;
  description: string;
  type: DependencyType;
  strength: 'weak' | 'medium' | 'strong';
  condition: (source: GeneratedCliCommand, target: GeneratedCliCommand, context: TemplateToCliContext) => boolean;
}

/**
 * Dependency Analyzer Class
 */
export class DependencyAnalyzer {
  private config: DependencyAnalyzerConfig;
  private dependencyRules: DependencyRule[] = [];
  private dependencyCache: Map<string, DependencyAnalysisResult> = new Map();

  constructor(config: DependencyAnalyzerConfig) {
    this.config = {
      enableCircularDependencyDetection: true,
      enableCriticalPathAnalysis: true,
      enableOptimizationSuggestions: true,
      maxDependencyDepth: 10,
      resolutionStrategy: 'balanced',
      ...config
    };

    this.initializeDependencyRules();
  }

  /**
   * Analyze command dependencies
   */
  public async analyze(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): Promise<DependencyAnalysisResult> {
    const startTime = Date.now();
    console.log(`Analyzing dependencies for ${commands.length} commands...`);

    // Check cache first
    const cacheKey = this.generateCacheKey(commands, context);
    if (this.dependencyCache.has(cacheKey)) {
      console.log('Using cached dependency analysis result');
      return this.dependencyCache.get(cacheKey)!;
    }

    try {
      // Phase 1: Create dependency graph
      const dependencyGraph = this.createDependencyGraph(commands, context);

      // Phase 2: Detect circular dependencies
      const circularDependencies = this.config.enableCircularDependencyDetection
        ? this.detectCircularDependencies(dependencyGraph)
        : [];

      // Phase 3: Determine critical path
      const criticalPath = this.config.enableCriticalPathAnalysis
        ? this.calculateCriticalPath(dependencyGraph)
        : commands.map(cmd => cmd.id);

      // Phase 4: Create execution levels
      const executionLevels = this.createExecutionLevels(dependencyGraph, criticalPath);

      // Phase 5: Generate optimization suggestions
      const optimizations = this.config.enableOptimizationSuggestions
        ? this.generateOptimizations(dependencyGraph, executionLevels, context)
        : [];

      // Phase 6: Create result
      const result: DependencyAnalysisResult = {
        dependencyGraph,
        criticalPath,
        executionLevels,
        circularDependencies,
        optimizations
      };

      // Cache result
      this.dependencyCache.set(cacheKey, result);

      const duration = Date.now() - startTime;
      console.log(`Dependency analysis completed in ${duration}ms: ${dependencyGraph.edges.length} dependencies found`);

      return result;

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      console.error(`Dependency analysis failed: ${errorMessage}`);
      throw new Error(`Dependency analysis failed: ${errorMessage}`);
    }
  }

  /**
   * Create dependency graph
   */
  private createDependencyGraph(
    commands: GeneratedCliCommand[],
    context: TemplateToCliContext
  ): DependencyGraph {
    const nodes: DependencyNode[] = [];
    const edges: DependencyEdge[] = [];

    // Create nodes
    for (const command of commands) {
      const node: DependencyNode = {
        id: command.id,
        type: command.type,
        critical: command.critical || false,
        estimatedDuration: command.metadata.estimatedDuration,
        riskLevel: command.metadata.riskLevel,
        dependencyCount: 0,
        dependentCount: 0
      };
      nodes.push(node);
    }

    // Create edges based on dependency rules
    for (let i = 0; i < commands.length; i++) {
      const sourceCommand = commands[i];
      for (let j = i + 1; j < commands.length; j++) {
        const targetCommand = commands[j];

        // Apply dependency rules
        for (const rule of this.dependencyRules) {
          if (rule.condition(sourceCommand, targetCommand, context)) {
            const edge: DependencyEdge = {
              from: sourceCommand.id,
              to: targetCommand.id,
              type: rule.type,
              strength: rule.strength,
              description: rule.description
            };
            edges.push(edge);

            // Update dependency counts
            const sourceNode = nodes.find(n => n.id === sourceCommand.id);
            const targetNode = nodes.find(n => n.id === targetCommand.id);
            if (sourceNode && targetNode) {
              sourceNode.dependentCount++;
              targetNode.dependencyCount++;
            }
          }
        }

        // Apply target-to-source dependencies (reverse)
        for (const rule of this.dependencyRules) {
          if (rule.condition(targetCommand, sourceCommand, context)) {
            const edge: DependencyEdge = {
              from: targetCommand.id,
              to: sourceCommand.id,
              type: rule.type,
              strength: rule.strength,
              description: rule.description
            };
            edges.push(edge);

            // Update dependency counts
            const sourceNode = nodes.find(n => n.id === targetCommand.id);
            const targetNode = nodes.find(n => n.id === sourceCommand.id);
            if (sourceNode && targetNode) {
              sourceNode.dependentCount++;
              targetNode.dependencyCount++;
            }
          }
        }
      }
    }

    // Calculate graph metrics
    const metrics = this.calculateGraphMetrics(nodes, edges);

    return {
      nodes,
      edges,
      metrics
    };
  }

  /**
   * Detect circular dependencies
   */
  private detectCircularDependencies(dependencyGraph: DependencyGraph): CircularDependency[] {
    const circularDependencies: CircularDependency[] = [];
    const visited = new Set<string>();
    const recursionStack = new Set<string>();
    const path: string[] = [];

    const dfs = (nodeId: string): boolean => {
      if (recursionStack.has(nodeId)) {
        // Found a cycle
        const cycleStart = path.indexOf(nodeId);
        const cycle = path.slice(cycleStart);
        cycle.push(nodeId);

        circularDependencies.push({
          commands: [...cycle],
          length: cycle.length,
          severity: this.calculateCycleSeverity(cycle.length),
          resolutions: this.generateCycleResolutions(cycle)
        });

        return true;
      }

      if (visited.has(nodeId)) {
        return false;
      }

      visited.add(nodeId);
      recursionStack.add(nodeId);
      path.push(nodeId);

      // Visit dependencies
      const dependencies = dependencyGraph.edges
        .filter(edge => edge.from === nodeId)
        .map(edge => edge.to);

      for (const depId of dependencies) {
        if (dfs(depId)) {
          return true;
        }
      }

      recursionStack.delete(nodeId);
      path.pop();

      return false;
    };

    // Start DFS from each node
    for (const node of dependencyGraph.nodes) {
      if (!visited.has(node.id)) {
        dfs(node.id);
      }
    }

    return circularDependencies;
  }

  /**
   * Calculate critical path
   */
  private calculateCriticalPath(dependencyGraph: DependencyGraph): string[] {
    // Build adjacency list
    const adjacencyList = new Map<string, Array<{ node: string; weight: number }>>();
    const inDegree = new Map<string, number>();

    // Initialize
    for (const node of dependencyGraph.nodes) {
      adjacencyList.set(node.id, []);
      inDegree.set(node.id, 0);
    }

    // Build graph
    for (const edge of dependencyGraph.edges) {
      adjacencyList.get(edge.from)!.push({ node: edge.to, weight: 1 });
      inDegree.set(edge.to, (inDegree.get(edge.to) || 0) + 1);
    }

    // Topological sort with longest path calculation
    const queue: string[] = [];
    const longestPath = new Map<string, { length: number; path: string[] }>();

    // Find nodes with no dependencies
    for (const [nodeId, degree] of inDegree) {
      if (degree === 0) {
        queue.push(nodeId);
        longestPath.set(nodeId, { length: 0, path: [nodeId] });
      }
    }

    // Process nodes
    while (queue.length > 0) {
      const current = queue.shift()!;
      const currentPath = longestPath.get(current)!;

      // Process neighbors
      for (const neighbor of adjacencyList.get(current)!) {
        const newLength = currentPath.length + neighbor.weight;
        const neighborPath = longestPath.get(neighbor);

        if (!neighborPath || newLength > neighborPath.length) {
          longestPath.set(neighbor, {
            length: newLength,
            path: [...currentPath.path, neighbor.node]
          });
        }

        // Update in-degree
        const newInDegree = (inDegree.get(neighbor.node) || 0) - 1;
        inDegree.set(neighbor.node, newInDegree);

        if (newInDegree === 0) {
          queue.push(neighbor.node);
        }
      }
    }

    // Find longest path
    let criticalPath: string[] = [];
    let maxLength = 0;

    for (const [nodeId, pathInfo] of longestPath) {
      if (pathInfo.length > maxLength) {
        maxLength = pathInfo.length;
        criticalPath = pathInfo.path;
      }
    }

    return criticalPath;
  }

  /**
   * Create execution levels
   */
  private createExecutionLevels(
    dependencyGraph: DependencyGraph,
    criticalPath: string[]
  ): ExecutionLevel[] {
    const levels: ExecutionLevel[] = [];
    const processed = new Set<string>();
    const commandToLevel = new Map<string, number>();

    // Build dependency map
    const dependencies = new Map<string, string[]>();
    const dependents = new Map<string, string[]>();

    for (const edge of dependencyGraph.edges) {
      if (!dependencies.has(edge.to)) {
        dependencies.set(edge.to, []);
      }
      dependencies.get(edge.to)!.push(edge.from);

      if (!dependents.has(edge.from)) {
        dependents.set(edge.from, []);
      }
      dependents.get(edge.from)!.push(edge.to);
    }

    let currentLevel = 0;
    let levelCommands: string[] = [];

    // Find commands with no dependencies
    for (const node of dependencyGraph.nodes) {
      if (!dependencies.has(node.id) || dependencies.get(node.id)!.length === 0) {
        levelCommands.push(node.id);
        commandToLevel.set(node.id, currentLevel);
        processed.add(node.id);
      }
    }

    // Add first level
    if (levelCommands.length > 0) {
      levels.push(this.createExecutionLevel(currentLevel, levelCommands, dependencyGraph));
    }

    // Process remaining levels
    while (processed.size < dependencyGraph.nodes.length) {
      currentLevel++;
      levelCommands = [];

      // Find commands whose dependencies are all processed
      for (const node of dependencyGraph.nodes) {
        if (!processed.has(node.id)) {
          const nodeDeps = dependencies.get(node.id) || [];
          const allDepsProcessed = nodeDeps.every(dep => processed.has(dep));

          if (allDepsProcessed) {
            levelCommands.push(node.id);
            commandToLevel.set(node.id, currentLevel);
            processed.add(node.id);
          }
        }
      }

      if (levelCommands.length > 0) {
        levels.push(this.createExecutionLevel(currentLevel, levelCommands, dependencyGraph));
      } else {
        // No progress made - likely circular dependencies
        console.warn('Cannot create execution levels - circular dependencies detected');
        break;
      }
    }

    return levels;
  }

  /**
   * Generate optimization suggestions
   */
  private generateOptimizations(
    dependencyGraph: DependencyGraph,
    executionLevels: ExecutionLevel[],
    context: TemplateToCliContext
  ): DependencyOptimization[] {
    const optimizations: DependencyOptimization[] = [];

    // Parallel execution opportunities
    for (const level of executionLevels) {
      if (level.commands.length > 1 && level.parallel) {
        optimizations.push({
          type: 'PARALLEL',
          targetCommands: level.commands,
          description: `Commands in level ${level.level} can be executed in parallel`,
          benefit: {
            timeReduction: (level.commands.length - 1) * level.estimatedDuration * 0.8
          },
          difficulty: 'easy'
        });
      }
    }

    // Command merging opportunities
    const mergeOpportunities = this.findMergeOpportunities(dependencyGraph);
    optimizations.push(...mergeOpportunities);

    // Reordering opportunities
    const reorderOpportunities = this.findReorderOpportunities(dependencyGraph, executionLevels);
    optimizations.push(...reorderOpportunities);

    // Dependency removal opportunities
    const removalOpportunities = this.findRemovalOpportunities(dependencyGraph);
    optimizations.push(...removalOpportunities);

    return optimizations;
  }

  /**
   * Create execution level object
   */
  private createExecutionLevel(
    level: number,
    commands: string[],
    dependencyGraph: DependencyGraph
  ): ExecutionLevel {
    const levelNodes = dependencyGraph.nodes.filter(node => commands.includes(node.id));
    const estimatedDuration = Math.max(...levelNodes.map(node => node.metadata.estimatedDuration));
    const riskLevel = this.calculateLevelRiskLevel(levelNodes);
    const parallel = this.canExecuteInParallel(levelNodes, dependencyGraph);

    return {
      level,
      commands,
      parallel,
      estimatedDuration,
      riskLevel
    };
  }

  /**
   * Helper methods
   */
  private generateCacheKey(commands: GeneratedCliCommand[], context: TemplateToCliContext): string {
    const commandIds = commands.map(cmd => cmd.id).sort().join(',');
    const contextKey = JSON.stringify(context.target);
    return `${commandIds}_${contextKey}`;
  }

  private calculateGraphMetrics(nodes: DependencyNode[], edges: DependencyEdge[]) {
    const totalNodes = nodes.length;
    const totalEdges = edges.length;
    const maxDepth = Math.max(...nodes.map(node => node.dependencyCount));
    const avgBranchingFactor = totalNodes > 0 ? totalEdges / totalNodes : 0;

    return {
      totalNodes,
      totalEdges,
      maxDepth,
      avgBranchingFactor
    };
  }

  private calculateCycleSeverity(cycleLength: number): 'low' | 'medium' | 'high' | 'critical' {
    if (cycleLength <= 2) return 'critical';
    if (cycleLength <= 4) return 'high';
    if (cycleLength <= 6) return 'medium';
    return 'low';
  }

  private generateCycleResolutions(cycle: string[]): string[] {
    return [
      `Remove one dependency from the cycle: ${cycle.join(' -> ')}`,
      'Review command order and requirements',
      'Consider breaking the cycle with intermediate steps',
      'Use conditional execution to break the dependency'
    ];
  }

  private canExecuteInParallel(nodes: DependencyNode[], dependencyGraph: DependencyGraph): boolean {
    // Check if there are any dependencies between nodes in this level
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const hasDependency = dependencyGraph.edges.some(edge =>
          (edge.from === nodes[i].id && edge.to === nodes[j].id) ||
          (edge.from === nodes[j].id && edge.to === nodes[i].id)
        );

        if (hasDependency) {
          return false;
        }
      }
    }

    return true;
  }

  private calculateLevelRiskLevel(nodes: DependencyNode[]): 'low' | 'medium' | 'high' {
    if (nodes.some(node => node.riskLevel === 'high' || node.critical)) {
      return 'high';
    }
    if (nodes.some(node => node.riskLevel === 'medium')) {
      return 'medium';
    }
    return 'low';
  }

  private findMergeOpportunities(dependencyGraph: DependencyGraph): DependencyOptimization[] {
    const optimizations: DependencyOptimization[] = [];

    // Look for consecutive SET commands on the same target
    const setCommandsByTarget = new Map<string, DependencyNode[]>();

    for (const node of dependencyGraph.nodes) {
      if (node.type === 'SET' && node.id.includes('config_')) {
        // Extract target from command (simplified)
        const target = 'unknown'; // Would extract from actual command
        if (!setCommandsByTarget.has(target)) {
          setCommandsByTarget.set(target, []);
        }
        setCommandsByTarget.get(target)!.push(node);
      }
    }

    for (const [target, commands] of setCommandsByTarget) {
      if (commands.length > 1) {
        optimizations.push({
          type: 'MERGE',
          targetCommands: commands.map(cmd => cmd.id),
          description: `Merge ${commands.length} SET commands on ${target}`,
          benefit: {
            timeReduction: (commands.length - 1) * 500,
            complexityReduction: commands.length - 1
          },
          difficulty: 'medium'
        });
      }
    }

    return optimizations;
  }

  private findReorderOpportunities(
    dependencyGraph: DependencyGraph,
    executionLevels: ExecutionLevel[]
  ): DependencyOptimization[] {
    const optimizations: DependencyOptimization[] = [];

    // Look for commands that can be moved to earlier levels
    for (let i = 1; i < executionLevels.length; i++) {
      const currentLevel = executionLevels[i];
      const previousLevel = executionLevels[i - 1];

      for (const commandId of currentLevel.commands) {
        const command = dependencyGraph.nodes.find(n => n.id === commandId);
        if (command && command.type === 'GET') {
          // GET commands can often be moved earlier
          optimizations.push({
            type: 'REORDER',
            targetCommands: [commandId],
            description: `Move GET command ${commandId} to earlier level`,
            benefit: {
              timeReduction: 1000,
              complexityReduction: 0.5
            },
            difficulty: 'easy'
          });
        }
      }
    }

    return optimizations;
  }

  private findRemovalOpportunities(dependencyGraph: DependencyGraph): DependencyOptimization[] {
    const optimizations: DependencyOptimization[] = [];

    // Look for weak dependencies that might be unnecessary
    const weakDependencies = dependencyGraph.edges.filter(edge => edge.strength === 'weak');

    for (const edge of weakDependencies) {
      optimizations.push({
        type: 'REMOVE',
        targetCommands: [edge.from, edge.to],
        description: `Consider removing weak dependency: ${edge.from} -> ${edge.to}`,
        benefit: {
          complexityReduction: 0.2,
          riskReduction: 0.1
        },
        difficulty: 'hard'
      });
    }

    return optimizations;
  }

  /**
   * Initialize dependency rules
   */
  private initializeDependencyRules(): void {
    this.dependencyRules = [
      // CREATE must come before SET on same object
      {
        name: 'Create before Set',
        description: 'CREATE operations must precede SET operations on the same object',
        type: 'REQUIRES',
        strength: 'strong',
        condition: (source, target, context) => {
          return source.type === 'CREATE' && target.type === 'SET' &&
            this.extractTarget(source) === this.extractTarget(target);
        }
      },

      // SET must come before DELETE on same object
      {
        name: 'Set before Delete',
        description: 'SET operations must precede DELETE operations on the same object',
        type: 'PRECEDES',
        strength: 'medium',
        condition: (source, target, context) => {
          return source.type === 'SET' && target.type === 'DELETE' &&
            this.extractTarget(source) === this.extractTarget(target);
        }
      },

      // Validation commands depend on configuration commands
      {
        name: 'Validate after Configure',
        description: 'Validation commands must follow configuration commands',
        type: 'VALIDATES',
        strength: 'medium',
        condition: (source, target, context) => {
          return source.type === 'SET' && target.type === 'VALIDATION' &&
            this.extractTarget(source) === this.extractTarget(target);
        }
      },

      // Critical commands create dependencies
      {
        name: 'Critical Dependency',
        description: 'Critical commands create dependency relationships',
        type: 'REQUIRES',
        strength: 'strong',
        condition: (source, target, context) => {
          return source.critical && !target.critical && this.areRelatedCommands(source, target);
        }
      },

      // Same target FDN creates dependency
      {
        name: 'Same Target Dependency',
        description: 'Commands on same target FDN have dependencies',
        type: 'RESOURCE',
        strength: 'weak',
        condition: (source, target, context) => {
          return this.extractTarget(source) === this.extractTarget(target) &&
            source.id !== target.id;
        }
      },

      // High-risk commands create validation dependencies
      {
        name: 'Risk Validation',
        description: 'High-risk commands require validation',
        type: 'VALIDATES',
        strength: 'medium',
        condition: (source, target, context) => {
          return source.metadata.riskLevel === 'high' && target.type === 'VALIDATION';
        }
      }
    ];
  }

  /**
   * Extract target from command
   */
  private extractTarget(command: GeneratedCliCommand): string {
    // Simplified target extraction
    if (command.targetFdn) {
      return command.targetFdn.split('=')[0];
    }

    // Extract from command string
    const match = command.command.match(/\s+([A-Za-z][A-Za-z0-9]*(?:=[^,\s]+)?)/);
    return match ? match[1] : 'unknown';
  }

  /**
   * Check if commands are related
   */
  private areRelatedCommands(source: GeneratedCliCommand, target: GeneratedCliCommand): boolean {
    const sourceTarget = this.extractTarget(source);
    const targetTarget = this.extractTarget(target);

    // Check for same target class
    if (sourceTarget === targetTarget) {
      return true;
    }

    // Check for hierarchical relationship (simplified)
    const sourceParts = sourceTarget.split('.');
    const targetParts = targetTarget.split('.');

    if (sourceParts.length > 0 && targetParts.length > 0) {
      return sourceParts[0] === targetParts[0];
    }

    return false;
  }

  /**
   * Clear cache
   */
  public clearCache(): void {
    this.dependencyCache.clear();
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): {
    size: number;
    hitRate: number;
    totalAnalyses: number;
  } {
    return {
      size: this.dependencyCache.size,
      hitRate: 0.7, // Placeholder - would track actual hits
      totalAnalyses: this.dependencyCache.size
    };
  }
}