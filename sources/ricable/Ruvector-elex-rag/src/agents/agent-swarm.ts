/**
 * Agent Swarm System for RAN Optimization
 *
 * Implements autonomous agent swarms for network optimization:
 * - Optimizer Agent: Finds optimal configurations using genetic algorithms
 * - Validator Agent: Sanity checks proposed changes
 * - Auditor Agent: Monitors post-implementation performance
 * - Coordinator Agent: Manages swarm coordination
 *
 * Solves the Tuning Paradox through coordinated multi-cell optimization.
 */

import { v4 as uuidv4 } from 'uuid';
import type {
  Agent,
  AgentRole,
  AgentState,
  AgentConfig,
  AgentAction,
  NetworkGraph,
  BayesianPrediction,
  PowerControlParams,
  ParameterChange,
  OptimizationCandidate,
  OptimizationResult,
  AlphaValue,
} from '../core/types.js';
import { AlphaValues, getConfig } from '../core/config.js';
import { GNNInferenceEngine } from '../gnn/gnn-engine.js';
import { logger, logAgentAction, logOptimization } from '../utils/logger.js';

/**
 * Base Agent class
 */
abstract class BaseAgent implements Agent {
  id: string;
  name: string;
  role: AgentRole;
  state: AgentState = 'idle';
  clusterId: string;
  config: AgentConfig;
  history: AgentAction[] = [];

  protected systemConfig = getConfig();
  protected gnnEngine: GNNInferenceEngine;

  constructor(role: AgentRole, clusterId: string, config?: Partial<AgentConfig>) {
    this.id = uuidv4();
    this.role = role;
    this.clusterId = clusterId;
    this.name = `${role}_${this.id.substring(0, 8)}`;
    this.config = {
      explorationRate: config?.explorationRate ?? 0.1,
      learningRate: config?.learningRate ?? 0.01,
      discountFactor: config?.discountFactor ?? 0.95,
      maxActionsPerCycle: config?.maxActionsPerCycle ?? 10,
      riskTolerance: config?.riskTolerance ?? 0.3,
      minAutoApprovalConfidence: config?.minAutoApprovalConfidence ?? 0.95,
    };
    this.gnnEngine = new GNNInferenceEngine();
  }

  abstract execute(graph: NetworkGraph): Promise<AgentAction | null>;

  protected setState(newState: AgentState): void {
    const oldState = this.state;
    this.state = newState;
    logAgentAction(this.id, 'state_change', { oldState, newState });
  }

  protected recordAction(action: AgentAction): void {
    this.history.push(action);
    // Keep only recent history
    if (this.history.length > 1000) {
      this.history = this.history.slice(-500);
    }
  }
}

/**
 * Optimizer Agent - "The Architect"
 *
 * Uses genetic algorithms to find optimal (P0, alpha) configurations
 * by simulating outcomes through the GNN.
 */
export class OptimizerAgent extends BaseAgent {
  private populationSize: number;
  private generations: number;
  private mutationRate: number;
  private crossoverRate: number;

  constructor(clusterId: string, config?: Partial<AgentConfig>) {
    super('optimizer', clusterId, config);
    this.populationSize = this.systemConfig.optimization.populationSize;
    this.generations = this.systemConfig.optimization.generations;
    this.mutationRate = this.systemConfig.optimization.mutationRate;
    this.crossoverRate = this.systemConfig.optimization.crossoverRate;
  }

  async execute(graph: NetworkGraph): Promise<AgentAction | null> {
    this.setState('optimizing');
    logOptimization(this.clusterId, 'started');

    try {
      const result = await this.runGeneticAlgorithm(graph);

      if (result.bestCandidate.fitness <= 0) {
        this.setState('idle');
        return null;
      }

      // Convert best candidate to action
      const action = this.createAction(graph, result);
      this.recordAction(action);

      this.setState('idle');
      logOptimization(this.clusterId, 'completed', {
        fitness: result.bestCandidate.fitness,
        sinrImprovement: result.bestCandidate.sinrImprovement,
        generations: result.totalGenerations,
      });

      return action;
    } catch (error) {
      logger.error('Optimizer agent failed', {
        agentId: this.id,
        error: (error as Error).message,
      });
      this.setState('idle');
      return null;
    }
  }

  /**
   * Run genetic algorithm optimization
   */
  private async runGeneticAlgorithm(graph: NetworkGraph): Promise<OptimizationResult> {
    const startTime = Date.now();
    const nodeIds = Array.from(graph.nodes.keys());

    // Initialize population with heuristic bias
    let population = this.initializePopulation(nodeIds, graph);

    // Evaluate initial population
    population = await this.evaluatePopulation(population, graph);

    const convergenceHistory: number[] = [];
    let bestEverCandidate = population[0];

    for (let gen = 0; gen < this.generations; gen++) {
      // Selection (tournament)
      const selected = this.tournamentSelection(population);

      // Crossover
      const offspring = this.crossover(selected, nodeIds);

      // Mutation
      const mutated = this.mutate(offspring, nodeIds);

      // Evaluate offspring
      const evaluatedOffspring = await this.evaluatePopulation(mutated, graph);

      // Elitism: keep best from previous generation
      population = this.elitistSelection(population, evaluatedOffspring);

      // Track best
      if (population[0].fitness > bestEverCandidate.fitness) {
        bestEverCandidate = { ...population[0] };
      }

      convergenceHistory.push(population[0].fitness);

      // Early stopping if converged
      if (gen > 20 && this.hasConverged(convergenceHistory.slice(-20))) {
        break;
      }
    }

    // Extract Pareto frontier for multi-objective
    const paretoFrontier = this.extractParetoFrontier(population);

    return {
      bestCandidate: bestEverCandidate,
      allCandidates: population,
      convergenceHistory,
      totalGenerations: convergenceHistory.length,
      totalEvaluations: convergenceHistory.length * this.populationSize,
      computationTime: Date.now() - startTime,
      paretoFrontier,
    };
  }

  /**
   * Initialize population with heuristic bias towards optimal regions
   */
  private initializePopulation(
    nodeIds: string[],
    graph: NetworkGraph
  ): OptimizationCandidate[] {
    const population: OptimizationCandidate[] = [];

    for (let i = 0; i < this.populationSize; i++) {
      const config = new Map<string, PowerControlParams>();

      for (const nodeId of nodeIds) {
        const node = graph.nodes.get(nodeId)!;
        const currentConfig = node.config.powerControl;

        // Bias towards known good regions
        // Alpha: prefer 0.6-0.8 range (fractional power control)
        // P0: adjust based on alpha change
        let alpha: AlphaValue;
        let p0: number;

        if (i < this.populationSize * 0.3) {
          // 30% use current config (baseline)
          alpha = currentConfig.alpha;
          p0 = currentConfig.pZeroNominalPusch;
        } else if (i < this.populationSize * 0.7) {
          // 40% biased towards optimal region
          alpha = this.biasedAlphaSelection();
          p0 = this.compensateP0(currentConfig.pZeroNominalPusch, currentConfig.alpha, alpha);
        } else {
          // 30% random exploration
          alpha = AlphaValues[Math.floor(Math.random() * AlphaValues.length)];
          p0 = this.systemConfig.parameters.p0Min +
            Math.random() * (this.systemConfig.parameters.p0Max - this.systemConfig.parameters.p0Min);
        }

        config.set(nodeId, {
          pZeroNominalPusch: Math.round(p0),
          alpha,
          pZeroNominalPucch: currentConfig.pZeroNominalPucch,
          pCmax: currentConfig.pCmax,
        });
      }

      population.push({
        id: uuidv4(),
        config,
        fitness: 0,
        sinrImprovement: 0,
        spectralEfficiency: 0,
        coveragePenalty: 0,
        uncertaintyPenalty: 0,
        generation: 0,
      });
    }

    return population;
  }

  /**
   * Biased alpha selection favoring optimal range (0.6-0.8)
   */
  private biasedAlphaSelection(): AlphaValue {
    const optimalAlphas: AlphaValue[] = [0.6, 0.7, 0.8];
    const otherAlphas: AlphaValue[] = [0.4, 0.5, 0.9, 1.0];

    // 70% chance of optimal, 30% chance of other
    if (Math.random() < 0.7) {
      return optimalAlphas[Math.floor(Math.random() * optimalAlphas.length)];
    }
    return otherAlphas[Math.floor(Math.random() * otherAlphas.length)];
  }

  /**
   * Compensate P0 for alpha change (key to solving Tuning Paradox)
   */
  private compensateP0(currentP0: number, currentAlpha: number, newAlpha: AlphaValue): number {
    // When reducing alpha, we need to increase P0 to maintain cell-edge coverage
    // Approximate compensation: ΔP0 = k * (currentAlpha - newAlpha) * avgPathLoss
    // Assuming average path loss around 100-120 dB
    const avgPathLoss = 110; // dB
    const compensation = (currentAlpha - newAlpha) * avgPathLoss * 0.5;

    return Math.max(
      this.systemConfig.parameters.p0Min,
      Math.min(this.systemConfig.parameters.p0Max, currentP0 + compensation)
    );
  }

  /**
   * Evaluate population using GNN
   */
  private async evaluatePopulation(
    population: OptimizationCandidate[],
    graph: NetworkGraph
  ): Promise<OptimizationCandidate[]> {
    const evaluated: OptimizationCandidate[] = [];

    for (const candidate of population) {
      // Convert config map to partial power control params
      const changes = new Map<string, Partial<PowerControlParams>>();
      for (const [nodeId, params] of candidate.config) {
        changes.set(nodeId, {
          pZeroNominalPusch: params.pZeroNominalPusch,
          alpha: params.alpha,
        });
      }

      // Get GNN prediction
      const prediction = this.gnnEngine.predictSINRImprovement(graph, changes);

      // Calculate fitness using multi-objective function
      const fitness = this.calculateFitness(prediction, candidate);

      evaluated.push({
        ...candidate,
        fitness,
        sinrImprovement: prediction.sinrImprovement,
        spectralEfficiency: prediction.spectralEfficiencyGain,
        coveragePenalty: this.calculateCoveragePenalty(prediction),
        uncertaintyPenalty: this.calculateUncertaintyPenalty(prediction),
      });
    }

    // Sort by fitness (descending)
    return evaluated.sort((a, b) => b.fitness - a.fitness);
  }

  /**
   * Calculate fitness score (from PRD equation)
   * F = SINR_improvement - λ₁ * Coverage_Penalty - λ₂ * Uncertainty_Penalty
   */
  private calculateFitness(
    prediction: BayesianPrediction,
    candidate: OptimizationCandidate
  ): number {
    const lambda1 = 2.0; // Coverage penalty weight
    const lambda2 = 1.0; // Uncertainty penalty weight

    const coveragePenalty = this.calculateCoveragePenalty(prediction);
    const uncertaintyPenalty = this.calculateUncertaintyPenalty(prediction);

    return prediction.sinrImprovement - lambda1 * coveragePenalty - lambda2 * uncertaintyPenalty;
  }

  /**
   * Calculate coverage penalty
   */
  private calculateCoveragePenalty(prediction: BayesianPrediction): number {
    // Penalize if coverage impact is negative (coverage degradation)
    if (prediction.coverageImpact < 0) {
      return Math.abs(prediction.coverageImpact);
    }
    return 0;
  }

  /**
   * Calculate uncertainty penalty
   */
  private calculateUncertaintyPenalty(prediction: BayesianPrediction): number {
    // Penalize high uncertainty
    const threshold = this.systemConfig.bayesian.uncertaintyThreshold;
    if (prediction.uncertainty > threshold) {
      return (prediction.uncertainty - threshold) * 2;
    }
    return 0;
  }

  /**
   * Tournament selection
   */
  private tournamentSelection(population: OptimizationCandidate[]): OptimizationCandidate[] {
    const selected: OptimizationCandidate[] = [];
    const tournamentSize = 3;

    while (selected.length < population.length) {
      const tournament: OptimizationCandidate[] = [];
      for (let i = 0; i < tournamentSize; i++) {
        const idx = Math.floor(Math.random() * population.length);
        tournament.push(population[idx]);
      }
      tournament.sort((a, b) => b.fitness - a.fitness);
      selected.push(tournament[0]);
    }

    return selected;
  }

  /**
   * Crossover operation
   */
  private crossover(
    selected: OptimizationCandidate[],
    nodeIds: string[]
  ): OptimizationCandidate[] {
    const offspring: OptimizationCandidate[] = [];

    for (let i = 0; i < selected.length; i += 2) {
      const parent1 = selected[i];
      const parent2 = selected[(i + 1) % selected.length];

      if (Math.random() < this.crossoverRate) {
        // Uniform crossover at node level
        const childConfig = new Map<string, PowerControlParams>();

        for (const nodeId of nodeIds) {
          const useParent1 = Math.random() < 0.5;
          const parentConfig = useParent1
            ? parent1.config.get(nodeId)!
            : parent2.config.get(nodeId)!;
          childConfig.set(nodeId, { ...parentConfig });
        }

        offspring.push({
          id: uuidv4(),
          config: childConfig,
          fitness: 0,
          sinrImprovement: 0,
          spectralEfficiency: 0,
          coveragePenalty: 0,
          uncertaintyPenalty: 0,
          generation: parent1.generation + 1,
        });
      } else {
        // No crossover, copy parent
        offspring.push({
          ...parent1,
          id: uuidv4(),
          generation: parent1.generation + 1,
        });
      }
    }

    return offspring;
  }

  /**
   * Mutation operation
   */
  private mutate(
    population: OptimizationCandidate[],
    nodeIds: string[]
  ): OptimizationCandidate[] {
    return population.map((candidate) => {
      const mutatedConfig = new Map<string, PowerControlParams>();

      for (const nodeId of nodeIds) {
        const params = candidate.config.get(nodeId)!;
        let newParams = { ...params };

        if (Math.random() < this.mutationRate) {
          // Mutate alpha
          const alphaIdx = AlphaValues.indexOf(params.alpha);
          const newAlphaIdx = Math.max(
            0,
            Math.min(AlphaValues.length - 1, alphaIdx + (Math.random() < 0.5 ? -1 : 1))
          );
          newParams.alpha = AlphaValues[newAlphaIdx];
        }

        if (Math.random() < this.mutationRate) {
          // Mutate P0 by ±1-3 dB
          const delta = (Math.random() < 0.5 ? -1 : 1) * (1 + Math.floor(Math.random() * 3));
          newParams.pZeroNominalPusch = Math.max(
            this.systemConfig.parameters.p0Min,
            Math.min(this.systemConfig.parameters.p0Max, params.pZeroNominalPusch + delta)
          );
        }

        mutatedConfig.set(nodeId, newParams);
      }

      return {
        ...candidate,
        config: mutatedConfig,
      };
    });
  }

  /**
   * Elitist selection: keep best from both populations
   */
  private elitistSelection(
    oldPop: OptimizationCandidate[],
    newPop: OptimizationCandidate[]
  ): OptimizationCandidate[] {
    const combined = [...oldPop, ...newPop];
    combined.sort((a, b) => b.fitness - a.fitness);
    return combined.slice(0, this.populationSize);
  }

  /**
   * Check if algorithm has converged
   */
  private hasConverged(recentFitness: number[]): boolean {
    if (recentFitness.length < 10) return false;

    const mean = recentFitness.reduce((a, b) => a + b, 0) / recentFitness.length;
    const variance =
      recentFitness.reduce((sum, f) => sum + Math.pow(f - mean, 2), 0) / recentFitness.length;

    // Converged if variance is very small
    return Math.sqrt(variance) < 0.001;
  }

  /**
   * Extract Pareto frontier for multi-objective optimization
   */
  private extractParetoFrontier(population: OptimizationCandidate[]): OptimizationCandidate[] {
    const frontier: OptimizationCandidate[] = [];

    for (const candidate of population) {
      let isDominated = false;

      for (const other of population) {
        if (candidate === other) continue;

        // Check if 'other' dominates 'candidate'
        const otherBetterSINR = other.sinrImprovement >= candidate.sinrImprovement;
        const otherBetterSE = other.spectralEfficiency >= candidate.spectralEfficiency;
        const otherStrictlyBetter =
          other.sinrImprovement > candidate.sinrImprovement ||
          other.spectralEfficiency > candidate.spectralEfficiency;

        if (otherBetterSINR && otherBetterSE && otherStrictlyBetter) {
          isDominated = true;
          break;
        }
      }

      if (!isDominated) {
        frontier.push(candidate);
      }
    }

    return frontier;
  }

  /**
   * Create action from optimization result
   */
  private createAction(graph: NetworkGraph, result: OptimizationResult): AgentAction {
    const changes: ParameterChange[] = [];
    const targetCells: string[] = [];

    for (const [nodeId, newParams] of result.bestCandidate.config) {
      const node = graph.nodes.get(nodeId);
      if (!node) continue;

      const oldParams = node.config.powerControl;

      // Only record actual changes
      if (
        newParams.pZeroNominalPusch !== oldParams.pZeroNominalPusch ||
        newParams.alpha !== oldParams.alpha
      ) {
        targetCells.push(nodeId);

        if (newParams.pZeroNominalPusch !== oldParams.pZeroNominalPusch) {
          changes.push({
            cellId: nodeId,
            parameter: 'pZeroNominalPusch',
            oldValue: oldParams.pZeroNominalPusch,
            newValue: newParams.pZeroNominalPusch,
          });
        }

        if (newParams.alpha !== oldParams.alpha) {
          changes.push({
            cellId: nodeId,
            parameter: 'alpha',
            oldValue: oldParams.alpha,
            newValue: newParams.alpha,
          });
        }
      }
    }

    // Get prediction for the best candidate
    const changesMap = new Map<string, Partial<PowerControlParams>>();
    for (const [nodeId, params] of result.bestCandidate.config) {
      changesMap.set(nodeId, {
        pZeroNominalPusch: params.pZeroNominalPusch,
        alpha: params.alpha,
      });
    }
    const prediction = this.gnnEngine.predictSINRImprovement(graph, changesMap);

    return {
      id: uuidv4(),
      agentId: this.id,
      type: 'parameter_change',
      targetCells,
      changes,
      prediction,
      status: 'proposed',
      timestamp: new Date(),
    };
  }
}

/**
 * Validator Agent - "The Skeptic"
 *
 * Sanity checks proposed configurations against hard constraints.
 */
export class ValidatorAgent extends BaseAgent {
  constructor(clusterId: string, config?: Partial<AgentConfig>) {
    super('validator', clusterId, config);
  }

  async execute(graph: NetworkGraph): Promise<AgentAction | null> {
    // Validator is reactive, not proactive
    return null;
  }

  /**
   * Validate a proposed action
   */
  validateAction(action: AgentAction): { valid: boolean; reasons: string[] } {
    const reasons: string[] = [];

    for (const change of action.changes) {
      // Check P0 range
      if (change.parameter === 'pZeroNominalPusch') {
        if (change.newValue < this.systemConfig.parameters.p0Min) {
          reasons.push(
            `P0 for ${change.cellId} (${change.newValue}) below minimum (${this.systemConfig.parameters.p0Min})`
          );
        }
        if (change.newValue > this.systemConfig.parameters.p0Max) {
          reasons.push(
            `P0 for ${change.cellId} (${change.newValue}) above maximum (${this.systemConfig.parameters.p0Max})`
          );
        }
        // Hard constraint: never set P0 > -90 dBm (from PRD)
        if (change.newValue > -90) {
          reasons.push(
            `P0 for ${change.cellId} (${change.newValue}) exceeds safety limit (-90 dBm)`
          );
        }
      }

      // Check alpha is valid enum value
      if (change.parameter === 'alpha') {
        if (!AlphaValues.includes(change.newValue as AlphaValue)) {
          reasons.push(
            `Alpha for ${change.cellId} (${change.newValue}) is not a valid 3GPP value`
          );
        }
      }
    }

    // Check prediction confidence
    if (action.prediction.uncertainty > this.systemConfig.bayesian.uncertaintyThreshold * 2) {
      reasons.push(
        `Prediction uncertainty (${action.prediction.uncertainty.toFixed(2)}) is too high`
      );
    }

    // Check coverage impact
    if (action.prediction.coverageImpact < -0.1) {
      reasons.push(
        `Predicted coverage degradation (${(action.prediction.coverageImpact * 100).toFixed(1)}%) is too severe`
      );
    }

    return {
      valid: reasons.length === 0,
      reasons,
    };
  }
}

/**
 * Auditor Agent - "The Historian"
 *
 * Monitors post-implementation performance using 3-ROP stability protocol.
 */
export class AuditorAgent extends BaseAgent {
  private monitoringActions: Map<
    string,
    {
      action: AgentAction;
      ropCount: number;
      kpiHistory: { rop: number; sinr: number; degraded: boolean }[];
    }
  > = new Map();

  constructor(clusterId: string, config?: Partial<AgentConfig>) {
    super('auditor', clusterId, config);
  }

  async execute(graph: NetworkGraph): Promise<AgentAction | null> {
    this.setState('validating');

    // Check all monitored actions
    const rollbackActions: AgentAction[] = [];

    for (const [actionId, monitoring] of this.monitoringActions) {
      // Simulate ROP check
      const currentSINR = this.measureCurrentSINR(graph, monitoring.action.targetCells);
      const baselineSINR = monitoring.action.prediction.sinrImprovement;

      const degraded = currentSINR < baselineSINR * 0.9; // More than 10% degradation

      monitoring.kpiHistory.push({
        rop: monitoring.ropCount,
        sinr: currentSINR,
        degraded,
      });
      monitoring.ropCount++;

      // 3-ROP Stability Protocol
      if (monitoring.ropCount >= 3) {
        const lastThree = monitoring.kpiHistory.slice(-3);
        const allDegraded = lastThree.every((h) => h.degraded);

        if (allDegraded) {
          // Trigger rollback
          rollbackActions.push(this.createRollbackAction(monitoring.action));
          this.monitoringActions.delete(actionId);

          logger.warn('3-ROP degradation detected, triggering rollback', {
            actionId,
            kpiHistory: lastThree,
          });
        } else if (!lastThree[2].degraded) {
          // Success - remove from monitoring
          this.monitoringActions.delete(actionId);
          logger.info('Action passed 3-ROP validation', { actionId });
        }
      }
    }

    this.setState('idle');

    // Return first rollback action if any
    if (rollbackActions.length > 0) {
      this.recordAction(rollbackActions[0]);
      return rollbackActions[0];
    }

    return null;
  }

  /**
   * Start monitoring an executed action
   */
  startMonitoring(action: AgentAction): void {
    this.monitoringActions.set(action.id, {
      action,
      ropCount: 0,
      kpiHistory: [],
    });

    logger.info('Started monitoring action', {
      actionId: action.id,
      targetCells: action.targetCells.length,
    });
  }

  /**
   * Measure current SINR (would use PM data in production)
   */
  private measureCurrentSINR(graph: NetworkGraph, targetCells: string[]): number {
    let totalSINR = 0;
    let count = 0;

    for (const cellId of targetCells) {
      const node = graph.nodes.get(cellId);
      if (node?.metrics) {
        totalSINR += node.metrics.pmPuschSinr.mean;
        count++;
      }
    }

    return count > 0 ? totalSINR / count : 0;
  }

  /**
   * Create rollback action
   */
  private createRollbackAction(originalAction: AgentAction): AgentAction {
    // Reverse the changes
    const rollbackChanges: ParameterChange[] = originalAction.changes.map((change) => ({
      ...change,
      oldValue: change.newValue,
      newValue: change.oldValue,
    }));

    return {
      id: uuidv4(),
      agentId: this.id,
      type: 'rollback',
      targetCells: originalAction.targetCells,
      changes: rollbackChanges,
      prediction: originalAction.prediction, // Use same prediction (reversed)
      status: 'proposed',
      timestamp: new Date(),
    };
  }
}

/**
 * Swarm Controller
 *
 * Coordinates multiple agents for distributed optimization.
 */
export class SwarmController {
  private agents: Map<string, BaseAgent> = new Map();
  private config = getConfig();
  private coordinationInterval: number;

  constructor() {
    this.coordinationInterval = this.config.swarm.coordinationInterval;
  }

  /**
   * Create agent swarm for a cluster
   */
  createSwarm(clusterId: string): { optimizer: OptimizerAgent; validator: ValidatorAgent; auditor: AuditorAgent } {
    const optimizer = new OptimizerAgent(clusterId);
    const validator = new ValidatorAgent(clusterId);
    const auditor = new AuditorAgent(clusterId);

    this.agents.set(optimizer.id, optimizer);
    this.agents.set(validator.id, validator);
    this.agents.set(auditor.id, auditor);

    logger.info('Created agent swarm', {
      clusterId,
      optimizerId: optimizer.id,
      validatorId: validator.id,
      auditorId: auditor.id,
    });

    return { optimizer, validator, auditor };
  }

  /**
   * Run optimization cycle
   */
  async runOptimizationCycle(
    graph: NetworkGraph,
    swarm: { optimizer: OptimizerAgent; validator: ValidatorAgent; auditor: AuditorAgent }
  ): Promise<AgentAction | null> {
    // 1. Check for any pending rollbacks from auditor
    const auditAction = await swarm.auditor.execute(graph);
    if (auditAction && auditAction.type === 'rollback') {
      return auditAction;
    }

    // 2. Run optimizer to get proposed action
    const proposedAction = await swarm.optimizer.execute(graph);
    if (!proposedAction) {
      return null;
    }

    // 3. Validate proposed action
    const validation = swarm.validator.validateAction(proposedAction);
    if (!validation.valid) {
      logger.warn('Proposed action rejected by validator', {
        actionId: proposedAction.id,
        reasons: validation.reasons,
      });
      proposedAction.status = 'rejected';
      return null;
    }

    // 4. Check if auto-approval confidence is met
    if (proposedAction.prediction.uncertainty < swarm.optimizer.config.minAutoApprovalConfidence) {
      proposedAction.status = 'approved';
      // Start monitoring
      swarm.auditor.startMonitoring(proposedAction);
    } else {
      // Needs human approval
      proposedAction.status = 'proposed';
    }

    return proposedAction;
  }

  /**
   * Get all agents
   */
  getAgents(): BaseAgent[] {
    return Array.from(this.agents.values());
  }

  /**
   * Get agent by ID
   */
  getAgent(agentId: string): BaseAgent | undefined {
    return this.agents.get(agentId);
  }
}

export default SwarmController;
