/**
 * Goal-Oriented Action Planning (GOAP) for RAN Configuration
 * Implements intelligent action sequencing using A* search
 */

import { v4 as uuidv4 } from 'uuid';
import {
  GOAPGoal,
  GOAPAction,
  GOAPPlan,
  ConfigurationChange,
  ManagedObjectType,
  CellConfiguration,
  CellMetrics,
} from '../core/types.js';
import { createLogger } from '../utils/logger.js';

const logger = createLogger('GOAPPlanner');

/**
 * World state representation for GOAP
 */
export interface WorldState {
  cells: Map<string, CellState>;
  networkKPIs: NetworkKPIs;
  constraints: ConstraintSet;
}

interface CellState {
  config: CellConfiguration;
  metrics: CellMetrics;
  isHealthy: boolean;
}

interface NetworkKPIs {
  averageThroughput: number;
  averageLatency: number;
  averageInterference: number;
  energyConsumption: number;
}

interface ConstraintSet {
  maxTiltChange: number;
  maxPowerChange: number;
  minCoverageRSRP: number;
  maxInterference: number;
  emergencyCallsActive: boolean;
}

/**
 * GOAP Action definition with costs and effects
 */
interface GOAPActionDef {
  name: string;
  preconditions: (state: WorldState, params: ActionParams) => boolean;
  effects: (state: WorldState, params: ActionParams) => WorldState;
  cost: (state: WorldState, params: ActionParams) => number;
  risk: (state: WorldState, params: ActionParams) => number;
  parameters: string[];
}

interface ActionParams {
  cellId?: string;
  value?: number;
  targetCellId?: string;
  [key: string]: unknown;
}

/**
 * A* search node for GOAP
 */
interface PlanNode {
  state: WorldState;
  action?: GOAPAction;
  parent?: PlanNode;
  gCost: number;  // Cost from start
  hCost: number;  // Heuristic to goal
  fCost: number;  // Total cost
}

/**
 * Goal-Oriented Action Planner
 */
export class GOAPPlanner {
  private actionDefinitions: Map<string, GOAPActionDef>;
  private maxPlanningDepth: number;
  private maxPlanningTime: number;

  constructor(options: { maxDepth?: number; maxTimeMs?: number } = {}) {
    this.maxPlanningDepth = options.maxDepth || 10;
    this.maxPlanningTime = options.maxTimeMs || 5000;
    this.actionDefinitions = this.initializeActions();

    logger.info('GOAP Planner initialized', {
      actionCount: this.actionDefinitions.size,
      maxDepth: this.maxPlanningDepth,
    });
  }

  /**
   * Create a plan to achieve a goal
   */
  async plan(
    goal: GOAPGoal,
    currentState: WorldState
  ): Promise<GOAPPlan | null> {
    const startTime = performance.now();

    logger.info('Starting GOAP planning', {
      goalId: goal.id,
      goalName: goal.name,
    });

    // A* search for optimal action sequence
    const openSet: PlanNode[] = [];
    const closedSet = new Set<string>();

    const startNode: PlanNode = {
      state: this.cloneState(currentState),
      gCost: 0,
      hCost: this.heuristic(currentState, goal),
      fCost: 0,
    };
    startNode.fCost = startNode.gCost + startNode.hCost;

    openSet.push(startNode);

    while (openSet.length > 0) {
      // Check timeout
      if (performance.now() - startTime > this.maxPlanningTime) {
        logger.warn('Planning timeout reached');
        break;
      }

      // Get node with lowest fCost
      openSet.sort((a, b) => a.fCost - b.fCost);
      const current = openSet.shift()!;

      // Check if goal is satisfied
      if (this.isGoalSatisfied(current.state, goal)) {
        const plan = this.reconstructPlan(current, goal);
        const planningTime = performance.now() - startTime;

        logger.info('Plan found', {
          goalId: goal.id,
          actionCount: plan.actions.length,
          totalCost: plan.totalCost,
          planningTimeMs: planningTime.toFixed(2),
        });

        return plan;
      }

      const stateHash = this.hashState(current.state);
      if (closedSet.has(stateHash)) continue;
      closedSet.add(stateHash);

      // Check depth limit
      const depth = this.getDepth(current);
      if (depth >= this.maxPlanningDepth) continue;

      // Expand neighbors (applicable actions)
      const applicableActions = this.getApplicableActions(current.state);

      for (const { actionDef, params } of applicableActions) {
        const newState = actionDef.effects(this.cloneState(current.state), params);
        const actionCost = actionDef.cost(current.state, params);
        const actionRisk = actionDef.risk(current.state, params);

        const action: GOAPAction = {
          name: actionDef.name,
          preconditions: {},
          effects: {},
          cost: actionCost,
          risk: actionRisk,
        };

        const neighbor: PlanNode = {
          state: newState,
          action,
          parent: current,
          gCost: current.gCost + actionCost + actionRisk * 0.5, // Risk penalty
          hCost: this.heuristic(newState, goal),
          fCost: 0,
        };
        neighbor.fCost = neighbor.gCost + neighbor.hCost;

        const neighborHash = this.hashState(newState);
        if (!closedSet.has(neighborHash)) {
          openSet.push(neighbor);
        }
      }
    }

    logger.warn('No plan found', { goalId: goal.id });
    return null;
  }

  /**
   * Create optimized plans for common goals
   */
  createGoal(type: GoalType, params: GoalParams): GOAPGoal {
    const goalTemplates: Record<GoalType, (p: GoalParams) => GOAPGoal> = {
      maximize_throughput: (p) => ({
        id: uuidv4(),
        name: 'Maximize Throughput',
        targetState: { throughput: 'maximized', cellId: p.cellId },
        priority: 1,
      }),
      minimize_interference: (p) => ({
        id: uuidv4(),
        name: 'Minimize Interference',
        targetState: { interference: 'minimized', cellId: p.cellId },
        priority: 2,
      }),
      optimize_coverage: (p) => ({
        id: uuidv4(),
        name: 'Optimize Coverage',
        targetState: { coverage: 'optimal', cellId: p.cellId },
        priority: 2,
      }),
      save_energy: (p) => ({
        id: uuidv4(),
        name: 'Save Energy',
        targetState: { energy: 'minimized', targetSleepRatio: p.targetSleepRatio },
        priority: 3,
      }),
      heal_cell: (p) => ({
        id: uuidv4(),
        name: 'Heal Cell',
        targetState: { cellHealthy: true, cellId: p.cellId },
        priority: 1,
      }),
    };

    return goalTemplates[type](params);
  }

  /**
   * Initialize available actions
   */
  private initializeActions(): Map<string, GOAPActionDef> {
    const actions = new Map<string, GOAPActionDef>();

    // Electrical Tilt adjustment
    actions.set('adjust_electrical_tilt', {
      name: 'Adjust Electrical Tilt',
      parameters: ['cellId', 'tiltDelta'],
      preconditions: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (!cell) return false;
        const newTilt = cell.config.electricalTilt + (params.value as number);
        return newTilt >= 0 && newTilt <= 100; // Valid range
      },
      effects: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (cell) {
          cell.config.electricalTilt += params.value as number;
          // Simulate effect on metrics
          if (params.value as number > 0) {
            // Downtilt: reduces interference, may reduce coverage
            cell.metrics.interferenceLevel *= 0.9;
            cell.metrics.rsrp -= 2;
          } else {
            // Uptilt: increases coverage, may increase interference
            cell.metrics.rsrp += 2;
            cell.metrics.interferenceLevel *= 1.1;
          }
        }
        return state;
      },
      cost: () => 1,
      risk: () => 0.2,
    });

    // Transmit power adjustment
    actions.set('adjust_transmit_power', {
      name: 'Adjust Transmit Power',
      parameters: ['cellId', 'powerDelta'],
      preconditions: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (!cell) return false;
        const newPower = cell.config.transmitPower + (params.value as number);
        return newPower >= 0 && newPower <= 46; // dBm range
      },
      effects: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (cell) {
          cell.config.transmitPower += params.value as number;
          // Power increase improves coverage but may increase interference
          if (params.value as number > 0) {
            cell.metrics.rsrp += params.value as number * 0.5;
            cell.metrics.throughputDl *= 1.05;
          } else {
            cell.metrics.rsrp += params.value as number * 0.5;
            cell.metrics.powerConsumption *= 0.95;
          }
        }
        return state;
      },
      cost: () => 1.5,
      risk: () => 0.3,
    });

    // PCI change
    actions.set('change_pci', {
      name: 'Change PCI',
      parameters: ['cellId', 'newPci'],
      preconditions: (state, params) => {
        const newPci = params.value as number;
        // Check PCI not already used by neighbors
        for (const [id, cell] of state.cells) {
          if (id !== params.cellId && cell.config.pci === newPci) {
            return false;
          }
        }
        return newPci >= 0 && newPci < 504;
      },
      effects: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (cell) {
          cell.config.pci = params.value as number;
          // PCI change can reduce interference with proper planning
          cell.metrics.interferenceLevel *= 0.8;
        }
        return state;
      },
      cost: () => 3, // Higher cost as it's more disruptive
      risk: () => 0.5,
    });

    // Handover parameter tuning
    actions.set('tune_handover_params', {
      name: 'Tune Handover Parameters',
      parameters: ['cellId', 'a3Offset', 'timeToTrigger'],
      preconditions: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        return cell !== undefined;
      },
      effects: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (cell && params.a3Offset !== undefined) {
          cell.config.a3Offset = params.a3Offset as number;
          if (params.timeToTrigger !== undefined) {
            cell.config.timeToTrigger = params.timeToTrigger as number;
          }
        }
        return state;
      },
      cost: () => 2,
      risk: () => 0.4,
    });

    // Enable sleep mode
    actions.set('enable_sleep_mode', {
      name: 'Enable Sleep Mode',
      parameters: ['cellId', 'sleepRatio'],
      preconditions: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (!cell) return false;
        // Can only sleep if load is low
        return cell.metrics.prbUtilizationDl < 0.3;
      },
      effects: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (cell) {
          cell.metrics.sleepRatio = params.value as number;
          cell.metrics.powerConsumption *= (1 - (params.value as number) * 0.3);
        }
        return state;
      },
      cost: () => 0.5,
      risk: () => 0.1,
    });

    // Restart cell (healing action)
    actions.set('restart_cell', {
      name: 'Restart Cell',
      parameters: ['cellId'],
      preconditions: (state) => {
        return !state.constraints.emergencyCallsActive;
      },
      effects: (state, params) => {
        const cell = state.cells.get(params.cellId as string);
        if (cell) {
          cell.isHealthy = true;
          // Reset metrics to baseline
          cell.metrics.interferenceLevel = -100;
        }
        return state;
      },
      cost: () => 5, // High cost due to service interruption
      risk: () => 0.8,
    });

    return actions;
  }

  /**
   * Get all applicable actions for current state
   */
  private getApplicableActions(
    state: WorldState
  ): { actionDef: GOAPActionDef; params: ActionParams }[] {
    const applicable: { actionDef: GOAPActionDef; params: ActionParams }[] = [];

    for (const [, actionDef] of this.actionDefinitions) {
      // Generate parameter combinations
      for (const [cellId] of state.cells) {
        const paramSets = this.generateParameterSets(actionDef, cellId);

        for (const params of paramSets) {
          if (actionDef.preconditions(state, params)) {
            applicable.push({ actionDef, params });
          }
        }
      }
    }

    return applicable;
  }

  /**
   * Generate parameter combinations for an action
   */
  private generateParameterSets(actionDef: GOAPActionDef, cellId: string): ActionParams[] {
    const params: ActionParams[] = [];

    switch (actionDef.name) {
      case 'Adjust Electrical Tilt':
        for (const delta of [-4, -2, 2, 4]) {
          params.push({ cellId, value: delta });
        }
        break;
      case 'Adjust Transmit Power':
        for (const delta of [-3, -1, 1, 3]) {
          params.push({ cellId, value: delta });
        }
        break;
      case 'Change PCI':
        // Generate a few PCI options
        for (let i = 0; i < 5; i++) {
          params.push({ cellId, value: Math.floor(Math.random() * 504) });
        }
        break;
      case 'Enable Sleep Mode':
        for (const ratio of [0.3, 0.5, 0.7]) {
          params.push({ cellId, value: ratio });
        }
        break;
      default:
        params.push({ cellId });
    }

    return params;
  }

  /**
   * Heuristic function for A* search
   */
  private heuristic(state: WorldState, goal: GOAPGoal): number {
    let h = 0;
    const targetState = goal.targetState;

    if (targetState.throughput === 'maximized') {
      // Estimate steps needed to maximize throughput
      h += Math.max(0, 100 - state.networkKPIs.averageThroughput) / 20;
    }

    if (targetState.interference === 'minimized') {
      // Estimate steps needed to minimize interference
      h += Math.max(0, state.networkKPIs.averageInterference + 80) / 10;
    }

    if (targetState.cellHealthy && targetState.cellId) {
      const cell = state.cells.get(targetState.cellId as string);
      if (cell && !cell.isHealthy) {
        h += 3; // At least restart action needed
      }
    }

    if (targetState.energy === 'minimized') {
      h += state.networkKPIs.energyConsumption / 100;
    }

    return h;
  }

  /**
   * Check if goal is satisfied
   */
  private isGoalSatisfied(state: WorldState, goal: GOAPGoal): boolean {
    const target = goal.targetState;

    if (target.throughput === 'maximized') {
      // Consider goal achieved if throughput improved significantly
      return state.networkKPIs.averageThroughput >= 80;
    }

    if (target.interference === 'minimized') {
      return state.networkKPIs.averageInterference <= -95;
    }

    if (target.cellHealthy && target.cellId) {
      const cell = state.cells.get(target.cellId as string);
      return cell?.isHealthy === true;
    }

    if (target.energy === 'minimized') {
      return state.networkKPIs.energyConsumption < 70;
    }

    return false;
  }

  /**
   * Reconstruct plan from search result
   */
  private reconstructPlan(node: PlanNode, goal: GOAPGoal): GOAPPlan {
    const actions: GOAPAction[] = [];
    let current: PlanNode | undefined = node;
    let totalCost = 0;
    let totalRisk = 0;

    while (current?.parent) {
      if (current.action) {
        actions.unshift(current.action);
        totalCost += current.action.cost;
        totalRisk = Math.max(totalRisk, current.action.risk);
      }
      current = current.parent;
    }

    return {
      goalId: goal.id,
      actions,
      totalCost,
      totalRisk,
      estimatedDuration: actions.length * 1000, // 1 second per action estimate
    };
  }

  /**
   * Get depth of node in search tree
   */
  private getDepth(node: PlanNode): number {
    let depth = 0;
    let current: PlanNode | undefined = node;
    while (current?.parent) {
      depth++;
      current = current.parent;
    }
    return depth;
  }

  /**
   * Clone world state (deep copy)
   */
  private cloneState(state: WorldState): WorldState {
    const newCells = new Map<string, CellState>();
    for (const [id, cell] of state.cells) {
      newCells.set(id, {
        config: { ...cell.config },
        metrics: { ...cell.metrics },
        isHealthy: cell.isHealthy,
      });
    }

    return {
      cells: newCells,
      networkKPIs: { ...state.networkKPIs },
      constraints: { ...state.constraints },
    };
  }

  /**
   * Create hash of state for visited set
   */
  private hashState(state: WorldState): string {
    const parts: string[] = [];
    for (const [id, cell] of state.cells) {
      parts.push(`${id}:${cell.config.electricalTilt}:${cell.config.pci}:${cell.isHealthy}`);
    }
    return parts.sort().join('|');
  }
}

export type GoalType =
  | 'maximize_throughput'
  | 'minimize_interference'
  | 'optimize_coverage'
  | 'save_energy'
  | 'heal_cell';

export interface GoalParams {
  cellId?: string;
  targetSleepRatio?: number;
  targetThroughput?: number;
}

/**
 * Create a configured GOAP planner instance
 */
export function createGOAPPlanner(
  options?: { maxDepth?: number; maxTimeMs?: number }
): GOAPPlanner {
  return new GOAPPlanner(options);
}
