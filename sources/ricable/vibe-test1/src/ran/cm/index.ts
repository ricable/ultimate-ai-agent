/**
 * Configuration Management (CM) Module
 *
 * Handles autonomous optimization of RAN parameters:
 * - Uplink Power Control (P0, Alpha, PUSCH, PUCCH)
 * - Mobility Parameters (A3, TTT, Hysteresis)
 * - Antenna Parameters (Tilt, Power)
 * - Scheduler Parameters
 *
 * Uses Decision Transformer for safe, offline RL-based optimization
 */

export { UplinkOptimizer, SliceAwareOptimizer } from './uplink-optimizer.js';
export type { UplinkOptimizerConfig } from './uplink-optimizer.js';
