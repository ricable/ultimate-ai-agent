/**
 * Self-Learning Agent for RAN CM/PM/FM Data
 * Integrates with midstream for live data streaming
 * Uses ruvector for spatial learning and agentdb for episode memory
 */

import { EventEmitter } from 'events';

// ============================================================
// Types & Interfaces
// ============================================================

export interface RANDataPoint {
  timestamp: number;
  cellId: string;
  dataType: 'CM' | 'PM' | 'FM';  // Configuration, Performance, Fault Management
  metrics: Record<string, number>;
  context?: Record<string, any>;
}

export interface PMCounters {
  // Uplink counters (3GPP TS 28.552)
  pmUlSinrMean?: number;
  pmUlBler?: number;
  pmUlRssi?: number;
  pmPuschPrbUsage?: number;
  // Downlink counters
  pmDlSinrMean?: number;
  pmDlBler?: number;
  pmPdschPrbUsage?: number;
  // Accessibility
  pmCssr?: number;
  pmErabSuccessRate?: number;
  pmRrcSetupSuccessRate?: number;
  // Retainability
  pmCallDropRate?: number;
  pmHoSuccessRate?: number;
  // Per-5QI
  pmPlrUrllc?: number;
  pmPlrEmbb?: number;
  pmPlrMiot?: number;
}

export interface CMParameters {
  p0NominalPUSCH?: number;
  alpha?: number;
  electricalTilt?: number;
  mechanicalTilt?: number;
  txPower?: number;
  crsGain?: number;
  beamWeights?: number[];
  ssbPeriodicity?: number;
}

export interface FMAlarm {
  alarmId: string;
  severity: 'CRITICAL' | 'MAJOR' | 'MINOR' | 'WARNING';
  category: string;
  description: string;
  raisedAt: number;
  clearedAt?: number;
}

export interface LearningEpisode {
  id: string;
  cellId: string;
  startTime: number;
  endTime: number;
  pmBefore: PMCounters;
  pmAfter: PMCounters;
  cmChange: Partial<CMParameters>;
  fmAlarms: FMAlarm[];
  outcome: 'SUCCESS' | 'FAILURE' | 'NEUTRAL';
  reward: number;
  embedding?: number[];
}

export interface LearningConfig {
  learningRate: number;
  discountFactor: number;  // gamma for RL
  explorationRate: number;  // epsilon for epsilon-greedy
  batchSize: number;
  memorySize: number;
  updateFrequency: number;  // milliseconds
}

// ============================================================
// Midstream Data Processor
// ============================================================

export class MidstreamProcessor extends EventEmitter {
  private buffer: RANDataPoint[] = [];
  private bufferSize: number = 1000;
  private flushInterval: number = 10000;  // 10 seconds
  private intervalId?: NodeJS.Timeout;

  constructor(config?: { bufferSize?: number; flushInterval?: number }) {
    super();
    this.bufferSize = config?.bufferSize || 1000;
    this.flushInterval = config?.flushInterval || 10000;
  }

  /**
   * Start the midstream processor
   */
  start(): void {
    console.log('[MIDSTREAM] Starting live data processor...');
    this.intervalId = setInterval(() => this.flush(), this.flushInterval);
    this.emit('started');
  }

  /**
   * Stop the processor
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    this.flush();
    this.emit('stopped');
  }

  /**
   * Ingest a data point from ENM/OSS
   */
  ingest(dataPoint: RANDataPoint): void {
    this.buffer.push(dataPoint);
    this.emit('data', dataPoint);

    // Emit real-time PM events
    if (dataPoint.dataType === 'PM') {
      this.emit('pm', dataPoint);
    } else if (dataPoint.dataType === 'FM') {
      this.emit('alarm', dataPoint);
    } else if (dataPoint.dataType === 'CM') {
      this.emit('config_change', dataPoint);
    }

    // Auto-flush if buffer full
    if (this.buffer.length >= this.bufferSize) {
      this.flush();
    }
  }

  /**
   * Flush buffer to persistent storage
   */
  private flush(): void {
    if (this.buffer.length === 0) return;

    const batch = [...this.buffer];
    this.buffer = [];

    this.emit('flush', batch);
    console.log(`[MIDSTREAM] Flushed ${batch.length} data points`);
  }

  /**
   * Calculate flow entropy for anomaly detection
   */
  calculateFlowEntropy(window: RANDataPoint[]): number {
    if (window.length === 0) return 0;

    // Count occurrences of each cell
    const counts: Record<string, number> = {};
    for (const dp of window) {
      counts[dp.cellId] = (counts[dp.cellId] || 0) + 1;
    }

    // Calculate Shannon entropy
    const total = window.length;
    let entropy = 0;
    for (const count of Object.values(counts)) {
      const p = count / total;
      if (p > 0) {
        entropy -= p * Math.log2(p);
      }
    }

    return entropy;
  }

  /**
   * Apply Dynamic Time Warping to align temporal patterns
   */
  applyDTW(seriesA: number[], seriesB: number[]): number {
    const n = seriesA.length;
    const m = seriesB.length;

    // DTW distance matrix
    const dtw: number[][] = Array(n + 1).fill(null)
      .map(() => Array(m + 1).fill(Infinity));
    dtw[0][0] = 0;

    for (let i = 1; i <= n; i++) {
      for (let j = 1; j <= m; j++) {
        const cost = Math.abs(seriesA[i - 1] - seriesB[j - 1]);
        dtw[i][j] = cost + Math.min(
          dtw[i - 1][j],      // insertion
          dtw[i][j - 1],      // deletion
          dtw[i - 1][j - 1]   // match
        );
      }
    }

    return dtw[n][m];
  }
}

// ============================================================
// Self-Learning Agent
// ============================================================

export class SelfLearningAgent extends EventEmitter {
  private config: LearningConfig;
  private midstream: MidstreamProcessor;
  private episodes: LearningEpisode[] = [];
  private qTable: Map<string, number[]> = new Map();
  private rewardHistory: number[] = [];

  constructor(config?: Partial<LearningConfig>) {
    super();
    this.config = {
      learningRate: config?.learningRate || 0.1,
      discountFactor: config?.discountFactor || 0.99,
      explorationRate: config?.explorationRate || 0.1,
      batchSize: config?.batchSize || 32,
      memorySize: config?.memorySize || 10000,
      updateFrequency: config?.updateFrequency || 60000
    };

    this.midstream = new MidstreamProcessor();
    this.setupMidstreamHandlers();
  }

  private setupMidstreamHandlers(): void {
    this.midstream.on('pm', (dp: RANDataPoint) => {
      this.processPMData(dp);
    });

    this.midstream.on('alarm', (dp: RANDataPoint) => {
      this.processFMAlarm(dp);
    });

    this.midstream.on('config_change', (dp: RANDataPoint) => {
      this.processCMChange(dp);
    });
  }

  /**
   * Start the self-learning agent
   */
  start(): void {
    console.log('[SELF-LEARNER] Starting self-learning agent...');
    console.log(`[SELF-LEARNER] Learning rate: ${this.config.learningRate}`);
    console.log(`[SELF-LEARNER] Exploration rate: ${this.config.explorationRate}`);

    this.midstream.start();
    this.emit('started');
  }

  /**
   * Stop the agent
   */
  stop(): void {
    this.midstream.stop();
    this.emit('stopped');
  }

  /**
   * Process PM counter data
   */
  private processPMData(dp: RANDataPoint): void {
    const pm = dp.metrics as unknown as PMCounters;

    // Detect anomalies
    if (pm.pmUlSinrMean !== undefined && pm.pmUlSinrMean < -10) {
      this.emit('anomaly', {
        type: 'LOW_SINR',
        cellId: dp.cellId,
        value: pm.pmUlSinrMean
      });
    }

    if (pm.pmCallDropRate !== undefined && pm.pmCallDropRate > 0.02) {
      this.emit('anomaly', {
        type: 'HIGH_DROP_RATE',
        cellId: dp.cellId,
        value: pm.pmCallDropRate
      });
    }
  }

  /**
   * Process FM alarm data
   */
  private processFMAlarm(dp: RANDataPoint): void {
    const alarm = dp.context as unknown as FMAlarm;

    if (alarm?.severity === 'CRITICAL') {
      this.emit('critical_alarm', {
        cellId: dp.cellId,
        alarm
      });
    }
  }

  /**
   * Process CM configuration change
   */
  private processCMChange(dp: RANDataPoint): void {
    // Log configuration change for correlation analysis
    this.emit('config_logged', {
      cellId: dp.cellId,
      params: dp.metrics
    });
  }

  /**
   * Calculate reward based on PM delta
   */
  calculateReward(pmBefore: PMCounters, pmAfter: PMCounters): number {
    let reward = 0;

    // SINR improvement (weight: 0.30)
    if (pmBefore.pmUlSinrMean !== undefined && pmAfter.pmUlSinrMean !== undefined) {
      const deltaSinr = pmAfter.pmUlSinrMean - pmBefore.pmUlSinrMean;
      reward += deltaSinr * 0.1 * 0.30;
    }

    // Accessibility (weight: 0.25)
    if (pmBefore.pmCssr !== undefined && pmAfter.pmCssr !== undefined) {
      const deltaCssr = pmAfter.pmCssr - pmBefore.pmCssr;
      reward += deltaCssr * 2.0 * 0.25;
    }

    // Retainability (weight: 0.20)
    if (pmBefore.pmCallDropRate !== undefined && pmAfter.pmCallDropRate !== undefined) {
      const deltaDrop = pmAfter.pmCallDropRate - pmBefore.pmCallDropRate;
      reward += -deltaDrop * 5.0 * 0.20;  // Negative because lower is better
    }

    // Spectral efficiency (weight: 0.15)
    if (pmBefore.pmPuschPrbUsage !== undefined && pmAfter.pmPuschPrbUsage !== undefined) {
      const deltaSe = pmAfter.pmPuschPrbUsage - pmBefore.pmPuschPrbUsage;
      reward += deltaSe * 0.5 * 0.15;
    }

    // Slice compliance (weight: 0.10)
    if (pmAfter.pmPlrUrllc !== undefined && pmAfter.pmPlrUrllc > 1e-5) {
      reward -= 1.0 * 0.10;  // Penalty for URLLC violation
    }

    return reward;
  }

  /**
   * Record a learning episode
   */
  recordEpisode(episode: LearningEpisode): void {
    // Calculate reward if not provided
    if (episode.reward === undefined) {
      episode.reward = this.calculateReward(episode.pmBefore, episode.pmAfter);
    }

    // Determine outcome based on reward
    if (episode.outcome === undefined) {
      episode.outcome = episode.reward > 0.1 ? 'SUCCESS' :
        episode.reward < -0.1 ? 'FAILURE' : 'NEUTRAL';
    }

    this.episodes.push(episode);
    this.rewardHistory.push(episode.reward);

    // Trim memory if exceeded
    if (this.episodes.length > this.config.memorySize) {
      this.episodes.shift();
    }

    console.log(`[SELF-LEARNER] Episode recorded: ${episode.id} (${episode.outcome}, reward: ${episode.reward.toFixed(3)})`);
    this.emit('episode_recorded', episode);

    // Trigger Q-learning update
    this.updateQTable(episode);
  }

  /**
   * Update Q-table using Q-learning
   */
  private updateQTable(episode: LearningEpisode): void {
    const state = this.getStateKey(episode.pmBefore);
    const action = this.getActionKey(episode.cmChange);
    const nextState = this.getStateKey(episode.pmAfter);

    // Initialize Q-values if not present
    if (!this.qTable.has(state)) {
      this.qTable.set(state, new Array(10).fill(0));  // 10 possible actions
    }
    if (!this.qTable.has(nextState)) {
      this.qTable.set(nextState, new Array(10).fill(0));
    }

    const qValues = this.qTable.get(state)!;
    const nextQValues = this.qTable.get(nextState)!;

    // Q-learning update
    const actionIdx = this.actionToIndex(action);
    const maxNextQ = Math.max(...nextQValues);
    const currentQ = qValues[actionIdx];

    qValues[actionIdx] = currentQ + this.config.learningRate *
      (episode.reward + this.config.discountFactor * maxNextQ - currentQ);

    this.qTable.set(state, qValues);
  }

  private getStateKey(pm: PMCounters): string {
    // Discretize PM counters into state bins
    const sinrBin = Math.floor((pm.pmUlSinrMean || 0) / 5);
    const blerBin = Math.floor((pm.pmUlBler || 0) * 10);
    return `s_${sinrBin}_${blerBin}`;
  }

  private getActionKey(cm: Partial<CMParameters>): string {
    // Encode CM change as action
    const tiltChange = cm.electricalTilt ? 'tilt' : '';
    const powerChange = cm.txPower ? 'power' : '';
    return `a_${tiltChange}_${powerChange}`;
  }

  private actionToIndex(action: string): number {
    // Map action string to index (0-9)
    const actions = ['none', 'tilt_up', 'tilt_down', 'power_up', 'power_down',
      'tilt_power_up', 'tilt_power_down', 'alpha_up', 'alpha_down', 'combo'];
    return Math.max(0, actions.indexOf(action.replace('a_', '')));
  }

  /**
   * Get recommended action for a given state (epsilon-greedy)
   */
  getRecommendedAction(pm: PMCounters): { action: string; confidence: number } {
    const state = this.getStateKey(pm);

    // Exploration: random action
    if (Math.random() < this.config.explorationRate) {
      return { action: 'explore', confidence: 0.5 };
    }

    // Exploitation: best Q-value action
    const qValues = this.qTable.get(state);
    if (!qValues) {
      return { action: 'none', confidence: 0.1 };
    }

    const maxQ = Math.max(...qValues);
    const bestAction = qValues.indexOf(maxQ);
    const actions = ['none', 'tilt_up', 'tilt_down', 'power_up', 'power_down',
      'tilt_power_up', 'tilt_power_down', 'alpha_up', 'alpha_down', 'combo'];

    return {
      action: actions[bestAction],
      confidence: Math.min(1, maxQ + 0.5)
    };
  }

  /**
   * Get learning statistics
   */
  getStats(): {
    episodeCount: number;
    avgReward: number;
    successRate: number;
    qTableSize: number;
  } {
    const successCount = this.episodes.filter(e => e.outcome === 'SUCCESS').length;
    const avgReward = this.rewardHistory.length > 0 ?
      this.rewardHistory.reduce((a, b) => a + b, 0) / this.rewardHistory.length : 0;

    return {
      episodeCount: this.episodes.length,
      avgReward,
      successRate: this.episodes.length > 0 ? successCount / this.episodes.length : 0,
      qTableSize: this.qTable.size
    };
  }
}

// ============================================================
// Ruvector Integration for Spatial Learning
// ============================================================

export class SpatialLearner extends EventEmitter {
  private cellEmbeddings: Map<string, number[]> = new Map();
  private neighborGraph: Map<string, Set<string>> = new Map();
  private dimension: number = 768;

  /**
   * Update cell embedding based on PM data
   */
  updateCellEmbedding(cellId: string, pm: PMCounters): void {
    // Create embedding from PM counters
    const embedding = this.pmToEmbedding(pm);
    this.cellEmbeddings.set(cellId, embedding);
    this.emit('embedding_updated', { cellId, embedding });
  }

  private pmToEmbedding(pm: PMCounters): number[] {
    // Convert PM counters to fixed-dimension embedding
    const values = [
      pm.pmUlSinrMean || 0,
      pm.pmDlSinrMean || 0,
      pm.pmUlBler || 0,
      pm.pmDlBler || 0,
      pm.pmCssr || 0,
      pm.pmCallDropRate || 0,
      pm.pmHoSuccessRate || 0,
      pm.pmPuschPrbUsage || 0
    ];

    // Pad to dimension size
    const embedding = new Array(this.dimension).fill(0);
    for (let i = 0; i < values.length; i++) {
      embedding[i] = values[i];
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, v) => sum + v * v, 0));
    if (norm > 0) {
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] /= norm;
      }
    }

    return embedding;
  }

  /**
   * Find similar cells based on embeddings
   */
  findSimilarCells(cellId: string, k: number = 5): { cellId: string; similarity: number }[] {
    const queryEmbedding = this.cellEmbeddings.get(cellId);
    if (!queryEmbedding) return [];

    const similarities: { cellId: string; similarity: number }[] = [];

    for (const [id, embedding] of this.cellEmbeddings) {
      if (id === cellId) continue;

      const similarity = this.cosineSimilarity(queryEmbedding, embedding);
      similarities.push({ cellId: id, similarity });
    }

    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Build neighbor graph from interference data
   */
  buildNeighborGraph(interferenceMatrix: { cell1: string; cell2: string; rsrp: number }[]): void {
    for (const { cell1, cell2 } of interferenceMatrix) {
      if (!this.neighborGraph.has(cell1)) {
        this.neighborGraph.set(cell1, new Set());
      }
      if (!this.neighborGraph.has(cell2)) {
        this.neighborGraph.set(cell2, new Set());
      }

      this.neighborGraph.get(cell1)!.add(cell2);
      this.neighborGraph.get(cell2)!.add(cell1);
    }

    this.emit('graph_updated', { nodeCount: this.neighborGraph.size });
  }

  /**
   * Get interference cluster for a cell
   */
  getInterferenceCluster(cellId: string, depth: number = 2): Set<string> {
    const cluster = new Set<string>();
    const queue: { id: string; level: number }[] = [{ id: cellId, level: 0 }];

    while (queue.length > 0) {
      const { id, level } = queue.shift()!;

      if (cluster.has(id) || level > depth) continue;
      cluster.add(id);

      const neighbors = this.neighborGraph.get(id);
      if (neighbors) {
        for (const neighbor of neighbors) {
          queue.push({ id: neighbor, level: level + 1 });
        }
      }
    }

    return cluster;
  }
}

// ============================================================
// Factory & Exports
// ============================================================

export function createSelfLearningPipeline(config?: Partial<LearningConfig>): {
  learner: SelfLearningAgent;
  spatial: SpatialLearner;
  midstream: MidstreamProcessor;
} {
  const learner = new SelfLearningAgent(config);
  const spatial = new SpatialLearner();

  // Connect spatial learning to main learner
  learner.on('episode_recorded', (episode: LearningEpisode) => {
    spatial.updateCellEmbedding(episode.cellId, episode.pmAfter);
  });

  return {
    learner,
    spatial,
    midstream: (learner as any).midstream
  };
}

// export { MidstreamProcessor, SelfLearningAgent, SpatialLearner };
