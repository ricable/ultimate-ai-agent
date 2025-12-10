/**
 * RuvSwarm - Swarm Orchestration Engine
 *
 * Ruv-swarm v1.0.20 binds individual RAN agents into a cohesive system with:
 * - Neural Routing (tiny-dancer): Routes tasks based on complexity
 * - Federated Learning: Privacy-preserving model aggregation (FedAvg)
 * - Leader Election: Bully algorithm for cluster coordination
 * - Pattern Propagation: Share successful optimization patterns across swarm
 * - Strange Loops Consensus: Blockchain-inspired model provenance tracking
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import {
  NanoAgentId,
  AgentState,
  ClusterInfo,
  SwarmMessage,
  SwarmMessageType,
  FederatedModelUpdate,
  ModelProvenanceRecord,
  Action,
  ThoughtTrajectory
} from '../../types/index.js';

// ============================================================================
// NEURAL ROUTER (TINY-DANCER)
// ============================================================================

export type TaskComplexity = 'trivial' | 'simple' | 'moderate' | 'complex' | 'critical';

export interface RoutingDecision {
  handler: 'rule' | 'lightweight' | 'reasoning' | 'gnn' | 'ensemble';
  confidence: number;
  estimatedLatencyMs: number;
}

class NeuralRouter {
  // Simple neural net for routing decisions
  private weights: Float32Array;
  private inputSize: number = 16;
  private hiddenSize: number = 32;
  private outputSize: number = 5; // 5 handler types

  constructor() {
    // Initialize weights
    const totalWeights = this.inputSize * this.hiddenSize + this.hiddenSize * this.outputSize;
    this.weights = new Float32Array(totalWeights);
    for (let i = 0; i < totalWeights; i++) {
      this.weights[i] = (Math.random() - 0.5) * 0.5;
    }
  }

  /**
   * Route a task to the appropriate handler based on complexity
   */
  route(taskFeatures: {
    anomalyScore: number;
    affectedCells: number;
    correlatedMetrics: number;
    historicalSimilarity: number;
    urgency: number;
    currentLoad: number;
  }): RoutingDecision {
    // Encode features
    const input = new Float32Array(this.inputSize);
    input[0] = taskFeatures.anomalyScore;
    input[1] = taskFeatures.affectedCells / 10;
    input[2] = taskFeatures.correlatedMetrics / 5;
    input[3] = taskFeatures.historicalSimilarity;
    input[4] = taskFeatures.urgency;
    input[5] = taskFeatures.currentLoad;

    // Forward pass
    const hidden = new Float32Array(this.hiddenSize);
    for (let h = 0; h < this.hiddenSize; h++) {
      let sum = 0;
      for (let i = 0; i < this.inputSize; i++) {
        sum += input[i] * this.weights[i * this.hiddenSize + h];
      }
      hidden[h] = Math.max(0, sum); // ReLU
    }

    const output = new Float32Array(this.outputSize);
    const hiddenOffset = this.inputSize * this.hiddenSize;
    for (let o = 0; o < this.outputSize; o++) {
      for (let h = 0; h < this.hiddenSize; h++) {
        output[o] += hidden[h] * this.weights[hiddenOffset + h * this.outputSize + o];
      }
    }

    // Softmax
    let maxVal = -Infinity;
    for (const v of output) maxVal = Math.max(maxVal, v);
    let sumExp = 0;
    for (let i = 0; i < output.length; i++) {
      output[i] = Math.exp(output[i] - maxVal);
      sumExp += output[i];
    }
    for (let i = 0; i < output.length; i++) {
      output[i] /= sumExp;
    }

    // Find winner
    let maxIdx = 0;
    for (let i = 1; i < output.length; i++) {
      if (output[i] > output[maxIdx]) maxIdx = i;
    }

    const handlers: RoutingDecision['handler'][] = ['rule', 'lightweight', 'reasoning', 'gnn', 'ensemble'];
    const latencies = [5, 20, 100, 200, 500];

    return {
      handler: handlers[maxIdx],
      confidence: output[maxIdx],
      estimatedLatencyMs: latencies[maxIdx]
    };
  }

  /**
   * Update router based on outcome feedback
   */
  feedback(decision: RoutingDecision, actualLatency: number, success: boolean): void {
    // Simplified online learning - would be full backprop in production
    const expectedIdx = ['rule', 'lightweight', 'reasoning', 'gnn', 'ensemble'].indexOf(decision.handler);
    const reward = success ? 1 : -1;

    // Small weight update in direction of reward
    const lr = 0.001 * reward;
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] += lr * (Math.random() - 0.5);
    }
  }
}

// ============================================================================
// FEDERATED LEARNING ENGINE
// ============================================================================

export type AggregationAlgorithm = 'FedAvg' | 'FedProx' | 'FedNova' | 'FedOpt';

interface FederatedLearningConfig {
  algorithm: AggregationAlgorithm;
  minClientsPerRound: number;
  roundIntervalMs: number;
  maxRoundsWithoutImprovement: number;
  differentialPrivacy: {
    enabled: boolean;
    epsilon: number;
    delta: number;
  };
  compression: {
    enabled: boolean;
    method: 'topK' | 'randomK' | 'quantization';
    ratio: number;
  };
}

class FederatedLearningEngine extends EventEmitter {
  config: FederatedLearningConfig;
  currentRound: number = 0;
  globalModel: Float32Array | null = null;

  // Track client contributions
  private clientContributions: Map<NanoAgentId, number> = new Map();
  private roundUpdates: Map<NanoAgentId, FederatedModelUpdate> = new Map();

  constructor(config: FederatedLearningConfig) {
    super();
    this.config = config;
  }

  /**
   * Submit local model update from an agent
   */
  submitUpdate(update: FederatedModelUpdate): void {
    this.roundUpdates.set(update.sourceAgent, update);
    this.clientContributions.set(
      update.sourceAgent,
      (this.clientContributions.get(update.sourceAgent) || 0) + 1
    );

    this.emit('update-received', {
      agent: update.sourceAgent,
      round: update.round,
      totalUpdates: this.roundUpdates.size
    });

    // Check if we have enough updates to aggregate
    if (this.roundUpdates.size >= this.config.minClientsPerRound) {
      this.aggregateRound();
    }
  }

  /**
   * Aggregate updates using FedAvg or other algorithms
   */
  private aggregateRound(): void {
    const updates = Array.from(this.roundUpdates.values());
    if (updates.length === 0) return;

    const modelSize = updates[0].weights?.length || updates[0].gradients?.length || 0;
    if (modelSize === 0) return;

    let aggregatedWeights: Float32Array;

    switch (this.config.algorithm) {
      case 'FedAvg':
        aggregatedWeights = this.fedAvg(updates, modelSize);
        break;
      case 'FedProx':
        aggregatedWeights = this.fedProx(updates, modelSize);
        break;
      case 'FedNova':
        aggregatedWeights = this.fedNova(updates, modelSize);
        break;
      default:
        aggregatedWeights = this.fedAvg(updates, modelSize);
    }

    // Apply differential privacy if enabled
    if (this.config.differentialPrivacy.enabled) {
      this.addDifferentialPrivacyNoise(aggregatedWeights);
    }

    // Apply compression if enabled
    if (this.config.compression.enabled) {
      aggregatedWeights = this.compressWeights(aggregatedWeights);
    }

    this.globalModel = aggregatedWeights;
    this.currentRound++;
    this.roundUpdates.clear();

    this.emit('round-complete', {
      round: this.currentRound,
      participantCount: updates.length,
      modelChecksum: this.computeChecksum(aggregatedWeights)
    });
  }

  /**
   * FedAvg: Weighted average by sample count
   */
  private fedAvg(updates: FederatedModelUpdate[], modelSize: number): Float32Array {
    const result = new Float32Array(modelSize);
    let totalSamples = 0;

    for (const update of updates) {
      totalSamples += update.sampleCount;
    }

    for (const update of updates) {
      const weight = update.sampleCount / totalSamples;
      const updateWeights = update.weights || update.gradients!;

      for (let i = 0; i < modelSize; i++) {
        result[i] += updateWeights[i] * weight;
      }
    }

    return result;
  }

  /**
   * FedProx: FedAvg with proximal term for heterogeneity
   */
  private fedProx(updates: FederatedModelUpdate[], modelSize: number): Float32Array {
    const result = this.fedAvg(updates, modelSize);

    // Add proximal regularization towards global model
    if (this.globalModel) {
      const mu = 0.01; // Proximal coefficient
      for (let i = 0; i < modelSize; i++) {
        result[i] = (1 - mu) * result[i] + mu * this.globalModel[i];
      }
    }

    return result;
  }

  /**
   * FedNova: Normalized averaging accounting for local steps
   */
  private fedNova(updates: FederatedModelUpdate[], modelSize: number): Float32Array {
    const result = new Float32Array(modelSize);

    // Assume equal local steps for simplicity
    // In production, track actual local epochs per client
    let totalWeight = 0;

    for (const update of updates) {
      const effectiveDataRatio = update.sampleCount;
      totalWeight += effectiveDataRatio;
    }

    for (const update of updates) {
      const weight = update.sampleCount / totalWeight;
      const updateWeights = update.weights || update.gradients!;

      for (let i = 0; i < modelSize; i++) {
        result[i] += updateWeights[i] * weight;
      }
    }

    return result;
  }

  /**
   * Add Gaussian noise for differential privacy
   */
  private addDifferentialPrivacyNoise(weights: Float32Array): void {
    const { epsilon, delta } = this.config.differentialPrivacy;

    // Compute noise scale using Gaussian mechanism
    // sigma = sqrt(2 * ln(1.25/delta)) / epsilon
    const sigma = Math.sqrt(2 * Math.log(1.25 / delta)) / epsilon;

    for (let i = 0; i < weights.length; i++) {
      // Box-Muller for Gaussian noise
      const u1 = Math.random();
      const u2 = Math.random();
      const noise = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2) * sigma;
      weights[i] += noise;
    }
  }

  /**
   * Compress weights for efficient transmission
   */
  private compressWeights(weights: Float32Array): Float32Array {
    switch (this.config.compression.method) {
      case 'topK':
        return this.topKCompression(weights);
      case 'quantization':
        return this.quantize(weights);
      default:
        return weights;
    }
  }

  private topKCompression(weights: Float32Array): Float32Array {
    const k = Math.floor(weights.length * this.config.compression.ratio);
    const indexed = Array.from(weights).map((v, i) => ({ v: Math.abs(v), i, orig: v }));
    indexed.sort((a, b) => b.v - a.v);

    const result = new Float32Array(weights.length);
    for (let i = 0; i < k; i++) {
      result[indexed[i].i] = indexed[i].orig;
    }
    return result;
  }

  private quantize(weights: Float32Array): Float32Array {
    // 8-bit quantization
    let min = Infinity, max = -Infinity;
    for (const w of weights) {
      min = Math.min(min, w);
      max = Math.max(max, w);
    }

    const scale = (max - min) / 255;
    const result = new Float32Array(weights.length);

    for (let i = 0; i < weights.length; i++) {
      const quantized = Math.round((weights[i] - min) / scale);
      result[i] = quantized * scale + min;
    }

    return result;
  }

  private computeChecksum(weights: Float32Array): string {
    let hash = 0;
    for (const w of weights) {
      hash = ((hash << 5) - hash) + Math.floor(w * 1000);
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }

  /**
   * Get global model for distribution
   */
  getGlobalModel(): Float32Array | null {
    return this.globalModel;
  }

  getStats(): {
    currentRound: number;
    totalContributors: number;
    pendingUpdates: number;
  } {
    return {
      currentRound: this.currentRound,
      totalContributors: this.clientContributions.size,
      pendingUpdates: this.roundUpdates.size
    };
  }
}

// ============================================================================
// LEADER ELECTION (BULLY ALGORITHM)
// ============================================================================

class LeaderElection extends EventEmitter {
  private agentId: NanoAgentId;
  private priority: number;
  private currentLeader: NanoAgentId | null = null;
  private electionInProgress: boolean = false;
  private peers: Map<NanoAgentId, number> = new Map(); // id -> priority

  constructor(agentId: NanoAgentId, priority: number) {
    super();
    this.agentId = agentId;
    this.priority = priority;
  }

  /**
   * Register a peer agent
   */
  registerPeer(peerId: NanoAgentId, peerPriority: number): void {
    this.peers.set(peerId, peerPriority);
  }

  /**
   * Remove a peer (e.g., on failure)
   */
  removePeer(peerId: NanoAgentId): void {
    this.peers.delete(peerId);

    // If leader failed, trigger election
    if (this.currentLeader === peerId) {
      this.startElection();
    }
  }

  /**
   * Start a leader election
   */
  startElection(): void {
    if (this.electionInProgress) return;

    this.electionInProgress = true;
    this.emit('election-started', { initiator: this.agentId });

    // Find all peers with higher priority
    const higherPeers: NanoAgentId[] = [];
    for (const [peerId, peerPriority] of this.peers) {
      if (peerPriority > this.priority) {
        higherPeers.push(peerId);
      }
    }

    if (higherPeers.length === 0) {
      // No higher priority peers, we become leader
      this.declareLeader(this.agentId);
    } else {
      // Wait for response from higher priority peers
      // In real impl, send ELECTION messages and wait for OK
      // For simplicity, just pick the highest
      let highestPeer = this.agentId;
      let highestPriority = this.priority;

      for (const [peerId, peerPriority] of this.peers) {
        if (peerPriority > highestPriority) {
          highestPeer = peerId;
          highestPriority = peerPriority;
        }
      }

      this.declareLeader(highestPeer);
    }
  }

  /**
   * Declare a new leader
   */
  private declareLeader(leaderId: NanoAgentId): void {
    this.currentLeader = leaderId;
    this.electionInProgress = false;

    this.emit('leader-elected', {
      leader: leaderId,
      isLocal: leaderId === this.agentId
    });
  }

  /**
   * Get current leader
   */
  getLeader(): NanoAgentId | null {
    return this.currentLeader;
  }

  /**
   * Check if this agent is the leader
   */
  isLeader(): boolean {
    return this.currentLeader === this.agentId;
  }
}

// ============================================================================
// PATTERN PROPAGATION ENGINE
// ============================================================================

interface OptimizationPattern {
  id: string;
  sourceAgent: NanoAgentId;
  symptom: string;
  actions: Action[];
  outcome: {
    success: boolean;
    improvement: Record<string, number>;
  };
  applicabilityScore: number;
  propagationCount: number;
  signatures: string[];
  timestamp: Date;
}

class PatternPropagation extends EventEmitter {
  private patterns: Map<string, OptimizationPattern> = new Map();
  private appliedPatterns: Set<string> = new Set();

  /**
   * Publish a successful optimization pattern
   */
  publishPattern(trajectory: ThoughtTrajectory, agentId: NanoAgentId): string {
    if (!trajectory.outcome.success) {
      throw new Error('Cannot publish failed pattern');
    }

    const patternId = uuidv4();
    const pattern: OptimizationPattern = {
      id: patternId,
      sourceAgent: agentId,
      symptom: trajectory.symptom,
      actions: trajectory.actionSequence,
      outcome: {
        success: trajectory.outcome.success,
        improvement: trajectory.outcome.deltaKPIs
      },
      applicabilityScore: this.computeApplicability(trajectory),
      propagationCount: 0,
      signatures: [this.signPattern(trajectory, agentId)],
      timestamp: new Date()
    };

    this.patterns.set(patternId, pattern);
    this.emit('pattern-published', { patternId, sourceAgent: agentId });

    return patternId;
  }

  /**
   * Find applicable patterns for a given symptom
   */
  findPatterns(symptom: string, context: any): OptimizationPattern[] {
    const matches: Array<{ pattern: OptimizationPattern; score: number }> = [];

    for (const pattern of this.patterns.values()) {
      const similarity = this.computeSimilarity(symptom, pattern.symptom);
      if (similarity > 0.5) {
        matches.push({
          pattern,
          score: similarity * pattern.applicabilityScore
        });
      }
    }

    matches.sort((a, b) => b.score - a.score);
    return matches.slice(0, 5).map(m => m.pattern);
  }

  /**
   * Record pattern application
   */
  recordApplication(patternId: string, success: boolean): void {
    const pattern = this.patterns.get(patternId);
    if (pattern) {
      pattern.propagationCount++;
      this.appliedPatterns.add(patternId);

      // Update applicability based on success
      const alpha = 0.1;
      pattern.applicabilityScore = pattern.applicabilityScore * (1 - alpha) +
        (success ? 1 : 0) * alpha;

      this.emit('pattern-applied', { patternId, success });
    }
  }

  /**
   * Verify pattern signature (blockchain-inspired provenance)
   */
  verifyPattern(pattern: OptimizationPattern): boolean {
    // Simplified verification - in production use actual crypto
    return pattern.signatures.length > 0 &&
           pattern.signatures[0].length > 0;
  }

  private computeApplicability(trajectory: ThoughtTrajectory): number {
    // Higher score for larger improvements
    let score = 0;
    for (const delta of Object.values(trajectory.outcome.deltaKPIs)) {
      score += Math.max(0, delta);
    }
    return Math.min(1, score / 10);
  }

  private computeSimilarity(s1: string, s2: string): number {
    const tokens1 = new Set(s1.toLowerCase().split(/\s+/));
    const tokens2 = new Set(s2.toLowerCase().split(/\s+/));

    const intersection = new Set([...tokens1].filter(t => tokens2.has(t)));
    const union = new Set([...tokens1, ...tokens2]);

    return intersection.size / union.size;
  }

  private signPattern(trajectory: ThoughtTrajectory, agentId: NanoAgentId): string {
    // Simplified signing - in production use Ed25519
    const data = JSON.stringify({
      symptom: trajectory.symptom,
      actions: trajectory.actionSequence.map(a => a.type),
      agentId
    });

    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      hash = ((hash << 5) - hash) + data.charCodeAt(i);
      hash = hash & hash;
    }
    return `sig-${agentId}-${Math.abs(hash).toString(16)}`;
  }

  getStats(): {
    totalPatterns: number;
    appliedPatterns: number;
    avgApplicability: number;
  } {
    let totalApp = 0;
    for (const p of this.patterns.values()) {
      totalApp += p.applicabilityScore;
    }

    return {
      totalPatterns: this.patterns.size,
      appliedPatterns: this.appliedPatterns.size,
      avgApplicability: this.patterns.size > 0 ? totalApp / this.patterns.size : 0
    };
  }
}

// ============================================================================
// MAIN SWARM ORCHESTRATOR
// ============================================================================

export interface SwarmOrchestratorConfig {
  agentId: NanoAgentId;
  clusterId: string;
  priority: number;
  federatedLearning: FederatedLearningConfig;
  heartbeatIntervalMs: number;
  leaderElectionTimeoutMs: number;
}

const DEFAULT_CONFIG: SwarmOrchestratorConfig = {
  agentId: 'default-agent',
  clusterId: 'default-cluster',
  priority: Math.random(),
  federatedLearning: {
    algorithm: 'FedAvg',
    minClientsPerRound: 3,
    roundIntervalMs: 300000,
    maxRoundsWithoutImprovement: 10,
    differentialPrivacy: {
      enabled: true,
      epsilon: 1.0,
      delta: 1e-5
    },
    compression: {
      enabled: true,
      method: 'topK',
      ratio: 0.1
    }
  },
  heartbeatIntervalMs: 1000,
  leaderElectionTimeoutMs: 5000
};

export class SwarmOrchestrator extends EventEmitter {
  config: SwarmOrchestratorConfig;
  router: NeuralRouter;
  federation: FederatedLearningEngine;
  election: LeaderElection;
  patterns: PatternPropagation;

  // Agent state
  private state: AgentState;
  private peers: Map<NanoAgentId, AgentState> = new Map();
  private messageQueue: SwarmMessage[] = [];

  // Provenance tracking
  private provenanceRecords: ModelProvenanceRecord[] = [];

  constructor(config: Partial<SwarmOrchestratorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.router = new NeuralRouter();
    this.federation = new FederatedLearningEngine(this.config.federatedLearning);
    this.election = new LeaderElection(this.config.agentId, this.config.priority);
    this.patterns = new PatternPropagation();

    // Initialize agent state
    this.state = {
      id: this.config.agentId,
      cgi: {
        mcc: '001',
        mnc: '01',
        gnbId: 1,
        cellId: 1,
        sectorId: 0
      },
      status: 'active',
      neighbors: [],
      localModel: {
        version: 0,
        lastTrained: new Date(),
        metrics: {}
      },
      reasoningStats: {
        trajectoriesStored: 0,
        queriesServed: 0,
        patternsLearned: 0
      },
      optimizationStats: {
        actionsExecuted: 0,
        actionsBlocked: 0,
        avgReward: 0
      },
      lastHeartbeat: new Date()
    };

    // Wire up events
    this.federation.on('round-complete', (data) => {
      this.recordProvenance(data);
      this.emit('federation-round', data);
    });

    this.election.on('leader-elected', (data) => {
      if (data.isLocal) {
        this.emit('became-leader', { clusterId: this.config.clusterId });
      }
    });

    this.patterns.on('pattern-published', (data) => {
      this.emit('pattern-available', data);
    });
  }

  /**
   * Route a task to appropriate handler
   */
  routeTask(taskFeatures: Parameters<NeuralRouter['route']>[0]): RoutingDecision {
    return this.router.route(taskFeatures);
  }

  /**
   * Submit local model update for federation
   */
  submitModelUpdate(weights: Float32Array, sampleCount: number, metrics: { loss: number }): void {
    const update: FederatedModelUpdate = {
      modelId: `model-${this.config.agentId}-${Date.now()}`,
      modelType: 'uplinkOptimizer',
      sourceAgent: this.config.agentId,
      round: this.federation.currentRound,
      weights: Array.from(weights),
      sampleCount,
      metrics
    };

    this.federation.submitUpdate(update);
    this.state.localModel.version++;
  }

  /**
   * Get aggregated global model
   */
  getGlobalModel(): Float32Array | null {
    return this.federation.getGlobalModel();
  }

  /**
   * Register peer agent
   */
  registerPeer(peer: AgentState): void {
    this.peers.set(peer.id, peer);
    this.state.neighbors.push(peer.id);
    this.election.registerPeer(peer.id, Math.random());
  }

  /**
   * Handle peer failure
   */
  handlePeerFailure(peerId: NanoAgentId): void {
    this.peers.delete(peerId);
    this.state.neighbors = this.state.neighbors.filter(n => n !== peerId);
    this.election.removePeer(peerId);
    this.emit('peer-failed', { peerId });
  }

  /**
   * Send message to peer or broadcast
   */
  sendMessage(
    type: SwarmMessageType,
    payload: any,
    targetAgent?: NanoAgentId
  ): SwarmMessage {
    const message: SwarmMessage = {
      id: uuidv4(),
      type,
      sourceAgent: this.config.agentId,
      targetAgent,
      timestamp: new Date(),
      ttl: 3,
      signature: this.signMessage(type, payload),
      payload
    };

    this.messageQueue.push(message);
    this.emit('message-sent', { messageId: message.id, type, target: targetAgent || 'broadcast' });

    return message;
  }

  /**
   * Process incoming message
   */
  processMessage(message: SwarmMessage): void {
    if (!this.verifySignature(message)) {
      this.emit('message-rejected', { messageId: message.id, reason: 'invalid signature' });
      return;
    }

    switch (message.type) {
      case 'HEARTBEAT':
        this.handleHeartbeat(message);
        break;
      case 'MODEL_UPDATE':
        this.handleModelUpdate(message);
        break;
      case 'PATTERN_SHARE':
        this.handlePatternShare(message);
        break;
      case 'LEADER_ELECTION':
        this.handleLeaderElection(message);
        break;
      case 'CONFLICT_RESOLUTION':
        this.handleConflictResolution(message);
        break;
      default:
        this.emit('unknown-message', { type: message.type });
    }
  }

  /**
   * Publish successful optimization pattern
   */
  publishPattern(trajectory: ThoughtTrajectory): string {
    const patternId = this.patterns.publishPattern(trajectory, this.config.agentId);

    // Broadcast to peers
    this.sendMessage('PATTERN_SHARE', {
      patternId,
      trajectory
    });

    this.state.reasoningStats.patternsLearned++;
    return patternId;
  }

  /**
   * Find patterns for a symptom
   */
  findPatterns(symptom: string, context: any): OptimizationPattern[] {
    return this.patterns.findPatterns(symptom, context);
  }

  /**
   * Start leader election
   */
  startElection(): void {
    this.election.startElection();
  }

  /**
   * Check if this agent is the cluster leader
   */
  isLeader(): boolean {
    return this.election.isLeader();
  }

  /**
   * Get current leader
   */
  getLeader(): NanoAgentId | null {
    return this.election.getLeader();
  }

  /**
   * Get agent state
   */
  getState(): AgentState {
    return { ...this.state };
  }

  /**
   * Update agent state
   */
  updateState(updates: Partial<AgentState>): void {
    this.state = { ...this.state, ...updates, lastHeartbeat: new Date() };
  }

  private handleHeartbeat(message: SwarmMessage): void {
    const peer = this.peers.get(message.sourceAgent);
    if (peer) {
      peer.lastHeartbeat = new Date();
      peer.status = message.payload.status || 'active';
    }
  }

  private handleModelUpdate(message: SwarmMessage): void {
    // Only leader processes model updates
    if (this.isLeader() && message.payload.update) {
      this.federation.submitUpdate(message.payload.update);
    }
  }

  private handlePatternShare(message: SwarmMessage): void {
    // Verify and store shared pattern
    if (message.payload.trajectory) {
      const trajectory = message.payload.trajectory as ThoughtTrajectory;
      if (trajectory.outcome.success) {
        this.patterns.publishPattern(trajectory, message.sourceAgent);
      }
    }
  }

  private handleLeaderElection(message: SwarmMessage): void {
    if (message.payload.action === 'start') {
      this.election.startElection();
    }
  }

  private handleConflictResolution(message: SwarmMessage): void {
    // Handle optimization conflicts (e.g., power wars)
    this.emit('conflict-detected', {
      type: message.payload.conflictType,
      involvedAgents: message.payload.agents
    });
  }

  private recordProvenance(data: { round: number; modelChecksum: string }): void {
    const record: ModelProvenanceRecord = {
      modelId: `global-${this.config.clusterId}-r${data.round}`,
      version: data.round,
      contributors: Array.from(this.peers.keys()),
      trainingRound: data.round,
      hash: data.modelChecksum,
      signature: this.signProvenance(data),
      timestamp: new Date()
    };

    this.provenanceRecords.push(record);

    // Keep last 100 records
    if (this.provenanceRecords.length > 100) {
      this.provenanceRecords.shift();
    }
  }

  private signMessage(type: SwarmMessageType, payload: any): string {
    const data = JSON.stringify({ type, payload, agent: this.config.agentId });
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      hash = ((hash << 5) - hash) + data.charCodeAt(i);
      hash = hash & hash;
    }
    return `msg-${Math.abs(hash).toString(16)}`;
  }

  private verifySignature(message: SwarmMessage): boolean {
    // Simplified verification
    return message.signature && message.signature.startsWith('msg-');
  }

  private signProvenance(data: any): string {
    const str = JSON.stringify(data);
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = ((hash << 5) - hash) + str.charCodeAt(i);
      hash = hash & hash;
    }
    return `prov-${Math.abs(hash).toString(16)}`;
  }

  getStats(): {
    agentId: NanoAgentId;
    peerCount: number;
    isLeader: boolean;
    federationStats: ReturnType<FederatedLearningEngine['getStats']>;
    patternStats: ReturnType<PatternPropagation['getStats']>;
    messageQueueSize: number;
    provenanceRecords: number;
  } {
    return {
      agentId: this.config.agentId,
      peerCount: this.peers.size,
      isLeader: this.isLeader(),
      federationStats: this.federation.getStats(),
      patternStats: this.patterns.getStats(),
      messageQueueSize: this.messageQueue.length,
      provenanceRecords: this.provenanceRecords.length
    };
  }
}
