/**
 * AIDefence - Security Layer for RAN Optimization
 *
 * Provides rigorous security for autonomous AI in critical infrastructure:
 * - Adversarial Input Detection
 * - Physical Consistency Validation
 * - Safety Guardrails Enforcement
 * - Model Provenance Tracking
 * - Neuro-symbolic Logic Validation
 *
 * Defense against:
 * - Adversarial examples (poisoning, evasion)
 * - Fake UE measurement reports
 * - Model poisoning in federated learning
 * - Parameter manipulation attacks
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import {
  CellKPIs,
  Action,
  ActionRecord,
  AdversarialDetectionResult,
  GuardrailViolation,
  ModelProvenanceRecord,
  FederatedModelUpdate,
  NanoAgentId
} from '../types/index.js';

// ============================================================================
// ADVERSARIAL DETECTOR
// ============================================================================

interface AdversarialDetectorConfig {
  sensitivityThreshold: number;
  maxDeviationSigma: number;
  enablePhysicsValidation: boolean;
}

class AdversarialDetector {
  config: AdversarialDetectorConfig;
  private baselineStats: Map<string, { mean: number; std: number }> = new Map();

  constructor(config: Partial<AdversarialDetectorConfig> = {}) {
    this.config = {
      sensitivityThreshold: 0.7,
      maxDeviationSigma: 4,
      enablePhysicsValidation: true,
      ...config
    };

    this.initializeBaselines();
  }

  private initializeBaselines(): void {
    // Expected ranges for common metrics based on physics
    this.baselineStats.set('rsrp', { mean: -90, std: 15 });
    this.baselineStats.set('rsrq', { mean: -12, std: 5 });
    this.baselineStats.set('sinr', { mean: 10, std: 8 });
    this.baselineStats.set('pathloss', { mean: 100, std: 20 });
    this.baselineStats.set('timing_advance', { mean: 10, std: 5 });
    this.baselineStats.set('power', { mean: 23, std: 10 });
  }

  /**
   * Detect adversarial inputs in measurement reports
   */
  detectAdversarial(
    input: Record<string, number>,
    context: { cellId: string; timestamp: Date }
  ): AdversarialDetectionResult {
    const anomalyScores: number[] = [];
    const violations: string[] = [];

    // Statistical anomaly detection
    for (const [metric, value] of Object.entries(input)) {
      const baseline = this.baselineStats.get(metric);
      if (baseline) {
        const zScore = Math.abs(value - baseline.mean) / baseline.std;
        if (zScore > this.config.maxDeviationSigma) {
          anomalyScores.push(1);
          violations.push(`${metric} z-score ${zScore.toFixed(2)} exceeds threshold`);
        } else {
          anomalyScores.push(zScore / this.config.maxDeviationSigma);
        }
      }
    }

    // Physics consistency checks
    let physicallyConsistent = true;
    if (this.config.enablePhysicsValidation) {
      physicallyConsistent = this.validatePhysicalConsistency(input);
      if (!physicallyConsistent) {
        violations.push('Failed physical consistency validation');
      }
    }

    // Calculate overall anomaly score
    const anomalyScore = anomalyScores.length > 0
      ? anomalyScores.reduce((a, b) => a + b, 0) / anomalyScores.length
      : 0;

    // Determine attack type if adversarial
    let attackType: 'poisoning' | 'evasion' | 'inference' | 'spoofing' | undefined;
    if (anomalyScore > this.config.sensitivityThreshold) {
      attackType = this.classifyAttackType(input, violations);
    }

    const isAdversarial = anomalyScore > this.config.sensitivityThreshold || !physicallyConsistent;

    return {
      isAdversarial,
      confidence: isAdversarial ? anomalyScore : 1 - anomalyScore,
      attackType,
      anomalyScore,
      physicalConsistency: physicallyConsistent,
      recommendation: isAdversarial
        ? (anomalyScore > 0.9 ? 'reject' : 'quarantine')
        : 'accept'
    };
  }

  /**
   * Validate physical consistency of measurements
   * Uses neuro-symbolic logic rules
   */
  private validatePhysicalConsistency(input: Record<string, number>): boolean {
    // Rule 1: RSRP and Pathloss relationship
    // Higher pathloss should correlate with lower RSRP
    if (input.rsrp !== undefined && input.pathloss !== undefined) {
      const expectedRsrp = 46 - input.pathloss; // TX Power - PL
      const deviation = Math.abs(input.rsrp - expectedRsrp);
      if (deviation > 20) return false; // Allow some margin for fading
    }

    // Rule 2: SINR vs RSRP vs Noise
    // SINR ≈ RSRP - Interference - Noise
    if (input.sinr !== undefined && input.rsrp !== undefined) {
      // Very high SINR with very low RSRP is suspicious
      if (input.sinr > 25 && input.rsrp < -115) return false;
      // Very low SINR with very high RSRP is suspicious
      if (input.sinr < -5 && input.rsrp > -70) return false;
    }

    // Rule 3: Timing Advance vs Distance
    // TA should be consistent with claimed location
    if (input.timing_advance !== undefined) {
      const impliedDistance = input.timing_advance * 78.125; // meters per TA unit
      if (impliedDistance > 100000) return false; // Max cell range ~100km
    }

    // Rule 4: Power levels must be within physical limits
    if (input.power !== undefined) {
      if (input.power < -50 || input.power > 50) return false;
    }

    // Rule 5: Rate of change checks (would need history)
    // Large instantaneous changes in metrics are suspicious

    return true;
  }

  private classifyAttackType(
    input: Record<string, number>,
    violations: string[]
  ): 'poisoning' | 'evasion' | 'inference' | 'spoofing' {
    // Heuristic classification based on violation patterns

    // Multiple extreme values suggest poisoning
    const extremeCount = violations.filter(v => v.includes('z-score')).length;
    if (extremeCount >= 3) return 'poisoning';

    // Physical inconsistency suggests spoofing
    if (violations.some(v => v.includes('physical consistency'))) return 'spoofing';

    // Single metric manipulation suggests evasion
    if (extremeCount === 1) return 'evasion';

    // Default to inference attack (probing)
    return 'inference';
  }

  /**
   * Update baselines with trusted data
   */
  updateBaseline(metric: string, values: number[]): void {
    if (values.length < 10) return;

    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((a, v) => a + Math.pow(v - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    this.baselineStats.set(metric, { mean, std: Math.max(std, 0.1) });
  }
}

// ============================================================================
// GUARDRAIL ENFORCER
// ============================================================================

interface GuardrailConfig {
  power: { min: number; max: number };
  p0: { min: number; max: number };
  alpha: { min: number; max: number };
  tilt: { min: number; max: number };
  maxActionsPerHour: number;
  forbiddenActions: string[];
}

class GuardrailEnforcer {
  config: GuardrailConfig;
  private actionHistory: Map<string, Date[]> = new Map();
  private violations: GuardrailViolation[] = [];

  constructor(config: Partial<GuardrailConfig> = {}) {
    this.config = {
      power: { min: -10, max: 46 },
      p0: { min: -126, max: -60 },
      alpha: { min: 0.4, max: 1.0 },
      tilt: { min: 0, max: 15 },
      maxActionsPerHour: 10,
      forbiddenActions: ['SHUTDOWN_CELL', 'DELETE_NEIGHBOR_ALL'],
      ...config
    };
  }

  /**
   * Check if action violates any guardrails
   */
  checkAction(action: Action): {
    allowed: boolean;
    violations: GuardrailViolation[];
    clippedParams?: Record<string, number>;
  } {
    const violations: GuardrailViolation[] = [];
    const clippedParams: Record<string, number> = {};

    // Check forbidden actions
    if (this.config.forbiddenActions.includes(action.type)) {
      violations.push({
        actionId: uuidv4(),
        violation: 'Forbidden action type',
        parameter: 'type',
        attemptedValue: 0,
        allowedRange: [0, 0],
        timestamp: new Date()
      });
    }

    // Check rate limiting
    const cellId = action.targetCgi;
    const now = Date.now();
    const hourAgo = now - 3600000;

    const cellHistory = this.actionHistory.get(cellId) || [];
    const recentActions = cellHistory.filter(d => d.getTime() > hourAgo);

    if (recentActions.length >= this.config.maxActionsPerHour) {
      violations.push({
        actionId: uuidv4(),
        violation: 'Rate limit exceeded',
        parameter: 'actionRate',
        attemptedValue: recentActions.length + 1,
        allowedRange: [0, this.config.maxActionsPerHour],
        timestamp: new Date()
      });
    }

    // Check parameter ranges
    const params = action.parameters;

    if (params.newPower !== undefined) {
      const power = Number(params.newPower);
      if (power < this.config.power.min || power > this.config.power.max) {
        violations.push({
          actionId: uuidv4(),
          violation: 'Power out of range',
          parameter: 'power',
          attemptedValue: power,
          allowedRange: [this.config.power.min, this.config.power.max],
          timestamp: new Date()
        });
        clippedParams.newPower = Math.max(
          this.config.power.min,
          Math.min(this.config.power.max, power)
        );
      }
    }

    if (params.newP0 !== undefined) {
      const p0 = Number(params.newP0);
      if (p0 < this.config.p0.min || p0 > this.config.p0.max) {
        violations.push({
          actionId: uuidv4(),
          violation: 'P0 out of range',
          parameter: 'p0',
          attemptedValue: p0,
          allowedRange: [this.config.p0.min, this.config.p0.max],
          timestamp: new Date()
        });
        clippedParams.newP0 = Math.max(
          this.config.p0.min,
          Math.min(this.config.p0.max, p0)
        );
      }
    }

    if (params.newAlpha !== undefined) {
      const alpha = Number(params.newAlpha);
      if (alpha < this.config.alpha.min || alpha > this.config.alpha.max) {
        violations.push({
          actionId: uuidv4(),
          violation: 'Alpha out of range',
          parameter: 'alpha',
          attemptedValue: alpha,
          allowedRange: [this.config.alpha.min, this.config.alpha.max],
          timestamp: new Date()
        });
        clippedParams.newAlpha = Math.max(
          this.config.alpha.min,
          Math.min(this.config.alpha.max, alpha)
        );
      }
    }

    if (params.tiltDelta !== undefined || params.newTilt !== undefined) {
      const tilt = Number(params.newTilt || params.tiltDelta);
      if (tilt < this.config.tilt.min || tilt > this.config.tilt.max) {
        violations.push({
          actionId: uuidv4(),
          violation: 'Tilt out of range',
          parameter: 'tilt',
          attemptedValue: tilt,
          allowedRange: [this.config.tilt.min, this.config.tilt.max],
          timestamp: new Date()
        });
      }
    }

    // Store violations
    this.violations.push(...violations);

    return {
      allowed: violations.length === 0,
      violations,
      clippedParams: Object.keys(clippedParams).length > 0 ? clippedParams : undefined
    };
  }

  /**
   * Record executed action for rate limiting
   */
  recordAction(cellId: string): void {
    if (!this.actionHistory.has(cellId)) {
      this.actionHistory.set(cellId, []);
    }
    this.actionHistory.get(cellId)!.push(new Date());

    // Clean old entries
    const hourAgo = Date.now() - 3600000;
    const history = this.actionHistory.get(cellId)!;
    this.actionHistory.set(cellId, history.filter(d => d.getTime() > hourAgo));
  }

  /**
   * Get recent violations
   */
  getViolations(limit: number = 100): GuardrailViolation[] {
    return this.violations.slice(-limit);
  }

  /**
   * Clear old violations
   */
  clearOldViolations(maxAge: number = 86400000): void {
    const cutoff = Date.now() - maxAge;
    this.violations = this.violations.filter(v => v.timestamp.getTime() > cutoff);
  }
}

// ============================================================================
// MODEL PROVENANCE TRACKER
// ============================================================================

interface ProvenanceEntry {
  modelId: string;
  version: number;
  hash: string;
  contributors: NanoAgentId[];
  parentHashes: string[];
  timestamp: Date;
  signature: string;
  metadata: Record<string, any>;
}

class ModelProvenanceTracker {
  private ledger: ProvenanceEntry[] = [];
  private modelHashes: Map<string, ProvenanceEntry[]> = new Map();

  /**
   * Record model version in provenance ledger
   */
  recordVersion(
    modelId: string,
    weights: Float32Array,
    contributors: NanoAgentId[],
    parentVersion?: number
  ): ProvenanceEntry {
    // Compute hash of weights
    const hash = this.computeHash(weights);

    // Get parent hashes
    const parentHashes: string[] = [];
    if (parentVersion !== undefined) {
      const modelHistory = this.modelHashes.get(modelId) || [];
      const parent = modelHistory.find(e => e.version === parentVersion);
      if (parent) {
        parentHashes.push(parent.hash);
      }
    }

    // Create entry
    const version = (this.modelHashes.get(modelId)?.length || 0) + 1;
    const entry: ProvenanceEntry = {
      modelId,
      version,
      hash,
      contributors,
      parentHashes,
      timestamp: new Date(),
      signature: this.signEntry(modelId, hash, contributors),
      metadata: {
        weightCount: weights.length,
        contributorCount: contributors.length
      }
    };

    // Store in ledger
    this.ledger.push(entry);
    if (!this.modelHashes.has(modelId)) {
      this.modelHashes.set(modelId, []);
    }
    this.modelHashes.get(modelId)!.push(entry);

    return entry;
  }

  /**
   * Verify model integrity
   */
  verifyModel(modelId: string, version: number, weights: Float32Array): boolean {
    const entry = this.getEntry(modelId, version);
    if (!entry) return false;

    const currentHash = this.computeHash(weights);
    return currentHash === entry.hash;
  }

  /**
   * Check for model poisoning (sudden changes)
   */
  detectPoisoning(
    modelId: string,
    newWeights: Float32Array,
    threshold: number = 0.5
  ): { isPoisoned: boolean; divergenceScore: number; suspiciousContributors: NanoAgentId[] } {
    const history = this.modelHashes.get(modelId) || [];
    if (history.length === 0) {
      return { isPoisoned: false, divergenceScore: 0, suspiciousContributors: [] };
    }

    // Compare with recent versions
    const recentEntries = history.slice(-5);
    let maxDivergence = 0;

    for (const entry of recentEntries) {
      // In real impl, would compare actual weights
      // Here we use hash comparison as proxy
      const divergence = this.estimateDivergence(entry.hash, this.computeHash(newWeights));
      maxDivergence = Math.max(maxDivergence, divergence);
    }

    const isPoisoned = maxDivergence > threshold;
    const suspiciousContributors = isPoisoned
      ? history[history.length - 1]?.contributors || []
      : [];

    return {
      isPoisoned,
      divergenceScore: maxDivergence,
      suspiciousContributors
    };
  }

  /**
   * Get provenance entry
   */
  getEntry(modelId: string, version: number): ProvenanceEntry | undefined {
    return this.modelHashes.get(modelId)?.find(e => e.version === version);
  }

  /**
   * Get full history for a model
   */
  getHistory(modelId: string): ProvenanceEntry[] {
    return this.modelHashes.get(modelId) || [];
  }

  private computeHash(weights: Float32Array): string {
    // Simple hash for demo - use SHA-256 in production
    let hash = 0;
    for (let i = 0; i < Math.min(weights.length, 1000); i++) {
      hash = ((hash << 5) - hash) + Math.floor(weights[i] * 1000);
      hash = hash & hash;
    }
    return `hash-${Math.abs(hash).toString(16).padStart(16, '0')}`;
  }

  private signEntry(modelId: string, hash: string, contributors: NanoAgentId[]): string {
    // Simple signature for demo - use Ed25519 in production
    const data = `${modelId}:${hash}:${contributors.join(',')}`;
    let sig = 0;
    for (let i = 0; i < data.length; i++) {
      sig = ((sig << 5) - sig) + data.charCodeAt(i);
      sig = sig & sig;
    }
    return `sig-${Math.abs(sig).toString(16)}`;
  }

  private estimateDivergence(hash1: string, hash2: string): number {
    // Compare hash strings as proxy for weight divergence
    if (hash1 === hash2) return 0;

    let diff = 0;
    const maxLen = Math.max(hash1.length, hash2.length);
    for (let i = 0; i < maxLen; i++) {
      if (hash1[i] !== hash2[i]) diff++;
    }
    return diff / maxLen;
  }

  getLedgerLength(): number {
    return this.ledger.length;
  }
}

// ============================================================================
// FEDERATED LEARNING VALIDATOR
// ============================================================================

class FederatedLearningValidator {
  private contributorScores: Map<NanoAgentId, number> = new Map();
  private byzantineThreshold: number = 0.3; // Max 30% Byzantine agents

  /**
   * Validate federated model update
   */
  validateUpdate(
    update: FederatedModelUpdate,
    globalModel: Float32Array | null
  ): { valid: boolean; reason?: string; trustScore: number } {
    const agentScore = this.contributorScores.get(update.sourceAgent) || 0.5;

    // Check for extreme gradients (gradient attack)
    if (update.gradients) {
      const maxGrad = Math.max(...update.gradients.map(Math.abs));
      if (maxGrad > 100) {
        this.decreaseScore(update.sourceAgent, 0.2);
        return {
          valid: false,
          reason: 'Extreme gradient values detected',
          trustScore: agentScore - 0.2
        };
      }
    }

    // Check weight divergence from global model
    if (globalModel && update.weights) {
      const divergence = this.computeDivergence(globalModel, new Float32Array(update.weights));
      if (divergence > 0.8) {
        this.decreaseScore(update.sourceAgent, 0.1);
        return {
          valid: false,
          reason: `Weight divergence ${divergence.toFixed(2)} too high`,
          trustScore: agentScore - 0.1
        };
      }
    }

    // Check sample count reasonability
    if (update.sampleCount < 10 || update.sampleCount > 100000) {
      return {
        valid: false,
        reason: 'Suspicious sample count',
        trustScore: agentScore
      };
    }

    // Valid update - increase trust
    this.increaseScore(update.sourceAgent, 0.01);

    return {
      valid: true,
      trustScore: Math.min(1, agentScore + 0.01)
    };
  }

  /**
   * Get agent trust score
   */
  getTrustScore(agentId: NanoAgentId): number {
    return this.contributorScores.get(agentId) || 0.5;
  }

  /**
   * Filter out potentially Byzantine contributors
   */
  filterByzantine(
    updates: FederatedModelUpdate[]
  ): { trusted: FederatedModelUpdate[]; excluded: FederatedModelUpdate[] } {
    const trusted: FederatedModelUpdate[] = [];
    const excluded: FederatedModelUpdate[] = [];

    // Sort by trust score
    const scored = updates.map(u => ({
      update: u,
      score: this.contributorScores.get(u.sourceAgent) || 0.5
    }));
    scored.sort((a, b) => b.score - a.score);

    // Take top (1 - byzantineThreshold) contributors
    const keepCount = Math.ceil(updates.length * (1 - this.byzantineThreshold));

    for (let i = 0; i < scored.length; i++) {
      if (i < keepCount && scored[i].score > 0.3) {
        trusted.push(scored[i].update);
      } else {
        excluded.push(scored[i].update);
      }
    }

    return { trusted, excluded };
  }

  private computeDivergence(a: Float32Array, b: Float32Array): number {
    let sumSqDiff = 0;
    let sumSqA = 0;

    const len = Math.min(a.length, b.length);
    for (let i = 0; i < len; i++) {
      sumSqDiff += (a[i] - b[i]) ** 2;
      sumSqA += a[i] ** 2;
    }

    return sumSqA > 0 ? Math.sqrt(sumSqDiff / sumSqA) : 0;
  }

  private increaseScore(agentId: NanoAgentId, delta: number): void {
    const current = this.contributorScores.get(agentId) || 0.5;
    this.contributorScores.set(agentId, Math.min(1, current + delta));
  }

  private decreaseScore(agentId: NanoAgentId, delta: number): void {
    const current = this.contributorScores.get(agentId) || 0.5;
    this.contributorScores.set(agentId, Math.max(0, current - delta));
  }
}

// ============================================================================
// MAIN AIDEFENCE CLASS
// ============================================================================

export interface AIDefenceConfig {
  enableAdversarialDetection: boolean;
  enableGuardrails: boolean;
  enableProvenance: boolean;
  enableFederatedValidation: boolean;
  adversarialSensitivity: number;
  maxActionsPerHour: number;
}

const DEFAULT_CONFIG: AIDefenceConfig = {
  enableAdversarialDetection: true,
  enableGuardrails: true,
  enableProvenance: true,
  enableFederatedValidation: true,
  adversarialSensitivity: 0.7,
  maxActionsPerHour: 10
};

export class AIDefence extends EventEmitter {
  config: AIDefenceConfig;
  adversarialDetector: AdversarialDetector;
  guardrailEnforcer: GuardrailEnforcer;
  provenanceTracker: ModelProvenanceTracker;
  federatedValidator: FederatedLearningValidator;

  // Statistics
  private inputsScanned: number = 0;
  private adversarialBlocked: number = 0;
  private actionsChecked: number = 0;
  private actionsBlocked: number = 0;

  constructor(config: Partial<AIDefenceConfig> = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };

    this.adversarialDetector = new AdversarialDetector({
      sensitivityThreshold: this.config.adversarialSensitivity
    });

    this.guardrailEnforcer = new GuardrailEnforcer({
      maxActionsPerHour: this.config.maxActionsPerHour
    });

    this.provenanceTracker = new ModelProvenanceTracker();
    this.federatedValidator = new FederatedLearningValidator();
  }

  /**
   * Scan input for adversarial content
   */
  scanInput(
    input: Record<string, number>,
    context: { cellId: string; timestamp: Date }
  ): AdversarialDetectionResult {
    this.inputsScanned++;

    if (!this.config.enableAdversarialDetection) {
      return {
        isAdversarial: false,
        confidence: 1,
        anomalyScore: 0,
        physicalConsistency: true,
        recommendation: 'accept'
      };
    }

    const result = this.adversarialDetector.detectAdversarial(input, context);

    if (result.isAdversarial) {
      this.adversarialBlocked++;
      this.emit('adversarial-detected', { input, context, result });
    }

    return result;
  }

  /**
   * Check action against guardrails
   */
  checkAction(action: Action): {
    allowed: boolean;
    violations: GuardrailViolation[];
    clippedParams?: Record<string, number>;
  } {
    this.actionsChecked++;

    if (!this.config.enableGuardrails) {
      return { allowed: true, violations: [] };
    }

    const result = this.guardrailEnforcer.checkAction(action);

    if (!result.allowed) {
      this.actionsBlocked++;
      this.emit('guardrail-violation', { action, violations: result.violations });
    }

    return result;
  }

  /**
   * Record action execution (for rate limiting)
   */
  recordAction(cellId: string): void {
    this.guardrailEnforcer.recordAction(cellId);
  }

  /**
   * Record model version in provenance ledger
   */
  recordModelVersion(
    modelId: string,
    weights: Float32Array,
    contributors: NanoAgentId[],
    parentVersion?: number
  ): ProvenanceEntry {
    if (!this.config.enableProvenance) {
      return {
        modelId,
        version: 0,
        hash: '',
        contributors,
        parentHashes: [],
        timestamp: new Date(),
        signature: '',
        metadata: {}
      };
    }

    const entry = this.provenanceTracker.recordVersion(
      modelId,
      weights,
      contributors,
      parentVersion
    );

    this.emit('model-recorded', { modelId, version: entry.version });
    return entry;
  }

  /**
   * Verify model integrity
   */
  verifyModel(modelId: string, version: number, weights: Float32Array): boolean {
    return this.provenanceTracker.verifyModel(modelId, version, weights);
  }

  /**
   * Detect model poisoning
   */
  detectModelPoisoning(
    modelId: string,
    newWeights: Float32Array
  ): { isPoisoned: boolean; divergenceScore: number } {
    const result = this.provenanceTracker.detectPoisoning(modelId, newWeights);

    if (result.isPoisoned) {
      this.emit('model-poisoning-detected', { modelId, ...result });
    }

    return result;
  }

  /**
   * Validate federated learning update
   */
  validateFederatedUpdate(
    update: FederatedModelUpdate,
    globalModel: Float32Array | null
  ): { valid: boolean; reason?: string; trustScore: number } {
    if (!this.config.enableFederatedValidation) {
      return { valid: true, trustScore: 1 };
    }

    const result = this.federatedValidator.validateUpdate(update, globalModel);

    if (!result.valid) {
      this.emit('federated-update-rejected', { update, reason: result.reason });
    }

    return result;
  }

  /**
   * Filter Byzantine contributors from federated updates
   */
  filterByzantine(
    updates: FederatedModelUpdate[]
  ): { trusted: FederatedModelUpdate[]; excluded: FederatedModelUpdate[] } {
    return this.federatedValidator.filterByzantine(updates);
  }

  /**
   * Get trust score for agent
   */
  getAgentTrustScore(agentId: NanoAgentId): number {
    return this.federatedValidator.getTrustScore(agentId);
  }

  /**
   * Get security statistics
   */
  getStats(): {
    inputsScanned: number;
    adversarialBlocked: number;
    actionsChecked: number;
    actionsBlocked: number;
    provenanceRecords: number;
    recentViolations: number;
  } {
    return {
      inputsScanned: this.inputsScanned,
      adversarialBlocked: this.adversarialBlocked,
      actionsChecked: this.actionsChecked,
      actionsBlocked: this.actionsBlocked,
      provenanceRecords: this.provenanceTracker.getLedgerLength(),
      recentViolations: this.guardrailEnforcer.getViolations(100).length
    };
  }
}
