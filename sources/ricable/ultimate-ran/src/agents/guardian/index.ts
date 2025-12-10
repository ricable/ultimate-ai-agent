/**
 * Guardian Agent - Adversarial Safety Agent
 * TITAN Neuro-Symbolic RAN Platform
 *
 * Role: Pre-commit audit, hallucination detection, Lyapunov stability analysis
 * Technology: Digital twin simulation (E2B), safety threshold validation
 *
 * Safety Thresholds (from PRD line 495-499):
 * - lyapunov_max: 0.0 (positive exponent = chaos)
 * - bler_max: 0.1 (10% BLER limit)
 * - power_max_dbm: 46 (maximum transmission power)
 */

// @ts-ignore - BaseAgent is JavaScript, ignore type errors
import { BaseAgent } from '../base-agent.js';
import { LyapunovAnalyzer } from './lyapunov-analyzer.js';
import { DigitalTwin } from './digital-twin.js';
import { SafetyThresholds } from './safety-thresholds.js';

export interface GuardianConfig {
  id?: string;
  thresholds?: {
    lyapunov_max?: number;
    bler_max?: number;
    power_max_dbm?: number;
  };
  // Dependency injection for testing (London School TDD)
  lyapunovAnalyzer?: LyapunovAnalyzer;
  digitalTwin?: DigitalTwin;
  safetyThresholds?: SafetyThresholds;
}

export interface Artifact {
  id: string;
  code: string;
  parameters: Record<string, any>;
}

export interface SimulationResult {
  id: string;
  sandboxId?: string;
  steps: SimulationStep[];
  finalState?: SimulationStep;
}

export interface SimulationStep {
  step: number;
  kpis: {
    throughput: number;
    bler: number;
    interference: number;
  };
  stable?: boolean;
}

export interface LyapunovResult {
  exponent: number;
  stable: boolean;
  interpretation: 'STABLE' | 'CHAOTIC';
  reliable: boolean;
  reason?: string;
}

export interface SafetyValidationResult {
  valid: boolean;
  violations: SafetyViolation[];
}

export interface SafetyViolation {
  type: string;
  threshold?: number;
  actual?: number;
  description?: string;
  severity?: string;
  parameter?: string;
  conflictingParams?: string[];
  line?: number;
  recommendation?: string;
}

export interface Hallucination {
  type: string;
  severity: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  description: string;
  parameter?: string;
  line?: number;
  recommendation?: string;
  conflictingParams?: string[];
}

export interface GuardianTaskResult {
  approved: boolean;
  simulation: SimulationResult;
  lyapunovResult: LyapunovResult;
  safetyValidation: SafetyValidationResult;
  hallucinations: Hallucination[];
  rejectionReason?: string;
}

export class GuardianAgent extends (BaseAgent as any) {
  public thresholds: {
    lyapunov_max: number;
    bler_max: number;
    power_max_dbm: number;
  };

  private lyapunovAnalyzer: LyapunovAnalyzer;
  private digitalTwin: DigitalTwin;
  private safetyThresholds: SafetyThresholds;

  // Properties from BaseAgent (for TypeScript)
  id!: string;
  type!: string;
  role!: string;
  capabilities!: string[];
  emitAGUI!: (eventType: string, payload: any) => void;

  constructor(config: GuardianConfig = {}) {
    super({
      ...config,
      type: 'guardian',
      role: 'Adversarial Safety Agent',
      capabilities: [
        'pre_commit_simulation',
        'hallucination_detection',
        'lyapunov_analysis',
        'safety_verification',
      ],
      tools: ['strange-loops', 'digital-twin', 'agentic-jujutsu'],
    });

    // Safety thresholds from PRD (line 495-499)
    this.thresholds = {
      lyapunov_max: config.thresholds?.lyapunov_max ?? 0.0,
      bler_max: config.thresholds?.bler_max ?? 0.1,
      power_max_dbm: config.thresholds?.power_max_dbm ?? 46,
    };

    // Initialize analysis components (with dependency injection support for testing)
    this.lyapunovAnalyzer = config.lyapunovAnalyzer ?? new LyapunovAnalyzer(this.thresholds);
    this.digitalTwin = config.digitalTwin ?? new DigitalTwin();
    this.safetyThresholds = config.safetyThresholds ?? new SafetyThresholds(this.thresholds);
  }

  /**
   * Analyze Lyapunov stability exponent (public API for testing)
   */
  async analyzeLyapunovStability(artifact: Artifact): Promise<LyapunovResult> {
    console.log('[GUARDIAN] Calculating Lyapunov exponent...');

    // First run simulation
    const simulation = await this.digitalTwin.simulate(artifact);

    // Analyze stability using Lyapunov exponent
    const result = await this.lyapunovAnalyzer.analyze(simulation);

    return result;
  }

  /**
   * Run pre-commit simulation in E2B digital twin sandbox (public API for testing)
   */
  async runPreCommitSimulation(artifact: Artifact): Promise<SimulationResult> {
    console.log('[GUARDIAN] Running pre-commit simulation in digital twin...');

    // Create E2B sandbox
    const sandboxId = await this.digitalTwin.createSandbox();

    try {
      // Run simulation
      const simulation = await this.digitalTwin.runPreCommitSimulation(
        sandboxId,
        artifact
      );

      return simulation;
    } finally {
      // Always destroy sandbox (cleanup)
      await this.digitalTwin.destroySandbox(sandboxId);
    }
  }

  /**
   * Validate against 3GPP safety thresholds (public API for testing)
   */
  async validateSafetyThresholds(artifact: Artifact): Promise<SafetyValidationResult> {
    console.log('[GUARDIAN] Validating safety thresholds...');

    const result = this.safetyThresholds.validate(artifact);

    return result;
  }

  /**
   * Detect hallucinations - syntactically correct but physically dangerous code (public API for testing)
   */
  async detectHallucinations(artifact: Artifact): Promise<Hallucination[]> {
    console.log('[GUARDIAN] Scanning for hallucinations...');

    const hallucinations = this.safetyThresholds.detectHallucinations(artifact);

    console.log(`[GUARDIAN] Found ${hallucinations.length} hallucinations`);
    return hallucinations;
  }

  /**
   * Main task processing: comprehensive safety audit
   */
  async processTask(task: { artifact: Artifact }): Promise<GuardianTaskResult> {
    console.log(`[GUARDIAN] Auditing artifact: ${task.artifact.id}`);

    // Step 1: Run pre-commit simulation in digital twin
    const simulation = await this.runPreCommitSimulation(task.artifact);

    // Step 2: Analyze Lyapunov stability
    const lyapunovResult = await this.analyzeLyapunovStability(task.artifact);

    // Step 3: Validate safety thresholds
    const safetyValidation = await this.validateSafetyThresholds(task.artifact);

    // Step 4: Detect hallucinations
    const hallucinations = await this.detectHallucinations(task.artifact);

    // Step 5: Render final verdict
    const approved = this.renderVerdict(
      simulation,
      lyapunovResult,
      safetyValidation,
      hallucinations
    );

    const result: GuardianTaskResult = {
      approved,
      simulation,
      lyapunovResult,
      safetyValidation,
      hallucinations,
    };

    // Add rejection reason if not approved
    if (!approved) {
      result.rejectionReason = this.buildRejectionReason(
        lyapunovResult,
        safetyValidation,
        hallucinations
      );
    }

    // Emit AG-UI event
    this.emitAGUI('agent_message', {
      type: 'markdown',
      content: this.formatAuditReport(result),
      agent_id: this.id,
    });

    return result;
  }

  /**
   * Render final verdict based on all safety checks
   */
  private renderVerdict(
    simulation: SimulationResult,
    lyapunovResult: LyapunovResult,
    safetyValidation: SafetyValidationResult,
    hallucinations: Hallucination[]
  ): boolean {
    // Reject if critical hallucinations detected
    const hasCriticalHallucinations = hallucinations.some(
      (h) => h.severity === 'CRITICAL'
    );
    if (hasCriticalHallucinations) {
      console.error('[GUARDIAN] REJECTED: Critical hallucinations detected');
      return false;
    }

    // Reject if chaotic behavior detected
    if (!lyapunovResult.stable && lyapunovResult.reliable) {
      console.error('[GUARDIAN] REJECTED: System exhibits chaotic behavior');
      return false;
    }

    // Reject if safety thresholds violated
    if (!safetyValidation.valid) {
      console.error('[GUARDIAN] REJECTED: Safety threshold violations');
      return false;
    }

    console.log('[GUARDIAN] APPROVED: All safety checks passed');
    return true;
  }

  /**
   * Build detailed rejection reason message
   */
  private buildRejectionReason(
    lyapunovResult: LyapunovResult,
    safetyValidation: SafetyValidationResult,
    hallucinations: Hallucination[]
  ): string {
    const reasons: string[] = [];

    if (!lyapunovResult.stable && lyapunovResult.reliable) {
      reasons.push(
        `CHAOTIC behavior detected (Lyapunov exponent: ${lyapunovResult.exponent.toFixed(3)})`
      );
    }

    if (!safetyValidation.valid) {
      reasons.push(
        `Safety violations: ${safetyValidation.violations.map((v) => v.type).join(', ')}`
      );
    }

    const criticalHalls = hallucinations.filter((h) => h.severity === 'CRITICAL');
    if (criticalHalls.length > 0) {
      reasons.push(
        `CRITICAL hallucinations: ${criticalHalls.map((h) => h.type).join(', ')}`
      );
    }

    return reasons.join('; ');
  }

  /**
   * Format audit report for AG-UI display
   */
  private formatAuditReport(result: GuardianTaskResult): string {
    const verdict = result.approved ? '✅ APPROVED' : '❌ REJECTED';

    let report = `## Safety Audit Result\n\n**Verdict:** ${verdict}\n\n`;

    // Lyapunov Analysis
    report += `### Lyapunov Stability Analysis\n`;
    report += `- Exponent: ${result.lyapunovResult.exponent.toFixed(3)}\n`;
    report += `- Status: ${result.lyapunovResult.interpretation}\n`;
    report += `- Reliable: ${result.lyapunovResult.reliable ? 'Yes' : 'No'}\n\n`;

    // Safety Thresholds
    report += `### Safety Threshold Validation\n`;
    report += `- Valid: ${result.safetyValidation.valid ? 'Yes' : 'No'}\n`;
    if (result.safetyValidation.violations.length > 0) {
      report += `- Violations:\n`;
      result.safetyValidation.violations.forEach((v) => {
        report += `  - ${v.type}: ${v.description || 'N/A'}\n`;
      });
    }
    report += `\n`;

    // Hallucinations
    report += `### Hallucination Detection\n`;
    report += `- Count: ${result.hallucinations.length}\n`;
    if (result.hallucinations.length > 0) {
      result.hallucinations.forEach((h) => {
        report += `  - ${h.type} (${h.severity}): ${h.description}\n`;
      });
    }
    report += `\n`;

    // Rejection reason
    if (result.rejectionReason) {
      report += `### Rejection Reason\n${result.rejectionReason}\n\n`;
    }

    return report;
  }
}
