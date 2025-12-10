/**
 * Safety Thresholds Validator
 * TITAN Neuro-Symbolic RAN Platform
 *
 * Validates RAN parameters against 3GPP safety thresholds and detects
 * hallucinations (syntactically correct but physically dangerous code).
 *
 * Safety Thresholds (from PRD line 495-499):
 * - lyapunov_max: 0.0 (positive exponent = chaos)
 * - bler_max: 0.1 (10% BLER limit)
 * - power_max_dbm: 46 (maximum transmission power)
 *
 * 3GPP Compliance (TS 38.213, TS 38.331):
 * - P0: -130 to -70 dBm
 * - Alpha: 0.0 to 1.0
 * - Interference: -105 dBm typical limit
 */

export interface SafetyConfig {
  lyapunov_max: number;
  bler_max: number;
  power_max_dbm: number;
  interference_max_dbm?: number;
}

export interface Artifact {
  id: string;
  code: string;
  parameters: Record<string, any>;
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

export class SafetyThresholds {
  private config: SafetyConfig;

  constructor(config: SafetyConfig) {
    this.config = {
      ...config,
      interference_max_dbm: config.interference_max_dbm ?? -105,
    };
  }

  /**
   * Validate artifact against all safety thresholds
   */
  validate(artifact: Artifact): SafetyValidationResult {
    const violations: SafetyViolation[] = [];

    // Check BLER
    const blerCheck = this.checkBLER(artifact);
    if (!blerCheck.valid) {
      violations.push({
        type: 'BLER_EXCEEDED',
        threshold: blerCheck.threshold,
        actual: blerCheck.actual,
        description: blerCheck.violation,
        severity: 'HIGH',
      });
    }

    // Check Power
    const powerCheck = this.checkPower(artifact);
    if (!powerCheck.valid) {
      violations.push({
        type: 'POWER_EXCEEDED',
        threshold: powerCheck.threshold,
        actual: powerCheck.actual,
        description: powerCheck.violation,
        severity: 'CRITICAL',
      });
    }

    // Check Interference
    const interferenceCheck = this.checkInterference(artifact);
    if (!interferenceCheck.valid) {
      violations.push({
        type: 'INTERFERENCE_HIGH',
        threshold: interferenceCheck.threshold,
        actual: interferenceCheck.actual,
        description: interferenceCheck.violation,
        severity: 'MEDIUM',
      });
    }

    // Check 3GPP parameter ranges
    const rangeViolations = this.check3GPPRanges(artifact);
    violations.push(...rangeViolations);

    return {
      valid: violations.length === 0,
      violations,
    };
  }

  /**
   * Check BLER threshold (max 10%)
   */
  checkBLER(artifact: Artifact): {
    valid: boolean;
    threshold?: number;
    actual?: number;
    violation?: string;
  } {
    const targetBLER = artifact.parameters.targetBLER;

    if (targetBLER !== undefined && targetBLER > this.config.bler_max) {
      return {
        valid: false,
        threshold: this.config.bler_max,
        actual: targetBLER,
        violation: `BLER ${(targetBLER * 100).toFixed(1)}% exceeds ${(this.config.bler_max * 100).toFixed(1)}% limit`,
      };
    }

    return { valid: true };
  }

  /**
   * Check power threshold (max 46 dBm)
   */
  checkPower(artifact: Artifact): {
    valid: boolean;
    threshold?: number;
    actual?: number;
    violation?: string;
  } {
    const maxPower = artifact.parameters.maxPower;

    if (maxPower !== undefined && maxPower > this.config.power_max_dbm) {
      return {
        valid: false,
        threshold: this.config.power_max_dbm,
        actual: maxPower,
        violation: `Power ${maxPower} dBm exceeds ${this.config.power_max_dbm} dBm limit`,
      };
    }

    return { valid: true };
  }

  /**
   * Check interference threshold (typical -105 dBm max)
   */
  checkInterference(artifact: Artifact): {
    valid: boolean;
    threshold?: number;
    actual?: number;
    violation?: string;
  } {
    const interferenceLevel = artifact.parameters.interferenceLevel;

    if (
      interferenceLevel !== undefined &&
      interferenceLevel > this.config.interference_max_dbm!
    ) {
      return {
        valid: false,
        threshold: this.config.interference_max_dbm,
        actual: interferenceLevel,
        violation: `Interference ${interferenceLevel} dBm exceeds ${this.config.interference_max_dbm} dBm limit`,
      };
    }

    return { valid: true };
  }

  /**
   * Check 3GPP parameter ranges (TS 38.213, TS 38.331)
   */
  private check3GPPRanges(artifact: Artifact): SafetyViolation[] {
    const violations: SafetyViolation[] = [];

    // P0 range: -130 to -70 dBm
    const p0 = artifact.parameters.p0NominalPUSCH;
    if (p0 !== undefined && (p0 < -130 || p0 > -70)) {
      violations.push({
        type: 'P0_OUT_OF_RANGE',
        threshold: -70, // upper limit
        actual: p0,
        description: `P0 ${p0} dBm out of range [-130, -70] (TS 38.213)`,
        severity: 'HIGH',
        parameter: 'p0NominalPUSCH',
      });
    }

    // Alpha range: 0.0 to 1.0
    const alpha = artifact.parameters.alpha;
    if (alpha !== undefined && (alpha < 0 || alpha > 1)) {
      violations.push({
        type: 'ALPHA_OUT_OF_RANGE',
        threshold: 1.0,
        actual: alpha,
        description: `Alpha ${alpha} out of range [0, 1] (TS 38.213)`,
        severity: 'HIGH',
        parameter: 'alpha',
      });
    }

    return violations;
  }

  /**
   * Detect hallucinations - syntactically correct but physically dangerous
   */
  detectHallucinations(artifact: Artifact): Hallucination[] {
    const hallucinations: Hallucination[] = [];

    // Check for infinite power loops
    if (this.hasInfinitePowerLoop(artifact)) {
      hallucinations.push({
        type: 'INFINITE_POWER_LOOP',
        severity: 'CRITICAL',
        description: 'Code contains loop that indefinitely increases transmission power',
        recommendation: 'Add explicit break condition or power limit',
      });
    }

    // Check for physics violations
    if (this.violatesPhysicsConstraints(artifact)) {
      hallucinations.push({
        type: 'PHYSICS_VIOLATION',
        severity: 'HIGH',
        description: 'Code violates physical RF constraints',
        recommendation: 'Review parameter values against 3GPP specifications',
      });
    }

    // Check for missing safety bounds
    if (!this.hasSafetyBounds(artifact)) {
      hallucinations.push({
        type: 'MISSING_SAFETY_BOUNDS',
        severity: 'MEDIUM',
        description: 'Code lacks explicit safety boundary checks',
        recommendation: 'Add min/max validation using Math.min/Math.max',
      });
    }

    // Check for parameter conflicts
    const conflicts = this.detectParameterConflicts(artifact);
    hallucinations.push(...conflicts);

    return hallucinations;
  }

  /**
   * Detect infinite power increase loops
   */
  private hasInfinitePowerLoop(artifact: Artifact): boolean {
    const code = artifact.code || '';

    // Check for unbounded power increase patterns
    const patterns = [
      /while\s*\([^)]*\)\s*\{[^}]*power\s*\+\+/,
      /while\s*\([^)]*\)\s*\{[^}]*power\s*\+=\s*\d+/,
      /for\s*\([^)]*\)\s*\{[^}]*power\s*\+\+/,
      /for\s*\([^)]*\)\s*\{[^}]*power\s*\+=\s*\d+/,
      /while\s*\(true\)[^}]*power/i,
    ];

    return patterns.some((pattern) => pattern.test(code));
  }

  /**
   * Check for values exceeding physical limits
   */
  private violatesPhysicsConstraints(artifact: Artifact): boolean {
    const code = artifact.code || '';

    // Check for power values exceeding physical limits
    const powerMatch = code.match(/power\s*=\s*(\d+)/);
    if (powerMatch && parseInt(powerMatch[1]) > this.config.power_max_dbm) {
      return true;
    }

    // Check for negative BLER values (impossible)
    const blerMatch = code.match(/bler\s*=\s*(-?\d+\.?\d*)/);
    if (blerMatch && parseFloat(blerMatch[1]) < 0) {
      return true;
    }

    // Check for P0 values outside physical range
    const p0Match = code.match(/p0\s*=\s*(-?\d+)/);
    if (p0Match) {
      const p0 = parseInt(p0Match[1]);
      if (p0 < -130 || p0 > -70) {
        return true;
      }
    }

    return false;
  }

  /**
   * Check if code has safety bounds
   */
  private hasSafetyBounds(artifact: Artifact): boolean {
    const code = artifact.code || '';

    // Check for common safety patterns
    const safetyPatterns = [
      /if\s*\([^)]*>\s*max/i,
      /if\s*\([^)]*<\s*min/i,
      /Math\.min\s*\(/,
      /Math\.max\s*\(/,
      /Math\.clamp\s*\(/,
      /\.clamp\s*\(/,
    ];

    return safetyPatterns.some((pattern) => pattern.test(code));
  }

  /**
   * Detect parameter conflicts (e.g., alpha=0 with high P0)
   */
  private detectParameterConflicts(artifact: Artifact): Hallucination[] {
    const conflicts: Hallucination[] = [];

    const { p0NominalPUSCH, alpha } = artifact.parameters;

    // Alpha = 0 with high P0 creates cell-edge coverage issues
    if (alpha !== undefined && p0NominalPUSCH !== undefined) {
      if (alpha === 0 && p0NominalPUSCH > -100) {
        conflicts.push({
          type: 'PARAMETER_CONFLICT',
          severity: 'HIGH',
          description: 'Alpha=0 with high P0 creates cell-edge coverage issues',
          conflictingParams: ['alpha', 'p0NominalPUSCH'],
          recommendation:
            'Increase alpha to 0.6-0.8 for cell-edge UEs or reduce P0 to -106 dBm',
        });
      }

      // Alpha = 1.0 with low P0 wastes power for cell-center UEs
      if (alpha === 1.0 && p0NominalPUSCH < -110) {
        conflicts.push({
          type: 'PARAMETER_CONFLICT',
          severity: 'MEDIUM',
          description: 'Alpha=1.0 with low P0 may waste power for cell-center UEs',
          conflictingParams: ['alpha', 'p0NominalPUSCH'],
          recommendation: 'Consider reducing alpha to 0.7-0.9 for better efficiency',
        });
      }
    }

    return conflicts;
  }

  /**
   * Get safety threshold summary
   */
  getThresholds(): SafetyConfig {
    return { ...this.config };
  }
}
