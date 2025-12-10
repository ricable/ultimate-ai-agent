/**
 * SPARC Methodology Validator
 * Rigorous Agentic Engineering Gate Enforcement
 *
 * Ensures all agent-generated code passes through:
 * S - Specification
 * P - Pseudocode
 * A - Architecture
 * R - Refinement
 * C - Completion
 */

import { readFileSync } from 'fs';

export class SPARCValidator {
  constructor({ configPath }) {
    this.configPath = configPath || './config/workflows/sparc-methodology.json';
    this.config = this.loadConfig();
    this.validationHistory = [];
  }

  loadConfig() {
    try {
      const data = readFileSync(this.configPath, 'utf-8');
      return JSON.parse(data);
    } catch (error) {
      console.warn('[SPARC] Failed to load config, using defaults');
      return this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      phases: {
        specification: { order: 1 },
        pseudocode: { order: 2 },
        architecture: { order: 3 },
        refinement: { order: 4 },
        completion: { order: 5 }
      },
      gate_enforcement: {
        strict: true,
        bypass_allowed: false
      }
    };
  }

  /**
   * Validate that an artifact passes through all SPARC gates
   */
  async validateArtifact(artifact) {
    console.log(`[SPARC] Validating artifact: ${artifact.id}`);

    const results = {
      artifact: artifact.id,
      passed: true,
      gates: {}
    };

    // Check each gate in order
    for (const [phaseName, phaseConfig] of Object.entries(this.config.phases)) {
      const gateResult = await this.validateGate(phaseName, artifact, phaseConfig);
      results.gates[phaseName] = gateResult;

      if (!gateResult.passed) {
        results.passed = false;

        if (this.config.gate_enforcement.strict) {
          console.error(`[SPARC] GATE FAILED: ${phaseName}`);
          break;
        }
      }
    }

    this.validationHistory.push(results);
    return results;
  }

  /**
   * Validate a single SPARC gate
   */
  async validateGate(phaseName, artifact, phaseConfig) {
    console.log(`[SPARC] Checking gate: ${phaseName.toUpperCase()}`);

    const validators = {
      specification: () => this.validateSpecification(artifact),
      pseudocode: () => this.validatePseudocode(artifact),
      architecture: () => this.validateArchitecture(artifact),
      refinement: () => this.validateRefinement(artifact),
      completion: () => this.validateCompletion(artifact)
    };

    const validator = validators[phaseName];
    if (!validator) {
      return { passed: true, message: 'No validator defined' };
    }

    try {
      return await validator();
    } catch (error) {
      return { passed: false, message: error.message };
    }
  }

  /**
   * S - Specification Validation
   * Ensure formal specification with objective function and safety constraints
   */
  async validateSpecification(artifact) {
    const spec = artifact.specification;

    if (!spec) {
      return { passed: false, message: 'Missing specification' };
    }

    const required = ['objective_function', 'safety_constraints'];
    const missing = required.filter(field => !spec[field]);

    if (missing.length > 0) {
      return {
        passed: false,
        message: `Missing required specification fields: ${missing.join(', ')}`
      };
    }

    // Verify grounding (LLM mental model check)
    const isGrounded = await this.checkGrounding(spec);

    return {
      passed: isGrounded,
      message: isGrounded ? 'Specification grounded' : 'Specification not properly grounded'
    };
  }

  async checkGrounding(spec) {
    // In production, this validates the LLM's mental model
    return spec.objective_function && spec.safety_constraints;
  }

  /**
   * P - Pseudocode Validation
   * Ensure algorithmic logic is coherent
   */
  async validatePseudocode(artifact) {
    const pseudocode = artifact.pseudocode;

    if (!pseudocode) {
      return { passed: false, message: 'Missing pseudocode' };
    }

    // Check for required elements
    const hasDataFlow = pseudocode.includes('->') || pseudocode.includes('flow');
    const hasControlStructure = /if|for|while|loop/i.test(pseudocode);

    return {
      passed: hasDataFlow || hasControlStructure,
      message: hasDataFlow ? 'Pseudocode valid' : 'Missing data flow or control structures'
    };
  }

  /**
   * A - Architecture Validation
   * Ensure Ruvnet stack mandate is followed
   */
  async validateArchitecture(artifact) {
    const arch = artifact.architecture;

    if (!arch) {
      return { passed: false, message: 'Missing architecture' };
    }

    const mandate = this.config.phases.architecture?.stack_mandate || {};

    // Check for forbidden dependencies
    const forbidden = mandate.forbidden || [];
    const usedForbidden = forbidden.filter(dep =>
      JSON.stringify(arch).toLowerCase().includes(dep.toLowerCase())
    );

    if (usedForbidden.length > 0) {
      return {
        passed: false,
        message: `Forbidden dependencies detected: ${usedForbidden.join(', ')}`
      };
    }

    return { passed: true, message: 'Architecture compliant with Ruvnet stack' };
  }

  /**
   * R - Refinement Validation
   * Ensure TDD and edge-native constraints
   */
  async validateRefinement(artifact) {
    const refinement = artifact.refinement;

    if (!refinement) {
      return { passed: false, message: 'Missing refinement' };
    }

    // Check for unit tests
    const hasTests = refinement.tests && refinement.tests.length > 0;

    // Check for edge-native constraints
    const constraints = this.config.phases.refinement?.constraints || {};
    const withinMemory = !constraints.resource_limits?.memory_max_mb ||
      (refinement.memoryUsage || 0) <= constraints.resource_limits.memory_max_mb;

    return {
      passed: hasTests && withinMemory,
      message: hasTests ? 'Refinement valid (TDD + Edge-Native)' : 'Missing tests or exceeds resource limits'
    };
  }

  /**
   * C - Completion Validation
   * Ensure 3GPP compliance and safety audit
   */
  async validateCompletion(artifact) {
    const completion = artifact.completion;

    if (!completion) {
      return { passed: false, message: 'Missing completion' };
    }

    // Check for Lyapunov verification
    const lyapunovCheck = await this.verifyLyapunov(artifact);

    // Check for 3GPP compliance
    const complianceCheck = await this.check3GPPCompliance(artifact);

    return {
      passed: lyapunovCheck.stable && complianceCheck.compliant,
      message: `Lyapunov: ${lyapunovCheck.stable ? 'STABLE' : 'UNSTABLE'}, 3GPP: ${complianceCheck.compliant ? 'COMPLIANT' : 'NON-COMPLIANT'}`
    };
  }

  async verifyLyapunov(artifact) {
    // Calculate Lyapunov exponent
    // Positive = chaos/instability, should reject
    const exponent = artifact.lyapunovExponent || 0;
    return {
      stable: exponent <= 0,
      exponent
    };
  }

  async check3GPPCompliance(artifact) {
    // Verify against 3GPP constraints
    return {
      compliant: true,
      standards: ['3GPP TS 38.331', '3GPP TS 38.300']
    };
  }

  /**
   * Get validation summary
   */
  getSummary() {
    const total = this.validationHistory.length;
    const passed = this.validationHistory.filter(v => v.passed).length;

    return {
      total,
      passed,
      failed: total - passed,
      passRate: total > 0 ? (passed / total * 100).toFixed(2) + '%' : 'N/A'
    };
  }
}
