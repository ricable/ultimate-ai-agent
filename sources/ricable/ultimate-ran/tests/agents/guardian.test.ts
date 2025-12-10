/**
 * Guardian Agent Tests - London School TDD
 * TITAN Neuro-Symbolic RAN Platform
 *
 * Test-First Approach:
 * 1. Write comprehensive mocks for all dependencies
 * 2. Test behavior and interactions (mockist style)
 * 3. Verify Lyapunov chaos detection, safety thresholds, hallucination detection
 * 4. Mock E2B sandbox for digital twin simulation
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { Mock } from 'vitest';
import { GuardianAgent } from '../../src/agents/guardian/index';

// No module-level mocks needed - using dependency injection instead

describe('GuardianAgent - Safety Gatekeeper', () => {
  let guardian: GuardianAgent;
  let mockLyapunovAnalyzer: any;
  let mockDigitalTwin: any;
  let mockSafetyThresholds: any;

  beforeEach(() => {
    // Reset all mocks before each test
    vi.clearAllMocks();

    // Create mock implementations using plain objects
    mockLyapunovAnalyzer = {
      analyze: vi.fn().mockResolvedValue({
        exponent: -0.05,
        stable: true,
        interpretation: 'STABLE',
        reliable: true,
      }),
      calculateExponent: vi.fn().mockReturnValue(-0.05),
    };

    mockDigitalTwin = {
      simulate: vi.fn().mockResolvedValue({
        id: 'sim-test',
        steps: Array(100).fill(null).map((_, i) => ({
          step: i,
          kpis: { throughput: 50 + Math.random() * 10, bler: 0.05, interference: -105 },
        })),
      }),
      runPreCommitSimulation: vi.fn().mockResolvedValue({
        id: 'sim-test',
        sandboxId: 'sandbox-test',
        steps: Array(100).fill(null).map((_, i) => ({
          step: i,
          kpis: { throughput: 50 + Math.random() * 10, bler: 0.05, interference: -105 },
        })),
      }),
      createSandbox: vi.fn().mockResolvedValue('sandbox-test'),
      destroySandbox: vi.fn().mockResolvedValue(undefined),
    };

    mockSafetyThresholds = {
      validate: vi.fn().mockReturnValue({
        valid: true,
        violations: [],
      }),
      checkBLER: vi.fn().mockReturnValue(true),
      checkPower: vi.fn().mockReturnValue(true),
      checkInterference: vi.fn().mockReturnValue(true),
      detectHallucinations: vi.fn().mockReturnValue([]),
    };

    // Create Guardian instance with injected mocks (dependency injection)
    guardian = new GuardianAgent({
      id: 'guardian-test-001',
      thresholds: {
        lyapunov_max: 0.0,
        bler_max: 0.1,
        power_max_dbm: 46,
      },
      lyapunovAnalyzer: mockLyapunovAnalyzer as any,
      digitalTwin: mockDigitalTwin as any,
      safetyThresholds: mockSafetyThresholds as any,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Constructor and Initialization', () => {
    it('should create Guardian agent with correct type and role', () => {
      expect(guardian.type).toBe('guardian');
      expect(guardian.role).toBe('Adversarial Safety Agent');
    });

    it('should initialize with correct capabilities', () => {
      expect(guardian.capabilities).toContain('pre_commit_simulation');
      expect(guardian.capabilities).toContain('hallucination_detection');
      expect(guardian.capabilities).toContain('lyapunov_analysis');
      expect(guardian.capabilities).toContain('safety_verification');
    });

    it('should set safety thresholds from PRD', () => {
      expect(guardian.thresholds.lyapunov_max).toBe(0.0);
      expect(guardian.thresholds.bler_max).toBe(0.1);
      expect(guardian.thresholds.power_max_dbm).toBe(46);
    });

    it('should be able to analyze Lyapunov stability', async () => {
      // Just verify method exists (London School focuses on behavior not implementation)
      expect(guardian.analyzeLyapunovStability).toBeDefined();
    });

    it('should be able to run pre-commit simulation', () => {
      expect(guardian.runPreCommitSimulation).toBeDefined();
    });

    it('should be able to validate safety thresholds', () => {
      expect(guardian.validateSafetyThresholds).toBeDefined();
    });
  });

  describe('Lyapunov Stability Analysis', () => {
    it('should detect stable system when exponent <= 0', async () => {
      // Mock stable simulation result
      const mockSimulation = {
        id: 'sim-001',
        steps: [
          { kpis: { throughput: 100, bler: 0.01, interference: -105 } },
          { kpis: { throughput: 101, bler: 0.01, interference: -105 } },
          { kpis: { throughput: 100.5, bler: 0.01, interference: -105 } },
        ],
      };

      mockDigitalTwin.simulate.mockResolvedValue(mockSimulation);
      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: -0.15,
        stable: true,
        interpretation: 'STABLE',
        reliable: true,
      });

      const artifact = {
        id: 'artifact-001',
        code: 'p0 = -106;',
        parameters: { p0NominalPUSCH: -106 },
      };

      const result = await guardian.analyzeLyapunovStability(artifact);

      expect(mockDigitalTwin.simulate).toHaveBeenCalledWith(artifact);
      expect(mockLyapunovAnalyzer.analyze).toHaveBeenCalledWith(mockSimulation);
      expect(result.stable).toBe(true);
      expect(result.interpretation).toBe('STABLE');
    });

    it('should detect chaotic system when exponent > 0', async () => {
      // Mock chaotic simulation result
      const mockSimulation = {
        id: 'sim-002',
        steps: [
          { kpis: { throughput: 100, bler: 0.01, interference: -105 } },
          { kpis: { throughput: 150, bler: 0.03, interference: -100 } },
          { kpis: { throughput: 80, bler: 0.06, interference: -95 } },
          { kpis: { throughput: 200, bler: 0.09, interference: -90 } },
        ],
      };

      mockDigitalTwin.simulate.mockResolvedValue(mockSimulation);
      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: 0.42,
        stable: false,
        interpretation: 'CHAOTIC',
        reliable: true,
      });

      const artifact = {
        id: 'artifact-002',
        code: 'while(true) { power++; }',
        parameters: { p0NominalPUSCH: -90 },
      };

      const result = await guardian.analyzeLyapunovStability(artifact);

      expect(result.stable).toBe(false);
      expect(result.interpretation).toBe('CHAOTIC');
      expect(result.exponent).toBeGreaterThan(0);
    });

    it('should flag unreliable Lyapunov when simulation too short', async () => {
      const mockSimulation = {
        id: 'sim-003',
        steps: [
          { kpis: { throughput: 100, bler: 0.01, interference: -105 } },
          { kpis: { throughput: 101, bler: 0.01, interference: -105 } },
        ],
      };

      mockDigitalTwin.simulate.mockResolvedValue(mockSimulation);
      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: 0.05,
        stable: false,
        interpretation: 'CHAOTIC',
        reliable: false, // Not enough data points
        reason: 'Insufficient simulation steps (2 < 10 minimum)',
      });

      const artifact = {
        id: 'artifact-003',
        code: 'p0 = -106;',
        parameters: { p0NominalPUSCH: -106 },
      };

      const result = await guardian.analyzeLyapunovStability(artifact);

      expect(result.reliable).toBe(false);
      expect(result.reason).toContain('Insufficient simulation steps');
    });
  });

  describe('Digital Twin Pre-Commit Simulation', () => {
    it('should run simulation in E2B sandbox', async () => {
      const artifact = {
        id: 'artifact-004',
        code: 'p0 = -103; alpha = 0.8;',
        parameters: { p0NominalPUSCH: -103, alpha: 0.8 },
      };

      mockDigitalTwin.createSandbox.mockResolvedValue('sandbox-001');
      mockDigitalTwin.runPreCommitSimulation.mockResolvedValue({
        sandboxId: 'sandbox-001',
        steps: Array.from({ length: 100 }, (_, i) => ({
          step: i,
          kpis: { throughput: 100 + i * 0.1, bler: 0.01, interference: -105 },
        })),
      });

      const result = await guardian.runPreCommitSimulation(artifact);

      expect(mockDigitalTwin.createSandbox).toHaveBeenCalled();
      expect(mockDigitalTwin.runPreCommitSimulation).toHaveBeenCalledWith(
        'sandbox-001',
        artifact
      );
      expect(result.sandboxId).toBe('sandbox-001');
      expect(result.steps).toHaveLength(100);
    });

    it('should destroy sandbox after simulation', async () => {
      const artifact = {
        id: 'artifact-005',
        code: 'p0 = -106;',
        parameters: { p0NominalPUSCH: -106 },
      };

      mockDigitalTwin.createSandbox.mockResolvedValue('sandbox-002');
      mockDigitalTwin.runPreCommitSimulation.mockResolvedValue({
        sandboxId: 'sandbox-002',
        steps: [],
      });
      mockDigitalTwin.destroySandbox.mockResolvedValue(true);

      await guardian.runPreCommitSimulation(artifact);

      expect(mockDigitalTwin.destroySandbox).toHaveBeenCalledWith('sandbox-002');
    });

    it('should handle sandbox creation failure gracefully', async () => {
      const artifact = {
        id: 'artifact-006',
        code: 'p0 = -106;',
        parameters: { p0NominalPUSCH: -106 },
      };

      mockDigitalTwin.createSandbox.mockRejectedValue(
        new Error('E2B sandbox creation failed')
      );

      await expect(guardian.runPreCommitSimulation(artifact)).rejects.toThrow(
        'E2B sandbox creation failed'
      );
    });
  });

  describe('Safety Threshold Validation', () => {
    it('should reject parameters exceeding BLER threshold', async () => {
      const artifact = {
        id: 'artifact-007',
        code: 'bler = 0.15;',
        parameters: { targetBLER: 0.15 },
      };

      mockSafetyThresholds.validate.mockReturnValue({
        valid: false,
        violations: [{
          type: 'BLER_EXCEEDED',
          threshold: 0.1,
          actual: 0.15,
          description: 'BLER exceeds 10% limit',
        }],
      });

      const result = await guardian.validateSafetyThresholds(artifact);

      expect(mockSafetyThresholds.validate).toHaveBeenCalledWith(artifact);
      expect(result.valid).toBe(false);
      expect(result.violations).toHaveLength(1);
      expect(result.violations[0].type).toBe('BLER_EXCEEDED');
    });

    it('should reject parameters exceeding power threshold', async () => {
      const artifact = {
        id: 'artifact-008',
        code: 'power = 50;',
        parameters: { maxPower: 50 },
      };

      mockSafetyThresholds.validate.mockReturnValue({
        valid: false,
        violations: [{
          type: 'POWER_EXCEEDED',
          threshold: 46,
          actual: 50,
          description: 'Power exceeds 46 dBm limit',
        }],
      });

      const result = await guardian.validateSafetyThresholds(artifact);

      expect(result.valid).toBe(false);
      expect(result.violations).toContainEqual(
        expect.objectContaining({
          type: 'POWER_EXCEEDED',
          threshold: 46,
          actual: 50,
        })
      );
    });

    it('should accept parameters within all safety thresholds', async () => {
      const artifact = {
        id: 'artifact-009',
        code: 'p0 = -106; alpha = 0.8;',
        parameters: { p0NominalPUSCH: -106, alpha: 0.8 },
      };

      mockSafetyThresholds.checkBLER.mockReturnValue({ valid: true });
      mockSafetyThresholds.checkPower.mockReturnValue({ valid: true });
      mockSafetyThresholds.checkInterference.mockReturnValue({ valid: true });
      mockSafetyThresholds.validate.mockReturnValue({
        valid: true,
        violations: [],
      });

      const result = await guardian.validateSafetyThresholds(artifact);

      expect(result.valid).toBe(true);
      expect(result.violations).toHaveLength(0);
    });

    it('should validate interference constraints', async () => {
      const artifact = {
        id: 'artifact-010',
        code: 'interference = -90;',
        parameters: { interferenceLevel: -90 },
      };

      mockSafetyThresholds.validate.mockReturnValue({
        valid: false,
        violations: [{
          type: 'INTERFERENCE_HIGH',
          threshold: -105,
          actual: -90,
          description: 'Interference level too high',
        }],
      });

      const result = await guardian.validateSafetyThresholds(artifact);

      expect(result.valid).toBe(false);
      expect(result.violations).toContainEqual(
        expect.objectContaining({
          type: 'INTERFERENCE_HIGH',
        })
      );
    });
  });

  describe('Hallucination Detection', () => {
    it('should detect infinite power loop hallucination', async () => {
      const artifact = {
        id: 'artifact-011',
        code: 'while (true) { power++; }',
        parameters: {},
      };

      mockSafetyThresholds.detectHallucinations.mockReturnValue([
        {
          type: 'INFINITE_POWER_LOOP',
          severity: 'CRITICAL',
          description: 'Code contains unbounded power increase',
          line: 1,
        },
      ]);

      const result = await guardian.detectHallucinations(artifact);

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('INFINITE_POWER_LOOP');
      expect(result[0].severity).toBe('CRITICAL');
    });

    it('should detect physically invalid parameters', async () => {
      const artifact = {
        id: 'artifact-012',
        code: 'p0 = -50;', // Too high for target received power
        parameters: { p0NominalPUSCH: -50 },
      };

      mockSafetyThresholds.detectHallucinations.mockReturnValue([
        {
          type: 'PHYSICS_VIOLATION',
          severity: 'HIGH',
          description: 'P0 value -50 dBm is physically invalid (range: -130 to -70)',
          parameter: 'p0NominalPUSCH',
        },
      ]);

      const result = await guardian.detectHallucinations(artifact);

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('PHYSICS_VIOLATION');
      expect(result[0].parameter).toBe('p0NominalPUSCH');
    });

    it('should detect missing safety bounds', async () => {
      const artifact = {
        id: 'artifact-013',
        code: 'power = userInput;', // No bounds checking
        parameters: {},
      };

      mockSafetyThresholds.detectHallucinations.mockReturnValue([
        {
          type: 'MISSING_SAFETY_BOUNDS',
          severity: 'MEDIUM',
          description: 'Code lacks explicit safety boundary checks',
          recommendation: 'Add min/max validation',
        },
      ]);

      const result = await guardian.detectHallucinations(artifact);

      expect(result).toHaveLength(1);
      expect(result[0].type).toBe('MISSING_SAFETY_BOUNDS');
      expect(result[0].severity).toBe('MEDIUM');
    });

    it('should detect parameter conflicts', async () => {
      const artifact = {
        id: 'artifact-014',
        code: 'alpha = 0.0; p0 = -70;', // Conflicting settings
        parameters: { alpha: 0.0, p0NominalPUSCH: -70 },
      };

      mockSafetyThresholds.detectHallucinations.mockReturnValue([
        {
          type: 'PARAMETER_CONFLICT',
          severity: 'HIGH',
          description: 'Alpha=0 with high P0 creates cell-edge coverage issues',
          conflictingParams: ['alpha', 'p0NominalPUSCH'],
        },
      ]);

      const result = await guardian.detectHallucinations(artifact);

      expect(result[0].type).toBe('PARAMETER_CONFLICT');
      expect(result[0].conflictingParams).toContain('alpha');
      expect(result[0].conflictingParams).toContain('p0NominalPUSCH');
    });

    it('should return empty array for safe code', async () => {
      const artifact = {
        id: 'artifact-015',
        code: 'p0 = Math.max(-110, Math.min(-100, value));',
        parameters: { p0NominalPUSCH: -106 },
      };

      mockSafetyThresholds.detectHallucinations.mockReturnValue([]);

      const result = await guardian.detectHallucinations(artifact);

      expect(result).toHaveLength(0);
    });
  });

  describe('End-to-End Safety Audit', () => {
    it('should approve safe artifact passing all checks', async () => {
      const artifact = {
        id: 'artifact-016',
        code: 'p0 = -106; alpha = 0.8;',
        parameters: { p0NominalPUSCH: -106, alpha: 0.8 },
      };

      // Mock all checks passing
      const mockSimulation = {
        id: 'sim-016',
        sandboxId: 'sandbox-016',
        steps: Array.from({ length: 100 }, (_, i) => ({
          step: i,
          kpis: { throughput: 100, bler: 0.01, interference: -105 },
        })),
      };

      // Mock Digital Twin methods
      mockDigitalTwin.createSandbox.mockResolvedValue('sandbox-016');
      mockDigitalTwin.runPreCommitSimulation.mockResolvedValue(mockSimulation);
      mockDigitalTwin.destroySandbox.mockResolvedValue(true);
      mockDigitalTwin.simulate.mockResolvedValue(mockSimulation);

      // Mock Lyapunov Analyzer
      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: -0.1,
        stable: true,
        interpretation: 'STABLE',
        reliable: true,
      });

      // Mock Safety Thresholds
      mockSafetyThresholds.validate.mockReturnValue({
        valid: true,
        violations: [],
      });

      mockSafetyThresholds.detectHallucinations.mockReturnValue([]);

      const result = await guardian.processTask({ artifact });

      expect(result.approved).toBe(true);
      expect(result.lyapunovResult.stable).toBe(true);
      expect(result.hallucinations).toHaveLength(0);
      expect(result.safetyValidation.valid).toBe(true);
    });

    it('should reject artifact with chaotic behavior', async () => {
      const artifact = {
        id: 'artifact-017',
        code: 'for(let i=0; i<100; i++) { power += 10; }',
        parameters: {},
      };

      mockDigitalTwin.simulate.mockResolvedValue({
        id: 'sim-017',
        steps: Array.from({ length: 100 }, (_, i) => ({
          step: i,
          kpis: { throughput: 100 * Math.pow(2, i / 10), bler: 0.01, interference: -105 },
        })),
      });

      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: 0.8,
        stable: false,
        interpretation: 'CHAOTIC',
        reliable: true,
      });

      mockSafetyThresholds.validate.mockReturnValue({ valid: true, violations: [] });
      mockSafetyThresholds.detectHallucinations.mockReturnValue([]);

      const result = await guardian.processTask({ artifact });

      expect(result.approved).toBe(false);
      expect(result.rejectionReason).toContain('CHAOTIC');
    });

    it('should reject artifact with critical hallucinations', async () => {
      const artifact = {
        id: 'artifact-018',
        code: 'while(true) { power = 100; }',
        parameters: {},
      };

      mockDigitalTwin.simulate.mockResolvedValue({
        id: 'sim-018',
        steps: [],
      });

      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: -0.1,
        stable: true,
        interpretation: 'STABLE',
        reliable: true,
      });

      mockSafetyThresholds.validate.mockReturnValue({ valid: true, violations: [] });
      mockSafetyThresholds.detectHallucinations.mockReturnValue([
        {
          type: 'INFINITE_POWER_LOOP',
          severity: 'CRITICAL',
          description: 'Infinite loop with power modification',
        },
      ]);

      const result = await guardian.processTask({ artifact });

      expect(result.approved).toBe(false);
      expect(result.rejectionReason).toContain('CRITICAL');
      expect(result.hallucinations).toHaveLength(1);
    });

    it('should reject artifact violating safety thresholds', async () => {
      const artifact = {
        id: 'artifact-019',
        code: 'power = 60; bler = 0.2;',
        parameters: { maxPower: 60, targetBLER: 0.2 },
      };

      mockDigitalTwin.simulate.mockResolvedValue({
        id: 'sim-019',
        steps: [],
      });

      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: -0.1,
        stable: true,
        interpretation: 'STABLE',
        reliable: true,
      });

      mockSafetyThresholds.validate.mockReturnValue({
        valid: false,
        violations: [
          { type: 'POWER_EXCEEDED', threshold: 46, actual: 60 },
          { type: 'BLER_EXCEEDED', threshold: 0.1, actual: 0.2 },
        ],
      });

      mockSafetyThresholds.detectHallucinations.mockReturnValue([]);

      const result = await guardian.processTask({ artifact });

      expect(result.approved).toBe(false);
      expect(result.safetyValidation.valid).toBe(false);
      expect(result.safetyValidation.violations).toHaveLength(2);
    });
  });

  describe('AG-UI Event Emission', () => {
    it('should emit agent_message event on approval', async () => {
      const emitSpy = vi.spyOn(guardian, 'emitAGUI');

      const artifact = {
        id: 'artifact-020',
        code: 'p0 = -106;',
        parameters: { p0NominalPUSCH: -106 },
      };

      // Mock all passing
      mockDigitalTwin.simulate.mockResolvedValue({ id: 'sim-020', steps: [] });
      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: -0.1,
        stable: true,
        interpretation: 'STABLE',
        reliable: true,
      });
      mockSafetyThresholds.validate.mockReturnValue({ valid: true, violations: [] });
      mockSafetyThresholds.detectHallucinations.mockReturnValue([]);

      await guardian.processTask({ artifact });

      expect(emitSpy).toHaveBeenCalledWith(
        'agent_message',
        expect.objectContaining({
          type: 'markdown',
          content: expect.stringContaining('APPROVED'),
        })
      );
    });

    it('should emit agent_message event on rejection', async () => {
      const emitSpy = vi.spyOn(guardian, 'emitAGUI');

      const artifact = {
        id: 'artifact-021',
        code: 'while(true) {}',
        parameters: {},
      };

      // Mock rejection
      mockDigitalTwin.simulate.mockResolvedValue({ id: 'sim-021', steps: [] });
      mockLyapunovAnalyzer.analyze.mockResolvedValue({
        exponent: 0.5,
        stable: false,
        interpretation: 'CHAOTIC',
        reliable: true,
      });
      mockSafetyThresholds.validate.mockReturnValue({ valid: true, violations: [] });
      mockSafetyThresholds.detectHallucinations.mockReturnValue([]);

      await guardian.processTask({ artifact });

      expect(emitSpy).toHaveBeenCalledWith(
        'agent_message',
        expect.objectContaining({
          type: 'markdown',
          content: expect.stringContaining('REJECTED'),
        })
      );
    });
  });
});
