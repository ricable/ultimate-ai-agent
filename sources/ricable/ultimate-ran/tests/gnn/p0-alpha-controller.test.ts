/**
 * P0/Alpha Parameter Controller Test Suite - London School TDD
 * TITAN Neuro-Symbolic RAN Platform
 *
 * Tests parameter optimization within 3GPP ranges
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { createMockCellNode, TEST_CONFIG } from '../setup';
import type { CellNode } from '../../src/gnn/types';

// Import to be implemented
import { P0AlphaController } from '../../src/gnn/p0-alpha-controller';

describe('P0AlphaController', () => {
  let controller: P0AlphaController;

  beforeEach(() => {
    controller = new P0AlphaController();
  });

  describe('P0 Optimization (-130 to -70 dBm)', () => {
    it('should optimize P0 within 3GPP range', () => {
      const cell = createMockCellNode({ p0: -106 });
      const optimizedP0 = controller.optimizeP0(cell);

      expect(optimizedP0).toBeGreaterThanOrEqual(-130);
      expect(optimizedP0).toBeLessThanOrEqual(-70);
    });

    it('should decrease P0 for low SINR cells', () => {
      const lowSinrCell = createMockCellNode({
        features: [5.0, -100.0, 80.0, 8.0], // SINR = 5 dB
        p0: -106,
      });

      const optimizedP0 = controller.optimizeP0(lowSinrCell);

      expect(optimizedP0).toBeLessThan(-106);
      expect(optimizedP0).toBeGreaterThanOrEqual(-130);
    });

    it('should increase P0 for high SINR cells', () => {
      const highSinrCell = createMockCellNode({
        features: [15.0, -90.0, 50.0, 15.0], // SINR = 15 dB
        p0: -106,
      });

      const optimizedP0 = controller.optimizeP0(highSinrCell);

      expect(optimizedP0).toBeGreaterThan(-106);
      expect(optimizedP0).toBeLessThanOrEqual(-70);
    });

    it('should clamp P0 at boundaries', () => {
      const extremeCell = createMockCellNode({ p0: -135 }); // Below min
      const optimizedP0 = controller.optimizeP0(extremeCell);

      expect(optimizedP0).toBe(-130);
    });

    it('should consider PRB utilization in P0 optimization', () => {
      const highLoadCell = createMockCellNode({
        features: [10.0, -95.0, 95.0, 12.0], // 95% PRB usage
        p0: -106,
      });

      const optimizedP0 = controller.optimizeP0(highLoadCell);

      // High load should reduce P0 to decrease interference
      expect(optimizedP0).toBeLessThan(-106);
    });
  });

  describe('Alpha Optimization (0 to 1)', () => {
    it('should optimize Alpha within valid range', () => {
      const cell = createMockCellNode({ alpha: 0.8 });
      const optimizedAlpha = controller.optimizeAlpha(cell);

      expect(optimizedAlpha).toBeGreaterThanOrEqual(0);
      expect(optimizedAlpha).toBeLessThanOrEqual(1);
    });

    it('should decrease Alpha for cell-edge users', () => {
      const cellEdgeCell = createMockCellNode({
        features: [6.0, -105.0, 75.0, 9.0], // Low SINR, high pathloss
        alpha: 0.85,
      });

      const optimizedAlpha = controller.optimizeAlpha(cellEdgeCell);

      expect(optimizedAlpha).toBeLessThan(0.85);
      expect(optimizedAlpha).toBeGreaterThan(0); // Partial compensation
    });

    it('should increase Alpha for cell-center users', () => {
      const cellCenterCell = createMockCellNode({
        features: [14.0, -88.0, 55.0, 15.0], // High SINR, low pathloss
        alpha: 0.75,
      });

      const optimizedAlpha = controller.optimizeAlpha(cellCenterCell);

      expect(optimizedAlpha).toBeGreaterThan(0.75);
    });

    it('should balance between coverage and interference', () => {
      const cell = createMockCellNode({ alpha: 0.8 });
      const optimizedAlpha = controller.optimizeAlpha(cell);

      // Should not be at extremes
      expect(optimizedAlpha).toBeGreaterThan(0.5);
      expect(optimizedAlpha).toBeLessThan(0.95);
    });
  });

  describe('Joint P0/Alpha Optimization', () => {
    it('should optimize both parameters jointly', () => {
      const cell = createMockCellNode({ p0: -106, alpha: 0.8 });
      const result = controller.optimizeJoint(cell);

      expect(result.p0).toBeDefined();
      expect(result.alpha).toBeDefined();
      expect(result.p0).toBeGreaterThanOrEqual(-130);
      expect(result.p0).toBeLessThanOrEqual(-70);
      expect(result.alpha).toBeGreaterThanOrEqual(0);
      expect(result.alpha).toBeLessThanOrEqual(1);
    });

    it('should maintain P0/Alpha consistency', () => {
      const cell = createMockCellNode({
        features: [8.0, -98.0, 70.0, 11.0],
        p0: -106,
        alpha: 0.8,
      });

      const result = controller.optimizeJoint(cell);

      // If P0 decreases, Alpha should typically decrease too
      if (result.p0 < -106) {
        expect(result.alpha).toBeLessThanOrEqual(0.85);
      }
    });

    it('should provide optimization rationale', () => {
      const cell = createMockCellNode({ p0: -106, alpha: 0.8 });
      const result = controller.optimizeJoint(cell);

      expect(result.rationale).toBeDefined();
      expect(result.rationale).toMatch(/P0|Alpha|SINR|interference/i);
    });
  });

  describe('Parameter Validation', () => {
    it('should validate P0 range', () => {
      const validation = controller.validateParameters(-106, 0.8);

      expect(validation.valid).toBe(true);
      expect(validation.violations).toHaveLength(0);
    });

    it('should reject P0 below minimum', () => {
      const validation = controller.validateParameters(-135, 0.8);

      expect(validation.valid).toBe(false);
      expect(validation.violations).toContain('P0 below -130 dBm');
    });

    it('should reject P0 above maximum', () => {
      const validation = controller.validateParameters(-65, 0.8);

      expect(validation.valid).toBe(false);
      expect(validation.violations).toContain('P0 above -70 dBm');
    });

    it('should reject Alpha below 0', () => {
      const validation = controller.validateParameters(-106, -0.1);

      expect(validation.valid).toBe(false);
      expect(validation.violations).toContain('Alpha below 0');
    });

    it('should reject Alpha above 1', () => {
      const validation = controller.validateParameters(-106, 1.2);

      expect(validation.valid).toBe(false);
      expect(validation.violations).toContain('Alpha above 1');
    });
  });

  describe('Optimization Strategies', () => {
    it('should support gradient-based optimization', () => {
      const cell = createMockCellNode({ p0: -106, alpha: 0.8 });
      const result = controller.optimizeJoint(cell, { strategy: 'gradient' });

      expect(result.strategy).toBe('gradient');
    });

    it('should support rule-based optimization', () => {
      const cell = createMockCellNode({ p0: -106, alpha: 0.8 });
      const result = controller.optimizeJoint(cell, { strategy: 'rules' });

      expect(result.strategy).toBe('rules');
    });

    it('should support hybrid optimization', () => {
      const cell = createMockCellNode({ p0: -106, alpha: 0.8 });
      const result = controller.optimizeJoint(cell, { strategy: 'hybrid' });

      expect(result.strategy).toBe('hybrid');
      expect(result.confidence).toBeGreaterThan(0.5);
    });
  });

  describe('Historical Learning', () => {
    it('should learn from past optimizations', () => {
      const cell = createMockCellNode({ p0: -106, alpha: 0.8 });

      // Simulate successful optimization history (need >4 items for >0.7 confidence)
      for (let i = 0; i < 5; i++) {
        controller.recordOutcome(cell, {
          p0: -103,
          alpha: 0.75,
          sinrDelta: 2.5, // Improvement
        });
      }

      const nextOptimization = controller.optimizeJoint(cell);
      expect(nextOptimization.confidence).toBeGreaterThan(0.7);
    });

    it('should avoid unsuccessful parameter combinations', () => {
      const cell = createMockCellNode({ p0: -106, alpha: 0.8 });

      // Simulate failed optimization
      controller.recordOutcome(cell, {
        p0: -110,
        alpha: 0.9,
        sinrDelta: -1.5, // Degradation
      });

      const nextOptimization = controller.optimizeJoint(cell);

      // Should not recommend similar parameters
      expect(
        Math.abs(nextOptimization.p0 - (-110)) > 2 ||
        Math.abs(nextOptimization.alpha - 0.9) > 0.05
      ).toBe(true);
    });
  });
});
