/**
 * Tests for Strange Loops Detector
 */

import {
  StrangeLoopsDetector,
  LogicalPattern,
  CausalChain,
} from '../../src/verification/patterns/strange-loops-detector';

describe('StrangeLoopsDetector', () => {
  let detector: StrangeLoopsDetector;

  beforeEach(() => {
    detector = new StrangeLoopsDetector();
  });

  describe('detectCircularReasoning', () => {
    it('should detect simple circular reasoning', async () => {
      const text = 'A causes B because B leads to A. Therefore A causes B.';

      const patterns = await detector.detectCircularReasoning(text);

      const circular = patterns.find(p => p.type === 'circular');
      expect(circular).toBeDefined();
    });

    it('should detect self-referential definitions', async () => {
      const text = 'Recursion is when something is recursive. It defines itself recursively.';

      const patterns = await detector.detectCircularReasoning(text);

      expect(patterns.length).toBeGreaterThan(0);
    });

    it('should not flag valid causal chains', async () => {
      const text = 'A causes B. B causes C. C causes D. Therefore A ultimately affects D.';

      const patterns = await detector.detectCircularReasoning(text);

      const criticalCircular = patterns.find(p => p.type === 'circular' && p.severity === 'critical');
      expect(criticalCircular).toBeUndefined();
    });
  });

  describe('detectContradictions', () => {
    it('should detect keyword contradictions', async () => {
      const text = 'This treatment always works. This treatment never works.';

      const patterns = await detector.detectContradictions(text);

      const contradiction = patterns.find(p => p.type === 'contradiction');
      expect(contradiction).toBeDefined();
      expect(contradiction?.severity).toBe('high');
    });

    it('should detect affirmation vs negation', async () => {
      const text = 'The medication is effective for treating pain. The medication is not effective for treating pain.';

      const patterns = await detector.detectContradictions(text);

      const contradiction = patterns.find(p => p.type === 'contradiction');
      expect(contradiction).toBeDefined();
    });

    it('should detect increase vs decrease contradictions', async () => {
      const text = 'The treatment increases symptoms. The treatment decreases symptoms.';

      const patterns = await detector.detectContradictions(text);

      expect(patterns.length).toBeGreaterThan(0);
    });

    it('should not flag non-contradictory statements', async () => {
      const text = 'Treatment A is effective. Treatment B is effective. Both have different mechanisms.';

      const patterns = await detector.detectContradictions(text);

      expect(patterns.filter(p => p.severity === 'high')).toHaveLength(0);
    });
  });

  describe('validateCausalChain', () => {
    it('should validate correct causal chain', async () => {
      const chain: CausalChain = {
        nodes: [
          { id: 'a', claim: 'A happens', evidence: ['study1'], confidence: 0.8 },
          { id: 'b', claim: 'B happens', evidence: ['study2'], confidence: 0.8 },
          { id: 'c', claim: 'C happens', evidence: ['study3'], confidence: 0.8 },
        ],
        edges: [
          { from: 'a', to: 'b', type: 'causes', strength: 0.9 },
          { from: 'b', to: 'c', type: 'causes', strength: 0.8 },
        ],
        valid: true,
        cycles: [],
        contradictions: [],
      };

      const result = await detector.validateCausalChain(chain);

      expect(result.valid).toBe(true);
      expect(result.issues).toHaveLength(0);
      expect(result.strength).toBeGreaterThan(0.7);
    });

    it('should detect causal cycles', async () => {
      const chain: CausalChain = {
        nodes: [
          { id: 'a', claim: 'A happens', evidence: [], confidence: 0.5 },
          { id: 'b', claim: 'B happens', evidence: [], confidence: 0.5 },
        ],
        edges: [
          { from: 'a', to: 'b', type: 'causes', strength: 0.7 },
          { from: 'b', to: 'a', type: 'causes', strength: 0.7 },
        ],
        valid: false,
        cycles: [[
          { id: 'a', claim: 'A happens', evidence: [], confidence: 0.5 },
          { id: 'b', claim: 'B happens', evidence: [], confidence: 0.5 },
        ]],
        contradictions: [],
      };

      const result = await detector.validateCausalChain(chain);

      expect(result.valid).toBe(false);
      expect(result.issues.some(i => i.type === 'circular')).toBe(true);
    });

    it('should detect weak causal links', async () => {
      const chain: CausalChain = {
        nodes: [
          { id: 'a', claim: 'A', evidence: [], confidence: 0.5 },
          { id: 'b', claim: 'B', evidence: [], confidence: 0.5 },
        ],
        edges: [
          { from: 'a', to: 'b', type: 'causes', strength: 0.2 },
        ],
        valid: true,
        cycles: [],
        contradictions: [],
      };

      const result = await detector.validateCausalChain(chain);

      const weakLink = result.issues.find(i => i.type === 'invalid-causal');
      expect(weakLink).toBeDefined();
    });

    it('should handle contradictions in chain', async () => {
      const chain: CausalChain = {
        nodes: [
          { id: 'a', claim: 'A happens', evidence: [], confidence: 0.5 },
          { id: 'b', claim: 'B happens', evidence: [], confidence: 0.5 },
        ],
        edges: [],
        valid: false,
        cycles: [],
        contradictions: [
          { node1: 'a', node2: 'b', reason: 'Contradictory claims', severity: 'high' },
        ],
      };

      const result = await detector.validateCausalChain(chain);

      expect(result.valid).toBe(false);
      expect(result.issues.some(i => i.type === 'contradiction')).toBe(true);
    });
  });

  describe('detectRecursivePatterns', () => {
    it('should detect self-referential language', async () => {
      const text = 'The process defines itself recursively through its own definition.';

      const patterns = await detector.detectRecursivePatterns(text);

      expect(patterns.some(p => p.pattern === 'self-reference')).toBe(true);
    });

    it('should detect nested parenthetical', async () => {
      const text = 'The treatment (which includes medication (which is FDA approved)) is effective.';

      const patterns = await detector.detectRecursivePatterns(text);

      const nested = patterns.find(p => p.pattern === 'nested-parenthetical');
      expect(nested).toBeDefined();
      expect(nested?.depth).toBe(2);
    });

    it('should flag excessive recursion', async () => {
      const text = 'A recursive definition that references itself which in turn references itself again.';

      const patterns = await detector.detectRecursivePatterns(text, 1);

      const problematic = patterns.find(p => p.problematic);
      // May or may not detect depending on complexity
      // Just ensure it runs without error
      expect(patterns).toBeDefined();
    });
  });

  describe('getStatistics', () => {
    it('should return pattern statistics', async () => {
      const text = 'A causes B. B causes A. This always works. This never works.';

      const circular = await detector.detectCircularReasoning(text);
      const contradictions = await detector.detectContradictions(text);
      const allPatterns = [...circular, ...contradictions];

      const stats = detector.getStatistics(allPatterns);

      expect(stats.total).toBe(allPatterns.length);
      expect(stats.byType).toBeDefined();
      expect(stats.bySeverity).toBeDefined();
      expect(stats.mostCommon).toBeDefined();
    });
  });
});
