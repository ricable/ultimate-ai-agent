/**
 * Integration tests for complete verification system
 */

import { VerificationSystem } from '../../src/verification';

describe('VerificationSystem Integration', () => {
  let system: VerificationSystem;

  beforeEach(() => {
    system = new VerificationSystem();
  });

  describe('end-to-end verification', () => {
    it('should perform complete verification workflow', async () => {
      const input = {
        claim: 'Studies suggest this treatment may improve outcomes in certain patient populations',
        citations: [
          {
            id: 'cit1',
            type: 'meta-analysis',
            title: 'Meta-analysis of treatment efficacy',
            year: 2023,
            citationCount: 500,
            impactFactor: 15,
            evidenceLevel: 'A',
            doi: '10.1234/test',
          },
          {
            id: 'cit2',
            type: 'clinical-trial',
            title: 'Randomized controlled trial',
            year: 2022,
            citationCount: 300,
            impactFactor: 10,
            evidenceLevel: 'A',
          },
        ],
        context: { sampleSize: 1000 },
        features: {
          citationCount: 2,
          peerReviewedRatio: 1.0,
          recencyScore: 0.95,
          evidenceLevelScore: 1.0,
          contradictionCount: 0,
          hallucinationFlags: 0,
          textLength: 100,
          quantitativeClaims: 0,
        },
        metadata: {
          timestamp: Date.now(),
          source: 'test',
        },
      };

      const result = await system.verify(input);

      expect(result).toBeDefined();
      expect(result.verified).toBeDefined();
      expect(result.confidence).toBeDefined();
      expect(result.hallucinations).toBeDefined();
      expect(result.logicalPatterns).toBeDefined();
      expect(result.contradictions).toBeDefined();
      expect(result.learningAdjustment).toBeDefined();
    });

    it('should detect and flag problematic claims', async () => {
      const input = {
        claim: 'This treatment always cures all conditions permanently and is 100% guaranteed to work with absolutely no side effects',
        citations: [],
        context: {},
        features: {
          citationCount: 0,
          peerReviewedRatio: 0,
          recencyScore: 0,
          evidenceLevelScore: 0,
          contradictionCount: 0,
          hallucinationFlags: 5,
          textLength: 100,
          quantitativeClaims: 1,
        },
      };

      const result = await system.verify(input);

      expect(result.verified).toBe(false);
      expect(result.confidence.overall).toBeLessThan(0.5);
      expect(result.hallucinations.length).toBeGreaterThan(0);
      expect(result.requiresReview).toBe(true);
    });

    it('should integrate all verification components', async () => {
      const input = {
        claim: 'Based on multiple high-quality studies, evidence suggests a moderate effect',
        citations: [
          {
            id: 'cit1',
            type: 'meta-analysis',
            title: 'Meta-analysis',
            year: 2023,
            citationCount: 500,
            evidenceLevel: 'A',
          },
        ],
        features: {
          citationCount: 1,
          peerReviewedRatio: 1.0,
          recencyScore: 0.9,
          evidenceLevelScore: 1.0,
          contradictionCount: 0,
          hallucinationFlags: 0,
          textLength: 80,
          quantitativeClaims: 0,
        },
        context: ['evidence-based-medicine'],
      };

      const result = await system.verify(input);

      // Should pass all checks
      expect(result.verified).toBe(true);
      expect(result.confidence.overall).toBeGreaterThan(0.7);

      // Should have minimal issues
      const criticalIssues = [
        ...result.hallucinations.filter((h: any) => h.severity === 'critical'),
        ...result.logicalPatterns.filter((p: any) => p.severity === 'critical'),
      ];
      expect(criticalIssues).toHaveLength(0);
    });
  });

  describe('system statistics', () => {
    it('should provide comprehensive statistics', () => {
      const stats = system.getStatistics();

      expect(stats).toBeDefined();
      expect(stats.pipeline).toBeDefined();
      expect(stats.patterns).toBeDefined();
      expect(stats.model).toBeDefined();
    });
  });

  describe('circular reasoning detection', () => {
    it('should detect circular reasoning in verification workflow', async () => {
      const input = {
        claim: 'A causes B because B leads to A, which then causes B again',
        citations: [],
        features: {
          citationCount: 0,
          peerReviewedRatio: 0,
          recencyScore: 0,
          evidenceLevelScore: 0,
          contradictionCount: 0,
          hallucinationFlags: 0,
          textLength: 60,
          quantitativeClaims: 0,
        },
      };

      const result = await system.verify(input);

      expect(result.logicalPatterns.length).toBeGreaterThan(0);
      const circular = result.logicalPatterns.find((p: any) => p.type === 'circular');
      expect(circular).toBeDefined();
    });
  });

  describe('contradiction detection', () => {
    it('should detect contradictions in verification workflow', async () => {
      const input = {
        claim: 'The treatment is always effective and never works',
        citations: [],
        features: {
          citationCount: 0,
          peerReviewedRatio: 0,
          recencyScore: 0,
          evidenceLevelScore: 0,
          contradictionCount: 0,
          hallucinationFlags: 0,
          textLength: 50,
          quantitativeClaims: 0,
        },
      };

      const result = await system.verify(input);

      expect(result.contradictions.length).toBeGreaterThan(0);
    });
  });

  describe('learning adjustment integration', () => {
    it('should apply learning adjustments to confidence', async () => {
      const input = {
        claim: 'Well-supported medical claim',
        citations: [
          {
            id: 'cit1',
            type: 'meta-analysis',
            title: 'Meta-analysis',
            year: 2023,
            citationCount: 500,
            evidenceLevel: 'A',
          },
        ],
        features: {
          citationCount: 1,
          peerReviewedRatio: 1.0,
          recencyScore: 0.9,
          evidenceLevelScore: 1.0,
          contradictionCount: 0,
          hallucinationFlags: 0,
          textLength: 30,
          quantitativeClaims: 0,
        },
        context: [],
      };

      const result = await system.verify(input);

      expect(result.learningAdjustment).toBeDefined();
      expect(result.learningAdjustment.adjustment).toBeGreaterThanOrEqual(-0.3);
      expect(result.learningAdjustment.adjustment).toBeLessThanOrEqual(0.3);
    });
  });
});
