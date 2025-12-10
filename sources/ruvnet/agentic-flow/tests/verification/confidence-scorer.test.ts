/**
 * Tests for Confidence Scorer
 */

import {
  ConfidenceScorer,
  MedicalCitation,
  ConfidenceScore,
} from '../../src/verification/core/confidence-scorer';

describe('ConfidenceScorer', () => {
  let scorer: ConfidenceScorer;

  beforeEach(() => {
    scorer = new ConfidenceScorer();
  });

  describe('calculateConfidence', () => {
    it('should return low confidence with no citations', async () => {
      const score = await scorer.calculateConfidence('Some claim', [], {});

      expect(score.overall).toBeLessThan(0.3);
      expect(score.citationStrength).toBe(0);
    });

    it('should return high confidence with quality citations', async () => {
      const citations: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'meta-analysis',
          title: 'Meta-analysis of treatment',
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
          doi: '10.1234/test2',
        },
      ];

      const score = await scorer.calculateConfidence('Treatment is effective', citations, {
        sampleSize: 1000,
      });

      expect(score.overall).toBeGreaterThan(0.7);
      expect(score.citationStrength).toBeGreaterThan(0.7);
      expect(score.medicalAgreement).toBeGreaterThan(0.8);
    });

    it('should detect contradictions in mixed evidence', async () => {
      const citations: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'meta-analysis',
          title: 'High quality study',
          year: 2023,
          citationCount: 500,
          evidenceLevel: 'A',
        },
        {
          id: 'cit2',
          type: 'expert-opinion',
          title: 'Opinion piece',
          year: 2010,
          citationCount: 10,
          evidenceLevel: 'D',
        },
      ];

      const score = await scorer.calculateConfidence('Some claim', citations);

      expect(score.contradictions).toContain('Mixed evidence quality detected (Level A and D present)');
    });

    it('should detect temporal contradictions', async () => {
      const citations: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'peer-reviewed',
          title: 'Old study',
          year: 2005,
          citationCount: 100,
          evidenceLevel: 'B',
        },
        {
          id: 'cit2',
          type: 'peer-reviewed',
          title: 'Recent study',
          year: 2023,
          citationCount: 50,
          evidenceLevel: 'A',
        },
      ];

      const score = await scorer.calculateConfidence('Claim', citations);

      expect(score.contradictions).toContain('Temporal evidence gap detected (10+ year span)');
    });
  });

  describe('isConfident', () => {
    it('should return true for high confidence scores', () => {
      const score: ConfidenceScore = {
        overall: 0.85,
        statistical: 0.8,
        citationStrength: 0.9,
        medicalAgreement: 0.85,
        expertConsensus: 0.8,
        contradictions: [],
        metadata: {
          sourceCount: 5,
          peerReviewedSources: 4,
          expertOpinions: 1,
          conflictingEvidence: 0,
          recencyScore: 0.9,
        },
      };

      expect(scorer.isConfident(score)).toBe(true);
    });

    it('should return false for low confidence scores', () => {
      const score: ConfidenceScore = {
        overall: 0.5,
        statistical: 0.4,
        citationStrength: 0.5,
        medicalAgreement: 0.6,
        expertConsensus: 0.4,
        contradictions: ['Some contradiction'],
        metadata: {
          sourceCount: 2,
          peerReviewedSources: 1,
          expertOpinions: 1,
          conflictingEvidence: 1,
          recencyScore: 0.5,
        },
      };

      expect(scorer.isConfident(score)).toBe(false);
    });
  });

  describe('getConfidenceLevel', () => {
    it('should return correct confidence levels', () => {
      expect(scorer.getConfidenceLevel(0.9)).toBe('high');
      expect(scorer.getConfidenceLevel(0.7)).toBe('medium');
      expect(scorer.getConfidenceLevel(0.5)).toBe('low');
      expect(scorer.getConfidenceLevel(0.3)).toBe('very-low');
    });
  });

  describe('statistical confidence calculation', () => {
    it('should weight recent studies higher', async () => {
      const recentCitations: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'peer-reviewed',
          title: 'Recent study',
          year: 2023,
          citationCount: 100,
          evidenceLevel: 'A',
        },
      ];

      const oldCitations: MedicalCitation[] = [
        {
          id: 'cit2',
          type: 'peer-reviewed',
          title: 'Old study',
          year: 2000,
          citationCount: 100,
          evidenceLevel: 'A',
        },
      ];

      const recentScore = await scorer.calculateConfidence('Claim', recentCitations);
      const oldScore = await scorer.calculateConfidence('Claim', oldCitations);

      expect(recentScore.statistical).toBeGreaterThan(oldScore.statistical);
    });

    it('should consider sample size', async () => {
      const citations: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'clinical-trial',
          title: 'Large trial',
          year: 2023,
          citationCount: 100,
          evidenceLevel: 'A',
        },
      ];

      const largeScore = await scorer.calculateConfidence('Claim', citations, {
        sampleSize: 10000,
      });

      const smallScore = await scorer.calculateConfidence('Claim', citations, {
        sampleSize: 50,
      });

      expect(largeScore.statistical).toBeGreaterThan(smallScore.statistical);
    });
  });

  describe('citation strength calculation', () => {
    it('should weight meta-analyses highest', async () => {
      const metaAnalysis: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'meta-analysis',
          title: 'Meta-analysis',
          year: 2023,
          citationCount: 100,
          impactFactor: 10,
          evidenceLevel: 'A',
        },
      ];

      const opinion: MedicalCitation[] = [
        {
          id: 'cit2',
          type: 'expert-opinion',
          title: 'Opinion',
          year: 2023,
          citationCount: 100,
          impactFactor: 10,
          evidenceLevel: 'A',
        },
      ];

      const metaScore = await scorer.calculateConfidence('Claim', metaAnalysis);
      const opinionScore = await scorer.calculateConfidence('Claim', opinion);

      expect(metaScore.citationStrength).toBeGreaterThan(opinionScore.citationStrength);
    });

    it('should consider impact factor', async () => {
      const highImpact: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'peer-reviewed',
          title: 'High impact study',
          year: 2023,
          citationCount: 100,
          impactFactor: 20,
          evidenceLevel: 'A',
        },
      ];

      const lowImpact: MedicalCitation[] = [
        {
          id: 'cit2',
          type: 'peer-reviewed',
          title: 'Low impact study',
          year: 2023,
          citationCount: 100,
          impactFactor: 2,
          evidenceLevel: 'A',
        },
      ];

      const highScore = await scorer.calculateConfidence('Claim', highImpact);
      const lowScore = await scorer.calculateConfidence('Claim', lowImpact);

      expect(highScore.citationStrength).toBeGreaterThan(lowScore.citationStrength);
    });
  });
});
