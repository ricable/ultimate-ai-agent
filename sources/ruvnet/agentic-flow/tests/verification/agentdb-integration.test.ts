/**
 * Tests for AgentDB Integration
 */

import {
  AgentDBIntegration,
  ProviderFeedback,
  FeatureVector,
  Correction,
} from '../../src/verification/learning/agentdb-integration';

describe('AgentDBIntegration', () => {
  let agentDB: AgentDBIntegration;

  beforeEach(() => {
    agentDB = new AgentDBIntegration();
  });

  describe('learnFromCorrection', () => {
    it('should learn from provider feedback', async () => {
      const claim = 'Treatment is effective';
      const originalConfidence = 0.6;
      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.8,
        reasoning: 'Good evidence provided',
        categories: ['cardiology'],
      };
      const features: FeatureVector = {
        citationCount: 5,
        peerReviewedRatio: 0.8,
        recencyScore: 0.9,
        evidenceLevelScore: 0.85,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 50,
        quantitativeClaims: 1,
      };

      await agentDB.learnFromCorrection(claim, originalConfidence, feedback, features);

      const stats = agentDB.getModelStatistics();
      expect(stats.trainingExamples).toBe(1);
    });

    it('should update model weights', async () => {
      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.9,
        reasoning: 'Excellent',
        categories: [],
      };

      const features: FeatureVector = {
        citationCount: 10,
        peerReviewedRatio: 1.0,
        recencyScore: 1.0,
        evidenceLevelScore: 1.0,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      const initialStats = agentDB.getModelStatistics();

      await agentDB.learnFromCorrection('Great claim', 0.7, feedback, features);

      const updatedStats = agentDB.getModelStatistics();

      expect(updatedStats.trainingExamples).toBeGreaterThan(initialStats.trainingExamples);
    });

    it('should track correction patterns', async () => {
      const corrections: Correction[] = [
        {
          type: 'factual',
          original: 'Treatment cures disease',
          corrected: 'Treatment may help manage disease',
          importance: 'high',
        },
      ];

      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: false,
        corrections,
        confidenceAssessment: 0.3,
        reasoning: 'Overstated claim',
        categories: [],
      };

      const features: FeatureVector = {
        citationCount: 1,
        peerReviewedRatio: 0.5,
        recencyScore: 0.5,
        evidenceLevelScore: 0.5,
        contradictionCount: 0,
        hallucinationFlags: 2,
        textLength: 50,
        quantitativeClaims: 0,
      };

      await agentDB.learnFromCorrection('Bad claim', 0.8, feedback, features);

      const patternStats = agentDB.getPatternStatistics();
      expect(patternStats.totalPatterns).toBeGreaterThan(0);
    });
  });

  describe('predictConfidence', () => {
    it('should predict confidence from features', () => {
      const features: FeatureVector = {
        citationCount: 5,
        peerReviewedRatio: 0.8,
        recencyScore: 0.9,
        evidenceLevelScore: 0.85,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 1,
      };

      const predicted = agentDB.predictConfidence(features);

      expect(predicted).toBeGreaterThan(0);
      expect(predicted).toBeLessThan(1);
    });

    it('should predict lower confidence for poor features', () => {
      const goodFeatures: FeatureVector = {
        citationCount: 10,
        peerReviewedRatio: 1.0,
        recencyScore: 1.0,
        evidenceLevelScore: 1.0,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      const poorFeatures: FeatureVector = {
        citationCount: 1,
        peerReviewedRatio: 0.2,
        recencyScore: 0.3,
        evidenceLevelScore: 0.3,
        contradictionCount: 3,
        hallucinationFlags: 5,
        textLength: 50,
        quantitativeClaims: 10,
      };

      const goodPrediction = agentDB.predictConfidence(goodFeatures);
      const poorPrediction = agentDB.predictConfidence(poorFeatures);

      expect(goodPrediction).toBeGreaterThan(poorPrediction);
    });
  });

  describe('getConfidenceAdjustment', () => {
    it('should return adjustment based on learned patterns', async () => {
      // Train with some data
      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.9,
        reasoning: 'High quality',
        categories: ['cardiology'],
      };

      const features: FeatureVector = {
        citationCount: 10,
        peerReviewedRatio: 1.0,
        recencyScore: 1.0,
        evidenceLevelScore: 1.0,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      // Train multiple times to establish pattern
      for (let i = 0; i < 5; i++) {
        await agentDB.learnFromCorrection('Good claim', 0.7, feedback, features);
      }

      const adjustment = await agentDB.getConfidenceAdjustment(features, ['cardiology']);

      expect(adjustment.adjustment).toBeGreaterThanOrEqual(-0.3);
      expect(adjustment.adjustment).toBeLessThanOrEqual(0.3);
      expect(adjustment.appliedPatterns).toBeDefined();
    });

    it('should provide reasoning for adjustments', async () => {
      const features: FeatureVector = {
        citationCount: 10,
        peerReviewedRatio: 1.0,
        recencyScore: 1.0,
        evidenceLevelScore: 1.0,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      const adjustment = await agentDB.getConfidenceAdjustment(features, []);

      expect(adjustment.reason).toBeDefined();
      expect(typeof adjustment.reason).toBe('string');
    });
  });

  describe('pattern recognition', () => {
    it('should identify reliable patterns', async () => {
      // Create multiple successful examples with same pattern
      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.9,
        reasoning: 'Good',
        categories: [],
      };

      const features: FeatureVector = {
        citationCount: 10,
        peerReviewedRatio: 1.0,
        recencyScore: 0.9,
        evidenceLevelScore: 0.9,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      // Train 10 times with similar pattern
      for (let i = 0; i < 10; i++) {
        await agentDB.learnFromCorrection(`Claim ${i}`, 0.7, feedback, features);
      }

      const stats = agentDB.getPatternStatistics();

      expect(stats.totalPatterns).toBeGreaterThan(0);
      expect(stats.reliablePatterns).toBeGreaterThan(0);
    });

    it('should track pattern frequency', async () => {
      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.9,
        reasoning: 'Good',
        categories: [],
      };

      const features: FeatureVector = {
        citationCount: 10,
        peerReviewedRatio: 1.0,
        recencyScore: 0.9,
        evidenceLevelScore: 0.9,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      await agentDB.learnFromCorrection('Claim 1', 0.7, feedback, features);
      await agentDB.learnFromCorrection('Claim 2', 0.7, feedback, features);

      const stats = agentDB.getPatternStatistics();
      expect(stats.topPatterns.length).toBeGreaterThan(0);
      expect(stats.topPatterns[0].frequency).toBeGreaterThan(0);
    });
  });

  describe('source reliability tracking', () => {
    it('should track source reliability', async () => {
      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.9,
        reasoning: 'Good',
        categories: ['cardiology'],
      };

      const features: FeatureVector = {
        citationCount: 5,
        peerReviewedRatio: 0.8,
        recencyScore: 0.9,
        evidenceLevelScore: 0.85,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      // Train 15 times (above minSampleSize)
      for (let i = 0; i < 15; i++) {
        await agentDB.learnFromCorrection(`Claim ${i}`, 0.7, feedback, features);
      }

      const rankings = agentDB.getSourceRankings(10);

      expect(rankings.length).toBeGreaterThan(0);
      expect(rankings[0].sourceId).toBe('cardiology');
      expect(rankings[0].sampleSize).toBeGreaterThanOrEqual(10);
    });

    it('should update reliability based on outcomes', async () => {
      // Add successful claims
      for (let i = 0; i < 10; i++) {
        await agentDB.learnFromCorrection(
          `Good claim ${i}`,
          0.7,
          {
            reviewerId: 'provider1',
            approved: true,
            corrections: [],
            confidenceAssessment: 0.9,
            reasoning: 'Good',
            categories: ['source1'],
          },
          {
            citationCount: 5,
            peerReviewedRatio: 0.8,
            recencyScore: 0.9,
            evidenceLevelScore: 0.85,
            contradictionCount: 0,
            hallucinationFlags: 0,
            textLength: 100,
            quantitativeClaims: 0,
          }
        );
      }

      const rankings = agentDB.getSourceRankings(5);
      const source1 = rankings.find(s => s.sourceId === 'source1');

      expect(source1).toBeDefined();
      expect(source1!.reliability).toBeGreaterThan(0.7);
    });
  });

  describe('exportLearningData', () => {
    it('should export all learning data', async () => {
      const feedback: ProviderFeedback = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        confidenceAssessment: 0.9,
        reasoning: 'Good',
        categories: ['test'],
      };

      const features: FeatureVector = {
        citationCount: 5,
        peerReviewedRatio: 0.8,
        recencyScore: 0.9,
        evidenceLevelScore: 0.85,
        contradictionCount: 0,
        hallucinationFlags: 0,
        textLength: 100,
        quantitativeClaims: 0,
      };

      await agentDB.learnFromCorrection('Claim', 0.7, feedback, features);

      const exported = agentDB.exportLearningData();

      expect(exported.records.length).toBeGreaterThan(0);
      expect(exported.model).toBeDefined();
      expect(exported.patterns).toBeDefined();
      expect(exported.sources).toBeDefined();
    });
  });
});
