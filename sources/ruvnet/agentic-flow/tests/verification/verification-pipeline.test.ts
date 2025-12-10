/**
 * Tests for Verification Pipeline
 */

import {
  VerificationPipeline,
  VerificationInput,
  HallucinationDetection,
} from '../../src/verification/pipeline/verification-pipeline';
import { MedicalCitation } from '../../src/verification/core/confidence-scorer';

describe('VerificationPipeline', () => {
  let pipeline: VerificationPipeline;

  beforeEach(() => {
    pipeline = new VerificationPipeline();
  });

  describe('preOutputVerification', () => {
    it('should warn when no citations provided', async () => {
      const input: VerificationInput = {
        claim: 'This treatment always works',
        context: {},
      };

      const result = await pipeline.preOutputVerification(input);

      expect(result.warnings).toContain('No citations provided - confidence will be limited');
      expect(result.requiresReview).toBe(true);
    });

    it('should detect overconfident language', async () => {
      const input: VerificationInput = {
        claim: 'This treatment always works and never fails. It is absolutely guaranteed to cure the condition.',
        citations: [],
      };

      const result = await pipeline.preOutputVerification(input);

      const overconfidentHallucination = result.hallucinations.find(
        h => h.type === 'factual' && h.description.includes('Overconfident')
      );

      expect(overconfidentHallucination).toBeDefined();
    });

    it('should detect unsupported quantitative claims', async () => {
      const input: VerificationInput = {
        claim: 'This treatment is 85% more effective and improves outcomes by 3 times',
        citations: [],
      };

      const result = await pipeline.preOutputVerification(input);

      const quantitativeHallucination = result.hallucinations.find(
        h => h.type === 'quantitative'
      );

      expect(quantitativeHallucination).toBeDefined();
      expect(quantitativeHallucination?.severity).toBe('high');
    });

    it('should detect unsupported medical claims', async () => {
      const input: VerificationInput = {
        claim: 'This treatment cures cancer and completely eliminates all symptoms permanently',
        citations: [],
      };

      const result = await pipeline.preOutputVerification(input);

      const medicalHallucination = result.hallucinations.find(
        h => h.type === 'factual' && h.severity === 'critical'
      );

      expect(medicalHallucination).toBeDefined();
    });

    it('should pass verification with good citations', async () => {
      const citations: MedicalCitation[] = [
        {
          id: 'cit1',
          type: 'meta-analysis',
          title: 'Meta-analysis',
          year: 2023,
          citationCount: 500,
          impactFactor: 15,
          evidenceLevel: 'A',
        },
        {
          id: 'cit2',
          type: 'clinical-trial',
          title: 'RCT',
          year: 2022,
          citationCount: 300,
          evidenceLevel: 'A',
        },
      ];

      const input: VerificationInput = {
        claim: 'Studies suggest this treatment may be effective for certain conditions',
        citations,
        context: { sampleSize: 1000 },
      };

      const result = await pipeline.preOutputVerification(input);

      expect(result.verified).toBe(true);
      expect(result.hallucinations.filter(h => h.severity === 'critical')).toHaveLength(0);
    });
  });

  describe('detectHallucinations', () => {
    it('should detect invalid percentages', async () => {
      const text = 'The treatment has a 150% success rate';

      const hallucinations = await pipeline.detectHallucinations(text);

      const invalidPercentage = hallucinations.find(
        h => h.type === 'quantitative' && h.severity === 'critical'
      );

      expect(invalidPercentage).toBeDefined();
      expect(invalidPercentage?.description).toContain('Invalid percentage');
    });

    it('should detect suspiciously precise numbers', async () => {
      const text = 'The treatment improved outcomes by 23.4567%';

      const hallucinations = await pipeline.detectHallucinations(text);

      const preciseNumber = hallucinations.find(
        h => h.type === 'quantitative' && h.description.includes('precise')
      );

      expect(preciseNumber).toBeDefined();
    });

    it('should detect future dates', async () => {
      const text = 'A 2026 study showed that this treatment is effective';

      const hallucinations = await pipeline.detectHallucinations(text);

      const futureDate = hallucinations.find(
        h => h.type === 'temporal' && h.severity === 'critical'
      );

      expect(futureDate).toBeDefined();
    });

    it('should detect logical contradictions', async () => {
      const text = 'The treatment is effective. The treatment is not effective.';

      const hallucinations = await pipeline.detectHallucinations(text);

      const contradiction = hallucinations.find(h => h.type === 'logical');

      expect(contradiction).toBeDefined();
    });
  });

  describe('postOutputValidation', () => {
    it('should detect output fidelity issues', async () => {
      const input: VerificationInput = {
        claim: 'Short claim',
        citations: [
          {
            id: 'cit1',
            type: 'peer-reviewed',
            title: 'Study',
            year: 2023,
            citationCount: 100,
            evidenceLevel: 'A',
          },
        ],
      };

      const output = 'Very long output that goes into extensive detail about many things that were not in the original claim, adding lots of new information, statistics, and facts that were not present before, including specific numbers like 85% effectiveness and 200% improvement.';

      const result = await pipeline.postOutputValidation(output, input);

      expect(result.warnings.length).toBeGreaterThan(0);
      expect(result.requiresReview).toBe(true);
    });
  });

  describe('provider review integration', () => {
    it('should store and retrieve provider reviews', async () => {
      const claimId = 'claim123';
      const review = {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        feedback: 'Looks good',
        timestamp: Date.now(),
      };

      await pipeline.addProviderReview(claimId, review);
      const reviews = pipeline.getProviderReviews(claimId);

      expect(reviews).toHaveLength(1);
      expect(reviews[0]).toEqual(review);
    });

    it('should handle multiple reviews for same claim', async () => {
      const claimId = 'claim123';

      await pipeline.addProviderReview(claimId, {
        reviewerId: 'provider1',
        approved: true,
        corrections: [],
        feedback: 'Good',
        timestamp: Date.now(),
      });

      await pipeline.addProviderReview(claimId, {
        reviewerId: 'provider2',
        approved: false,
        corrections: [{ type: 'factual', original: 'X', corrected: 'Y', importance: 'high' }],
        feedback: 'Needs correction',
        timestamp: Date.now(),
      });

      const reviews = pipeline.getProviderReviews(claimId);

      expect(reviews).toHaveLength(2);
    });
  });

  describe('suggestion generation', () => {
    it('should generate helpful suggestions', async () => {
      const input: VerificationInput = {
        claim: 'Weak claim with no support',
        citations: [
          {
            id: 'cit1',
            type: 'expert-opinion',
            title: 'Opinion',
            year: 2010,
            citationCount: 5,
            evidenceLevel: 'D',
          },
        ],
      };

      const result = await pipeline.preOutputVerification(input);

      expect(result.suggestions.length).toBeGreaterThan(0);
      expect(result.suggestions.some(s => s.includes('high-quality'))).toBe(true);
    });
  });
});
