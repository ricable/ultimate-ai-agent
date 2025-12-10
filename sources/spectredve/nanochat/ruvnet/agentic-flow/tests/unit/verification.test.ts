// Unit Tests for Verification System
import { VerificationService } from '../../src/services/verification-service';
import { Citation, HallucinationCheck } from '../../src/types/medical';

describe('VerificationService', () => {
  let service: VerificationService;

  beforeEach(() => {
    service = new VerificationService();
  });

  describe('verifyAnalysis', () => {
    it('should pass verification for valid analysis', async () => {
      const validData = {
        analysis: 'Patient presents with symptoms consistent with upper respiratory infection',
        diagnosis: ['Upper Respiratory Infection'],
        citations: [
          {
            source: 'Medical Journal 1',
            reference: 'DOI: 10.1234/medj.1000',
            relevance: 0.9,
            verified: true,
          },
        ],
        recommendations: ['Follow up with primary care physician', 'Monitor symptoms'],
      };

      const result = await service.verifyAnalysis(validData);

      expect(result.passed).toBe(true);
      expect(result.score).toBeGreaterThanOrEqual(0.85);
      expect(result.checks.medicalAccuracy).toBe(true);
      expect(result.checks.citationValidity).toBe(true);
      expect(result.checks.hallucinationFree).toBe(true);
    });

    it('should fail verification for analysis without citations', async () => {
      const invalidData = {
        analysis: 'Patient has condition',
        diagnosis: ['Unknown condition'],
        citations: [],
        recommendations: [],
      };

      const result = await service.verifyAnalysis(invalidData);

      expect(result.passed).toBe(false);
      expect(result.checks.citationValidity).toBe(false);
    });

    it('should fail verification for suspicious content', async () => {
      const suspiciousData = {
        analysis: 'This guaranteed cure will be 100% effective',
        diagnosis: ['Miracle cure'],
        citations: [{
          source: 'Unknown',
          reference: 'None',
          relevance: 0.5,
          verified: false,
        }],
      };

      const result = await service.verifyAnalysis(suspiciousData);

      expect(result.passed).toBe(false);
      expect(result.checks.hallucinationFree).toBe(false);
    });

    it('should detect hallucination patterns', async () => {
      const hallucinationData = {
        analysis: 'Call 555-123-4567 for miracle treatment',
        diagnosis: ['Secret cure'],
        citations: [],
      };

      const result = await service.verifyAnalysis(hallucinationData);

      expect(result.passed).toBe(false);
      expect(result.checks.hallucinationFree).toBe(false);
      expect(result.issues.some(i => i.type === 'hallucination')).toBe(true);
    });
  });

  describe('calculateVerificationScore', () => {
    it('should calculate high score for valid analysis', async () => {
      const validData = {
        analysis: 'Patient diagnosed with condition based on symptoms',
        diagnosis: ['Valid Diagnosis'],
        citations: [
          { source: 'Journal', reference: 'DOI: 10.1234/test', relevance: 0.9, verified: true },
        ],
        hallucinationChecks: [
          { type: 'factual' as const, passed: true, confidence: 0.95, details: 'All facts verified' },
          { type: 'logical' as const, passed: true, confidence: 0.92, details: 'Logically consistent' },
        ],
      };

      const score = await service.calculateVerificationScore(validData);

      expect(score).toBeGreaterThanOrEqual(0.85);
      expect(score).toBeLessThanOrEqual(1.0);
    });

    it('should calculate low score for invalid analysis', async () => {
      const invalidData = {
        analysis: 'Invalid analysis',
        diagnosis: [],
        citations: [],
        hallucinationChecks: [
          { type: 'factual' as const, passed: false, confidence: 0.5, details: 'Verification failed' },
        ],
      };

      const score = await service.calculateVerificationScore(invalidData);

      expect(score).toBeLessThan(0.85);
    });
  });

  describe('verification checks', () => {
    it('should verify medical accuracy', async () => {
      const medicalData = {
        analysis: 'Patient presents with elevated blood pressure and symptoms of hypertension',
        diagnosis: ['Hypertension', 'Essential Hypertension'],
        citations: [
          { source: 'Medical Journal', reference: 'DOI: 10.1234/test', relevance: 0.9, verified: true },
        ],
      };

      const result = await service.verifyAnalysis(medicalData);

      expect(result.checks.medicalAccuracy).toBe(true);
    });

    it('should verify citation validity', async () => {
      const citedData = {
        analysis: 'Analysis with proper citations',
        diagnosis: ['Valid Diagnosis'],
        citations: [
          { source: 'Medical Journal 1', reference: 'DOI: 10.1234/1', relevance: 0.85, verified: true },
          { source: 'Medical Journal 2', reference: 'DOI: 10.1234/2', relevance: 0.80, verified: true },
        ],
      };

      const result = await service.verifyAnalysis(citedData);

      expect(result.checks.citationValidity).toBe(true);
    });

    it('should verify logical consistency', async () => {
      const consistentData = {
        analysis: 'Patient shows consistent symptoms throughout examination',
        diagnosis: ['Consistent Diagnosis'],
        citations: [
          { source: 'Journal', reference: 'DOI: 10.1234/test', relevance: 0.9, verified: true },
        ],
      };

      const result = await service.verifyAnalysis(consistentData);

      expect(result.checks.logicalConsistency).toBe(true);
    });

    it('should verify guideline compliance', async () => {
      const compliantData = {
        analysis: 'Analysis follows medical guidelines',
        diagnosis: ['Guideline-compliant diagnosis'],
        citations: [
          { source: 'Journal', reference: 'DOI: 10.1234/test', relevance: 0.9, verified: true },
        ],
        recommendations: ['Follow up with physician', 'Monitor symptoms closely'],
      };

      const result = await service.verifyAnalysis(compliantData);

      expect(result.checks.guidelineCompliance).toBe(true);
    });
  });

  describe('issue identification', () => {
    it('should identify medical accuracy issues', async () => {
      const inaccurateData = {
        analysis: 'Random text without medical terms',
        diagnosis: ['X'],
        citations: [],
      };

      const result = await service.verifyAnalysis(inaccurateData);

      const medicalIssues = result.issues.filter(i => i.type === 'medical-accuracy');
      expect(medicalIssues.length).toBeGreaterThan(0);
      expect(medicalIssues[0].severity).toBe('error');
    });

    it('should identify citation issues', async () => {
      const uncitedData = {
        analysis: 'Patient has medical condition',
        diagnosis: ['Some condition'],
        citations: [],
      };

      const result = await service.verifyAnalysis(uncitedData);

      const citationIssues = result.issues.filter(i => i.type === 'citation-validity');
      expect(citationIssues.length).toBeGreaterThan(0);
    });

    it('should provide suggested fixes', async () => {
      const problematicData = {
        analysis: 'Problematic analysis',
        diagnosis: ['Diagnosis'],
        citations: [],
      };

      const result = await service.verifyAnalysis(problematicData);

      result.issues.forEach(issue => {
        if (issue.suggestedFix) {
          expect(typeof issue.suggestedFix).toBe('string');
          expect(issue.suggestedFix.length).toBeGreaterThan(0);
        }
      });
    });
  });
});
