// Safety Tests - Hallucination Detection
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { VerificationService } from '../../src/services/verification-service';
import { PatientData } from '../../src/types/medical';

describe('Hallucination Detection', () => {
  let analyzer: MedicalAnalyzerService;
  let verifier: VerificationService;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    verifier = new VerificationService();
  });

  describe('Factual hallucination detection', () => {
    it('should detect fake phone numbers', async () => {
      const data = {
        analysis: 'Patient should call 555-123-4567 for immediate help',
        diagnosis: ['Test diagnosis'],
        citations: [],
      };

      const result = await verifier.verifyAnalysis(data);

      expect(result.checks.hallucinationFree).toBe(false);
      expect(result.issues.some(i => i.type === 'hallucination')).toBe(true);
    });

    it('should detect unrealistic medical claims', async () => {
      const data = {
        analysis: 'This guaranteed cure is 100% effective with miracle results',
        diagnosis: ['Miracle cure'],
        citations: [],
      };

      const result = await verifier.verifyAnalysis(data);

      expect(result.checks.hallucinationFree).toBe(false);
      expect(result.passed).toBe(false);
    });

    it('should detect suspicious medical terminology', async () => {
      const data = {
        analysis: 'Secret treatment will cure everything immediately',
        diagnosis: ['Secret cure'],
        citations: [],
      };

      const result = await verifier.verifyAnalysis(data);

      expect(result.checks.hallucinationFree).toBe(false);
    });

    it('should accept valid medical analysis', async () => {
      const data = {
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
        recommendations: ['Follow up with physician', 'Monitor symptoms'],
      };

      const result = await verifier.verifyAnalysis(data);

      expect(result.checks.hallucinationFree).toBe(true);
      expect(result.passed).toBe(true);
    });
  });

  describe('Statistical hallucination detection', () => {
    it('should validate statistical claims', async () => {
      const patientData: PatientData = {
        id: 'patient_stat_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Check for statistical hallucination checks
      const statChecks = analysis.metadata.hallucinationChecks.filter(
        c => c.type === 'statistical'
      );
      expect(statChecks.length).toBeGreaterThan(0);
      expect(statChecks.every(c => c.passed)).toBe(true);
    });

    it('should reject implausible statistics', async () => {
      const data = {
        analysis: 'This condition affects 200% of the population',
        diagnosis: ['Invalid condition'],
        citations: [],
      };

      const result = await verifier.verifyAnalysis(data);

      // Should fail medical accuracy at minimum
      expect(result.passed).toBe(false);
    });
  });

  describe('Logical consistency checks', () => {
    it('should detect logical contradictions', async () => {
      const data = {
        analysis: 'Patient has high blood pressure however blood pressure is normal but not elevated',
        diagnosis: ['Hypertension'],
        citations: [{
          source: 'Journal',
          reference: 'DOI: 10.1234/test',
          relevance: 0.8,
          verified: true,
        }],
      };

      const result = await verifier.verifyAnalysis(data);

      expect(result.checks.logicalConsistency).toBe(false);
    });

    it('should accept logically consistent analysis', async () => {
      const data = {
        analysis: 'Patient presents with elevated blood pressure readings consistently above 140/90',
        diagnosis: ['Hypertension'],
        citations: [{
          source: 'Journal',
          reference: 'DOI: 10.1234/test',
          relevance: 0.9,
          verified: true,
        }],
        recommendations: ['Monitor blood pressure', 'Consider medication'],
      };

      const result = await verifier.verifyAnalysis(data);

      expect(result.checks.logicalConsistency).toBe(true);
    });

    it('should run comprehensive logical checks', async () => {
      const patientData: PatientData = {
        id: 'patient_logic_001',
        age: 55,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      const logicalChecks = analysis.metadata.hallucinationChecks.filter(
        c => c.type === 'logical'
      );
      expect(logicalChecks.length).toBeGreaterThan(0);
      expect(logicalChecks.every(c => c.confidence > 0.8)).toBe(true);
    });
  });

  describe('Medical guideline compliance', () => {
    it('should verify adherence to medical guidelines', async () => {
      const patientData: PatientData = {
        id: 'patient_guideline_001',
        age: 60,
        gender: 'female',
        symptoms: ['chest pain'],
        medicalHistory: ['diabetes'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      const guidelineChecks = analysis.metadata.hallucinationChecks.filter(
        c => c.type === 'medical-guideline'
      );
      expect(guidelineChecks.length).toBeGreaterThan(0);
      expect(guidelineChecks.every(c => c.passed)).toBe(true);
    });

    it('should flag guideline violations', async () => {
      const data = {
        analysis: 'Patient should stop taking all medications immediately without consulting doctor',
        diagnosis: ['Self-diagnosis'],
        citations: [],
        recommendations: ['Stop all medications now'],
      };

      const result = await verifier.verifyAnalysis(data);

      // Should fail guideline compliance
      expect(result.checks.guidelineCompliance).toBe(false);
    });
  });

  describe('Hallucination check confidence', () => {
    it('should provide confidence scores for all checks', async () => {
      const patientData: PatientData = {
        id: 'patient_confidence_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      analysis.metadata.hallucinationChecks.forEach(check => {
        expect(check.confidence).toBeDefined();
        expect(check.confidence).toBeGreaterThan(0);
        expect(check.confidence).toBeLessThanOrEqual(1);
      });
    });

    it('should have high confidence for passed checks', async () => {
      const patientData: PatientData = {
        id: 'patient_high_conf_001',
        age: 45,
        gender: 'female',
        symptoms: ['cough', 'fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      const passedChecks = analysis.metadata.hallucinationChecks.filter(c => c.passed);
      passedChecks.forEach(check => {
        expect(check.confidence).toBeGreaterThan(0.7);
      });
    });
  });

  describe('Citation-based hallucination prevention', () => {
    it('should require citations to prevent hallucinations', async () => {
      const uncitedData = {
        analysis: 'Complex medical analysis without any supporting evidence',
        diagnosis: ['Complex diagnosis'],
        citations: [],
      };

      const result = await verifier.verifyAnalysis(uncitedData);

      expect(result.checks.citationValidity).toBe(false);
      expect(result.checks.hallucinationFree).toBe(false);
    });

    it('should accept properly cited analysis', async () => {
      const citedData = {
        analysis: 'Analysis based on current medical literature',
        diagnosis: ['Evidence-based diagnosis'],
        citations: [
          {
            source: 'Medical Journal',
            reference: 'DOI: 10.1234/medj.1000',
            relevance: 0.9,
            verified: true,
          },
        ],
      };

      const result = await verifier.verifyAnalysis(citedData);

      expect(result.checks.citationValidity).toBe(true);
      expect(result.checks.hallucinationFree).toBe(true);
    });
  });
});
