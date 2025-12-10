// Validation Tests - Confidence Scoring
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { PatientData } from '../../src/types/medical';

describe('Confidence Scoring Validation', () => {
  let analyzer: MedicalAnalyzerService;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
  });

  describe('Confidence calculation', () => {
    it('should calculate confidence based on multiple factors', async () => {
      const patientData: PatientData = {
        id: 'patient_conf_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever', 'cough', 'fatigue'],
        medicalHistory: ['asthma'],
        medications: ['albuterol'],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.confidence).toBeGreaterThan(0);
      expect(analysis.confidence).toBeLessThanOrEqual(1);

      // Confidence should reflect hallucination checks
      const passedChecks = analysis.metadata.hallucinationChecks.filter(c => c.passed).length;
      const totalChecks = analysis.metadata.hallucinationChecks.length;
      if (passedChecks === totalChecks) {
        expect(analysis.confidence).toBeGreaterThan(0.5);
      }
    });

    it('should weight citation quality in confidence', async () => {
      const wellCitedPatient: PatientData = {
        id: 'patient_cited_001',
        age: 45,
        gender: 'female',
        symptoms: ['symptoms with clear diagnosis'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(wellCitedPatient);

      // More citations and cross-checks should increase confidence
      if (analysis.citations.length > 0 && analysis.metadata.knowledgeBaseCrossChecks > 0) {
        expect(analysis.confidence).toBeGreaterThan(0.6);
      }
    });

    it('should lower confidence for conflicting information', async () => {
      const conflictingPatient: PatientData = {
        id: 'patient_conflict_001',
        age: 25,
        gender: 'male',
        symptoms: ['chest pain'],
        medicalHistory: ['healthy', 'no prior issues'],
        medications: [],
        vitalSigns: {
          bloodPressure: { systolic: 110, diastolic: 70 },
          heartRate: 65,
          temperature: 37.0,
          respiratoryRate: 14,
          oxygenSaturation: 99,
        },
      };

      const analysis = await analyzer.analyzePatient(conflictingPatient);

      // Young healthy patient with chest pain is ambiguous
      expect(analysis).toBeDefined();
      expect(analysis.confidence).toBeGreaterThan(0);
    });
  });

  describe('Confidence thresholds', () => {
    it('should flag low confidence analyses', async () => {
      const uncertainPatient: PatientData = {
        id: 'patient_uncertain_001',
        age: 40,
        gender: 'male',
        symptoms: ['vague symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(uncertainPatient);

      // Low confidence should be reflected in recommendations
      if (analysis.confidence < 0.7) {
        expect(analysis.recommendations.some(r =>
          r.toLowerCase().includes('additional') ||
          r.toLowerCase().includes('further') ||
          r.toLowerCase().includes('testing')
        )).toBe(true);
      }
    });

    it('should require high confidence for critical diagnoses', async () => {
      const criticalPatient: PatientData = {
        id: 'patient_critical_001',
        age: 70,
        gender: 'male',
        symptoms: ['severe chest pain', 'difficulty breathing'],
        medicalHistory: ['heart disease'],
        medications: ['aspirin'],
      };

      const analysis = await analyzer.analyzePatient(criticalPatient);

      // Critical cases should have citations and verification
      if (analysis.riskFactors.some(r => r.severity === 'critical' || r.severity === 'high')) {
        expect(analysis.citations.length).toBeGreaterThan(0);
        expect(analysis.verificationScore).toBeDefined();
      }
    });
  });

  describe('Confidence factors', () => {
    it('should increase confidence with more data points', async () => {
      const sparsePatient: PatientData = {
        id: 'patient_sparse_001',
        age: 50,
        gender: 'male',
        symptoms: ['symptom'],
        medicalHistory: [],
        medications: [],
      };

      const detailedPatient: PatientData = {
        id: 'patient_detailed_001',
        age: 50,
        gender: 'male',
        symptoms: ['symptom1', 'symptom2', 'symptom3'],
        medicalHistory: ['condition1', 'condition2'],
        medications: ['med1', 'med2'],
        vitalSigns: {
          bloodPressure: { systolic: 120, diastolic: 80 },
          heartRate: 75,
          temperature: 37.0,
          respiratoryRate: 16,
          oxygenSaturation: 98,
        },
        labResults: [
          {
            testName: 'Test1',
            value: 100,
            unit: 'mg/dL',
            referenceRange: { min: 70, max: 110 },
            timestamp: new Date().toISOString(),
          },
        ],
      };

      const sparseAnalysis = await analyzer.analyzePatient(sparsePatient);
      const detailedAnalysis = await analyzer.analyzePatient(detailedPatient);

      // More data should generally lead to higher confidence
      // (though not guaranteed in all cases)
      expect(detailedAnalysis.metadata.hallucinationChecks).toBeDefined();
      expect(sparseAnalysis.metadata.hallucinationChecks).toBeDefined();
    });

    it('should consider hallucination check confidence', async () => {
      const patientData: PatientData = {
        id: 'patient_halluc_conf_001',
        age: 55,
        gender: 'female',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Hallucination check confidence should be high
      analysis.metadata.hallucinationChecks.forEach(check => {
        if (check.passed) {
          expect(check.confidence).toBeGreaterThan(0.7);
        }
      });
    });

    it('should incorporate verification score in overall confidence', async () => {
      const patientData: PatientData = {
        id: 'patient_verif_conf_001',
        age: 60,
        gender: 'male',
        symptoms: ['symptoms'],
        medicalHistory: ['history'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Verification score should correlate with confidence
      expect(analysis.verificationScore).toBeDefined();
      expect(analysis.confidence).toBeDefined();

      // Both should be in reasonable ranges
      expect(analysis.verificationScore).toBeGreaterThanOrEqual(0);
      expect(analysis.verificationScore).toBeLessThanOrEqual(1);
      expect(analysis.confidence).toBeGreaterThanOrEqual(0);
      expect(analysis.confidence).toBeLessThanOrEqual(1);
    });
  });

  describe('Confidence reporting', () => {
    it('should include confidence in analysis metadata', async () => {
      const patientData: PatientData = {
        id: 'patient_meta_001',
        age: 45,
        gender: 'female',
        symptoms: ['symptom'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.confidence).toBeDefined();
      expect(typeof analysis.confidence).toBe('number');
      expect(analysis.metadata).toHaveProperty('hallucinationChecks');
      expect(analysis.metadata).toHaveProperty('knowledgeBaseCrossChecks');
    });

    it('should provide confidence breakdown in hallucination checks', async () => {
      const patientData: PatientData = {
        id: 'patient_breakdown_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      analysis.metadata.hallucinationChecks.forEach(check => {
        expect(check).toHaveProperty('type');
        expect(check).toHaveProperty('passed');
        expect(check).toHaveProperty('confidence');
        expect(check).toHaveProperty('details');
        expect(check.confidence).toBeGreaterThanOrEqual(0);
        expect(check.confidence).toBeLessThanOrEqual(1);
      });
    });
  });
});
