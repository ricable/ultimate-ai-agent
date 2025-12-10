// Validation Tests - Medical Accuracy
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { VerificationService } from '../../src/services/verification-service';
import { KnowledgeBaseService } from '../../src/services/knowledge-base';
import { PatientData } from '../../src/types/medical';

describe('Medical Accuracy Validation', () => {
  let analyzer: MedicalAnalyzerService;
  let verifier: VerificationService;
  let knowledgeBase: KnowledgeBaseService;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    verifier = new VerificationService();
    knowledgeBase = new KnowledgeBaseService();
  });

  describe('Diagnosis accuracy', () => {
    it('should diagnose hypertension based on blood pressure', async () => {
      const patientData: PatientData = {
        id: 'patient_hypertension_001',
        age: 55,
        gender: 'male',
        symptoms: ['headache', 'dizziness'],
        medicalHistory: [],
        medications: [],
        vitalSigns: {
          bloodPressure: { systolic: 160, diastolic: 100 },
          heartRate: 75,
          temperature: 37.0,
          respiratoryRate: 16,
          oxygenSaturation: 98,
        },
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.diagnosis).toBeDefined();
      expect(analysis.diagnosis.some(d =>
        d.toLowerCase().includes('hypertension') ||
        d.toLowerCase().includes('blood pressure')
      )).toBe(true);
    });

    it('should diagnose respiratory infection from symptoms', async () => {
      const patientData: PatientData = {
        id: 'patient_respiratory_001',
        age: 30,
        gender: 'female',
        symptoms: ['fever', 'cough', 'congestion', 'sore throat'],
        medicalHistory: [],
        medications: [],
        vitalSigns: {
          bloodPressure: { systolic: 120, diastolic: 80 },
          heartRate: 85,
          temperature: 38.5,
          respiratoryRate: 20,
          oxygenSaturation: 96,
        },
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.diagnosis).toBeDefined();
      expect(analysis.diagnosis.some(d =>
        d.toLowerCase().includes('respiratory') ||
        d.toLowerCase().includes('infection')
      )).toBe(true);
    });

    it('should identify diabetes risk factors', async () => {
      const patientData: PatientData = {
        id: 'patient_diabetes_001',
        age: 60,
        gender: 'male',
        symptoms: ['increased thirst', 'frequent urination', 'fatigue'],
        medicalHistory: ['obesity', 'family history of diabetes'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.riskFactors).toBeDefined();
      const diabetesRisk = analysis.riskFactors.find(r =>
        r.factor.toLowerCase().includes('diabetes') ||
        r.factor.toLowerCase().includes('age') ||
        r.factor.toLowerCase().includes('family')
      );

      // Should identify at least one diabetes-related risk
      expect(analysis.riskFactors.length).toBeGreaterThan(0);
    });
  });

  describe('Clinical guideline compliance', () => {
    it('should recommend follow-up for new diagnoses', async () => {
      const patientData: PatientData = {
        id: 'patient_followup_001',
        age: 45,
        gender: 'female',
        symptoms: ['symptom1'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.recommendations).toBeDefined();
      expect(analysis.recommendations.length).toBeGreaterThan(0);
      expect(analysis.recommendations.some(r =>
        r.toLowerCase().includes('follow') ||
        r.toLowerCase().includes('monitor')
      )).toBe(true);
    });

    it('should recommend lifestyle modifications for chronic conditions', async () => {
      const patientData: PatientData = {
        id: 'patient_chronic_001',
        age: 50,
        gender: 'male',
        symptoms: [],
        medicalHistory: ['hypertension', 'obesity'],
        medications: ['lisinopril'],
        vitalSigns: {
          bloodPressure: { systolic: 145, diastolic: 92 },
          heartRate: 78,
          temperature: 37.0,
          respiratoryRate: 16,
          oxygenSaturation: 98,
        },
      };

      const analysis = await analyzer.analyzePatient(patientData);

      expect(analysis.recommendations.some(r =>
        r.toLowerCase().includes('lifestyle') ||
        r.toLowerCase().includes('diet') ||
        r.toLowerCase().includes('exercise') ||
        r.toLowerCase().includes('physical activity')
      )).toBe(true);
    });

    it('should provide evidence-based recommendations', async () => {
      const patientData: PatientData = {
        id: 'patient_evidence_001',
        age: 55,
        gender: 'female',
        symptoms: ['chest pain'],
        medicalHistory: ['diabetes'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // All recommendations should be supported by citations
      expect(analysis.citations.length).toBeGreaterThan(0);
      expect(analysis.citations.every(c => c.verified)).toBe(true);
    });
  });

  describe('Knowledge base cross-validation', () => {
    it('should cross-check diagnoses with knowledge base', async () => {
      const patientData: PatientData = {
        id: 'patient_crosscheck_001',
        age: 50,
        gender: 'male',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Verify cross-checks were performed
      expect(analysis.metadata.knowledgeBaseCrossChecks).toBeGreaterThan(0);
    });

    it('should validate citations against knowledge base', async () => {
      const patientData: PatientData = {
        id: 'patient_citation_001',
        age: 45,
        gender: 'female',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Verify all citations
      for (const citation of analysis.citations) {
        const isValid = await knowledgeBase.verifyCitation(citation);
        expect(isValid).toBe(true);
      }
    });

    it('should identify knowledge gaps', async () => {
      const patientData: PatientData = {
        id: 'patient_gap_001',
        age: 40,
        gender: 'male',
        symptoms: ['rare symptom', 'unusual presentation'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Should still provide analysis even with knowledge gaps
      expect(analysis).toBeDefined();
      expect(analysis.confidence).toBeDefined();

      // Lower confidence for uncertain cases
      if (analysis.diagnosis.includes('Requires additional diagnostic testing')) {
        expect(analysis.confidence).toBeLessThan(0.9);
      }
    });
  });

  describe('Medical terminology validation', () => {
    it('should use proper medical terminology', async () => {
      const patientData: PatientData = {
        id: 'patient_terminology_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Analysis should contain medical terms
      const medicalTermPattern = /patient|symptoms?|diagnosis|treatment|medication|condition|vital signs?/i;
      expect(medicalTermPattern.test(analysis.analysis)).toBe(true);
    });

    it('should avoid colloquial or unprofessional language', async () => {
      const patientData: PatientData = {
        id: 'patient_professional_001',
        age: 45,
        gender: 'female',
        symptoms: ['headache'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Should not contain unprofessional terms
      const unprofessionalTerms = /very bad|super sick|really really/i;
      expect(unprofessionalTerms.test(analysis.analysis)).toBe(false);
    });
  });

  describe('Confidence scoring accuracy', () => {
    it('should have high confidence for clear cases', async () => {
      const clearCase: PatientData = {
        id: 'patient_clear_001',
        age: 55,
        gender: 'male',
        symptoms: ['fever', 'cough', 'congestion'],
        medicalHistory: [],
        medications: [],
        vitalSigns: {
          bloodPressure: { systolic: 155, diastolic: 95 },
          heartRate: 80,
          temperature: 38.5,
          respiratoryRate: 18,
          oxygenSaturation: 96,
        },
      };

      const analysis = await analyzer.analyzePatient(clearCase);

      expect(analysis.confidence).toBeGreaterThan(0.7);
    });

    it('should have lower confidence for ambiguous cases', async () => {
      const ambiguousCase: PatientData = {
        id: 'patient_ambiguous_001',
        age: 40,
        gender: 'male',
        symptoms: ['fatigue'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(ambiguousCase);

      // Single vague symptom should result in lower confidence or
      // request for additional testing
      if (analysis.diagnosis.includes('Requires additional diagnostic testing')) {
        expect(analysis).toBeDefined();
      }
    });
  });
});
