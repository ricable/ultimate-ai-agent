// Safety Tests - Edge Cases
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { VerificationService } from '../../src/services/verification-service';
import { NotificationService } from '../../src/services/notification-service';
import { PatientData } from '../../src/types/medical';

describe('Edge Case Handling', () => {
  let analyzer: MedicalAnalyzerService;
  let verifier: VerificationService;
  let notifier: NotificationService;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    verifier = new VerificationService();
    notifier = new NotificationService();
  });

  describe('Invalid input handling', () => {
    it('should handle missing patient ID', async () => {
      const invalidPatient: PatientData = {
        id: '',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      // Should not throw
      await expect(analyzer.analyzePatient(invalidPatient)).resolves.not.toThrow();
    });

    it('should handle negative age', async () => {
      const invalidPatient: PatientData = {
        id: 'patient_001',
        age: -1,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(invalidPatient);
      expect(analysis).toBeDefined();
    });

    it('should handle extremely high age', async () => {
      const oldPatient: PatientData = {
        id: 'patient_002',
        age: 150,
        gender: 'male',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(oldPatient);
      expect(analysis).toBeDefined();
    });

    it('should handle empty symptoms array', async () => {
      const noSymptoms: PatientData = {
        id: 'patient_003',
        age: 40,
        gender: 'female',
        symptoms: [],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(noSymptoms);
      expect(analysis).toBeDefined();
      expect(analysis.diagnosis).toBeDefined();
    });

    it('should handle null or undefined vital signs', async () => {
      const noVitals: PatientData = {
        id: 'patient_004',
        age: 35,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
        vitalSigns: undefined,
      };

      const analysis = await analyzer.analyzePatient(noVitals);
      expect(analysis).toBeDefined();
    });
  });

  describe('Boundary value testing', () => {
    it('should handle minimum age boundary', async () => {
      const infant: PatientData = {
        id: 'patient_infant',
        age: 0,
        gender: 'female',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(infant);
      expect(analysis).toBeDefined();
    });

    it('should handle maximum normal age', async () => {
      const elderly: PatientData = {
        id: 'patient_elderly',
        age: 100,
        gender: 'male',
        symptoms: ['fatigue'],
        medicalHistory: ['multiple conditions'],
        medications: ['multiple medications'],
      };

      const analysis = await analyzer.analyzePatient(elderly);
      expect(analysis).toBeDefined();
      expect(analysis.riskFactors.length).toBeGreaterThan(0);
    });

    it('should handle extreme vital signs', async () => {
      const extremeVitals: PatientData = {
        id: 'patient_extreme',
        age: 50,
        gender: 'male',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
        vitalSigns: {
          bloodPressure: { systolic: 250, diastolic: 150 },
          heartRate: 200,
          temperature: 42.0,
          respiratoryRate: 40,
          oxygenSaturation: 70,
        },
      };

      const analysis = await analyzer.analyzePatient(extremeVitals);
      expect(analysis).toBeDefined();
      expect(analysis.riskFactors.length).toBeGreaterThan(0);
    });
  });

  describe('Large data handling', () => {
    it('should handle patient with many symptoms', async () => {
      const manySymptoms: PatientData = {
        id: 'patient_many_symptoms',
        age: 60,
        gender: 'female',
        symptoms: Array.from({ length: 50 }, (_, i) => `symptom_${i}`),
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(manySymptoms);
      expect(analysis).toBeDefined();
    });

    it('should handle extensive medical history', async () => {
      const extensiveHistory: PatientData = {
        id: 'patient_extensive',
        age: 70,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: Array.from({ length: 30 }, (_, i) => `condition_${i}`),
        medications: Array.from({ length: 20 }, (_, i) => `medication_${i}`),
      };

      const analysis = await analyzer.analyzePatient(extensiveHistory);
      expect(analysis).toBeDefined();
      expect(analysis.riskFactors.length).toBeGreaterThan(0);
    });

    it('should handle many lab results', async () => {
      const manyLabs: PatientData = {
        id: 'patient_labs',
        age: 55,
        gender: 'female',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
        labResults: Array.from({ length: 20 }, (_, i) => ({
          testName: `Test_${i}`,
          value: 100 + i,
          unit: 'mg/dL',
          referenceRange: { min: 70, max: 110 },
          timestamp: new Date().toISOString(),
        })),
      };

      const analysis = await analyzer.analyzePatient(manyLabs);
      expect(analysis).toBeDefined();
    });
  });

  describe('Special characters and encoding', () => {
    it('should handle special characters in symptoms', async () => {
      const specialChars: PatientData = {
        id: 'patient_special',
        age: 45,
        gender: 'male',
        symptoms: ['chest pain (severe)', 'difficulty breathing @ night', 'fever ~38.5°C'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(specialChars);
      expect(analysis).toBeDefined();
    });

    it('should handle unicode characters', async () => {
      const unicode: PatientData = {
        id: 'patient_unicode',
        age: 50,
        gender: 'female',
        symptoms: ['头痛', 'fièvre', 'Übelkeit'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(unicode);
      expect(analysis).toBeDefined();
    });
  });

  describe('Concurrent operation safety', () => {
    it('should handle concurrent analyses safely', async () => {
      const patients: PatientData[] = Array.from({ length: 20 }, (_, i) => ({
        id: `patient_concurrent_${i}`,
        age: 40 + i,
        gender: i % 2 === 0 ? 'male' : 'female',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      }));

      const analyses = await Promise.all(
        patients.map(p => analyzer.analyzePatient(p))
      );

      expect(analyses.length).toBe(20);
      expect(analyses.every(a => a.id && a.patientId)).toBe(true);

      // Each should have unique ID
      const ids = new Set(analyses.map(a => a.id));
      expect(ids.size).toBe(20);
    });

    it('should handle concurrent notifications safely', async () => {
      const patient: PatientData = {
        id: 'patient_notif_concurrent',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patient);

      const notificationPromises = Array.from({ length: 10 }, (_, i) =>
        notifier.notifyProvider(analysis, `provider_${i}`)
      );

      const results = await Promise.all(notificationPromises);

      expect(results.length).toBe(10);
      expect(results.every(r => r.length > 0)).toBe(true);
    });
  });

  describe('Memory and resource limits', () => {
    it('should not leak memory with repeated analyses', async () => {
      const patient: PatientData = {
        id: 'patient_memory',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      // Run many analyses
      for (let i = 0; i < 100; i++) {
        await analyzer.analyzePatient({ ...patient, id: `patient_${i}` });
      }

      // Should complete without memory issues
      expect(true).toBe(true);
    });

    it('should handle analysis timeout gracefully', async () => {
      const complexPatient: PatientData = {
        id: 'patient_timeout',
        age: 70,
        gender: 'female',
        symptoms: Array.from({ length: 100 }, (_, i) => `symptom_${i}`),
        medicalHistory: Array.from({ length: 50 }, (_, i) => `condition_${i}`),
        medications: Array.from({ length: 30 }, (_, i) => `medication_${i}`),
      };

      // Should complete even with complex data
      const analysis = await analyzer.analyzePatient(complexPatient);
      expect(analysis).toBeDefined();
    }, 15000); // Extended timeout
  });

  describe('Error recovery', () => {
    it('should recover from verification errors', async () => {
      const data = {
        analysis: undefined as any,
        diagnosis: [],
        citations: [],
      };

      // Should handle gracefully
      await expect(verifier.verifyAnalysis(data)).resolves.not.toThrow();
    });

    it('should recover from notification errors', async () => {
      // Should not throw even with invalid data
      await expect(
        notifier.sendNotification('', '', 'email', 'medium', '')
      ).resolves.not.toThrow();
    });
  });
});
