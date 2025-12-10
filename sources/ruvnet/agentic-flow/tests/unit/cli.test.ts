// Unit Tests for CLI Commands
import { MedicalCLI } from '../../src/cli/medical-cli';
import { PatientData, MedicalAnalysis } from '../../src/types/medical';

describe('MedicalCLI', () => {
  let cli: MedicalCLI;
  let mockPatientData: PatientData;

  beforeEach(() => {
    cli = new MedicalCLI();
    mockPatientData = {
      id: 'patient_001',
      age: 45,
      gender: 'female',
      symptoms: ['fever', 'cough', 'fatigue'],
      medicalHistory: ['asthma'],
      medications: ['albuterol'],
      vitalSigns: {
        bloodPressure: { systolic: 120, diastolic: 80 },
        heartRate: 75,
        temperature: 38.5,
        respiratoryRate: 18,
        oxygenSaturation: 96,
      },
    };
  });

  describe('analyzeCommand', () => {
    it('should analyze patient data successfully', async () => {
      const analysis = await cli.analyzeCommand(mockPatientData);

      expect(analysis).toBeDefined();
      expect(analysis.patientId).toBe(mockPatientData.id);
      expect(analysis.diagnosis).toBeInstanceOf(Array);
      expect(analysis.confidence).toBeGreaterThan(0);
      expect(analysis.confidence).toBeLessThanOrEqual(1);
    });

    it('should return analysis with citations', async () => {
      const analysis = await cli.analyzeCommand(mockPatientData);

      expect(analysis.citations).toBeInstanceOf(Array);
      expect(analysis.citations.length).toBeGreaterThan(0);
      expect(analysis.citations[0]).toHaveProperty('source');
      expect(analysis.citations[0]).toHaveProperty('reference');
    });

    it('should include risk factors in analysis', async () => {
      const analysis = await cli.analyzeCommand(mockPatientData);

      expect(analysis.riskFactors).toBeInstanceOf(Array);
      if (analysis.riskFactors.length > 0) {
        expect(analysis.riskFactors[0]).toHaveProperty('factor');
        expect(analysis.riskFactors[0]).toHaveProperty('severity');
        expect(analysis.riskFactors[0]).toHaveProperty('confidence');
      }
    });

    it('should run hallucination checks', async () => {
      const analysis = await cli.analyzeCommand(mockPatientData);

      expect(analysis.metadata.hallucinationChecks).toBeInstanceOf(Array);
      expect(analysis.metadata.hallucinationChecks.length).toBeGreaterThan(0);
      expect(analysis.metadata.hallucinationChecks[0]).toHaveProperty('type');
      expect(analysis.metadata.hallucinationChecks[0]).toHaveProperty('passed');
    });

    it('should calculate verification score', async () => {
      const analysis = await cli.analyzeCommand(mockPatientData);

      expect(analysis.verificationScore).toBeDefined();
      expect(analysis.verificationScore).toBeGreaterThanOrEqual(0);
      expect(analysis.verificationScore).toBeLessThanOrEqual(1);
    });
  });

  describe('verifyCommand', () => {
    it('should verify analysis successfully', async () => {
      const analysis = await cli.analyzeCommand(mockPatientData);

      // Should not throw
      await expect(cli.verifyCommand(analysis)).resolves.not.toThrow();
    });
  });

  describe('helpCommand', () => {
    it('should display help information', async () => {
      const consoleSpy = jest.spyOn(console, 'log');

      await cli.helpCommand();

      expect(consoleSpy).toHaveBeenCalled();
      expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Available commands'));
    });
  });

  describe('edge cases', () => {
    it('should handle patient with minimal data', async () => {
      const minimalPatient: PatientData = {
        id: 'patient_002',
        age: 30,
        gender: 'male',
        symptoms: ['headache'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await cli.analyzeCommand(minimalPatient);

      expect(analysis).toBeDefined();
      expect(analysis.patientId).toBe(minimalPatient.id);
    });

    it('should handle patient with complex medical history', async () => {
      const complexPatient: PatientData = {
        ...mockPatientData,
        medicalHistory: ['diabetes', 'hypertension', 'obesity', 'sleep apnea'],
        medications: ['metformin', 'lisinopril', 'atorvastatin'],
      };

      const analysis = await cli.analyzeCommand(complexPatient);

      expect(analysis).toBeDefined();
      expect(analysis.riskFactors.length).toBeGreaterThan(0);
    });
  });
});
