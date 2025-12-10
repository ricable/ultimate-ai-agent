// Test Helper Utilities
import { PatientData, MedicalAnalysis, Citation } from '../src/types/medical';

export class TestHelpers {
  /**
   * Generate mock patient data for testing
   */
  static createMockPatient(overrides?: Partial<PatientData>): PatientData {
    return {
      id: `patient_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      age: 50,
      gender: 'male',
      symptoms: ['fever', 'cough'],
      medicalHistory: [],
      medications: [],
      ...overrides,
    };
  }

  /**
   * Generate mock citation
   */
  static createMockCitation(overrides?: Partial<Citation>): Citation {
    return {
      source: 'Medical Journal',
      reference: `DOI: 10.1234/medj.${Math.floor(Math.random() * 9000 + 1000)}`,
      relevance: 0.85,
      verified: true,
      ...overrides,
    };
  }

  /**
   * Wait for a specified duration
   */
  static async wait(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Generate array of mock patients
   */
  static createMockPatients(count: number): PatientData[] {
    return Array.from({ length: count }, (_, i) =>
      this.createMockPatient({
        id: `patient_batch_${i}`,
        age: 30 + i,
      })
    );
  }

  /**
   * Verify analysis structure
   */
  static isValidAnalysis(analysis: any): boolean {
    return (
      typeof analysis.id === 'string' &&
      typeof analysis.patientId === 'string' &&
      typeof analysis.analysis === 'string' &&
      Array.isArray(analysis.diagnosis) &&
      typeof analysis.confidence === 'number' &&
      Array.isArray(analysis.citations) &&
      Array.isArray(analysis.recommendations) &&
      Array.isArray(analysis.riskFactors) &&
      typeof analysis.verificationScore === 'number' &&
      typeof analysis.timestamp === 'string' &&
      typeof analysis.metadata === 'object'
    );
  }

  /**
   * Create high-risk patient for testing urgent scenarios
   */
  static createHighRiskPatient(): PatientData {
    return this.createMockPatient({
      age: 75,
      symptoms: ['severe chest pain', 'difficulty breathing', 'confusion'],
      medicalHistory: ['heart disease', 'diabetes', 'hypertension'],
      medications: ['aspirin', 'insulin', 'lisinopril'],
      vitalSigns: {
        bloodPressure: { systolic: 180, diastolic: 110 },
        heartRate: 110,
        temperature: 38.5,
        respiratoryRate: 28,
        oxygenSaturation: 88,
      },
    });
  }

  /**
   * Create minimal patient data for edge case testing
   */
  static createMinimalPatient(): PatientData {
    return {
      id: 'minimal_patient',
      age: 40,
      gender: 'male',
      symptoms: [],
      medicalHistory: [],
      medications: [],
    };
  }

  /**
   * Validate confidence score range
   */
  static isValidConfidence(confidence: number): boolean {
    return confidence >= 0 && confidence <= 1;
  }

  /**
   * Validate verification score range
   */
  static isValidVerificationScore(score: number): boolean {
    return score >= 0 && score <= 1;
  }

  /**
   * Check if analysis contains hallucination indicators
   */
  static hasHallucinationIndicators(analysis: string): boolean {
    const suspiciousPatterns = [
      /\d{3,}-\d{3,}-\d{4}/, // Phone numbers
      /guaranteed cure/i,
      /100% effective/i,
      /miracle/i,
      /secret/i,
    ];

    return suspiciousPatterns.some(pattern => pattern.test(analysis));
  }

  /**
   * Generate random medical terminology
   */
  static generateMedicalTerms(count: number): string[] {
    const terms = [
      'hypertension', 'diabetes', 'asthma', 'arthritis',
      'hyperlipidemia', 'hypothyroidism', 'COPD', 'CAD',
      'CHF', 'CKD', 'GERD', 'IBS',
    ];

    return Array.from({ length: count }, () =>
      terms[Math.floor(Math.random() * terms.length)]
    );
  }

  /**
   * Measure test execution time
   */
  static async measureExecutionTime<T>(
    fn: () => Promise<T>
  ): Promise<{ result: T; duration: number }> {
    const start = Date.now();
    const result = await fn();
    const duration = Date.now() - start;
    return { result, duration };
  }

  /**
   * Assert analysis quality
   */
  static assertAnalysisQuality(analysis: MedicalAnalysis): void {
    if (!this.isValidAnalysis(analysis)) {
      throw new Error('Invalid analysis structure');
    }

    if (!this.isValidConfidence(analysis.confidence)) {
      throw new Error('Invalid confidence score');
    }

    if (!this.isValidVerificationScore(analysis.verificationScore)) {
      throw new Error('Invalid verification score');
    }

    if (analysis.citations.length === 0) {
      console.warn('Analysis has no citations');
    }

    if (this.hasHallucinationIndicators(analysis.analysis)) {
      throw new Error('Analysis contains hallucination indicators');
    }
  }
}

export default TestHelpers;
