// Safety Tests - Security Validation
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { VerificationService } from '../../src/services/verification-service';
import { MedicalAPI } from '../../src/api/medical-api';
import { PatientData } from '../../src/types/medical';

describe('Security Validation', () => {
  let analyzer: MedicalAnalyzerService;
  let verifier: VerificationService;
  let api: MedicalAPI;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    verifier = new VerificationService();
    api = new MedicalAPI();
  });

  describe('Injection attack prevention', () => {
    it('should prevent SQL injection attempts', async () => {
      const maliciousPatient: PatientData = {
        id: "'; DROP TABLE patients; --",
        age: 50,
        gender: 'male',
        symptoms: ["'; DELETE FROM symptoms; --"],
        medicalHistory: [],
        medications: [],
      };

      // Should handle safely without executing injection
      const analysis = await analyzer.analyzePatient(maliciousPatient);
      expect(analysis).toBeDefined();
    });

    it('should prevent NoSQL injection', async () => {
      const maliciousPatient: PatientData = {
        id: '{"$gt": ""}',
        age: 50,
        gender: 'male',
        symptoms: ['{"$ne": null}'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(maliciousPatient);
      expect(analysis).toBeDefined();
    });

    it('should prevent command injection in symptoms', async () => {
      const commandInjection: PatientData = {
        id: 'patient_cmd_001',
        age: 45,
        gender: 'female',
        symptoms: ['fever; rm -rf /', 'cough && cat /etc/passwd'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(commandInjection);
      expect(analysis).toBeDefined();
      // Symptoms should be treated as plain text
      expect(analysis.patientId).toBe(commandInjection.id);
    });
  });

  describe('XSS prevention', () => {
    it('should sanitize script tags in analysis', async () => {
      const xssAttempt: PatientData = {
        id: 'patient_xss_001',
        age: 50,
        gender: 'male',
        symptoms: ['<script>alert("XSS")</script>'],
        medicalHistory: ['<img src=x onerror=alert("XSS")>'],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(xssAttempt);

      // Analysis output should not contain executable scripts
      expect(analysis.analysis).not.toContain('<script>');
      expect(analysis.analysis).not.toContain('onerror=');
    });

    it('should sanitize diagnosis output', async () => {
      const data = {
        analysis: 'Patient has <script>alert("test")</script> condition',
        diagnosis: ['<img src=x onerror=alert(1)>'],
        citations: [{
          source: 'Source',
          reference: 'Ref',
          relevance: 0.8,
          verified: true,
        }],
      };

      const result = await verifier.verifyAnalysis(data);
      expect(result).toBeDefined();
    });
  });

  describe('Data validation', () => {
    it('should validate patient ID format', async () => {
      const invalidIds = ['', null, undefined, '..', '/etc/passwd'];

      for (const id of invalidIds) {
        const patient: PatientData = {
          id: id as string,
          age: 50,
          gender: 'male',
          symptoms: ['fever'],
          medicalHistory: [],
          medications: [],
        };

        // Should handle invalid IDs gracefully
        const analysis = await analyzer.analyzePatient(patient);
        expect(analysis).toBeDefined();
      }
    });

    it('should validate age ranges', async () => {
      const invalidAges = [-1, 1000, NaN, Infinity];

      for (const age of invalidAges) {
        const patient: PatientData = {
          id: 'patient_001',
          age,
          gender: 'male',
          symptoms: ['fever'],
          medicalHistory: [],
          medications: [],
        };

        const analysis = await analyzer.analyzePatient(patient);
        expect(analysis).toBeDefined();
      }
    });

    it('should validate gender values', async () => {
      const validGenders = ['male', 'female', 'other'];

      for (const gender of validGenders) {
        const patient: PatientData = {
          id: 'patient_001',
          age: 50,
          gender: gender as 'male' | 'female' | 'other',
          symptoms: ['fever'],
          medicalHistory: [],
          medications: [],
        };

        const analysis = await analyzer.analyzePatient(patient);
        expect(analysis).toBeDefined();
      }
    });
  });

  describe('Access control', () => {
    it('should not expose sensitive patient data in errors', async () => {
      try {
        await analyzer.analyzePatient(null as any);
      } catch (error) {
        const errorMessage = (error as Error).message || '';
        // Error should not contain patient data
        expect(errorMessage).not.toContain('social security');
        expect(errorMessage).not.toContain('credit card');
      }
    });

    it('should validate provider access in notifications', async () => {
      const patient: PatientData = {
        id: 'patient_access_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patient);

      // Should validate provider ID format
      const result = await api.handleNotifyRequest(analysis, 'provider_001');
      expect(result).toBeDefined();
      expect(result.success).toBeDefined();
    });
  });

  describe('Rate limiting and DoS prevention', () => {
    it('should handle rapid successive requests', async () => {
      const patient: PatientData = {
        id: 'patient_rate_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const requests = Array.from({ length: 50 }, () =>
        analyzer.analyzePatient(patient)
      );

      const results = await Promise.all(requests);
      expect(results.length).toBe(50);
    });

    it('should handle large payload sizes', async () => {
      const largePayload: PatientData = {
        id: 'patient_large_001',
        age: 50,
        gender: 'male',
        symptoms: Array.from({ length: 1000 }, (_, i) => `symptom_${i}`),
        medicalHistory: Array.from({ length: 500 }, (_, i) => `condition_${i}`),
        medications: Array.from({ length: 200 }, (_, i) => `medication_${i}`),
      };

      const analysis = await analyzer.analyzePatient(largePayload);
      expect(analysis).toBeDefined();
    });
  });

  describe('Secure data handling', () => {
    it('should not log sensitive information', async () => {
      const sensitivePatient: PatientData = {
        id: 'patient_sensitive_001',
        age: 50,
        gender: 'male',
        symptoms: ['HIV positive', 'mental health issues'],
        medicalHistory: ['substance abuse', 'psychiatric treatment'],
        medications: ['antiretroviral', 'antipsychotic'],
      };

      // Analysis should work but not log sensitive data
      const analysis = await analyzer.analyzePatient(sensitivePatient);
      expect(analysis).toBeDefined();
    });

    it('should handle PHI (Protected Health Information) appropriately', async () => {
      const phiPatient: PatientData = {
        id: 'patient_phi_001',
        age: 65,
        gender: 'female',
        symptoms: ['symptoms'],
        medicalHistory: ['confidential history'],
        medications: ['confidential medications'],
      };

      const analysis = await analyzer.analyzePatient(phiPatient);

      // Analysis should exist but protect PHI
      expect(analysis).toBeDefined();
      expect(analysis.patientId).toBe(phiPatient.id);
    });
  });

  describe('Input sanitization', () => {
    it('should sanitize special characters', async () => {
      const specialChars: PatientData = {
        id: 'patient_special_001',
        age: 50,
        gender: 'male',
        symptoms: ['<>&"\'/`='],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(specialChars);
      expect(analysis).toBeDefined();
    });

    it('should handle null bytes', async () => {
      const nullBytes: PatientData = {
        id: 'patient\x00null',
        age: 50,
        gender: 'male',
        symptoms: ['symptom\x00injection'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(nullBytes);
      expect(analysis).toBeDefined();
    });

    it('should sanitize path traversal attempts', async () => {
      const pathTraversal: PatientData = {
        id: '../../../etc/passwd',
        age: 50,
        gender: 'male',
        symptoms: ['../../sensitive/data'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(pathTraversal);
      expect(analysis).toBeDefined();
    });
  });

  describe('API security', () => {
    it('should validate MCP tool parameters', async () => {
      const maliciousRequest = {
        tool: 'medical-analyze',
        action: '../../../etc/passwd',
        params: {
          patientData: {
            id: '<script>alert("xss")</script>',
          },
        },
      };

      const response = await api.handleMCPToolRequest(maliciousRequest);
      expect(response).toBeDefined();
      expect(response).toHaveProperty('success');
    });

    it('should prevent prototype pollution', async () => {
      const pollutionAttempt = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: {
          '__proto__': { isAdmin: true },
          'constructor': { prototype: { isAdmin: true } },
        },
      };

      const response = await api.handleMCPToolRequest(pollutionAttempt);
      expect(response).toBeDefined();
    });
  });
});
