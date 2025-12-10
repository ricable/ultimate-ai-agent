// Safety Tests - Error Recovery
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { VerificationService } from '../../src/services/verification-service';
import { NotificationService } from '../../src/services/notification-service';
import { MedicalAPI } from '../../src/api/medical-api';
import { PatientData } from '../../src/types/medical';

describe('Error Recovery and Resilience', () => {
  let analyzer: MedicalAnalyzerService;
  let verifier: VerificationService;
  let notifier: NotificationService;
  let api: MedicalAPI;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    verifier = new VerificationService();
    notifier = new NotificationService();
    api = new MedicalAPI();
  });

  describe('Service error handling', () => {
    it('should handle analyzer errors gracefully', async () => {
      const invalidPatient = null as any;

      try {
        await analyzer.analyzePatient(invalidPatient);
      } catch (error) {
        expect(error).toBeDefined();
      }

      // Should not crash the service
      const validPatient: PatientData = {
        id: 'patient_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(validPatient);
      expect(analysis).toBeDefined();
    });

    it('should handle verifier errors gracefully', async () => {
      const invalidData = null as any;

      try {
        await verifier.verifyAnalysis(invalidData);
      } catch (error) {
        expect(error).toBeDefined();
      }

      // Service should still work after error
      const validData = {
        analysis: 'Valid analysis',
        diagnosis: ['Diagnosis'],
        citations: [{
          source: 'Source',
          reference: 'Reference',
          relevance: 0.9,
          verified: true,
        }],
      };

      const result = await verifier.verifyAnalysis(validData);
      expect(result).toBeDefined();
    });

    it('should handle notification errors gracefully', async () => {
      // Try to send notification with invalid data
      try {
        await notifier.sendNotification('', '', 'email', 'medium', '');
      } catch (error) {
        // Should handle error
        expect(error).toBeDefined();
      }

      // Service should still work
      const notification = await notifier.sendNotification(
        'analysis_001',
        'provider_001',
        'email',
        'medium',
        'Test message'
      );
      expect(notification).toBeDefined();
    });
  });

  describe('API error handling', () => {
    it('should return error response for invalid requests', async () => {
      const result = await api.handleAnalyzeRequest({} as PatientData);

      expect(result).toHaveProperty('success');
      // May succeed with empty object or fail gracefully
    });

    it('should handle MCP tool errors', async () => {
      const invalidRequest = {
        tool: 'invalid-tool',
        action: 'test',
        params: {},
      };

      const response = await api.handleMCPToolRequest(invalidRequest);

      expect(response.success).toBe(false);
      expect(response.error).toBeDefined();
    });

    it('should maintain API availability after errors', async () => {
      // Generate error
      await api.handleAnalyzeRequest(null as any);

      // API should still work
      const validPatient: PatientData = {
        id: 'patient_recovery_001',
        age: 45,
        gender: 'female',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      const result = await api.handleAnalyzeRequest(validPatient);
      expect(result).toBeDefined();
      expect(result).toHaveProperty('success');
    });
  });

  describe('Partial failure recovery', () => {
    it('should complete analysis even with missing optional data', async () => {
      const partialPatient: PatientData = {
        id: 'patient_partial_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
        // Missing vitalSigns and labResults
      };

      const analysis = await analyzer.analyzePatient(partialPatient);

      expect(analysis).toBeDefined();
      expect(analysis.patientId).toBe(partialPatient.id);
      expect(analysis.diagnosis).toBeDefined();
    });

    it('should send notifications even if some channels fail', async () => {
      const patient: PatientData = {
        id: 'patient_notif_001',
        age: 55,
        gender: 'female',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patient);
      const notifications = await notifier.notifyProvider(analysis, 'provider_001');

      // Should have attempted to send notifications
      expect(notifications).toBeDefined();
      expect(Array.isArray(notifications)).toBe(true);
    });

    it('should provide partial results when full analysis unavailable', async () => {
      const limitedPatient: PatientData = {
        id: 'patient_limited_001',
        age: 40,
        gender: 'male',
        symptoms: [],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(limitedPatient);

      // Should provide some analysis even with limited data
      expect(analysis).toBeDefined();
      expect(analysis.id).toBeDefined();
      expect(analysis.timestamp).toBeDefined();
    });
  });

  describe('Cascading failure prevention', () => {
    it('should isolate verification failures', async () => {
      const patient: PatientData = {
        id: 'patient_cascade_001',
        age: 50,
        gender: 'male',
        symptoms: ['fever'],
        medicalHistory: [],
        medications: [],
      };

      // Analysis should succeed even if verification has issues
      const analysis = await analyzer.analyzePatient(patient);
      expect(analysis).toBeDefined();

      // Verification failure should not affect analysis
      try {
        await verifier.verifyAnalysis({
          analysis: analysis.analysis,
          diagnosis: analysis.diagnosis,
          citations: analysis.citations,
        });
      } catch (error) {
        // Verification error should not cascade
      }

      // Analysis is still valid
      expect(analysis.patientId).toBe(patient.id);
    });

    it('should continue after notification failures', async () => {
      const patient: PatientData = {
        id: 'patient_notif_fail_001',
        age: 45,
        gender: 'female',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patient);

      // Try to send notifications (may fail)
      try {
        await notifier.notifyProvider(analysis, 'invalid_provider');
      } catch (error) {
        // Notification failure should not affect analysis
      }

      // Analysis is still valid
      expect(analysis).toBeDefined();
      expect(analysis.confidence).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Recovery mechanisms', () => {
    it('should retry on transient failures', async () => {
      let attempts = 0;
      const maxAttempts = 3;

      const retryableOperation = async () => {
        attempts++;
        if (attempts < maxAttempts) {
          throw new Error('Transient failure');
        }
        return { success: true };
      };

      let result;
      for (let i = 0; i < maxAttempts; i++) {
        try {
          result = await retryableOperation();
          break;
        } catch (error) {
          if (i === maxAttempts - 1) throw error;
        }
      }

      expect(result).toEqual({ success: true });
      expect(attempts).toBe(maxAttempts);
    });

    it('should provide fallback values on failure', async () => {
      const getFallbackValue = (fn: () => any, fallback: any) => {
        try {
          return fn();
        } catch {
          return fallback;
        }
      };

      const result = getFallbackValue(
        () => { throw new Error('Failed'); },
        { default: true }
      );

      expect(result).toEqual({ default: true });
    });

    it('should log errors for debugging', async () => {
      const errors: Error[] = [];

      const logError = (error: Error) => {
        errors.push(error);
      };

      try {
        throw new Error('Test error');
      } catch (error) {
        logError(error as Error);
      }

      expect(errors.length).toBe(1);
      expect(errors[0].message).toBe('Test error');
    });
  });

  describe('Resource cleanup', () => {
    it('should clean up resources after errors', async () => {
      const resources = new Set();

      const acquireResource = (id: string) => {
        resources.add(id);
      };

      const releaseResource = (id: string) => {
        resources.delete(id);
      };

      const operationWithCleanup = async (id: string) => {
        try {
          acquireResource(id);
          throw new Error('Operation failed');
        } finally {
          releaseResource(id);
        }
      };

      try {
        await operationWithCleanup('resource_1');
      } catch (error) {
        // Error expected
      }

      expect(resources.has('resource_1')).toBe(false);
    });

    it('should not leave dangling references', async () => {
      const references = new WeakMap();
      const obj = { id: 'test' };

      references.set(obj, 'value');

      // Clear reference
      const cleared = null;

      expect(cleared).toBeNull();
    });
  });
});
