// Unit Tests for API Endpoints
import { MedicalAPI } from '../../src/api/medical-api';
import { PatientData, MedicalAnalysis, MCPToolRequest } from '../../src/types/medical';

describe('MedicalAPI', () => {
  let api: MedicalAPI;
  let mockPatientData: PatientData;

  beforeEach(() => {
    api = new MedicalAPI();
    mockPatientData = {
      id: 'patient_001',
      age: 55,
      gender: 'male',
      symptoms: ['chest pain', 'shortness of breath'],
      medicalHistory: ['diabetes', 'hypertension'],
      medications: ['metformin', 'lisinopril'],
      vitalSigns: {
        bloodPressure: { systolic: 145, diastolic: 92 },
        heartRate: 88,
        temperature: 37.0,
        respiratoryRate: 20,
        oxygenSaturation: 94,
      },
    };
  });

  describe('handleAnalyzeRequest', () => {
    it('should return successful analysis', async () => {
      const result = await api.handleAnalyzeRequest(mockPatientData);

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
      expect(result.error).toBeUndefined();
      expect(result.data?.patientId).toBe(mockPatientData.id);
    });

    it('should handle invalid patient data gracefully', async () => {
      const invalidData = {} as PatientData;
      const result = await api.handleAnalyzeRequest(invalidData);

      // Should still return a response structure
      expect(result).toHaveProperty('success');
    });

    it('should return analysis with all required fields', async () => {
      const result = await api.handleAnalyzeRequest(mockPatientData);

      expect(result.success).toBe(true);
      expect(result.data).toMatchObject({
        id: expect.any(String),
        patientId: expect.any(String),
        analysis: expect.any(String),
        diagnosis: expect.any(Array),
        confidence: expect.any(Number),
        citations: expect.any(Array),
        recommendations: expect.any(Array),
        riskFactors: expect.any(Array),
        verificationScore: expect.any(Number),
        timestamp: expect.any(String),
        metadata: expect.any(Object),
      });
    });
  });

  describe('handleVerifyRequest', () => {
    it('should verify analysis successfully', async () => {
      const analyzeResult = await api.handleAnalyzeRequest(mockPatientData);
      const analysis = analyzeResult.data!;

      const verifyResult = await api.handleVerifyRequest(analysis.id, analysis);

      expect(verifyResult.success).toBe(true);
      expect(verifyResult.data).toBeDefined();
      expect(verifyResult.data).toHaveProperty('passed');
      expect(verifyResult.data).toHaveProperty('score');
      expect(verifyResult.data).toHaveProperty('checks');
    });

    it('should return verification issues when present', async () => {
      const analyzeResult = await api.handleAnalyzeRequest(mockPatientData);
      const analysis = analyzeResult.data!;

      const verifyResult = await api.handleVerifyRequest(analysis.id, analysis);

      expect(verifyResult.data).toHaveProperty('issues');
      expect(verifyResult.data.issues).toBeInstanceOf(Array);
    });
  });

  describe('handleNotifyRequest', () => {
    it('should send notifications successfully', async () => {
      const analyzeResult = await api.handleAnalyzeRequest(mockPatientData);
      const analysis = analyzeResult.data!;
      const providerId = 'provider_001';

      const notifyResult = await api.handleNotifyRequest(analysis, providerId);

      expect(notifyResult.success).toBe(true);
      expect(notifyResult.data).toBeDefined();
      expect(notifyResult.data.notifications).toBeInstanceOf(Array);
      expect(notifyResult.data.notifications.length).toBeGreaterThan(0);
    });

    it('should send multiple channel notifications for urgent cases', async () => {
      const urgentPatient: PatientData = {
        ...mockPatientData,
        symptoms: ['severe chest pain', 'difficulty breathing', 'dizziness'],
      };

      const analyzeResult = await api.handleAnalyzeRequest(urgentPatient);
      const analysis = analyzeResult.data!;
      const providerId = 'provider_001';

      const notifyResult = await api.handleNotifyRequest(analysis, providerId);

      expect(notifyResult.success).toBe(true);
      expect(notifyResult.data.notifications.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('handleStatusRequest', () => {
    it('should return notification status', async () => {
      const analyzeResult = await api.handleAnalyzeRequest(mockPatientData);
      const analysis = analyzeResult.data!;
      const notifyResult = await api.handleNotifyRequest(analysis, 'provider_001');
      const notificationId = notifyResult.data.notifications[0].id;

      const statusResult = await api.handleStatusRequest(notificationId);

      expect(statusResult.success).toBe(true);
      expect(statusResult.data).toBeDefined();
      expect(statusResult.data.id).toBe(notificationId);
      expect(statusResult.data).toHaveProperty('status');
    });

    it('should return error for non-existent notification', async () => {
      const statusResult = await api.handleStatusRequest('invalid_id');

      expect(statusResult.success).toBe(false);
      expect(statusResult.error).toBeDefined();
    });
  });

  describe('handleMCPToolRequest', () => {
    it('should handle medical-analyze tool request', async () => {
      const request: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: { patientData: mockPatientData },
      };

      const result = await api.handleMCPToolRequest(request);

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });

    it('should handle unknown tool gracefully', async () => {
      const request: MCPToolRequest = {
        tool: 'unknown-tool',
        action: 'test',
        params: {},
      };

      const result = await api.handleMCPToolRequest(request);

      expect(result.success).toBe(false);
      expect(result.error).toContain('Unknown tool');
    });

    it('should handle medical-verify tool request', async () => {
      const analyzeResult = await api.handleAnalyzeRequest(mockPatientData);
      const analysis = analyzeResult.data!;

      const request: MCPToolRequest = {
        tool: 'medical-verify',
        action: 'verify',
        params: { analysisId: analysis.id, analysis },
      };

      const result = await api.handleMCPToolRequest(request);

      expect(result.success).toBe(true);
      expect(result.data).toBeDefined();
    });
  });
});
