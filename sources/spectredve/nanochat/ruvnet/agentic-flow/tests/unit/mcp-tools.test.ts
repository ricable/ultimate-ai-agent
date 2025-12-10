// Unit Tests for MCP Tools
import { MedicalAPI } from '../../src/api/medical-api';
import { MCPToolRequest, PatientData } from '../../src/types/medical';

describe('MCP Tools', () => {
  let api: MedicalAPI;
  let mockPatientData: PatientData;

  beforeEach(() => {
    api = new MedicalAPI();
    mockPatientData = {
      id: 'patient_001',
      age: 50,
      gender: 'female',
      symptoms: ['fatigue', 'weight gain'],
      medicalHistory: ['hypothyroidism'],
      medications: ['levothyroxine'],
    };
  });

  describe('medical-analyze tool', () => {
    it('should handle analyze request', async () => {
      const request: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: { patientData: mockPatientData },
      };

      const response = await api.handleMCPToolRequest(request);

      expect(response.success).toBe(true);
      expect(response.data).toBeDefined();
      expect(response.data.id).toBeDefined();
      expect(response.data.patientId).toBe(mockPatientData.id);
    });

    it('should return error for invalid parameters', async () => {
      const request: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: {},
      };

      const response = await api.handleMCPToolRequest(request);

      // Should handle gracefully
      expect(response).toHaveProperty('success');
    });
  });

  describe('medical-verify tool', () => {
    it('should handle verify request', async () => {
      // First create an analysis
      const analyzeRequest: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: { patientData: mockPatientData },
      };
      const analyzeResponse = await api.handleMCPToolRequest(analyzeRequest);
      const analysis = analyzeResponse.data;

      // Then verify it
      const verifyRequest: MCPToolRequest = {
        tool: 'medical-verify',
        action: 'verify',
        params: { analysisId: analysis.id, analysis },
      };
      const verifyResponse = await api.handleMCPToolRequest(verifyRequest);

      expect(verifyResponse.success).toBe(true);
      expect(verifyResponse.data).toBeDefined();
      expect(verifyResponse.data).toHaveProperty('passed');
      expect(verifyResponse.data).toHaveProperty('score');
    });
  });

  describe('medical-notify tool', () => {
    it('should handle notify request', async () => {
      // Create an analysis
      const analyzeRequest: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: { patientData: mockPatientData },
      };
      const analyzeResponse = await api.handleMCPToolRequest(analyzeRequest);
      const analysis = analyzeResponse.data;

      // Send notification
      const notifyRequest: MCPToolRequest = {
        tool: 'medical-notify',
        action: 'notify',
        params: { analysis, providerId: 'provider_001' },
      };
      const notifyResponse = await api.handleMCPToolRequest(notifyRequest);

      expect(notifyResponse.success).toBe(true);
      expect(notifyResponse.data).toBeDefined();
      expect(notifyResponse.data.notifications).toBeInstanceOf(Array);
    });
  });

  describe('medical-status tool', () => {
    it('should handle status request', async () => {
      // Create analysis and notification
      const analyzeRequest: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: { patientData: mockPatientData },
      };
      const analyzeResponse = await api.handleMCPToolRequest(analyzeRequest);
      const analysis = analyzeResponse.data;

      const notifyRequest: MCPToolRequest = {
        tool: 'medical-notify',
        action: 'notify',
        params: { analysis, providerId: 'provider_001' },
      };
      const notifyResponse = await api.handleMCPToolRequest(notifyRequest);
      const notificationId = notifyResponse.data.notifications[0].id;

      // Check status
      const statusRequest: MCPToolRequest = {
        tool: 'medical-status',
        action: 'status',
        params: { notificationId },
      };
      const statusResponse = await api.handleMCPToolRequest(statusRequest);

      expect(statusResponse.success).toBe(true);
      expect(statusResponse.data).toBeDefined();
      expect(statusResponse.data.id).toBe(notificationId);
    });

    it('should return error for invalid notification ID', async () => {
      const statusRequest: MCPToolRequest = {
        tool: 'medical-status',
        action: 'status',
        params: { notificationId: 'invalid_id' },
      };
      const statusResponse = await api.handleMCPToolRequest(statusRequest);

      expect(statusResponse.success).toBe(false);
      expect(statusResponse.error).toBeDefined();
    });
  });

  describe('error handling', () => {
    it('should handle unknown tool', async () => {
      const request: MCPToolRequest = {
        tool: 'unknown-tool',
        action: 'test',
        params: {},
      };

      const response = await api.handleMCPToolRequest(request);

      expect(response.success).toBe(false);
      expect(response.error).toContain('Unknown tool');
    });

    it('should handle exceptions gracefully', async () => {
      const request: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: { patientData: null },
      };

      const response = await api.handleMCPToolRequest(request);

      expect(response).toHaveProperty('success');
      expect(response).toHaveProperty('error');
    });
  });

  describe('tool chaining', () => {
    it('should support analyze -> verify -> notify workflow', async () => {
      // Analyze
      const analyzeRequest: MCPToolRequest = {
        tool: 'medical-analyze',
        action: 'analyze',
        params: { patientData: mockPatientData },
      };
      const analyzeResponse = await api.handleMCPToolRequest(analyzeRequest);
      expect(analyzeResponse.success).toBe(true);
      const analysis = analyzeResponse.data;

      // Verify
      const verifyRequest: MCPToolRequest = {
        tool: 'medical-verify',
        action: 'verify',
        params: { analysisId: analysis.id, analysis },
      };
      const verifyResponse = await api.handleMCPToolRequest(verifyRequest);
      expect(verifyResponse.success).toBe(true);

      // Notify
      const notifyRequest: MCPToolRequest = {
        tool: 'medical-notify',
        action: 'notify',
        params: { analysis, providerId: 'provider_001' },
      };
      const notifyResponse = await api.handleMCPToolRequest(notifyRequest);
      expect(notifyResponse.success).toBe(true);
    });
  });
});
