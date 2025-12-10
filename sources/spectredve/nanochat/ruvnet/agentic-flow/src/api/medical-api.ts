// Medical Analysis API Handlers
import { MedicalAnalyzerService } from '../services/medical-analyzer';
import { NotificationService } from '../services/notification-service';
import { VerificationService } from '../services/verification-service';
import { PatientData, MedicalAnalysis, MCPToolRequest, MCPToolResponse } from '../types/medical';

export class MedicalAPI {
  private analyzer: MedicalAnalyzerService;
  private notificationService: NotificationService;
  private verificationService: VerificationService;

  constructor() {
    this.analyzer = new MedicalAnalyzerService();
    this.notificationService = new NotificationService();
    this.verificationService = new VerificationService();
  }

  async handleAnalyzeRequest(patientData: PatientData): Promise<{ success: boolean; data?: MedicalAnalysis; error?: string }> {
    try {
      const analysis = await this.analyzer.analyzePatient(patientData);
      return { success: true, data: analysis };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async handleVerifyRequest(analysisId: string, analysis: MedicalAnalysis): Promise<{ success: boolean; data?: any; error?: string }> {
    try {
      const result = await this.verificationService.verifyAnalysis({
        analysis: analysis.analysis,
        diagnosis: analysis.diagnosis,
        citations: analysis.citations,
        recommendations: analysis.recommendations,
      });
      return { success: true, data: result };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async handleNotifyRequest(analysis: MedicalAnalysis, providerId: string): Promise<{ success: boolean; data?: any; error?: string }> {
    try {
      const notifications = await this.notificationService.notifyProvider(analysis, providerId);
      return { success: true, data: { notifications } };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async handleStatusRequest(notificationId: string): Promise<{ success: boolean; data?: any; error?: string }> {
    try {
      const notification = await this.notificationService.getNotificationStatus(notificationId);
      if (!notification) {
        return { success: false, error: 'Notification not found' };
      }
      return { success: true, data: notification };
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  async handleMCPToolRequest(request: MCPToolRequest): Promise<MCPToolResponse> {
    try {
      switch (request.tool) {
        case 'medical-analyze':
          return await this.mcpAnalyze(request.params);
        case 'medical-verify':
          return await this.mcpVerify(request.params);
        case 'medical-notify':
          return await this.mcpNotify(request.params);
        case 'medical-status':
          return await this.mcpStatus(request.params);
        default:
          return { success: false, error: `Unknown tool: ${request.tool}` };
      }
    } catch (error) {
      return { success: false, error: error instanceof Error ? error.message : 'Unknown error' };
    }
  }

  private async mcpAnalyze(params: any): Promise<MCPToolResponse> {
    const result = await this.handleAnalyzeRequest(params.patientData);
    return result;
  }

  private async mcpVerify(params: any): Promise<MCPToolResponse> {
    const result = await this.handleVerifyRequest(params.analysisId, params.analysis);
    return result;
  }

  private async mcpNotify(params: any): Promise<MCPToolResponse> {
    const result = await this.handleNotifyRequest(params.analysis, params.providerId);
    return result;
  }

  private async mcpStatus(params: any): Promise<MCPToolResponse> {
    const result = await this.handleStatusRequest(params.notificationId);
    return result;
  }
}
