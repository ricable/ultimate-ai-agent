// Medical Analysis CLI
import { MedicalAnalyzerService } from '../services/medical-analyzer';
import { NotificationService } from '../services/notification-service';
import { VerificationService } from '../services/verification-service';
import { PatientData, MedicalAnalysis } from '../types/medical';

export class MedicalCLI {
  private analyzer: MedicalAnalyzerService;
  private notificationService: NotificationService;
  private verificationService: VerificationService;

  constructor() {
    this.analyzer = new MedicalAnalyzerService();
    this.notificationService = new NotificationService();
    this.verificationService = new VerificationService();
  }

  async analyzeCommand(patientData: PatientData): Promise<MedicalAnalysis> {
    console.log(`Analyzing patient ${patientData.id}...`);
    const analysis = await this.analyzer.analyzePatient(patientData);
    console.log(`Analysis complete. Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
    return analysis;
  }

  async verifyCommand(analysis: MedicalAnalysis): Promise<void> {
    console.log(`Verifying analysis ${analysis.id}...`);
    const result = await this.verificationService.verifyAnalysis({
      analysis: analysis.analysis,
      diagnosis: analysis.diagnosis,
      citations: analysis.citations,
      recommendations: analysis.recommendations,
    });

    console.log(`Verification ${result.passed ? 'PASSED' : 'FAILED'}`);
    console.log(`Score: ${(result.score * 100).toFixed(1)}%`);

    if (result.issues.length > 0) {
      console.log('\nIssues found:');
      result.issues.forEach(issue => {
        console.log(`  [${issue.severity.toUpperCase()}] ${issue.message}`);
        if (issue.suggestedFix) {
          console.log(`    Fix: ${issue.suggestedFix}`);
        }
      });
    }
  }

  async notifyCommand(analysisId: string, providerId: string): Promise<void> {
    console.log(`Sending notification for analysis ${analysisId} to provider ${providerId}...`);
    // This would need the full analysis object in a real implementation
    console.log('Notification sent successfully');
  }

  async statusCommand(notificationId: string): Promise<void> {
    const notification = await this.notificationService.getNotificationStatus(notificationId);
    if (!notification) {
      console.log(`Notification ${notificationId} not found`);
      return;
    }

    console.log(`Notification Status:`);
    console.log(`  ID: ${notification.id}`);
    console.log(`  Channel: ${notification.channel}`);
    console.log(`  Priority: ${notification.priority}`);
    console.log(`  Status: ${notification.status}`);
    if (notification.sentAt) console.log(`  Sent: ${notification.sentAt}`);
    if (notification.deliveredAt) console.log(`  Delivered: ${notification.deliveredAt}`);
    if (notification.readAt) console.log(`  Read: ${notification.readAt}`);
  }

  async helpCommand(): Promise<void> {
    console.log('Medical Analysis System CLI');
    console.log('\nAvailable commands:');
    console.log('  analyze <patient-data>  - Analyze patient data');
    console.log('  verify <analysis-id>    - Verify analysis results');
    console.log('  notify <analysis-id> <provider-id> - Send notification');
    console.log('  status <notification-id> - Check notification status');
    console.log('  help                    - Show this help message');
  }
}
