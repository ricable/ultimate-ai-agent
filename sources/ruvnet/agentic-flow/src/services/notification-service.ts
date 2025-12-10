// Multi-Channel Notification Service
import { ProviderNotification, MedicalAnalysis } from '../types/medical';

export class NotificationService {
  private notifications: Map<string, ProviderNotification> = new Map();

  async sendNotification(
    analysisId: string,
    providerId: string,
    channel: ProviderNotification['channel'],
    priority: ProviderNotification['priority'],
    message: string
  ): Promise<ProviderNotification> {
    const notification: ProviderNotification = {
      id: this.generateNotificationId(),
      analysisId,
      providerId,
      channel,
      priority,
      message,
      status: 'pending',
    };

    // Simulate sending notification
    await this.sendViaChannel(notification);

    this.notifications.set(notification.id, notification);
    return notification;
  }

  async notifyProvider(analysis: MedicalAnalysis, providerId: string): Promise<ProviderNotification[]> {
    const notifications: ProviderNotification[] = [];
    const priority = this.determinePriority(analysis);

    // Determine channels based on priority
    const channels = this.getChannelsForPriority(priority);

    for (const channel of channels) {
      const message = this.formatMessage(analysis, channel);
      const notification = await this.sendNotification(
        analysis.id,
        providerId,
        channel,
        priority,
        message
      );
      notifications.push(notification);
    }

    return notifications;
  }

  async getNotificationStatus(notificationId: string): Promise<ProviderNotification | null> {
    return this.notifications.get(notificationId) || null;
  }

  async markAsDelivered(notificationId: string): Promise<void> {
    const notification = this.notifications.get(notificationId);
    if (notification) {
      notification.status = 'delivered';
      notification.deliveredAt = new Date().toISOString();
    }
  }

  async markAsRead(notificationId: string): Promise<void> {
    const notification = this.notifications.get(notificationId);
    if (notification) {
      notification.status = 'read';
      notification.readAt = new Date().toISOString();
    }
  }

  private async sendViaChannel(notification: ProviderNotification): Promise<void> {
    // Simulate channel-specific sending
    await this.delay(100);

    switch (notification.channel) {
      case 'email':
        await this.sendEmail(notification);
        break;
      case 'sms':
        await this.sendSMS(notification);
        break;
      case 'push':
        await this.sendPushNotification(notification);
        break;
      case 'in-app':
        await this.sendInAppNotification(notification);
        break;
    }

    notification.status = 'sent';
    notification.sentAt = new Date().toISOString();
  }

  private async sendEmail(notification: ProviderNotification): Promise<void> {
    // Email sending logic
    console.log(`Sending email to provider ${notification.providerId}`);
  }

  private async sendSMS(notification: ProviderNotification): Promise<void> {
    // SMS sending logic
    console.log(`Sending SMS to provider ${notification.providerId}`);
  }

  private async sendPushNotification(notification: ProviderNotification): Promise<void> {
    // Push notification logic
    console.log(`Sending push notification to provider ${notification.providerId}`);
  }

  private async sendInAppNotification(notification: ProviderNotification): Promise<void> {
    // In-app notification logic
    console.log(`Sending in-app notification to provider ${notification.providerId}`);
  }

  private determinePriority(analysis: MedicalAnalysis): ProviderNotification['priority'] {
    const criticalRisks = analysis.riskFactors.filter(r => r.severity === 'critical');
    const highRisks = analysis.riskFactors.filter(r => r.severity === 'high');

    if (criticalRisks.length > 0 || analysis.verificationScore < 0.7) {
      return 'urgent';
    }
    if (highRisks.length > 0 || analysis.confidence < 0.8) {
      return 'high';
    }
    if (analysis.confidence < 0.9) {
      return 'medium';
    }
    return 'low';
  }

  private getChannelsForPriority(priority: ProviderNotification['priority']): ProviderNotification['channel'][] {
    switch (priority) {
      case 'urgent':
        return ['sms', 'push', 'email', 'in-app'];
      case 'high':
        return ['push', 'email', 'in-app'];
      case 'medium':
        return ['email', 'in-app'];
      case 'low':
        return ['in-app'];
    }
  }

  private formatMessage(analysis: MedicalAnalysis, channel: ProviderNotification['channel']): string {
    const shortMessage = `New analysis for patient ${analysis.patientId}`;
    const detailedMessage = `${shortMessage}\nDiagnosis: ${analysis.diagnosis.join(', ')}\nConfidence: ${(analysis.confidence * 100).toFixed(1)}%`;

    switch (channel) {
      case 'sms':
        return shortMessage;
      case 'push':
        return shortMessage;
      case 'email':
      case 'in-app':
        return detailedMessage;
    }
  }

  private generateNotificationId(): string {
    return `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}
