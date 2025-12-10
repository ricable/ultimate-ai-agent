/**
 * Provider Notification Tool
 * Notifies healthcare providers about analyses requiring review
 */

import type { ProviderNotification, MedicalAnalysis, MCPToolResponse } from '../types';
import { ProviderWorkflow } from '../anti-hallucination/provider-workflow';

export class ProviderNotifyTool {
  private readonly workflow: ProviderWorkflow;

  constructor() {
    this.workflow = new ProviderWorkflow();
  }

  /**
   * Notify provider about analysis
   */
  async execute(args: {
    analysisId: string;
    analysis?: MedicalAnalysis;
    urgency?: 'low' | 'medium' | 'high' | 'critical';
    recipient?: string;
    channel?: 'email' | 'sms' | 'pager' | 'app';
    message?: string;
  }): Promise<MCPToolResponse> {
    try {
      const notification = this.createNotification(args);

      // Send notification
      const sent = await this.sendNotification(notification);

      const response = {
        notificationId: notification.id,
        status: sent ? 'sent' : 'failed',
        notification: {
          timestamp: notification.timestamp,
          urgency: notification.urgency,
          recipient: notification.recipient,
          channel: notification.channel,
        },
        message: sent
          ? `Provider notification sent successfully via ${notification.channel}`
          : 'Failed to send provider notification',
      };

      return {
        content: [
          {
            type: 'json',
            json: response,
          },
          {
            type: 'text',
            text: this.formatNotificationSummary(response),
          },
        ],
      };
    } catch (error: any) {
      return {
        content: [
          {
            type: 'text',
            text: `❌ Provider notification failed: ${error.message}`,
          },
        ],
        isError: true,
      };
    }
  }

  /**
   * Create notification
   */
  private createNotification(args: any): ProviderNotification {
    const urgency = args.urgency || this.determineUrgency(args.analysis);
    const channel = args.channel || this.selectChannel(urgency);

    return {
      id: `notif-${Date.now()}`,
      timestamp: Date.now(),
      analysisId: args.analysisId,
      urgency,
      recipient: args.recipient || 'provider@medical.example.com',
      channel,
      status: 'sent',
      message: args.message || this.generateMessage(args.analysis, urgency),
    };
  }

  /**
   * Send notification (simulated)
   */
  private async sendNotification(notification: ProviderNotification): Promise<boolean> {
    // In production, integrate with:
    // - Email service (SendGrid, AWS SES)
    // - SMS service (Twilio, AWS SNS)
    // - Pager service (PagerDuty, Opsgenie)
    // - Mobile app push notifications (Firebase, OneSignal)

    console.log(`📤 Sending ${notification.channel} to ${notification.recipient}:`);
    console.log(`   Urgency: ${notification.urgency}`);
    console.log(`   Message: ${notification.message.substring(0, 100)}...`);

    // Simulate async send
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve(true);
      }, 100);
    });
  }

  /**
   * Determine urgency from analysis
   */
  private determineUrgency(
    analysis?: MedicalAnalysis
  ): 'low' | 'medium' | 'high' | 'critical' {
    if (!analysis) return 'medium';

    if (analysis.urgencyLevel === 'emergency') return 'critical';
    if (analysis.urgencyLevel === 'urgent') return 'high';
    if (analysis.conditions.some(c => c.severity === 'severe')) return 'high';
    if (analysis.confidence < 0.7) return 'high';

    return 'medium';
  }

  /**
   * Select notification channel based on urgency
   */
  private selectChannel(urgency: string): 'email' | 'sms' | 'pager' | 'app' {
    switch (urgency) {
      case 'critical':
        return 'pager';
      case 'high':
        return 'sms';
      case 'medium':
        return 'app';
      default:
        return 'email';
    }
  }

  /**
   * Generate notification message
   */
  private generateMessage(analysis: MedicalAnalysis | undefined, urgency: string): string {
    if (!analysis) {
      return `
${urgency === 'critical' ? '🚨 URGENT' : '📋'} Medical Analysis Notification

A medical analysis requires your review.
Please access the system to review details.
      `.trim();
    }

    const conditions = analysis.conditions.map(c => c.name).join(', ');

    return `
${urgency === 'critical' ? '🚨 URGENT' : '📋'} Medical Analysis Notification

Analysis ID: ${analysis.id}
Urgency: ${analysis.urgencyLevel.toUpperCase()}
Confidence: ${(analysis.confidence * 100).toFixed(1)}%

Conditions Identified:
${conditions}

${analysis.requiresProviderReview ? '⚠️ PROVIDER REVIEW REQUIRED' : 'ℹ️ For your information'}

Please review and provide feedback at your earliest convenience.
    `.trim();
  }

  /**
   * Format notification summary
   */
  private formatNotificationSummary(response: any): string {
    let summary = '📤 Provider Notification Summary\n\n';

    summary += `📋 Notification ID: ${response.notificationId}\n`;
    summary += `📊 Status: ${response.status.toUpperCase()}\n`;
    summary += `⏰ Sent: ${new Date(response.notification.timestamp).toISOString()}\n`;
    summary += `🎯 Urgency: ${response.notification.urgency}\n`;
    summary += `👤 Recipient: ${response.notification.recipient}\n`;
    summary += `📱 Channel: ${response.notification.channel}\n\n`;
    summary += `💬 ${response.message}\n`;

    return summary;
  }
}
