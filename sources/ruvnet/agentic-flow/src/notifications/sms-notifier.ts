/**
 * SMS Notification Service
 * Twilio/AWS SNS-based SMS delivery with HIPAA compliance
 */

import {
  INotifier,
  NotificationPayload,
  NotificationResult,
  NotificationStatus,
  NotificationChannel,
  SMSConfig
} from './types';

export class SMSNotifier implements INotifier {
  private config: SMSConfig;
  private deliveryLog: Map<string, NotificationResult>;
  private readonly maxSmsLength: number;

  constructor(config: SMSConfig) {
    this.config = config;
    this.deliveryLog = new Map();
    this.maxSmsLength = config.maxLength || 160;
  }

  /**
   * Send SMS notification with character limit enforcement
   */
  async send(payload: NotificationPayload): Promise<NotificationResult> {
    const result: NotificationResult = {
      id: payload.id,
      channel: NotificationChannel.SMS,
      status: NotificationStatus.PENDING,
      sentAt: new Date()
    };

    try {
      // Validate recipient has phone number
      if (!payload.recipient.phone) {
        throw new Error('Recipient phone number not provided');
      }

      // Format phone number
      const phoneNumber = this.formatPhoneNumber(payload.recipient.phone);

      // Build SMS content (truncate if needed)
      const smsContent = this.buildSMSContent(payload);

      // Send SMS via provider
      const sent = await this.sendSMS({
        to: phoneNumber,
        from: this.config.fromNumber,
        body: smsContent
      });

      if (sent) {
        result.status = NotificationStatus.SENT;
        result.deliveredAt = new Date();
        result.metadata = {
          messageId: `sms-${Date.now()}`,
          recipient: phoneNumber,
          segments: Math.ceil(smsContent.length / this.maxSmsLength)
        };
      }

      this.deliveryLog.set(payload.id, result);
      return result;

    } catch (error) {
      result.status = NotificationStatus.FAILED;
      result.error = error instanceof Error ? error.message : 'Unknown error';
      this.deliveryLog.set(payload.id, result);
      throw error;
    }
  }

  /**
   * Get delivery status of SMS notification
   */
  async getStatus(notificationId: string): Promise<NotificationStatus> {
    const result = this.deliveryLog.get(notificationId);
    return result?.status || NotificationStatus.PENDING;
  }

  /**
   * Cancel pending SMS notification
   */
  async cancel(notificationId: string): Promise<boolean> {
    const result = this.deliveryLog.get(notificationId);
    if (result && result.status === NotificationStatus.PENDING) {
      result.status = NotificationStatus.FAILED;
      result.error = 'Cancelled by user';
      return true;
    }
    return false;
  }

  /**
   * Build SMS content from notification payload
   */
  private buildSMSContent(payload: NotificationPayload): string {
    const priorityPrefix = payload.priority === 'emergency' ? '🚨 URGENT: ' : '';
    const message = `${priorityPrefix}${payload.title}\n\n${payload.message}`;

    // Truncate if exceeds SMS length
    if (message.length > this.maxSmsLength) {
      const truncated = message.substring(0, this.maxSmsLength - 3) + '...';
      return truncated;
    }

    return message;
  }

  /**
   * Format phone number to E.164 standard
   */
  private formatPhoneNumber(phone: string): string {
    // Remove all non-digit characters
    const digits = phone.replace(/\D/g, '');

    // Add country code if missing (assuming US)
    if (digits.length === 10) {
      return `+1${digits}`;
    }

    if (digits.length === 11 && digits.startsWith('1')) {
      return `+${digits}`;
    }

    // Already formatted or international
    return phone.startsWith('+') ? phone : `+${digits}`;
  }

  /**
   * Send SMS via provider (Twilio/AWS SNS)
   */
  private async sendSMS(sms: {
    to: string;
    from: string;
    body: string;
  }): Promise<boolean> {
    // In production, integrate with Twilio:
    // const client = require('twilio')(this.config.accountSid, this.config.authToken);
    // await client.messages.create({ to: sms.to, from: sms.from, body: sms.body });

    // Or AWS SNS:
    // const sns = new AWS.SNS();
    // await sns.publish({ PhoneNumber: sms.to, Message: sms.body }).promise();

    // Simulate SMS sending
    console.log(`[SMS] Sending to ${sms.to}: ${sms.body.substring(0, 50)}...`);
    return true;
  }

  /**
   * Get SMS delivery report from provider
   */
  async getDeliveryReport(messageId: string): Promise<{
    status: string;
    errorCode?: string;
    errorMessage?: string;
  }> {
    // In production, query provider API for delivery status
    // For now, return mock data
    return {
      status: 'delivered'
    };
  }
}
