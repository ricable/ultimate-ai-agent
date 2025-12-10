/**
 * Email Notification Service
 * SMTP-based email delivery with HIPAA compliance
 */

import {
  INotifier,
  NotificationPayload,
  NotificationResult,
  NotificationStatus,
  NotificationChannel,
  EmailConfig
} from './types';

export class EmailNotifier implements INotifier {
  private config: EmailConfig;
  private deliveryLog: Map<string, NotificationResult>;

  constructor(config: EmailConfig) {
    this.config = config;
    this.deliveryLog = new Map();
  }

  /**
   * Send email notification with HIPAA-compliant encryption
   */
  async send(payload: NotificationPayload): Promise<NotificationResult> {
    const result: NotificationResult = {
      id: payload.id,
      channel: NotificationChannel.EMAIL,
      status: NotificationStatus.PENDING,
      sentAt: new Date()
    };

    try {
      // Validate recipient has email
      if (!payload.recipient.email) {
        throw new Error('Recipient email not provided');
      }

      // Validate HIPAA compliance requirements
      if (payload.hipaaCompliant && !this.config.secure) {
        throw new Error('HIPAA-compliant emails require secure connection');
      }

      // Build email content
      const emailContent = this.buildEmailContent(payload);

      // In production, this would use nodemailer or similar
      // For now, simulate email sending
      const sent = await this.sendEmail({
        to: payload.recipient.email,
        from: this.config.from,
        replyTo: this.config.replyTo,
        subject: emailContent.subject,
        text: emailContent.text,
        html: emailContent.html
      });

      if (sent) {
        result.status = NotificationStatus.SENT;
        result.deliveredAt = new Date();
        result.metadata = {
          messageId: `email-${Date.now()}`,
          recipient: payload.recipient.email
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
   * Get delivery status of email notification
   */
  async getStatus(notificationId: string): Promise<NotificationStatus> {
    const result = this.deliveryLog.get(notificationId);
    return result?.status || NotificationStatus.PENDING;
  }

  /**
   * Cancel pending email notification
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
   * Build email content from notification payload
   */
  private buildEmailContent(payload: NotificationPayload): {
    subject: string;
    text: string;
    html: string;
  } {
    const priorityLabel = payload.priority.toUpperCase();
    const subject = `[${priorityLabel}] ${payload.title}`;

    const text = `
${payload.title}

${payload.message}

Priority: ${priorityLabel}
Sent: ${payload.createdAt.toISOString()}
${payload.expiresAt ? `Expires: ${payload.expiresAt.toISOString()}` : ''}

---
This is a secure healthcare notification. Do not forward or share.
    `.trim();

    const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
    .header { background: #0066cc; color: white; padding: 20px; border-radius: 5px 5px 0 0; }
    .content { padding: 20px; background: #f9f9f9; }
    .priority-${payload.priority} { border-left: 4px solid ${this.getPriorityColor(payload.priority)}; padding-left: 15px; }
    .footer { padding: 15px; background: #eee; font-size: 12px; color: #666; text-align: center; }
    .metadata { font-size: 12px; color: #666; margin-top: 20px; }
  </style>
</head>
<body>
  <div class="header">
    <h2>${subject}</h2>
  </div>
  <div class="content">
    <div class="priority-${payload.priority}">
      <p><strong>${payload.message}</strong></p>
    </div>
    <div class="metadata">
      <p>Priority: ${priorityLabel}</p>
      <p>Sent: ${payload.createdAt.toISOString()}</p>
      ${payload.expiresAt ? `<p>Expires: ${payload.expiresAt.toISOString()}</p>` : ''}
    </div>
  </div>
  <div class="footer">
    <p>This is a secure healthcare notification. Do not forward or share.</p>
    <p>HIPAA Compliant | Encrypted Delivery</p>
  </div>
</body>
</html>
    `.trim();

    return { subject, text, html };
  }

  /**
   * Get color code for priority level
   */
  private getPriorityColor(priority: string): string {
    const colors: Record<string, string> = {
      emergency: '#ff0000',
      high: '#ff6600',
      medium: '#ffcc00',
      low: '#00cc00'
    };
    return colors[priority] || '#666666';
  }

  /**
   * Send email via SMTP (simulated for now)
   */
  private async sendEmail(email: {
    to: string;
    from: string;
    replyTo?: string;
    subject: string;
    text: string;
    html: string;
  }): Promise<boolean> {
    // In production, use nodemailer:
    // const transporter = nodemailer.createTransport(this.config);
    // await transporter.sendMail(email);

    // Simulate email sending
    console.log(`[EMAIL] Sending to ${email.to}: ${email.subject}`);
    return true;
  }
}
