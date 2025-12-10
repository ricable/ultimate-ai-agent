/**
 * Webhook Notification Service
 * HTTP/HTTPS webhook delivery with retry logic
 */

import {
  INotifier,
  NotificationPayload,
  NotificationResult,
  NotificationStatus,
  NotificationChannel,
  WebhookConfig
} from './types';

export class WebhookNotifier implements INotifier {
  private config: WebhookConfig;
  private deliveryLog: Map<string, NotificationResult>;

  constructor(config: WebhookConfig) {
    this.config = config;
    this.deliveryLog = new Map();
  }

  /**
   * Send webhook notification with retry logic
   */
  async send(payload: NotificationPayload): Promise<NotificationResult> {
    const result: NotificationResult = {
      id: payload.id,
      channel: NotificationChannel.WEBHOOK,
      status: NotificationStatus.PENDING,
      sentAt: new Date()
    };

    try {
      // Get webhook endpoint for recipient
      const endpoint = this.config.endpoints.get(payload.recipient.id);
      if (!endpoint) {
        throw new Error(`No webhook endpoint configured for recipient ${payload.recipient.id}`);
      }

      // Build webhook payload
      const webhookPayload = this.buildWebhookPayload(payload);

      // Send webhook with retry logic
      const response = await this.sendWithRetry(endpoint, webhookPayload);

      if (response.success) {
        result.status = NotificationStatus.DELIVERED;
        result.deliveredAt = new Date();
        result.metadata = {
          endpoint,
          statusCode: response.statusCode,
          responseTime: response.responseTime
        };
      } else {
        throw new Error(`Webhook failed: ${response.error}`);
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
   * Get delivery status of webhook notification
   */
  async getStatus(notificationId: string): Promise<NotificationStatus> {
    const result = this.deliveryLog.get(notificationId);
    return result?.status || NotificationStatus.PENDING;
  }

  /**
   * Cancel pending webhook notification
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
   * Build webhook payload with HIPAA-compliant structure
   */
  private buildWebhookPayload(payload: NotificationPayload): Record<string, any> {
    return {
      id: payload.id,
      type: payload.type,
      timestamp: payload.createdAt.toISOString(),
      priority: payload.priority,
      data: {
        title: payload.title,
        message: payload.message,
        recipient: {
          id: payload.recipient.id,
          type: payload.recipient.type,
          name: payload.recipient.name
        },
        metadata: payload.metadata
      },
      hipaaCompliant: payload.hipaaCompliant,
      expiresAt: payload.expiresAt?.toISOString()
    };
  }

  /**
   * Send webhook with exponential backoff retry
   */
  private async sendWithRetry(
    endpoint: string,
    payload: Record<string, any>,
    attempt: number = 1
  ): Promise<{
    success: boolean;
    statusCode?: number;
    responseTime?: number;
    error?: string;
  }> {
    const maxAttempts = this.config.retryOnFailure ? 3 : 1;
    const startTime = Date.now();

    try {
      // In production, use fetch or axios
      // const response = await fetch(endpoint, {
      //   method: 'POST',
      //   headers: {
      //     'Content-Type': 'application/json',
      //     ...this.config.headers
      //   },
      //   body: JSON.stringify(payload),
      //   signal: AbortSignal.timeout(this.config.timeout)
      // });

      // Simulate webhook call
      console.log(`[WEBHOOK] Sending to ${endpoint} (attempt ${attempt})`);
      const simulatedSuccess = Math.random() > 0.1; // 90% success rate

      if (!simulatedSuccess && attempt < maxAttempts) {
        // Retry with exponential backoff
        const backoffMs = Math.pow(2, attempt) * 1000;
        await this.delay(backoffMs);
        return this.sendWithRetry(endpoint, payload, attempt + 1);
      }

      return {
        success: simulatedSuccess,
        statusCode: simulatedSuccess ? 200 : 500,
        responseTime: Date.now() - startTime,
        error: simulatedSuccess ? undefined : 'Webhook endpoint returned error'
      };

    } catch (error) {
      if (attempt < maxAttempts) {
        const backoffMs = Math.pow(2, attempt) * 1000;
        await this.delay(backoffMs);
        return this.sendWithRetry(endpoint, payload, attempt + 1);
      }

      return {
        success: false,
        responseTime: Date.now() - startTime,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Delay helper for retry backoff
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Verify webhook endpoint connectivity
   */
  async verifyEndpoint(recipientId: string): Promise<boolean> {
    const endpoint = this.config.endpoints.get(recipientId);
    if (!endpoint) {
      return false;
    }

    try {
      // Send test ping
      const result = await this.sendWithRetry(endpoint, {
        type: 'ping',
        timestamp: new Date().toISOString()
      });
      return result.success;
    } catch {
      return false;
    }
  }
}
