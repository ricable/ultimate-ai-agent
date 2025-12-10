/**
 * Notification Manager
 * Orchestrates multi-channel notification delivery with fallback
 */

import {
  NotificationPayload,
  NotificationResult,
  NotificationChannel,
  NotificationConfig,
  INotifier
} from './types';
import { EmailNotifier } from './email-notifier';
import { SMSNotifier } from './sms-notifier';
import { WebhookNotifier } from './webhook-notifier';
import { WebSocketNotifier } from './websocket-notifier';
import { InAppNotifier } from './inapp-notifier';

export class NotificationManager {
  private config: NotificationConfig;
  private notifiers: Map<NotificationChannel, INotifier>;
  private auditLog: Array<{
    timestamp: Date;
    payload: NotificationPayload;
    results: NotificationResult[];
  }>;

  constructor(config: NotificationConfig) {
    this.config = config;
    this.notifiers = new Map();
    this.auditLog = [];
    this.initializeNotifiers();
  }

  /**
   * Initialize all configured notifiers
   */
  private initializeNotifiers(): void {
    if (this.config.channels.email) {
      this.notifiers.set(
        NotificationChannel.EMAIL,
        new EmailNotifier(this.config.channels.email)
      );
    }

    if (this.config.channels.sms) {
      this.notifiers.set(
        NotificationChannel.SMS,
        new SMSNotifier(this.config.channels.sms)
      );
    }

    if (this.config.channels.webhook) {
      this.notifiers.set(
        NotificationChannel.WEBHOOK,
        new WebhookNotifier(this.config.channels.webhook)
      );
    }

    if (this.config.channels.websocket) {
      this.notifiers.set(
        NotificationChannel.WEBSOCKET,
        new WebSocketNotifier(this.config.channels.websocket)
      );
    }

    if (this.config.channels.inapp) {
      this.notifiers.set(
        NotificationChannel.INAPP,
        new InAppNotifier(this.config.channels.inapp)
      );
    }
  }

  /**
   * Send notification via all specified channels
   */
  async sendNotification(payload: NotificationPayload): Promise<NotificationResult[]> {
    const results: NotificationResult[] = [];

    // Validate payload
    this.validatePayload(payload);

    // Filter channels based on recipient preferences
    const channels = this.filterChannelsByPreference(
      payload.channels,
      payload.recipient.preferredChannels
    );

    // Send via each channel
    for (const channel of channels) {
      try {
        const notifier = this.notifiers.get(channel);
        if (!notifier) {
          console.warn(`No notifier configured for channel: ${channel}`);
          continue;
        }

        const result = await this.sendWithRetry(notifier, payload);
        results.push(result);

      } catch (error) {
        console.error(`Failed to send notification via ${channel}:`, error);
        results.push({
          id: payload.id,
          channel,
          status: 'failed' as any,
          error: error instanceof Error ? error.message : 'Unknown error'
        });
      }
    }

    // Log to audit trail
    if (this.config.security.auditLog) {
      this.auditLog.push({
        timestamp: new Date(),
        payload,
        results
      });
    }

    return results;
  }

  /**
   * Send notification with retry logic
   */
  private async sendWithRetry(
    notifier: INotifier,
    payload: NotificationPayload
  ): Promise<NotificationResult> {
    const maxAttempts = this.config.retryPolicy.maxAttempts;
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const result = await notifier.send(payload);
        return result;

      } catch (error) {
        lastError = error instanceof Error ? error : new Error('Unknown error');
        console.warn(`Attempt ${attempt}/${maxAttempts} failed:`, lastError.message);

        if (attempt < maxAttempts) {
          const delay = this.calculateBackoff(attempt);
          await this.delay(delay);
        }
      }
    }

    throw lastError || new Error('Max retry attempts exceeded');
  }

  /**
   * Calculate backoff delay for retry
   */
  private calculateBackoff(attempt: number): number {
    if (this.config.retryPolicy.exponentialBackoff) {
      return this.config.retryPolicy.backoffMs * Math.pow(2, attempt - 1);
    }
    return this.config.retryPolicy.backoffMs;
  }

  /**
   * Delay helper
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Validate notification payload
   */
  private validatePayload(payload: NotificationPayload): void {
    if (!payload.id || !payload.type || !payload.title || !payload.message) {
      throw new Error('Invalid notification payload: missing required fields');
    }

    if (!payload.recipient || !payload.recipient.id) {
      throw new Error('Invalid recipient: missing recipient ID');
    }

    if (!payload.channels || payload.channels.length === 0) {
      throw new Error('No notification channels specified');
    }

    // Validate HIPAA compliance requirements
    if (payload.hipaaCompliant) {
      if (!this.config.security.encryptPayload) {
        throw new Error('HIPAA-compliant notifications require payload encryption');
      }
      if (!this.config.security.auditLog) {
        throw new Error('HIPAA-compliant notifications require audit logging');
      }
    }
  }

  /**
   * Filter channels by recipient preference
   */
  private filterChannelsByPreference(
    requestedChannels: NotificationChannel[],
    preferredChannels: NotificationChannel[]
  ): NotificationChannel[] {
    // If no preferences, use all requested channels
    if (preferredChannels.length === 0) {
      return requestedChannels;
    }

    // Use intersection of requested and preferred channels
    return requestedChannels.filter(channel =>
      preferredChannels.includes(channel)
    );
  }

  /**
   * Send notification with fallback channels
   */
  async sendWithFallback(
    payload: NotificationPayload,
    primaryChannel: NotificationChannel,
    fallbackChannels: NotificationChannel[]
  ): Promise<NotificationResult> {
    // Try primary channel first
    try {
      const notifier = this.notifiers.get(primaryChannel);
      if (notifier) {
        const result = await notifier.send(payload);
        if (result.status === 'delivered' || result.status === 'sent') {
          return result;
        }
      }
    } catch (error) {
      console.warn(`Primary channel ${primaryChannel} failed:`, error);
    }

    // Try fallback channels
    for (const channel of fallbackChannels) {
      try {
        const notifier = this.notifiers.get(channel);
        if (notifier) {
          const result = await notifier.send(payload);
          if (result.status === 'delivered' || result.status === 'sent') {
            console.log(`Fallback successful via ${channel}`);
            return result;
          }
        }
      } catch (error) {
        console.warn(`Fallback channel ${channel} failed:`, error);
      }
    }

    throw new Error('All notification channels failed');
  }

  /**
   * Broadcast notification to multiple recipients
   */
  async broadcast(
    payloads: NotificationPayload[]
  ): Promise<Map<string, NotificationResult[]>> {
    const results = new Map<string, NotificationResult[]>();

    // Send all notifications in parallel
    const promises = payloads.map(async payload => {
      const notificationResults = await this.sendNotification(payload);
      return { recipientId: payload.recipient.id, results: notificationResults };
    });

    const allResults = await Promise.allSettled(promises);

    for (const result of allResults) {
      if (result.status === 'fulfilled') {
        results.set(result.value.recipientId, result.value.results);
      }
    }

    return results;
  }

  /**
   * Get audit log for compliance reporting
   */
  getAuditLog(filter?: {
    startDate?: Date;
    endDate?: Date;
    recipientId?: string;
    channel?: NotificationChannel;
  }): Array<{
    timestamp: Date;
    payload: NotificationPayload;
    results: NotificationResult[];
  }> {
    let filtered = this.auditLog;

    if (filter?.startDate) {
      filtered = filtered.filter(entry => entry.timestamp >= filter.startDate!);
    }

    if (filter?.endDate) {
      filtered = filtered.filter(entry => entry.timestamp <= filter.endDate!);
    }

    if (filter?.recipientId) {
      filtered = filtered.filter(entry =>
        entry.payload.recipient.id === filter.recipientId
      );
    }

    if (filter?.channel) {
      filtered = filtered.filter(entry =>
        entry.results.some(result => result.channel === filter.channel)
      );
    }

    return filtered;
  }

  /**
   * Get notification statistics
   */
  getStats(): {
    totalSent: number;
    successRate: number;
    channelStats: Map<NotificationChannel, {
      sent: number;
      delivered: number;
      failed: number;
    }>;
  } {
    const channelStats = new Map<NotificationChannel, {
      sent: number;
      delivered: number;
      failed: number;
    }>();

    let totalSent = 0;
    let totalSuccess = 0;

    for (const entry of this.auditLog) {
      for (const result of entry.results) {
        totalSent++;

        const stats = channelStats.get(result.channel) || {
          sent: 0,
          delivered: 0,
          failed: 0
        };

        stats.sent++;
        if (result.status === 'delivered' || result.status === 'sent') {
          stats.delivered++;
          totalSuccess++;
        } else if (result.status === 'failed') {
          stats.failed++;
        }

        channelStats.set(result.channel, stats);
      }
    }

    return {
      totalSent,
      successRate: totalSent > 0 ? totalSuccess / totalSent : 0,
      channelStats
    };
  }
}
