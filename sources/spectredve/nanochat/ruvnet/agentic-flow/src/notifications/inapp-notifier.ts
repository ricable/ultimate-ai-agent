/**
 * In-App Notification Service
 * Persistent notification storage and retrieval
 */

import {
  INotifier,
  NotificationPayload,
  NotificationResult,
  NotificationStatus,
  NotificationChannel,
  InAppConfig
} from './types';

interface StoredNotification {
  payload: NotificationPayload;
  result: NotificationResult;
  read: boolean;
  readAt?: Date;
  deletedAt?: Date;
}

export class InAppNotifier implements INotifier {
  private config: InAppConfig;
  private notifications: Map<string, StoredNotification>;
  private userNotifications: Map<string, Set<string>>;

  constructor(config: InAppConfig) {
    this.config = config;
    this.notifications = new Map();
    this.userNotifications = new Map();
    this.startCleanupTask();
  }

  /**
   * Store in-app notification
   */
  async send(payload: NotificationPayload): Promise<NotificationResult> {
    const result: NotificationResult = {
      id: payload.id,
      channel: NotificationChannel.INAPP,
      status: NotificationStatus.DELIVERED,
      sentAt: new Date(),
      deliveredAt: new Date()
    };

    try {
      // Check user notification limit
      const userNotifs = this.userNotifications.get(payload.recipient.id) || new Set();
      if (userNotifs.size >= this.config.maxNotificationsPerUser) {
        // Remove oldest notification
        const oldestId = Array.from(userNotifs)[0];
        this.removeNotification(oldestId);
      }

      // Store notification
      const stored: StoredNotification = {
        payload,
        result,
        read: false
      };

      this.notifications.set(payload.id, stored);
      userNotifs.add(payload.id);
      this.userNotifications.set(payload.recipient.id, userNotifs);

      result.metadata = {
        stored: true,
        unreadCount: this.getUnreadCount(payload.recipient.id)
      };

      console.log(`[INAPP] Notification stored: ${payload.id} for user ${payload.recipient.id}`);
      return result;

    } catch (error) {
      result.status = NotificationStatus.FAILED;
      result.error = error instanceof Error ? error.message : 'Unknown error';
      throw error;
    }
  }

  /**
   * Get delivery status (always delivered for in-app)
   */
  async getStatus(notificationId: string): Promise<NotificationStatus> {
    const notification = this.notifications.get(notificationId);
    return notification?.result.status || NotificationStatus.PENDING;
  }

  /**
   * Cancel/delete in-app notification
   */
  async cancel(notificationId: string): Promise<boolean> {
    return this.removeNotification(notificationId);
  }

  /**
   * Get all notifications for user
   */
  async getNotificationsForUser(
    userId: string,
    options?: {
      unreadOnly?: boolean;
      limit?: number;
      offset?: number;
    }
  ): Promise<StoredNotification[]> {
    const userNotifIds = this.userNotifications.get(userId);
    if (!userNotifIds) {
      return [];
    }

    let notifications = Array.from(userNotifIds)
      .map(id => this.notifications.get(id))
      .filter((n): n is StoredNotification => n !== undefined && !n.deletedAt);

    // Filter unread only
    if (options?.unreadOnly) {
      notifications = notifications.filter(n => !n.read);
    }

    // Sort by date (newest first)
    notifications.sort((a, b) =>
      b.payload.createdAt.getTime() - a.payload.createdAt.getTime()
    );

    // Apply pagination
    const offset = options?.offset || 0;
    const limit = options?.limit || notifications.length;
    return notifications.slice(offset, offset + limit);
  }

  /**
   * Mark notification as read
   */
  async markAsRead(notificationId: string): Promise<boolean> {
    const notification = this.notifications.get(notificationId);
    if (notification && !notification.read) {
      notification.read = true;
      notification.readAt = new Date();
      notification.result.status = NotificationStatus.READ;
      return true;
    }
    return false;
  }

  /**
   * Mark all notifications as read for user
   */
  async markAllAsRead(userId: string): Promise<number> {
    const userNotifIds = this.userNotifications.get(userId);
    if (!userNotifIds) {
      return 0;
    }

    let count = 0;
    for (const id of userNotifIds) {
      const notification = this.notifications.get(id);
      if (notification && !notification.read) {
        notification.read = true;
        notification.readAt = new Date();
        notification.result.status = NotificationStatus.READ;
        count++;
      }
    }

    return count;
  }

  /**
   * Get unread count for user
   */
  getUnreadCount(userId: string): number {
    const userNotifIds = this.userNotifications.get(userId);
    if (!userNotifIds) {
      return 0;
    }

    let count = 0;
    for (const id of userNotifIds) {
      const notification = this.notifications.get(id);
      if (notification && !notification.read && !notification.deletedAt) {
        count++;
      }
    }

    return count;
  }

  /**
   * Delete notification
   */
  async deleteNotification(notificationId: string): Promise<boolean> {
    const notification = this.notifications.get(notificationId);
    if (notification) {
      notification.deletedAt = new Date();
      return true;
    }
    return false;
  }

  /**
   * Delete all notifications for user
   */
  async deleteAllForUser(userId: string): Promise<number> {
    const userNotifIds = this.userNotifications.get(userId);
    if (!userNotifIds) {
      return 0;
    }

    let count = 0;
    for (const id of userNotifIds) {
      const notification = this.notifications.get(id);
      if (notification && !notification.deletedAt) {
        notification.deletedAt = new Date();
        count++;
      }
    }

    return count;
  }

  /**
   * Remove notification from storage
   */
  private removeNotification(notificationId: string): boolean {
    const notification = this.notifications.get(notificationId);
    if (notification) {
      const userId = notification.payload.recipient.id;
      const userNotifs = this.userNotifications.get(userId);
      if (userNotifs) {
        userNotifs.delete(notificationId);
      }
      this.notifications.delete(notificationId);
      return true;
    }
    return false;
  }

  /**
   * Start cleanup task to remove expired notifications
   */
  private startCleanupTask(): void {
    setInterval(() => {
      this.cleanupExpiredNotifications();
    }, 60 * 60 * 1000); // Run every hour
  }

  /**
   * Clean up expired notifications based on retention policy
   */
  private cleanupExpiredNotifications(): void {
    const now = new Date();
    const retentionMs = this.config.retentionDays * 24 * 60 * 60 * 1000;

    for (const [id, notification] of this.notifications.entries()) {
      const age = now.getTime() - notification.payload.createdAt.getTime();

      // Remove if expired or past retention period
      const isExpired = notification.payload.expiresAt &&
        now > notification.payload.expiresAt;
      const isPastRetention = age > retentionMs;
      const isDeleted = notification.deletedAt &&
        (now.getTime() - notification.deletedAt.getTime()) > 24 * 60 * 60 * 1000; // 24h grace period

      if (isExpired || isPastRetention || isDeleted) {
        this.removeNotification(id);
        console.log(`[INAPP] Cleaned up notification: ${id}`);
      }
    }
  }

  /**
   * Get storage statistics
   */
  getStats(): {
    totalNotifications: number;
    userCount: number;
    unreadTotal: number;
    storageUsage: number;
  } {
    let unreadTotal = 0;

    for (const userNotifs of this.userNotifications.values()) {
      for (const id of userNotifs) {
        const notification = this.notifications.get(id);
        if (notification && !notification.read && !notification.deletedAt) {
          unreadTotal++;
        }
      }
    }

    return {
      totalNotifications: this.notifications.size,
      userCount: this.userNotifications.size,
      unreadTotal,
      storageUsage: JSON.stringify(Array.from(this.notifications.values())).length
    };
  }
}
