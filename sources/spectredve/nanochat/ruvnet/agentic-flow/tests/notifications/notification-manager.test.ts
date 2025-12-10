/**
 * Notification Manager Tests
 */

import { NotificationManager } from '../../src/notifications/notification-manager';
import { NotificationPayload, NotificationChannel, NotificationPriority } from '../../src/notifications/types';

describe('NotificationManager', () => {
  let manager: NotificationManager;

  beforeEach(() => {
    manager = new NotificationManager({
      channels: {
        inapp: {
          storageType: 'memory',
          maxNotificationsPerUser: 100,
          retentionDays: 30
        }
      },
      retryPolicy: {
        maxAttempts: 3,
        backoffMs: 1000,
        exponentialBackoff: true
      },
      security: {
        encryptPayload: true,
        requireSignature: true,
        auditLog: true
      }
    });
  });

  describe('sendNotification', () => {
    it('should send notification via in-app channel', async () => {
      const payload: NotificationPayload = {
        id: 'test-1',
        type: 'test',
        title: 'Test Notification',
        message: 'This is a test',
        priority: NotificationPriority.LOW,
        channels: [NotificationChannel.INAPP],
        recipient: {
          id: 'user-1',
          type: 'provider',
          name: 'Test User',
          preferredChannels: [NotificationChannel.INAPP]
        },
        createdAt: new Date(),
        hipaaCompliant: true
      };

      const results = await manager.sendNotification(payload);

      expect(results).toHaveLength(1);
      expect(results[0].channel).toBe(NotificationChannel.INAPP);
      expect(results[0].status).toBe('delivered');
    });

    it('should validate HIPAA compliance requirements', async () => {
      const payload: NotificationPayload = {
        id: 'test-2',
        type: 'test',
        title: 'Test',
        message: 'Test',
        priority: NotificationPriority.LOW,
        channels: [NotificationChannel.INAPP],
        recipient: {
          id: 'user-1',
          type: 'provider',
          name: 'Test User',
          preferredChannels: []
        },
        createdAt: new Date(),
        hipaaCompliant: true
      };

      // Should succeed with encryption enabled
      const results = await manager.sendNotification(payload);
      expect(results).toHaveLength(1);
    });
  });

  describe('getStats', () => {
    it('should return notification statistics', () => {
      const stats = manager.getStats();

      expect(stats).toHaveProperty('totalSent');
      expect(stats).toHaveProperty('successRate');
      expect(stats).toHaveProperty('channelStats');
    });
  });
});
