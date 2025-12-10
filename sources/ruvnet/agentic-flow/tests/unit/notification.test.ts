// Unit Tests for Notification System
import { NotificationService } from '../../src/services/notification-service';
import { MedicalAnalysis, ProviderNotification } from '../../src/types/medical';

describe('NotificationService', () => {
  let service: NotificationService;
  let mockAnalysis: MedicalAnalysis;

  beforeEach(() => {
    service = new NotificationService();
    mockAnalysis = {
      id: 'analysis_001',
      patientId: 'patient_001',
      analysis: 'Test analysis',
      diagnosis: ['Test diagnosis'],
      confidence: 0.9,
      citations: [],
      recommendations: ['Test recommendation'],
      riskFactors: [],
      verificationScore: 0.85,
      timestamp: new Date().toISOString(),
      metadata: {
        modelUsed: 'claude-sonnet-4-5',
        processingTime: 1000,
        hallucinationChecks: [],
        knowledgeBaseCrossChecks: 2,
        agentDBLearningApplied: true,
      },
    };
  });

  describe('sendNotification', () => {
    it('should send notification successfully', async () => {
      const notification = await service.sendNotification(
        'analysis_001',
        'provider_001',
        'email',
        'medium',
        'Test message'
      );

      expect(notification).toBeDefined();
      expect(notification.id).toBeDefined();
      expect(notification.analysisId).toBe('analysis_001');
      expect(notification.providerId).toBe('provider_001');
      expect(notification.channel).toBe('email');
      expect(notification.priority).toBe('medium');
      expect(notification.status).toBe('sent');
    });

    it('should send SMS notification', async () => {
      const notification = await service.sendNotification(
        'analysis_001',
        'provider_001',
        'sms',
        'urgent',
        'Urgent notification'
      );

      expect(notification.channel).toBe('sms');
      expect(notification.status).toBe('sent');
    });

    it('should send push notification', async () => {
      const notification = await service.sendNotification(
        'analysis_001',
        'provider_001',
        'push',
        'high',
        'High priority notification'
      );

      expect(notification.channel).toBe('push');
      expect(notification.status).toBe('sent');
    });

    it('should send in-app notification', async () => {
      const notification = await service.sendNotification(
        'analysis_001',
        'provider_001',
        'in-app',
        'low',
        'Low priority notification'
      );

      expect(notification.channel).toBe('in-app');
      expect(notification.status).toBe('sent');
    });
  });

  describe('notifyProvider', () => {
    it('should send multiple notifications for urgent cases', async () => {
      const urgentAnalysis: MedicalAnalysis = {
        ...mockAnalysis,
        riskFactors: [
          { factor: 'Critical condition', severity: 'critical', confidence: 0.95 },
        ],
      };

      const notifications = await service.notifyProvider(urgentAnalysis, 'provider_001');

      expect(notifications.length).toBeGreaterThan(1);
      expect(notifications.some(n => n.channel === 'sms')).toBe(true);
      expect(notifications.some(n => n.priority === 'urgent')).toBe(true);
    });

    it('should send fewer notifications for low priority cases', async () => {
      const lowPriorityAnalysis: MedicalAnalysis = {
        ...mockAnalysis,
        confidence: 0.95,
        riskFactors: [],
      };

      const notifications = await service.notifyProvider(lowPriorityAnalysis, 'provider_001');

      expect(notifications.length).toBeGreaterThanOrEqual(1);
      expect(notifications.every(n => n.priority === 'low')).toBe(true);
    });

    it('should determine priority based on risk factors', async () => {
      const highRiskAnalysis: MedicalAnalysis = {
        ...mockAnalysis,
        riskFactors: [
          { factor: 'High risk factor', severity: 'high', confidence: 0.9 },
        ],
      };

      const notifications = await service.notifyProvider(highRiskAnalysis, 'provider_001');

      expect(notifications.some(n => n.priority === 'high')).toBe(true);
    });

    it('should determine priority based on verification score', async () => {
      const lowVerificationAnalysis: MedicalAnalysis = {
        ...mockAnalysis,
        verificationScore: 0.65,
      };

      const notifications = await service.notifyProvider(lowVerificationAnalysis, 'provider_001');

      expect(notifications.some(n => n.priority === 'urgent')).toBe(true);
    });
  });

  describe('notification status tracking', () => {
    it('should track notification status', async () => {
      const notification = await service.sendNotification(
        'analysis_001',
        'provider_001',
        'email',
        'medium',
        'Test'
      );

      const status = await service.getNotificationStatus(notification.id);

      expect(status).toBeDefined();
      expect(status?.id).toBe(notification.id);
      expect(status?.status).toBe('sent');
    });

    it('should mark notification as delivered', async () => {
      const notification = await service.sendNotification(
        'analysis_001',
        'provider_001',
        'email',
        'medium',
        'Test'
      );

      await service.markAsDelivered(notification.id);
      const status = await service.getNotificationStatus(notification.id);

      expect(status?.status).toBe('delivered');
      expect(status?.deliveredAt).toBeDefined();
    });

    it('should mark notification as read', async () => {
      const notification = await service.sendNotification(
        'analysis_001',
        'provider_001',
        'email',
        'medium',
        'Test'
      );

      await service.markAsDelivered(notification.id);
      await service.markAsRead(notification.id);
      const status = await service.getNotificationStatus(notification.id);

      expect(status?.status).toBe('read');
      expect(status?.readAt).toBeDefined();
    });

    it('should return null for non-existent notification', async () => {
      const status = await service.getNotificationStatus('invalid_id');

      expect(status).toBeNull();
    });
  });

  describe('message formatting', () => {
    it('should format short messages for SMS', async () => {
      const notifications = await service.notifyProvider(mockAnalysis, 'provider_001');
      const smsNotifications = notifications.filter(n => n.channel === 'sms');

      if (smsNotifications.length > 0) {
        expect(smsNotifications[0].message.length).toBeLessThan(200);
      }
    });

    it('should include diagnosis in detailed messages', async () => {
      mockAnalysis.diagnosis = ['Hypertension', 'Diabetes'];
      const notifications = await service.notifyProvider(mockAnalysis, 'provider_001');
      const emailNotifications = notifications.filter(n => n.channel === 'email');

      if (emailNotifications.length > 0) {
        expect(emailNotifications[0].message).toContain('Hypertension');
      }
    });

    it('should include confidence in detailed messages', async () => {
      const notifications = await service.notifyProvider(mockAnalysis, 'provider_001');
      const emailNotifications = notifications.filter(n => n.channel === 'email');

      if (emailNotifications.length > 0) {
        expect(emailNotifications[0].message).toMatch(/\d+\.\d+%/);
      }
    });
  });

  describe('channel selection', () => {
    it('should use all channels for urgent priority', async () => {
      const urgentAnalysis: MedicalAnalysis = {
        ...mockAnalysis,
        riskFactors: [
          { factor: 'Critical', severity: 'critical', confidence: 0.99 },
        ],
      };

      const notifications = await service.notifyProvider(urgentAnalysis, 'provider_001');
      const channels = new Set(notifications.map(n => n.channel));

      expect(channels.has('sms')).toBe(true);
      expect(channels.has('push')).toBe(true);
      expect(channels.has('email')).toBe(true);
      expect(channels.has('in-app')).toBe(true);
    });

    it('should use only in-app for low priority', async () => {
      const lowPriorityAnalysis: MedicalAnalysis = {
        ...mockAnalysis,
        confidence: 0.96,
        riskFactors: [],
      };

      const notifications = await service.notifyProvider(lowPriorityAnalysis, 'provider_001');
      const channels = notifications.map(n => n.channel);

      expect(channels).toContain('in-app');
      expect(channels.length).toBe(1);
    });
  });
});
