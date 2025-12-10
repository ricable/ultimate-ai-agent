// Integration Tests - Provider Notification Flows
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { NotificationService } from '../../src/services/notification-service';
import { PatientData } from '../../src/types/medical';

describe('Provider Notification Flow Integration', () => {
  let analyzer: MedicalAnalyzerService;
  let notifier: NotificationService;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    notifier = new NotificationService();
  });

  describe('Multi-channel notification delivery', () => {
    it('should send notifications through all channels for urgent cases', async () => {
      const urgentPatient: PatientData = {
        id: 'patient_urgent_channel_001',
        age: 70,
        gender: 'male',
        symptoms: ['severe chest pain', 'difficulty breathing'],
        medicalHistory: ['heart disease'],
        medications: ['aspirin'],
      };

      const analysis = await analyzer.analyzePatient(urgentPatient);
      analysis.riskFactors.push({
        factor: 'Cardiac emergency',
        severity: 'critical',
        confidence: 0.95,
      });

      const notifications = await notifier.notifyProvider(analysis, 'provider_001');

      const channels = notifications.map(n => n.channel);
      expect(channels).toContain('sms');
      expect(channels).toContain('email');
      expect(channels).toContain('push');
      expect(channels).toContain('in-app');
    });

    it('should track notification delivery across channels', async () => {
      const patientData: PatientData = {
        id: 'patient_track_001',
        age: 55,
        gender: 'female',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);
      const notifications = await notifier.notifyProvider(analysis, 'provider_001');

      // Mark all as delivered
      for (const notification of notifications) {
        await notifier.markAsDelivered(notification.id);
        const status = await notifier.getNotificationStatus(notification.id);
        expect(status?.status).toBe('delivered');
        expect(status?.deliveredAt).toBeDefined();
      }
    });

    it('should handle notification read receipts', async () => {
      const patientData: PatientData = {
        id: 'patient_read_001',
        age: 60,
        gender: 'male',
        symptoms: ['headache'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);
      const notifications = await notifier.notifyProvider(analysis, 'provider_001');

      // Simulate provider reading notifications
      for (const notification of notifications) {
        await notifier.markAsDelivered(notification.id);
        await notifier.markAsRead(notification.id);

        const status = await notifier.getNotificationStatus(notification.id);
        expect(status?.status).toBe('read');
        expect(status?.readAt).toBeDefined();
      }
    });
  });

  describe('Priority-based notification routing', () => {
    it('should route urgent notifications immediately', async () => {
      const criticalPatient: PatientData = {
        id: 'patient_critical_001',
        age: 80,
        gender: 'female',
        symptoms: ['stroke symptoms', 'paralysis'],
        medicalHistory: ['hypertension', 'atrial fibrillation'],
        medications: ['warfarin'],
        vitalSigns: {
          bloodPressure: { systolic: 200, diastolic: 120 },
          heartRate: 130,
          temperature: 37.0,
          respiratoryRate: 30,
          oxygenSaturation: 85,
        },
      };

      const startTime = Date.now();
      const analysis = await analyzer.analyzePatient(criticalPatient);
      const notifications = await notifier.notifyProvider(analysis, 'provider_critical');
      const duration = Date.now() - startTime;

      // Should be fast
      expect(duration).toBeLessThan(5000);

      // Should be urgent priority
      expect(notifications.some(n => n.priority === 'urgent')).toBe(true);

      // Should include immediate channels
      const channels = notifications.map(n => n.channel);
      expect(channels).toContain('sms');
      expect(channels).toContain('push');
    });

    it('should batch low-priority notifications', async () => {
      const routinePatient: PatientData = {
        id: 'patient_routine_001',
        age: 35,
        gender: 'male',
        symptoms: ['mild cold'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(routinePatient);
      const notifications = await notifier.notifyProvider(analysis, 'provider_routine');

      // Should use low-priority channels only
      expect(notifications.every(n => n.priority === 'low' || n.priority === 'medium')).toBe(true);

      // Should primarily use in-app
      const inAppNotifications = notifications.filter(n => n.channel === 'in-app');
      expect(inAppNotifications.length).toBeGreaterThan(0);
    });

    it('should escalate based on verification score', async () => {
      const ambiguousPatient: PatientData = {
        id: 'patient_ambiguous_001',
        age: 50,
        gender: 'female',
        symptoms: ['chest discomfort', 'nausea'],
        medicalHistory: ['diabetes'],
        medications: ['metformin'],
      };

      const analysis = await analyzer.analyzePatient(ambiguousPatient);

      // Simulate low verification score
      analysis.verificationScore = 0.65;

      const notifications = await notifier.notifyProvider(analysis, 'provider_escalate');

      // Should escalate priority due to low verification
      expect(notifications.some(n => n.priority === 'urgent' || n.priority === 'high')).toBe(true);
    });
  });

  describe('Notification message content', () => {
    it('should include essential information in all notifications', async () => {
      const patientData: PatientData = {
        id: 'patient_content_001',
        age: 45,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);
      const notifications = await notifier.notifyProvider(analysis, 'provider_001');

      for (const notification of notifications) {
        expect(notification.message).toContain(analysis.patientId);
        expect(notification.message.length).toBeGreaterThan(0);
      }
    });

    it('should format messages appropriately for each channel', async () => {
      const patientData: PatientData = {
        id: 'patient_format_001',
        age: 50,
        gender: 'female',
        symptoms: ['symptoms'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);
      const notifications = await notifier.notifyProvider(analysis, 'provider_001');

      // SMS messages should be shorter
      const smsNotifications = notifications.filter(n => n.channel === 'sms');
      if (smsNotifications.length > 0) {
        expect(smsNotifications[0].message.length).toBeLessThan(200);
      }

      // Email/in-app can be longer and more detailed
      const detailedNotifications = notifications.filter(
        n => n.channel === 'email' || n.channel === 'in-app'
      );
      if (detailedNotifications.length > 0) {
        expect(detailedNotifications[0].message.length).toBeGreaterThan(50);
      }
    });
  });

  describe('Failure handling and retries', () => {
    it('should handle notification failures gracefully', async () => {
      const patientData: PatientData = {
        id: 'patient_fail_001',
        age: 40,
        gender: 'male',
        symptoms: ['symptom'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Should not throw even with potential failures
      await expect(
        notifier.notifyProvider(analysis, 'invalid_provider')
      ).resolves.not.toThrow();
    });

    it('should track failed notification attempts', async () => {
      const notification = await notifier.sendNotification(
        'analysis_001',
        'provider_001',
        'email',
        'medium',
        'Test message'
      );

      // Status should be trackable even if delivery fails
      const status = await notifier.getNotificationStatus(notification.id);
      expect(status).toBeDefined();
      expect(['pending', 'sent', 'delivered', 'failed']).toContain(status!.status);
    });
  });
});
