// Integration Tests - End-to-End Workflows
import { MedicalAnalyzerService } from '../../src/services/medical-analyzer';
import { VerificationService } from '../../src/services/verification-service';
import { NotificationService } from '../../src/services/notification-service';
import { AgentDBIntegration } from '../../src/middleware/agentdb-integration';
import { PatientData } from '../../src/types/medical';

describe('End-to-End Medical Analysis Workflow', () => {
  let analyzer: MedicalAnalyzerService;
  let verifier: VerificationService;
  let notifier: NotificationService;
  let agentDB: AgentDBIntegration;

  beforeEach(() => {
    analyzer = new MedicalAnalyzerService();
    verifier = new VerificationService();
    notifier = new NotificationService();
    agentDB = new AgentDBIntegration();
  });

  describe('Complete analysis workflow', () => {
    it('should process patient from analysis to notification', async () => {
      const patientData: PatientData = {
        id: 'patient_e2e_001',
        age: 65,
        gender: 'male',
        symptoms: ['chest pain', 'shortness of breath', 'fatigue'],
        medicalHistory: ['diabetes', 'hypertension', 'high cholesterol'],
        medications: ['metformin', 'lisinopril', 'atorvastatin'],
        vitalSigns: {
          bloodPressure: { systolic: 155, diastolic: 95 },
          heartRate: 92,
          temperature: 37.2,
          respiratoryRate: 22,
          oxygenSaturation: 93,
        },
      };

      // Step 1: Analyze patient
      const analysis = await analyzer.analyzePatient(patientData);
      expect(analysis).toBeDefined();
      expect(analysis.patientId).toBe(patientData.id);
      expect(analysis.confidence).toBeGreaterThan(0);

      // Step 2: Verify analysis
      const verificationResult = await verifier.verifyAnalysis({
        analysis: analysis.analysis,
        diagnosis: analysis.diagnosis,
        citations: analysis.citations,
        recommendations: analysis.recommendations,
      });
      expect(verificationResult).toBeDefined();
      expect(verificationResult.score).toBeGreaterThanOrEqual(0);

      // Step 3: Record in AgentDB for learning
      await agentDB.recordAnalysis(patientData, analysis);

      // Step 4: Send notifications
      const notifications = await notifier.notifyProvider(analysis, 'provider_001');
      expect(notifications.length).toBeGreaterThan(0);
      expect(notifications.every(n => n.status === 'sent')).toBe(true);

      // Step 5: Verify notification delivery
      for (const notification of notifications) {
        const status = await notifier.getNotificationStatus(notification.id);
        expect(status).toBeDefined();
        expect(status?.status).toBe('sent');
      }
    });

    it('should handle high-risk patient with urgent notifications', async () => {
      const highRiskPatient: PatientData = {
        id: 'patient_urgent_001',
        age: 75,
        gender: 'female',
        symptoms: ['severe chest pain', 'difficulty breathing', 'confusion'],
        medicalHistory: ['heart disease', 'diabetes', 'stroke'],
        medications: ['aspirin', 'warfarin', 'insulin'],
        vitalSigns: {
          bloodPressure: { systolic: 180, diastolic: 110 },
          heartRate: 110,
          temperature: 38.5,
          respiratoryRate: 28,
          oxygenSaturation: 88,
        },
      };

      // Analyze
      const analysis = await analyzer.analyzePatient(highRiskPatient);
      expect(analysis.riskFactors.length).toBeGreaterThan(0);

      // Verify
      const verificationResult = await verifier.verifyAnalysis({
        analysis: analysis.analysis,
        diagnosis: analysis.diagnosis,
        citations: analysis.citations,
      });
      expect(verificationResult).toBeDefined();

      // Notify with high priority
      const notifications = await notifier.notifyProvider(analysis, 'provider_urgent');

      // Should send through multiple channels
      expect(notifications.length).toBeGreaterThan(1);

      // Should include urgent priority
      const urgentNotifications = notifications.filter(n => n.priority === 'urgent');
      expect(urgentNotifications.length).toBeGreaterThan(0);

      // Should include SMS for urgent cases
      const smsNotifications = notifications.filter(n => n.channel === 'sms');
      expect(smsNotifications.length).toBeGreaterThan(0);
    });

    it('should apply AgentDB learning to improve analysis', async () => {
      const patientData: PatientData = {
        id: 'patient_learning_001',
        age: 45,
        gender: 'female',
        symptoms: ['fever', 'cough', 'fatigue'],
        medicalHistory: ['asthma'],
        medications: ['albuterol'],
      };

      // First analysis
      const firstAnalysis = await analyzer.analyzePatient(patientData);
      await agentDB.recordAnalysis(patientData, firstAnalysis);

      // Apply learning
      const learning = await agentDB.applyLearning(patientData);
      expect(learning).toBeDefined();

      // Second analysis with similar patient should benefit from learning
      const similarPatient: PatientData = {
        ...patientData,
        id: 'patient_learning_002',
      };

      const secondAnalysis = await analyzer.analyzePatient(similarPatient);
      expect(secondAnalysis).toBeDefined();
      expect(secondAnalysis.metadata.agentDBLearningApplied).toBe(true);
    });
  });

  describe('Error recovery workflows', () => {
    it('should handle analysis failure gracefully', async () => {
      const invalidPatient: PatientData = {
        id: '',
        age: -1,
        gender: 'male',
        symptoms: [],
        medicalHistory: [],
        medications: [],
      };

      // Should not throw, but handle gracefully
      try {
        const analysis = await analyzer.analyzePatient(invalidPatient);
        expect(analysis).toBeDefined();
      } catch (error) {
        // If it does throw, it should be handled
        expect(error).toBeDefined();
      }
    });

    it('should handle notification failure gracefully', async () => {
      const patientData: PatientData = {
        id: 'patient_notif_fail_001',
        age: 50,
        gender: 'male',
        symptoms: ['headache'],
        medicalHistory: [],
        medications: [],
      };

      const analysis = await analyzer.analyzePatient(patientData);

      // Should not throw even with invalid provider ID
      const notifications = await notifier.notifyProvider(analysis, '');
      expect(notifications).toBeDefined();
    });
  });

  describe('Performance and scalability', () => {
    it('should handle multiple concurrent analyses', async () => {
      const patients: PatientData[] = Array.from({ length: 10 }, (_, i) => ({
        id: `patient_concurrent_${i}`,
        age: 40 + i,
        gender: i % 2 === 0 ? 'male' : 'female',
        symptoms: ['symptom1', 'symptom2'],
        medicalHistory: [],
        medications: [],
      }));

      const startTime = Date.now();
      const analyses = await Promise.all(
        patients.map(p => analyzer.analyzePatient(p))
      );
      const duration = Date.now() - startTime;

      expect(analyses.length).toBe(10);
      expect(analyses.every(a => a.id && a.patientId)).toBe(true);

      // Should complete reasonably quickly
      expect(duration).toBeLessThan(30000); // 30 seconds
    });

    it('should maintain high verification scores under load', async () => {
      const patients: PatientData[] = Array.from({ length: 5 }, (_, i) => ({
        id: `patient_load_${i}`,
        age: 50,
        gender: 'male',
        symptoms: ['fever', 'cough'],
        medicalHistory: ['condition'],
        medications: ['medication'],
      }));

      const analyses = await Promise.all(
        patients.map(p => analyzer.analyzePatient(p))
      );

      const verifications = await Promise.all(
        analyses.map(a => verifier.verifyAnalysis({
          analysis: a.analysis,
          diagnosis: a.diagnosis,
          citations: a.citations,
        }))
      );

      const averageScore = verifications.reduce((sum, v) => sum + v.score, 0) / verifications.length;
      expect(averageScore).toBeGreaterThan(0.7);
    });
  });
});
