/**
 * Emergency Detector Tests
 */

import { EmergencyDetector } from '../../src/routing/emergency-detector';
import { EmergencyType } from '../../src/routing/types';
import { PatientQuery, QueryPriority, QueryStatus } from '../../src/providers/types';

describe('EmergencyDetector', () => {
  let detector: EmergencyDetector;

  beforeEach(() => {
    detector = new EmergencyDetector();
  });

  describe('detect', () => {
    it('should detect life-threatening emergency', () => {
      const query: PatientQuery = {
        id: 'query-1',
        patientId: 'patient-1',
        patientName: 'Emergency Patient',
        queryType: 'emergency',
        priority: QueryPriority.EMERGENCY,
        status: QueryStatus.PENDING,
        description: 'Patient experiencing severe chest pain and difficulty breathing',
        symptoms: ['chest pain', 'difficulty breathing'],
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      const emergencyType = detector.detect(query);
      expect(emergencyType).toBe(EmergencyType.LIFE_THREATENING);
    });

    it('should detect urgent care need', () => {
      const query: PatientQuery = {
        id: 'query-2',
        patientId: 'patient-2',
        patientName: 'Urgent Patient',
        queryType: 'urgent',
        priority: QueryPriority.URGENT,
        status: QueryStatus.PENDING,
        description: 'High fever and severe pain',
        symptoms: ['high fever', 'severe pain'],
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      const emergencyType = detector.detect(query);
      expect(emergencyType).toBe(EmergencyType.URGENT_CARE);
    });

    it('should classify routine queries correctly', () => {
      const query: PatientQuery = {
        id: 'query-3',
        patientId: 'patient-3',
        patientName: 'Routine Patient',
        queryType: 'consultation',
        priority: QueryPriority.ROUTINE,
        status: QueryStatus.PENDING,
        description: 'General checkup needed',
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      const emergencyType = detector.detect(query);
      expect(emergencyType).toBe(EmergencyType.ROUTINE);
    });
  });

  describe('getMatchedSignals', () => {
    it('should return matched emergency signals', () => {
      const query: PatientQuery = {
        id: 'query-4',
        patientId: 'patient-4',
        patientName: 'Test Patient',
        queryType: 'emergency',
        priority: QueryPriority.EMERGENCY,
        status: QueryStatus.PENDING,
        description: 'Patient has chest pain and is unconscious',
        symptoms: ['chest pain', 'unconscious'],
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      const signals = detector.getMatchedSignals(query);
      expect(signals.length).toBeGreaterThan(0);
      expect(signals[0].weight).toBeGreaterThanOrEqual(0.9);
    });
  });
});
