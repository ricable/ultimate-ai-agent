/**
 * Patient Queue Tests
 */

import { PatientQueue } from '../../src/providers/patient-queue';
import { PatientQuery, QueryPriority, QueryStatus } from '../../src/providers/types';

describe('PatientQueue', () => {
  let queue: PatientQueue;

  beforeEach(() => {
    queue = new PatientQueue({
      autoAssignment: false,
      priorityWeighting: new Map([
        [QueryPriority.EMERGENCY, 4],
        [QueryPriority.URGENT, 3],
        [QueryPriority.ROUTINE, 2],
        [QueryPriority.LOW, 1]
      ]),
      maxQueueSize: 100,
      stalePeriodMinutes: 30
    });
  });

  describe('enqueue', () => {
    it('should add query to queue', async () => {
      const query: PatientQuery = {
        id: 'query-1',
        patientId: 'patient-1',
        patientName: 'John Doe',
        queryType: 'consultation',
        priority: QueryPriority.ROUTINE,
        status: QueryStatus.PENDING,
        description: 'Test query',
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      await queue.enqueue(query);

      const stats = queue.getStats();
      expect(stats.totalQueries).toBe(1);
    });

    it('should respect priority ordering', async () => {
      const emergencyQuery: PatientQuery = {
        id: 'query-emergency',
        patientId: 'patient-1',
        patientName: 'Emergency Patient',
        queryType: 'emergency',
        priority: QueryPriority.EMERGENCY,
        status: QueryStatus.PENDING,
        description: 'Emergency',
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      const routineQuery: PatientQuery = {
        id: 'query-routine',
        patientId: 'patient-2',
        patientName: 'Routine Patient',
        queryType: 'consultation',
        priority: QueryPriority.ROUTINE,
        status: QueryStatus.PENDING,
        description: 'Routine',
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      await queue.enqueue(routineQuery);
      await queue.enqueue(emergencyQuery);

      const next = queue.getNext();
      expect(next?.id).toBe('query-emergency');
    });
  });

  describe('assignToProvider', () => {
    it('should assign query to provider', async () => {
      const query: PatientQuery = {
        id: 'query-2',
        patientId: 'patient-1',
        patientName: 'Jane Doe',
        queryType: 'consultation',
        priority: QueryPriority.URGENT,
        status: QueryStatus.PENDING,
        description: 'Test query',
        createdAt: new Date(),
        updatedAt: new Date(),
        requiresConsent: false
      };

      await queue.enqueue(query);
      const success = await queue.assignToProvider('query-2', 'provider-1');

      expect(success).toBe(true);

      const providerQueries = queue.getQueriesForProvider('provider-1');
      expect(providerQueries).toHaveLength(1);
      expect(providerQueries[0].id).toBe('query-2');
      expect(providerQueries[0].status).toBe(QueryStatus.IN_REVIEW);
    });
  });

  describe('getStats', () => {
    it('should return queue statistics', () => {
      const stats = queue.getStats();

      expect(stats).toHaveProperty('totalQueries');
      expect(stats).toHaveProperty('byStatus');
      expect(stats).toHaveProperty('byPriority');
      expect(stats).toHaveProperty('averageWaitTime');
    });
  });
});
