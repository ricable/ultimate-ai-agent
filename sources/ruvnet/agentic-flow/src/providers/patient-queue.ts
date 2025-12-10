/**
 * Patient Query Queue Management
 * Handles queuing, prioritization, and assignment of patient queries
 */

import {
  PatientQuery,
  Provider,
  QueryStatus,
  QueryPriority,
  ProviderStatus,
  QueueConfig
} from './types';

export class PatientQueue {
  private queue: Map<string, PatientQuery>;
  private priorityQueue: Map<QueryPriority, Set<string>>;
  private providerAssignments: Map<string, Set<string>>;
  private config: QueueConfig;

  constructor(config: QueueConfig) {
    this.queue = new Map();
    this.priorityQueue = new Map();
    this.providerAssignments = new Map();
    this.config = config;

    // Initialize priority queues
    Object.values(QueryPriority).forEach(priority => {
      this.priorityQueue.set(priority, new Set());
    });
  }

  /**
   * Add query to queue
   */
  async enqueue(query: PatientQuery): Promise<void> {
    // Check queue size limit
    if (this.queue.size >= this.config.maxQueueSize) {
      throw new Error('Queue is full');
    }

    // Validate query
    if (!query.id || !query.patientId) {
      throw new Error('Invalid query: missing required fields');
    }

    // Add to main queue
    this.queue.set(query.id, query);

    // Add to priority queue
    const prioritySet = this.priorityQueue.get(query.priority);
    if (prioritySet) {
      prioritySet.add(query.id);
    }

    console.log(`[QUEUE] Added query ${query.id} with priority ${query.priority}`);

    // Auto-assign if configured
    if (this.config.autoAssignment) {
      await this.autoAssignQuery(query.id);
    }
  }

  /**
   * Remove query from queue
   */
  dequeue(queryId: string): PatientQuery | undefined {
    const query = this.queue.get(queryId);
    if (!query) {
      return undefined;
    }

    // Remove from main queue
    this.queue.delete(queryId);

    // Remove from priority queue
    const prioritySet = this.priorityQueue.get(query.priority);
    if (prioritySet) {
      prioritySet.delete(queryId);
    }

    // Remove from provider assignments
    if (query.assignedProviderId) {
      const assignments = this.providerAssignments.get(query.assignedProviderId);
      if (assignments) {
        assignments.delete(queryId);
      }
    }

    console.log(`[QUEUE] Removed query ${queryId}`);
    return query;
  }

  /**
   * Get next query based on priority
   */
  getNext(): PatientQuery | undefined {
    // Check priorities in order: emergency, urgent, routine, low
    const priorityOrder = [
      QueryPriority.EMERGENCY,
      QueryPriority.URGENT,
      QueryPriority.ROUTINE,
      QueryPriority.LOW
    ];

    for (const priority of priorityOrder) {
      const prioritySet = this.priorityQueue.get(priority);
      if (prioritySet && prioritySet.size > 0) {
        const queryId = Array.from(prioritySet)[0];
        const query = this.queue.get(queryId);
        if (query && query.status === QueryStatus.PENDING) {
          return query;
        }
      }
    }

    return undefined;
  }

  /**
   * Assign query to provider
   */
  async assignToProvider(queryId: string, providerId: string): Promise<boolean> {
    const query = this.queue.get(queryId);
    if (!query) {
      throw new Error(`Query not found: ${queryId}`);
    }

    if (query.status !== QueryStatus.PENDING) {
      throw new Error(`Query ${queryId} is not in pending status`);
    }

    // Update query
    query.assignedProviderId = providerId;
    query.status = QueryStatus.IN_REVIEW;
    query.updatedAt = new Date();

    // Update provider assignments
    const assignments = this.providerAssignments.get(providerId) || new Set();
    assignments.add(queryId);
    this.providerAssignments.set(providerId, assignments);

    console.log(`[QUEUE] Assigned query ${queryId} to provider ${providerId}`);
    return true;
  }

  /**
   * Auto-assign query to available provider
   */
  private async autoAssignQuery(queryId: string): Promise<boolean> {
    const query = this.queue.get(queryId);
    if (!query) {
      return false;
    }

    // This would integrate with provider service to find available provider
    // For now, just mark as pending
    console.log(`[QUEUE] Query ${queryId} ready for auto-assignment`);
    return true;
  }

  /**
   * Get queries for provider
   */
  getQueriesForProvider(providerId: string): PatientQuery[] {
    const assignments = this.providerAssignments.get(providerId);
    if (!assignments) {
      return [];
    }

    return Array.from(assignments)
      .map(id => this.queue.get(id))
      .filter((q): q is PatientQuery => q !== undefined);
  }

  /**
   * Get queries by status
   */
  getQueriesByStatus(status: QueryStatus): PatientQuery[] {
    return Array.from(this.queue.values())
      .filter(q => q.status === status);
  }

  /**
   * Get queries by priority
   */
  getQueriesByPriority(priority: QueryPriority): PatientQuery[] {
    const prioritySet = this.priorityQueue.get(priority);
    if (!prioritySet) {
      return [];
    }

    return Array.from(prioritySet)
      .map(id => this.queue.get(id))
      .filter((q): q is PatientQuery => q !== undefined);
  }

  /**
   * Update query status
   */
  updateQueryStatus(queryId: string, status: QueryStatus): boolean {
    const query = this.queue.get(queryId);
    if (!query) {
      return false;
    }

    query.status = status;
    query.updatedAt = new Date();

    if (status === QueryStatus.COMPLETED) {
      query.completedAt = new Date();
      // Optionally dequeue completed queries
      this.dequeue(queryId);
    }

    return true;
  }

  /**
   * Get queue statistics
   */
  getStats(): {
    totalQueries: number;
    byStatus: Map<QueryStatus, number>;
    byPriority: Map<QueryPriority, number>;
    averageWaitTime: number;
  } {
    const byStatus = new Map<QueryStatus, number>();
    const byPriority = new Map<QueryPriority, number>();
    let totalWaitTime = 0;

    for (const query of this.queue.values()) {
      // Count by status
      const statusCount = byStatus.get(query.status) || 0;
      byStatus.set(query.status, statusCount + 1);

      // Count by priority
      const priorityCount = byPriority.get(query.priority) || 0;
      byPriority.set(query.priority, priorityCount + 1);

      // Calculate wait time for pending queries
      if (query.status === QueryStatus.PENDING) {
        const waitTime = Date.now() - query.createdAt.getTime();
        totalWaitTime += waitTime;
      }
    }

    const pendingCount = byStatus.get(QueryStatus.PENDING) || 0;
    const averageWaitTime = pendingCount > 0 ? totalWaitTime / pendingCount / 60000 : 0; // in minutes

    return {
      totalQueries: this.queue.size,
      byStatus,
      byPriority,
      averageWaitTime
    };
  }

  /**
   * Check for stale queries and escalate
   */
  async checkStaleQueries(): Promise<string[]> {
    const staleQueries: string[] = [];
    const now = Date.now();
    const staleThreshold = this.config.stalePeriodMinutes * 60 * 1000;

    for (const query of this.queue.values()) {
      if (query.status === QueryStatus.PENDING || query.status === QueryStatus.IN_REVIEW) {
        const age = now - query.updatedAt.getTime();
        if (age > staleThreshold) {
          staleQueries.push(query.id);
          // Escalate priority if not already emergency
          if (query.priority !== QueryPriority.EMERGENCY) {
            query.priority = query.priority === QueryPriority.URGENT
              ? QueryPriority.EMERGENCY
              : QueryPriority.URGENT;
            console.log(`[QUEUE] Escalated stale query ${query.id} to ${query.priority}`);
          }
        }
      }
    }

    return staleQueries;
  }
}
