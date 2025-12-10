/**
 * Emergency Escalation System
 * Handles emergency alerts and escalation workflows
 */

import {
  EmergencyAlert,
  PatientQuery,
  Provider,
  QueryPriority
} from './types';

export class EmergencyEscalationService {
  private alerts: Map<string, EmergencyAlert>;
  private escalationChains: Map<string, string[]>; // specialization -> providerIds
  private activeAlerts: Set<string>;

  constructor() {
    this.alerts = new Map();
    this.escalationChains = new Map();
    this.activeAlerts = new Set();
  }

  /**
   * Create emergency alert
   */
  async createAlert(
    query: PatientQuery,
    severity: 'critical' | 'high' | 'moderate',
    triggeredBy: string
  ): Promise<EmergencyAlert> {
    const alert: EmergencyAlert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      queryId: query.id,
      patientId: query.patientId,
      severity,
      description: this.generateAlertDescription(query, severity),
      triggeredBy,
      assignedProviders: [],
      escalationChain: [],
      createdAt: new Date()
    };

    this.alerts.set(alert.id, alert);
    this.activeAlerts.add(alert.id);

    console.log(`[EMERGENCY] ${severity.toUpperCase()} alert created: ${alert.id} for query ${query.id}`);

    // Auto-assign to escalation chain
    await this.assignToEscalationChain(alert);

    return alert;
  }

  /**
   * Acknowledge emergency alert
   */
  async acknowledgeAlert(alertId: string, providerId: string): Promise<boolean> {
    const alert = this.alerts.get(alertId);
    if (!alert) {
      return false;
    }

    if (!alert.acknowledgedBy) {
      alert.acknowledgedBy = [];
    }

    if (!alert.acknowledgedBy.includes(providerId)) {
      alert.acknowledgedBy.push(providerId);
      alert.acknowledgedAt = new Date();
      console.log(`[EMERGENCY] Alert ${alertId} acknowledged by provider ${providerId}`);
    }

    return true;
  }

  /**
   * Resolve emergency alert
   */
  async resolveAlert(alertId: string, providerId: string, resolution: string): Promise<boolean> {
    const alert = this.alerts.get(alertId);
    if (!alert) {
      return false;
    }

    alert.resolvedBy = providerId;
    alert.resolvedAt = new Date();
    this.activeAlerts.delete(alertId);

    console.log(`[EMERGENCY] Alert ${alertId} resolved by provider ${providerId}: ${resolution}`);

    return true;
  }

  /**
   * Escalate to next level
   */
  async escalate(alertId: string, reason: string): Promise<boolean> {
    const alert = this.alerts.get(alertId);
    if (!alert) {
      return false;
    }

    // Increase severity
    if (alert.severity === 'moderate') {
      alert.severity = 'high';
    } else if (alert.severity === 'high') {
      alert.severity = 'critical';
    }

    // Add to escalation chain
    alert.escalationChain.push(`Escalated: ${reason} at ${new Date().toISOString()}`);

    // Assign to higher-level providers
    await this.assignToEscalationChain(alert);

    console.log(`[EMERGENCY] Alert ${alertId} escalated to ${alert.severity}: ${reason}`);

    return true;
  }

  /**
   * Configure escalation chain for specialization
   */
  setEscalationChain(specialization: string, providerIds: string[]): void {
    this.escalationChains.set(specialization, providerIds);
    console.log(`[EMERGENCY] Escalation chain configured for ${specialization}: ${providerIds.length} providers`);
  }

  /**
   * Assign alert to escalation chain
   */
  private async assignToEscalationChain(alert: EmergencyAlert): Promise<void> {
    // Get appropriate escalation chain based on query type
    // This would integrate with provider service to find suitable providers
    const providers = this.findProvidersForAlert(alert);

    alert.assignedProviders = providers;
    console.log(`[EMERGENCY] Alert ${alert.id} assigned to ${providers.length} providers`);
  }

  /**
   * Find suitable providers for alert
   */
  private findProvidersForAlert(alert: EmergencyAlert): string[] {
    // This would integrate with provider service to find available providers
    // For now, return empty array
    return [];
  }

  /**
   * Generate alert description
   */
  private generateAlertDescription(query: PatientQuery, severity: string): string {
    const severityPrefix = severity === 'critical' ? '🚨 CRITICAL: ' :
                          severity === 'high' ? '⚠️ HIGH PRIORITY: ' :
                          '⚡ URGENT: ';

    return `${severityPrefix}${query.queryType} - Patient ${query.patientName}. ${query.description}`;
  }

  /**
   * Check for unacknowledged alerts
   */
  async checkUnacknowledgedAlerts(timeoutMinutes: number = 5): Promise<EmergencyAlert[]> {
    const now = Date.now();
    const timeout = timeoutMinutes * 60 * 1000;
    const unacknowledged: EmergencyAlert[] = [];

    for (const alertId of this.activeAlerts) {
      const alert = this.alerts.get(alertId);
      if (!alert) continue;

      const age = now - alert.createdAt.getTime();

      if (!alert.acknowledgedBy || alert.acknowledgedBy.length === 0) {
        if (age > timeout) {
          unacknowledged.push(alert);
          // Auto-escalate
          await this.escalate(alert.id, `No acknowledgment after ${timeoutMinutes} minutes`);
        }
      }
    }

    return unacknowledged;
  }

  /**
   * Get active alerts
   */
  getActiveAlerts(): EmergencyAlert[] {
    return Array.from(this.activeAlerts)
      .map(id => this.alerts.get(id))
      .filter((a): a is EmergencyAlert => a !== undefined)
      .sort((a, b) => {
        // Sort by severity first, then by time
        const severityOrder = { critical: 0, high: 1, moderate: 2 };
        const severityDiff = severityOrder[a.severity] - severityOrder[b.severity];
        if (severityDiff !== 0) return severityDiff;
        return a.createdAt.getTime() - b.createdAt.getTime();
      });
  }

  /**
   * Get alerts for provider
   */
  getAlertsForProvider(providerId: string): EmergencyAlert[] {
    return Array.from(this.alerts.values())
      .filter(alert =>
        alert.assignedProviders.includes(providerId) &&
        this.activeAlerts.has(alert.id)
      );
  }

  /**
   * Get alert statistics
   */
  getStats(): {
    totalAlerts: number;
    activeAlerts: number;
    bySeverity: Map<string, number>;
    averageResponseTime: number;
    averageResolutionTime: number;
  } {
    const bySeverity = new Map<string, number>();
    let totalResponseTime = 0;
    let totalResolutionTime = 0;
    let acknowledgedCount = 0;
    let resolvedCount = 0;

    for (const alert of this.alerts.values()) {
      // Count by severity
      const count = bySeverity.get(alert.severity) || 0;
      bySeverity.set(alert.severity, count + 1);

      // Calculate response time
      if (alert.acknowledgedAt) {
        const responseTime = alert.acknowledgedAt.getTime() - alert.createdAt.getTime();
        totalResponseTime += responseTime;
        acknowledgedCount++;
      }

      // Calculate resolution time
      if (alert.resolvedAt) {
        const resolutionTime = alert.resolvedAt.getTime() - alert.createdAt.getTime();
        totalResolutionTime += resolutionTime;
        resolvedCount++;
      }
    }

    return {
      totalAlerts: this.alerts.size,
      activeAlerts: this.activeAlerts.size,
      bySeverity,
      averageResponseTime: acknowledgedCount > 0 ? totalResponseTime / acknowledgedCount / 60000 : 0, // minutes
      averageResolutionTime: resolvedCount > 0 ? totalResolutionTime / resolvedCount / 60000 : 0 // minutes
    };
  }
}
