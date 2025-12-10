/**
 * HIPAA Compliance Helpers
 * Utilities for HIPAA compliance tracking and enforcement
 */

import {
  HIPAACompliance,
  AccessLogEntry
} from './types';

export class HIPAAComplianceService {
  private complianceRecords: Map<string, HIPAACompliance>;
  private accessLogs: Map<string, AccessLogEntry[]>; // patientId -> logs

  constructor() {
    this.complianceRecords = new Map();
    this.accessLogs = new Map();
  }

  /**
   * Initialize HIPAA compliance for patient
   */
  async initializeCompliance(patientId: string): Promise<HIPAACompliance> {
    const compliance: HIPAACompliance = {
      patientId,
      consentDocuments: [],
      privacyPracticesAcknowledged: false,
      breachNotificationMethod: 'email',
      authorizedRepresentatives: [],
      accessLog: [],
      lastAudit: new Date(),
      complianceStatus: 'pending'
    };

    this.complianceRecords.set(patientId, compliance);
    this.accessLogs.set(patientId, []);

    console.log(`[HIPAA] Initialized compliance record for patient ${patientId}`);

    return compliance;
  }

  /**
   * Log data access
   */
  async logAccess(entry: AccessLogEntry): Promise<void> {
    const logs = this.accessLogs.get(entry.patientId) || [];
    logs.push(entry);
    this.accessLogs.set(entry.patientId, logs);

    // Update compliance record
    const compliance = this.complianceRecords.get(entry.patientId);
    if (compliance) {
      compliance.accessLog.push(entry);
    }

    // Log unauthorized access attempts
    if (!entry.authorized) {
      console.warn(`[HIPAA] Unauthorized access attempt: ${entry.userId} -> Patient ${entry.patientId}`);
      console.warn(`[HIPAA] Denial reason: ${entry.denialReason}`);
    }
  }

  /**
   * Verify access authorization
   */
  async verifyAccess(
    userId: string,
    userType: 'provider' | 'admin' | 'patient',
    patientId: string,
    action: string,
    resource: string
  ): Promise<{ authorized: boolean; reason?: string }> {
    const compliance = this.complianceRecords.get(patientId);

    if (!compliance) {
      return {
        authorized: false,
        reason: 'No compliance record found'
      };
    }

    // Check if privacy practices are acknowledged
    if (!compliance.privacyPracticesAcknowledged) {
      return {
        authorized: false,
        reason: 'Privacy practices not acknowledged'
      };
    }

    // Check compliance status
    if (compliance.complianceStatus === 'non_compliant') {
      return {
        authorized: false,
        reason: 'Patient record is non-compliant'
      };
    }

    // Patient always has access to their own data
    if (userType === 'patient' && userId === patientId) {
      return { authorized: true };
    }

    // Check authorized representatives
    if (userType === 'patient' && compliance.authorizedRepresentatives.includes(userId)) {
      return { authorized: true };
    }

    // For providers and admins, additional checks would be performed
    // This would integrate with consent and authorization services

    return { authorized: true };
  }

  /**
   * Acknowledge privacy practices
   */
  async acknowledgePrivacyPractices(patientId: string, documentId: string): Promise<boolean> {
    const compliance = this.complianceRecords.get(patientId);
    if (!compliance) {
      return false;
    }

    compliance.privacyPracticesAcknowledged = true;
    compliance.consentDocuments.push(documentId);

    console.log(`[HIPAA] Privacy practices acknowledged for patient ${patientId}`);

    return true;
  }

  /**
   * Add authorized representative
   */
  async addAuthorizedRepresentative(patientId: string, representativeId: string): Promise<boolean> {
    const compliance = this.complianceRecords.get(patientId);
    if (!compliance) {
      return false;
    }

    if (!compliance.authorizedRepresentatives.includes(representativeId)) {
      compliance.authorizedRepresentatives.push(representativeId);
      console.log(`[HIPAA] Added authorized representative ${representativeId} for patient ${patientId}`);
    }

    return true;
  }

  /**
   * Update breach notification method
   */
  async updateBreachNotificationMethod(
    patientId: string,
    method: 'email' | 'sms' | 'mail'
  ): Promise<boolean> {
    const compliance = this.complianceRecords.get(patientId);
    if (!compliance) {
      return false;
    }

    compliance.breachNotificationMethod = method;
    console.log(`[HIPAA] Updated breach notification method for patient ${patientId} to ${method}`);

    return true;
  }

  /**
   * Perform compliance audit
   */
  async performAudit(patientId: string): Promise<{
    compliant: boolean;
    issues: string[];
    recommendations: string[];
  }> {
    const compliance = this.complianceRecords.get(patientId);
    if (!compliance) {
      return {
        compliant: false,
        issues: ['No compliance record found'],
        recommendations: ['Initialize compliance record']
      };
    }

    const issues: string[] = [];
    const recommendations: string[] = [];

    // Check privacy practices acknowledgment
    if (!compliance.privacyPracticesAcknowledged) {
      issues.push('Privacy practices not acknowledged');
      recommendations.push('Obtain privacy practices acknowledgment');
    }

    // Check for consent documents
    if (compliance.consentDocuments.length === 0) {
      issues.push('No consent documents on file');
      recommendations.push('Obtain required consent documents');
    }

    // Check audit recency
    const daysSinceAudit = (Date.now() - compliance.lastAudit.getTime()) / (24 * 60 * 60 * 1000);
    if (daysSinceAudit > 365) {
      issues.push('Audit overdue (>365 days)');
      recommendations.push('Perform annual compliance audit');
    }

    // Check access log
    const logs = this.accessLogs.get(patientId) || [];
    const unauthorizedAccess = logs.filter(log => !log.authorized);
    if (unauthorizedAccess.length > 0) {
      issues.push(`${unauthorizedAccess.length} unauthorized access attempts`);
      recommendations.push('Review and address unauthorized access attempts');
    }

    // Update compliance status
    compliance.compliant = issues.length === 0;
    compliance.lastAudit = new Date();
    compliance.complianceStatus = issues.length === 0 ? 'compliant' : 'non_compliant';

    console.log(`[HIPAA] Audit completed for patient ${patientId}: ${compliance.complianceStatus}`);

    return {
      compliant: issues.length === 0,
      issues,
      recommendations
    };
  }

  /**
   * Get access logs for patient
   */
  getAccessLogs(patientId: string, filter?: {
    startDate?: Date;
    endDate?: Date;
    userType?: string;
    authorized?: boolean;
  }): AccessLogEntry[] {
    let logs = this.accessLogs.get(patientId) || [];

    if (filter?.startDate) {
      logs = logs.filter(log => log.timestamp >= filter.startDate!);
    }

    if (filter?.endDate) {
      logs = logs.filter(log => log.timestamp <= filter.endDate!);
    }

    if (filter?.userType) {
      logs = logs.filter(log => log.userType === filter.userType);
    }

    if (filter?.authorized !== undefined) {
      logs = logs.filter(log => log.authorized === filter.authorized);
    }

    return logs.sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }

  /**
   * Generate compliance report
   */
  generateReport(patientId: string): {
    patientId: string;
    complianceStatus: string;
    lastAudit: Date;
    totalAccesses: number;
    unauthorizedAccesses: number;
    uniqueAccessors: number;
    privacyPracticesAcknowledged: boolean;
    consentDocumentsCount: number;
  } {
    const compliance = this.complianceRecords.get(patientId);
    const logs = this.accessLogs.get(patientId) || [];

    const unauthorizedAccesses = logs.filter(log => !log.authorized).length;
    const uniqueAccessors = new Set(logs.map(log => log.userId)).size;

    return {
      patientId,
      complianceStatus: compliance?.complianceStatus || 'unknown',
      lastAudit: compliance?.lastAudit || new Date(0),
      totalAccesses: logs.length,
      unauthorizedAccesses,
      uniqueAccessors,
      privacyPracticesAcknowledged: compliance?.privacyPracticesAcknowledged || false,
      consentDocumentsCount: compliance?.consentDocuments.length || 0
    };
  }

  /**
   * Get statistics
   */
  getStats(): {
    totalPatients: number;
    compliantPatients: number;
    pendingPatients: number;
    nonCompliantPatients: number;
    totalAccessLogs: number;
    unauthorizedAttempts: number;
  } {
    let compliant = 0;
    let pending = 0;
    let nonCompliant = 0;
    let totalAccessLogs = 0;
    let unauthorizedAttempts = 0;

    for (const compliance of this.complianceRecords.values()) {
      switch (compliance.complianceStatus) {
        case 'compliant':
          compliant++;
          break;
        case 'pending':
          pending++;
          break;
        case 'non_compliant':
          nonCompliant++;
          break;
      }
    }

    for (const logs of this.accessLogs.values()) {
      totalAccessLogs += logs.length;
      unauthorizedAttempts += logs.filter(log => !log.authorized).length;
    }

    return {
      totalPatients: this.complianceRecords.size,
      compliantPatients: compliant,
      pendingPatients: pending,
      nonCompliantPatients: nonCompliant,
      totalAccessLogs,
      unauthorizedAttempts
    };
  }
}
