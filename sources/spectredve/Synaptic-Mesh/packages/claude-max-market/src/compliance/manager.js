/**
 * Compliance Manager
 * 
 * Central compliance management and monitoring for Claude Max market
 */

import { ComplianceChecker } from './checker.js';
import { LegalNotice } from '../legal/notice.js';

export class ComplianceManager {
  constructor() {
    this.checker = new ComplianceChecker();
    this.legal = new LegalNotice();
    this.complianceStatus = {
      verified: false,
      lastCheck: null,
      score: 0,
      issues: []
    };
  }

  /**
   * Verify system compliance
   */
  async verifyCompliance() {
    try {
      const isCompliant = await this.checker.checkCompliance();
      const report = await this.checker.generateComplianceReport();
      
      this.complianceStatus = {
        verified: isCompliant,
        lastCheck: new Date().toISOString(),
        score: report.summary.score,
        issues: report.issues,
        criticalFailures: report.summary.criticalFailures
      };
      
      return isCompliant;
    } catch (error) {
      console.error('Compliance verification failed:', error.message);
      return false;
    }
  }

  /**
   * Get current compliance status
   */
  async getComplianceStatus() {
    return {
      anthropicTos: this.complianceStatus.verified,
      noSharedKeys: true, // Always true by design
      peerOrchestrated: true, // Always true by design
      voluntaryParticipation: this.complianceStatus.verified,
      userControl: this.complianceStatus.verified,
      score: this.complianceStatus.score,
      lastVerified: this.complianceStatus.lastCheck
    };
  }

  /**
   * Quick compliance check
   */
  async quickCheck() {
    return await this.checker.quickCheck();
  }

  /**
   * Generate comprehensive audit report
   */
  async generateAuditReport() {
    const report = await this.checker.generateComplianceReport();
    const legal = this.legal.generateComplianceReport();
    
    return {
      ...report,
      legal,
      recommendations: [
        ...report.recommendations,
        'Regular compliance monitoring',
        'User education on terms and controls',
        'Periodic legal review of terms'
      ]
    };
  }
}