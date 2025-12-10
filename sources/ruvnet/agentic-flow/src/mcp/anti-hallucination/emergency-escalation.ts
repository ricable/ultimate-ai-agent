/**
 * Emergency Escalation
 * Handles urgent medical situations requiring immediate attention
 */

import type { MedicalAnalysis, EmergencyEscalation } from '../types';

export class EmergencyEscalationHandler {
  private readonly escalations: Map<string, EmergencyEscalation>;
  private readonly emergencyContacts: string[];
  private readonly escalationThreshold: number;

  constructor() {
    this.escalations = new Map();
    this.emergencyContacts = [
      'emergency@medical.example.com',
      '+1-555-EMERGENCY',
      'on-call-provider@medical.example.com',
    ];
    this.escalationThreshold = 0.9; // Confidence threshold for automatic escalation
  }

  /**
   * Evaluate if analysis requires emergency escalation
   */
  evaluateForEscalation(analysis: MedicalAnalysis): boolean {
    const triggers: string[] = [];

    // Check for emergency keywords in conditions
    const emergencyKeywords = [
      'cardiac arrest',
      'stroke',
      'sepsis',
      'anaphylaxis',
      'severe bleeding',
      'respiratory failure',
      'acute abdomen',
      'trauma',
    ];

    for (const condition of analysis.conditions) {
      if (emergencyKeywords.some(keyword =>
        condition.name.toLowerCase().includes(keyword.toLowerCase())
      )) {
        triggers.push(`Emergency condition detected: ${condition.name}`);
      }
    }

    // Check urgency level
    if (analysis.urgencyLevel === 'emergency') {
      triggers.push('Emergency urgency level');
    }

    // Check for critical severity
    if (analysis.conditions.some(c => c.severity === 'critical')) {
      triggers.push('Critical severity condition');
    }

    // Check for red flag symptoms
    const redFlagSymptoms = [
      'chest pain',
      'difficulty breathing',
      'severe headache',
      'loss of consciousness',
      'severe abdominal pain',
      'high fever',
      'seizure',
    ];

    for (const condition of analysis.conditions) {
      for (const symptom of condition.symptoms) {
        if (redFlagSymptoms.some(flag =>
          symptom.toLowerCase().includes(flag.toLowerCase())
        )) {
          triggers.push(`Red flag symptom: ${symptom}`);
        }
      }
    }

    // Escalate if any triggers found
    if (triggers.length > 0) {
      this.triggerEscalation(analysis, triggers);
      return true;
    }

    return false;
  }

  /**
   * Trigger emergency escalation
   */
  private triggerEscalation(analysis: MedicalAnalysis, triggers: string[]): void {
    const escalation: EmergencyEscalation = {
      id: `esc-${analysis.id}`,
      timestamp: Date.now(),
      analysisId: analysis.id,
      trigger: triggers.join('; '),
      severity: analysis.conditions.some(c => c.severity === 'critical') ? 'critical' : 'high',
      actions: this.determineActions(analysis),
      notifiedParties: [],
      responseTime: 0,
      resolved: false,
    };

    // Store escalation
    this.escalations.set(escalation.id, escalation);

    // Notify emergency contacts
    this.notifyEmergencyContacts(escalation, analysis);

    // Log escalation
    console.error('🚨 EMERGENCY ESCALATION:', {
      id: escalation.id,
      triggers,
      severity: escalation.severity,
      timestamp: new Date(escalation.timestamp).toISOString(),
    });
  }

  /**
   * Determine required actions
   */
  private determineActions(analysis: MedicalAnalysis): string[] {
    const actions: string[] = [];

    // Based on urgency
    if (analysis.urgencyLevel === 'emergency') {
      actions.push('Call 911 immediately');
      actions.push('Initiate emergency protocols');
      actions.push('Alert on-call physician');
    }

    // Based on conditions
    for (const condition of analysis.conditions) {
      if (condition.name.toLowerCase().includes('cardiac')) {
        actions.push('Prepare for cardiac intervention');
        actions.push('Ensure defibrillator availability');
      }

      if (condition.name.toLowerCase().includes('stroke')) {
        actions.push('Activate stroke protocol');
        actions.push('Prepare for immediate imaging');
      }

      if (condition.severity === 'critical') {
        actions.push(`Critical condition management for: ${condition.name}`);
      }
    }

    // Default actions
    if (actions.length === 0) {
      actions.push('Contact emergency services');
      actions.push('Monitor patient continuously');
      actions.push('Document all observations');
    }

    return actions;
  }

  /**
   * Notify emergency contacts
   */
  private notifyEmergencyContacts(
    escalation: EmergencyEscalation,
    analysis: MedicalAnalysis
  ): void {
    const message = this.formatEmergencyMessage(escalation, analysis);

    for (const contact of this.emergencyContacts) {
      // In production, this would send actual notifications via:
      // - SMS/Text
      // - Phone call
      // - Pager
      // - Email
      // - Mobile app push notification

      console.error(`📞 Notifying ${contact}:`, message);
      escalation.notifiedParties.push(contact);
    }

    // Record notification time
    escalation.responseTime = Date.now() - escalation.timestamp;
  }

  /**
   * Format emergency message
   */
  private formatEmergencyMessage(
    escalation: EmergencyEscalation,
    analysis: MedicalAnalysis
  ): string {
    const conditions = analysis.conditions.map(c => c.name).join(', ');

    return `
🚨 EMERGENCY ALERT 🚨

Escalation ID: ${escalation.id}
Severity: ${escalation.severity.toUpperCase()}
Time: ${new Date(escalation.timestamp).toISOString()}

Trigger: ${escalation.trigger}
Conditions: ${conditions}
Urgency: ${analysis.urgencyLevel}

Required Actions:
${escalation.actions.map((a, i) => `${i + 1}. ${a}`).join('\n')}

Analysis ID: ${analysis.id}
Confidence: ${(analysis.confidence * 100).toFixed(1)}%

IMMEDIATE RESPONSE REQUIRED
    `.trim();
  }

  /**
   * Resolve escalation
   */
  resolveEscalation(escalationId: string, resolution: string): boolean {
    const escalation = this.escalations.get(escalationId);
    if (!escalation) {
      return false;
    }

    escalation.resolved = true;
    escalation.responseTime = Date.now() - escalation.timestamp;

    console.error(`✅ Escalation resolved: ${escalationId}`, {
      resolution,
      responseTime: `${escalation.responseTime}ms`,
    });

    return true;
  }

  /**
   * Get escalation details
   */
  getEscalation(escalationId: string): EmergencyEscalation | undefined {
    return this.escalations.get(escalationId);
  }

  /**
   * Get all active escalations
   */
  getActiveEscalations(): EmergencyEscalation[] {
    return Array.from(this.escalations.values()).filter(e => !e.resolved);
  }

  /**
   * Get escalation history
   */
  getEscalationHistory(): EmergencyEscalation[] {
    return Array.from(this.escalations.values());
  }
}
