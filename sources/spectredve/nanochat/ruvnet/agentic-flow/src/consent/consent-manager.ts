/**
 * Consent Manager
 * Manages patient consent lifecycle and validation
 */

import {
  Consent,
  ConsentType,
  ConsentStatus
} from './types';

export class ConsentManager {
  private consents: Map<string, Consent>;
  private patientConsents: Map<string, Set<string>>; // patientId -> consentIds

  constructor() {
    this.consents = new Map();
    this.patientConsents = new Map();
  }

  /**
   * Create new consent
   */
  async createConsent(consent: Omit<Consent, 'id' | 'createdAt' | 'updatedAt'>): Promise<Consent> {
    const newConsent: Consent = {
      ...consent,
      id: `consent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      createdAt: new Date(),
      updatedAt: new Date()
    };

    this.consents.set(newConsent.id, newConsent);

    // Index by patient
    const patientConsentSet = this.patientConsents.get(consent.patientId) || new Set();
    patientConsentSet.add(newConsent.id);
    this.patientConsents.set(consent.patientId, patientConsentSet);

    console.log(`[CONSENT] Created consent ${newConsent.id} for patient ${consent.patientId}`);

    return newConsent;
  }

  /**
   * Grant consent
   */
  async grantConsent(
    consentId: string,
    signature: string,
    witnessSignature?: string
  ): Promise<boolean> {
    const consent = this.consents.get(consentId);
    if (!consent) {
      return false;
    }

    consent.status = ConsentStatus.GRANTED;
    consent.signature = signature;
    consent.witnessSignature = witnessSignature;
    consent.updatedAt = new Date();

    console.log(`[CONSENT] Consent ${consentId} granted`);

    return true;
  }

  /**
   * Revoke consent
   */
  async revokeConsent(consentId: string, revokedBy: string, reason?: string): Promise<boolean> {
    const consent = this.consents.get(consentId);
    if (!consent) {
      return false;
    }

    consent.status = ConsentStatus.REVOKED;
    consent.revokedAt = new Date();
    consent.revokedBy = revokedBy;
    consent.updatedAt = new Date();

    if (reason && consent.metadata) {
      consent.metadata.revocationReason = reason;
    }

    console.log(`[CONSENT] Consent ${consentId} revoked by ${revokedBy}`);

    return true;
  }

  /**
   * Check if consent is valid
   */
  isConsentValid(consentId: string): boolean {
    const consent = this.consents.get(consentId);
    if (!consent) {
      return false;
    }

    // Check status
    if (consent.status !== ConsentStatus.GRANTED) {
      return false;
    }

    // Check expiry
    if (consent.expiryDate && consent.expiryDate < new Date()) {
      // Auto-expire
      consent.status = ConsentStatus.EXPIRED;
      consent.updatedAt = new Date();
      return false;
    }

    return true;
  }

  /**
   * Check if provider has consent to access patient data
   */
  hasConsentFor(
    patientId: string,
    providerId: string,
    type: ConsentType,
    dataCategory?: string
  ): boolean {
    const patientConsentIds = this.patientConsents.get(patientId);
    if (!patientConsentIds) {
      return false;
    }

    for (const consentId of patientConsentIds) {
      const consent = this.consents.get(consentId);
      if (!consent) continue;

      // Check if consent is valid
      if (!this.isConsentValid(consentId)) continue;

      // Check type
      if (consent.type !== type) continue;

      // Check if granted to provider
      if (consent.grantedTo && !consent.grantedTo.includes(providerId)) {
        continue;
      }

      // Check data category if specified
      if (dataCategory && !consent.dataCategories.includes(dataCategory)) {
        continue;
      }

      return true;
    }

    return false;
  }

  /**
   * Get consents for patient
   */
  getConsentsForPatient(patientId: string, filter?: {
    type?: ConsentType;
    status?: ConsentStatus;
  }): Consent[] {
    const patientConsentIds = this.patientConsents.get(patientId);
    if (!patientConsentIds) {
      return [];
    }

    let consents = Array.from(patientConsentIds)
      .map(id => this.consents.get(id))
      .filter((c): c is Consent => c !== undefined);

    // Apply filters
    if (filter?.type) {
      consents = consents.filter(c => c.type === filter.type);
    }

    if (filter?.status) {
      consents = consents.filter(c => c.status === filter.status);
    }

    return consents.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
  }

  /**
   * Get consents granted to provider
   */
  getConsentsForProvider(providerId: string): Consent[] {
    const consents: Consent[] = [];

    for (const consent of this.consents.values()) {
      if (consent.grantedTo && consent.grantedTo.includes(providerId)) {
        if (this.isConsentValid(consent.id)) {
          consents.push(consent);
        }
      }
    }

    return consents;
  }

  /**
   * Check expiring consents
   */
  getExpiringConsents(daysBeforeExpiry: number = 30): Consent[] {
    const now = Date.now();
    const threshold = daysBeforeExpiry * 24 * 60 * 60 * 1000;
    const expiring: Consent[] = [];

    for (const consent of this.consents.values()) {
      if (consent.status === ConsentStatus.GRANTED && consent.expiryDate) {
        const timeToExpiry = consent.expiryDate.getTime() - now;
        if (timeToExpiry > 0 && timeToExpiry <= threshold) {
          expiring.push(consent);
        }
      }
    }

    return expiring;
  }

  /**
   * Renew consent
   */
  async renewConsent(consentId: string, newExpiryDate: Date): Promise<boolean> {
    const consent = this.consents.get(consentId);
    if (!consent) {
      return false;
    }

    consent.expiryDate = newExpiryDate;
    consent.updatedAt = new Date();

    // Reactivate if expired
    if (consent.status === ConsentStatus.EXPIRED) {
      consent.status = ConsentStatus.GRANTED;
    }

    console.log(`[CONSENT] Consent ${consentId} renewed until ${newExpiryDate.toISOString()}`);

    return true;
  }

  /**
   * Get consent statistics
   */
  getStats(): {
    totalConsents: number;
    byStatus: Map<ConsentStatus, number>;
    byType: Map<ConsentType, number>;
    expiringCount: number;
  } {
    const byStatus = new Map<ConsentStatus, number>();
    const byType = new Map<ConsentType, number>();

    for (const consent of this.consents.values()) {
      // Count by status
      const statusCount = byStatus.get(consent.status) || 0;
      byStatus.set(consent.status, statusCount + 1);

      // Count by type
      const typeCount = byType.get(consent.type) || 0;
      byType.set(consent.type, typeCount + 1);
    }

    return {
      totalConsents: this.consents.size,
      byStatus,
      byType,
      expiringCount: this.getExpiringConsents(30).length
    };
  }
}
