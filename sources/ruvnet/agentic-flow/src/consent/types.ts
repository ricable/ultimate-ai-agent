/**
 * Type definitions for consent management
 */

export enum ConsentType {
  TREATMENT = 'treatment',
  DATA_SHARING = 'data_sharing',
  RESEARCH = 'research',
  MARKETING = 'marketing',
  THIRD_PARTY = 'third_party'
}

export enum ConsentStatus {
  PENDING = 'pending',
  GRANTED = 'granted',
  DENIED = 'denied',
  REVOKED = 'revoked',
  EXPIRED = 'expired'
}

export enum DataAccessLevel {
  FULL = 'full',
  LIMITED = 'limited',
  READ_ONLY = 'read_only',
  NONE = 'none'
}

export interface Consent {
  id: string;
  patientId: string;
  type: ConsentType;
  status: ConsentStatus;
  grantedTo?: string[]; // providerIds or organizationIds
  purpose: string;
  dataCategories: string[]; // e.g., 'medical_history', 'test_results'
  startDate: Date;
  expiryDate?: Date;
  revokedAt?: Date;
  revokedBy?: string;
  createdAt: Date;
  updatedAt: Date;
  signature?: string; // Digital signature
  witnessSignature?: string;
  metadata?: Record<string, any>;
}

export interface Authorization {
  id: string;
  providerId: string;
  patientId: string;
  accessLevel: DataAccessLevel;
  allowedActions: string[]; // e.g., 'read', 'write', 'prescribe'
  dataScopes: string[]; // Specific data types authorized
  validFrom: Date;
  validUntil?: Date;
  grantedBy: string; // patientId or authorized representative
  purpose: string;
  active: boolean;
  auditTrail: AuthorizationEvent[];
}

export interface AuthorizationEvent {
  timestamp: Date;
  action: 'granted' | 'revoked' | 'accessed' | 'modified';
  performedBy: string;
  reason?: string;
  metadata?: Record<string, any>;
}

export interface DataSharingPolicy {
  id: string;
  patientId: string;
  allowedProviders: string[];
  allowedOrganizations: string[];
  dataCategories: string[];
  restrictions: DataRestriction[];
  autoApprove: boolean;
  requiresNotification: boolean;
  active: boolean;
}

export interface DataRestriction {
  type: 'time_based' | 'location_based' | 'purpose_based';
  rules: Record<string, any>;
  description: string;
}

export interface HIPAACompliance {
  patientId: string;
  consentDocuments: string[]; // Document IDs
  privacyPracticesAcknowledged: boolean;
  breachNotificationMethod: 'email' | 'sms' | 'mail';
  authorizedRepresentatives: string[];
  accessLog: AccessLogEntry[];
  lastAudit: Date;
  complianceStatus: 'compliant' | 'pending' | 'non_compliant';
}

export interface AccessLogEntry {
  timestamp: Date;
  userId: string; // providerId
  userType: 'provider' | 'admin' | 'patient';
  action: string;
  resource: string;
  patientId: string;
  authorized: boolean;
  denialReason?: string;
  ipAddress?: string;
  metadata?: Record<string, any>;
}
