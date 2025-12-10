/**
 * Type definitions for provider dashboard system
 */

export enum ProviderType {
  PHYSICIAN = 'physician',
  NURSE = 'nurse',
  SPECIALIST = 'specialist',
  ADMINISTRATOR = 'administrator'
}

export enum ProviderStatus {
  AVAILABLE = 'available',
  BUSY = 'busy',
  OFFLINE = 'offline',
  ON_CALL = 'on_call'
}

export enum QueryStatus {
  PENDING = 'pending',
  IN_REVIEW = 'in_review',
  APPROVED = 'approved',
  REJECTED = 'rejected',
  ESCALATED = 'escalated',
  COMPLETED = 'completed'
}

export enum QueryPriority {
  EMERGENCY = 'emergency',
  URGENT = 'urgent',
  ROUTINE = 'routine',
  LOW = 'low'
}

export interface Provider {
  id: string;
  name: string;
  type: ProviderType;
  specialization?: string[];
  credentials: string[];
  status: ProviderStatus;
  email: string;
  phone: string;
  maxConcurrentCases: number;
  currentCaseLoad: number;
  metadata?: Record<string, any>;
}

export interface PatientQuery {
  id: string;
  patientId: string;
  patientName: string;
  queryType: string;
  priority: QueryPriority;
  status: QueryStatus;
  description: string;
  symptoms?: string[];
  medicalHistory?: string;
  attachments?: string[];
  assignedProviderId?: string;
  createdAt: Date;
  updatedAt: Date;
  reviewedAt?: Date;
  completedAt?: Date;
  requiresConsent: boolean;
  consentGranted?: boolean;
  metadata?: Record<string, any>;
}

export interface QueryReview {
  id: string;
  queryId: string;
  providerId: string;
  action: 'approve' | 'reject' | 'escalate' | 'request_info';
  notes: string;
  diagnosis?: string;
  recommendations?: string[];
  prescriptions?: Prescription[];
  referrals?: Referral[];
  followUpRequired: boolean;
  followUpDate?: Date;
  reviewedAt: Date;
}

export interface Prescription {
  medication: string;
  dosage: string;
  frequency: string;
  duration: string;
  instructions: string;
}

export interface Referral {
  specialization: string;
  reason: string;
  urgency: 'immediate' | 'urgent' | 'routine';
  notes?: string;
}

export interface ProviderCommunication {
  id: string;
  from: string; // providerId or patientId
  to: string; // providerId or patientId
  queryId: string;
  message: string;
  attachments?: string[];
  encrypted: boolean;
  sentAt: Date;
  readAt?: Date;
}

export interface EmergencyAlert {
  id: string;
  queryId: string;
  patientId: string;
  severity: 'critical' | 'high' | 'moderate';
  description: string;
  triggeredBy: string;
  assignedProviders: string[];
  acknowledgedBy?: string[];
  resolvedBy?: string;
  createdAt: Date;
  acknowledgedAt?: Date;
  resolvedAt?: Date;
  escalationChain: string[];
}

export interface QueueConfig {
  autoAssignment: boolean;
  priorityWeighting: Map<QueryPriority, number>;
  maxQueueSize: number;
  stalePeriodMinutes: number;
}

export interface ProviderMetrics {
  providerId: string;
  period: {
    start: Date;
    end: Date;
  };
  queriesReviewed: number;
  averageReviewTime: number; // minutes
  approvalRate: number; // percentage
  patientSatisfaction?: number; // rating
  responseTime: number; // minutes
}
