/**
 * Type definitions for routing system
 */

export enum SeverityLevel {
  CRITICAL = 'critical',
  HIGH = 'high',
  MODERATE = 'moderate',
  LOW = 'low'
}

export enum EmergencyType {
  LIFE_THREATENING = 'life_threatening',
  URGENT_CARE = 'urgent_care',
  ROUTINE = 'routine'
}

export interface RoutingDecision {
  queryId: string;
  severity: SeverityLevel;
  emergencyType: EmergencyType;
  recommendedProviderId?: string;
  alternativeProviders: string[];
  escalationRequired: boolean;
  estimatedResponseTime: number; // minutes
  reasoning: string;
  confidence: number; // 0-1
}

export interface EmergencySignal {
  keyword: string;
  weight: number;
  category: 'symptom' | 'condition' | 'urgency';
}

export interface SeverityScore {
  totalScore: number;
  components: {
    symptomSeverity: number;
    urgency: number;
    riskFactors: number;
    patientHistory: number;
  };
  level: SeverityLevel;
}

export interface ProviderMatch {
  providerId: string;
  matchScore: number;
  availability: boolean;
  estimatedWaitTime: number;
  specializations: string[];
  currentLoad: number;
  performanceScore: number;
}
