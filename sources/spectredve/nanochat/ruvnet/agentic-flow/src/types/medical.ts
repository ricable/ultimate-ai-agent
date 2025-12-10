// Medical Analysis System Types

export interface PatientData {
  id: string;
  age: number;
  gender: 'male' | 'female' | 'other';
  symptoms: string[];
  medicalHistory: string[];
  medications: string[];
  vitalSigns?: VitalSigns;
  labResults?: LabResult[];
}

export interface VitalSigns {
  bloodPressure: { systolic: number; diastolic: number };
  heartRate: number;
  temperature: number;
  respiratoryRate: number;
  oxygenSaturation: number;
}

export interface LabResult {
  testName: string;
  value: number;
  unit: string;
  referenceRange: { min: number; max: number };
  timestamp: string;
}

export interface MedicalAnalysis {
  id: string;
  patientId: string;
  analysis: string;
  diagnosis: string[];
  confidence: number;
  citations: Citation[];
  recommendations: string[];
  riskFactors: RiskFactor[];
  verificationScore: number;
  timestamp: string;
  metadata: AnalysisMetadata;
}

export interface Citation {
  source: string;
  reference: string;
  relevance: number;
  verified: boolean;
}

export interface RiskFactor {
  factor: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
}

export interface AnalysisMetadata {
  modelUsed: string;
  processingTime: number;
  hallucinationChecks: HallucinationCheck[];
  knowledgeBaseCrossChecks: number;
  agentDBLearningApplied: boolean;
}

export interface HallucinationCheck {
  type: 'factual' | 'statistical' | 'logical' | 'medical-guideline';
  passed: boolean;
  confidence: number;
  details: string;
}

export interface ProviderNotification {
  id: string;
  analysisId: string;
  providerId: string;
  channel: 'email' | 'sms' | 'push' | 'in-app';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  message: string;
  sentAt?: string;
  deliveredAt?: string;
  readAt?: string;
  status: 'pending' | 'sent' | 'delivered' | 'read' | 'failed';
}

export interface VerificationResult {
  passed: boolean;
  score: number;
  checks: {
    medicalAccuracy: boolean;
    citationValidity: boolean;
    logicalConsistency: boolean;
    guidelineCompliance: boolean;
    hallucinationFree: boolean;
  };
  issues: VerificationIssue[];
}

export interface VerificationIssue {
  type: string;
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  suggestedFix?: string;
}

export interface MCPToolRequest {
  tool: string;
  action: string;
  params: Record<string, any>;
}

export interface MCPToolResponse {
  success: boolean;
  data?: any;
  error?: string;
  metadata?: Record<string, any>;
}
