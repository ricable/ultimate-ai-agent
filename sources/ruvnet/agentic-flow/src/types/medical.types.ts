/**
 * Medical Analysis System - Type Definitions
 * Comprehensive type safety for medical AI analysis
 */

// ============= Core Analysis Types =============

export interface MedicalCondition {
  name: string;
  icd10Code?: string;
  severity: 'mild' | 'moderate' | 'severe' | 'critical';
  symptoms: string[];
  relatedConditions?: string[];
}

export interface MedicalQuery {
  id: string;
  patientId?: string;
  condition: string;
  symptoms: string[];
  medicalHistory?: string[];
  medications?: string[];
  timestamp: Date;
  priority: 'low' | 'medium' | 'high' | 'urgent';
}

export interface AnalysisResult {
  id: string;
  queryId: string;
  diagnosis: Diagnosis[];
  recommendations: Recommendation[];
  confidenceScore: ConfidenceScore;
  citations: Citation[];
  warnings: Warning[];
  requiresProviderReview: boolean;
  reviewedBy?: ProviderReview;
  timestamp: Date;
  status: 'pending' | 'analyzing' | 'completed' | 'provider_review' | 'approved' | 'rejected';
}

export interface Diagnosis {
  condition: string;
  icd10Code: string;
  probability: number;
  confidence: number;
  reasoning: string;
  differentialDiagnoses: DifferentialDiagnosis[];
  supportingEvidence: Evidence[];
  contradictions?: Contradiction[];
}

export interface DifferentialDiagnosis {
  condition: string;
  probability: number;
  reasoning: string;
  distinguishingFeatures: string[];
}

export interface Recommendation {
  type: 'treatment' | 'diagnostic_test' | 'lifestyle' | 'follow_up' | 'emergency';
  description: string;
  priority: 'low' | 'medium' | 'high' | 'urgent';
  rationale: string;
  citations: Citation[];
  confidence: number;
  alternatives?: string[];
}

export interface Evidence {
  type: 'symptom' | 'lab_result' | 'imaging' | 'history' | 'literature';
  description: string;
  source?: string;
  reliability: number;
  citation?: Citation;
}

export interface Contradiction {
  statement1: string;
  statement2: string;
  severity: 'minor' | 'moderate' | 'major';
  resolution?: string;
  requiresProviderReview: boolean;
}

// ============= Anti-Hallucination Types =============

export interface ConfidenceScore {
  overall: number; // 0-1
  breakdown: {
    diagnosisConfidence: number;
    citationVerification: number;
    knowledgeBaseValidation: number;
    contradictionCheck: number;
    providerAlignment: number;
  };
  factors: ConfidenceFactor[];
  thresholds: {
    requiresProviderReview: number;
    autoApprove: number;
    flagForReview: number;
  };
  explanation: string;
}

export interface ConfidenceFactor {
  name: string;
  impact: 'positive' | 'negative' | 'neutral';
  weight: number;
  description: string;
}

export interface Citation {
  id: string;
  source: string;
  title: string;
  authors?: string[];
  year?: number;
  doi?: string;
  pubmedId?: string;
  url?: string;
  relevanceScore: number;
  verified: boolean;
  verificationMethod?: string;
  excerpt?: string;
}

export interface Warning {
  type: 'hallucination' | 'low_confidence' | 'contradiction' | 'outdated_info' | 'missing_data' | 'emergency';
  severity: 'info' | 'warning' | 'error' | 'critical';
  message: string;
  details?: string;
  suggestedAction?: string;
}

// ============= Knowledge Base Types =============

export interface MedicalKnowledgeBase {
  id: string;
  condition: string;
  symptoms: KnowledgeSymptom[];
  treatments: KnowledgeTreatment[];
  diagnosticCriteria: DiagnosticCriteria[];
  contraindications: Contraindication[];
  references: Citation[];
  lastUpdated: Date;
  source: string;
  reliability: number;
}

export interface KnowledgeSymptom {
  name: string;
  prevalence: number; // 0-1
  specificity: number; // 0-1
  sensitivity: number; // 0-1
  differentialSymptoms?: string[];
}

export interface KnowledgeTreatment {
  name: string;
  type: 'medication' | 'procedure' | 'therapy' | 'lifestyle';
  efficacy: number;
  evidenceLevel: 'A' | 'B' | 'C' | 'D';
  sideEffects?: string[];
  contraindications?: string[];
}

export interface DiagnosticCriteria {
  name: string;
  required: string[];
  optional: string[];
  exclusion: string[];
  minimumRequired: number;
}

export interface Contraindication {
  medication: string;
  condition?: string;
  severity: 'absolute' | 'relative';
  reasoning: string;
}

// ============= Provider Types =============

export interface Provider {
  id: string;
  name: string;
  specialty: string;
  credentials: string[];
  licenseNumber: string;
  email: string;
  phone?: string;
  notificationPreferences: NotificationPreferences;
}

export interface ProviderReview {
  providerId: string;
  providerName: string;
  timestamp: Date;
  decision: 'approved' | 'rejected' | 'modified';
  comments?: string;
  modifications?: string[];
  signature?: string;
}

export interface NotificationPreferences {
  email: boolean;
  sms: boolean;
  push: boolean;
  urgentOnly: boolean;
  minimumSeverity: 'low' | 'medium' | 'high' | 'urgent';
}

// ============= API Types =============

export interface AnalysisRequest {
  condition?: string;
  symptoms: string[];
  patientContext?: PatientContext;
  options?: AnalysisOptions;
}

export interface PatientContext {
  age?: number;
  gender?: 'male' | 'female' | 'other';
  medicalHistory?: string[];
  currentMedications?: string[];
  allergies?: string[];
  familyHistory?: string[];
}

export interface AnalysisOptions {
  includeDifferentials?: boolean;
  maxDifferentials?: number;
  requireHighConfidence?: boolean;
  includeEmergencyCheck?: boolean;
  language?: string;
}

export interface AnalysisResponse {
  success: boolean;
  data?: AnalysisResult;
  error?: ApiError;
  metadata: ResponseMetadata;
}

export interface ApiError {
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
}

export interface ResponseMetadata {
  requestId: string;
  timestamp: Date;
  processingTimeMs: number;
  version: string;
}

// ============= WebSocket Types =============

export interface WebSocketMessage {
  type: 'analysis_update' | 'confidence_update' | 'provider_notification' | 'warning' | 'error';
  payload: any;
  timestamp: Date;
}

export interface AnalysisUpdate {
  analysisId: string;
  status: AnalysisResult['status'];
  progress: number; // 0-100
  currentStep: string;
  estimatedTimeRemaining?: number;
}

// ============= Learning & Pattern Recognition Types =============

export interface LearningPattern {
  id: string;
  patternType: 'diagnosis' | 'treatment' | 'symptom_cluster' | 'provider_decision';
  frequency: number;
  accuracy: number;
  examples: string[];
  metadata: Record<string, any>;
  createdAt: Date;
  lastUsed: Date;
}

export interface PatternRecognitionResult {
  patterns: LearningPattern[];
  confidence: number;
  applicableToQuery: boolean;
  reasoning: string;
}

// ============= Configuration Types =============

export interface SystemConfig {
  antiHallucination: AntiHallucinationConfig;
  providers: ProviderConfig;
  api: ApiConfig;
  learning: LearningConfig;
}

export interface AntiHallucinationConfig {
  minimumConfidence: number;
  requireProviderReviewThreshold: number;
  autoApproveThreshold: number;
  enableContradictionDetection: boolean;
  enableCitationVerification: boolean;
  knowledgeBaseSources: string[];
}

export interface ProviderConfig {
  notificationEnabled: boolean;
  autoAssignProvider: boolean;
  requiredForHighRisk: boolean;
  responseTimeoutMinutes: number;
}

export interface ApiConfig {
  port: number;
  enableWebSocket: boolean;
  rateLimitPerMinute: number;
  requireAuthentication: boolean;
  maxConcurrentAnalyses: number;
}

export interface LearningConfig {
  enablePatternLearning: boolean;
  agentDbPath: string;
  minimumPatternFrequency: number;
  retrainIntervalHours: number;
}
