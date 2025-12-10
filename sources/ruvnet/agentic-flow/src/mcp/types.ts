/**
 * Medical MCP Server - Type Definitions
 * Comprehensive types for medical analysis with anti-hallucination features
 */

export interface MedicalCondition {
  name: string;
  icd10Code?: string;
  severity: 'mild' | 'moderate' | 'severe' | 'critical';
  confidence: number;
  symptoms: string[];
  differential?: string[];
}

export interface MedicalAnalysis {
  id: string;
  timestamp: number;
  conditions: MedicalCondition[];
  recommendations: string[];
  urgencyLevel: 'routine' | 'urgent' | 'emergency';
  confidence: number;
  citations: Citation[];
  requiresProviderReview: boolean;
  providerApproved?: boolean;
  providerNotes?: string;
}

export interface Citation {
  id: string;
  source: string;
  sourceType: 'clinical_guideline' | 'research_paper' | 'textbook' | 'database';
  title: string;
  authors?: string[];
  year?: number;
  url?: string;
  excerpt?: string;
  relevanceScore: number;
  verified: boolean;
}

export interface ConfidenceMetrics {
  overall: number;
  byComponent: {
    diagnosis: number;
    treatment: number;
    prognosis: number;
  };
  uncertaintyFactors: string[];
  dataQuality: number;
  modelAgreement?: number;
}

export interface ProviderNotification {
  id: string;
  timestamp: number;
  analysisId: string;
  urgency: 'low' | 'medium' | 'high' | 'critical';
  recipient: string;
  channel: 'email' | 'sms' | 'pager' | 'app';
  status: 'sent' | 'delivered' | 'acknowledged' | 'failed';
  message: string;
}

export interface EmergencyEscalation {
  id: string;
  timestamp: number;
  analysisId: string;
  trigger: string;
  severity: 'high' | 'critical';
  actions: string[];
  notifiedParties: string[];
  responseTime: number;
  resolved: boolean;
}

export interface KnowledgeSearchQuery {
  query: string;
  filters?: {
    sourceTypes?: string[];
    minRelevance?: number;
    dateRange?: { start: Date; end: Date };
    specialties?: string[];
  };
  maxResults?: number;
}

export interface KnowledgeSearchResult {
  id: string;
  title: string;
  content: string;
  source: string;
  relevanceScore: number;
  citations: Citation[];
  lastUpdated: Date;
}

export interface ValidationResult {
  isValid: boolean;
  confidence: number;
  issues: ValidationIssue[];
  recommendations: string[];
}

export interface ValidationIssue {
  type: 'inconsistency' | 'missing_citation' | 'low_confidence' | 'contraindication' | 'red_flag';
  severity: 'info' | 'warning' | 'error' | 'critical';
  description: string;
  suggestion?: string;
}

export interface MCPToolRequest {
  name: string;
  arguments?: Record<string, any>;
}

export interface MCPToolResponse {
  content: Array<{
    type: 'text' | 'json';
    text?: string;
    json?: any;
  }>;
  isError?: boolean;
}

export interface AgentDBPattern {
  taskType: string;
  approach: string;
  successRate: number;
  tags: string[];
  metadata: Record<string, any>;
}

export interface LearningFeedback {
  analysisId: string;
  accuracy: number;
  providerFeedback: string;
  corrections?: string[];
  timestamp: number;
}
