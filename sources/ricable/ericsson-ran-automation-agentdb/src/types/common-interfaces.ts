/**
 * Common Interface Definitions for Phase 5 Integration
 *
 * This file provides shared interface definitions used across
 * Phase 5 components to resolve compilation issues.
 */

// ============================================================================
// Search and Recommendation Types
// ============================================================================

export interface SearchContext {
  query: string;
  filters: Record<string, any>;
  pagination?: {
    offset: number;
    limit: number;
  };
}

export interface SemanticSearchResult {
  id: string;
  content: string;
  relevanceScore: number;
  metadata: SearchMetadata;
}

export interface SearchMetadata {
  source: string;
  timestamp: Date;
  confidence: number;
  context?: Record<string, any>;
}

export interface RelevanceScore {
  score: number;
  factors: string[];
  confidence: number;
}

export interface SearchSuggestion {
  text: string;
  type: string;
  relevance: number;
}

export interface SimilarityMetrics {
  cosineSimilarity: number;
  jaccardSimilarity: number;
  euclideanDistance: number;
  semanticSimilarity: number;
}

export interface SimilarityRecommendation {
  item: any;
  similarity: SimilarityMetrics;
  explanation: string;
}

export interface SimilarityFactor {
  factor: string;
  weight: number;
  value: number;
}

export interface AdaptationSuggestion {
  suggestion: string;
  reasoning: string;
  confidence: number;
  impact: string;
}

// ============================================================================
// Learning and Recommendation Types
// ============================================================================

export interface RecommendationContext {
  currentConfiguration: any;
  performanceMetrics: any;
  constraints: any[];
  goals: string[];
}

export interface PatternRequirements {
  patternType: string;
  parameters: string[];
  constraints: any[];
  expectedOutcomes: string[];
}

export interface OptimalPatternRecommendation {
  pattern: any;
  confidence: number;
  expectedBenefits: ExpectedBenefits;
  adaptationSuggestions: AdaptationSuggestion[];
}

export interface UsageData {
  usageCount: number;
  lastUsed: Date;
  successRate: number;
  averagePerformance: number;
}

export interface LearningInsight {
  insight: string;
  confidence: number;
  impact: string;
  supportingData: any[];
}

export interface RecommendationFeedback {
  recommendationId: string;
  wasApplied: boolean;
  outcome: string;
  feedback: string;
  effectiveness: number;
}

export interface ExpectedBenefits {
  performanceImprovement: number;
  costReduction: number;
  efficiencyGain: number;
  riskMitigation: string[];
}

export interface RecommendationRiskAssessment {
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  riskFactors: string[];
  mitigationStrategies: string[];
  probabilityOfFailure: number;
}

export interface AlternativeOption {
  option: any;
  benefits: ExpectedBenefits;
  risks: RecommendationRiskAssessment;
  feasibility: number;
}

// ============================================================================
// Template and Constraint Types
// ============================================================================

export interface ConstraintResolver {
  resolveConstraints(constraints: any[]): Promise<any[]>;
  validateConstraints(solution: any, constraints: any[]): Promise<boolean>;
  applyConstraints(base: any, constraints: any[]): Promise<any>;
}

export interface InheritanceProcessor {
  processInheritance(template: any, baseTemplates: any[]): Promise<any>;
  resolveConflicts(template1: any, template2: any[]): Promise<any>;
  applyInheritanceRules(template: any, rules: any[]): Promise<any>;
}

export interface TemplateSearchCriteria {
  type?: string;
  priority?: number;
  tags?: string[];
  dateRange?: {
    start: Date;
    end: Date;
  };
}

export interface TemplateVersion {
  version: string;
  changes: string[];
  author: string;
  timestamp: Date;
  compatibility: string[];
}

export interface CLITemplateStructure {
  id: string;
  name: string;
  version: string;
  description: string;
  parameters: CLIParameter[];
  constraints: CLIConstraint[];
  metadata: CLIMetadata;
}

export interface CLIParameter {
  name: string;
  type: string;
  required: boolean;
  defaultValue?: any;
  validation?: any;
  description?: string;
}

export interface CLIConstraint {
  type: string;
  parameters: string[];
  condition: string;
  message: string;
}

export interface CLIMetadata {
  author: string;
  created: Date;
  modified: Date;
  tags: string[];
  version: string;
  dependencies: string[];
}

export interface MORelationship {
  source: string;
  target: string;
  type: string;
  constraints: any[];
  description?: string;
}

// ============================================================================
// AgentDB Integration Types
// ============================================================================

export interface AgentDBPatternStorage {
  storePattern(pattern: any): Promise<void>;
  retrievePattern(id: string): Promise<any>;
  searchPatterns(query: any): Promise<any[]>;
  deletePattern(id: string): Promise<void>;
}

export interface AgentDBLearningIntegration {
  recordLearning(learning: any): Promise<void>;
  retrieveLearnings(context: any): Promise<any[]>;
  updateLearning(id: string, updates: any): Promise<void>;
}

export interface AgentDBTemporalStorage {
  storeTemporalData(data: any, timestamp?: Date): Promise<void>;
  retrieveTemporalData(timeRange: { start: Date; end: Date }): Promise<any[]>;
  queryTemporalData(query: any): Promise<any[]>;
}

export interface StorageResult<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  metadata?: any;
}

export interface PatternQuery {
  type?: string;
  parameters?: Record<string, any>;
  timeRange?: { start: Date; end: Date };
  similarity?: {
    vector?: number[];
    threshold?: number;
  };
  limit?: number;
  offset?: number;
}

export interface ExecutionHistory {
  id: string;
  timestamp: Date;
  type: string;
  input: any;
  output: any;
  success: boolean;
  duration: number;
  metadata?: any;
}

export interface SequenceContext {
  sequenceId: string;
  step: number;
  totalSteps: number;
  context: Record<string, any>;
  previousResults?: any[];
}

export interface OptimalSequence {
  sequence: any[];
  expectedDuration: number;
  confidence: number;
  risks: string[];
  benefits: string[];
}

export interface CognitiveInsight {
  insight: string;
  confidence: number;
  factors: string[];
  recommendations: string[];
  evidence: any[];
}

// ============================================================================
// Performance and Monitoring Types
// ============================================================================

export interface IndexOptimizationResult {
  indexType: string;
  optimizationApplied: boolean;
  performanceGain: number;
  beforeMetrics: any;
  afterMetrics: any;
}

export interface EmbeddingUpdateResult {
  embeddingId: string;
  success: boolean;
  dimensions: number;
  updateTimestamp: Date;
  performanceMetrics: any;
}

export interface ExpertSystemMetrics {
  accuracy: number;
  responseTime: number;
  successRate: number;
  learningRate: number;
  knowledgeBaseSize: number;
  lastUpdate: Date;
}

export interface PatternLibraryMetrics {
  totalPatterns: number;
  activePatterns: number;
  usageFrequency: Record<string, number>;
  averagePerformance: number;
  recentAdditions: number;
  deprecatedPatterns: number;
}

export interface PatternUsageStats {
  patternId: string;
  usageCount: number;
  successRate: number;
  averagePerformance: number;
  lastUsed: Date;
  userFeedback: number[];
}

// ============================================================================
// Validation Types
// ============================================================================

export interface ValidationWarning {
  code: string;
  message: string;
  severity: 'low' | 'medium' | 'high';
  suggestion?: string;
  location?: string;
}

export interface ValidationSuggestion {
  type: string;
  description: string;
  action: string;
  impact: string;
}

export interface ValidationMetadata {
  timestamp: Date;
  validator: string;
  version: string;
  rules: string[];
  results: any[];
}

export interface ErrorSeverity {
  level: 'info' | 'warning' | 'error' | 'critical';
  impact: string;
  actionRequired: boolean;
}

export interface SpecializedResource {
  type: string;
  capacity: number;
  availability: number;
  constraints: any[];
  costStructure: any;
}