/**
 * Phase 5: Type-Safe Template Export System Types
 *
 * Comprehensive type definitions for template export with Pydantic schema generation,
 * validation, and metadata management for production integration.
 */

export interface ExportConfig {
  outputFormat: 'json' | 'yaml' | 'pydantic' | 'typescript';
  includeMetadata: boolean;
  includeValidation: boolean;
  includeDocumentation: boolean;
  compressionLevel?: 'none' | 'gzip' | 'brotli';
  encryptionEnabled?: boolean;
  outputDirectory: string;
  filenameTemplate?: string;
  batchProcessing?: boolean;
  parallelExecution?: boolean;
  maxConcurrency?: number;
}

export interface PydanticSchemaConfig {
  className: string;
  moduleName: string;
  includeValidators: boolean;
  includeSerializers: boolean;
  includeFieldValidators: boolean;
  strictTypes: boolean;
  optionalFields: string[];
  requiredFields: string[];
  fieldAnnotations: Record<string, string>;
  customMethods?: string[];
  imports?: string[];
  docstring?: string;
}

export interface ExportValidationConfig {
  strictMode: boolean;
  validateConstraints: boolean;
  validateDependencies: boolean;
  validateTypes: boolean;
  validateInheritance: boolean;
  validatePerformance: boolean;
  maxProcessingTime: number;
  maxMemoryUsage: number;
  allowedViolations: string[];
  customValidators?: ValidationRule[];
}

export interface ValidationRule {
  name: string;
  type: 'constraint' | 'dependency' | 'type' | 'performance' | 'custom';
  condition: string;
  action: 'error' | 'warning' | 'info';
  message: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  enabled: boolean;
}

export interface ExportMetadata {
  exportId: string;
  exportTimestamp: Date;
  exportConfig: ExportConfig;
  templateInfo: TemplateExportInfo;
  validationResults: ValidationResults;
  performanceMetrics: ExportPerformanceMetrics;
  cognitiveInsights?: CognitiveInsights;
  agentdbIntegration?: AgentDBIntegrationInfo;
}

export interface TemplateExportInfo {
  templateId: string;
  templateName: string;
  templateVersion: string;
  templateType: string;
  variantType?: string;
  priority: number;
  parameterCount: number;
  constraintCount: number;
  inheritanceChain: string[];
  dependencies: string[];
  tags: string[];
  exportFormat: string;
  schemaInfo: SchemaInfo;
}

export interface SchemaInfo {
  schemaType: 'pydantic' | 'typescript' | 'json' | 'yaml';
  schemaVersion: string;
  fieldCount: number;
  requiredFields: number;
  optionalFields: number;
  complexTypes: ComplexTypeInfo[];
  validationRules: SchemaValidationRule[];
  documentationFields: DocumentationField[];
}

export interface ComplexTypeInfo {
  fieldName: string;
  fieldType: string;
  nestedFields: number;
  isRecursive: boolean;
  isGeneric: boolean;
  genericParameters?: string[];
  constraints: FieldConstraint[];
}

export interface FieldConstraint {
  type: 'min' | 'max' | 'pattern' | 'enum' | 'custom';
  value: any;
  description: string;
  validationFunction?: string;
}

export interface SchemaValidationRule {
  fieldName: string;
  ruleType: string;
  condition: string;
  errorMessage: string;
  validationCode: string;
  isRequired: boolean;
}

export interface DocumentationField {
  fieldName: string;
  description: string;
  dataType: string;
  defaultValue?: any;
  examples: any[];
  relatedFields: string[];
  constraints: string[];
  notes: string[];
}

export interface ValidationResults {
  isValid: boolean;
  validationScore: number; // 0-1
  errors: ValidationError[];
  warnings: ValidationWarning[];
  infos: ValidationInfo[];
  suggestions: ValidationSuggestion[];
  totalChecks: number;
  passedChecks: number;
  failedChecks: number;
  processingTime: number;
}

export interface ValidationError {
  id: string;
  type: 'constraint' | 'type' | 'dependency' | 'performance' | 'schema' | 'custom';
  severity: 'error';
  code: string;
  message: string;
  field?: string;
  value?: any;
  expectedValue?: any;
  suggestion?: string;
  fixable: boolean;
  autoFix?: AutoFix;
}

export interface ValidationWarning {
  id: string;
  type: 'performance' | 'deprecation' | 'style' | 'best_practice';
  severity: 'warning';
  code: string;
  message: string;
  field?: string;
  value?: any;
  recommendation?: string;
}

export interface ValidationInfo {
  id: string;
  type: 'info' | 'optimization' | 'suggestion';
  severity: 'info';
  code: string;
  message: string;
  field?: string;
  value?: any;
  improvement?: string;
}

export interface ValidationSuggestion {
  id: string;
  type: 'improvement' | 'optimization' | 'refactoring';
  priority: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  impact: string;
  effort: 'low' | 'medium' | 'high';
  codeExample?: string;
  relatedIssues: string[];
}

export interface AutoFix {
  type: 'replace' | 'add' | 'remove' | 'modify';
  target: string;
  oldValue?: any;
  newValue?: any;
  code: string;
  confidence: number;
}

export interface ExportPerformanceMetrics {
  totalProcessingTime: number;
  templateProcessingTime: number;
  validationTime: number;
  schemaGenerationTime: number;
  metadataGenerationTime: number;
  fileWriteTime: number;
  memoryUsage: MemoryUsageMetrics;
  throughputMetrics: ThroughputMetrics;
  cacheMetrics: CacheMetrics;
  errorMetrics: ErrorMetrics;
}

export interface MemoryUsageMetrics {
  peakMemoryUsage: number;
  averageMemoryUsage: number;
  memoryLeaks: number;
  gcCollections: number;
  heapSize: number;
  externalMemory: number;
}

export interface ThroughputMetrics {
  templatesProcessed: number;
  parametersProcessed: number;
  validationsPerformed: number;
  schemasGenerated: number;
  filesWritten: number;
  averageProcessingRate: number; // templates per second
  peakProcessingRate: number;
}

export interface CacheMetrics {
  cacheHitRate: number;
  cacheMissRate: number;
  totalCacheHits: number;
  totalCacheMisses: number;
  cacheSize: number;
  evictions: number;
  averageLookupTime: number;
}

export interface ErrorMetrics {
  totalErrors: number;
  errorsByType: Record<string, number>;
  errorsBySeverity: Record<string, number>;
  fixableErrors: number;
  autoFixedErrors: number;
  errorRecoveryTime: number;
}

export interface CognitiveInsights {
  consciousnessLevel: number;
  temporalAnalysisDepth: number;
  strangeLoopOptimizations: StrangeLoopOptimization[];
  learningPatterns: LearningPattern[];
  consciousnessEvolution: ConsciousnessEvolution;
  recommendations: CognitiveRecommendation[];
}

export interface StrangeLoopOptimization {
  iteration: number;
  optimizationType: string;
  effectiveness: number;
  improvements: string[];
  processingTime: number;
  confidence: number;
}

export interface LearningPattern {
  patternId: string;
  patternType: string;
  frequency: number;
  successRate: number;
  lastUsed: Date;
  effectiveness: number;
  relatedPatterns: string[];
}

export interface ConsciousnessEvolution {
  previousLevel: number;
  currentLevel: number;
  evolutionRate: number;
  evolutionFactors: string[];
  adaptationStrategies: string[];
  metaOptimizations: string[];
}

export interface CognitiveRecommendation {
  type: 'optimization' | 'validation' | 'documentation' | 'performance';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  reasoning: string;
  expectedBenefit: string;
  implementationSteps: string[];
  confidence: number;
}

export interface AgentDBIntegrationInfo {
  connected: boolean;
  syncStatus: 'synced' | 'syncing' | 'error';
  lastSyncTime: Date;
  storedPatterns: number;
  retrievedPatterns: number;
  synchronizationTime: number;
  patterns: AgentDBPattern[];
}

export interface AgentDBPattern {
  patternId: string;
  patternType: string;
  content: any;
  confidence: number;
  createdAt: Date;
  lastAccessed: Date;
  accessCount: number;
  relatedPatterns: string[];
}

export interface ExportJob {
  jobId: string;
  config: ExportConfig;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  progress: number; // 0-1
  startTime: Date;
  endTime?: Date;
  templates: string[];
  results?: ExportResult[];
  errors?: ExportError[];
  metadata?: ExportMetadata;
}

export interface ExportResult {
  templateId: string;
  outputPath: string;
  outputFormat: string;
  fileSize: number;
  checksum: string;
  validationResults: ValidationResults;
  performanceMetrics: Partial<ExportPerformanceMetrics>;
  warnings: string[];
  errors: string[];
}

export interface ExportError {
  errorId: string;
  type: 'template_error' | 'validation_error' | 'schema_error' | 'file_error' | 'system_error';
  severity: 'error' | 'warning' | 'info';
  message: string;
  details?: any;
  stackTrace?: string;
  templateId?: string;
  recoverable: boolean;
  recoveryAction?: string;
  timestamp: Date;
}

export interface BatchExportConfig {
  jobConfig: ExportConfig;
  templateGroups: TemplateGroup[];
  parallelProcessing: boolean;
  maxConcurrency: number;
  progressCallback?: (progress: BatchExportProgress) => void;
  errorHandling: 'fail_fast' | 'continue_on_error' | 'retry';
  retryConfig?: RetryConfig;
}

export interface TemplateGroup {
  groupId: string;
  groupName: string;
  templateIds: string[];
  exportConfig?: Partial<ExportConfig>;
  dependencies?: string[];
}

export interface BatchExportProgress {
  totalGroups: number;
  completedGroups: number;
  totalTemplates: number;
  completedTemplates: number;
  currentGroup?: string;
  currentTemplate?: string;
  processingTime: number;
  estimatedTimeRemaining: number;
  errors: number;
  warnings: number;
}

export interface RetryConfig {
  maxRetries: number;
  retryDelay: number;
  exponentialBackoff: boolean;
  retryCondition: (error: ExportError) => boolean;
}

export interface ExportCache {
  enabled: boolean;
  maxSize: number;
  ttl: number;
  evictionPolicy: 'lru' | 'lfu' | 'fifo';
  compressionEnabled: boolean;
  compressionLevel: number;
  keyPrefix: string;
}

export interface ExportStatistics {
  totalExports: number;
  successfulExports: number;
  failedExports: number;
  averageProcessingTime: number;
  totalProcessingTime: number;
  totalTemplatesProcessed: number;
  totalParametersProcessed: number;
  totalValidationChecks: number;
  totalSchemasGenerated: number;
  exportsByFormat: Record<string, number>;
  exportsByType: Record<string, number>;
  errorDistribution: Record<string, number>;
  performanceDistribution: PerformanceDistribution[];
}

export interface PerformanceDistribution {
  timeRange: string;
  count: number;
  averageTime: number;
  minTime: number;
  maxTime: number;
  medianTime: number;
  p95Time: number;
  p99Time: number;
}

// Export event types
export interface ExportEvent {
  eventType: 'export_started' | 'export_completed' | 'export_failed' | 'template_processed' | 'validation_completed' | 'schema_generated';
  jobId?: string;
  templateId?: string;
  timestamp: Date;
  data: any;
  processingTime?: number;
  error?: Error;
}

// Export status types
export interface ExportStatus {
  isActive: boolean;
  currentJobs: number;
  queuedJobs: number;
  completedJobs: number;
  failedJobs: number;
  averageProcessingTime: number;
  systemLoad: number;
  memoryUsage: number;
  cacheStatus: CacheStatus;
}

export interface CacheStatus {
  enabled: boolean;
  currentSize: number;
  hitRate: number;
  missRate: number;
  evictions: number;
  lastCleanup: Date;
}