/**
 * Verification System Main Export
 * Comprehensive anti-hallucination and verification system
 */

export { ConfidenceScorer } from './core/confidence-scorer';
export type {
  ConfidenceScore,
  ConfidenceMetadata,
  MedicalCitation,
} from './core/confidence-scorer';

export { VerificationPipeline } from './pipeline/verification-pipeline';
export type {
  VerificationInput,
  VerificationResult,
  VerificationMetadata,
  HallucinationDetection,
  ProviderReview,
} from './pipeline/verification-pipeline';

export { LeanAgenticIntegration } from './integrations/lean-agentic-integration';
export type {
  CausalModel,
  CausalInferenceResult,
  StatisticalTest,
  PowerAnalysis,
} from './integrations/lean-agentic-integration';

export { StrangeLoopsDetector } from './patterns/strange-loops-detector';
export type {
  LogicalPattern,
  CausalChain,
  RecursivePattern,
} from './patterns/strange-loops-detector';

export { AgentDBIntegration } from './learning/agentdb-integration';
export type {
  LearningRecord,
  ProviderFeedback,
  Pattern,
  SourceReliability,
} from './learning/agentdb-integration';

/**
 * Main Verification System Class
 * Orchestrates all verification components
 */
export class VerificationSystem {
  private confidenceScorer: ConfidenceScorer;
  private pipeline: VerificationPipeline;
  private leanAgentic: LeanAgenticIntegration;
  private loopsDetector: StrangeLoopsDetector;
  private agentDB: AgentDBIntegration;

  constructor() {
    this.confidenceScorer = new ConfidenceScorer();
    this.pipeline = new VerificationPipeline();
    this.leanAgentic = new LeanAgenticIntegration();
    this.loopsDetector = new StrangeLoopsDetector();
    this.agentDB = new AgentDBIntegration();
  }

  /**
   * Perform comprehensive verification
   */
  async verify(input: any): Promise<any> {
    // Pre-output verification
    const verificationResult = await this.pipeline.preOutputVerification(input);

    // Detect logical issues
    const logicalPatterns = await this.loopsDetector.detectCircularReasoning(
      input.claim
    );
    const contradictions = await this.loopsDetector.detectContradictions(
      input.claim
    );

    // Get learning-based adjustments
    const adjustment = await this.agentDB.getConfidenceAdjustment(
      input.features || {},
      input.context || []
    );

    return {
      ...verificationResult,
      logicalPatterns,
      contradictions,
      learningAdjustment: adjustment,
    };
  }

  /**
   * Get system statistics
   */
  getStatistics(): any {
    return {
      pipeline: this.pipeline.getStatistics(),
      patterns: this.agentDB.getPatternStatistics(),
      model: this.agentDB.getModelStatistics(),
    };
  }
}
