/**
 * Verification Pipeline
 * Pre-output verification, real-time hallucination detection, post-output validation
 */

import { ConfidenceScorer, ConfidenceScore, MedicalCitation } from '../core/confidence-scorer';

export interface VerificationInput {
  claim: string;
  context?: Record<string, any>;
  citations?: MedicalCitation[];
  metadata?: VerificationMetadata;
}

export interface VerificationMetadata {
  agentId?: string;
  timestamp: number;
  source: string;
  category?: string;
  requiresProviderReview?: boolean;
}

export interface VerificationResult {
  verified: boolean;
  confidence: ConfidenceScore;
  hallucinations: HallucinationDetection[];
  warnings: string[];
  requiresReview: boolean;
  suggestions: string[];
  timestamp: number;
}

export interface HallucinationDetection {
  type: 'factual' | 'citation' | 'logical' | 'temporal' | 'quantitative';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  location?: string;
  suggestion?: string;
}

export interface ProviderReview {
  reviewerId: string;
  approved: boolean;
  corrections: string[];
  feedback: string;
  timestamp: number;
}

export class VerificationPipeline {
  private confidenceScorer: ConfidenceScorer;
  private hallucinationPatterns: Map<string, RegExp>;
  private providerReviews: Map<string, ProviderReview[]>;

  constructor() {
    this.confidenceScorer = new ConfidenceScorer();
    this.hallucinationPatterns = new Map();
    this.providerReviews = new Map();
    this.initializeHallucinationPatterns();
  }

  /**
   * Initialize common hallucination patterns
   */
  private initializeHallucinationPatterns(): void {
    // Overly confident language
    this.hallucinationPatterns.set(
      'overconfident',
      /\b(always|never|absolutely|definitely|certainly|guaranteed|100%|impossible)\b/gi
    );

    // Unsupported quantitative claims
    this.hallucinationPatterns.set(
      'unsupported-numbers',
      /\b(\d+%|\d+\s*times?\s*(more|less|better|worse))\b/gi
    );

    // Temporal hallucinations
    this.hallucinationPatterns.set(
      'temporal-vague',
      /\b(recent studies|latest research|modern medicine)\b/gi
    );

    // Medical jargon without support
    this.hallucinationPatterns.set(
      'unsupported-medical',
      /\b(cures?|eliminates?|completely|permanently fixes?)\b/gi
    );
  }

  /**
   * Pre-output verification - check before generating output
   */
  async preOutputVerification(input: VerificationInput): Promise<VerificationResult> {
    const startTime = Date.now();

    // Step 1: Check for immediate red flags
    const warnings: string[] = [];
    const hallucinations: HallucinationDetection[] = [];

    // Check citation availability
    if (!input.citations || input.citations.length === 0) {
      warnings.push('No citations provided - confidence will be limited');
      hallucinations.push({
        type: 'citation',
        severity: 'medium',
        description: 'Claim made without supporting citations',
        suggestion: 'Provide peer-reviewed sources to support this claim',
      });
    }

    // Step 2: Pattern-based hallucination detection
    const patternHallucinations = this.detectPatternHallucinations(input.claim);
    hallucinations.push(...patternHallucinations);

    // Step 3: Calculate confidence
    const confidence = await this.confidenceScorer.calculateConfidence(
      input.claim,
      input.citations || [],
      input.context
    );

    // Step 4: Determine if verification passed
    const verified = this.confidenceScorer.isConfident(confidence) &&
                     hallucinations.filter(h => h.severity === 'critical').length === 0;

    // Step 5: Determine if provider review required
    const requiresReview = !verified ||
                          input.metadata?.requiresProviderReview === true ||
                          hallucinations.some(h => h.severity === 'high' || h.severity === 'critical');

    return {
      verified,
      confidence,
      hallucinations,
      warnings,
      requiresReview,
      suggestions: this.generateSuggestions(confidence, hallucinations),
      timestamp: Date.now() - startTime,
    };
  }

  /**
   * Real-time hallucination detection
   */
  async detectHallucinations(text: string, context?: Record<string, any>): Promise<HallucinationDetection[]> {
    const hallucinations: HallucinationDetection[] = [];

    // Pattern-based detection
    hallucinations.push(...this.detectPatternHallucinations(text));

    // Logical consistency checks
    hallucinations.push(...this.detectLogicalInconsistencies(text));

    // Quantitative claim validation
    hallucinations.push(...this.detectQuantitativeHallucinations(text));

    // Temporal accuracy checks
    hallucinations.push(...this.detectTemporalHallucinations(text));

    return hallucinations;
  }

  /**
   * Pattern-based hallucination detection
   */
  private detectPatternHallucinations(text: string): HallucinationDetection[] {
    const hallucinations: HallucinationDetection[] = [];

    // Check overconfident language
    const overconfidentMatches = text.match(this.hallucinationPatterns.get('overconfident')!);
    if (overconfidentMatches && overconfidentMatches.length > 2) {
      hallucinations.push({
        type: 'factual',
        severity: 'medium',
        description: 'Overconfident language detected (always/never/guaranteed)',
        suggestion: 'Use more measured language with appropriate qualifiers',
      });
    }

    // Check unsupported quantitative claims
    const numberMatches = text.match(this.hallucinationPatterns.get('unsupported-numbers')!);
    if (numberMatches) {
      hallucinations.push({
        type: 'quantitative',
        severity: 'high',
        description: 'Quantitative claims detected without citations',
        suggestion: 'Provide peer-reviewed sources for statistical claims',
      });
    }

    // Check temporal vagueness
    const temporalMatches = text.match(this.hallucinationPatterns.get('temporal-vague')!);
    if (temporalMatches) {
      hallucinations.push({
        type: 'temporal',
        severity: 'low',
        description: 'Vague temporal references (recent studies, latest research)',
        suggestion: 'Specify exact years and studies',
      });
    }

    // Check unsupported medical claims
    const medicalMatches = text.match(this.hallucinationPatterns.get('unsupported-medical')!);
    if (medicalMatches) {
      hallucinations.push({
        type: 'factual',
        severity: 'critical',
        description: 'Absolute medical claims detected (cures, eliminates, completely)',
        suggestion: 'Use evidence-based language and provide citations',
      });
    }

    return hallucinations;
  }

  /**
   * Detect logical inconsistencies
   */
  private detectLogicalInconsistencies(text: string): HallucinationDetection[] {
    const hallucinations: HallucinationDetection[] = [];

    // Check for contradictory statements within text
    const sentences = text.split(/[.!?]+/).filter(s => s.trim());

    // Simple contradiction detection (can be enhanced with NLP)
    const negationWords = ['not', 'no', 'never', 'none'];
    const hasPositive = sentences.some(s => !negationWords.some(n => s.toLowerCase().includes(n)));
    const hasNegative = sentences.some(s => negationWords.some(n => s.toLowerCase().includes(n)));

    if (hasPositive && hasNegative && sentences.length < 5) {
      hallucinations.push({
        type: 'logical',
        severity: 'medium',
        description: 'Potential logical contradiction detected',
        suggestion: 'Review for internal consistency',
      });
    }

    return hallucinations;
  }

  /**
   * Detect quantitative hallucinations
   */
  private detectQuantitativeHallucinations(text: string): HallucinationDetection[] {
    const hallucinations: HallucinationDetection[] = [];

    // Check for percentages over 100%
    const percentageMatch = text.match(/(\d+)%/g);
    if (percentageMatch) {
      percentageMatch.forEach(match => {
        const value = parseInt(match);
        if (value > 100) {
          hallucinations.push({
            type: 'quantitative',
            severity: 'critical',
            description: `Invalid percentage: ${match}`,
            suggestion: 'Percentages cannot exceed 100%',
          });
        }
      });
    }

    // Check for suspiciously precise numbers
    const preciseMatch = text.match(/\b(\d+\.\d{3,})\b/g);
    if (preciseMatch) {
      hallucinations.push({
        type: 'quantitative',
        severity: 'low',
        description: 'Suspiciously precise numbers detected',
        suggestion: 'Round to appropriate precision with confidence intervals',
      });
    }

    return hallucinations;
  }

  /**
   * Detect temporal hallucinations
   */
  private detectTemporalHallucinations(text: string): HallucinationDetection[] {
    const hallucinations: HallucinationDetection[] = [];

    // Check for future dates
    const currentYear = new Date().getFullYear();
    const yearMatches = text.match(/\b(19\d{2}|20\d{2})\b/g);

    if (yearMatches) {
      yearMatches.forEach(year => {
        const yearNum = parseInt(year);
        if (yearNum > currentYear) {
          hallucinations.push({
            type: 'temporal',
            severity: 'critical',
            description: `Future date referenced: ${year}`,
            suggestion: 'Cannot cite studies from the future',
          });
        }
      });
    }

    return hallucinations;
  }

  /**
   * Post-output validation
   */
  async postOutputValidation(
    output: string,
    originalInput: VerificationInput
  ): Promise<VerificationResult> {
    // Re-run verification on the generated output
    const verificationInput: VerificationInput = {
      claim: output,
      context: originalInput.context,
      citations: originalInput.citations,
      metadata: originalInput.metadata,
    };

    const result = await this.preOutputVerification(verificationInput);

    // Add additional post-output checks
    if (result.verified) {
      // Check if output stayed faithful to input
      const fidelityCheck = this.checkOutputFidelity(originalInput.claim, output);
      if (!fidelityCheck.faithful) {
        result.warnings.push(fidelityCheck.reason);
        result.requiresReview = true;
      }
    }

    return result;
  }

  /**
   * Check if output is faithful to input
   */
  private checkOutputFidelity(input: string, output: string): { faithful: boolean; reason: string } {
    // Check length inflation (potential hallucination)
    if (output.length > input.length * 3) {
      return {
        faithful: false,
        reason: 'Output significantly longer than input - potential elaboration beyond evidence',
      };
    }

    // Check for added quantitative claims
    const inputNumbers = (input.match(/\d+/g) || []).length;
    const outputNumbers = (output.match(/\d+/g) || []).length;

    if (outputNumbers > inputNumbers * 2) {
      return {
        faithful: false,
        reason: 'Output contains many more numbers than input - potential quantitative hallucination',
      };
    }

    return { faithful: true, reason: '' };
  }

  /**
   * Integrate provider review
   */
  async addProviderReview(claimId: string, review: ProviderReview): Promise<void> {
    if (!this.providerReviews.has(claimId)) {
      this.providerReviews.set(claimId, []);
    }
    this.providerReviews.get(claimId)!.push(review);
  }

  /**
   * Get provider reviews for a claim
   */
  getProviderReviews(claimId: string): ProviderReview[] {
    return this.providerReviews.get(claimId) || [];
  }

  /**
   * Generate suggestions for improvement
   */
  private generateSuggestions(
    confidence: ConfidenceScore,
    hallucinations: HallucinationDetection[]
  ): string[] {
    const suggestions: string[] = [];

    if (confidence.citationStrength < 0.6) {
      suggestions.push('Add more high-quality peer-reviewed sources');
    }

    if (confidence.medicalAgreement < 0.7) {
      suggestions.push('Ensure consensus across multiple medical sources');
    }

    if (confidence.expertConsensus < 0.5) {
      suggestions.push('Include clinical guidelines or meta-analyses');
    }

    if (hallucinations.length > 0) {
      suggestions.push('Address detected hallucinations and unsupported claims');
    }

    if (confidence.contradictions.length > 0) {
      suggestions.push('Resolve contradictions in evidence base');
    }

    return suggestions;
  }

  /**
   * Get pipeline statistics
   */
  getStatistics(): {
    totalVerifications: number;
    verifiedCount: number;
    reviewCount: number;
    averageConfidence: number;
  } {
    // Placeholder - implement with actual tracking
    return {
      totalVerifications: 0,
      verifiedCount: 0,
      reviewCount: 0,
      averageConfidence: 0,
    };
  }
}
