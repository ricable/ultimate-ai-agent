/**
 * Anti-Hallucination Service
 * Implements confidence scoring, citation verification, and hallucination detection
 */

import type {
  AnalysisResult,
  ConfidenceScore,
  ConfidenceFactor,
  Citation,
  Warning,
  Contradiction,
  Diagnosis,
  MedicalKnowledgeBase
} from '../types/medical.types';

export class AntiHallucinationService {
  private readonly CONFIDENCE_THRESHOLDS = {
    HIGH: 0.85,
    MEDIUM: 0.70,
    LOW: 0.50,
    REQUIRES_REVIEW: 0.75,
    AUTO_APPROVE: 0.90
  };

  private knowledgeBase: Map<string, MedicalKnowledgeBase> = new Map();

  /**
   * Calculate comprehensive confidence score for analysis result
   */
  public calculateConfidenceScore(analysis: AnalysisResult): ConfidenceScore {
    const factors: ConfidenceFactor[] = [];

    // 1. Diagnosis confidence
    const diagnosisConfidence = this.assessDiagnosisConfidence(analysis.diagnosis);
    factors.push({
      name: 'diagnosis_confidence',
      impact: diagnosisConfidence > 0.8 ? 'positive' : diagnosisConfidence < 0.5 ? 'negative' : 'neutral',
      weight: 0.3,
      description: `Diagnosis confidence: ${(diagnosisConfidence * 100).toFixed(1)}%`
    });

    // 2. Citation verification
    const citationScore = this.verifyCitations(analysis.citations);
    factors.push({
      name: 'citation_verification',
      impact: citationScore > 0.8 ? 'positive' : citationScore < 0.6 ? 'negative' : 'neutral',
      weight: 0.25,
      description: `${analysis.citations.filter(c => c.verified).length}/${analysis.citations.length} citations verified`
    });

    // 3. Knowledge base validation
    const knowledgeBaseScore = this.validateAgainstKnowledgeBase(analysis);
    factors.push({
      name: 'knowledge_base_validation',
      impact: knowledgeBaseScore > 0.8 ? 'positive' : knowledgeBaseScore < 0.6 ? 'negative' : 'neutral',
      weight: 0.2,
      description: `Knowledge base alignment: ${(knowledgeBaseScore * 100).toFixed(1)}%`
    });

    // 4. Contradiction check
    const contradictionScore = this.checkContradictions(analysis);
    factors.push({
      name: 'contradiction_check',
      impact: contradictionScore > 0.9 ? 'positive' : contradictionScore < 0.7 ? 'negative' : 'neutral',
      weight: 0.15,
      description: `${analysis.diagnosis.reduce((sum, d) => sum + (d.contradictions?.length || 0), 0)} contradictions found`
    });

    // 5. Provider alignment (if reviewed)
    const providerScore = analysis.reviewedBy ? 1.0 : 0.5;
    factors.push({
      name: 'provider_alignment',
      impact: analysis.reviewedBy ? 'positive' : 'neutral',
      weight: 0.1,
      description: analysis.reviewedBy ? 'Provider reviewed' : 'Awaiting provider review'
    });

    // Calculate overall score
    const overall = factors.reduce((sum, factor) => {
      const score = factor.impact === 'positive' ? 1 : factor.impact === 'negative' ? 0 : 0.7;
      return sum + (score * factor.weight);
    }, 0);

    return {
      overall,
      breakdown: {
        diagnosisConfidence,
        citationVerification: citationScore,
        knowledgeBaseValidation: knowledgeBaseScore,
        contradictionCheck: contradictionScore,
        providerAlignment: providerScore
      },
      factors,
      thresholds: {
        requiresProviderReview: this.CONFIDENCE_THRESHOLDS.REQUIRES_REVIEW,
        autoApprove: this.CONFIDENCE_THRESHOLDS.AUTO_APPROVE,
        flagForReview: this.CONFIDENCE_THRESHOLDS.MEDIUM
      },
      explanation: this.generateConfidenceExplanation(overall, factors)
    };
  }

  /**
   * Assess confidence in diagnosis based on multiple factors
   */
  private assessDiagnosisConfidence(diagnoses: Diagnosis[]): number {
    if (diagnoses.length === 0) return 0;

    const primaryDiagnosis = diagnoses[0];
    let score = primaryDiagnosis.confidence;

    // Reduce score if supporting evidence is weak
    const evidenceStrength = primaryDiagnosis.supportingEvidence.reduce(
      (sum, ev) => sum + ev.reliability,
      0
    ) / Math.max(primaryDiagnosis.supportingEvidence.length, 1);

    score *= evidenceStrength;

    // Reduce score for contradictions
    if (primaryDiagnosis.contradictions && primaryDiagnosis.contradictions.length > 0) {
      const contradictionPenalty = primaryDiagnosis.contradictions.length * 0.1;
      score *= (1 - Math.min(contradictionPenalty, 0.5));
    }

    // Boost score if differential diagnoses are well-reasoned
    if (primaryDiagnosis.differentialDiagnoses.length >= 2) {
      score *= 1.1;
    }

    return Math.min(score, 1.0);
  }

  /**
   * Verify citations against known sources
   */
  private verifyCitations(citations: Citation[]): number {
    if (citations.length === 0) return 0.5; // Neutral if no citations

    const verifiedCount = citations.filter(c => {
      // Check if citation has verifiable identifiers
      return c.verified && (c.doi || c.pubmedId || c.url);
    }).length;

    const verificationRate = verifiedCount / citations.length;

    // Weight by relevance scores
    const avgRelevance = citations.reduce((sum, c) => sum + c.relevanceScore, 0) / citations.length;

    return (verificationRate * 0.7) + (avgRelevance * 0.3);
  }

  /**
   * Validate analysis against medical knowledge base
   */
  private validateAgainstKnowledgeBase(analysis: AnalysisResult): number {
    let validationScore = 0;
    let validationCount = 0;

    for (const diagnosis of analysis.diagnosis) {
      const knowledge = this.knowledgeBase.get(diagnosis.condition);
      if (!knowledge) {
        validationCount++;
        validationScore += 0.5; // Unknown condition, neutral score
        continue;
      }

      // Check if diagnostic criteria match
      let criteriaMatch = 0;
      for (const criteria of knowledge.diagnosticCriteria) {
        const evidenceTypes = diagnosis.supportingEvidence.map(e => e.type);
        const hasRequired = criteria.required.every(req =>
          diagnosis.supportingEvidence.some(ev => ev.description.toLowerCase().includes(req.toLowerCase()))
        );
        if (hasRequired) criteriaMatch++;
      }

      validationCount++;
      validationScore += criteriaMatch / Math.max(knowledge.diagnosticCriteria.length, 1);
    }

    return validationCount > 0 ? validationScore / validationCount : 0.5;
  }

  /**
   * Check for contradictions in analysis
   */
  private checkContradictions(analysis: AnalysisResult): number {
    const contradictions: Contradiction[] = [];

    // Check for contradictions between diagnoses
    for (let i = 0; i < analysis.diagnosis.length; i++) {
      for (let j = i + 1; j < analysis.diagnosis.length; j++) {
        const d1 = analysis.diagnosis[i];
        const d2 = analysis.diagnosis[j];

        // Check for mutually exclusive conditions
        if (this.areMutuallyExclusive(d1.condition, d2.condition)) {
          contradictions.push({
            statement1: `Diagnosis: ${d1.condition}`,
            statement2: `Diagnosis: ${d2.condition}`,
            severity: 'major',
            requiresProviderReview: true
          });
        }
      }
    }

    // Check for contradictions in recommendations
    for (let i = 0; i < analysis.recommendations.length; i++) {
      for (let j = i + 1; j < analysis.recommendations.length; j++) {
        const r1 = analysis.recommendations[i];
        const r2 = analysis.recommendations[j];

        if (this.areContradictoryRecommendations(r1.description, r2.description)) {
          contradictions.push({
            statement1: r1.description,
            statement2: r2.description,
            severity: 'moderate',
            requiresProviderReview: true
          });
        }
      }
    }

    // Score based on contradiction severity
    const severityPenalty = contradictions.reduce((sum, c) => {
      return sum + (c.severity === 'major' ? 0.3 : c.severity === 'moderate' ? 0.2 : 0.1);
    }, 0);

    return Math.max(0, 1 - severityPenalty);
  }

  /**
   * Generate warnings based on analysis quality
   */
  public generateWarnings(analysis: AnalysisResult, confidenceScore: ConfidenceScore): Warning[] {
    const warnings: Warning[] = [];

    // Low confidence warning
    if (confidenceScore.overall < this.CONFIDENCE_THRESHOLDS.LOW) {
      warnings.push({
        type: 'low_confidence',
        severity: 'warning',
        message: 'Analysis confidence is below acceptable threshold',
        details: `Overall confidence: ${(confidenceScore.overall * 100).toFixed(1)}%`,
        suggestedAction: 'Provider review required before clinical use'
      });
    }

    // Hallucination detection
    if (confidenceScore.breakdown.citationVerification < 0.6) {
      warnings.push({
        type: 'hallucination',
        severity: 'error',
        message: 'Potential hallucination detected',
        details: 'Citation verification score is low, indicating possible fabricated information',
        suggestedAction: 'Verify all claims with authoritative sources'
      });
    }

    // Contradiction warning
    if (confidenceScore.breakdown.contradictionCheck < 0.7) {
      warnings.push({
        type: 'contradiction',
        severity: 'warning',
        message: 'Contradictions detected in analysis',
        details: 'Multiple contradictory statements found',
        suggestedAction: 'Review and resolve contradictions before use'
      });
    }

    // Missing data warning
    if (analysis.diagnosis.length === 0) {
      warnings.push({
        type: 'missing_data',
        severity: 'error',
        message: 'No diagnosis provided',
        details: 'Analysis did not produce any diagnostic conclusions',
        suggestedAction: 'Provide additional symptoms or context'
      });
    }

    // Emergency check
    const hasUrgentRecommendations = analysis.recommendations.some(r => r.priority === 'urgent');
    if (hasUrgentRecommendations) {
      warnings.push({
        type: 'emergency',
        severity: 'critical',
        message: 'Urgent medical attention may be required',
        details: 'Analysis indicates potential emergency situation',
        suggestedAction: 'Contact emergency services or seek immediate medical care'
      });
    }

    return warnings;
  }

  /**
   * Determine if analysis requires provider review
   */
  public requiresProviderReview(confidenceScore: ConfidenceScore): boolean {
    return confidenceScore.overall < this.CONFIDENCE_THRESHOLDS.REQUIRES_REVIEW;
  }

  // Helper methods
  private generateConfidenceExplanation(score: number, factors: ConfidenceFactor[]): string {
    const level = score >= 0.85 ? 'HIGH' : score >= 0.70 ? 'MEDIUM' : 'LOW';
    const positiveFactors = factors.filter(f => f.impact === 'positive').map(f => f.name);
    const negativeFactors = factors.filter(f => f.impact === 'negative').map(f => f.name);

    let explanation = `Confidence Level: ${level} (${(score * 100).toFixed(1)}%). `;

    if (positiveFactors.length > 0) {
      explanation += `Strengths: ${positiveFactors.join(', ')}. `;
    }

    if (negativeFactors.length > 0) {
      explanation += `Concerns: ${negativeFactors.join(', ')}. `;
    }

    return explanation;
  }

  private areMutuallyExclusive(condition1: string, condition2: string): boolean {
    // Simplified logic - in production, use knowledge base
    const exclusivePairs = [
      ['hypertension', 'hypotension'],
      ['hyperthyroidism', 'hypothyroidism'],
      ['diabetes type 1', 'diabetes type 2']
    ];

    return exclusivePairs.some(pair =>
      (condition1.toLowerCase().includes(pair[0]) && condition2.toLowerCase().includes(pair[1])) ||
      (condition1.toLowerCase().includes(pair[1]) && condition2.toLowerCase().includes(pair[0]))
    );
  }

  private areContradictoryRecommendations(rec1: string, rec2: string): boolean {
    // Simplified logic - in production, use NLP and knowledge base
    const contradictoryPhrases = [
      ['increase', 'decrease'],
      ['start', 'stop'],
      ['elevate', 'lower']
    ];

    return contradictoryPhrases.some(pair => {
      const has1 = rec1.toLowerCase().includes(pair[0]);
      const has2 = rec2.toLowerCase().includes(pair[1]);
      const reverse1 = rec1.toLowerCase().includes(pair[1]);
      const reverse2 = rec2.toLowerCase().includes(pair[0]);
      return (has1 && has2) || (reverse1 && reverse2);
    });
  }

  /**
   * Load knowledge base entry
   */
  public loadKnowledgeBase(entry: MedicalKnowledgeBase): void {
    this.knowledgeBase.set(entry.condition, entry);
  }
}
