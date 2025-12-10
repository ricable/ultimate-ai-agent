/**
 * Real-time Confidence Monitoring
 * Continuously monitors analysis confidence and flags low-confidence outputs
 */

import type { MedicalAnalysis, ConfidenceMetrics, ValidationIssue } from '../types';

export class ConfidenceMonitor {
  private readonly confidenceThreshold: number;
  private readonly criticalThreshold: number;

  constructor(
    confidenceThreshold: number = 0.8,
    criticalThreshold: number = 0.6
  ) {
    this.confidenceThreshold = confidenceThreshold;
    this.criticalThreshold = criticalThreshold;
  }

  /**
   * Monitor confidence levels in real-time
   */
  monitorConfidence(analysis: MedicalAnalysis): ConfidenceMetrics {
    const metrics: ConfidenceMetrics = {
      overall: analysis.confidence,
      byComponent: {
        diagnosis: this.calculateDiagnosisConfidence(analysis),
        treatment: this.calculateTreatmentConfidence(analysis),
        prognosis: this.calculatePrognosisConfidence(analysis),
      },
      uncertaintyFactors: this.identifyUncertaintyFactors(analysis),
      dataQuality: this.assessDataQuality(analysis),
    };

    // Check for agreement across multiple sources
    metrics.modelAgreement = this.checkModelAgreement(analysis);

    return metrics;
  }

  /**
   * Validate confidence levels and flag issues
   */
  validateConfidence(metrics: ConfidenceMetrics): ValidationIssue[] {
    const issues: ValidationIssue[] = [];

    // Check overall confidence
    if (metrics.overall < this.criticalThreshold) {
      issues.push({
        type: 'low_confidence',
        severity: 'critical',
        description: `Overall confidence (${metrics.overall.toFixed(2)}) is critically low`,
        suggestion: 'Require immediate provider review before acting on analysis',
      });
    } else if (metrics.overall < this.confidenceThreshold) {
      issues.push({
        type: 'low_confidence',
        severity: 'warning',
        description: `Overall confidence (${metrics.overall.toFixed(2)}) is below threshold`,
        suggestion: 'Consider additional verification or provider consultation',
      });
    }

    // Check component-specific confidence
    for (const [component, confidence] of Object.entries(metrics.byComponent)) {
      if (confidence < this.confidenceThreshold) {
        issues.push({
          type: 'low_confidence',
          severity: 'warning',
          description: `${component} confidence (${confidence.toFixed(2)}) is low`,
          suggestion: `Review ${component} analysis with additional sources`,
        });
      }
    }

    // Check data quality
    if (metrics.dataQuality < 0.7) {
      issues.push({
        type: 'missing_citation',
        severity: 'error',
        description: `Data quality (${metrics.dataQuality.toFixed(2)}) is insufficient`,
        suggestion: 'Gather more reliable data before proceeding',
      });
    }

    // Check model agreement
    if (metrics.modelAgreement && metrics.modelAgreement < 0.6) {
      issues.push({
        type: 'inconsistency',
        severity: 'warning',
        description: `Low model agreement (${metrics.modelAgreement.toFixed(2)})`,
        suggestion: 'Multiple models disagree; seek expert consensus',
      });
    }

    // Flag uncertainty factors
    if (metrics.uncertaintyFactors.length > 3) {
      issues.push({
        type: 'low_confidence',
        severity: 'info',
        description: `Multiple uncertainty factors identified: ${metrics.uncertaintyFactors.join(', ')}`,
        suggestion: 'Address uncertainty factors to improve confidence',
      });
    }

    return issues;
  }

  /**
   * Calculate diagnosis confidence
   */
  private calculateDiagnosisConfidence(analysis: MedicalAnalysis): number {
    if (analysis.conditions.length === 0) return 0;

    const avgConfidence = analysis.conditions.reduce(
      (sum, condition) => sum + condition.confidence,
      0
    ) / analysis.conditions.length;

    // Penalize if no ICD-10 codes
    const codePenalty = analysis.conditions.every(c => c.icd10Code) ? 1.0 : 0.9;

    // Penalize if too many differential diagnoses
    const differentialPenalty = analysis.conditions.some(c =>
      c.differential && c.differential.length > 3
    ) ? 0.95 : 1.0;

    return avgConfidence * codePenalty * differentialPenalty;
  }

  /**
   * Calculate treatment confidence
   */
  private calculateTreatmentConfidence(analysis: MedicalAnalysis): number {
    const hasRecommendations = analysis.recommendations.length > 0;
    const hasCitations = analysis.citations.length > 0;
    const citationsVerified = analysis.citations.every(c => c.verified);

    let confidence = 0.5;

    if (hasRecommendations) confidence += 0.2;
    if (hasCitations) confidence += 0.2;
    if (citationsVerified) confidence += 0.1;

    return Math.min(confidence, 1.0);
  }

  /**
   * Calculate prognosis confidence
   */
  private calculatePrognosisConfidence(analysis: MedicalAnalysis): number {
    // Base confidence on severity and urgency alignment
    const severityLevels = analysis.conditions.map(c => c.severity);
    const mostSevere = this.getMostSevere(severityLevels);

    const urgencyAligned = this.checkUrgencyAlignment(mostSevere, analysis.urgencyLevel);

    return urgencyAligned ? 0.9 : 0.7;
  }

  /**
   * Identify factors contributing to uncertainty
   */
  private identifyUncertaintyFactors(analysis: MedicalAnalysis): string[] {
    const factors: string[] = [];

    if (analysis.conditions.length > 3) {
      factors.push('Multiple possible conditions');
    }

    if (analysis.citations.length < 2) {
      factors.push('Limited supporting evidence');
    }

    if (analysis.citations.some(c => !c.verified)) {
      factors.push('Unverified citations');
    }

    if (analysis.conditions.some(c => c.differential && c.differential.length > 2)) {
      factors.push('Broad differential diagnoses');
    }

    if (analysis.confidence < 0.8) {
      factors.push('Low model confidence');
    }

    return factors;
  }

  /**
   * Assess overall data quality
   */
  private assessDataQuality(analysis: MedicalAnalysis): number {
    let score = 0.5;

    // Check citations
    if (analysis.citations.length >= 3) score += 0.2;
    if (analysis.citations.every(c => c.verified)) score += 0.15;
    if (analysis.citations.some(c => c.sourceType === 'clinical_guideline')) score += 0.1;

    // Check completeness
    if (analysis.conditions.every(c => c.icd10Code)) score += 0.05;

    return Math.min(score, 1.0);
  }

  /**
   * Check agreement across multiple models (simulated for now)
   */
  private checkModelAgreement(analysis: MedicalAnalysis): number {
    // In a real implementation, this would compare outputs from multiple models
    // For now, use citation consensus as a proxy
    const consensusCitations = analysis.citations.filter(
      c => c.relevanceScore > 0.8
    ).length;

    return Math.min(consensusCitations / 3, 1.0);
  }

  /**
   * Get most severe condition level
   */
  private getMostSevere(levels: Array<'mild' | 'moderate' | 'severe' | 'critical'>): string {
    if (levels.includes('critical')) return 'critical';
    if (levels.includes('severe')) return 'severe';
    if (levels.includes('moderate')) return 'moderate';
    return 'mild';
  }

  /**
   * Check if urgency aligns with severity
   */
  private checkUrgencyAlignment(
    severity: string,
    urgency: 'routine' | 'urgent' | 'emergency'
  ): boolean {
    const alignments: Record<string, string[]> = {
      critical: ['urgent', 'emergency'],
      severe: ['urgent', 'emergency'],
      moderate: ['routine', 'urgent'],
      mild: ['routine'],
    };

    return alignments[severity]?.includes(urgency) ?? false;
  }
}
