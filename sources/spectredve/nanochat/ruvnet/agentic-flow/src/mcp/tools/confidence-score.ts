/**
 * Confidence Score Tool
 * Calculates and reports confidence scores for medical analyses
 */

import type { MedicalAnalysis, ConfidenceMetrics, MCPToolResponse } from '../types';
import { ConfidenceMonitor } from '../anti-hallucination/confidence-monitor';

export class ConfidenceScoreTool {
  private readonly monitor: ConfidenceMonitor;

  constructor() {
    this.monitor = new ConfidenceMonitor(0.8, 0.6);
  }

  /**
   * Calculate confidence scores
   */
  async execute(args: {
    analysis: MedicalAnalysis;
    detailedBreakdown?: boolean;
  }): Promise<MCPToolResponse> {
    try {
      const metrics = this.monitor.monitorConfidence(args.analysis);
      const issues = this.monitor.validateConfidence(metrics);

      const score = this.calculateCompositeScore(metrics);
      const grade = this.gradeConfidence(score);
      const interpretation = this.interpretConfidence(score, issues);

      const response = {
        score,
        grade,
        metrics,
        issues,
        interpretation,
        breakdown: args.detailedBreakdown ? this.generateDetailedBreakdown(metrics, args.analysis) : undefined,
      };

      return {
        content: [
          {
            type: 'json',
            json: response,
          },
          {
            type: 'text',
            text: this.formatConfidenceReport(response),
          },
        ],
      };
    } catch (error: any) {
      return {
        content: [
          {
            type: 'text',
            text: `❌ Confidence calculation failed: ${error.message}`,
          },
        ],
        isError: true,
      };
    }
  }

  /**
   * Calculate composite confidence score
   */
  private calculateCompositeScore(metrics: ConfidenceMetrics): number {
    // Weighted average of components
    const weights = {
      overall: 0.4,
      diagnosis: 0.25,
      treatment: 0.2,
      prognosis: 0.15,
    };

    const score =
      metrics.overall * weights.overall +
      metrics.byComponent.diagnosis * weights.diagnosis +
      metrics.byComponent.treatment * weights.treatment +
      metrics.byComponent.prognosis * weights.prognosis;

    // Adjust for data quality
    const adjustedScore = score * (0.5 + metrics.dataQuality * 0.5);

    // Penalty for uncertainty factors
    const uncertaintyPenalty = Math.max(0, (metrics.uncertaintyFactors.length - 2) * 0.05);

    return Math.max(0, Math.min(1, adjustedScore - uncertaintyPenalty));
  }

  /**
   * Grade confidence level
   */
  private gradeConfidence(score: number): string {
    if (score >= 0.9) return 'A+ (Excellent)';
    if (score >= 0.85) return 'A (Very High)';
    if (score >= 0.8) return 'B+ (High)';
    if (score >= 0.75) return 'B (Good)';
    if (score >= 0.7) return 'C+ (Acceptable)';
    if (score >= 0.65) return 'C (Fair)';
    if (score >= 0.6) return 'D (Marginal)';
    return 'F (Insufficient)';
  }

  /**
   * Interpret confidence score
   */
  private interpretConfidence(score: number, issues: any[]): string {
    if (score >= 0.9) {
      return 'Excellent confidence. Analysis is well-supported and reliable for clinical decision-making.';
    }

    if (score >= 0.8) {
      return 'High confidence. Analysis is reliable with minor reservations. Suitable for most clinical applications.';
    }

    if (score >= 0.7) {
      return 'Good confidence. Analysis is generally reliable but may benefit from additional verification.';
    }

    if (score >= 0.6) {
      return 'Acceptable confidence. Analysis requires careful review and corroboration before clinical use.';
    }

    return 'Insufficient confidence. Analysis should not be used for clinical decisions without significant additional verification and expert review.';
  }

  /**
   * Generate detailed breakdown
   */
  private generateDetailedBreakdown(
    metrics: ConfidenceMetrics,
    analysis: MedicalAnalysis
  ): any {
    return {
      overall: {
        score: metrics.overall,
        factors: [
          {
            name: 'Model Confidence',
            value: metrics.overall,
            weight: 0.4,
            contribution: metrics.overall * 0.4,
          },
          {
            name: 'Data Quality',
            value: metrics.dataQuality,
            weight: 0.3,
            contribution: metrics.dataQuality * 0.3,
          },
          {
            name: 'Citation Quality',
            value: this.assessCitationQuality(analysis),
            weight: 0.2,
            contribution: this.assessCitationQuality(analysis) * 0.2,
          },
          {
            name: 'Completeness',
            value: this.assessCompleteness(analysis),
            weight: 0.1,
            contribution: this.assessCompleteness(analysis) * 0.1,
          },
        ],
      },
      byComponent: {
        diagnosis: {
          score: metrics.byComponent.diagnosis,
          factors: [
            `${analysis.conditions.length} conditions identified`,
            `Average condition confidence: ${(analysis.conditions.reduce((sum, c) => sum + c.confidence, 0) / analysis.conditions.length * 100).toFixed(1)}%`,
            `ICD-10 coding: ${analysis.conditions.filter(c => c.icd10Code).length}/${analysis.conditions.length}`,
          ],
        },
        treatment: {
          score: metrics.byComponent.treatment,
          factors: [
            `${analysis.recommendations.length} recommendations provided`,
            `${analysis.citations.length} citations supporting recommendations`,
            `Citation verification: ${analysis.citations.filter(c => c.verified).length}/${analysis.citations.length}`,
          ],
        },
        prognosis: {
          score: metrics.byComponent.prognosis,
          factors: [
            `Urgency level: ${analysis.urgencyLevel}`,
            `Severity range: ${this.getSeverityRange(analysis)}`,
            `Provider review ${analysis.requiresProviderReview ? 'required' : 'not required'}`,
          ],
        },
      },
      uncertaintyFactors: metrics.uncertaintyFactors.map(factor => ({
        factor,
        impact: 'reduces confidence by ~5%',
        mitigation: this.suggestMitigation(factor),
      })),
      modelAgreement: metrics.modelAgreement ? {
        score: metrics.modelAgreement,
        interpretation: this.interpretModelAgreement(metrics.modelAgreement),
      } : undefined,
    };
  }

  /**
   * Assess citation quality
   */
  private assessCitationQuality(analysis: MedicalAnalysis): number {
    if (analysis.citations.length === 0) return 0;

    const verifiedRatio = analysis.citations.filter(c => c.verified).length / analysis.citations.length;
    const avgRelevance = analysis.citations.reduce((sum, c) => sum + c.relevanceScore, 0) / analysis.citations.length;

    return (verifiedRatio + avgRelevance) / 2;
  }

  /**
   * Assess completeness
   */
  private assessCompleteness(analysis: MedicalAnalysis): number {
    let score = 0.5;

    if (analysis.conditions.length > 0) score += 0.1;
    if (analysis.recommendations.length > 0) score += 0.1;
    if (analysis.citations.length >= 2) score += 0.1;
    if (analysis.conditions.every(c => c.icd10Code)) score += 0.1;
    if (analysis.conditions.some(c => c.differential && c.differential.length > 0)) score += 0.1;

    return Math.min(1, score);
  }

  /**
   * Get severity range
   */
  private getSeverityRange(analysis: MedicalAnalysis): string {
    const severities = analysis.conditions.map(c => c.severity);
    const uniqueSeverities = [...new Set(severities)];
    return uniqueSeverities.join(', ');
  }

  /**
   * Suggest mitigation for uncertainty
   */
  private suggestMitigation(factor: string): string {
    const mitigations: Record<string, string> = {
      'Multiple possible conditions': 'Conduct additional diagnostic tests to narrow differential',
      'Limited supporting evidence': 'Search for additional citations from trusted sources',
      'Unverified citations': 'Verify citations against original sources',
      'Broad differential diagnoses': 'Gather more specific clinical data',
      'Low model confidence': 'Seek expert consultation or second opinion',
    };

    return mitigations[factor] || 'Consult with healthcare professional';
  }

  /**
   * Interpret model agreement
   */
  private interpretModelAgreement(score: number): string {
    if (score >= 0.9) return 'Excellent agreement across models';
    if (score >= 0.75) return 'Good agreement with minor variations';
    if (score >= 0.6) return 'Moderate agreement, some disagreement exists';
    return 'Low agreement, significant model disagreement';
  }

  /**
   * Format confidence report
   */
  private formatConfidenceReport(response: any): string {
    let report = '📊 Confidence Score Report\n\n';

    report += `🎯 Overall Score: ${(response.score * 100).toFixed(1)}% (${response.grade})\n\n`;
    report += `📝 Interpretation:\n${response.interpretation}\n\n`;

    report += `📈 Component Scores:\n`;
    report += `  • Overall: ${(response.metrics.overall * 100).toFixed(1)}%\n`;
    report += `  • Diagnosis: ${(response.metrics.byComponent.diagnosis * 100).toFixed(1)}%\n`;
    report += `  • Treatment: ${(response.metrics.byComponent.treatment * 100).toFixed(1)}%\n`;
    report += `  • Prognosis: ${(response.metrics.byComponent.prognosis * 100).toFixed(1)}%\n`;
    report += `  • Data Quality: ${(response.metrics.dataQuality * 100).toFixed(1)}%\n`;

    if (response.metrics.modelAgreement) {
      report += `  • Model Agreement: ${(response.metrics.modelAgreement * 100).toFixed(1)}%\n`;
    }

    if (response.metrics.uncertaintyFactors.length > 0) {
      report += `\n⚠️  Uncertainty Factors (${response.metrics.uncertaintyFactors.length}):\n`;
      for (const factor of response.metrics.uncertaintyFactors) {
        report += `  • ${factor}\n`;
      }
    }

    if (response.issues.length > 0) {
      report += `\n🔍 Issues Detected (${response.issues.length}):\n`;
      const critical = response.issues.filter((i: any) => i.severity === 'critical');
      const errors = response.issues.filter((i: any) => i.severity === 'error');
      const warnings = response.issues.filter((i: any) => i.severity === 'warning');

      if (critical.length > 0) report += `  🚨 Critical: ${critical.length}\n`;
      if (errors.length > 0) report += `  ❌ Errors: ${errors.length}\n`;
      if (warnings.length > 0) report += `  ⚠️  Warnings: ${warnings.length}\n`;
    }

    return report;
  }
}
