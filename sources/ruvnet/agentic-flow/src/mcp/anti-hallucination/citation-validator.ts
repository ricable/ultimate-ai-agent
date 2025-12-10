/**
 * Citation Validation
 * Validates medical citations against trusted sources
 */

import type { Citation, ValidationResult, ValidationIssue } from '../types';

export class CitationValidator {
  private readonly trustedSources: Set<string>;
  private readonly minimumYear: number;

  constructor() {
    // Trusted medical sources
    this.trustedSources = new Set([
      'PubMed',
      'Cochrane Library',
      'NICE Guidelines',
      'UpToDate',
      'New England Journal of Medicine',
      'The Lancet',
      'JAMA',
      'BMJ',
      'CDC',
      'WHO',
      'FDA',
      'American Medical Association',
      'Mayo Clinic',
    ]);

    // Minimum acceptable year for citations (last 10 years)
    this.minimumYear = new Date().getFullYear() - 10;
  }

  /**
   * Validate a single citation
   */
  validateCitation(citation: Citation): ValidationResult {
    const issues: ValidationIssue[] = [];

    // Check source trustworthiness
    if (!this.isTrustedSource(citation.source)) {
      issues.push({
        type: 'missing_citation',
        severity: 'warning',
        description: `Source "${citation.source}" is not in trusted sources list`,
        suggestion: 'Verify citation from trusted medical databases',
      });
    }

    // Check citation age
    if (citation.year && citation.year < this.minimumYear) {
      issues.push({
        type: 'missing_citation',
        severity: 'info',
        description: `Citation is from ${citation.year}, may be outdated`,
        suggestion: 'Consider more recent sources for current best practices',
      });
    }

    // Check for required fields
    if (!citation.url && !citation.excerpt) {
      issues.push({
        type: 'missing_citation',
        severity: 'error',
        description: 'Citation lacks both URL and excerpt',
        suggestion: 'Provide URL or excerpt for verification',
      });
    }

    // Check relevance score
    if (citation.relevanceScore < 0.7) {
      issues.push({
        type: 'low_confidence',
        severity: 'warning',
        description: `Low relevance score (${citation.relevanceScore.toFixed(2)})`,
        suggestion: 'Find more relevant citations to support claims',
      });
    }

    // Check verification status
    if (!citation.verified) {
      issues.push({
        type: 'missing_citation',
        severity: 'error',
        description: 'Citation has not been verified',
        suggestion: 'Verify citation against original source',
      });
    }

    const isValid = issues.every(issue => issue.severity !== 'error' && issue.severity !== 'critical');
    const confidence = this.calculateCitationConfidence(citation, issues);

    return {
      isValid,
      confidence,
      issues,
      recommendations: this.generateRecommendations(issues),
    };
  }

  /**
   * Validate multiple citations for consistency
   */
  validateCitations(citations: Citation[]): ValidationResult {
    const issues: ValidationIssue[] = [];

    // Check minimum citation count
    if (citations.length < 2) {
      issues.push({
        type: 'missing_citation',
        severity: 'error',
        description: 'Insufficient citations (minimum 2 required)',
        suggestion: 'Add more citations from trusted sources',
      });
    }

    // Check for citation diversity
    const uniqueSources = new Set(citations.map(c => c.source));
    if (uniqueSources.size < Math.min(2, citations.length)) {
      issues.push({
        type: 'inconsistency',
        severity: 'warning',
        description: 'Citations lack source diversity',
        suggestion: 'Include citations from multiple independent sources',
      });
    }

    // Check for consistency across citations
    const verifiedCount = citations.filter(c => c.verified).length;
    if (verifiedCount < citations.length * 0.8) {
      issues.push({
        type: 'missing_citation',
        severity: 'error',
        description: `Only ${verifiedCount}/${citations.length} citations verified`,
        suggestion: 'Verify all citations before use',
      });
    }

    // Check average relevance
    const avgRelevance = citations.reduce((sum, c) => sum + c.relevanceScore, 0) / citations.length;
    if (avgRelevance < 0.75) {
      issues.push({
        type: 'low_confidence',
        severity: 'warning',
        description: `Average relevance score (${avgRelevance.toFixed(2)}) is low`,
        suggestion: 'Find more relevant citations',
      });
    }

    // Validate each citation individually
    for (const citation of citations) {
      const result = this.validateCitation(citation);
      issues.push(...result.issues);
    }

    const isValid = issues.every(issue => issue.severity !== 'error' && issue.severity !== 'critical');
    const confidence = this.calculateOverallConfidence(citations, issues);

    return {
      isValid,
      confidence,
      issues,
      recommendations: this.generateRecommendations(issues),
    };
  }

  /**
   * Verify citation against source (simulated)
   */
  async verifyCitationSource(citation: Citation): Promise<boolean> {
    // In production, this would:
    // 1. Fetch the original source
    // 2. Compare the excerpt or title
    // 3. Verify authors and publication details
    // 4. Check for retractions

    // Simulated verification
    return new Promise((resolve) => {
      setTimeout(() => {
        const isVerified =
          this.isTrustedSource(citation.source) &&
          citation.relevanceScore > 0.7 &&
          (!citation.year || citation.year >= this.minimumYear);
        resolve(isVerified);
      }, 100);
    });
  }

  /**
   * Check if source is trusted
   */
  private isTrustedSource(source: string): boolean {
    return Array.from(this.trustedSources).some(trusted =>
      source.toLowerCase().includes(trusted.toLowerCase())
    );
  }

  /**
   * Calculate citation confidence
   */
  private calculateCitationConfidence(citation: Citation, issues: ValidationIssue[]): number {
    let confidence = 1.0;

    // Penalize for issues
    for (const issue of issues) {
      switch (issue.severity) {
        case 'critical':
          confidence -= 0.4;
          break;
        case 'error':
          confidence -= 0.2;
          break;
        case 'warning':
          confidence -= 0.1;
          break;
        case 'info':
          confidence -= 0.05;
          break;
      }
    }

    // Boost for trusted source
    if (this.isTrustedSource(citation.source)) {
      confidence += 0.1;
    }

    // Boost for verification
    if (citation.verified) {
      confidence += 0.1;
    }

    return Math.max(0, Math.min(1.0, confidence));
  }

  /**
   * Calculate overall confidence across citations
   */
  private calculateOverallConfidence(citations: Citation[], issues: ValidationIssue[]): number {
    if (citations.length === 0) return 0;

    const avgCitationScore = citations.reduce(
      (sum, c) => sum + c.relevanceScore,
      0
    ) / citations.length;

    const verificationRate = citations.filter(c => c.verified).length / citations.length;

    let confidence = (avgCitationScore + verificationRate) / 2;

    // Penalize for critical/error issues
    const criticalCount = issues.filter(i => i.severity === 'critical').length;
    const errorCount = issues.filter(i => i.severity === 'error').length;

    confidence -= criticalCount * 0.3;
    confidence -= errorCount * 0.15;

    return Math.max(0, Math.min(1.0, confidence));
  }

  /**
   * Generate recommendations based on issues
   */
  private generateRecommendations(issues: ValidationIssue[]): string[] {
    const recommendations = new Set<string>();

    for (const issue of issues) {
      if (issue.suggestion) {
        recommendations.add(issue.suggestion);
      }
    }

    // Add general recommendations
    if (issues.some(i => i.type === 'missing_citation')) {
      recommendations.add('Add citations from peer-reviewed medical literature');
    }

    if (issues.some(i => i.type === 'low_confidence')) {
      recommendations.add('Seek additional supporting evidence');
    }

    return Array.from(recommendations);
  }
}
