/**
 * AgentDB Integration for Medical MCP
 * Enables pattern learning and experience tracking for medical analyses
 */

import type { MedicalAnalysis, LearningFeedback, AgentDBPattern } from './types';

export class AgentDBIntegration {
  private readonly patterns: Map<string, AgentDBPattern>;
  private readonly feedback: Map<string, LearningFeedback>;

  constructor() {
    this.patterns = new Map();
    this.feedback = new Map();
  }

  /**
   * Store analysis pattern for learning
   */
  async storeAnalysisPattern(analysis: MedicalAnalysis): Promise<void> {
    // In production, integrate with actual AgentDB
    // using packages/agentdb/src/index.ts

    const pattern: AgentDBPattern = {
      taskType: 'medical_analysis',
      approach: this.summarizeApproach(analysis),
      successRate: analysis.providerApproved ? 1.0 : 0.5,
      tags: this.generateTags(analysis),
      metadata: {
        analysisId: analysis.id,
        timestamp: analysis.timestamp,
        conditions: analysis.conditions.map(c => c.name),
        urgencyLevel: analysis.urgencyLevel,
        confidence: analysis.confidence,
      },
    };

    this.patterns.set(analysis.id, pattern);

    // Would call AgentDB in production:
    // const { ReasoningBank } = await import('../../packages/agentdb/src/index.js');
    // const reasoningBank = new ReasoningBank(db, embeddingService);
    // await reasoningBank.storePattern(pattern);
  }

  /**
   * Record provider feedback for learning
   */
  async recordFeedback(
    analysisId: string,
    accuracy: number,
    providerFeedback: string,
    corrections?: string[]
  ): Promise<void> {
    const feedback: LearningFeedback = {
      analysisId,
      accuracy,
      providerFeedback,
      corrections,
      timestamp: Date.now(),
    };

    this.feedback.set(analysisId, feedback);

    // Update pattern success rate
    const pattern = this.patterns.get(analysisId);
    if (pattern) {
      pattern.successRate = accuracy;
      await this.updatePattern(pattern);
    }

    // Would call AgentDB learning system in production:
    // const { LearningSystem } = await import('../../packages/agentdb/src/index.js');
    // await learningSystem.submitFeedback({...});
  }

  /**
   * Search for similar analysis patterns
   */
  async findSimilarPatterns(
    symptoms: string[],
    k: number = 5
  ): Promise<AgentDBPattern[]> {
    // In production, use AgentDB vector search
    // For now, simple matching

    const symptomText = symptoms.join(' ').toLowerCase();
    const matches: Array<{ pattern: AgentDBPattern; similarity: number }> = [];

    for (const pattern of this.patterns.values()) {
      const patternConditions = (pattern.metadata.conditions as string[] || []).join(' ').toLowerCase();
      const similarity = this.calculateSimilarity(symptomText, patternConditions);

      if (similarity > 0.5) {
        matches.push({ pattern, similarity });
      }
    }

    // Sort by similarity and success rate
    matches.sort((a, b) => {
      const scoreA = a.similarity * 0.7 + a.pattern.successRate * 0.3;
      const scoreB = b.similarity * 0.7 + b.pattern.successRate * 0.3;
      return scoreB - scoreA;
    });

    return matches.slice(0, k).map(m => m.pattern);

    // Would use AgentDB in production:
    // const { ReasoningBank } = await import('../../packages/agentdb/src/index.js');
    // return await reasoningBank.searchPatterns({ task: symptomText, k });
  }

  /**
   * Get learning metrics
   */
  async getLearningMetrics(): Promise<{
    totalAnalyses: number;
    avgAccuracy: number;
    patternsLearned: number;
    topConditions: Array<{ name: string; count: number }>;
  }> {
    const feedbackArray = Array.from(this.feedback.values());
    const avgAccuracy =
      feedbackArray.length > 0
        ? feedbackArray.reduce((sum, f) => sum + f.accuracy, 0) / feedbackArray.length
        : 0;

    // Count conditions
    const conditionCounts = new Map<string, number>();
    for (const pattern of this.patterns.values()) {
      const conditions = pattern.metadata.conditions as string[] || [];
      for (const condition of conditions) {
        conditionCounts.set(condition, (conditionCounts.get(condition) || 0) + 1);
      }
    }

    const topConditions = Array.from(conditionCounts.entries())
      .map(([name, count]) => ({ name, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 10);

    return {
      totalAnalyses: this.patterns.size,
      avgAccuracy,
      patternsLearned: this.patterns.size,
      topConditions,
    };
  }

  /**
   * Summarize analysis approach
   */
  private summarizeApproach(analysis: MedicalAnalysis): string {
    const conditionNames = analysis.conditions.map(c => c.name).join(', ');
    return `Analyzed ${analysis.conditions.length} conditions (${conditionNames}) with ${analysis.citations.length} citations, confidence ${(analysis.confidence * 100).toFixed(1)}%`;
  }

  /**
   * Generate tags for pattern
   */
  private generateTags(analysis: MedicalAnalysis): string[] {
    const tags: string[] = [
      'medical_analysis',
      `urgency_${analysis.urgencyLevel}`,
      `confidence_${analysis.confidence >= 0.8 ? 'high' : analysis.confidence >= 0.6 ? 'medium' : 'low'}`,
    ];

    // Add condition-based tags
    for (const condition of analysis.conditions) {
      tags.push(`severity_${condition.severity}`);
      if (condition.icd10Code) {
        tags.push(`icd10_${condition.icd10Code.substring(0, 3)}`);
      }
    }

    return tags;
  }

  /**
   * Update pattern in storage
   */
  private async updatePattern(pattern: AgentDBPattern): Promise<void> {
    this.patterns.set(pattern.metadata.analysisId as string, pattern);

    // Would update in AgentDB in production
  }

  /**
   * Calculate text similarity (simple Jaccard)
   */
  private calculateSimilarity(text1: string, text2: string): number {
    const words1 = new Set(text1.split(/\s+/));
    const words2 = new Set(text2.split(/\s+/));

    const intersection = new Set([...words1].filter(w => words2.has(w)));
    const union = new Set([...words1, ...words2]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }

  /**
   * Export patterns for analysis
   */
  exportPatterns(): AgentDBPattern[] {
    return Array.from(this.patterns.values());
  }

  /**
   * Import patterns from external source
   */
  importPatterns(patterns: AgentDBPattern[]): void {
    for (const pattern of patterns) {
      this.patterns.set(pattern.metadata.analysisId as string, pattern);
    }
  }
}
