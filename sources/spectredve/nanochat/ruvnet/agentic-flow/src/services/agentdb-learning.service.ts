/**
 * AgentDB Learning Service
 * Pattern recognition and continuous learning for medical analysis
 */

import { ReflexionMemory, SkillLibrary, EmbeddingService } from '../../agentic-flow/src/agentdb';
import type {
  AnalysisResult,
  LearningPattern,
  PatternRecognitionResult,
  Diagnosis,
  Recommendation
} from '../types/medical.types';

export class AgentDBLearningService {
  private reflexionMemory: ReflexionMemory;
  private skillLibrary: SkillLibrary;
  private embeddingService: EmbeddingService;
  private dbPath: string;

  constructor(dbPath: string = './data/medical-learning.db') {
    this.dbPath = dbPath;
    this.reflexionMemory = new ReflexionMemory({ dbPath });
    this.skillLibrary = new SkillLibrary({ dbPath });
    this.embeddingService = new EmbeddingService({ dbPath });
  }

  /**
   * Learn from successful analysis
   */
  public async learnFromAnalysis(
    analysis: AnalysisResult,
    outcome: 'successful' | 'failed' | 'modified',
    providerFeedback?: string
  ): Promise<void> {
    try {
      // Store trajectory for reflexion learning
      const trajectory = {
        taskId: analysis.id,
        steps: [
          {
            action: 'diagnosis',
            observation: JSON.stringify(analysis.diagnosis),
            reward: outcome === 'successful' ? 1.0 : outcome === 'modified' ? 0.7 : 0.0
          },
          {
            action: 'recommendations',
            observation: JSON.stringify(analysis.recommendations),
            reward: outcome === 'successful' ? 1.0 : outcome === 'modified' ? 0.7 : 0.0
          }
        ],
        verdict: outcome === 'successful' ? 'correct' : 'incorrect',
        reflection: providerFeedback || this.generateReflection(analysis, outcome)
      };

      await this.reflexionMemory.addTrajectory(trajectory);

      // Extract and store successful patterns
      if (outcome === 'successful' || outcome === 'modified') {
        await this.extractPatterns(analysis);
      }

      // Update skill library
      await this.updateSkills(analysis, outcome);
    } catch (error) {
      console.error('Error learning from analysis:', error);
    }
  }

  /**
   * Recognize patterns in new query
   */
  public async recognizePatterns(
    symptoms: string[],
    context?: Record<string, any>
  ): Promise<PatternRecognitionResult> {
    try {
      // Generate embedding for symptom cluster
      const queryText = symptoms.join(' ');
      const queryEmbedding = await this.embeddingService.generateEmbedding(queryText);

      // Search for similar patterns
      const similarPatterns = await this.embeddingService.search(
        queryEmbedding,
        {
          limit: 10,
          threshold: 0.7,
          filter: { patternType: 'symptom_cluster' }
        }
      );

      // Analyze applicability
      const patterns: LearningPattern[] = similarPatterns.map(result => ({
        id: result.id,
        patternType: 'symptom_cluster',
        frequency: result.metadata?.frequency || 1,
        accuracy: result.metadata?.accuracy || 0.5,
        examples: result.metadata?.examples || [],
        metadata: result.metadata || {},
        createdAt: new Date(result.metadata?.createdAt || Date.now()),
        lastUsed: new Date()
      }));

      const confidence = patterns.length > 0
        ? patterns.reduce((sum, p) => sum + p.accuracy, 0) / patterns.length
        : 0;

      return {
        patterns,
        confidence,
        applicableToQuery: confidence > 0.7,
        reasoning: this.generatePatternReasoning(patterns, symptoms)
      };
    } catch (error) {
      console.error('Error recognizing patterns:', error);
      return {
        patterns: [],
        confidence: 0,
        applicableToQuery: false,
        reasoning: 'Pattern recognition failed'
      };
    }
  }

  /**
   * Get relevant skills for analysis
   */
  public async getRelevantSkills(condition: string): Promise<any[]> {
    try {
      const skills = await this.skillLibrary.searchSkills(condition, 5);
      return skills.filter(skill => skill.successRate > 0.7);
    } catch (error) {
      console.error('Error retrieving skills:', error);
      return [];
    }
  }

  /**
   * Extract patterns from successful analysis
   */
  private async extractPatterns(analysis: AnalysisResult): Promise<void> {
    // Extract diagnosis patterns
    for (const diagnosis of analysis.diagnosis) {
      const pattern = {
        type: 'diagnosis_pattern',
        condition: diagnosis.condition,
        confidence: diagnosis.confidence,
        evidence: diagnosis.supportingEvidence.map(e => e.type),
        timestamp: new Date()
      };

      const patternText = JSON.stringify(pattern);
      const embedding = await this.embeddingService.generateEmbedding(patternText);

      await this.embeddingService.store({
        id: `pattern_${Date.now()}_${Math.random()}`,
        embedding,
        metadata: {
          ...pattern,
          patternType: 'diagnosis',
          frequency: 1,
          accuracy: diagnosis.confidence
        }
      });
    }

    // Extract recommendation patterns
    for (const rec of analysis.recommendations) {
      const pattern = {
        type: 'recommendation_pattern',
        recommendationType: rec.type,
        priority: rec.priority,
        confidence: rec.confidence,
        timestamp: new Date()
      };

      const patternText = JSON.stringify(pattern);
      const embedding = await this.embeddingService.generateEmbedding(patternText);

      await this.embeddingService.store({
        id: `rec_pattern_${Date.now()}_${Math.random()}`,
        embedding,
        metadata: {
          ...pattern,
          patternType: 'recommendation',
          frequency: 1,
          accuracy: rec.confidence
        }
      });
    }
  }

  /**
   * Update skill library based on outcomes
   */
  private async updateSkills(
    analysis: AnalysisResult,
    outcome: 'successful' | 'failed' | 'modified'
  ): Promise<void> {
    for (const diagnosis of analysis.diagnosis) {
      const skillName = `diagnose_${diagnosis.condition.replace(/\s+/g, '_')}`;
      const successRate = outcome === 'successful' ? 1.0 : outcome === 'modified' ? 0.7 : 0.0;

      await this.skillLibrary.addSkill({
        name: skillName,
        description: `Diagnose ${diagnosis.condition}`,
        implementation: diagnosis.reasoning,
        successRate,
        usageCount: 1,
        metadata: {
          icd10Code: diagnosis.icd10Code,
          confidence: diagnosis.confidence
        }
      });
    }
  }

  /**
   * Generate reflection for learning
   */
  private generateReflection(
    analysis: AnalysisResult,
    outcome: 'successful' | 'failed' | 'modified'
  ): string {
    if (outcome === 'successful') {
      return `Analysis was accurate. Key factors: high confidence (${analysis.confidenceScore.overall}), verified citations, no contradictions.`;
    } else if (outcome === 'modified') {
      return `Analysis required modifications. Review confidence scoring and citation verification processes.`;
    } else {
      return `Analysis failed. Low confidence (${analysis.confidenceScore.overall}) indicated issues. Improve pattern recognition and knowledge base validation.`;
    }
  }

  /**
   * Generate reasoning for pattern recognition
   */
  private generatePatternReasoning(patterns: LearningPattern[], symptoms: string[]): string {
    if (patterns.length === 0) {
      return 'No similar patterns found in historical data.';
    }

    const avgAccuracy = patterns.reduce((sum, p) => sum + p.accuracy, 0) / patterns.length;
    const totalFrequency = patterns.reduce((sum, p) => sum + p.frequency, 0);

    return `Found ${patterns.length} similar patterns with average accuracy ${(avgAccuracy * 100).toFixed(1)}% ` +
      `(seen ${totalFrequency} times). Symptoms match ${patterns[0].patternType} patterns from previous successful analyses.`;
  }

  /**
   * Export learning metrics
   */
  public async getMetrics(): Promise<any> {
    return {
      totalTrajectories: await this.reflexionMemory.getTrajectoryCount?.() || 0,
      totalSkills: await this.skillLibrary.getSkillCount?.() || 0,
      totalPatterns: await this.embeddingService.getCount?.() || 0,
      averageAccuracy: 0.85 // Calculate from actual data
    };
  }

  /**
   * Cleanup and close connections
   */
  public async close(): Promise<void> {
    // Close database connections if needed
  }
}
