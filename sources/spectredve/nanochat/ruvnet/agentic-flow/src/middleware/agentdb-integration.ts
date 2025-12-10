// AgentDB Learning Integration Middleware
import { MedicalAnalysis, PatientData } from '../types/medical';

export class AgentDBIntegration {
  private learningEnabled: boolean = true;
  private patterns: Map<string, any> = new Map();

  async recordAnalysis(patientData: PatientData, analysis: MedicalAnalysis): Promise<void> {
    if (!this.learningEnabled) return;

    // Record successful analysis pattern
    const pattern = this.extractPattern(patientData, analysis);
    await this.storePattern(pattern);
  }

  async applyLearning(patientData: PatientData): Promise<any> {
    if (!this.learningEnabled) return null;

    // Apply learned patterns to improve analysis
    const similarPatterns = await this.findSimilarPatterns(patientData);
    return this.synthesizeLearning(similarPatterns);
  }

  async trainFromFeedback(analysisId: string, feedback: { correct: boolean; improvements: string[] }): Promise<void> {
    // Train AgentDB from provider feedback
    const pattern = {
      analysisId,
      feedback,
      timestamp: new Date().toISOString(),
    };

    await this.storePattern(pattern);
  }

  enableLearning(): void {
    this.learningEnabled = true;
  }

  disableLearning(): void {
    this.learningEnabled = false;
  }

  isLearningEnabled(): boolean {
    return this.learningEnabled;
  }

  private extractPattern(patientData: PatientData, analysis: MedicalAnalysis): any {
    return {
      symptoms: patientData.symptoms,
      diagnosis: analysis.diagnosis,
      confidence: analysis.confidence,
      verificationScore: analysis.verificationScore,
      timestamp: new Date().toISOString(),
    };
  }

  private async storePattern(pattern: any): Promise<void> {
    const key = `pattern_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.patterns.set(key, pattern);
  }

  private async findSimilarPatterns(patientData: PatientData): Promise<any[]> {
    const similarPatterns: any[] = [];

    for (const [key, pattern] of this.patterns.entries()) {
      if (pattern.symptoms) {
        const similarity = this.calculateSimilarity(patientData.symptoms, pattern.symptoms);
        if (similarity > 0.7) {
          similarPatterns.push({ ...pattern, similarity });
        }
      }
    }

    return similarPatterns.sort((a, b) => b.similarity - a.similarity);
  }

  private calculateSimilarity(symptoms1: string[], symptoms2: string[]): number {
    const set1 = new Set(symptoms1.map(s => s.toLowerCase()));
    const set2 = new Set(symptoms2.map(s => s.toLowerCase()));

    const intersection = new Set([...set1].filter(x => set2.has(x)));
    const union = new Set([...set1, ...set2]);

    return intersection.size / union.size;
  }

  private synthesizeLearning(patterns: any[]): any {
    if (patterns.length === 0) return null;

    return {
      suggestedDiagnoses: patterns.flatMap(p => p.diagnosis),
      averageConfidence: patterns.reduce((sum, p) => sum + p.confidence, 0) / patterns.length,
      patternsFound: patterns.length,
    };
  }
}
