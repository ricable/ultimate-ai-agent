/**
 * Severity Classifier
 * Classifies patient query severity based on multiple factors
 */

import { SeverityLevel, SeverityScore } from './types';
import { PatientQuery } from '../providers/types';

export class SeverityClassifier {
  /**
   * Classify severity of patient query
   */
  classify(query: PatientQuery): SeverityScore {
    const symptomScore = this.calculateSymptomSeverity(query);
    const urgencyScore = this.calculateUrgency(query);
    const riskScore = this.calculateRiskFactors(query);
    const historyScore = this.calculatePatientHistoryImpact(query);

    // Weighted scoring
    const totalScore =
      symptomScore * 0.4 +
      urgencyScore * 0.3 +
      riskScore * 0.2 +
      historyScore * 0.1;

    const level = this.scoreToLevel(totalScore);

    console.log(`[SEVERITY_CLASSIFIER] Query ${query.id} classified as ${level} (score: ${totalScore.toFixed(2)})`);

    return {
      totalScore,
      components: {
        symptomSeverity: symptomScore,
        urgency: urgencyScore,
        riskFactors: riskScore,
        patientHistory: historyScore
      },
      level
    };
  }

  /**
   * Calculate symptom severity score
   */
  private calculateSymptomSeverity(query: PatientQuery): number {
    if (!query.symptoms || query.symptoms.length === 0) {
      return 0.3; // Default moderate severity if no symptoms specified
    }

    const severeSymptoms = [
      'chest pain', 'difficulty breathing', 'severe pain',
      'bleeding', 'unconscious', 'seizure', 'severe headache'
    ];

    const moderateSymptoms = [
      'pain', 'fever', 'cough', 'nausea', 'vomiting',
      'dizziness', 'weakness', 'fatigue'
    ];

    let score = 0;
    for (const symptom of query.symptoms) {
      const lowerSymptom = symptom.toLowerCase();

      if (severeSymptoms.some(s => lowerSymptom.includes(s))) {
        score += 0.3;
      } else if (moderateSymptoms.some(s => lowerSymptom.includes(s))) {
        score += 0.15;
      } else {
        score += 0.1;
      }
    }

    return Math.min(score, 1.0);
  }

  /**
   * Calculate urgency score
   */
  private calculateUrgency(query: PatientQuery): number {
    const text = query.description.toLowerCase();

    const urgencyIndicators = [
      { keyword: 'emergency', score: 1.0 },
      { keyword: 'urgent', score: 0.8 },
      { keyword: 'immediately', score: 0.9 },
      { keyword: 'asap', score: 0.7 },
      { keyword: 'right now', score: 0.9 },
      { keyword: 'cant wait', score: 0.8 },
      { keyword: 'getting worse', score: 0.7 }
    ];

    let maxScore = 0;
    for (const indicator of urgencyIndicators) {
      if (text.includes(indicator.keyword)) {
        maxScore = Math.max(maxScore, indicator.score);
      }
    }

    return maxScore;
  }

  /**
   * Calculate risk factors score
   */
  private calculateRiskFactors(query: PatientQuery): number {
    const text = `${query.description} ${query.medicalHistory || ''}`.toLowerCase();

    const riskFactors = [
      'diabetes', 'heart disease', 'cancer', 'immunocompromised',
      'elderly', 'pregnant', 'hypertension', 'asthma',
      'chronic', 'allergies'
    ];

    let riskCount = 0;
    for (const risk of riskFactors) {
      if (text.includes(risk)) {
        riskCount++;
      }
    }

    // Cap at 3 risk factors for scoring
    return Math.min(riskCount / 3, 1.0) * 0.8;
  }

  /**
   * Calculate patient history impact
   */
  private calculatePatientHistoryImpact(query: PatientQuery): number {
    if (!query.medicalHistory) {
      return 0.3;
    }

    const history = query.medicalHistory.toLowerCase();

    // Check for previous similar issues
    if (history.includes('recurring') || history.includes('chronic')) {
      return 0.7;
    }

    // Check for recent hospitalizations
    if (history.includes('hospitalized') || history.includes('admitted')) {
      return 0.8;
    }

    // Check for multiple conditions
    const conditionCount = (history.match(/diagnosed/g) || []).length;
    return Math.min(0.3 + (conditionCount * 0.15), 1.0);
  }

  /**
   * Convert score to severity level
   */
  private scoreToLevel(score: number): SeverityLevel {
    if (score >= 0.8) return SeverityLevel.CRITICAL;
    if (score >= 0.6) return SeverityLevel.HIGH;
    if (score >= 0.4) return SeverityLevel.MODERATE;
    return SeverityLevel.LOW;
  }

  /**
   * Get severity explanation
   */
  getExplanation(score: SeverityScore): string {
    const parts: string[] = [];

    if (score.components.symptomSeverity > 0.7) {
      parts.push('severe symptoms detected');
    }

    if (score.components.urgency > 0.7) {
      parts.push('high urgency indicated');
    }

    if (score.components.riskFactors > 0.6) {
      parts.push('significant risk factors present');
    }

    if (score.components.patientHistory > 0.6) {
      parts.push('relevant medical history');
    }

    const explanation = parts.length > 0
      ? `Classified as ${score.level}: ${parts.join(', ')}`
      : `Classified as ${score.level} based on standard assessment`;

    return explanation;
  }
}
