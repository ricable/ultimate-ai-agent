/**
 * Emergency Detector
 * Detects emergency conditions from patient queries
 */

import { EmergencyType, EmergencySignal } from './types';
import { PatientQuery } from '../providers/types';

export class EmergencyDetector {
  private emergencyKeywords: EmergencySignal[];
  private urgentKeywords: EmergencySignal[];

  constructor() {
    this.initializeKeywords();
  }

  /**
   * Initialize emergency and urgent keywords
   */
  private initializeKeywords(): void {
    this.emergencyKeywords = [
      // Life-threatening symptoms
      { keyword: 'chest pain', weight: 1.0, category: 'symptom' },
      { keyword: 'difficulty breathing', weight: 1.0, category: 'symptom' },
      { keyword: 'unconscious', weight: 1.0, category: 'symptom' },
      { keyword: 'severe bleeding', weight: 1.0, category: 'symptom' },
      { keyword: 'heart attack', weight: 1.0, category: 'condition' },
      { keyword: 'stroke', weight: 1.0, category: 'condition' },
      { keyword: 'seizure', weight: 0.9, category: 'symptom' },
      { keyword: 'anaphylaxis', weight: 1.0, category: 'condition' },
      { keyword: 'poisoning', weight: 0.9, category: 'condition' },
      { keyword: 'severe head injury', weight: 1.0, category: 'symptom' },
      // Urgent indicators
      { keyword: 'emergency', weight: 0.8, category: 'urgency' },
      { keyword: 'urgent', weight: 0.7, category: 'urgency' },
      { keyword: 'immediately', weight: 0.7, category: 'urgency' },
      { keyword: 'cant breathe', weight: 1.0, category: 'symptom' },
      { keyword: 'choking', weight: 1.0, category: 'symptom' }
    ];

    this.urgentKeywords = [
      { keyword: 'high fever', weight: 0.7, category: 'symptom' },
      { keyword: 'severe pain', weight: 0.8, category: 'symptom' },
      { keyword: 'bleeding', weight: 0.6, category: 'symptom' },
      { keyword: 'broken bone', weight: 0.7, category: 'condition' },
      { keyword: 'infection', weight: 0.5, category: 'condition' },
      { keyword: 'dehydrated', weight: 0.6, category: 'symptom' },
      { keyword: 'burn', weight: 0.6, category: 'symptom' },
      { keyword: 'allergic reaction', weight: 0.7, category: 'condition' },
      { keyword: 'asap', weight: 0.6, category: 'urgency' },
      { keyword: 'as soon as possible', weight: 0.6, category: 'urgency' }
    ];
  }

  /**
   * Detect emergency type from query
   */
  detect(query: PatientQuery): EmergencyType {
    const text = `${query.description} ${query.symptoms?.join(' ') || ''}`.toLowerCase();

    // Check for life-threatening conditions
    const emergencyScore = this.calculateKeywordScore(text, this.emergencyKeywords);
    if (emergencyScore >= 0.8) {
      console.log(`[EMERGENCY_DETECTOR] Life-threatening condition detected in query ${query.id}`);
      return EmergencyType.LIFE_THREATENING;
    }

    // Check for urgent care
    const urgentScore = this.calculateKeywordScore(text, this.urgentKeywords);
    if (urgentScore >= 0.5 || emergencyScore >= 0.5) {
      console.log(`[EMERGENCY_DETECTOR] Urgent care needed for query ${query.id}`);
      return EmergencyType.URGENT_CARE;
    }

    return EmergencyType.ROUTINE;
  }

  /**
   * Calculate keyword match score
   */
  private calculateKeywordScore(text: string, keywords: EmergencySignal[]): number {
    let totalWeight = 0;
    let matchedWeight = 0;

    for (const signal of keywords) {
      totalWeight += signal.weight;
      if (text.includes(signal.keyword)) {
        matchedWeight += signal.weight;
      }
    }

    return totalWeight > 0 ? matchedWeight / totalWeight : 0;
  }

  /**
   * Get matched emergency signals
   */
  getMatchedSignals(query: PatientQuery): EmergencySignal[] {
    const text = `${query.description} ${query.symptoms?.join(' ') || ''}`.toLowerCase();
    const matched: EmergencySignal[] = [];

    const allKeywords = [...this.emergencyKeywords, ...this.urgentKeywords];

    for (const signal of allKeywords) {
      if (text.includes(signal.keyword)) {
        matched.push(signal);
      }
    }

    return matched.sort((a, b) => b.weight - a.weight);
  }

  /**
   * Add custom emergency keyword
   */
  addEmergencyKeyword(signal: EmergencySignal): void {
    this.emergencyKeywords.push(signal);
  }

  /**
   * Add custom urgent keyword
   */
  addUrgentKeyword(signal: EmergencySignal): void {
    this.urgentKeywords.push(signal);
  }
}
