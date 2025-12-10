/**
 * Escalation Router
 * Handles query escalation workflows and routing decisions
 */

import { RoutingDecision, SeverityLevel, EmergencyType } from './types';
import { PatientQuery } from '../providers/types';
import { Provider } from '../providers/types';
import { EmergencyDetector } from './emergency-detector';
import { SeverityClassifier } from './severity-classifier';
import { ProviderMatcher } from './provider-matcher';

export class EscalationRouter {
  private emergencyDetector: EmergencyDetector;
  private severityClassifier: SeverityClassifier;
  private providerMatcher: ProviderMatcher;

  constructor() {
    this.emergencyDetector = new EmergencyDetector();
    this.severityClassifier = new SeverityClassifier();
    this.providerMatcher = new ProviderMatcher();
  }

  /**
   * Make routing decision for query
   */
  async route(
    query: PatientQuery,
    availableProviders: Provider[]
  ): Promise<RoutingDecision> {
    // Detect emergency type
    const emergencyType = this.emergencyDetector.detect(query);

    // Classify severity
    const severityScore = this.severityClassifier.classify(query);

    // Find matching providers
    const matches = await this.providerMatcher.findMatches(query, availableProviders, 5);

    // Determine if escalation is required
    const escalationRequired = this.shouldEscalate(severityScore.level, emergencyType, matches);

    // Select best provider
    const bestMatch = matches.length > 0 ? matches[0] : undefined;

    // Calculate estimated response time
    const estimatedResponseTime = this.calculateResponseTime(
      emergencyType,
      severityScore.level,
      bestMatch?.estimatedWaitTime || 60
    );

    // Generate reasoning
    const reasoning = this.generateReasoning(
      emergencyType,
      severityScore,
      matches.length,
      escalationRequired
    );

    // Calculate confidence
    const confidence = this.calculateConfidence(matches, severityScore);

    const decision: RoutingDecision = {
      queryId: query.id,
      severity: severityScore.level,
      emergencyType,
      recommendedProviderId: bestMatch?.providerId,
      alternativeProviders: matches.slice(1).map(m => m.providerId),
      escalationRequired,
      estimatedResponseTime,
      reasoning,
      confidence
    };

    console.log(`[ESCALATION_ROUTER] Routing decision for query ${query.id}:`);
    console.log(`  Emergency Type: ${emergencyType}`);
    console.log(`  Severity: ${severityScore.level}`);
    console.log(`  Recommended Provider: ${decision.recommendedProviderId || 'None'}`);
    console.log(`  Escalation Required: ${escalationRequired}`);

    return decision;
  }

  /**
   * Determine if escalation is required
   */
  private shouldEscalate(
    severity: SeverityLevel,
    emergencyType: EmergencyType,
    matches: any[]
  ): boolean {
    // Always escalate life-threatening emergencies
    if (emergencyType === EmergencyType.LIFE_THREATENING) {
      return true;
    }

    // Escalate critical severity
    if (severity === SeverityLevel.CRITICAL) {
      return true;
    }

    // Escalate if no suitable providers available
    if (matches.length === 0) {
      return true;
    }

    // Escalate if best match score is too low for high severity
    if (severity === SeverityLevel.HIGH && matches[0].matchScore < 0.6) {
      return true;
    }

    return false;
  }

  /**
   * Calculate estimated response time
   */
  private calculateResponseTime(
    emergencyType: EmergencyType,
    severity: SeverityLevel,
    providerWaitTime: number
  ): number {
    let baseTime = providerWaitTime;

    // Adjust based on emergency type
    if (emergencyType === EmergencyType.LIFE_THREATENING) {
      baseTime = Math.min(baseTime, 5); // Maximum 5 minutes
    } else if (emergencyType === EmergencyType.URGENT_CARE) {
      baseTime = Math.min(baseTime, 15); // Maximum 15 minutes
    }

    // Adjust based on severity
    const severityMultipliers = {
      [SeverityLevel.CRITICAL]: 0.5,
      [SeverityLevel.HIGH]: 0.7,
      [SeverityLevel.MODERATE]: 1.0,
      [SeverityLevel.LOW]: 1.5
    };

    baseTime *= severityMultipliers[severity];

    return Math.round(baseTime);
  }

  /**
   * Generate routing reasoning
   */
  private generateReasoning(
    emergencyType: EmergencyType,
    severityScore: any,
    matchCount: number,
    escalationRequired: boolean
  ): string {
    const parts: string[] = [];

    // Emergency type reasoning
    if (emergencyType === EmergencyType.LIFE_THREATENING) {
      parts.push('Life-threatening emergency detected requiring immediate attention');
    } else if (emergencyType === EmergencyType.URGENT_CARE) {
      parts.push('Urgent care needed based on symptoms');
    } else {
      parts.push('Routine consultation appropriate');
    }

    // Severity reasoning
    parts.push(this.severityClassifier.getExplanation(severityScore));

    // Provider availability reasoning
    if (matchCount === 0) {
      parts.push('No suitable providers currently available');
    } else if (matchCount === 1) {
      parts.push('One suitable provider identified');
    } else {
      parts.push(`${matchCount} suitable providers available`);
    }

    // Escalation reasoning
    if (escalationRequired) {
      parts.push('Escalation recommended due to severity or lack of available resources');
    }

    return parts.join('. ') + '.';
  }

  /**
   * Calculate decision confidence
   */
  private calculateConfidence(matches: any[], severityScore: any): number {
    let confidence = 0.5;

    // Increase confidence if we have good matches
    if (matches.length > 0) {
      const bestMatchScore = matches[0].matchScore;
      confidence += bestMatchScore * 0.3;
    }

    // Increase confidence for clear severity classification
    if (severityScore.totalScore > 0.7 || severityScore.totalScore < 0.3) {
      confidence += 0.2;
    }

    return Math.min(confidence, 1.0);
  }

  /**
   * Escalate query to next level
   */
  async escalateToNextLevel(
    query: PatientQuery,
    currentProviderId: string,
    reason: string
  ): Promise<RoutingDecision> {
    console.log(`[ESCALATION_ROUTER] Escalating query ${query.id} from provider ${currentProviderId}: ${reason}`);

    // In a real system, this would:
    // 1. Find higher-level providers (specialists, senior physicians)
    // 2. Create emergency alert
    // 3. Notify escalation chain
    // 4. Update query priority

    // For now, return a routing decision with escalation flag
    return {
      queryId: query.id,
      severity: SeverityLevel.HIGH,
      emergencyType: EmergencyType.URGENT_CARE,
      alternativeProviders: [],
      escalationRequired: true,
      estimatedResponseTime: 5,
      reasoning: `Escalated: ${reason}`,
      confidence: 0.8
    };
  }

  /**
   * Get routing statistics
   */
  getStats(decisions: RoutingDecision[]): {
    total: number;
    byEmergencyType: Map<EmergencyType, number>;
    bySeverity: Map<SeverityLevel, number>;
    escalationRate: number;
    averageResponseTime: number;
    averageConfidence: number;
  } {
    const byEmergencyType = new Map<EmergencyType, number>();
    const bySeverity = new Map<SeverityLevel, number>();
    let escalationCount = 0;
    let totalResponseTime = 0;
    let totalConfidence = 0;

    for (const decision of decisions) {
      // Count by emergency type
      const emergencyCount = byEmergencyType.get(decision.emergencyType) || 0;
      byEmergencyType.set(decision.emergencyType, emergencyCount + 1);

      // Count by severity
      const severityCount = bySeverity.get(decision.severity) || 0;
      bySeverity.set(decision.severity, severityCount + 1);

      // Count escalations
      if (decision.escalationRequired) {
        escalationCount++;
      }

      // Sum metrics
      totalResponseTime += decision.estimatedResponseTime;
      totalConfidence += decision.confidence;
    }

    const total = decisions.length;

    return {
      total,
      byEmergencyType,
      bySeverity,
      escalationRate: total > 0 ? escalationCount / total : 0,
      averageResponseTime: total > 0 ? totalResponseTime / total : 0,
      averageConfidence: total > 0 ? totalConfidence / total : 0
    };
  }
}
