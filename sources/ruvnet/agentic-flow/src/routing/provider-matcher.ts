/**
 * Provider Matcher
 * Matches patient queries to the most suitable providers
 */

import { ProviderMatch } from './types';
import { PatientQuery } from '../providers/types';
import { Provider, ProviderStatus } from '../providers/types';

export class ProviderMatcher {
  /**
   * Find best matching providers for query
   */
  async findMatches(
    query: PatientQuery,
    availableProviders: Provider[],
    topN: number = 5
  ): Promise<ProviderMatch[]> {
    const matches: ProviderMatch[] = [];

    for (const provider of availableProviders) {
      // Skip if provider is offline or too busy
      if (provider.status === ProviderStatus.OFFLINE) {
        continue;
      }

      if (provider.currentCaseLoad >= provider.maxConcurrentCases) {
        continue;
      }

      const matchScore = this.calculateMatchScore(query, provider);
      const estimatedWaitTime = this.estimateWaitTime(provider);

      matches.push({
        providerId: provider.id,
        matchScore,
        availability: provider.status === ProviderStatus.AVAILABLE,
        estimatedWaitTime,
        specializations: provider.specialization || [],
        currentLoad: provider.currentCaseLoad / provider.maxConcurrentCases,
        performanceScore: 0.8 // Would come from provider metrics
      });
    }

    // Sort by match score (descending) and wait time (ascending)
    matches.sort((a, b) => {
      const scoreDiff = b.matchScore - a.matchScore;
      if (Math.abs(scoreDiff) > 0.1) {
        return scoreDiff;
      }
      return a.estimatedWaitTime - b.estimatedWaitTime;
    });

    return matches.slice(0, topN);
  }

  /**
   * Calculate match score between query and provider
   */
  private calculateMatchScore(query: PatientQuery, provider: Provider): number {
    let score = 0;

    // Base score for availability
    if (provider.status === ProviderStatus.AVAILABLE) {
      score += 0.3;
    } else if (provider.status === ProviderStatus.ON_CALL) {
      score += 0.15;
    }

    // Score based on specialization match
    if (provider.specialization && provider.specialization.length > 0) {
      const specializationScore = this.matchSpecialization(query, provider.specialization);
      score += specializationScore * 0.4;
    } else {
      score += 0.2; // General practitioner baseline
    }

    // Score based on current load
    const loadFactor = 1 - (provider.currentCaseLoad / provider.maxConcurrentCases);
    score += loadFactor * 0.2;

    // Score based on query type match
    const queryTypeScore = this.matchQueryType(query, provider);
    score += queryTypeScore * 0.1;

    return Math.min(score, 1.0);
  }

  /**
   * Match query to provider specialization
   */
  private matchSpecialization(query: PatientQuery, specializations: string[]): number {
    const queryText = `${query.queryType} ${query.description}`.toLowerCase();

    const specializationKeywords: Record<string, string[]> = {
      'cardiology': ['heart', 'chest pain', 'cardiac', 'blood pressure'],
      'pulmonology': ['breathing', 'lungs', 'respiratory', 'asthma'],
      'neurology': ['headache', 'seizure', 'neurological', 'brain'],
      'orthopedics': ['bone', 'fracture', 'joint', 'muscle'],
      'dermatology': ['skin', 'rash', 'dermatological'],
      'pediatrics': ['child', 'infant', 'pediatric'],
      'emergency': ['emergency', 'urgent', 'critical'],
      'internal_medicine': ['fever', 'infection', 'general']
    };

    let bestMatch = 0;

    for (const specialization of specializations) {
      const keywords = specializationKeywords[specialization.toLowerCase()];
      if (!keywords) continue;

      for (const keyword of keywords) {
        if (queryText.includes(keyword)) {
          bestMatch = Math.max(bestMatch, 1.0);
          break;
        }
      }

      if (bestMatch === 1.0) break;
    }

    return bestMatch;
  }

  /**
   * Match query type to provider
   */
  private matchQueryType(query: PatientQuery, provider: Provider): number {
    // Simple query type matching - would be more sophisticated in production
    const emergencyTypes = ['emergency', 'urgent', 'critical'];
    const isEmergency = emergencyTypes.some(type =>
      query.queryType.toLowerCase().includes(type)
    );

    if (isEmergency && provider.type === 'physician') {
      return 1.0;
    }

    return 0.5;
  }

  /**
   * Estimate wait time for provider
   */
  private estimateWaitTime(provider: Provider): number {
    // Base wait time
    let waitTime = 5; // minutes

    // Add time based on current load
    const loadFactor = provider.currentCaseLoad / provider.maxConcurrentCases;
    waitTime += loadFactor * 30;

    // Add time based on status
    if (provider.status === ProviderStatus.BUSY) {
      waitTime += 15;
    } else if (provider.status === ProviderStatus.ON_CALL) {
      waitTime += 20;
    }

    return Math.round(waitTime);
  }

  /**
   * Get match explanation
   */
  getMatchExplanation(match: ProviderMatch): string {
    const reasons: string[] = [];

    if (match.matchScore > 0.8) {
      reasons.push('excellent match');
    } else if (match.matchScore > 0.6) {
      reasons.push('good match');
    } else {
      reasons.push('suitable match');
    }

    if (match.availability) {
      reasons.push('immediately available');
    } else {
      reasons.push(`estimated ${match.estimatedWaitTime} min wait`);
    }

    if (match.specializations.length > 0) {
      reasons.push(`specializes in ${match.specializations.join(', ')}`);
    }

    if (match.currentLoad < 0.5) {
      reasons.push('low current workload');
    }

    return reasons.join(', ');
  }
}
