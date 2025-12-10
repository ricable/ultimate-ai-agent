/**
 * Provider Service
 * Manages healthcare providers and their availability
 */

import {
  Provider,
  ProviderType,
  ProviderStatus,
  ProviderMetrics,
  PatientQuery,
  QueryStatus
} from './types';

export class ProviderService {
  private providers: Map<string, Provider>;
  private onlineProviders: Set<string>;
  private metrics: Map<string, ProviderMetrics>;

  constructor() {
    this.providers = new Map();
    this.onlineProviders = new Set();
    this.metrics = new Map();
  }

  /**
   * Register new provider
   */
  async registerProvider(provider: Provider): Promise<void> {
    if (this.providers.has(provider.id)) {
      throw new Error(`Provider already registered: ${provider.id}`);
    }

    this.providers.set(provider.id, provider);
    console.log(`[PROVIDER] Registered provider: ${provider.id} (${provider.type})`);
  }

  /**
   * Update provider status
   */
  async updateStatus(providerId: string, status: ProviderStatus): Promise<boolean> {
    const provider = this.providers.get(providerId);
    if (!provider) {
      return false;
    }

    provider.status = status;

    // Update online set
    if (status === ProviderStatus.AVAILABLE || status === ProviderStatus.ON_CALL) {
      this.onlineProviders.add(providerId);
    } else {
      this.onlineProviders.delete(providerId);
    }

    console.log(`[PROVIDER] ${providerId} status updated to ${status}`);
    return true;
  }

  /**
   * Find available provider by specialization
   */
  async findAvailableProvider(options?: {
    specialization?: string;
    providerType?: ProviderType;
    maxCaseLoad?: number;
  }): Promise<Provider | undefined> {
    for (const providerId of this.onlineProviders) {
      const provider = this.providers.get(providerId);
      if (!provider || provider.status !== ProviderStatus.AVAILABLE) {
        continue;
      }

      // Check case load
      if (provider.currentCaseLoad >= provider.maxConcurrentCases) {
        continue;
      }

      // Check provider type
      if (options?.providerType && provider.type !== options.providerType) {
        continue;
      }

      // Check specialization
      if (options?.specialization) {
        if (!provider.specialization?.includes(options.specialization)) {
          continue;
        }
      }

      return provider;
    }

    return undefined;
  }

  /**
   * Assign query to provider
   */
  async assignQuery(providerId: string, queryId: string): Promise<boolean> {
    const provider = this.providers.get(providerId);
    if (!provider) {
      throw new Error(`Provider not found: ${providerId}`);
    }

    if (provider.currentCaseLoad >= provider.maxConcurrentCases) {
      throw new Error(`Provider ${providerId} has reached maximum case load`);
    }

    provider.currentCaseLoad++;
    console.log(`[PROVIDER] Assigned query ${queryId} to ${providerId}`);
    return true;
  }

  /**
   * Release query from provider
   */
  async releaseQuery(providerId: string, queryId: string): Promise<boolean> {
    const provider = this.providers.get(providerId);
    if (!provider) {
      return false;
    }

    if (provider.currentCaseLoad > 0) {
      provider.currentCaseLoad--;
    }

    console.log(`[PROVIDER] Released query ${queryId} from ${providerId}`);
    return true;
  }

  /**
   * Get provider by ID
   */
  getProvider(providerId: string): Provider | undefined {
    return this.providers.get(providerId);
  }

  /**
   * Get all providers
   */
  getAllProviders(): Provider[] {
    return Array.from(this.providers.values());
  }

  /**
   * Get providers by type
   */
  getProvidersByType(type: ProviderType): Provider[] {
    return Array.from(this.providers.values())
      .filter(p => p.type === type);
  }

  /**
   * Get providers by specialization
   */
  getProvidersBySpecialization(specialization: string): Provider[] {
    return Array.from(this.providers.values())
      .filter(p => p.specialization?.includes(specialization));
  }

  /**
   * Get online providers
   */
  getOnlineProviders(): Provider[] {
    return Array.from(this.onlineProviders)
      .map(id => this.providers.get(id))
      .filter((p): p is Provider => p !== undefined);
  }

  /**
   * Record provider metrics
   */
  recordMetric(metric: ProviderMetrics): void {
    this.metrics.set(metric.providerId, metric);
  }

  /**
   * Get provider metrics
   */
  getMetrics(providerId: string): ProviderMetrics | undefined {
    return this.metrics.get(providerId);
  }

  /**
   * Calculate provider performance score
   */
  calculatePerformanceScore(providerId: string): number {
    const metric = this.metrics.get(providerId);
    if (!metric) {
      return 0;
    }

    // Weighted scoring
    const approvalScore = metric.approvalRate * 0.3;
    const speedScore = Math.max(0, 100 - metric.averageReviewTime) / 100 * 0.3;
    const responseScore = Math.max(0, 100 - metric.responseTime) / 100 * 0.2;
    const satisfactionScore = (metric.patientSatisfaction || 0) / 5 * 0.2;

    return approvalScore + speedScore + responseScore + satisfactionScore;
  }

  /**
   * Get provider statistics
   */
  getStats(): {
    totalProviders: number;
    onlineProviders: number;
    byType: Map<ProviderType, number>;
    byStatus: Map<ProviderStatus, number>;
    averageCaseLoad: number;
  } {
    const byType = new Map<ProviderType, number>();
    const byStatus = new Map<ProviderStatus, number>();
    let totalCaseLoad = 0;

    for (const provider of this.providers.values()) {
      // Count by type
      const typeCount = byType.get(provider.type) || 0;
      byType.set(provider.type, typeCount + 1);

      // Count by status
      const statusCount = byStatus.get(provider.status) || 0;
      byStatus.set(provider.status, statusCount + 1);

      // Sum case load
      totalCaseLoad += provider.currentCaseLoad;
    }

    return {
      totalProviders: this.providers.size,
      onlineProviders: this.onlineProviders.size,
      byType,
      byStatus,
      averageCaseLoad: this.providers.size > 0 ? totalCaseLoad / this.providers.size : 0
    };
  }
}
