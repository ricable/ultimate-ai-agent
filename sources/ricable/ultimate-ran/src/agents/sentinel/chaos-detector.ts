/**
 * Chaos Detector - Lyapunov Monitoring
 *
 * Monitors system stability using Lyapunov exponent analysis.
 * Detects chaotic behavior in real-time network metrics.
 */

import { EventEmitter } from 'events';

export enum ChaosLevel {
  NONE = 'NONE',
  WARNING = 'WARNING',
  HIGH = 'HIGH',
  CRITICAL = 'CRITICAL'
}

export interface ChaosEvent {
  level: ChaosLevel;
  lyapunovExponent: number;
  systemStability: number;
  interferenceLevel?: number;
  timestamp: number;
}

export interface ChaosDetectorConfig {
  lyapunovCritical: number;
  systemStability: number;
  iotMaxDbm: number;
  monitoringInterval: number;
}

export interface SystemMetrics {
  sinr: number[];
  throughput: number[];
  latency: number[];
}

export class ChaosDetector extends EventEmitter {
  private config: ChaosDetectorConfig;
  private currentLevel: ChaosLevel = ChaosLevel.NONE;
  private monitoringInterval?: NodeJS.Timeout;
  private metricsBuffer: SystemMetrics = {
    sinr: [],
    throughput: [],
    latency: []
  };

  constructor(config: ChaosDetectorConfig) {
    super();
    this.config = config;
  }

  getCurrentLevel(): ChaosLevel {
    return this.currentLevel;
  }

  startMonitoring(): void {
    this.monitoringInterval = setInterval(async () => {
      const lyapunov = await this.calculateLyapunovExponent(this.metricsBuffer);
      const stability = this.calculateSystemStability();
      const interference = this.measureInterference();

      const event: ChaosEvent = {
        level: this.classifyChaosLevel({
          lyapunovExponent: lyapunov,
          systemStability: stability,
          interferenceLevel: interference
        }),
        lyapunovExponent: lyapunov,
        systemStability: stability,
        interferenceLevel: interference,
        timestamp: Date.now()
      };

      this.currentLevel = event.level;
      this.emit('chaos', event);
    }, this.config.monitoringInterval);
  }

  stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }
  }

  /**
   * Calculate Lyapunov exponent from system metrics
   * Measures the rate of separation of infinitesimally close trajectories
   */
  async calculateLyapunovExponent(metrics: SystemMetrics): Promise<number> {
    if (!metrics.sinr || metrics.sinr.length < 2) {
      return 0;
    }

    // Calculate divergence rate using finite differences
    const divergences: number[] = [];

    for (let i = 1; i < metrics.sinr.length; i++) {
      const delta = Math.abs(metrics.sinr[i] - metrics.sinr[i - 1]);
      if (delta > 0) {
        divergences.push(Math.log(delta));
      }
    }

    if (divergences.length === 0) {
      return 0;
    }

    // Average log divergence approximates Lyapunov exponent
    const avgDivergence = divergences.reduce((sum, d) => sum + d, 0) / divergences.length;

    // Normalize to [0, 1] range
    const normalized = Math.max(0, Math.min(1, avgDivergence / 10));

    return normalized;
  }

  classifyChaosLevel(event: {
    lyapunovExponent: number;
    systemStability: number;
    interferenceLevel?: number;
  }): ChaosLevel {
    let violations = 0;

    // Check Lyapunov threshold
    if (event.lyapunovExponent > this.config.lyapunovCritical) {
      violations++;
    }

    // Check stability threshold
    if (event.systemStability < this.config.systemStability) {
      violations++;
    }

    // Check interference threshold
    if (event.interferenceLevel && event.interferenceLevel > this.config.iotMaxDbm) {
      violations++;
    }

    // Classify based on number of violations
    if (violations >= 2) {
      return ChaosLevel.CRITICAL;
    } else if (violations === 1) {
      if (event.lyapunovExponent > this.config.lyapunovCritical) {
        return ChaosLevel.HIGH;
      }
      return ChaosLevel.WARNING;
    }

    // Check if approaching thresholds (within 20% of threshold)
    const lyapunovWarning = event.lyapunovExponent > this.config.lyapunovCritical * 0.8 &&
                            event.lyapunovExponent <= this.config.lyapunovCritical;
    const stabilityWarning = event.systemStability < this.config.systemStability &&
                             event.systemStability >= this.config.systemStability * 0.98;
    const interferenceWarning = event.interferenceLevel &&
                               event.interferenceLevel > this.config.iotMaxDbm * 1.05 &&
                               event.interferenceLevel <= this.config.iotMaxDbm;

    if (lyapunovWarning || stabilityWarning || interferenceWarning) {
      return ChaosLevel.WARNING;
    }

    return ChaosLevel.NONE;
  }

  async testStability(): Promise<boolean> {
    const lyapunov = await this.calculateLyapunovExponent(this.metricsBuffer);
    const stability = this.calculateSystemStability();

    return (
      lyapunov <= this.config.lyapunovCritical &&
      stability >= this.config.systemStability
    );
  }

  private calculateSystemStability(): number {
    if (this.metricsBuffer.sinr.length < 2) {
      return 1.0;
    }

    // Calculate coefficient of variation (inverse of stability)
    const mean = this.metricsBuffer.sinr.reduce((sum, v) => sum + v, 0) / this.metricsBuffer.sinr.length;
    const variance = this.metricsBuffer.sinr.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / this.metricsBuffer.sinr.length;
    const stdDev = Math.sqrt(variance);

    const cv = mean !== 0 ? stdDev / Math.abs(mean) : 0;

    // Convert to stability score (1 = perfectly stable, 0 = highly unstable)
    return Math.max(0, 1 - cv);
  }

  private measureInterference(): number {
    // Placeholder for interference measurement
    // In production, this would interface with RAN metrics
    return -110; // dBm
  }

  addMetrics(metrics: Partial<SystemMetrics>): void {
    if (metrics.sinr) {
      this.metricsBuffer.sinr.push(...metrics.sinr);
      // Keep buffer size manageable
      if (this.metricsBuffer.sinr.length > 100) {
        this.metricsBuffer.sinr = this.metricsBuffer.sinr.slice(-100);
      }
    }
    if (metrics.throughput) {
      this.metricsBuffer.throughput.push(...metrics.throughput);
      if (this.metricsBuffer.throughput.length > 100) {
        this.metricsBuffer.throughput = this.metricsBuffer.throughput.slice(-100);
      }
    }
    if (metrics.latency) {
      this.metricsBuffer.latency.push(...metrics.latency);
      if (this.metricsBuffer.latency.length > 100) {
        this.metricsBuffer.latency = this.metricsBuffer.latency.slice(-100);
      }
    }
  }
}
