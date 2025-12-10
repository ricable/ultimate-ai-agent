/**
 * Classification Models for RAN Domain Analysis
 * Classifies cells, anomalies, and issues into categories
 */

import * as ss from 'simple-statistics';
import type {
  CellKPISnapshot,
  DetectedAnomaly,
  RootCauseCategory,
} from '../models/ran-kpi.js';

// ============================================================================
// CELL HEALTH CLASSIFICATION
// ============================================================================

export type CellHealthStatus = 'healthy' | 'degraded' | 'critical' | 'failed';

export interface CellHealthClassification {
  cellId: string;
  timestamp: Date;
  overallHealth: CellHealthStatus;
  healthScore: number; // 0-100
  domainScores: {
    accessibility: number;
    retainability: number;
    radioQuality: number;
    mobility: number;
    uplinkInterference: number;
    uplinkPowerControl: number;
  };
  issues: string[];
  recommendations: string[];
}

export class CellHealthClassifier {
  private weights = {
    accessibility: 0.2,
    retainability: 0.2,
    radioQuality: 0.2,
    mobility: 0.15,
    uplinkInterference: 0.15,
    uplinkPowerControl: 0.1,
  };

  classifyCell(snapshot: CellKPISnapshot): CellHealthClassification {
    const domainScores = {
      accessibility: this.scoreAccessibility(snapshot),
      retainability: this.scoreRetainability(snapshot),
      radioQuality: this.scoreRadioQuality(snapshot),
      mobility: this.scoreMobility(snapshot),
      uplinkInterference: this.scoreUplinkInterference(snapshot),
      uplinkPowerControl: this.scoreUplinkPowerControl(snapshot),
    };

    const healthScore =
      domainScores.accessibility * this.weights.accessibility +
      domainScores.retainability * this.weights.retainability +
      domainScores.radioQuality * this.weights.radioQuality +
      domainScores.mobility * this.weights.mobility +
      domainScores.uplinkInterference * this.weights.uplinkInterference +
      domainScores.uplinkPowerControl * this.weights.uplinkPowerControl;

    const overallHealth = this.healthScoreToStatus(healthScore);
    const { issues, recommendations } = this.identifyIssues(snapshot, domainScores);

    return {
      cellId: snapshot.cell.cellId,
      timestamp: snapshot.timestamp,
      overallHealth,
      healthScore,
      domainScores,
      issues,
      recommendations,
    };
  }

  private scoreAccessibility(snapshot: CellKPISnapshot): number {
    const acc = snapshot.accessibility;
    const rrcScore = Math.max(0, (acc.rrcSetupSuccessRate - 90) / 10 * 100);
    const erabScore = Math.max(0, (acc.erabSetupSuccessRate - 90) / 10 * 100);
    const contextScore = Math.max(0, (acc.initialContextSetupSuccessRate - 90) / 10 * 100);
    return (rrcScore * 0.4 + erabScore * 0.4 + contextScore * 0.2);
  }

  private scoreRetainability(snapshot: CellKPISnapshot): number {
    const ret = snapshot.retainability;
    // Invert drop rates (lower is better)
    const dropScore = Math.max(0, 100 - ret.erabDropRate * 20);
    const voiceDropScore = Math.max(0, 100 - ret.voiceCallDropRate * 10);
    const sessionScore = Math.max(0, ret.dataSessionRetainability);
    return (dropScore * 0.4 + voiceDropScore * 0.3 + sessionScore * 0.3);
  }

  private scoreRadioQuality(snapshot: CellKPISnapshot): number {
    const rq = snapshot.radioQuality;
    const cqiScore = Math.min(100, (rq.dlAvgCqi / 15) * 100);
    const sinrScore = Math.min(100, Math.max(0, ((rq.ulSinrAvg + 5) / 35) * 100));
    const rsrpScore = Math.min(100, Math.max(0, ((rq.rsrpAvg + 140) / 60) * 100));
    const rsrqScore = Math.min(100, Math.max(0, ((rq.rsrqAvg + 25) / 25) * 100));
    return (cqiScore * 0.3 + sinrScore * 0.3 + rsrpScore * 0.2 + rsrqScore * 0.2);
  }

  private scoreMobility(snapshot: CellKPISnapshot): number {
    const mob = snapshot.mobility;
    const intraScore = Math.max(0, mob.intraFreqHoSuccessRate);
    const interScore = Math.max(0, mob.interFreqHoSuccessRate);
    const x2Score = Math.max(0, mob.x2HoSuccessRate);
    // Penalize ping-pong handovers
    const pingPongPenalty = Math.min(30, (mob.pingPongHo / (mob.intraFreqHoSuccess + 1)) * 100);
    return Math.max(0, (intraScore * 0.4 + interScore * 0.3 + x2Score * 0.3) - pingPongPenalty);
  }

  private scoreUplinkInterference(snapshot: CellKPISnapshot): number {
    const ui = snapshot.uplinkInterference;
    // Lower IoT is better
    const iotScore = Math.max(0, 100 - ui.iotAvg * 5);
    const prbScore = Math.max(0, 100 - ui.highInterferencePrbRatio);
    const externalPenalty = ui.externalInterferenceDetected ? 20 : 0;
    return Math.max(0, (iotScore * 0.5 + prbScore * 0.5) - externalPenalty);
  }

  private scoreUplinkPowerControl(snapshot: CellKPISnapshot): number {
    const upc = snapshot.uplinkPowerControl;
    // Lower power limited UE ratio is better
    const limitedScore = Math.max(0, 100 - upc.powerLimitedUeRatio * 3);
    const phrScore = Math.max(0, 100 - upc.negativePowerHeadroomRatio * 2);
    // Good headroom distribution
    const headroomScore = Math.min(100, Math.max(0, (upc.powerHeadroomAvg + 5) / 30 * 100));
    return (limitedScore * 0.4 + phrScore * 0.3 + headroomScore * 0.3);
  }

  private healthScoreToStatus(score: number): CellHealthStatus {
    if (score >= 80) return 'healthy';
    if (score >= 60) return 'degraded';
    if (score >= 30) return 'critical';
    return 'failed';
  }

  private identifyIssues(
    snapshot: CellKPISnapshot,
    scores: Record<string, number>
  ): { issues: string[]; recommendations: string[] } {
    const issues: string[] = [];
    const recommendations: string[] = [];

    // Accessibility issues
    if (snapshot.accessibility.rrcSetupSuccessRate < 98) {
      issues.push('Low RRC setup success rate');
      recommendations.push('Check RACH configuration and preamble allocation');
    }
    if (snapshot.accessibility.erabSetupSuccessRate < 98) {
      issues.push('Low E-RAB setup success rate');
      recommendations.push('Verify S1 connection and MME capacity');
    }

    // Retainability issues
    if (snapshot.retainability.erabDropRate > 0.5) {
      issues.push('High E-RAB drop rate');
      recommendations.push('Investigate UL interference and handover parameters');
    }

    // Radio quality issues
    if (snapshot.radioQuality.dlAvgCqi < 7) {
      issues.push('Poor DL radio quality (low CQI)');
      recommendations.push('Check for DL interference and antenna issues');
    }
    if (snapshot.radioQuality.ulSinrAvg < 5) {
      issues.push('Poor UL radio quality (low SINR)');
      recommendations.push('Review UL power control settings and interference sources');
    }

    // Mobility issues
    if (snapshot.mobility.intraFreqHoSuccessRate < 95) {
      issues.push('Low intra-frequency handover success rate');
      recommendations.push('Review A3 offset and neighbor relations');
    }
    if (snapshot.mobility.pingPongHo > 10) {
      issues.push('High ping-pong handover rate');
      recommendations.push('Increase hysteresis or time-to-trigger');
    }

    // UL interference
    if (snapshot.uplinkInterference.iotAvg > 10) {
      issues.push('High uplink interference over thermal');
      recommendations.push('Investigate external interference sources');
    }

    // Power control issues
    if (snapshot.uplinkPowerControl.powerLimitedUeRatio > 10) {
      issues.push('High ratio of power-limited UEs');
      recommendations.push('Review P0 nominal PUSCH and alpha settings');
    }
    if (snapshot.uplinkPowerControl.negativePowerHeadroomRatio > 15) {
      issues.push('High ratio of UEs with negative power headroom');
      recommendations.push('Consider increasing P0 nominal PUSCH or coverage enhancement');
    }

    return { issues, recommendations };
  }
}

// ============================================================================
// ANOMALY CLASSIFICATION
// ============================================================================

export interface AnomalyClassification {
  anomalyId: string;
  category: RootCauseCategory;
  confidence: number;
  evidence: string[];
  correlatedKpis: string[];
  impactAssessment: {
    userImpact: 'none' | 'minor' | 'moderate' | 'severe';
    networkImpact: 'isolated' | 'local' | 'regional' | 'widespread';
  };
}

export class AnomalyClassifier {
  private domainToCategory: Record<string, RootCauseCategory[]> = {
    accessibility: ['hardware_failure', 'software_issue', 'capacity_exhaustion', 'configuration_error'],
    retainability: ['interference', 'coverage_issue', 'mobility_issue', 'backhaul_issue'],
    radioQuality: ['interference', 'hardware_failure', 'coverage_issue', 'parameter_drift'],
    mobility: ['neighbor_relation_issue', 'configuration_error', 'coverage_issue', 'mobility_issue'],
    uplinkInterference: ['interference', 'external_factor', 'hardware_failure'],
    uplinkPowerControl: ['power_control_issue', 'configuration_error', 'coverage_issue', 'parameter_drift'],
  };

  classifyAnomaly(
    anomaly: DetectedAnomaly,
    relatedAnomalies: DetectedAnomaly[] = [],
    cellSnapshot?: CellKPISnapshot
  ): AnomalyClassification {
    const possibleCategories = this.domainToCategory[anomaly.domain] || ['unknown'];
    const evidence: string[] = [];
    const correlatedKpis: string[] = [];

    // Analyze anomaly characteristics
    let category: RootCauseCategory = 'unknown';
    let confidence = 0.5;

    // Pattern matching based on anomaly type and domain
    if (anomaly.anomalyType === 'spike' || anomaly.anomalyType === 'dip') {
      if (anomaly.domain === 'uplinkInterference') {
        category = 'interference';
        evidence.push('Sudden interference change detected');
        confidence = 0.7;
      } else if (anomaly.domain === 'accessibility') {
        if (anomaly.kpiName.includes('rrc')) {
          category = 'capacity_exhaustion';
          evidence.push('RRC failures indicate possible capacity issue');
          confidence = 0.6;
        }
      }
    }

    if (anomaly.anomalyType === 'level_shift') {
      category = 'configuration_error';
      evidence.push('Sustained level change suggests configuration change');
      confidence = 0.65;
    }

    if (anomaly.anomalyType === 'trend_shift') {
      category = 'parameter_drift';
      evidence.push('Gradual change indicates parameter drift');
      confidence = 0.55;
    }

    // Check for correlated anomalies
    for (const related of relatedAnomalies) {
      if (
        Math.abs(related.timestamp.getTime() - anomaly.timestamp.getTime()) < 15 * 60 * 1000 &&
        related.cellId === anomaly.cellId
      ) {
        correlatedKpis.push(related.kpiName);

        // Correlation-based refinement
        if (related.domain === 'uplinkInterference' && anomaly.domain === 'radioQuality') {
          category = 'interference';
          evidence.push(`Correlated with ${related.kpiName}`);
          confidence = Math.min(0.9, confidence + 0.1);
        }

        if (related.domain === 'mobility' && anomaly.domain === 'retainability') {
          category = 'mobility_issue';
          evidence.push('Mobility issues correlate with retainability degradation');
          confidence = Math.min(0.9, confidence + 0.15);
        }
      }
    }

    // Use cell snapshot for context if available
    if (cellSnapshot) {
      if (
        anomaly.domain === 'uplinkPowerControl' &&
        cellSnapshot.uplinkPowerControl.negativePowerHeadroomRatio > 20
      ) {
        category = 'power_control_issue';
        evidence.push('High negative PHR ratio confirms power control issue');
        confidence = 0.85;
      }

      if (
        anomaly.domain === 'mobility' &&
        cellSnapshot.mobility.pingPongHo > 15
      ) {
        category = 'neighbor_relation_issue';
        evidence.push('High ping-pong rate indicates neighbor relation issue');
        confidence = 0.8;
      }
    }

    // Impact assessment
    const userImpact = this.assessUserImpact(anomaly);
    const networkImpact = this.assessNetworkImpact(anomaly, relatedAnomalies);

    return {
      anomalyId: anomaly.id,
      category,
      confidence,
      evidence,
      correlatedKpis,
      impactAssessment: {
        userImpact,
        networkImpact,
      },
    };
  }

  private assessUserImpact(
    anomaly: DetectedAnomaly
  ): 'none' | 'minor' | 'moderate' | 'severe' {
    if (anomaly.severity === 'critical') return 'severe';
    if (anomaly.severity === 'high') return 'moderate';
    if (anomaly.severity === 'medium') return 'minor';
    return 'none';
  }

  private assessNetworkImpact(
    anomaly: DetectedAnomaly,
    relatedAnomalies: DetectedAnomaly[]
  ): 'isolated' | 'local' | 'regional' | 'widespread' {
    const uniqueCells = new Set([
      anomaly.cellId,
      ...relatedAnomalies.map(a => a.cellId),
    ]);

    if (uniqueCells.size === 1) return 'isolated';
    if (uniqueCells.size <= 5) return 'local';
    if (uniqueCells.size <= 20) return 'regional';
    return 'widespread';
  }
}

// ============================================================================
// ISSUE PATTERN CLASSIFIER
// ============================================================================

export type IssuePattern =
  | 'external_interference'
  | 'pim_interference'
  | 'coverage_hole'
  | 'overshooting'
  | 'pilot_pollution'
  | 'capacity_exhaustion'
  | 'backhaul_congestion'
  | 'hardware_degradation'
  | 'software_bug'
  | 'parameter_misconfiguration'
  | 'neighbor_missing'
  | 'neighbor_excess'
  | 'power_control_aggressive'
  | 'power_control_conservative'
  | 'handover_too_early'
  | 'handover_too_late'
  | 'unknown';

export interface IssuePatternClassification {
  pattern: IssuePattern;
  confidence: number;
  indicators: string[];
  suggestedActions: string[];
}

export class IssuePatternClassifier {
  classifyPattern(
    snapshot: CellKPISnapshot,
    anomalies: DetectedAnomaly[]
  ): IssuePatternClassification[] {
    const patterns: IssuePatternClassification[] = [];

    // Check for external interference pattern
    patterns.push(...this.checkExternalInterference(snapshot, anomalies));

    // Check for PIM pattern
    patterns.push(...this.checkPIMInterference(snapshot, anomalies));

    // Check for coverage issues
    patterns.push(...this.checkCoverageIssues(snapshot, anomalies));

    // Check for power control issues
    patterns.push(...this.checkPowerControlIssues(snapshot, anomalies));

    // Check for handover issues
    patterns.push(...this.checkHandoverIssues(snapshot, anomalies));

    // Check for capacity issues
    patterns.push(...this.checkCapacityIssues(snapshot, anomalies));

    // Sort by confidence
    return patterns.sort((a, b) => b.confidence - a.confidence);
  }

  private checkExternalInterference(
    snapshot: CellKPISnapshot,
    anomalies: DetectedAnomaly[]
  ): IssuePatternClassification[] {
    const patterns: IssuePatternClassification[] = [];
    const ui = snapshot.uplinkInterference;

    if (ui.externalInterferenceDetected || ui.iotAvg > 10) {
      const interferenceAnomalies = anomalies.filter(
        a => a.domain === 'uplinkInterference' && a.anomalyType === 'spike'
      );

      if (interferenceAnomalies.length > 0 || ui.iotAvg > 12) {
        patterns.push({
          pattern: 'external_interference',
          confidence: Math.min(0.95, 0.6 + interferenceAnomalies.length * 0.1),
          indicators: [
            `IoT: ${ui.iotAvg.toFixed(1)} dB`,
            `High interference PRB ratio: ${ui.highInterferencePrbRatio}%`,
            ui.externalInterferenceDetected ? 'External interference flag active' : '',
          ].filter(Boolean),
          suggestedActions: [
            'Spectrum scan to identify interference source',
            'Check for illegal transmitters or nearby radar',
            'Consider frequency retuning if persistent',
          ],
        });
      }
    }

    return patterns;
  }

  private checkPIMInterference(
    snapshot: CellKPISnapshot,
    anomalies: DetectedAnomaly[]
  ): IssuePatternClassification[] {
    const patterns: IssuePatternClassification[] = [];
    const ui = snapshot.uplinkInterference;
    const rq = snapshot.radioQuality;

    // PIM pattern: high interference on specific PRBs, correlated with DL power
    if (ui.highInterferencePrbRatio > 15 && rq.ulSinrAvg < 5) {
      patterns.push({
        pattern: 'pim_interference',
        confidence: 0.5, // Medium confidence - needs more evidence
        indicators: [
          'High interference on specific PRBs',
          `UL SINR degradation: ${rq.ulSinrAvg.toFixed(1)} dB`,
          'Possible PIM from antenna system',
        ],
        suggestedActions: [
          'PIM test with portable analyzer',
          'Check antenna connectors and jumpers',
          'Inspect passive components for corrosion',
        ],
      });
    }

    return patterns;
  }

  private checkCoverageIssues(
    snapshot: CellKPISnapshot,
    anomalies: DetectedAnomaly[]
  ): IssuePatternClassification[] {
    const patterns: IssuePatternClassification[] = [];
    const rq = snapshot.radioQuality;
    const upc = snapshot.uplinkPowerControl;

    // Coverage hole pattern
    if (rq.rsrpP10 < -115 && upc.powerLimitedUeRatio > 15) {
      patterns.push({
        pattern: 'coverage_hole',
        confidence: 0.7,
        indicators: [
          `RSRP P10: ${rq.rsrpP10} dBm (weak signal at cell edge)`,
          `Power limited UEs: ${upc.powerLimitedUeRatio}%`,
          'Users struggling at cell boundary',
        ],
        suggestedActions: [
          'Review antenna tilt and azimuth',
          'Consider coverage extension solutions',
          'Verify neighbor relations for handover',
        ],
      });
    }

    // Overshooting pattern
    if (rq.rsrpP90 > -70 && snapshot.mobility.pingPongHo > 10) {
      patterns.push({
        pattern: 'overshooting',
        confidence: 0.65,
        indicators: [
          `RSRP P90: ${rq.rsrpP90} dBm (strong signal far from cell)`,
          `Ping-pong handovers: ${snapshot.mobility.pingPongHo}`,
          'Cell coverage exceeds intended area',
        ],
        suggestedActions: [
          'Increase antenna electrical tilt',
          'Review transmit power settings',
          'Adjust cell individual offset',
        ],
      });
    }

    return patterns;
  }

  private checkPowerControlIssues(
    snapshot: CellKPISnapshot,
    anomalies: DetectedAnomaly[]
  ): IssuePatternClassification[] {
    const patterns: IssuePatternClassification[] = [];
    const upc = snapshot.uplinkPowerControl;
    const ui = snapshot.uplinkInterference;

    // Conservative power control (P0 too high)
    if (upc.ueTxPowerAvg > 15 && ui.iotAvg > 8) {
      patterns.push({
        pattern: 'power_control_conservative',
        confidence: 0.6,
        indicators: [
          `Average UE TX power: ${upc.ueTxPowerAvg} dBm (high)`,
          `IoT: ${ui.iotAvg} dB (elevated)`,
          'P0 nominal may be too high',
        ],
        suggestedActions: [
          'Reduce P0 nominal PUSCH by 2-3 dB',
          'Consider increasing alpha for better path loss compensation',
        ],
      });
    }

    // Aggressive power control (P0 too low)
    if (upc.powerLimitedUeRatio > 20 && upc.negativePowerHeadroomRatio > 25) {
      patterns.push({
        pattern: 'power_control_aggressive',
        confidence: 0.75,
        indicators: [
          `Power limited UEs: ${upc.powerLimitedUeRatio}%`,
          `Negative PHR ratio: ${upc.negativePowerHeadroomRatio}%`,
          'P0 nominal may be too low',
        ],
        suggestedActions: [
          'Increase P0 nominal PUSCH by 3-5 dB',
          'Review alpha setting for edge users',
          'Consider path loss based optimization',
        ],
      });
    }

    return patterns;
  }

  private checkHandoverIssues(
    snapshot: CellKPISnapshot,
    anomalies: DetectedAnomaly[]
  ): IssuePatternClassification[] {
    const patterns: IssuePatternClassification[] = [];
    const mob = snapshot.mobility;

    // Too early handover
    if (mob.tooEarlyHo > 5) {
      patterns.push({
        pattern: 'handover_too_early',
        confidence: 0.7,
        indicators: [
          `Too early HO count: ${mob.tooEarlyHo}`,
          'UEs returning to source cell after handover',
        ],
        suggestedActions: [
          'Increase A3 offset',
          'Increase time-to-trigger',
          'Review source cell CIO',
        ],
      });
    }

    // Too late handover
    if (mob.tooLateHo > 5) {
      patterns.push({
        pattern: 'handover_too_late',
        confidence: 0.7,
        indicators: [
          `Too late HO count: ${mob.tooLateHo}`,
          'Radio link failures before handover completion',
        ],
        suggestedActions: [
          'Decrease A3 offset',
          'Decrease time-to-trigger',
          'Review hysteresis settings',
        ],
      });
    }

    // Missing neighbor
    if (mob.intraFreqHoSuccessRate < 95 && mob.intraFreqHoAttempts > 100) {
      patterns.push({
        pattern: 'neighbor_missing',
        confidence: 0.55,
        indicators: [
          `Intra-freq HO success: ${mob.intraFreqHoSuccessRate}%`,
          'Possible missing neighbor relation',
        ],
        suggestedActions: [
          'Review ANR (Automatic Neighbor Relations) output',
          'Check for HO failures to unknown PCI',
          'Add missing neighbor relations manually',
        ],
      });
    }

    return patterns;
  }

  private checkCapacityIssues(
    snapshot: CellKPISnapshot,
    anomalies: DetectedAnomaly[]
  ): IssuePatternClassification[] {
    const patterns: IssuePatternClassification[] = [];
    const acc = snapshot.accessibility;

    // RRC congestion
    if (acc.rrcSetupSuccessRate < 97 && acc.rrcFailureCauses?.congestion && acc.rrcFailureCauses.congestion > 10) {
      patterns.push({
        pattern: 'capacity_exhaustion',
        confidence: 0.8,
        indicators: [
          `RRC success rate: ${acc.rrcSetupSuccessRate}%`,
          `Congestion failures: ${acc.rrcFailureCauses.congestion}`,
          'Cell capacity may be exhausted',
        ],
        suggestedActions: [
          'Review maximum connected users setting',
          'Consider load balancing to adjacent cells',
          'Evaluate capacity expansion options',
        ],
      });
    }

    return patterns;
  }
}

export default {
  CellHealthClassifier,
  AnomalyClassifier,
  IssuePatternClassifier,
};
