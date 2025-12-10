/**
 * Sample Data Generator for RAN Network Analysis
 * Generates realistic test data for development and testing
 */

import { v4 as uuidv4 } from 'uuid';
import type {
  CellKPISnapshot,
  NeighborRelation,
  KPITimeSeries,
  CellId,
  AccessibilityKPI,
  RetainabilityKPI,
  RadioQualityKPI,
  MobilityKPI,
  UplinkInterferenceKPI,
  UplinkPowerControlKPI,
} from '../models/ran-kpi.js';

// ============================================================================
// RANDOM UTILITIES
// ============================================================================

function randomNormal(mean: number, stdDev: number): number {
  const u1 = Math.random();
  const u2 = Math.random();
  const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  return mean + z * stdDev;
}

function randomBetween(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function randomInt(min: number, max: number): number {
  return Math.floor(randomBetween(min, max + 1));
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// ============================================================================
// CELL GENERATOR
// ============================================================================

export interface CellGenerationConfig {
  numCells: number;
  technology: 'LTE' | 'NR' | '5G-NSA';
  healthDistribution: {
    healthy: number;
    degraded: number;
    critical: number;
  };
  includeAnomalies: boolean;
  anomalyRate: number;
}

export const DEFAULT_CELL_CONFIG: CellGenerationConfig = {
  numCells: 50,
  technology: 'LTE',
  healthDistribution: {
    healthy: 0.7,
    degraded: 0.2,
    critical: 0.1,
  },
  includeAnomalies: true,
  anomalyRate: 0.15,
};

export class SampleDataGenerator {
  private config: CellGenerationConfig;
  private cellIds: string[] = [];

  constructor(config: Partial<CellGenerationConfig> = {}) {
    this.config = { ...DEFAULT_CELL_CONFIG, ...config };
  }

  /**
   * Generate complete sample dataset
   */
  generateDataset(): {
    cellSnapshots: Map<string, CellKPISnapshot>;
    timeSeriesData: Map<string, KPITimeSeries[]>;
    neighborRelations: NeighborRelation[];
  } {
    // Generate cells
    const cellSnapshots = this.generateCellSnapshots();

    // Generate time series
    const timeSeriesData = this.generateTimeSeriesData(cellSnapshots);

    // Generate neighbor relations
    const neighborRelations = this.generateNeighborRelations();

    return {
      cellSnapshots,
      timeSeriesData,
      neighborRelations,
    };
  }

  /**
   * Generate cell KPI snapshots
   */
  generateCellSnapshots(): Map<string, CellKPISnapshot> {
    const snapshots = new Map<string, CellKPISnapshot>();
    const timestamp = new Date();

    for (let i = 0; i < this.config.numCells; i++) {
      const cellId = `CELL_${String(i).padStart(3, '0')}`;
      this.cellIds.push(cellId);

      // Determine cell health based on distribution
      const healthRoll = Math.random();
      let healthType: 'healthy' | 'degraded' | 'critical';
      if (healthRoll < this.config.healthDistribution.healthy) {
        healthType = 'healthy';
      } else if (healthRoll < this.config.healthDistribution.healthy + this.config.healthDistribution.degraded) {
        healthType = 'degraded';
      } else {
        healthType = 'critical';
      }

      const snapshot = this.generateCellSnapshot(cellId, timestamp, healthType);
      snapshots.set(cellId, snapshot);
    }

    return snapshots;
  }

  private generateCellSnapshot(
    cellId: string,
    timestamp: Date,
    healthType: 'healthy' | 'degraded' | 'critical'
  ): CellKPISnapshot {
    const cell = this.generateCellId(cellId);
    const modifiers = this.getHealthModifiers(healthType);

    return {
      timestamp,
      cell,
      accessibility: this.generateAccessibilityKPI(modifiers),
      retainability: this.generateRetainabilityKPI(modifiers),
      radioQuality: this.generateRadioQualityKPI(modifiers),
      mobility: this.generateMobilityKPI(modifiers),
      uplinkInterference: this.generateUplinkInterferenceKPI(modifiers),
      uplinkPowerControl: this.generateUplinkPowerControlKPI(modifiers),
    };
  }

  private generateCellId(cellId: string): CellId {
    const cellNum = parseInt(cellId.split('_')[1]);
    return {
      cellId,
      enodebId: `ENB_${String(Math.floor(cellNum / 3)).padStart(3, '0')}`,
      sectorId: cellNum % 3,
      frequency: [1800, 2100, 2600][cellNum % 3],
      band: ['B3', 'B1', 'B7'][cellNum % 3],
      technology: this.config.technology,
      pci: cellNum % 504,
      tac: 1000 + Math.floor(cellNum / 10),
      latitude: 51.5 + (cellNum % 10) * 0.01,
      longitude: -0.1 + Math.floor(cellNum / 10) * 0.01,
    };
  }

  private getHealthModifiers(healthType: 'healthy' | 'degraded' | 'critical'): {
    successRateMod: number;
    dropRateMod: number;
    qualityMod: number;
    interferenceMod: number;
  } {
    switch (healthType) {
      case 'healthy':
        return { successRateMod: 0, dropRateMod: 1, qualityMod: 0, interferenceMod: 1 };
      case 'degraded':
        return { successRateMod: -2, dropRateMod: 3, qualityMod: -3, interferenceMod: 2 };
      case 'critical':
        return { successRateMod: -5, dropRateMod: 8, qualityMod: -6, interferenceMod: 4 };
    }
  }

  private generateAccessibilityKPI(modifiers: ReturnType<typeof this.getHealthModifiers>): Omit<AccessibilityKPI, 'timestamp' | 'cellId'> {
    const rrcAttempts = randomInt(5000, 20000);
    const rrcSuccessRate = clamp(randomNormal(99, 1) + modifiers.successRateMod, 85, 100);
    const rrcSuccess = Math.round(rrcAttempts * rrcSuccessRate / 100);

    const erabAttempts = Math.round(rrcSuccess * 0.95);
    const erabSuccessRate = clamp(randomNormal(99, 0.8) + modifiers.successRateMod, 88, 100);
    const erabSuccess = Math.round(erabAttempts * erabSuccessRate / 100);

    const s1Attempts = Math.round(rrcAttempts * 0.98);
    const s1SuccessRate = clamp(randomNormal(99.5, 0.5) + modifiers.successRateMod, 90, 100);

    const contextAttempts = erabAttempts;
    const contextSuccessRate = (rrcSuccessRate * erabSuccessRate) / 100;

    return {
      rrcSetupAttempts: rrcAttempts,
      rrcSetupSuccess: rrcSuccess,
      rrcSetupFailure: rrcAttempts - rrcSuccess,
      rrcSetupSuccessRate: rrcSuccessRate,
      erabSetupAttempts: erabAttempts,
      erabSetupSuccess: erabSuccess,
      erabSetupFailure: erabAttempts - erabSuccess,
      erabSetupSuccessRate: erabSuccessRate,
      s1SigConnEstabAttempts: s1Attempts,
      s1SigConnEstabSuccess: Math.round(s1Attempts * s1SuccessRate / 100),
      s1SigConnEstabSuccessRate: s1SuccessRate,
      initialContextSetupAttempts: contextAttempts,
      initialContextSetupSuccess: Math.round(contextAttempts * contextSuccessRate / 100),
      initialContextSetupSuccessRate: contextSuccessRate,
      rrcFailureCauses: {
        congestion: randomInt(0, 50 * Math.abs(modifiers.successRateMod)),
        unspecified: randomInt(0, 20),
        timer: randomInt(0, 10),
        radioResourceNotAvailable: randomInt(0, 30),
      },
    };
  }

  private generateRetainabilityKPI(modifiers: ReturnType<typeof this.getHealthModifiers>): Omit<RetainabilityKPI, 'timestamp' | 'cellId'> {
    const baseDropRate = 0.3;
    const dropRate = clamp(baseDropRate * modifiers.dropRateMod + randomNormal(0, 0.2), 0, 10);

    return {
      erabNormalRelease: randomInt(4000, 15000),
      erabAbnormalRelease: randomInt(10, 500) * modifiers.dropRateMod,
      erabDropRate: dropRate,
      voiceCallAttempts: randomInt(1000, 5000),
      voiceCallDrops: randomInt(5, 100) * modifiers.dropRateMod,
      voiceCallDropRate: clamp(dropRate * 0.8, 0, 5),
      dataSessionAttempts: randomInt(3000, 12000),
      dataSessionDrops: randomInt(20, 200) * modifiers.dropRateMod,
      dataSessionRetainability: clamp(100 - dropRate, 90, 100),
      contextReleaseCauses: {
        radioConnectionWithUeLost: randomInt(10, 100) * modifiers.dropRateMod,
        userInactivity: randomInt(500, 3000),
        s1UPathSwitch: randomInt(0, 50),
        interRatRedirection: randomInt(0, 100),
        intraLteRedirection: randomInt(0, 200),
        x2Handover: randomInt(100, 1000),
        s1Handover: randomInt(50, 500),
        other: randomInt(0, 100),
      },
    };
  }

  private generateRadioQualityKPI(modifiers: ReturnType<typeof this.getHealthModifiers>): Omit<RadioQualityKPI, 'timestamp' | 'cellId'> {
    const cqi = clamp(randomNormal(10, 2) + modifiers.qualityMod, 1, 15);
    const sinr = clamp(randomNormal(15, 5) + modifiers.qualityMod, -5, 30);
    const rsrp = clamp(randomNormal(-95, 10) + modifiers.qualityMod * 2, -140, -60);
    const rsrq = clamp(randomNormal(-10, 3) + modifiers.qualityMod, -25, 0);

    return {
      dlAvgCqi: cqi,
      dlCqiDistribution: Array(16).fill(0).map((_, i) => {
        const dist = Math.exp(-0.5 * Math.pow((i - cqi) / 2, 2));
        return Math.round(dist * 100);
      }),
      dlRi1Ratio: clamp(randomNormal(40, 15), 0, 100),
      dlRi2Ratio: clamp(randomNormal(60, 15), 0, 100),
      dlBlerPercent: clamp(randomNormal(5, 3) - modifiers.qualityMod, 0, 30),
      ulSinrAvg: sinr,
      ulSinrP10: sinr - randomBetween(8, 12),
      ulSinrP50: sinr - randomBetween(2, 5),
      ulSinrP90: sinr + randomBetween(5, 10),
      ulBlerPercent: clamp(randomNormal(5, 3) - modifiers.qualityMod, 0, 30),
      rsrpAvg: rsrp,
      rsrpP10: rsrp - randomBetween(15, 25),
      rsrpP50: rsrp - randomBetween(3, 8),
      rsrpP90: rsrp + randomBetween(10, 20),
      rsrqAvg: rsrq,
      rsrqP10: rsrq - randomBetween(5, 8),
      rsrqP50: rsrq - randomBetween(1, 3),
      rsrqP90: rsrq + randomBetween(3, 6),
      dlSpectralEfficiency: clamp(randomNormal(4, 1.5) + modifiers.qualityMod * 0.3, 0.5, 10),
      ulSpectralEfficiency: clamp(randomNormal(2, 0.8) + modifiers.qualityMod * 0.2, 0.3, 5),
    };
  }

  private generateMobilityKPI(modifiers: ReturnType<typeof this.getHealthModifiers>): Omit<MobilityKPI, 'timestamp' | 'cellId'> {
    const intraAttempts = randomInt(500, 3000);
    const intraSuccessRate = clamp(randomNormal(97, 2) + modifiers.successRateMod, 80, 100);
    const intraSuccess = Math.round(intraAttempts * intraSuccessRate / 100);

    const interAttempts = randomInt(100, 800);
    const interSuccessRate = clamp(randomNormal(95, 3) + modifiers.successRateMod, 75, 100);
    const interSuccess = Math.round(interAttempts * interSuccessRate / 100);

    return {
      intraFreqHoAttempts: intraAttempts,
      intraFreqHoSuccess: intraSuccess,
      intraFreqHoFailure: intraAttempts - intraSuccess,
      intraFreqHoSuccessRate: intraSuccessRate,
      interFreqHoAttempts: interAttempts,
      interFreqHoSuccess: interSuccess,
      interFreqHoFailure: interAttempts - interSuccess,
      interFreqHoSuccessRate: interSuccessRate,
      interRatHoAttempts: randomInt(50, 200),
      interRatHoSuccess: randomInt(40, 180),
      interRatHoFailure: randomInt(0, 30),
      interRatHoSuccessRate: clamp(randomNormal(92, 4), 70, 100),
      x2HoAttempts: Math.round(intraAttempts * 0.9),
      x2HoSuccess: Math.round(intraSuccess * 0.92),
      x2HoSuccessRate: clamp(randomNormal(96, 2) + modifiers.successRateMod, 80, 100),
      s1HoAttempts: Math.round(intraAttempts * 0.1),
      s1HoSuccess: Math.round(intraSuccess * 0.08),
      s1HoSuccessRate: clamp(randomNormal(94, 3) + modifiers.successRateMod, 75, 100),
      tooEarlyHo: randomInt(0, 20) * Math.max(1, -modifiers.successRateMod),
      tooLateHo: randomInt(0, 15) * Math.max(1, -modifiers.successRateMod),
      wrongCellHo: randomInt(0, 10),
      pingPongHo: randomInt(5, 30) * Math.max(1, -modifiers.successRateMod / 2),
      incomingHoTotal: intraSuccess + interSuccess,
      outgoingHoTotal: randomInt(500, 2500),
    };
  }

  private generateUplinkInterferenceKPI(modifiers: ReturnType<typeof this.getHealthModifiers>): Omit<UplinkInterferenceKPI, 'timestamp' | 'cellId'> {
    const iot = clamp(randomNormal(5, 2) * modifiers.interferenceMod, 0, 20);
    const interference = -110 + iot;

    return {
      prbUlInterferenceAvg: interference,
      prbUlInterferenceP10: interference - randomBetween(3, 6),
      prbUlInterferenceP50: interference - randomBetween(1, 3),
      prbUlInterferenceP90: interference + randomBetween(5, 10),
      prbUlInterferenceP99: interference + randomBetween(10, 15),
      iotAvg: iot,
      iotP95: iot + randomBetween(3, 6),
      rip: interference + 5,
      externalInterferenceDetected: iot > 10,
      externalInterferenceLevel: iot > 12 ? 'high' : iot > 8 ? 'medium' : iot > 5 ? 'low' : 'none',
      puschSinrDegradation: clamp(iot - 5, 0, 10),
      highInterferencePrbRatio: clamp(randomNormal(10, 8) * modifiers.interferenceMod / 2, 0, 100),
    };
  }

  private generateUplinkPowerControlKPI(modifiers: ReturnType<typeof this.getHealthModifiers>): Omit<UplinkPowerControlKPI, 'timestamp' | 'cellId'> {
    const p0 = randomInt(-100, -90);
    const alpha = [0.6, 0.7, 0.8, 0.9, 1.0][randomInt(0, 4)];
    const pathLoss = clamp(randomNormal(120, 15), 80, 160);

    const txPower = p0 + alpha * pathLoss;
    const avgTxPower = clamp(txPower * 0.15, -10, 23);
    const headroom = 23 - avgTxPower;

    const powerLimited = clamp(
      randomNormal(8, 5) * (modifiers.interferenceMod > 1 ? modifiers.interferenceMod : 1),
      0,
      50
    );

    return {
      p0NominalPusch: p0,
      p0NominalPucch: p0 - 5,
      alpha,
      ueTxPowerAvg: avgTxPower,
      ueTxPowerP10: avgTxPower - randomBetween(5, 10),
      ueTxPowerP50: avgTxPower - randomBetween(1, 3),
      ueTxPowerP90: Math.min(23, avgTxPower + randomBetween(5, 10)),
      ueTxPowerMax: 23,
      powerHeadroomAvg: headroom,
      powerHeadroomP10: headroom - randomBetween(10, 20),
      powerHeadroomP50: headroom - randomBetween(3, 7),
      powerHeadroomP90: headroom + randomBetween(5, 10),
      negativePowerHeadroomRatio: clamp(powerLimited * 0.8, 0, 50),
      pathLossAvg: pathLoss,
      pathLossP10: pathLoss - randomBetween(20, 30),
      pathLossP50: pathLoss - randomBetween(5, 10),
      pathLossP90: pathLoss + randomBetween(15, 25),
      tpcUpCommands: randomInt(1000, 10000),
      tpcDownCommands: randomInt(1000, 10000),
      tpcAccumulatedOffset: randomNormal(0, 3),
      powerLimitedUeRatio: powerLimited,
    };
  }

  /**
   * Generate time series data
   */
  generateTimeSeriesData(
    cellSnapshots: Map<string, CellKPISnapshot>
  ): Map<string, KPITimeSeries[]> {
    const timeSeriesData = new Map<string, KPITimeSeries[]>();
    const now = new Date();
    const pointsPerDay = 96; // 15-min granularity
    const daysOfData = 7;
    const totalPoints = pointsPerDay * daysOfData;

    for (const [cellId, snapshot] of cellSnapshots) {
      const seriesList: KPITimeSeries[] = [];

      // Generate a few key KPI time series
      const kpis = [
        { name: 'rrcSetupSuccessRate', domain: 'accessibility' as const, baseline: snapshot.accessibility.rrcSetupSuccessRate },
        { name: 'erabDropRate', domain: 'retainability' as const, baseline: snapshot.retainability.erabDropRate },
        { name: 'ulSinrAvg', domain: 'radioQuality' as const, baseline: snapshot.radioQuality.ulSinrAvg },
        { name: 'iotAvg', domain: 'uplinkInterference' as const, baseline: snapshot.uplinkInterference.iotAvg },
        { name: 'powerLimitedUeRatio', domain: 'uplinkPowerControl' as const, baseline: snapshot.uplinkPowerControl.powerLimitedUeRatio },
      ];

      for (const kpi of kpis) {
        const series = this.generateKPITimeSeries(
          cellId,
          kpi.name,
          kpi.domain,
          kpi.baseline,
          totalPoints,
          now
        );
        seriesList.push(series);
      }

      timeSeriesData.set(cellId, seriesList);
    }

    return timeSeriesData;
  }

  private generateKPITimeSeries(
    cellId: string,
    kpiName: string,
    domain: KPITimeSeries['domain'],
    baseline: number,
    points: number,
    endTime: Date
  ): KPITimeSeries {
    const startTime = new Date(endTime.getTime() - points * 15 * 60 * 1000);
    const dataPoints: KPITimeSeries['dataPoints'] = [];

    let value = baseline;
    const volatility = Math.abs(baseline) * 0.1;

    for (let i = 0; i < points; i++) {
      const timestamp = new Date(startTime.getTime() + i * 15 * 60 * 1000);

      // Add random walk
      value += randomNormal(0, volatility);

      // Add daily seasonality
      const hourOfDay = timestamp.getHours();
      const dailyFactor = 1 + 0.1 * Math.sin((hourOfDay - 6) * Math.PI / 12);
      value *= dailyFactor;

      // Add anomalies
      if (this.config.includeAnomalies && Math.random() < this.config.anomalyRate / points) {
        value *= randomBetween(1.5, 2.5); // Spike
      }

      // Clamp to reasonable range
      value = Math.max(baseline * 0.5, Math.min(baseline * 1.5, value));

      dataPoints.push({ timestamp, value });
    }

    return {
      cellId,
      kpiName,
      domain,
      granularity: '15min',
      startTime,
      endTime,
      dataPoints,
    };
  }

  /**
   * Generate neighbor relations
   */
  generateNeighborRelations(): NeighborRelation[] {
    const relations: NeighborRelation[] = [];

    for (let i = 0; i < this.cellIds.length; i++) {
      const sourceCell = this.cellIds[i];

      // Each cell has 3-6 intra-freq neighbors
      const numNeighbors = randomInt(3, 6);
      const neighborIndices = new Set<number>();

      while (neighborIndices.size < numNeighbors) {
        const neighborIdx = randomInt(0, this.cellIds.length - 1);
        if (neighborIdx !== i) {
          neighborIndices.add(neighborIdx);
        }
      }

      for (const neighborIdx of neighborIndices) {
        const targetCell = this.cellIds[neighborIdx];

        relations.push({
          sourceCellId: sourceCell,
          targetCellId: targetCell,
          relationshipType: Math.random() < 0.8 ? 'intra-freq' : 'inter-freq',
          sourcePci: i % 504,
          sourceFrequency: 1800,
          sourceRsrp: randomNormal(-95, 10),
          sourceSinr: randomNormal(15, 5),
          targetPci: neighborIdx % 504,
          targetFrequency: Math.random() < 0.8 ? 1800 : 2100,
          targetRsrp: randomNormal(-100, 12),
          targetSinr: randomNormal(12, 6),
          hoAttempts: randomInt(100, 1000),
          hoSuccess: randomInt(90, 950),
          hoFailure: randomInt(5, 100),
          hoSuccessRate: randomNormal(95, 3),
          a3Offset: randomInt(-3, 3),
          hysteresis: randomBetween(1, 3),
          timeToTrigger: [40, 64, 80, 100, 128, 160][randomInt(0, 5)],
          neighborQuality: ['excellent', 'good', 'fair', 'poor'][randomInt(0, 3)] as NeighborRelation['neighborQuality'],
          distance: randomInt(100, 2000),
          azimuthDifference: randomInt(0, 180),
        });
      }
    }

    return relations;
  }
}

export default SampleDataGenerator;
