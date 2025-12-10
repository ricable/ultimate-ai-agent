/**
 * PM Counter Collector for Ericsson SMO Integration
 * 3GPP TS 28.552 - Performance Management (PM)
 *
 * Collects PM counters from RAN cells via Ericsson ENM/OSS
 * Streams data to midstream processor for real-time analytics
 * Stores in agentdb for GNN training and historical analysis
 *
 * @module smo/pm-collector
 * @track PM Data Pipeline
 * @agent agent-03
 */

import { EventEmitter } from 'events';
import { MidstreamProcessor, RANDataPoint } from '../learning/self-learner';

// ============================================================
// 3GPP TS 28.552 PM Counter Interfaces
// ============================================================

/**
 * PM Counters as defined in 3GPP TS 28.552
 * ROP (Result Output Period): 10 minutes (configurable)
 */
export interface PMCounters {
  // Uplink Performance Counters
  pmUlSinrMean: number;           // Mean UL SINR in dB (-20 to 40 dB)
  pmUlBler: number;               // UL Block Error Rate (0 to 1)
  pmPuschPrbUsage: number;        // PUSCH PRB utilization (0 to 100%)
  pmUlRssi: number;               // UL RSSI in dBm (-130 to 0 dBm)

  // Downlink Performance Counters
  pmDlSinrMean: number;           // Mean DL SINR in dB (-20 to 40 dB)
  pmDlBler: number;               // DL Block Error Rate (0 to 1)
  pmPdschPrbUsage: number;        // PDSCH PRB utilization (0 to 100%)

  // Accessibility KPIs (3GPP TS 28.554)
  pmRrcConnEstabSucc: number;     // RRC Connection Establishment Success count
  pmRrcConnEstabAtt: number;      // RRC Connection Establishment Attempts
  pmErabEstabSuccQci: {           // E-RAB Establishment Success per QCI
    [qci: number]: number;
  };

  // Retainability KPIs
  pmErabRelNormal: number;        // E-RAB Release Normal
  pmErabRelAbnormal: number;      // E-RAB Release Abnormal (drops)
  pmCallDropRate?: number;        // Calculated: pmErabRelAbnormal / (pmErabRelNormal + pmErabRelAbnormal)

  // Additional counters
  pmHoSuccessRate?: number;       // Handover success rate
  pmCssr?: number;                // Call Setup Success Rate (calculated)
  pmPrbUsageDl?: number;          // DL PRB usage
  pmPrbUsageUl?: number;          // UL PRB usage
  pmActiveDrb?: number;           // Active Data Radio Bearers
  pmActiveUe?: number;            // Active UEs

  // 5G-specific (NR)
  pmNrSinrMean?: number;          // NR SINR mean
  pmNrBler?: number;              // NR BLER
  pmNrPrbUsage?: number;          // NR PRB usage
}

/**
 * PM File structure from Ericsson ENM
 * XML format following 3GPP TS 32.435
 */
export interface PMFile {
  fileFormatVersion: string;      // e.g., "32.435 V10.0"
  vendorName: string;              // "Ericsson"
  collectionBeginTime: Date;       // Start of ROP
  measCollec: MeasurementCollection[];
}

export interface MeasurementCollection {
  managedElement: string;          // Cell DN (Distinguished Name)
  measTypes: string[];             // Array of PM counter names
  measValues: number[];            // Corresponding values
  granularityPeriod: number;       // ROP in seconds (600 = 10 min)
  repPeriod: number;               // Reporting period
}

/**
 * PM Collection Configuration
 */
export interface PMCollectorConfig {
  ropInterval: number;             // Result Output Period in milliseconds (default: 600000 = 10 min)
  enmEndpoint?: string;            // Ericsson ENM endpoint URL
  cells: string[];                 // List of cell DNs to collect from
  counters: string[];              // Specific PM counters to collect
  enableStreaming: boolean;        // Enable real-time streaming
  storageEnabled: boolean;         // Enable agentdb storage
  aggregationLevel: 'cell' | 'sector' | 'site';  // Aggregation level
}

/**
 * PM Data Point for storage
 */
export interface PMDataPoint {
  cellId: string;
  timestamp: number;
  rop: number;                     // ROP duration in seconds
  counters: PMCounters;
  metadata?: {
    siteId?: string;
    sectorId?: string;
    technology?: '4G' | '5G';
    band?: string;
  };
}

// ============================================================
// PM Collector Implementation
// ============================================================

/**
 * PMCollector - Real-time PM counter collection from Ericsson ENM/OSS
 *
 * Features:
 * - 10-minute ROP collection (configurable)
 * - Real-time streaming via midstream
 * - agentdb storage for GNN training
 * - KPI calculation (CSSR, Drop Rate, etc.)
 * - Anomaly detection integration
 */
export class PMCollector extends EventEmitter {
  private config: PMCollectorConfig;
  private midstream: MidstreamProcessor;
  private collectionInterval?: NodeJS.Timeout;
  private pmBuffer: Map<string, PMDataPoint[]>;
  private isRunning: boolean;
  private totalCollections: number;

  constructor(config: Partial<PMCollectorConfig> = {}) {
    super();

    this.config = {
      ropInterval: config.ropInterval || 600000,  // 10 minutes default
      enmEndpoint: config.enmEndpoint || 'http://enm.ericsson.local/pm/v1',
      cells: config.cells || [],
      counters: config.counters || this.getDefaultCounters(),
      enableStreaming: config.enableStreaming !== false,
      storageEnabled: config.storageEnabled !== false,
      aggregationLevel: config.aggregationLevel || 'cell'
    };

    // Initialize midstream processor with 10-minute window
    this.midstream = new MidstreamProcessor({
      bufferSize: 1000,
      flushInterval: this.config.ropInterval
    });

    this.pmBuffer = new Map();
    this.isRunning = false;
    this.totalCollections = 0;

    this.setupMidstreamHandlers();
  }

  /**
   * Get default PM counters to collect (3GPP TS 28.552)
   */
  private getDefaultCounters(): string[] {
    return [
      // Uplink
      'pmUlSinrMean',
      'pmUlBler',
      'pmPuschPrbUsage',
      'pmUlRssi',

      // Downlink
      'pmDlSinrMean',
      'pmDlBler',
      'pmPdschPrbUsage',

      // Accessibility
      'pmRrcConnEstabSucc',
      'pmRrcConnEstabAtt',

      // Retainability
      'pmErabRelNormal',
      'pmErabRelAbnormal'
    ];
  }

  /**
   * Setup midstream event handlers
   */
  private setupMidstreamHandlers(): void {
    this.midstream.on('flush', (batch: RANDataPoint[]) => {
      this.handleFlush(batch);
    });

    this.midstream.on('pm', (dataPoint: RANDataPoint) => {
      this.emit('pm_received', dataPoint);
    });
  }

  /**
   * Start PM collection
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      console.warn('[PMCollector] Already running');
      return;
    }

    console.log('[PMCollector] Starting PM collection...');
    console.log(`[PMCollector] ROP Interval: ${this.config.ropInterval / 1000}s`);
    console.log(`[PMCollector] Cells: ${this.config.cells.length}`);
    console.log(`[PMCollector] Counters: ${this.config.counters.length}`);

    this.isRunning = true;

    // Start midstream processor
    if (this.config.enableStreaming) {
      this.midstream.start();
    }

    // Start periodic collection
    this.collectionInterval = setInterval(() => {
      this.collectPMData();
    }, this.config.ropInterval);

    // Immediate first collection
    await this.collectPMData();

    this.emit('started');
  }

  /**
   * Stop PM collection
   */
  stop(): void {
    if (!this.isRunning) return;

    console.log('[PMCollector] Stopping PM collection...');

    this.isRunning = false;

    if (this.collectionInterval) {
      clearInterval(this.collectionInterval);
      this.collectionInterval = undefined;
    }

    this.midstream.stop();

    this.emit('stopped');
  }

  /**
   * Collect PM data from all configured cells
   */
  async collectPMData(): Promise<void> {
    const startTime = Date.now();
    const timestamp = startTime;

    console.log(`[PMCollector] Collecting PM data at ${new Date(timestamp).toISOString()}`);

    try {
      // In production, this would fetch from Ericsson ENM API
      // For now, we simulate PM data collection
      const pmDataPoints = await this.fetchPMFromENM(this.config.cells);

      // Process each cell's PM data
      for (const pmData of pmDataPoints) {
        // Calculate derived KPIs
        this.calculateKPIs(pmData.counters);

        // Create RAN data point for midstream
        const ranDataPoint: RANDataPoint = {
          timestamp,
          cellId: pmData.cellId,
          dataType: 'PM',
          metrics: pmData.counters as unknown as Record<string, number>,
          context: {
            rop: pmData.rop,
            metadata: pmData.metadata
          }
        };

        // Stream to midstream processor
        if (this.config.enableStreaming) {
          this.midstream.ingest(ranDataPoint);
        }

        // Store in buffer for batch processing
        if (!this.pmBuffer.has(pmData.cellId)) {
          this.pmBuffer.set(pmData.cellId, []);
        }
        this.pmBuffer.get(pmData.cellId)!.push(pmData);

        // Detect anomalies
        this.detectAnomalies(pmData);
      }

      // Store to agentdb if enabled
      if (this.config.storageEnabled) {
        await this.storePMData(pmDataPoints);
      }

      this.totalCollections++;

      const duration = Date.now() - startTime;
      console.log(`[PMCollector] Collection completed in ${duration}ms (${pmDataPoints.length} cells)`);

      this.emit('collection_complete', {
        timestamp,
        cellCount: pmDataPoints.length,
        duration
      });

    } catch (error) {
      console.error('[PMCollector] Collection failed:', error);
      this.emit('collection_error', error);
    }
  }

  /**
   * Fetch PM data from Ericsson ENM
   * In production, this would use ENM REST API or file-based collection
   */
  private async fetchPMFromENM(cells: string[]): Promise<PMDataPoint[]> {
    // Simulate ENM API response with realistic PM data
    const pmDataPoints: PMDataPoint[] = [];

    for (const cellId of cells) {
      const counters = this.generateMockPMCounters(cellId);

      pmDataPoints.push({
        cellId,
        timestamp: Date.now(),
        rop: this.config.ropInterval / 1000,  // Convert to seconds
        counters,
        metadata: {
          siteId: cellId.split('-')[0],
          technology: cellId.includes('NR') ? '5G' : '4G'
        }
      });
    }

    return pmDataPoints;
  }

  /**
   * Generate mock PM counters for testing
   * In production, this is replaced by actual ENM data
   */
  private generateMockPMCounters(cellId: string): PMCounters {
    // Generate realistic PM values with some variation
    const baseQuality = Math.random() * 0.3 + 0.7;  // 0.7 to 1.0

    const rrcAttempts = Math.floor(Math.random() * 500 + 100);
    const rrcSuccess = Math.floor(rrcAttempts * baseQuality);

    const erabNormal = Math.floor(Math.random() * 300 + 50);
    const erabAbnormal = Math.floor(erabNormal * (1 - baseQuality) * 0.1);

    return {
      // Uplink
      pmUlSinrMean: Math.random() * 15 + 10,  // 10-25 dB
      pmUlBler: Math.random() * 0.05,         // 0-5%
      pmPuschPrbUsage: Math.random() * 40 + 30,  // 30-70%
      pmUlRssi: Math.random() * 20 - 90,      // -90 to -70 dBm

      // Downlink
      pmDlSinrMean: Math.random() * 15 + 10,  // 10-25 dB
      pmDlBler: Math.random() * 0.05,         // 0-5%
      pmPdschPrbUsage: Math.random() * 50 + 25,  // 25-75%

      // Accessibility
      pmRrcConnEstabSucc: rrcSuccess,
      pmRrcConnEstabAtt: rrcAttempts,
      pmErabEstabSuccQci: {
        9: Math.floor(Math.random() * 100 + 50),   // QCI 9 (default bearer)
        5: Math.floor(Math.random() * 50 + 10),    // QCI 5 (IMS signaling)
        1: Math.floor(Math.random() * 30 + 5)      // QCI 1 (conversational voice)
      },

      // Retainability
      pmErabRelNormal: erabNormal,
      pmErabRelAbnormal: erabAbnormal,

      // Additional
      pmActiveUe: Math.floor(Math.random() * 200 + 50),
      pmActiveDrb: Math.floor(Math.random() * 300 + 100)
    };
  }

  /**
   * Calculate derived KPIs from raw PM counters
   */
  private calculateKPIs(counters: PMCounters): void {
    // Call Setup Success Rate (CSSR)
    if (counters.pmRrcConnEstabAtt > 0) {
      counters.pmCssr = counters.pmRrcConnEstabSucc / counters.pmRrcConnEstabAtt;
    }

    // Call Drop Rate
    const totalReleases = counters.pmErabRelNormal + counters.pmErabRelAbnormal;
    if (totalReleases > 0) {
      counters.pmCallDropRate = counters.pmErabRelAbnormal / totalReleases;
    }

    // Aggregate PRB usage
    counters.pmPrbUsageDl = counters.pmPdschPrbUsage;
    counters.pmPrbUsageUl = counters.pmPuschPrbUsage;
  }

  /**
   * Detect anomalies in PM data
   */
  private detectAnomalies(pmData: PMDataPoint): void {
    const { counters, cellId } = pmData;

    // Low SINR anomaly
    if (counters.pmUlSinrMean < 5 || counters.pmDlSinrMean < 5) {
      this.emit('anomaly_detected', {
        type: 'LOW_SINR',
        cellId,
        severity: 'major',
        metric: 'SINR',
        value: Math.min(counters.pmUlSinrMean, counters.pmDlSinrMean),
        threshold: 5,
        timestamp: pmData.timestamp
      });
    }

    // High BLER anomaly
    if (counters.pmUlBler > 0.1 || counters.pmDlBler > 0.1) {
      this.emit('anomaly_detected', {
        type: 'HIGH_BLER',
        cellId,
        severity: 'major',
        metric: 'BLER',
        value: Math.max(counters.pmUlBler, counters.pmDlBler),
        threshold: 0.1,
        timestamp: pmData.timestamp
      });
    }

    // Low CSSR anomaly
    if (counters.pmCssr && counters.pmCssr < 0.95) {
      this.emit('anomaly_detected', {
        type: 'LOW_CSSR',
        cellId,
        severity: counters.pmCssr < 0.90 ? 'critical' : 'major',
        metric: 'CSSR',
        value: counters.pmCssr,
        threshold: 0.95,
        timestamp: pmData.timestamp
      });
    }

    // High drop rate anomaly
    if (counters.pmCallDropRate && counters.pmCallDropRate > 0.02) {
      this.emit('anomaly_detected', {
        type: 'HIGH_DROP_RATE',
        cellId,
        severity: counters.pmCallDropRate > 0.05 ? 'critical' : 'major',
        metric: 'DROP_RATE',
        value: counters.pmCallDropRate,
        threshold: 0.02,
        timestamp: pmData.timestamp
      });
    }
  }

  /**
   * Store PM data to agentdb for GNN training
   */
  private async storePMData(pmDataPoints: PMDataPoint[]): Promise<void> {
    try {
      // In production, this would use agentdb API
      // For now, we log the storage intent
      console.log(`[PMCollector] Storing ${pmDataPoints.length} PM data points to agentdb`);

      this.emit('pm_stored', {
        count: pmDataPoints.length,
        timestamp: Date.now()
      });

    } catch (error) {
      console.error('[PMCollector] Storage failed:', error);
      this.emit('storage_error', error);
    }
  }

  /**
   * Handle midstream flush event
   */
  private handleFlush(batch: RANDataPoint[]): void {
    console.log(`[PMCollector] Midstream flushed ${batch.length} PM data points`);

    // Calculate batch statistics
    const pmBatch = batch.filter(dp => dp.dataType === 'PM');

    if (pmBatch.length > 0) {
      const avgSinr = pmBatch.reduce((sum, dp) => {
        return sum + (dp.metrics.pmUlSinrMean || 0);
      }, 0) / pmBatch.length;

      console.log(`[PMCollector] Batch avg SINR: ${avgSinr.toFixed(2)} dB`);
    }

    this.emit('batch_flushed', {
      totalCount: batch.length,
      pmCount: pmBatch.length,
      timestamp: Date.now()
    });
  }

  /**
   * Get PM data for a specific cell
   */
  getPMData(cellId: string, limit: number = 100): PMDataPoint[] {
    const buffer = this.pmBuffer.get(cellId) || [];
    return buffer.slice(-limit);
  }

  /**
   * Get aggregated PM statistics
   */
  getAggregatedStats(): {
    avgSinr: number;
    avgCssr: number;
    avgDropRate: number;
    avgPrbUsage: number;
    cellCount: number;
  } {
    let totalSinr = 0;
    let totalCssr = 0;
    let totalDropRate = 0;
    let totalPrbUsage = 0;
    let count = 0;

    for (const [cellId, pmData] of this.pmBuffer) {
      if (pmData.length === 0) continue;

      const latest = pmData[pmData.length - 1];
      totalSinr += (latest.counters.pmUlSinrMean + latest.counters.pmDlSinrMean) / 2;
      totalCssr += latest.counters.pmCssr || 0;
      totalDropRate += latest.counters.pmCallDropRate || 0;
      totalPrbUsage += ((latest.counters.pmPrbUsageDl || 0) + (latest.counters.pmPrbUsageUl || 0)) / 2;
      count++;
    }

    return {
      avgSinr: count > 0 ? totalSinr / count : 0,
      avgCssr: count > 0 ? totalCssr / count : 0,
      avgDropRate: count > 0 ? totalDropRate / count : 0,
      avgPrbUsage: count > 0 ? totalPrbUsage / count : 0,
      cellCount: count
    };
  }

  /**
   * Get collector statistics
   */
  getStats() {
    return {
      isRunning: this.isRunning,
      totalCollections: this.totalCollections,
      configuredCells: this.config.cells.length,
      bufferedCells: this.pmBuffer.size,
      ropInterval: this.config.ropInterval,
      midstreamStats: {
        bufferSize: (this.midstream as any).buffer?.length || 0
      }
    };
  }

  /**
   * Add cells to collection
   */
  addCells(cells: string[]): void {
    for (const cell of cells) {
      if (!this.config.cells.includes(cell)) {
        this.config.cells.push(cell);
      }
    }

    console.log(`[PMCollector] Added ${cells.length} cells. Total: ${this.config.cells.length}`);
    this.emit('cells_added', { cells, total: this.config.cells.length });
  }

  /**
   * Remove cells from collection
   */
  removeCells(cells: string[]): void {
    this.config.cells = this.config.cells.filter(c => !cells.includes(c));

    for (const cell of cells) {
      this.pmBuffer.delete(cell);
    }

    console.log(`[PMCollector] Removed ${cells.length} cells. Total: ${this.config.cells.length}`);
    this.emit('cells_removed', { cells, total: this.config.cells.length });
  }
}

// ============================================================
// Exports
// ============================================================

// export { PMCollector };
// export type { PMCollectorConfig, PMDataPoint, PMFile, MeasurementCollection };
