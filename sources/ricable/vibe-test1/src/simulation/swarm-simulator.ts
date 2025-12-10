/**
 * Swarm Simulator - Network Simulation Engine
 *
 * High-fidelity simulation of the RAN environment for:
 * - Training RL agents safely
 * - Testing optimization strategies
 * - Validating fault management workflows
 * - Benchmarking swarm performance
 *
 * Features:
 * - Realistic traffic patterns
 * - RF propagation modeling
 * - User mobility simulation
 * - Fault injection
 * - Multi-cell coordination
 */

import { EventEmitter } from 'eventemitter3';
import { v4 as uuidv4 } from 'uuid';
import {
  CellKPIs,
  CellGlobalIdentity,
  SimulationConfig,
  SimulationState,
  UserState,
  RANGraph,
  RANGraphNode,
  RANGraphEdge,
  Problem,
  ProblemCategory,
  UplinkPowerControlParams,
  AntennaParams
} from '../types/index.js';

// ============================================================================
// RF PROPAGATION MODEL
// ============================================================================

class PropagationModel {
  // 3GPP TR 38.901 Urban Macro parameters
  private readonly hBs = 25; // Base station height (m)
  private readonly hUt = 1.5; // User terminal height (m)
  private readonly fc = 3500; // Carrier frequency (MHz)

  /**
   * Calculate pathloss using 3GPP 38.901 UMa model
   */
  calculatePathloss(distance: number, los: boolean): number {
    const d3d = Math.sqrt(distance ** 2 + (this.hBs - this.hUt) ** 2);
    const dBp = 4 * this.hBs * this.hUt * this.fc / 300; // Breakpoint distance

    if (los) {
      if (d3d <= dBp) {
        return 28 + 22 * Math.log10(d3d) + 20 * Math.log10(this.fc / 1000);
      } else {
        return 28 + 40 * Math.log10(d3d) + 20 * Math.log10(this.fc / 1000)
          - 9 * Math.log10((dBp) ** 2 + (this.hBs - this.hUt) ** 2);
      }
    } else {
      // NLOS
      const plNlos = 13.54 + 39.08 * Math.log10(d3d) + 20 * Math.log10(this.fc / 1000)
        - 0.6 * (this.hUt - 1.5);
      const plLos = this.calculatePathloss(distance, true);
      return Math.max(plNlos, plLos);
    }
  }

  /**
   * Calculate shadow fading (log-normal)
   */
  getShadowFading(los: boolean): number {
    const sigma = los ? 4 : 6; // dB
    return this.gaussianRandom() * sigma;
  }

  /**
   * Calculate LOS probability
   */
  losProbaility(distance: number): number {
    if (distance <= 18) return 1;
    return (18 / distance + Math.exp(-distance / 63) * (1 - 18 / distance));
  }

  private gaussianRandom(): number {
    const u1 = Math.random();
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
}

// ============================================================================
// USER MOBILITY MODEL
// ============================================================================

class MobilityModel {
  /**
   * Update user position based on mobility model
   */
  updatePosition(
    user: UserState,
    deltaTimeMs: number,
    bounds: { minLat: number; maxLat: number; minLng: number; maxLng: number }
  ): void {
    const deltaTimeSec = deltaTimeMs / 1000;

    // Update position
    const latDelta = user.velocity.speed * Math.cos(user.velocity.direction) * deltaTimeSec / 111000;
    const lngDelta = user.velocity.speed * Math.sin(user.velocity.direction) * deltaTimeSec / (111000 * Math.cos(user.position.lat * Math.PI / 180));

    user.position.lat += latDelta;
    user.position.lng += lngDelta;

    // Bounce off boundaries
    if (user.position.lat < bounds.minLat || user.position.lat > bounds.maxLat) {
      user.velocity.direction = Math.PI - user.velocity.direction;
      user.position.lat = Math.max(bounds.minLat, Math.min(bounds.maxLat, user.position.lat));
    }
    if (user.position.lng < bounds.minLng || user.position.lng > bounds.maxLng) {
      user.velocity.direction = -user.velocity.direction;
      user.position.lng = Math.max(bounds.minLng, Math.min(bounds.maxLng, user.position.lng));
    }

    // Random direction change (random waypoint model)
    if (Math.random() < 0.01) {
      user.velocity.direction = Math.random() * 2 * Math.PI;
    }
  }

  /**
   * Generate initial user distribution
   */
  generateUsers(
    count: number,
    bounds: { minLat: number; maxLat: number; minLng: number; maxLng: number },
    scenario: 'uniform' | 'hotspot' | 'commute'
  ): UserState[] {
    const users: UserState[] = [];

    for (let i = 0; i < count; i++) {
      let lat: number, lng: number;

      switch (scenario) {
        case 'hotspot':
          // 80% near center, 20% uniform
          if (Math.random() < 0.8) {
            const centerLat = (bounds.minLat + bounds.maxLat) / 2;
            const centerLng = (bounds.minLng + bounds.maxLng) / 2;
            lat = centerLat + (Math.random() - 0.5) * 0.01;
            lng = centerLng + (Math.random() - 0.5) * 0.01;
          } else {
            lat = bounds.minLat + Math.random() * (bounds.maxLat - bounds.minLat);
            lng = bounds.minLng + Math.random() * (bounds.maxLng - bounds.minLng);
          }
          break;
        default:
          lat = bounds.minLat + Math.random() * (bounds.maxLat - bounds.minLat);
          lng = bounds.minLng + Math.random() * (bounds.maxLng - bounds.minLng);
      }

      users.push({
        imsi: `imsi-${uuidv4().slice(0, 8)}`,
        servingCell: '',
        position: { lat, lng },
        velocity: {
          speed: Math.random() < 0.3 ? 0 : 1 + Math.random() * 15, // m/s
          direction: Math.random() * 2 * Math.PI
        },
        rsrp: -90,
        sinr: 10,
        throughput: 0
      });
    }

    return users;
  }
}

// ============================================================================
// TRAFFIC MODEL
// ============================================================================

class TrafficModel {
  /**
   * Generate traffic load factor based on time of day
   */
  getLoadFactor(hour: number, scenario: string): number {
    // Typical urban traffic pattern
    const basePattern = [
      0.3, 0.2, 0.15, 0.1, 0.1, 0.15,  // 0-5
      0.3, 0.6, 0.9, 1.0, 0.95, 0.9,   // 6-11
      0.85, 0.9, 0.85, 0.8, 0.85, 0.95, // 12-17
      1.0, 0.95, 0.85, 0.7, 0.5, 0.4   // 18-23
    ];

    let factor = basePattern[hour];

    // Scenario adjustments
    if (scenario === 'event') {
      factor *= 1.5; // Surge during events
    } else if (scenario === 'commute') {
      if ((hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19)) {
        factor *= 1.3;
      }
    }

    return Math.min(1, factor);
  }

  /**
   * Calculate user throughput based on SINR
   */
  calculateThroughput(sinr: number, bandwidth: number = 100): number {
    // Shannon capacity with efficiency factor
    const snrLinear = Math.pow(10, sinr / 10);
    const spectralEfficiency = Math.log2(1 + snrLinear);
    const efficiency = 0.75; // Account for overhead
    return bandwidth * spectralEfficiency * efficiency; // Mbps
  }
}

// ============================================================================
// CELL SIMULATOR
// ============================================================================

interface SimulatedCell {
  cgi: CellGlobalIdentity;
  position: { lat: number; lng: number };
  antenna: AntennaParams;
  powerControl: UplinkPowerControlParams;
  connectedUsers: Set<string>;
  interferingCells: Set<string>;
  kpis: CellKPIs;
}

class CellSimulator {
  cells: Map<string, SimulatedCell> = new Map();
  propagation: PropagationModel;

  constructor() {
    this.propagation = new PropagationModel();
  }

  /**
   * Create simulated cell
   */
  createCell(
    gnbId: number,
    cellId: number,
    position: { lat: number; lng: number },
    azimuth: number
  ): SimulatedCell {
    const cgi: CellGlobalIdentity = {
      mcc: '001',
      mnc: '01',
      gnbId,
      cellId,
      sectorId: cellId % 3
    };

    const cell: SimulatedCell = {
      cgi,
      position,
      antenna: {
        electricalTilt: 6,
        mechanicalTilt: 0,
        azimuth,
        beamwidth: 65,
        height: 25,
        maxPower: 46,
        referenceSignalPower: 15
      },
      powerControl: {
        p0NominalPusch: -96,
        alpha: 0.8,
        p0NominalPucch: -106,
        deltaMcs: true,
        accumulationEnabled: true,
        msg3DeltaPreamble: 0,
        deltaF_PUCCH_Format: {
          format1: 0,
          format1a: 0,
          format1b: 1,
          format2: 0,
          format2a: 0,
          format2b: 2,
          format3: 0,
          format4: 0
        }
      },
      connectedUsers: new Set(),
      interferingCells: new Set(),
      kpis: this.createInitialKpis(cgi)
    };

    const cellKey = this.cgiToString(cgi);
    this.cells.set(cellKey, cell);
    return cell;
  }

  /**
   * Calculate RSRP from cell to user position
   */
  calculateRsrp(cell: SimulatedCell, userPos: { lat: number; lng: number }): number {
    const distance = this.calculateDistance(cell.position, userPos);
    const los = Math.random() < this.propagation.losProbaility(distance);
    const pathloss = this.propagation.calculatePathloss(distance, los);
    const shadowFading = this.propagation.getShadowFading(los);
    const antennaGain = this.calculateAntennaGain(cell, userPos);

    return cell.antenna.referenceSignalPower + antennaGain - pathloss + shadowFading;
  }

  /**
   * Calculate antenna gain towards user (simplified)
   */
  private calculateAntennaGain(cell: SimulatedCell, userPos: { lat: number; lng: number }): number {
    const bearing = this.calculateBearing(cell.position, userPos);
    const angleDiff = Math.abs(bearing - cell.antenna.azimuth);
    const normalizedAngle = Math.min(angleDiff, 360 - angleDiff);

    // Simple antenna pattern: max gain at boresight, -12dB at ±65°
    const gain = 17 - 12 * Math.pow(normalizedAngle / cell.antenna.beamwidth, 2);
    return Math.max(-10, gain);
  }

  /**
   * Update cell KPIs based on current state
   */
  updateKpis(cell: SimulatedCell, users: UserState[], hour: number, loadFactor: number): void {
    const connectedUsers = users.filter(u => cell.connectedUsers.has(u.imsi));

    // Calculate aggregated metrics
    const avgRsrp = connectedUsers.length > 0
      ? connectedUsers.reduce((sum, u) => sum + u.rsrp, 0) / connectedUsers.length
      : -95;

    const avgSinr = connectedUsers.length > 0
      ? connectedUsers.reduce((sum, u) => sum + u.sinr, 0) / connectedUsers.length
      : 10;

    const totalThroughput = connectedUsers.reduce((sum, u) => sum + u.throughput, 0);

    cell.kpis = {
      ...cell.kpis,
      timestamp: new Date(),
      activeUsers: connectedUsers.length,
      rrcConnectedUsers: connectedUsers.length,

      avgRsrp,
      avgRsrq: avgRsrp - 90 + Math.random() * 5,
      avgSinrDl: avgSinr,
      avgSinrUl: avgSinr - 2 + Math.random() * 2,

      dlThroughput: totalThroughput,
      ulThroughput: totalThroughput * 0.3,
      dlUserThroughput5Pct: connectedUsers.length > 0 ? Math.min(...connectedUsers.map(u => u.throughput)) : 0,
      ulUserThroughput5Pct: connectedUsers.length > 0 ? Math.min(...connectedUsers.map(u => u.throughput)) * 0.3 : 0,

      prbUtilizationDl: Math.min(100, connectedUsers.length * 3 * loadFactor),
      prbUtilizationUl: Math.min(100, connectedUsers.length * 2 * loadFactor),

      iotUl: -6 + Math.random() * 10 + (cell.interferingCells.size * 2),
      rtwp: -110 + cell.interferingCells.size * 3 + connectedUsers.length * 0.5,

      callDropRate: Math.max(0, 0.3 + Math.random() * 0.3 - avgSinr * 0.02),
      rrcAbnormalRelease: Math.random() * 0.5,
      erabAbnormalRelease: Math.random() * 0.3,

      rrcSetupSuccessRate: Math.min(100, 95 + Math.random() * 5),
      erabSetupSuccessRate: Math.min(100, 96 + Math.random() * 4),
      ngSetupSuccessRate: Math.min(100, 97 + Math.random() * 3),

      hoSuccessRate: Math.min(100, 94 + Math.random() * 6),
      hoAttempts: Math.floor(connectedUsers.length * 0.2 * loadFactor),
      hoFailures: Math.floor(Math.random() * 3),
      pingPongRate: Math.random() * 3,

      blerDl: Math.max(0, 0.05 - avgSinr * 0.002 + Math.random() * 0.03),
      blerUl: Math.max(0, 0.06 - avgSinr * 0.002 + Math.random() * 0.04),
      cqiAverage: Math.min(15, Math.max(1, Math.floor(avgSinr / 2) + 5)),

      dataVolumeDl: totalThroughput * 900 / 8000, // GB per ROP
      dataVolumeUl: totalThroughput * 0.3 * 900 / 8000
    };
  }

  private createInitialKpis(cgi: CellGlobalIdentity): CellKPIs {
    return {
      timestamp: new Date(),
      cgi,
      rrcSetupSuccessRate: 98,
      erabSetupSuccessRate: 97,
      ngSetupSuccessRate: 98,
      callDropRate: 0.5,
      rrcAbnormalRelease: 0.2,
      erabAbnormalRelease: 0.1,
      hoSuccessRate: 96,
      hoAttempts: 50,
      hoFailures: 2,
      pingPongRate: 1.5,
      dlThroughput: 500,
      ulThroughput: 150,
      dlUserThroughput5Pct: 10,
      ulUserThroughput5Pct: 3,
      prbUtilizationDl: 45,
      prbUtilizationUl: 30,
      activeUsers: 100,
      rrcConnectedUsers: 120,
      avgRsrp: -85,
      avgRsrq: -10,
      avgSinrDl: 12,
      avgSinrUl: 10,
      iotUl: 5,
      rtwp: -105,
      blerDl: 0.05,
      blerUl: 0.06,
      cqiAverage: 10,
      dataVolumeDl: 50,
      dataVolumeUl: 15
    };
  }

  private calculateDistance(pos1: { lat: number; lng: number }, pos2: { lat: number; lng: number }): number {
    const R = 6371000; // Earth radius in meters
    const dLat = (pos2.lat - pos1.lat) * Math.PI / 180;
    const dLng = (pos2.lng - pos1.lng) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(pos1.lat * Math.PI / 180) * Math.cos(pos2.lat * Math.PI / 180) * Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  private calculateBearing(from: { lat: number; lng: number }, to: { lat: number; lng: number }): number {
    const dLng = (to.lng - from.lng) * Math.PI / 180;
    const lat1 = from.lat * Math.PI / 180;
    const lat2 = to.lat * Math.PI / 180;
    const x = Math.sin(dLng) * Math.cos(lat2);
    const y = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLng);
    return (Math.atan2(x, y) * 180 / Math.PI + 360) % 360;
  }

  cgiToString(cgi: CellGlobalIdentity): string {
    return `${cgi.mcc}-${cgi.mnc}-${cgi.gnbId}-${cgi.cellId}`;
  }
}

// ============================================================================
// FAULT INJECTION ENGINE
// ============================================================================

interface FaultInjectionConfig {
  enabled: boolean;
  faultTypes: ProblemCategory[];
  probability: number;
}

class FaultInjector extends EventEmitter {
  config: FaultInjectionConfig;
  activeFaults: Map<string, { type: ProblemCategory; startTime: Date; duration: number }> = new Map();

  constructor(config: FaultInjectionConfig) {
    super();
    this.config = config;
  }

  /**
   * Potentially inject a fault
   */
  maybeInjectFault(cellId: string, tick: number): void {
    if (!this.config.enabled) return;
    if (this.activeFaults.has(cellId)) return;
    if (Math.random() > this.config.probability) return;

    const faultType = this.config.faultTypes[Math.floor(Math.random() * this.config.faultTypes.length)];
    const duration = 5 + Math.floor(Math.random() * 20); // 5-25 ticks

    this.activeFaults.set(cellId, {
      type: faultType,
      startTime: new Date(),
      duration
    });

    this.emit('fault-injected', { cellId, faultType, duration });
  }

  /**
   * Apply fault effects to cell KPIs
   */
  applyFaultEffects(cell: SimulatedCell, cellId: string): void {
    const fault = this.activeFaults.get(cellId);
    if (!fault) return;

    switch (fault.type) {
      case 'UPLINK_INTERFERENCE':
        cell.kpis.iotUl += 10;
        cell.kpis.avgSinrUl -= 5;
        cell.kpis.blerUl += 0.1;
        break;
      case 'SLEEPING_CELL':
        cell.kpis.activeUsers = 0;
        cell.kpis.hoAttempts = 0;
        cell.kpis.dlThroughput = 0;
        break;
      case 'CAPACITY_SATURATION':
        cell.kpis.prbUtilizationDl = 95;
        cell.kpis.prbUtilizationUl = 90;
        cell.kpis.callDropRate += 2;
        break;
      case 'COVERAGE_HOLE':
        cell.kpis.avgRsrp -= 15;
        cell.kpis.avgSinrDl -= 10;
        cell.kpis.callDropRate += 3;
        break;
      case 'PCI_CONFLICT':
        cell.kpis.hoSuccessRate -= 20;
        cell.kpis.hoFailures += 10;
        break;
    }
  }

  /**
   * Update fault durations, remove expired
   */
  updateFaults(): void {
    for (const [cellId, fault] of this.activeFaults) {
      fault.duration--;
      if (fault.duration <= 0) {
        this.activeFaults.delete(cellId);
        this.emit('fault-cleared', { cellId, faultType: fault.type });
      }
    }
  }

  getActiveFaults(): Array<{ cellId: string; type: ProblemCategory }> {
    return Array.from(this.activeFaults.entries()).map(([cellId, fault]) => ({
      cellId,
      type: fault.type
    }));
  }
}

// ============================================================================
// MAIN SWARM SIMULATOR
// ============================================================================

export interface SwarmSimulatorConfig {
  numCells: number;
  numUsers: number;
  areaKm2: number;
  tickIntervalMs: number;
  scenario: 'urban' | 'suburban' | 'rural';
  trafficProfile: 'uniform' | 'hotspot' | 'event' | 'commute';
  faultInjection: FaultInjectionConfig;
}

const DEFAULT_SIM_CONFIG: SwarmSimulatorConfig = {
  numCells: 19,
  numUsers: 1000,
  areaKm2: 4,
  tickIntervalMs: 1000,
  scenario: 'urban',
  trafficProfile: 'uniform',
  faultInjection: {
    enabled: true,
    faultTypes: ['UPLINK_INTERFERENCE', 'CAPACITY_SATURATION', 'SLEEPING_CELL'],
    probability: 0.001
  }
};

export class SwarmSimulator extends EventEmitter {
  config: SwarmSimulatorConfig;
  cellSimulator: CellSimulator;
  mobilityModel: MobilityModel;
  trafficModel: TrafficModel;
  faultInjector: FaultInjector;

  // Simulation state
  tick: number = 0;
  running: boolean = false;
  users: UserState[] = [];
  bounds: { minLat: number; maxLat: number; minLng: number; maxLng: number };

  // Statistics
  totalTicks: number = 0;
  totalHandovers: number = 0;
  totalDrops: number = 0;

  constructor(config: Partial<SwarmSimulatorConfig> = {}) {
    super();
    this.config = { ...DEFAULT_SIM_CONFIG, ...config };

    this.cellSimulator = new CellSimulator();
    this.mobilityModel = new MobilityModel();
    this.trafficModel = new TrafficModel();
    this.faultInjector = new FaultInjector(this.config.faultInjection);

    // Calculate bounds based on area
    const sideLengthDeg = Math.sqrt(this.config.areaKm2) / 111;
    const centerLat = 40.7128;
    const centerLng = -74.0060;
    this.bounds = {
      minLat: centerLat - sideLengthDeg / 2,
      maxLat: centerLat + sideLengthDeg / 2,
      minLng: centerLng - sideLengthDeg / 2,
      maxLng: centerLng + sideLengthDeg / 2
    };

    this.faultInjector.on('fault-injected', (data) => this.emit('fault-injected', data));
    this.faultInjector.on('fault-cleared', (data) => this.emit('fault-cleared', data));
  }

  /**
   * Initialize simulation
   */
  initialize(): void {
    // Create cells in hexagonal pattern
    this.createCellGrid();

    // Generate users
    this.users = this.mobilityModel.generateUsers(
      this.config.numUsers,
      this.bounds,
      this.config.trafficProfile as any
    );

    // Initial cell selection for users
    this.assignUsersToServingCells();

    this.emit('initialized', {
      cellCount: this.cellSimulator.cells.size,
      userCount: this.users.length
    });
  }

  /**
   * Run simulation step
   */
  step(): SimulationState {
    this.tick++;
    this.totalTicks++;

    const hour = (new Date().getHours() + Math.floor(this.tick / 60)) % 24;
    const loadFactor = this.trafficModel.getLoadFactor(hour, this.config.trafficProfile);

    // Update user positions
    for (const user of this.users) {
      this.mobilityModel.updatePosition(user, this.config.tickIntervalMs, this.bounds);
    }

    // Update user RF conditions and handle handovers
    this.updateUserRfConditions();

    // Potentially inject faults
    for (const cellId of this.cellSimulator.cells.keys()) {
      this.faultInjector.maybeInjectFault(cellId, this.tick);
    }

    // Update cell KPIs
    for (const [cellId, cell] of this.cellSimulator.cells) {
      this.cellSimulator.updateKpis(cell, this.users, hour, loadFactor);
      this.faultInjector.applyFaultEffects(cell, cellId);
    }

    // Update fault durations
    this.faultInjector.updateFaults();

    const state: SimulationState = {
      tick: this.tick,
      timestamp: new Date(),
      cells: new Map(Array.from(this.cellSimulator.cells.entries()).map(([k, v]) => [k, v.kpis])),
      users: [...this.users],
      activeProblems: [],
      agentDecisions: []
    };

    this.emit('tick', state);
    return state;
  }

  /**
   * Start continuous simulation
   */
  start(): void {
    if (this.running) return;
    this.running = true;

    const loop = () => {
      if (!this.running) return;
      this.step();
      setTimeout(loop, this.config.tickIntervalMs);
    };

    loop();
    this.emit('started');
  }

  /**
   * Stop simulation
   */
  stop(): void {
    this.running = false;
    this.emit('stopped');
  }

  /**
   * Get current RAN graph for GNN
   */
  getRANGraph(): RANGraph {
    const nodes = new Map<string, RANGraphNode>();
    const edges: RANGraphEdge[] = [];

    // Build nodes
    for (const [cellId, cell] of this.cellSimulator.cells) {
      nodes.set(cellId, {
        id: cellId,
        cgi: cell.cgi,
        staticFeatures: {
          azimuth: cell.antenna.azimuth,
          tilt: cell.antenna.electricalTilt,
          height: cell.antenna.height,
          beamwidth: cell.antenna.beamwidth,
          frequency: 3500,
          bandwidth: 100,
          technology: 'NR',
          latitude: cell.position.lat,
          longitude: cell.position.lng
        },
        dynamicFeatures: {
          txPower: cell.antenna.maxPower,
          load: cell.kpis.prbUtilizationDl,
          rtwp: cell.kpis.rtwp,
          activeUsers: cell.kpis.activeUsers,
          throughput: cell.kpis.dlThroughput
        }
      });
    }

    // Build edges based on distance and interference
    const cellIds = Array.from(this.cellSimulator.cells.keys());
    for (let i = 0; i < cellIds.length; i++) {
      for (let j = i + 1; j < cellIds.length; j++) {
        const cell1 = this.cellSimulator.cells.get(cellIds[i])!;
        const cell2 = this.cellSimulator.cells.get(cellIds[j])!;

        const distance = this.calculateDistance(cell1.position, cell2.position);

        if (distance < 2000) { // Neighbor if within 2km
          edges.push({
            sourceId: cellIds[i],
            targetId: cellIds[j],
            edgeType: 'geographic',
            weight: 1 / (distance + 1),
            features: { distance }
          });

          // Bidirectional
          edges.push({
            sourceId: cellIds[j],
            targetId: cellIds[i],
            edgeType: 'geographic',
            weight: 1 / (distance + 1),
            features: { distance }
          });
        }
      }
    }

    return {
      nodes,
      edges,
      hyperedges: [],
      lastUpdated: new Date()
    };
  }

  /**
   * Get cell KPIs
   */
  getCellKpis(cellId: string): CellKPIs | undefined {
    return this.cellSimulator.cells.get(cellId)?.kpis;
  }

  /**
   * Get all cell KPIs
   */
  getAllCellKpis(): CellKPIs[] {
    return Array.from(this.cellSimulator.cells.values()).map(c => c.kpis);
  }

  /**
   * Apply parameter change to cell
   */
  applyParameterChange(cellId: string, params: Partial<UplinkPowerControlParams & AntennaParams>): void {
    const cell = this.cellSimulator.cells.get(cellId);
    if (!cell) return;

    if (params.p0NominalPusch !== undefined) cell.powerControl.p0NominalPusch = params.p0NominalPusch;
    if (params.alpha !== undefined) cell.powerControl.alpha = params.alpha;
    if (params.electricalTilt !== undefined) cell.antenna.electricalTilt = params.electricalTilt;
    if (params.maxPower !== undefined) cell.antenna.maxPower = params.maxPower;

    this.emit('parameter-changed', { cellId, params });
  }

  /**
   * Get simulation statistics
   */
  getStats(): {
    tick: number;
    cellCount: number;
    userCount: number;
    avgLoad: number;
    avgThroughput: number;
    activeFaults: number;
    totalHandovers: number;
  } {
    const cells = Array.from(this.cellSimulator.cells.values());
    const avgLoad = cells.reduce((sum, c) => sum + c.kpis.prbUtilizationDl, 0) / cells.length;
    const avgThroughput = cells.reduce((sum, c) => sum + c.kpis.dlThroughput, 0) / cells.length;

    return {
      tick: this.tick,
      cellCount: cells.length,
      userCount: this.users.length,
      avgLoad,
      avgThroughput,
      activeFaults: this.faultInjector.getActiveFaults().length,
      totalHandovers: this.totalHandovers
    };
  }

  private createCellGrid(): void {
    // Create hexagonal cell layout
    const numRings = Math.ceil(Math.sqrt(this.config.numCells / 3));
    const cellSpacing = Math.sqrt(this.config.areaKm2 / this.config.numCells) / 111;
    const centerLat = (this.bounds.minLat + this.bounds.maxLat) / 2;
    const centerLng = (this.bounds.minLng + this.bounds.maxLng) / 2;

    let cellCount = 0;
    const azimuths = [0, 120, 240]; // Tri-sector sites

    // Center site
    for (let sector = 0; sector < 3 && cellCount < this.config.numCells; sector++) {
      this.cellSimulator.createCell(1, cellCount++, { lat: centerLat, lng: centerLng }, azimuths[sector]);
    }

    // Surrounding sites in rings
    for (let ring = 1; ring <= numRings && cellCount < this.config.numCells; ring++) {
      const numSites = 6 * ring;
      for (let i = 0; i < numSites && cellCount < this.config.numCells; i++) {
        const angle = (i / numSites) * 2 * Math.PI;
        const lat = centerLat + ring * cellSpacing * Math.cos(angle);
        const lng = centerLng + ring * cellSpacing * Math.sin(angle);

        for (let sector = 0; sector < 3 && cellCount < this.config.numCells; sector++) {
          this.cellSimulator.createCell(
            ring * 100 + i + 1,
            cellCount++,
            { lat, lng },
            azimuths[sector]
          );
        }
      }
    }
  }

  private assignUsersToServingCells(): void {
    for (const user of this.users) {
      let bestCell: string | null = null;
      let bestRsrp = -Infinity;

      for (const [cellId, cell] of this.cellSimulator.cells) {
        const rsrp = this.cellSimulator.calculateRsrp(cell, user.position);
        if (rsrp > bestRsrp) {
          bestRsrp = rsrp;
          bestCell = cellId;
        }
      }

      if (bestCell) {
        user.servingCell = bestCell;
        user.rsrp = bestRsrp;
        this.cellSimulator.cells.get(bestCell)!.connectedUsers.add(user.imsi);
      }
    }
  }

  private updateUserRfConditions(): void {
    for (const user of this.users) {
      const currentCell = this.cellSimulator.cells.get(user.servingCell);
      if (!currentCell) continue;

      // Calculate RSRP from serving and find best cell
      const servingRsrp = this.cellSimulator.calculateRsrp(currentCell, user.position);
      let bestCell = user.servingCell;
      let bestRsrp = servingRsrp;

      // Check neighbors for potential handover
      for (const [cellId, cell] of this.cellSimulator.cells) {
        if (cellId === user.servingCell) continue;
        const rsrp = this.cellSimulator.calculateRsrp(cell, user.position);
        if (rsrp > bestRsrp + 3) { // A3 offset
          bestRsrp = rsrp;
          bestCell = cellId;
        }
      }

      // Perform handover if needed
      if (bestCell !== user.servingCell) {
        currentCell.connectedUsers.delete(user.imsi);
        this.cellSimulator.cells.get(bestCell)!.connectedUsers.add(user.imsi);
        user.servingCell = bestCell;
        this.totalHandovers++;
      }

      // Update user metrics
      user.rsrp = bestRsrp;
      user.sinr = bestRsrp - (-110) + Math.random() * 5; // Simplified SINR
      user.throughput = this.trafficModel.calculateThroughput(user.sinr);
    }
  }

  private calculateDistance(pos1: { lat: number; lng: number }, pos2: { lat: number; lng: number }): number {
    const R = 6371000;
    const dLat = (pos2.lat - pos1.lat) * Math.PI / 180;
    const dLng = (pos2.lng - pos1.lng) * Math.PI / 180;
    const a = Math.sin(dLat / 2) ** 2 + Math.cos(pos1.lat * Math.PI / 180) * Math.cos(pos2.lat * Math.PI / 180) * Math.sin(dLng / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }
}
