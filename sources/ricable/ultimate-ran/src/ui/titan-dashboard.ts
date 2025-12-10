/**
 * TITAN Dashboard - AG-UI + OpenTUI Integration
 * Real-time RAN optimization dashboard with Glass Box observability
 *
 * @module ui/titan-dashboard
 * @version 7.0.0-alpha.1
 */

import { EventEmitter } from 'events';
import { AGUIServer } from '../agui/server.js';
import type {
  CellStatus,
  FMAlarm,
  PMCounter,
  OptimizationEvent,
  ApprovalRequest,
  DashboardState,
  InterferenceMatrix,
  ParameterChange
} from './types.js';

// ============================================================================
// TitanDashboard Class
// ============================================================================

export interface TitanDashboardConfig {
  port?: number;
  aguiPort?: number;
  sseUpdateInterval?: number;
  enableOpenTUI?: boolean;
  enableAGUI?: boolean;
}

export class TitanDashboard extends EventEmitter {
  private aguiServer: AGUIServer;
  private state: DashboardState;
  private sseClients: Set<any>;
  private config: Required<TitanDashboardConfig>;
  private updateInterval?: NodeJS.Timeout;

  constructor(config: TitanDashboardConfig = {}) {
    super();

    this.config = {
      port: config.port || 8080,
      aguiPort: config.aguiPort || 3000,
      sseUpdateInterval: config.sseUpdateInterval || 10000, // 10 seconds
      enableOpenTUI: config.enableOpenTUI !== false,
      enableAGUI: config.enableAGUI !== false
    };

    // Initialize AG-UI server
    this.aguiServer = new AGUIServer({
      port: this.config.aguiPort,
      protocolPath: './config/ag-ui/protocol.json'
    });

    // Initialize state
    this.state = {
      cells: new Map(),
      alarms: [],
      optimizationTimeline: [],
      pendingApprovals: [],
      selectedCells: [],
      viewMode: 'overview'
    };

    this.sseClients = new Set();
  }

  // ==========================================================================
  // Lifecycle Methods
  // ==========================================================================

  async start(): Promise<void> {
    console.log('[TITAN Dashboard] Starting dashboard services...');

    // Start AG-UI server
    if (this.config.enableAGUI) {
      await this.aguiServer.start();
      console.log(`[TITAN Dashboard] AG-UI server running on port ${this.config.aguiPort}`);
    }

    // Start SSE update loop
    this.startSSEUpdates();

    // Start OpenTUI if enabled
    if (this.config.enableOpenTUI) {
      this.initializeOpenTUI();
    }

    console.log(`[TITAN Dashboard] Dashboard ready on port ${this.config.port}`);
    this.emit('ready');
  }

  async stop(): Promise<void> {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    console.log('[TITAN Dashboard] Dashboard stopped');
  }

  // ==========================================================================
  // Real-time KPI Heatmap
  // ==========================================================================

  renderInterferenceHeatmap(
    cells: CellStatus[],
    interferenceMatrix: InterferenceMatrix,
    threshold: number = -90
  ): void {
    console.log('[TITAN Dashboard] Rendering interference heatmap...');

    // Update state
    this.state.interferenceMatrix = interferenceMatrix;

    // Send to AG-UI
    if (this.config.enableAGUI) {
      this.aguiServer.renderInterferenceHeatmap(
        cells,
        interferenceMatrix.matrix,
        threshold
      );
    }

    // Broadcast via SSE
    this.broadcastSSE({
      type: 'interference_heatmap_update',
      data: {
        cells: cells.map(c => ({
          id: c.cell_id,
          lat: c.latitude,
          lon: c.longitude,
          power: c.power_dbm,
          status: c.status
        })),
        matrix: interferenceMatrix.matrix,
        threshold
      }
    });

    this.emit('heatmap_rendered', { cells, interferenceMatrix });
  }

  // ==========================================================================
  // GNN Optimization Timeline
  // ==========================================================================

  renderOptimizationTimeline(events?: OptimizationEvent[]): void {
    if (events) {
      this.state.optimizationTimeline = events;
    }

    console.log(
      `[TITAN Dashboard] Rendering optimization timeline (${this.state.optimizationTimeline.length} events)...`
    );

    // Send to AG-UI as custom component
    if (this.config.enableAGUI) {
      this.aguiServer.renderComponent('OptimizationTimeline', {
        events: this.state.optimizationTimeline,
        maxEvents: 100
      });
    }

    // Broadcast via SSE
    this.broadcastSSE({
      type: 'optimization_timeline_update',
      data: {
        events: this.state.optimizationTimeline.slice(-50) // Last 50 events
      }
    });

    this.emit('timeline_rendered', { events: this.state.optimizationTimeline });
  }

  addOptimizationEvent(event: OptimizationEvent): void {
    this.state.optimizationTimeline.push(event);

    // Keep only last 1000 events
    if (this.state.optimizationTimeline.length > 1000) {
      this.state.optimizationTimeline = this.state.optimizationTimeline.slice(-1000);
    }

    this.renderOptimizationTimeline();
  }

  // ==========================================================================
  // P0/Alpha Parameter Controls
  // ==========================================================================

  renderParameterControls(cellIds: string[]): void {
    console.log(`[TITAN Dashboard] Rendering parameter controls for ${cellIds.length} cells...`);

    const cells = cellIds.map(id => this.state.cells.get(id)).filter(Boolean) as CellStatus[];

    // Send to AG-UI
    if (this.config.enableAGUI) {
      this.aguiServer.renderComponent('ParameterControls', {
        cells,
        parameters: [
          {
            name: 'power_dbm',
            label: 'Transmit Power (dBm)',
            min: -130,
            max: 46,
            step: 0.1,
            unit: 'dBm'
          },
          {
            name: 'tilt_deg',
            label: 'Antenna Tilt (degrees)',
            min: 0,
            max: 15,
            step: 0.5,
            unit: '°'
          },
          {
            name: 'azimuth_deg',
            label: 'Azimuth (degrees)',
            min: 0,
            max: 360,
            step: 1,
            unit: '°'
          }
        ]
      });
    }

    this.emit('parameter_controls_rendered', { cells });
  }

  async updateParameter(
    cellId: string,
    parameter: string,
    value: number,
    requireApproval: boolean = true
  ): Promise<void> {
    const cell = this.state.cells.get(cellId);
    if (!cell) {
      throw new Error(`Cell ${cellId} not found`);
    }

    const change: ParameterChange = {
      parameter,
      old_value: (cell as any)[parameter],
      new_value: value,
      cell_id: cellId,
      bounds: this.getParameterBounds(parameter)
    };

    if (requireApproval) {
      // Create approval request
      const approval = this.createApprovalRequest({
        action: `Update ${parameter} for cell ${cellId}`,
        target: [cellId],
        changes: [change],
        riskLevel: this.assessRiskLevel(change)
      });

      this.renderApprovalQueue();
    } else {
      // Direct update (low-risk changes)
      await this.executeParameterChange(change);
    }
  }

  private getParameterBounds(parameter: string): [number, number] {
    const bounds: Record<string, [number, number]> = {
      power_dbm: [-130, 46],
      tilt_deg: [0, 15],
      azimuth_deg: [0, 360]
    };
    return bounds[parameter] || [0, 100];
  }

  private assessRiskLevel(change: ParameterChange): 'low' | 'medium' | 'high' | 'critical' {
    const delta = Math.abs(change.new_value - change.old_value);
    const range = change.bounds[1] - change.bounds[0];
    const percentChange = (delta / range) * 100;

    if (percentChange < 5) return 'low';
    if (percentChange < 15) return 'medium';
    if (percentChange < 30) return 'high';
    return 'critical';
  }

  private async executeParameterChange(change: ParameterChange): Promise<void> {
    const cell = this.state.cells.get(change.cell_id);
    if (!cell) return;

    (cell as any)[change.parameter] = change.new_value;

    console.log(
      `[TITAN Dashboard] Executed parameter change: ${change.parameter} = ${change.new_value} for cell ${change.cell_id}`
    );

    // Create optimization event
    this.addOptimizationEvent({
      id: `evt_${Date.now()}`,
      timestamp: new Date().toISOString(),
      event_type: 'execution',
      cell_ids: [change.cell_id],
      parameters_changed: [change],
      reasoning: 'Manual parameter adjustment via dashboard',
      confidence: 1.0,
      status: 'executed'
    });

    this.emit('parameter_updated', { change });
  }

  // ==========================================================================
  // HITL Approval Queue
  // ==========================================================================

  renderApprovalQueue(): void {
    console.log(
      `[TITAN Dashboard] Rendering approval queue (${this.state.pendingApprovals.length} pending)...`
    );

    // Send to AG-UI
    if (this.config.enableAGUI) {
      this.aguiServer.renderComponent('ApprovalQueue', {
        approvals: this.state.pendingApprovals,
        sortBy: 'risk_level' // critical first
      });
    }

    // Broadcast via SSE
    this.broadcastSSE({
      type: 'approval_queue_update',
      data: {
        pending: this.state.pendingApprovals.filter(a => a.status === 'pending'),
        recent: this.state.pendingApprovals.filter(a => a.status !== 'pending').slice(-10)
      }
    });

    this.emit('approval_queue_rendered', {
      approvals: this.state.pendingApprovals
    });
  }

  createApprovalRequest(params: {
    action: string;
    target: string[];
    changes: ParameterChange[];
    riskLevel: 'low' | 'medium' | 'high' | 'critical';
    justification?: string;
  }): ApprovalRequest {
    const approval: ApprovalRequest = {
      id: `approval_${Date.now()}`,
      risk_level: params.riskLevel,
      action: params.action,
      target: params.target,
      justification: params.justification || 'Automated optimization proposal',
      proposed_changes: params.changes,
      predicted_impact: this.predictImpact(params.changes),
      safety_checks: this.runSafetyChecks(params.changes),
      status: 'pending',
      created_at: new Date().toISOString(),
      expires_at: new Date(Date.now() + 3600000).toISOString() // 1 hour expiry
    };

    this.state.pendingApprovals.push(approval);

    // Notify AG-UI
    if (this.config.enableAGUI) {
      this.aguiServer.emit('request_approval', approval);
    }

    return approval;
  }

  async processApproval(approvalId: string, authorized: boolean, signature: string): Promise<void> {
    const approval = this.state.pendingApprovals.find(a => a.id === approvalId);

    if (!approval) {
      throw new Error(`Approval ${approvalId} not found`);
    }

    approval.status = authorized ? 'approved' : 'rejected';

    console.log(`[TITAN Dashboard] Approval ${approvalId}: ${approval.status.toUpperCase()}`);

    if (authorized) {
      // Execute the approved changes
      for (const change of approval.proposed_changes) {
        await this.executeParameterChange(change);
      }
    }

    // Notify AG-UI
    if (this.config.enableAGUI) {
      await this.aguiServer.processApproval(approvalId, authorized, signature);
    }

    this.renderApprovalQueue();
    this.emit('approval_processed', { approvalId, authorized });
  }

  private predictImpact(changes: ParameterChange[]): string {
    // Simplified impact prediction
    const impacts: string[] = [];

    for (const change of changes) {
      if (change.parameter === 'power_dbm') {
        const delta = change.new_value - change.old_value;
        impacts.push(
          `Cell ${change.cell_id}: ${delta > 0 ? 'Increased' : 'Decreased'} coverage by ~${Math.abs(delta * 5).toFixed(0)}m`
        );
      } else if (change.parameter === 'tilt_deg') {
        impacts.push(`Cell ${change.cell_id}: Modified vertical coverage pattern`);
      }
    }

    return impacts.join('; ');
  }

  private runSafetyChecks(changes: ParameterChange[]) {
    return changes.map(change => ({
      check_name: '3GPP Bounds Check',
      passed: change.new_value >= change.bounds[0] && change.new_value <= change.bounds[1],
      message: `Value ${change.new_value} within bounds [${change.bounds[0]}, ${change.bounds[1]}]`,
      severity: 'info' as const
    }));
  }

  // ==========================================================================
  // SSE Streaming
  // ==========================================================================

  private startSSEUpdates(): void {
    this.updateInterval = setInterval(() => {
      this.broadcastKPIUpdates();
    }, this.config.sseUpdateInterval);
  }

  private broadcastKPIUpdates(): void {
    const kpiData = Array.from(this.state.cells.values()).map(cell => ({
      cell_id: cell.cell_id,
      kpi: cell.kpi,
      status: cell.status,
      timestamp: new Date().toISOString()
    }));

    this.broadcastSSE({
      type: 'kpi_update',
      data: kpiData
    });
  }

  private broadcastSSE(message: any): void {
    const data = JSON.stringify(message);

    for (const client of this.sseClients) {
      try {
        client.write(`data: ${data}\n\n`);
      } catch (error) {
        console.error('[TITAN Dashboard] SSE write error:', error);
        this.sseClients.delete(client);
      }
    }
  }

  registerSSEClient(client: any): void {
    this.sseClients.add(client);
    console.log(`[TITAN Dashboard] SSE client connected (total: ${this.sseClients.size})`);

    // Send initial state
    client.write(
      `data: ${JSON.stringify({
        type: 'initial_state',
        data: {
          cellCount: this.state.cells.size,
          alarmCount: this.state.alarms.length,
          pendingApprovals: this.state.pendingApprovals.filter(a => a.status === 'pending').length
        }
      })}\n\n`
    );
  }

  unregisterSSEClient(client: any): void {
    this.sseClients.delete(client);
    console.log(`[TITAN Dashboard] SSE client disconnected (total: ${this.sseClients.size})`);
  }

  // ==========================================================================
  // OpenTUI Terminal Interface
  // ==========================================================================

  private initializeOpenTUI(): void {
    console.log('[TITAN Dashboard] Initializing OpenTUI terminal interface...');

    // In production, this would initialize blessed/ink terminal UI
    // For now, we provide CLI commands

    this.registerCommands();
  }

  private registerCommands(): void {
    // Command palette actions
    const commands = [
      {
        id: 'show_cells',
        label: 'Show Cell Status',
        description: 'Display all cell statuses in table format',
        handler: () => this.showCellTable()
      },
      {
        id: 'show_alarms',
        label: 'Show Alarms',
        description: 'Display FM alarms',
        handler: () => this.showAlarmList()
      },
      {
        id: 'show_kpi_chart',
        label: 'Show KPI Chart',
        description: 'Display KPI trends',
        handler: () => this.showKPIChart()
      },
      {
        id: 'approve_pending',
        label: 'Approve Pending Actions',
        description: 'Review and approve pending optimizations',
        handler: () => this.showApprovalQueue()
      }
    ];

    console.log(`[TITAN Dashboard] Registered ${commands.length} OpenTUI commands`);
  }

  private showCellTable(): void {
    console.log('\n=== CELL STATUS TABLE ===\n');
    console.log(
      'CELL_ID       | PCI  | POWER  | TILT | STATUS     | RSRP    | SINR   | PRB%'
    );
    console.log('-'.repeat(80));

    for (const cell of this.state.cells.values()) {
      console.log(
        `${cell.cell_id.padEnd(13)} | ${cell.pci.toString().padStart(4)} | ${cell.power_dbm.toFixed(1).padStart(6)} | ${cell.tilt_deg.toFixed(1).padStart(4)} | ${cell.status.padEnd(10)} | ${cell.kpi.rsrp_avg.toFixed(1).padStart(7)} | ${cell.kpi.sinr_avg.toFixed(1).padStart(6)} | ${cell.kpi.prb_utilization.toFixed(0).padStart(4)}`
      );
    }
    console.log('');
  }

  private showAlarmList(): void {
    console.log('\n=== FM ALARMS ===\n');

    if (this.state.alarms.length === 0) {
      console.log('No active alarms.\n');
      return;
    }

    for (const alarm of this.state.alarms) {
      console.log(`[${alarm.severity.toUpperCase()}] ${alarm.alarm_text}`);
      console.log(`  MO: ${alarm.mo_class}/${alarm.mo_id}`);
      console.log(`  Raised: ${alarm.raised_at}`);
      console.log('');
    }
  }

  private showKPIChart(): void {
    console.log('\n=== KPI TRENDS (Last 24h) ===\n');
    console.log('(Chart rendering requires terminal with graphics support)');
    console.log('');
  }

  private showApprovalQueue(): void {
    console.log('\n=== APPROVAL QUEUE ===\n');

    const pending = this.state.pendingApprovals.filter(a => a.status === 'pending');

    if (pending.length === 0) {
      console.log('No pending approvals.\n');
      return;
    }

    for (const approval of pending) {
      console.log(`[${approval.risk_level.toUpperCase()}] ${approval.action}`);
      console.log(`  ID: ${approval.id}`);
      console.log(`  Target: ${approval.target.join(', ')}`);
      console.log(`  Justification: ${approval.justification}`);
      console.log(`  Predicted Impact: ${approval.predicted_impact}`);
      console.log('');
    }
  }

  // ==========================================================================
  // State Management
  // ==========================================================================

  updateCellStatus(cell: CellStatus): void {
    this.state.cells.set(cell.cell_id, cell);
    this.emit('cell_updated', { cell });
  }

  addAlarm(alarm: FMAlarm): void {
    this.state.alarms.push(alarm);
    this.emit('alarm_added', { alarm });
  }

  clearAlarm(alarmId: string): void {
    const alarm = this.state.alarms.find(a => a.id === alarmId);
    if (alarm) {
      alarm.cleared_at = new Date().toISOString();
      this.emit('alarm_cleared', { alarm });
    }
  }

  getState(): Readonly<DashboardState> {
    return { ...this.state };
  }
}
