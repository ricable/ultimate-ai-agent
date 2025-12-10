/**
 * TITAN Dashboard Demo
 * Demonstrates the full dashboard capabilities with mock data
 *
 * @module ui/demo
 * @version 7.0.0-alpha.1
 */

import { TitanDashboard } from './titan-dashboard.js';
import { TitanAPIServer } from './api-server.js';
import type { CellStatus, InterferenceMatrix, OptimizationEvent, FMAlarm } from './types.js';

// ============================================================================
// Mock Data Generation
// ============================================================================

function generateMockCells(count: number): CellStatus[] {
  const cells: CellStatus[] = [];

  for (let i = 0; i < count; i++) {
    cells.push({
      cell_id: `Cell-${String(i).padStart(3, '0')}`,
      pci: 100 + i,
      earfcn: 6300 + i * 10,
      power_dbm: -5 + Math.random() * 10,
      tilt_deg: 3 + Math.random() * 8,
      azimuth_deg: (i * 120) % 360,
      latitude: 37.7749 + (Math.random() - 0.5) * 0.1,
      longitude: -122.4194 + (Math.random() - 0.5) * 0.1,
      status: ['active', 'degraded', 'active', 'active'][Math.floor(Math.random() * 4)] as any,
      kpi: {
        rsrp_avg: -85 + Math.random() * 30,
        rsrq_avg: -10 + Math.random() * 5,
        sinr_avg: 10 + Math.random() * 15,
        throughput_mbps: 50 + Math.random() * 100,
        rrc_connections: Math.floor(Math.random() * 500),
        prb_utilization: 30 + Math.random() * 50,
        ho_success_rate: 95 + Math.random() * 4,
        drop_rate: Math.random() * 2
      },
      last_optimized: new Date(Date.now() - Math.random() * 86400000).toISOString()
    });
  }

  return cells;
}

function generateInterferenceMatrix(cells: CellStatus[]): InterferenceMatrix {
  const n = cells.length;
  const matrix: number[][] = [];

  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) {
        matrix[i][j] = 0; // No self-interference
      } else {
        // Distance-based interference simulation
        const cellA = cells[i];
        const cellB = cells[j];
        const distance = Math.sqrt(
          Math.pow(cellA.latitude - cellB.latitude, 2) +
          Math.pow(cellA.longitude - cellB.longitude, 2)
        );
        const interference = -130 + (1 / (distance * 1000 + 0.1)) * 40;
        matrix[i][j] = interference;
      }
    }
  }

  return {
    cells: cells.map(c => c.cell_id),
    matrix,
    threshold: -90,
    timestamp: new Date().toISOString()
  };
}

function generateMockAlarms(cells: CellStatus[]): FMAlarm[] {
  const alarms: FMAlarm[] = [];

  // Random alarms for degraded cells
  for (const cell of cells) {
    if (cell.status === 'degraded' || Math.random() < 0.1) {
      alarms.push({
        id: `alarm_${Date.now()}_${cell.cell_id}`,
        severity: ['critical', 'major', 'minor', 'warning'][Math.floor(Math.random() * 4)] as any,
        mo_class: 'EUtranCellFDD',
        mo_id: cell.cell_id,
        alarm_text: [
          'High interference detected',
          'PRB utilization exceeds threshold',
          'Handover failure rate high',
          'Low SINR detected'
        ][Math.floor(Math.random() * 4)],
        raised_at: new Date(Date.now() - Math.random() * 3600000).toISOString()
      });
    }
  }

  return alarms;
}

function generateMockOptimizationEvents(cells: CellStatus[]): OptimizationEvent[] {
  const events: OptimizationEvent[] = [];
  const eventTypes: OptimizationEvent['event_type'][] = [
    'gnn_decision',
    'council_debate',
    'hitl_approval',
    'execution',
    'rollback'
  ];

  for (let i = 0; i < 20; i++) {
    const targetCells = cells
      .slice(0, Math.floor(Math.random() * 3) + 1)
      .map(c => c.cell_id);

    events.push({
      id: `evt_${Date.now()}_${i}`,
      timestamp: new Date(Date.now() - i * 300000).toISOString(), // 5 min intervals
      event_type: eventTypes[Math.floor(Math.random() * eventTypes.length)],
      cell_ids: targetCells,
      parameters_changed: targetCells.map(cellId => ({
        parameter: ['power_dbm', 'tilt_deg'][Math.floor(Math.random() * 2)],
        old_value: 5 + Math.random() * 5,
        new_value: 5 + Math.random() * 5,
        cell_id: cellId,
        bounds: [0, 15]
      })),
      reasoning: [
        'Detected high interference, optimizing power levels',
        'Load balancing required across neighboring cells',
        'Coverage hole identified, increasing tilt',
        'Rollback due to KPI degradation'
      ][Math.floor(Math.random() * 4)],
      confidence: 0.7 + Math.random() * 0.3,
      status: ['executed', 'approved', 'pending'][Math.floor(Math.random() * 3)] as any,
      agent_id: `agent_${Math.floor(Math.random() * 5)}`
    });
  }

  return events.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
}

// ============================================================================
// Demo Runner
// ============================================================================

async function runDemo() {
  console.log('='.repeat(80));
  console.log('TITAN RAN DASHBOARD DEMO');
  console.log('AG-UI + OpenTUI Integration');
  console.log('='.repeat(80));
  console.log('');

  // Generate mock data
  console.log('[Demo] Generating mock data...');
  const cells = generateMockCells(10);
  const interferenceMatrix = generateInterferenceMatrix(cells);
  const alarms = generateMockAlarms(cells);
  const optimizationEvents = generateMockOptimizationEvents(cells);

  console.log(`[Demo] Generated ${cells.length} cells`);
  console.log(`[Demo] Generated ${interferenceMatrix.matrix.length}x${interferenceMatrix.matrix.length} interference matrix`);
  console.log(`[Demo] Generated ${alarms.length} alarms`);
  console.log(`[Demo] Generated ${optimizationEvents.length} optimization events`);
  console.log('');

  // Initialize dashboard
  console.log('[Demo] Initializing TITAN Dashboard...');
  const dashboard = new TitanDashboard({
    port: 8080,
    aguiPort: 3000,
    sseUpdateInterval: 10000,
    enableOpenTUI: true,
    enableAGUI: true
  });

  // Start dashboard
  await dashboard.start();
  console.log('');

  // Populate with mock data
  console.log('[Demo] Populating dashboard with mock data...');
  for (const cell of cells) {
    dashboard.updateCellStatus(cell);
  }

  for (const alarm of alarms) {
    dashboard.addAlarm(alarm);
  }

  for (const event of optimizationEvents) {
    dashboard.addOptimizationEvent(event);
  }

  // Render visualizations
  console.log('[Demo] Rendering visualizations...');
  dashboard.renderInterferenceHeatmap(cells, interferenceMatrix, -90);
  dashboard.renderOptimizationTimeline();
  dashboard.renderParameterControls(cells.slice(0, 3).map(c => c.cell_id));
  console.log('');

  // Create approval request
  console.log('[Demo] Creating mock approval request...');
  const approval = dashboard.createApprovalRequest({
    action: 'Optimize power levels for high-interference cells',
    target: cells.slice(0, 3).map(c => c.cell_id),
    changes: [
      {
        parameter: 'power_dbm',
        old_value: cells[0].power_dbm,
        new_value: cells[0].power_dbm - 2,
        cell_id: cells[0].cell_id,
        bounds: [-130, 46]
      }
    ],
    riskLevel: 'medium',
    justification: 'GNN detected excessive interference pattern'
  });
  dashboard.renderApprovalQueue();
  console.log(`[Demo] Created approval request: ${approval.id}`);
  console.log('');

  // Start API server
  console.log('[Demo] Starting API server...');
  const apiServer = new TitanAPIServer(dashboard, 8080);
  await apiServer.start();
  console.log('');

  // Display OpenTUI terminal views
  console.log('[Demo] Displaying OpenTUI terminal views...');
  console.log('');
  dashboard['showCellTable']();
  console.log('');
  dashboard['showAlarmList']();
  console.log('');

  // Simulate real-time updates
  console.log('[Demo] Simulating real-time KPI updates (press Ctrl+C to stop)...');
  console.log('');

  let updateCounter = 0;
  const updateInterval = setInterval(() => {
    updateCounter++;

    // Update random cell KPIs
    const randomCell = cells[Math.floor(Math.random() * cells.length)];
    randomCell.kpi.rsrp_avg += (Math.random() - 0.5) * 2;
    randomCell.kpi.sinr_avg += (Math.random() - 0.5);
    randomCell.kpi.prb_utilization += (Math.random() - 0.5) * 5;
    dashboard.updateCellStatus(randomCell);

    if (updateCounter % 5 === 0) {
      console.log(`[Demo] Update #${updateCounter} - Cell ${randomCell.cell_id} KPIs updated`);
    }

    // Occasionally add new optimization event
    if (updateCounter % 10 === 0) {
      dashboard.addOptimizationEvent({
        id: `evt_realtime_${Date.now()}`,
        timestamp: new Date().toISOString(),
        event_type: 'gnn_decision',
        cell_ids: [randomCell.cell_id],
        parameters_changed: [],
        reasoning: 'Real-time optimization triggered',
        confidence: 0.85,
        status: 'pending'
      });
      console.log(`[Demo] Added new optimization event for ${randomCell.cell_id}`);
    }
  }, 5000);

  // Cleanup on exit
  process.on('SIGINT', async () => {
    console.log('\n\n[Demo] Shutting down...');
    clearInterval(updateInterval);
    await apiServer.stop();
    await dashboard.stop();
    console.log('[Demo] Goodbye!');
    process.exit(0);
  });

  // Keep demo running
  console.log('[Demo] Dashboard running. Access:');
  console.log(`  - API Server:     http://localhost:8080`);
  console.log(`  - Health Check:   http://localhost:8080/health`);
  console.log(`  - KPI Stream:     http://localhost:8080/api/stream/kpi`);
  console.log(`  - Event Stream:   http://localhost:8080/api/stream/events`);
  console.log(`  - AG-UI Server:   http://localhost:3000`);
  console.log('');
  console.log('Press Ctrl+C to exit');
}

// ============================================================================
// Run Demo
// ============================================================================

if (import.meta.url === `file://${process.argv[1]}`) {
  runDemo().catch(error => {
    console.error('[Demo] Fatal error:', error);
    process.exit(1);
  });
}

export { runDemo };
