/**
 * TITAN Dashboard API Server
 * Express-based HTTP/SSE server for real-time dashboard updates
 *
 * @module ui/api-server
 * @version 7.0.0-alpha.1
 */

import { createServer, IncomingMessage, ServerResponse } from 'http';
import { EventEmitter } from 'events';
import { TitanDashboard } from './titan-dashboard.js';
import type { CellStatus, FMAlarm, PMCounter, OptimizationEvent, ApprovalRequest } from './types.js';

// ============================================================================
// Types
// ============================================================================

interface APIRoute {
  method: string;
  path: string;
  handler: (req: IncomingMessage, res: ServerResponse, params?: any) => void | Promise<void>;
}

// ============================================================================
// TitanAPIServer Class
// ============================================================================

export class TitanAPIServer extends EventEmitter {
  private dashboard: TitanDashboard;
  private server: any;
  private port: number;
  private routes: APIRoute[];

  constructor(dashboard: TitanDashboard, port: number = 8080) {
    super();
    this.dashboard = dashboard;
    this.port = port;
    this.routes = [];

    this.setupRoutes();
  }

  // ==========================================================================
  // Route Setup
  // ==========================================================================

  private setupRoutes(): void {
    // SSE Streaming Endpoints
    this.addRoute('GET', '/api/stream/kpi', this.handleKPIStream.bind(this));
    this.addRoute('GET', '/api/stream/events', this.handleEventStream.bind(this));
    this.addRoute('GET', '/api/stream/alarms', this.handleAlarmStream.bind(this));
    this.addRoute('GET', '/api/stream/approvals', this.handleApprovalStream.bind(this));

    // REST Endpoints
    this.addRoute('GET', '/api/cells', this.handleGetCells.bind(this));
    this.addRoute('GET', '/api/cells/:cellId', this.handleGetCell.bind(this));
    this.addRoute('POST', '/api/cells/:cellId/parameters', this.handleUpdateParameter.bind(this));

    this.addRoute('GET', '/api/alarms', this.handleGetAlarms.bind(this));
    this.addRoute('POST', '/api/alarms/:alarmId/clear', this.handleClearAlarm.bind(this));

    this.addRoute('GET', '/api/optimization/events', this.handleGetOptimizationEvents.bind(this));
    this.addRoute('POST', '/api/optimization/events', this.handleCreateOptimizationEvent.bind(this));

    this.addRoute('GET', '/api/approvals', this.handleGetApprovals.bind(this));
    this.addRoute('POST', '/api/approvals/:approvalId/approve', this.handleApproveRequest.bind(this));
    this.addRoute('POST', '/api/approvals/:approvalId/reject', this.handleRejectRequest.bind(this));

    this.addRoute('GET', '/api/interference/matrix', this.handleGetInterferenceMatrix.bind(this));

    // Health check
    this.addRoute('GET', '/health', this.handleHealth.bind(this));
  }

  private addRoute(method: string, path: string, handler: APIRoute['handler']): void {
    this.routes.push({ method, path, handler });
  }

  // ==========================================================================
  // SSE Stream Handlers
  // ==========================================================================

  private async handleKPIStream(req: IncomingMessage, res: ServerResponse): Promise<void> {
    console.log('[API Server] KPI stream client connected');

    // Set SSE headers
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    });

    // Register client with dashboard
    this.dashboard.registerSSEClient(res);

    // Send initial heartbeat
    res.write('data: {"type":"connected","timestamp":"' + new Date().toISOString() + '"}\n\n');

    // Stream PM counters every 10 seconds
    const interval = setInterval(() => {
      const state = this.dashboard.getState();
      const kpiData: PMCounter[] = [];

      for (const cell of state.cells.values()) {
        kpiData.push(
          { counter_name: 'rsrp_avg', value: cell.kpi.rsrp_avg, cell_id: cell.cell_id, timestamp: new Date().toISOString() },
          { counter_name: 'sinr_avg', value: cell.kpi.sinr_avg, cell_id: cell.cell_id, timestamp: new Date().toISOString() },
          { counter_name: 'throughput_mbps', value: cell.kpi.throughput_mbps, cell_id: cell.cell_id, timestamp: new Date().toISOString() },
          { counter_name: 'prb_utilization', value: cell.kpi.prb_utilization, cell_id: cell.cell_id, timestamp: new Date().toISOString() }
        );
      }

      try {
        res.write(`data: ${JSON.stringify({ type: 'kpi_update', counters: kpiData })}\n\n`);
      } catch (error) {
        console.error('[API Server] SSE write error:', error);
        clearInterval(interval);
        this.dashboard.unregisterSSEClient(res);
      }
    }, 10000);

    // Handle client disconnect
    req.on('close', () => {
      console.log('[API Server] KPI stream client disconnected');
      clearInterval(interval);
      this.dashboard.unregisterSSEClient(res);
    });
  }

  private async handleEventStream(req: IncomingMessage, res: ServerResponse): Promise<void> {
    console.log('[API Server] Event stream client connected');

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    });

    this.dashboard.registerSSEClient(res);

    res.write('data: {"type":"connected"}\n\n');

    // Listen to optimization events
    const eventHandler = (data: { events: OptimizationEvent[] }) => {
      try {
        res.write(`data: ${JSON.stringify({ type: 'optimization_events', events: data.events })}\n\n`);
      } catch (error) {
        console.error('[API Server] Event stream write error:', error);
      }
    };

    this.dashboard.on('timeline_rendered', eventHandler);

    req.on('close', () => {
      console.log('[API Server] Event stream client disconnected');
      this.dashboard.off('timeline_rendered', eventHandler);
      this.dashboard.unregisterSSEClient(res);
    });
  }

  private async handleAlarmStream(req: IncomingMessage, res: ServerResponse): Promise<void> {
    console.log('[API Server] Alarm stream client connected');

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    });

    res.write('data: {"type":"connected"}\n\n');

    // Listen to alarm events
    const alarmAddedHandler = (data: { alarm: FMAlarm }) => {
      try {
        res.write(`data: ${JSON.stringify({ type: 'alarm_added', alarm: data.alarm })}\n\n`);
      } catch (error) {
        console.error('[API Server] Alarm stream write error:', error);
      }
    };

    const alarmClearedHandler = (data: { alarm: FMAlarm }) => {
      try {
        res.write(`data: ${JSON.stringify({ type: 'alarm_cleared', alarm: data.alarm })}\n\n`);
      } catch (error) {
        console.error('[API Server] Alarm stream write error:', error);
      }
    };

    this.dashboard.on('alarm_added', alarmAddedHandler);
    this.dashboard.on('alarm_cleared', alarmClearedHandler);

    req.on('close', () => {
      console.log('[API Server] Alarm stream client disconnected');
      this.dashboard.off('alarm_added', alarmAddedHandler);
      this.dashboard.off('alarm_cleared', alarmClearedHandler);
    });
  }

  private async handleApprovalStream(req: IncomingMessage, res: ServerResponse): Promise<void> {
    console.log('[API Server] Approval stream client connected');

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*'
    });

    this.dashboard.registerSSEClient(res);

    res.write('data: {"type":"connected"}\n\n');

    // Listen to approval events
    const approvalQueueHandler = (data: { approvals: ApprovalRequest[] }) => {
      try {
        res.write(`data: ${JSON.stringify({ type: 'approval_queue_update', approvals: data.approvals })}\n\n`);
      } catch (error) {
        console.error('[API Server] Approval stream write error:', error);
      }
    };

    this.dashboard.on('approval_queue_rendered', approvalQueueHandler);

    req.on('close', () => {
      console.log('[API Server] Approval stream client disconnected');
      this.dashboard.off('approval_queue_rendered', approvalQueueHandler);
      this.dashboard.unregisterSSEClient(res);
    });
  }

  // ==========================================================================
  // REST Handlers
  // ==========================================================================

  private async handleGetCells(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const state = this.dashboard.getState();
    const cells = Array.from(state.cells.values());

    this.sendJSON(res, { cells });
  }

  private async handleGetCell(req: IncomingMessage, res: ServerResponse, params: any): Promise<void> {
    const state = this.dashboard.getState();
    const cell = state.cells.get(params.cellId);

    if (!cell) {
      this.sendError(res, 404, `Cell ${params.cellId} not found`);
      return;
    }

    this.sendJSON(res, { cell });
  }

  private async handleUpdateParameter(req: IncomingMessage, res: ServerResponse, params: any): Promise<void> {
    const body = await this.parseBody(req);

    if (!body.parameter || typeof body.value !== 'number') {
      this.sendError(res, 400, 'Missing parameter or value');
      return;
    }

    try {
      await this.dashboard.updateParameter(
        params.cellId,
        body.parameter,
        body.value,
        body.requireApproval ?? true
      );

      this.sendJSON(res, { success: true, message: 'Parameter update initiated' });
    } catch (error: any) {
      this.sendError(res, 500, error.message);
    }
  }

  private async handleGetAlarms(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const state = this.dashboard.getState();
    this.sendJSON(res, { alarms: state.alarms });
  }

  private async handleClearAlarm(req: IncomingMessage, res: ServerResponse, params: any): Promise<void> {
    this.dashboard.clearAlarm(params.alarmId);
    this.sendJSON(res, { success: true });
  }

  private async handleGetOptimizationEvents(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const state = this.dashboard.getState();
    this.sendJSON(res, { events: state.optimizationTimeline });
  }

  private async handleCreateOptimizationEvent(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const body = await this.parseBody(req);
    this.dashboard.addOptimizationEvent(body as OptimizationEvent);
    this.sendJSON(res, { success: true });
  }

  private async handleGetApprovals(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const state = this.dashboard.getState();
    this.sendJSON(res, { approvals: state.pendingApprovals });
  }

  private async handleApproveRequest(req: IncomingMessage, res: ServerResponse, params: any): Promise<void> {
    const body = await this.parseBody(req);

    if (!body.signature) {
      this.sendError(res, 400, 'Signature is required');
      return;
    }

    try {
      await this.dashboard.processApproval(params.approvalId, true, body.signature);
      this.sendJSON(res, { success: true, message: 'Approval granted' });
    } catch (error: any) {
      this.sendError(res, 500, error.message);
    }
  }

  private async handleRejectRequest(req: IncomingMessage, res: ServerResponse, params: any): Promise<void> {
    const body = await this.parseBody(req);

    if (!body.signature) {
      this.sendError(res, 400, 'Signature is required');
      return;
    }

    try {
      await this.dashboard.processApproval(params.approvalId, false, body.signature);
      this.sendJSON(res, { success: true, message: 'Approval rejected' });
    } catch (error: any) {
      this.sendError(res, 500, error.message);
    }
  }

  private async handleGetInterferenceMatrix(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const state = this.dashboard.getState();

    if (!state.interferenceMatrix) {
      this.sendError(res, 404, 'Interference matrix not available');
      return;
    }

    this.sendJSON(res, { matrix: state.interferenceMatrix });
  }

  private async handleHealth(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const state = this.dashboard.getState();

    this.sendJSON(res, {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      stats: {
        cells: state.cells.size,
        alarms: state.alarms.length,
        optimizationEvents: state.optimizationTimeline.length,
        pendingApprovals: state.pendingApprovals.filter(a => a.status === 'pending').length
      }
    });
  }

  // ==========================================================================
  // HTTP Server
  // ==========================================================================

  async start(): Promise<void> {
    this.server = createServer((req, res) => this.handleRequest(req, res));

    return new Promise((resolve, reject) => {
      this.server.listen(this.port, () => {
        console.log(`[API Server] TITAN Dashboard API listening on port ${this.port}`);
        console.log(`[API Server] SSE endpoints:`);
        console.log(`  - /api/stream/kpi         (KPI counters, 10s interval)`);
        console.log(`  - /api/stream/events      (Optimization events)`);
        console.log(`  - /api/stream/alarms      (FM alarms)`);
        console.log(`  - /api/stream/approvals   (HITL approvals)`);
        console.log(`[API Server] REST endpoints:`);
        console.log(`  - GET  /api/cells`);
        console.log(`  - GET  /api/cells/:cellId`);
        console.log(`  - POST /api/cells/:cellId/parameters`);
        console.log(`  - GET  /api/alarms`);
        console.log(`  - GET  /api/approvals`);
        console.log(`  - POST /api/approvals/:approvalId/approve`);
        console.log(`  - POST /api/approvals/:approvalId/reject`);
        console.log(`  - GET  /health`);
        resolve();
      });

      this.server.on('error', reject);
    });
  }

  async stop(): Promise<void> {
    if (this.server) {
      return new Promise((resolve) => {
        this.server.close(() => {
          console.log('[API Server] Server stopped');
          resolve();
        });
      });
    }
  }

  // ==========================================================================
  // Utilities
  // ==========================================================================

  private async handleRequest(req: IncomingMessage, res: ServerResponse): Promise<void> {
    const url = req.url || '/';
    const method = req.method || 'GET';

    console.log(`[API Server] ${method} ${url}`);

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

    if (method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    // Match route
    for (const route of this.routes) {
      const match = this.matchRoute(route, method, url);
      if (match) {
        try {
          await route.handler(req, res, match.params);
          return;
        } catch (error: any) {
          console.error('[API Server] Route handler error:', error);
          this.sendError(res, 500, error.message);
          return;
        }
      }
    }

    // 404
    this.sendError(res, 404, 'Not Found');
  }

  private matchRoute(route: APIRoute, method: string, url: string): { params: any } | null {
    if (route.method !== method) return null;

    const routeParts = route.path.split('/').filter(Boolean);
    const urlParts = url.split('?')[0].split('/').filter(Boolean);

    if (routeParts.length !== urlParts.length) return null;

    const params: any = {};

    for (let i = 0; i < routeParts.length; i++) {
      if (routeParts[i].startsWith(':')) {
        const paramName = routeParts[i].substring(1);
        params[paramName] = urlParts[i];
      } else if (routeParts[i] !== urlParts[i]) {
        return null;
      }
    }

    return { params };
  }

  private async parseBody(req: IncomingMessage): Promise<any> {
    return new Promise((resolve, reject) => {
      let body = '';
      req.on('data', chunk => (body += chunk));
      req.on('end', () => {
        try {
          resolve(body ? JSON.parse(body) : {});
        } catch (error) {
          reject(new Error('Invalid JSON'));
        }
      });
      req.on('error', reject);
    });
  }

  private sendJSON(res: ServerResponse, data: any): void {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(data));
  }

  private sendError(res: ServerResponse, statusCode: number, message: string): void {
    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: message }));
  }
}

// ============================================================================
// Export
// ============================================================================

export default TitanAPIServer;
