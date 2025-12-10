/**
 * AG-UI Server
 * Agentic User Interface Protocol Implementation
 *
 * Provides "Glass Box" operational experience with
 * Generative UI and Human-in-the-Loop (HITL) safety.
 */

import { readFileSync } from 'fs';
import { EventEmitter } from 'events';

export class AGUIServer extends EventEmitter {
  constructor({ port, protocolPath }) {
    super();
    this.port = port || 3000;
    this.protocolPath = protocolPath || './config/ag-ui/protocol.json';

    this.protocol = this.loadProtocol();
    this.clients = new Set();
    this.pendingApprovals = new Map();
  }

  loadProtocol() {
    try {
      const data = readFileSync(this.protocolPath, 'utf-8');
      return JSON.parse(data);
    } catch (error) {
      console.warn('[AG-UI] Failed to load protocol, using defaults');
      return { events: {} };
    }
  }

  /**
   * Start the AG-UI server
   */
  async start() {
    console.log(`[AG-UI] Starting Glass Box interface on port ${this.port}...`);
    console.log('[AG-UI] Deprecated interfaces: telegram, slack, chatops');
    console.log('[AG-UI] Generative UI mode: ENABLED');

    // In production, this would start an HTTP/WebSocket server
    this.running = true;
  }

  /**
   * Emit an AG-UI event to all connected clients
   */
  emit(eventType, payload) {
    const eventConfig = this.protocol.events[eventType];

    if (!eventConfig) {
      console.warn(`[AG-UI] Unknown event type: ${eventType}`);
      return;
    }

    const event = {
      type: eventType,
      timestamp: new Date().toISOString(),
      payload
    };

    console.log(`[AG-UI] Emitting ${eventType}:`, JSON.stringify(payload).substring(0, 100));

    // Broadcast to all connected clients
    for (const client of this.clients) {
      this.sendToClient(client, event);
    }

    // Handle special event types
    if (eventType === 'request_approval') {
      return this.handleApprovalRequest(payload);
    }

    return event;
  }

  sendToClient(client, event) {
    // In production, sends via WebSocket
    console.log(`[AG-UI] -> Client ${client.id}: ${event.type}`);
  }

  /**
   * Emit a Generative UI render event
   */
  renderComponent(component, props) {
    return this.emit('gen_ui_render', {
      component,
      props,
      interactive: true
    });
  }

  /**
   * Render an interference heatmap
   */
  renderInterferenceHeatmap(cells, interferenceMatrix, threshold) {
    return this.renderComponent('InterferenceHeatmap', {
      cells,
      interference_matrix: interferenceMatrix,
      threshold
    });
  }

  /**
   * Render a neighbor graph (force-directed)
   */
  renderNeighborGraph(nodes, edges) {
    return this.renderComponent('NeighborGraph', {
      nodes,
      edges,
      weights: edges.map(e => e.weight || 1)
    });
  }

  /**
   * Render a causal diagram
   */
  renderCausalDiagram(causes, effects, probabilities) {
    return this.renderComponent('CausalDiagram', {
      causes,
      effects,
      probabilities
    });
  }

  /**
   * Handle HITL Approval Request
   * Critical actions require human authorization
   */
  async handleApprovalRequest(payload) {
    const { risk_level, action, target, justification } = payload;

    console.log(`[AG-UI] HITL APPROVAL REQUIRED`);
    console.log(`[AG-UI] Risk Level: ${risk_level}`);
    console.log(`[AG-UI] Action: ${action}`);
    console.log(`[AG-UI] Justification: ${justification}`);

    const approvalId = `approval-${Date.now()}`;

    const request = {
      id: approvalId,
      payload,
      status: 'pending',
      createdAt: new Date().toISOString()
    };

    this.pendingApprovals.set(approvalId, request);

    // In production, this displays a Critical Action Card
    console.log(`[AG-UI] Waiting for human authorization (ID: ${approvalId})...`);

    return request;
  }

  /**
   * Process approval response
   */
  async processApproval(approvalId, authorized, signature) {
    const request = this.pendingApprovals.get(approvalId);

    if (!request) {
      throw new Error(`Unknown approval request: ${approvalId}`);
    }

    request.status = authorized ? 'approved' : 'rejected';
    request.signature = signature;
    request.resolvedAt = new Date().toISOString();

    console.log(`[AG-UI] Approval ${approvalId}: ${request.status}`);

    return request;
  }

  /**
   * Send agent reasoning explanation
   */
  explainReasoning(agentId, reasoning) {
    return this.emit('agent_message', {
      type: 'markdown',
      content: reasoning,
      agent_id: agentId,
      reasoning_chain: reasoning.split('\n')
    });
  }

  /**
   * Sync Managed Object state
   */
  syncMOState(moClass, moId, param, value) {
    return this.emit('state_sync', {
      mo_class: moClass,
      mo_id: moId,
      param,
      value,
      timestamp: new Date().toISOString(),
      source: 'titan-orchestrator'
    });
  }

  /**
   * Report tool execution
   */
  reportToolCall(tool, command, args, status, result) {
    return this.emit('tool_call', {
      tool,
      command,
      args,
      status,
      result
    });
  }
}
