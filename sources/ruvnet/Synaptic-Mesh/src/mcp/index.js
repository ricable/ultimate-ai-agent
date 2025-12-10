/**
 * Synaptic Neural Mesh - MCP Integration Entry Point
 * 
 * This module provides Model Context Protocol (MCP) integration for the
 * Synaptic Neural Mesh, enabling AI assistants to interact with the
 * distributed neural fabric through standardized JSON-RPC 2.0 protocols.
 */

import { McpServer } from './server/mcp-server.js';
import { NeuralMeshTools } from './neural-mesh/neural-mesh-tools.js';
import { TransportManager } from './transport/transport-manager.js';
import { AuthManager } from './auth/auth-manager.js';
import { EventStreamer } from './events/event-streamer.js';
import { WasmBridge } from './wasm-bridge/wasm-bridge.js';

export class SynapticMeshMCP {
  constructor(config = {}) {
    this.config = {
      transport: 'stdio',
      port: 3000,
      enableAuth: false,
      enableEvents: true,
      wasmEnabled: true,
      logLevel: 'info',
      ...config
    };

    this.server = null;
    this.tools = null;
    this.transport = null;
    this.auth = null;
    this.events = null;
    this.wasmBridge = null;
  }

  /**
   * Initialize the MCP server with all components
   */
  async initialize() {
    try {
      // Initialize core components
      this.transport = new TransportManager(this.config);
      this.auth = new AuthManager(this.config);
      this.events = new EventStreamer(this.config);
      this.wasmBridge = new WasmBridge(this.config);

      // Initialize neural mesh tools
      this.tools = new NeuralMeshTools({
        wasmBridge: this.wasmBridge,
        events: this.events,
        auth: this.auth
      });

      // Initialize MCP server
      this.server = new McpServer({
        tools: this.tools,
        transport: this.transport,
        auth: this.auth,
        events: this.events,
        config: this.config
      });

      await this.server.initialize();
      
      console.log('🧠 Synaptic Neural Mesh MCP Server initialized');
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize MCP server:', error);
      throw error;
    }
  }

  /**
   * Start the MCP server
   */
  async start() {
    if (!this.server) {
      await this.initialize();
    }
    
    await this.server.start();
    console.log(`🚀 MCP Server running on ${this.config.transport}${this.config.transport === 'http' ? ':' + this.config.port : ''}`);
  }

  /**
   * Stop the MCP server
   */
  async stop() {
    if (this.server) {
      await this.server.stop();
      console.log('🛑 MCP Server stopped');
    }
  }

  /**
   * Get server status
   */
  getStatus() {
    return {
      initialized: !!this.server,
      running: this.server?.isRunning() || false,
      config: this.config,
      toolsCount: this.tools?.getToolCount() || 0,
      activeConnections: this.transport?.getActiveConnections() || 0
    };
  }
}

// Export individual components for advanced usage
export {
  McpServer,
  NeuralMeshTools,
  TransportManager,
  AuthManager,
  EventStreamer,
  WasmBridge
};

// Default export
export default SynapticMeshMCP;