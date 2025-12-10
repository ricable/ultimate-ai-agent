#!/usr/bin/env node
/**
 * Synaptic Neural Mesh MCP Server Startup Script
 * Initializes and runs the MCP server with neural mesh extensions
 */

import { SynapticMCPServer } from './synaptic-mcp-server.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

interface MCPMessage {
  jsonrpc: string;
  id?: string | number;
  method?: string;
  params?: any;
  result?: any;
  error?: any;
}

class SynapticMCPRunner {
  private server: SynapticMCPServer;
  private buffer: string = '';
  
  constructor() {
    this.server = new SynapticMCPServer();
  }
  
  async start() {
    console.error(`[${new Date().toISOString()}] INFO [synaptic-mcp] Starting Synaptic Neural Mesh MCP Server`);
    console.error({
      mode: 'stdio',
      protocol: 'MCP 2024-11-05',
      extensions: 'synaptic-neural-mesh',
      tools: {
        claude_flow: 27,
        synaptic: 20,
        total: 47
      },
      capabilities: {
        streaming: true,
        batch: true,
        ai_integration: true
      }
    });
    
    // Send initial capabilities
    this.sendMessage({
      jsonrpc: '2.0',
      method: 'server.initialized',
      params: {
        serverInfo: {
          name: 'synaptic-neural-mesh',
          version: '1.0.0',
          capabilities: {
            tools: { listChanged: true },
            resources: { subscribe: true, listChanged: true },
            streaming: { supported: true },
            batch: { supported: true }
          }
        }
      }
    });
    
    // Setup input handling
    this.setupInputHandling();
    
    // Setup graceful shutdown
    this.setupShutdownHandlers();
  }
  
  private setupInputHandling() {
    process.stdin.on('data', async (chunk) => {
      this.buffer += chunk.toString();
      
      // Process complete JSON messages
      const lines = this.buffer.split('\n');
      this.buffer = lines.pop() || ''; // Keep incomplete line in buffer
      
      for (const line of lines) {
        if (line.trim()) {
          try {
            const message: MCPMessage = JSON.parse(line);
            await this.handleMessage(message);
          } catch (error) {
            console.error(`[${new Date().toISOString()}] ERROR [synaptic-mcp] Failed to parse message:`, error);
            this.sendError(null, -32700, 'Parse error', error.message);
          }
        }
      }
    });
    
    process.stdin.on('end', () => {
      console.error(`[${new Date().toISOString()}] INFO [synaptic-mcp] Connection closed`);
      process.exit(0);
    });
  }
  
  private async handleMessage(message: MCPMessage) {
    try {
      console.error(`[${new Date().toISOString()}] INFO [synaptic-mcp] Received: ${message.method || 'response'}`);
      
      // Delegate to server
      const response = await this.server.handleMessage(message);
      
      if (response) {
        this.sendMessage(response);
      }
    } catch (error) {
      console.error(`[${new Date().toISOString()}] ERROR [synaptic-mcp] Handler error:`, error);
      this.sendError(message.id, -32603, 'Internal error', error.message);
    }
  }
  
  private sendMessage(message: MCPMessage) {
    console.log(JSON.stringify(message));
  }
  
  private sendError(id: string | number | null, code: number, message: string, data?: any) {
    const error: MCPMessage = {
      jsonrpc: '2.0',
      id: id || null,
      error: { code, message }
    };
    
    if (data) {
      error.error.data = data;
    }
    
    this.sendMessage(error);
  }
  
  private setupShutdownHandlers() {
    const shutdown = async (signal: string) => {
      console.error(`[${new Date().toISOString()}] INFO [synaptic-mcp] Received ${signal}, shutting down...`);
      
      // Cleanup operations
      try {
        // Save mesh states
        // Close connections
        // Persist memory
        
        console.error(`[${new Date().toISOString()}] INFO [synaptic-mcp] Shutdown complete`);
      } catch (error) {
        console.error(`[${new Date().toISOString()}] ERROR [synaptic-mcp] Shutdown error:`, error);
      }
      
      process.exit(0);
    };
    
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGTERM', () => shutdown('SIGTERM'));
    
    // Handle uncaught errors
    process.on('uncaughtException', (error) => {
      console.error(`[${new Date().toISOString()}] ERROR [synaptic-mcp] Uncaught exception:`, error);
      process.exit(1);
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      console.error(`[${new Date().toISOString()}] ERROR [synaptic-mcp] Unhandled rejection:`, reason);
      process.exit(1);
    });
  }
}

// Start the server
if (import.meta.url === `file://${process.argv[1]}`) {
  const runner = new SynapticMCPRunner();
  runner.start().catch((error) => {
    console.error(`[${new Date().toISOString()}] ERROR [synaptic-mcp] Failed to start:`, error);
    process.exit(1);
  });
}

export { SynapticMCPRunner };