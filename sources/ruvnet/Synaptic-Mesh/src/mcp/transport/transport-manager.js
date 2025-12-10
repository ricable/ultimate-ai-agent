/**
 * Transport Manager for MCP Communication
 * Handles stdio, HTTP, and WebSocket transport layers
 */

import { EventEmitter } from 'events';
import { createServer } from 'http';
import WebSocket, { WebSocketServer } from 'ws';

export class TransportManager extends EventEmitter {
  constructor(config) {
    super();
    this.config = config;
    this.activeConnections = new Set();
    this.connectionPool = new Map();
    this.server = null;
    this.wss = null;
    
    this.stats = {
      totalConnections: 0,
      activeConnections: 0,
      messagesProcessed: 0,
      errors: 0
    };
  }

  async initialize() {
    switch (this.config.transport) {
      case 'stdio':
        await this.initializeStdio();
        break;
      case 'http':
        await this.initializeHttp();
        break;
      case 'websocket':
        await this.initializeWebSocket();
        break;
      default:
        throw new Error(`Unsupported transport: ${this.config.transport}`);
    }
  }

  async initializeStdio() {
    // stdio transport is handled by the MCP SDK directly
    console.log('📡 Stdio transport initialized');
  }

  async initializeHttp() {
    this.server = createServer();
    
    this.server.on('request', async (req, res) => {
      if (req.method === 'POST' && req.url === '/mcp') {
        await this.handleHttpRequest(req, res);
      } else if (req.method === 'GET' && req.url === '/health') {
        await this.handleHealthCheck(req, res);
      } else {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not Found' }));
      }
    });

    await new Promise((resolve) => {
      this.server.listen(this.config.port, () => {
        console.log(`🌐 HTTP transport listening on port ${this.config.port}`);
        resolve();
      });
    });
  }

  async initializeWebSocket() {
    if (!this.server) {
      await this.initializeHttp();
    }

    this.wss = new WebSocketServer({ 
      server: this.server,
      path: '/ws'
    });

    this.wss.on('connection', (ws, req) => {
      this.handleWebSocketConnection(ws, req);
    });

    console.log('🔌 WebSocket transport initialized');
  }

  async handleHttpRequest(req, res) {
    try {
      let body = '';
      req.on('data', chunk => { body += chunk; });
      
      await new Promise(resolve => req.on('end', resolve));
      
      const requestData = JSON.parse(body);
      this.stats.messagesProcessed++;
      
      // Set CORS headers
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      
      if (req.method === 'OPTIONS') {
        res.writeHead(200);
        res.end();
        return;
      }

      // Emit request for processing
      const responseData = await this.processRequest(requestData);
      
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(responseData));
      
    } catch (error) {
      this.stats.errors++;
      console.error('HTTP request error:', error);
      
      res.writeHead(500, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        jsonrpc: '2.0',
        error: {
          code: -32603,
          message: 'Internal error',
          data: error.message
        }
      }));
    }
  }

  async handleHealthCheck(req, res) {
    const health = {
      status: 'healthy',
      timestamp: Date.now(),
      transport: this.config.transport,
      stats: this.getStats(),
      uptime: process.uptime()
    };

    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify(health));
  }

  handleWebSocketConnection(ws, req) {
    const connectionId = this.generateConnectionId();
    
    const connection = {
      id: connectionId,
      ws,
      req,
      createdAt: Date.now(),
      lastActivity: Date.now(),
      messageCount: 0
    };

    this.activeConnections.add(connection);
    this.connectionPool.set(connectionId, connection);
    this.stats.totalConnections++;
    this.stats.activeConnections++;

    console.log(`🔗 WebSocket connection established: ${connectionId}`);

    ws.on('message', async (data) => {
      try {
        const message = JSON.parse(data.toString());
        connection.lastActivity = Date.now();
        connection.messageCount++;
        this.stats.messagesProcessed++;

        const response = await this.processRequest(message);
        ws.send(JSON.stringify(response));

      } catch (error) {
        this.stats.errors++;
        console.error('WebSocket message error:', error);
        
        ws.send(JSON.stringify({
          jsonrpc: '2.0',
          error: {
            code: -32700,
            message: 'Parse error',
            data: error.message
          }
        }));
      }
    });

    ws.on('close', () => {
      this.activeConnections.delete(connection);
      this.connectionPool.delete(connectionId);
      this.stats.activeConnections--;
      console.log(`🔌 WebSocket connection closed: ${connectionId}`);
    });

    ws.on('error', (error) => {
      console.error(`WebSocket error for ${connectionId}:`, error);
      this.stats.errors++;
    });

    // Send welcome message
    ws.send(JSON.stringify({
      type: 'welcome',
      connectionId,
      timestamp: Date.now(),
      capabilities: ['tools', 'resources', 'prompts']
    }));
  }

  async processRequest(request) {
    // This would be handled by the MCP server
    // Emit event for the server to handle
    return new Promise((resolve, reject) => {
      this.emit('request', request, (error, response) => {
        if (error) reject(error);
        else resolve(response);
      });
    });
  }

  generateConnectionId() {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  getActiveConnections() {
    return this.activeConnections.size;
  }

  getStats() {
    return {
      ...this.stats,
      activeConnections: this.activeConnections.size,
      connectionPoolSize: this.connectionPool.size
    };
  }

  async broadcastToAll(message) {
    const promises = [];
    
    for (const connection of this.activeConnections) {
      if (connection.ws && connection.ws.readyState === WebSocket.OPEN) {
        promises.push(
          new Promise(resolve => {
            connection.ws.send(JSON.stringify(message), resolve);
          })
        );
      }
    }

    await Promise.allSettled(promises);
  }

  async close() {
    // Close all WebSocket connections
    for (const connection of this.activeConnections) {
      if (connection.ws) {
        connection.ws.close();
      }
    }

    // Close HTTP server
    if (this.server) {
      await new Promise(resolve => {
        this.server.close(resolve);
      });
    }

    console.log('🔌 Transport manager closed');
  }
}

export default TransportManager;