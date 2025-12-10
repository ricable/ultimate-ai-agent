#!/usr/bin/env node

/**
 * QuDAG Native MCP Server - SSE Transport
 * Model Context Protocol server with Server-Sent Events transport
 */

const qudag = require('../index.js');
const http = require('http');
const { URL } = require('url');

class QudagMcpSseServer {
  constructor(port = 3000) {
    this.port = port;
    this.clients = new Map();
    this.tools = this.getToolDefinitions();
  }

  getToolDefinitions() {
    return [
      {
        name: 'mlkem768_keygen',
        description: 'Generate ML-KEM-768 keypair for quantum-resistant encryption',
        inputSchema: { type: 'object', properties: {}, required: [] }
      },
      {
        name: 'mlkem768_encapsulate',
        description: 'Encapsulate shared secret using ML-KEM-768 public key',
        inputSchema: {
          type: 'object',
          properties: {
            publicKey: { type: 'string', description: 'Hex-encoded public key (1184 bytes)' }
          },
          required: ['publicKey']
        }
      },
      {
        name: 'mlkem768_decapsulate',
        description: 'Decapsulate shared secret using ML-KEM-768 secret key',
        inputSchema: {
          type: 'object',
          properties: {
            ciphertext: { type: 'string', description: 'Hex-encoded ciphertext (1088 bytes)' },
            secretKey: { type: 'string', description: 'Hex-encoded secret key (2400 bytes)' }
          },
          required: ['ciphertext', 'secretKey']
        }
      },
      {
        name: 'mldsa65_keygen',
        description: 'Generate ML-DSA-65 keypair for quantum-resistant signatures',
        inputSchema: { type: 'object', properties: {}, required: [] }
      },
      {
        name: 'mldsa65_sign',
        description: 'Sign message with ML-DSA-65',
        inputSchema: {
          type: 'object',
          properties: {
            message: { type: 'string', description: 'Hex-encoded message to sign' },
            secretKey: { type: 'string', description: 'Hex-encoded secret key (4032 bytes)' }
          },
          required: ['message', 'secretKey']
        }
      },
      {
        name: 'mldsa65_verify',
        description: 'Verify ML-DSA-65 signature',
        inputSchema: {
          type: 'object',
          properties: {
            message: { type: 'string', description: 'Hex-encoded message' },
            signature: { type: 'string', description: 'Hex-encoded signature (3309 bytes)' },
            publicKey: { type: 'string', description: 'Hex-encoded public key (1952 bytes)' }
          },
          required: ['message', 'signature', 'publicKey']
        }
      },
      {
        name: 'blake3_hash',
        description: 'Compute BLAKE3 cryptographic hash',
        inputSchema: {
          type: 'object',
          properties: {
            data: { type: 'string', description: 'Hex-encoded data to hash' }
          },
          required: ['data']
        }
      },
      {
        name: 'quantum_fingerprint',
        description: 'Generate quantum-resistant fingerprint',
        inputSchema: {
          type: 'object',
          properties: {
            data: { type: 'string', description: 'Hex-encoded data to fingerprint' }
          },
          required: ['data']
        }
      },
      {
        name: 'random_bytes',
        description: 'Generate cryptographically secure random bytes',
        inputSchema: {
          type: 'object',
          properties: {
            length: { type: 'number', description: 'Number of bytes to generate' }
          },
          required: ['length']
        }
      }
    ];
  }

  start() {
    this.server = http.createServer((req, res) => {
      this.handleRequest(req, res);
    });

    this.server.listen(this.port, () => {
      console.log(`🚀 QuDAG MCP Server (SSE) listening on http://localhost:${this.port}`);
      console.log(`   Endpoints:`);
      console.log(`   • GET  /sse    - SSE connection`);
      console.log(`   • POST /message - Send MCP message`);
      console.log(`   • GET  /health - Health check`);
    });
  }

  handleRequest(req, res) {
    const url = new URL(req.url, `http://${req.headers.host}`);

    // CORS headers
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
      res.writeHead(200);
      res.end();
      return;
    }

    if (url.pathname === '/sse' && req.method === 'GET') {
      this.handleSseConnection(req, res);
    } else if (url.pathname === '/message' && req.method === 'POST') {
      this.handleMessage(req, res);
    } else if (url.pathname === '/health' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ status: 'ok', version: qudag.version() }));
    } else {
      res.writeHead(404);
      res.end('Not found');
    }
  }

  handleSseConnection(req, res) {
    const clientId = Date.now().toString();

    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });

    this.clients.set(clientId, res);

    // Send initial connection event
    this.sendEvent(clientId, 'connected', { clientId, timestamp: new Date().toISOString() });

    req.on('close', () => {
      this.clients.delete(clientId);
    });
  }

  async handleMessage(req, res) {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
      try {
        const message = JSON.parse(body);
        const response = this.processMessage(message);

        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify(response));
      } catch (err) {
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message }));
      }
    });
  }

  processMessage(message) {
    const { method, params, id } = message;

    try {
      switch (method) {
        case 'initialize':
          return {
            id,
            result: {
              protocolVersion: '2024-11-05',
              serverInfo: { name: 'qudag-native', version: qudag.version() },
              capabilities: { tools: true, resources: false, prompts: false }
            }
          };

        case 'tools/list':
          return { id, result: { tools: this.tools } };

        case 'tools/call':
          return { id, result: this.executeToolCall(params) };

        default:
          return { id, error: { code: -32601, message: `Unknown method: ${method}` } };
      }
    } catch (err) {
      return { id, error: { code: -32603, message: err.message } };
    }
  }

  executeToolCall(params) {
    const { name, arguments: args } = params;

    switch (name) {
      case 'mlkem768_keygen': {
        const keypair = qudag.mlkem768GenerateKeypair();
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              publicKey: qudag.bytesToHex(keypair.publicKey),
              secretKey: qudag.bytesToHex(keypair.secretKey),
              fingerprint: qudag.quantumFingerprint(keypair.publicKey)
            }, null, 2)
          }]
        };
      }

      case 'mlkem768_encapsulate': {
        const publicKey = qudag.hexToBytes(args.publicKey);
        const result = qudag.mlkem768Encapsulate(publicKey);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              ciphertext: qudag.bytesToHex(result.ciphertext),
              sharedSecret: qudag.bytesToHex(result.sharedSecret)
            }, null, 2)
          }]
        };
      }

      case 'mlkem768_decapsulate': {
        const ciphertext = qudag.hexToBytes(args.ciphertext);
        const secretKey = qudag.hexToBytes(args.secretKey);
        const sharedSecret = qudag.mlkem768Decapsulate(ciphertext, secretKey);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              sharedSecret: qudag.bytesToHex(sharedSecret)
            }, null, 2)
          }]
        };
      }

      case 'mldsa65_keygen': {
        const keypair = qudag.mldsa65GenerateKeypair();
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              publicKey: qudag.bytesToHex(keypair.publicKey),
              secretKey: qudag.bytesToHex(keypair.secretKey),
              fingerprint: qudag.quantumFingerprint(keypair.publicKey)
            }, null, 2)
          }]
        };
      }

      case 'mldsa65_sign': {
        const message = qudag.hexToBytes(args.message);
        const secretKey = qudag.hexToBytes(args.secretKey);
        const signature = qudag.mldsa65Sign(message, secretKey);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              signature: qudag.bytesToHex(signature)
            }, null, 2)
          }]
        };
      }

      case 'mldsa65_verify': {
        const message = qudag.hexToBytes(args.message);
        const signature = qudag.hexToBytes(args.signature);
        const publicKey = qudag.hexToBytes(args.publicKey);
        const valid = qudag.mldsa65Verify(message, signature, publicKey);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ valid }, null, 2)
          }]
        };
      }

      case 'blake3_hash': {
        const data = qudag.hexToBytes(args.data);
        const hash = qudag.blake3HashHex(data);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ hash }, null, 2)
          }]
        };
      }

      case 'quantum_fingerprint': {
        const data = qudag.hexToBytes(args.data);
        const fingerprint = qudag.quantumFingerprint(data);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({ fingerprint }, null, 2)
          }]
        };
      }

      case 'random_bytes': {
        const randomData = qudag.randomBytes(args.length);
        return {
          content: [{
            type: 'text',
            text: JSON.stringify({
              bytes: qudag.bytesToHex(randomData)
            }, null, 2)
          }]
        };
      }

      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  }

  sendEvent(clientId, event, data) {
    const client = this.clients.get(clientId);
    if (client) {
      client.write(`event: ${event}\n`);
      client.write(`data: ${JSON.stringify(data)}\n\n`);
    }
  }
}

module.exports = (port) => {
  const server = new QudagMcpSseServer(port);
  server.start();
};

// Run directly if executed
if (require.main === module) {
  const port = parseInt(process.argv[2] || '3000');
  module.exports(port);
}
