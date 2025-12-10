#!/usr/bin/env node

/**
 * QuDAG Native MCP Server - STDIO Transport
 * Model Context Protocol server for quantum cryptography operations
 */

const qudag = require('../index.js');
const readline = require('readline');

class QudagMcpServer {
  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    this.capabilities = {
      tools: true,
      resources: false,
      prompts: false
    };

    this.tools = [
      {
        name: 'mlkem768_keygen',
        description: 'Generate ML-KEM-768 keypair for quantum-resistant encryption',
        inputSchema: {
          type: 'object',
          properties: {},
          required: []
        }
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
        inputSchema: {
          type: 'object',
          properties: {},
          required: []
        }
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
    this.rl.on('line', (line) => {
      try {
        const request = JSON.parse(line);
        this.handleRequest(request);
      } catch (err) {
        this.sendError(null, -32700, 'Parse error: ' + err.message);
      }
    });

    this.rl.on('close', () => {
      process.exit(0);
    });

    // Send server info on startup (optional)
    process.stderr.write('QuDAG MCP Server started (STDIO transport)\n');
  }

  handleRequest(request) {
    const { id, method, params } = request;

    try {
      switch (method) {
        case 'initialize':
          this.sendResponse(id, {
            protocolVersion: '2024-11-05',
            serverInfo: {
              name: 'qudag-native',
              version: qudag.version()
            },
            capabilities: this.capabilities
          });
          break;

        case 'tools/list':
          this.sendResponse(id, { tools: this.tools });
          break;

        case 'tools/call':
          this.handleToolCall(id, params);
          break;

        default:
          this.sendError(id, -32601, `Method not found: ${method}`);
      }
    } catch (err) {
      this.sendError(id, -32603, 'Internal error: ' + err.message);
    }
  }

  handleToolCall(id, params) {
    const { name, arguments: args } = params;

    try {
      let result;

      switch (name) {
        case 'mlkem768_keygen': {
          const keypair = qudag.mlkem768GenerateKeypair();
          result = {
            publicKey: qudag.bytesToHex(keypair.publicKey),
            secretKey: qudag.bytesToHex(keypair.secretKey),
            fingerprint: qudag.quantumFingerprint(keypair.publicKey)
          };
          break;
        }

        case 'mlkem768_encapsulate': {
          const publicKey = qudag.hexToBytes(args.publicKey);
          const encapsulated = qudag.mlkem768Encapsulate(publicKey);
          result = {
            ciphertext: qudag.bytesToHex(encapsulated.ciphertext),
            sharedSecret: qudag.bytesToHex(encapsulated.sharedSecret)
          };
          break;
        }

        case 'mlkem768_decapsulate': {
          const ciphertext = qudag.hexToBytes(args.ciphertext);
          const secretKey = qudag.hexToBytes(args.secretKey);
          const sharedSecret = qudag.mlkem768Decapsulate(ciphertext, secretKey);
          result = {
            sharedSecret: qudag.bytesToHex(sharedSecret)
          };
          break;
        }

        case 'mldsa65_keygen': {
          const keypair = qudag.mldsa65GenerateKeypair();
          result = {
            publicKey: qudag.bytesToHex(keypair.publicKey),
            secretKey: qudag.bytesToHex(keypair.secretKey),
            fingerprint: qudag.quantumFingerprint(keypair.publicKey)
          };
          break;
        }

        case 'mldsa65_sign': {
          const message = qudag.hexToBytes(args.message);
          const secretKey = qudag.hexToBytes(args.secretKey);
          const signature = qudag.mldsa65Sign(message, secretKey);
          result = {
            signature: qudag.bytesToHex(signature)
          };
          break;
        }

        case 'mldsa65_verify': {
          const message = qudag.hexToBytes(args.message);
          const signature = qudag.hexToBytes(args.signature);
          const publicKey = qudag.hexToBytes(args.publicKey);
          const valid = qudag.mldsa65Verify(message, signature, publicKey);
          result = { valid };
          break;
        }

        case 'blake3_hash': {
          const data = qudag.hexToBytes(args.data);
          const hash = qudag.blake3HashHex(data);
          result = { hash };
          break;
        }

        case 'quantum_fingerprint': {
          const data = qudag.hexToBytes(args.data);
          const fingerprint = qudag.quantumFingerprint(data);
          result = { fingerprint };
          break;
        }

        case 'random_bytes': {
          const randomData = qudag.randomBytes(args.length);
          result = {
            bytes: qudag.bytesToHex(randomData)
          };
          break;
        }

        default:
          throw new Error(`Unknown tool: ${name}`);
      }

      this.sendResponse(id, {
        content: [{
          type: 'text',
          text: JSON.stringify(result, null, 2)
        }]
      });
    } catch (err) {
      this.sendError(id, -32000, `Tool execution error: ${err.message}`);
    }
  }

  sendResponse(id, result) {
    const response = {
      jsonrpc: '2.0',
      id,
      result
    };
    console.log(JSON.stringify(response));
  }

  sendError(id, code, message) {
    const response = {
      jsonrpc: '2.0',
      id,
      error: { code, message }
    };
    console.log(JSON.stringify(response));
  }
}

module.exports = () => {
  const server = new QudagMcpServer();
  server.start();
};

// Run directly if executed
if (require.main === module) {
  module.exports();
}
