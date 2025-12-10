#!/usr/bin/env node
// FastMCP HTTP server with Server-Sent Events (SSE) streaming support
import { FastMCP } from 'fastmcp';
import { z } from 'zod';
import { execSync } from 'child_process';
import http from 'http';

console.error('🚀 Starting FastMCP HTTP Streaming Server...');
console.error('🌐 Port: 3000');
console.error('📡 SSE endpoint: /events');
console.error('🔧 Tools: 6 (memory_store, memory_retrieve, memory_search, swarm_init, agent_spawn, task_orchestrate)');

// Create FastMCP server
const mcp = new FastMCP({
  name: 'fastmcp-http-streaming',
  version: '1.0.0'
});

// Add 6 tools
mcp.addTool({
  name: 'memory_store',
  description: 'Store a value in persistent memory',
  parameters: z.object({
    key: z.string().min(1),
    value: z.string(),
    namespace: z.string().optional().default('default'),
    ttl: z.number().positive().optional()
  }),
  execute: async ({ key, value, namespace, ttl }) => {
    const cmd = [
      'npx claude-flow@alpha memory store',
      `"${key}"`, `"${value}"`,
      `--namespace "${namespace}"`,
      ttl ? `--ttl ${ttl}` : ''
    ].filter(Boolean).join(' ');
    const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    return JSON.stringify({ success: true, key, namespace, size: value.length, ttl, timestamp: new Date().toISOString() }, null, 2);
  }
});

mcp.addTool({
  name: 'memory_retrieve',
  description: 'Retrieve a value from memory',
  parameters: z.object({
    key: z.string().min(1),
    namespace: z.string().optional().default('default')
  }),
  execute: async ({ key, namespace }) => {
    const cmd = `npx claude-flow@alpha memory retrieve "${key}" --namespace "${namespace}"`;
    const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    return JSON.stringify({ success: true, key, namespace, value: result.trim(), timestamp: new Date().toISOString() }, null, 2);
  }
});

mcp.addTool({
  name: 'memory_search',
  description: 'Search for keys matching a pattern',
  parameters: z.object({
    pattern: z.string().min(1),
    namespace: z.string().optional(),
    limit: z.number().positive().optional().default(10)
  }),
  execute: async ({ pattern, namespace, limit }) => {
    const cmd = `npx claude-flow@alpha memory search "${pattern}"${namespace ? ` --namespace "${namespace}"` : ''} --limit ${limit}`;
    const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    return JSON.stringify({ success: true, pattern, namespace: namespace || 'all', results: result.trim(), limit, timestamp: new Date().toISOString() }, null, 2);
  }
});

mcp.addTool({
  name: 'swarm_init',
  description: 'Initialize a multi-agent swarm',
  parameters: z.object({
    topology: z.enum(['mesh', 'hierarchical', 'ring', 'star']),
    maxAgents: z.number().positive().optional().default(8),
    strategy: z.enum(['balanced', 'specialized', 'adaptive']).optional().default('balanced')
  }),
  execute: async ({ topology, maxAgents, strategy }) => {
    const cmd = `npx claude-flow@alpha swarm init --topology ${topology} --max-agents ${maxAgents} --strategy ${strategy}`;
    const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    return JSON.stringify({ success: true, topology, maxAgents, strategy, result: result.trim(), timestamp: new Date().toISOString() }, null, 2);
  }
});

mcp.addTool({
  name: 'agent_spawn',
  description: 'Spawn a new agent in the swarm',
  parameters: z.object({
    type: z.enum(['researcher', 'coder', 'analyst', 'optimizer', 'coordinator']),
    capabilities: z.array(z.string()).optional(),
    name: z.string().optional()
  }),
  execute: async ({ type, capabilities, name }) => {
    const capStr = capabilities ? ` --capabilities "${capabilities.join(',')}"` : '';
    const nameStr = name ? ` --name "${name}"` : '';
    const cmd = `npx claude-flow@alpha agent spawn --type ${type}${capStr}${nameStr}`;
    const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    return JSON.stringify({ success: true, type, capabilities, name, result: result.trim(), timestamp: new Date().toISOString() }, null, 2);
  }
});

mcp.addTool({
  name: 'task_orchestrate',
  description: 'Orchestrate a task across the swarm',
  parameters: z.object({
    task: z.string().min(1),
    strategy: z.enum(['parallel', 'sequential', 'adaptive']).optional().default('adaptive'),
    priority: z.enum(['low', 'medium', 'high', 'critical']).optional().default('medium'),
    maxAgents: z.number().positive().optional()
  }),
  execute: async ({ task, strategy, priority, maxAgents }) => {
    const maxStr = maxAgents ? ` --max-agents ${maxAgents}` : '';
    const cmd = `npx claude-flow@alpha task orchestrate "${task}" --strategy ${strategy} --priority ${priority}${maxStr}`;
    const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
    return JSON.stringify({ success: true, task, strategy, priority, maxAgents, result: result.trim(), timestamp: new Date().toISOString() }, null, 2);
  }
});

// HTTP Server with SSE
const server = http.createServer((req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    res.writeHead(200);
    res.end();
    return;
  }

  // SSE endpoint
  if (req.url === '/events' && req.method === 'GET') {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive'
    });

    // Send initial connection event
    res.write(`event: connected\ndata: ${JSON.stringify({ status: 'connected', timestamp: new Date().toISOString() })}\n\n`);

    // Keep-alive ping every 30s
    const pingInterval = setInterval(() => {
      res.write(`event: ping\ndata: ${JSON.stringify({ type: 'ping', timestamp: new Date().toISOString() })}\n\n`);
    }, 30000);

    req.on('close', () => {
      clearInterval(pingInterval);
    });
    return;
  }

  // MCP endpoint
  if (req.url === '/mcp' && req.method === 'POST') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      try {
        const request = JSON.parse(body);

        // Handle tools/list
        if (request.method === 'tools/list') {
          const tools = [
            { name: 'memory_store', description: 'Store a value in persistent memory' },
            { name: 'memory_retrieve', description: 'Retrieve a value from memory' },
            { name: 'memory_search', description: 'Search for keys matching a pattern' },
            { name: 'swarm_init', description: 'Initialize a multi-agent swarm' },
            { name: 'agent_spawn', description: 'Spawn a new agent in the swarm' },
            { name: 'task_orchestrate', description: 'Orchestrate a task across the swarm' }
          ];
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: { tools } }));
          return;
        }

        // Handle tools/call
        if (request.method === 'tools/call') {
          const { name, arguments: args } = request.params;
          // Execute tool via FastMCP
          const result = await executeTool(name, args);
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ jsonrpc: '2.0', id: request.id, result: { content: [{ type: 'text', text: result }] } }));
          return;
        }

        // Unknown method
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ jsonrpc: '2.0', id: request.id, error: { code: -32601, message: 'Method not found' } }));
      } catch (error: any) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ jsonrpc: '2.0', error: { code: -32603, message: error.message } }));
      }
    });
    return;
  }

  // Health check
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'healthy', timestamp: new Date().toISOString() }));
    return;
  }

  // 404
  res.writeHead(404);
  res.end('Not Found');
});

// Tool execution helper
async function executeTool(name: string, args: any): Promise<string> {
  switch (name) {
    case 'memory_store': {
      const { key, value, namespace = 'default', ttl } = args;
      const cmd = [
        'npx claude-flow@alpha memory store',
        `"${key}"`, `"${value}"`,
        `--namespace "${namespace}"`,
        ttl ? `--ttl ${ttl}` : ''
      ].filter(Boolean).join(' ');
      const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
      return JSON.stringify({ success: true, key, namespace, size: value.length, ttl, timestamp: new Date().toISOString() }, null, 2);
    }
    case 'memory_retrieve': {
      const { key, namespace = 'default' } = args;
      const cmd = `npx claude-flow@alpha memory retrieve "${key}" --namespace "${namespace}"`;
      const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
      return JSON.stringify({ success: true, key, namespace, value: result.trim(), timestamp: new Date().toISOString() }, null, 2);
    }
    case 'memory_search': {
      const { pattern, namespace, limit = 10 } = args;
      const cmd = `npx claude-flow@alpha memory search "${pattern}"${namespace ? ` --namespace "${namespace}"` : ''} --limit ${limit}`;
      const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
      return JSON.stringify({ success: true, pattern, namespace: namespace || 'all', results: result.trim(), limit, timestamp: new Date().toISOString() }, null, 2);
    }
    case 'swarm_init': {
      const { topology, maxAgents = 8, strategy = 'balanced' } = args;
      const cmd = `npx claude-flow@alpha swarm init --topology ${topology} --max-agents ${maxAgents} --strategy ${strategy}`;
      const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
      return JSON.stringify({ success: true, topology, maxAgents, strategy, result: result.trim(), timestamp: new Date().toISOString() }, null, 2);
    }
    case 'agent_spawn': {
      const { type, capabilities, name } = args;
      const capStr = capabilities ? ` --capabilities "${capabilities.join(',')}"` : '';
      const nameStr = name ? ` --name "${name}"` : '';
      const cmd = `npx claude-flow@alpha agent spawn --type ${type}${capStr}${nameStr}`;
      const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
      return JSON.stringify({ success: true, type, capabilities, name, result: result.trim(), timestamp: new Date().toISOString() }, null, 2);
    }
    case 'task_orchestrate': {
      const { task, strategy = 'adaptive', priority = 'medium', maxAgents } = args;
      const maxStr = maxAgents ? ` --max-agents ${maxAgents}` : '';
      const cmd = `npx claude-flow@alpha task orchestrate "${task}" --strategy ${strategy} --priority ${priority}${maxStr}`;
      const result = execSync(cmd, { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 });
      return JSON.stringify({ success: true, task, strategy, priority, maxAgents, result: result.trim(), timestamp: new Date().toISOString() }, null, 2);
    }
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

server.listen(3000, () => {
  console.error('✅ FastMCP HTTP Streaming Server running on http://localhost:3000');
  console.error('📡 SSE endpoint: http://localhost:3000/events');
  console.error('🔧 MCP endpoint: http://localhost:3000/mcp');
  console.error('❤️  Health check: http://localhost:3000/health');
});
