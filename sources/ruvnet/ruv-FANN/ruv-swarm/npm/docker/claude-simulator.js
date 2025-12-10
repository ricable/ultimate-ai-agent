const axios = require('axios');
const WebSocket = require('ws');
const jsonrpc = require('jsonrpc-lite');
const winston = require('winston');
const promClient = require('prom-client');
const fs = require('fs').promises;
const path = require('path');

// Logger setup
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'claude-simulator' },
  transports: [
    new winston.transports.Console({
      format: winston.format.simple()
    }),
    new winston.transports.File({ 
      filename: '/app/logs/claude-simulator.log',
      maxsize: 10485760, // 10MB
      maxFiles: 5
    })
  ]
});

// Prometheus metrics
const register = new promClient.Registry();
promClient.collectDefaultMetrics({ register });

const metrics = {
  connections: new promClient.Counter({
    name: 'mcp_connections_total',
    help: 'Total number of MCP connections attempted',
    labelNames: ['status']
  }),
  disconnections: new promClient.Counter({
    name: 'mcp_disconnections_total',
    help: 'Total number of MCP disconnections',
    labelNames: ['reason']
  }),
  requests: new promClient.Counter({
    name: 'mcp_requests_total',
    help: 'Total number of MCP requests sent',
    labelNames: ['method', 'status']
  }),
  requestDuration: new promClient.Histogram({
    name: 'mcp_request_duration_seconds',
    help: 'MCP request duration in seconds',
    labelNames: ['method'],
    buckets: [0.1, 0.5, 1, 2, 5, 10]
  }),
  reconnectAttempts: new promClient.Counter({
    name: 'mcp_reconnect_attempts_total',
    help: 'Total number of reconnection attempts',
    labelNames: ['success']
  }),
  sessionDuration: new promClient.Histogram({
    name: 'mcp_session_duration_seconds',
    help: 'Duration of MCP sessions in seconds',
    buckets: [60, 300, 600, 1800, 3600, 7200]
  })
};

// Register all metrics
Object.values(metrics).forEach(metric => register.registerMetric(metric));

// Claude Simulator Class
class ClaudeSimulator {
  constructor(config) {
    this.config = {
      serverUrl: process.env.MCP_SERVER_URL || 'http://localhost:3000',
      sessionDuration: parseInt(process.env.SESSION_DURATION) || 3600,
      requestInterval: parseInt(process.env.REQUEST_INTERVAL) || 100,
      failureInjection: process.env.FAILURE_INJECTION === 'true',
      testScenarios: (process.env.TEST_SCENARIOS || 'connection').split(','),
      ...config
    };
    
    this.ws = null;
    this.sessionStartTime = null;
    this.requestId = 0;
    this.pendingRequests = new Map();
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000;
    this.scenarios = [];
  }

  async loadScenarios() {
    try {
      const scenarioFiles = await fs.readdir('/app/scenarios');
      for (const file of scenarioFiles) {
        if (file.endsWith('.json')) {
          const content = await fs.readFile(path.join('/app/scenarios', file), 'utf8');
          this.scenarios.push(JSON.parse(content));
        }
      }
      logger.info(`Loaded ${this.scenarios.length} test scenarios`);
    } catch (error) {
      logger.warn('No scenario files found, using default scenarios');
      this.scenarios = this.getDefaultScenarios();
    }
  }

  getDefaultScenarios() {
    return [
      {
        name: 'basic-connection',
        steps: [
          { action: 'connect' },
          { action: 'initialize' },
          { action: 'list-tools' },
          { action: 'wait', duration: 5000 },
          { action: 'disconnect' }
        ]
      },
      {
        name: 'long-running-session',
        steps: [
          { action: 'connect' },
          { action: 'initialize' },
          { action: 'repeat', count: 100, interval: 1000, steps: [
            { action: 'call-tool', tool: 'swarm_status' },
            { action: 'call-tool', tool: 'memory_usage' }
          ]},
          { action: 'disconnect' }
        ]
      },
      {
        name: 'reconnection-test',
        steps: [
          { action: 'connect' },
          { action: 'initialize' },
          { action: 'call-tool', tool: 'swarm_init', params: { topology: 'mesh' } },
          { action: 'force-disconnect' },
          { action: 'wait', duration: 2000 },
          { action: 'reconnect' },
          { action: 'call-tool', tool: 'swarm_status' },
          { action: 'verify-state' }
        ]
      },
      {
        name: 'heavy-load',
        steps: [
          { action: 'connect' },
          { action: 'initialize' },
          { action: 'parallel', count: 10, steps: [
            { action: 'call-tool', tool: 'agent_spawn', params: { type: 'researcher' } },
            { action: 'call-tool', tool: 'task_orchestrate', params: { task: 'test-task' } }
          ]},
          { action: 'wait', duration: 5000 },
          { action: 'disconnect' }
        ]
      }
    ];
  }

  async connect() {
    return new Promise((resolve, reject) => {
      try {
        const wsUrl = this.config.serverUrl.replace('http://', 'ws://').replace('https://', 'wss://');
        logger.info(`Connecting to MCP server at ${wsUrl}`);
        
        this.ws = new WebSocket(wsUrl);
        this.sessionStartTime = Date.now();
        
        this.ws.on('open', () => {
          logger.info('WebSocket connection established');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          metrics.connections.inc({ status: 'success' });
          resolve();
        });

        this.ws.on('message', (data) => {
          this.handleMessage(data.toString());
        });

        this.ws.on('close', (code, reason) => {
          logger.warn(`WebSocket connection closed: ${code} - ${reason}`);
          this.isConnected = false;
          metrics.disconnections.inc({ reason: code.toString() });
          
          if (this.sessionStartTime) {
            const duration = (Date.now() - this.sessionStartTime) / 1000;
            metrics.sessionDuration.observe(duration);
          }
          
          // Attempt reconnection if not a clean close
          if (code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.attemptReconnection();
          }
        });

        this.ws.on('error', (error) => {
          logger.error('WebSocket error:', error);
          metrics.connections.inc({ status: 'error' });
          reject(error);
        });

        // Connection timeout
        setTimeout(() => {
          if (!this.isConnected) {
            logger.error('Connection timeout');
            this.ws.close();
            reject(new Error('Connection timeout'));
          }
        }, 30000);

      } catch (error) {
        logger.error('Failed to connect:', error);
        metrics.connections.inc({ status: 'failed' });
        reject(error);
      }
    });
  }

  async attemptReconnection() {
    this.reconnectAttempts++;
    logger.info(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
    metrics.reconnectAttempts.inc({ success: 'pending' });
    
    await new Promise(resolve => setTimeout(resolve, this.reconnectDelay * this.reconnectAttempts));
    
    try {
      await this.connect();
      metrics.reconnectAttempts.inc({ success: 'true' });
      logger.info('Reconnection successful');
    } catch (error) {
      metrics.reconnectAttempts.inc({ success: 'false' });
      logger.error('Reconnection failed:', error);
      
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.attemptReconnection();
      }
    }
  }

  handleMessage(data) {
    try {
      const parsed = jsonrpc.parseObject(data);
      
      if (parsed.type === 'success' || parsed.type === 'error') {
        const requestId = parsed.payload.id;
        const pending = this.pendingRequests.get(requestId);
        
        if (pending) {
          const duration = (Date.now() - pending.startTime) / 1000;
          metrics.requestDuration.observe({ method: pending.method }, duration);
          
          if (parsed.type === 'success') {
            metrics.requests.inc({ method: pending.method, status: 'success' });
            pending.resolve(parsed.payload.result);
          } else {
            metrics.requests.inc({ method: pending.method, status: 'error' });
            pending.reject(new Error(parsed.payload.error.message));
          }
          
          this.pendingRequests.delete(requestId);
        }
      } else if (parsed.type === 'notification') {
        logger.info('Received notification:', parsed.payload);
      }
    } catch (error) {
      logger.error('Failed to parse message:', error);
    }
  }

  async sendRequest(method, params = {}) {
    return new Promise((resolve, reject) => {
      if (!this.isConnected) {
        reject(new Error('Not connected to MCP server'));
        return;
      }

      const id = ++this.requestId;
      const request = jsonrpc.request(id, method, params);
      
      this.pendingRequests.set(id, {
        method,
        startTime: Date.now(),
        resolve,
        reject
      });

      logger.debug(`Sending request: ${method}`, params);
      this.ws.send(JSON.stringify(request));

      // Request timeout
      setTimeout(() => {
        if (this.pendingRequests.has(id)) {
          this.pendingRequests.delete(id);
          metrics.requests.inc({ method, status: 'timeout' });
          reject(new Error(`Request timeout: ${method}`));
        }
      }, 30000);
    });
  }

  async initialize() {
    logger.info('Initializing MCP session');
    const result = await this.sendRequest('initialize', {
      protocolVersion: '1.0',
      clientInfo: {
        name: 'claude-simulator',
        version: '1.0.0'
      }
    });
    logger.info('MCP session initialized:', result);
    return result;
  }

  async listTools() {
    logger.info('Listing available tools');
    const result = await this.sendRequest('tools/list');
    logger.info(`Found ${result.tools?.length || 0} tools`);
    return result;
  }

  async callTool(name, arguments = {}) {
    logger.info(`Calling tool: ${name}`, arguments);
    const result = await this.sendRequest('tools/call', {
      name,
      arguments
    });
    return result;
  }

  async executeScenario(scenario) {
    logger.info(`Executing scenario: ${scenario.name}`);
    const results = [];

    for (const step of scenario.steps) {
      try {
        const result = await this.executeStep(step);
        results.push({ step, result, success: true });
      } catch (error) {
        logger.error(`Step failed:`, error);
        results.push({ step, error: error.message, success: false });
        
        // Continue with next step unless it's a critical failure
        if (step.action === 'connect' || step.action === 'initialize') {
          break;
        }
      }
    }

    return results;
  }

  async executeStep(step) {
    logger.debug(`Executing step: ${step.action}`);

    switch (step.action) {
      case 'connect':
        return await this.connect();
      
      case 'disconnect':
        if (this.ws) {
          this.ws.close(1000, 'Normal closure');
        }
        return { disconnected: true };
      
      case 'force-disconnect':
        if (this.ws) {
          this.ws.terminate();
        }
        return { forcedDisconnect: true };
      
      case 'reconnect':
        await this.connect();
        return await this.initialize();
      
      case 'initialize':
        return await this.initialize();
      
      case 'list-tools':
        return await this.listTools();
      
      case 'call-tool':
        return await this.callTool(step.tool, step.params);
      
      case 'wait':
        await new Promise(resolve => setTimeout(resolve, step.duration));
        return { waited: step.duration };
      
      case 'repeat':
        const results = [];
        for (let i = 0; i < step.count; i++) {
          for (const subStep of step.steps) {
            results.push(await this.executeStep(subStep));
          }
          if (step.interval) {
            await new Promise(resolve => setTimeout(resolve, step.interval));
          }
        }
        return results;
      
      case 'parallel':
        const promises = [];
        for (let i = 0; i < step.count; i++) {
          for (const subStep of step.steps) {
            promises.push(this.executeStep(subStep));
          }
        }
        return await Promise.all(promises);
      
      case 'verify-state':
        // Verify that the connection is still valid and tools are available
        const tools = await this.listTools();
        const status = await this.callTool('swarm_status');
        return { verified: true, tools: tools.tools?.length, status };
      
      default:
        throw new Error(`Unknown step action: ${step.action}`);
    }
  }

  async runSimulation() {
    logger.info('Starting Claude Code simulation');
    logger.info('Configuration:', this.config);

    // Load test scenarios
    await this.loadScenarios();

    // Start metrics server
    const metricsServer = require('http').createServer((req, res) => {
      if (req.url === '/metrics') {
        res.setHeader('Content-Type', register.contentType);
        register.metrics().then(data => res.end(data));
      } else {
        res.statusCode = 404;
        res.end();
      }
    });
    metricsServer.listen(9091, () => {
      logger.info('Metrics server listening on port 9091');
    });

    // Execute test scenarios
    const results = {
      scenarios: [],
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        startTime: new Date().toISOString(),
        endTime: null
      }
    };

    for (const scenario of this.scenarios) {
      if (this.config.testScenarios.includes('all') || 
          this.config.testScenarios.some(s => scenario.name.includes(s))) {
        results.summary.total++;
        
        try {
          const scenarioResults = await this.executeScenario(scenario);
          const failed = scenarioResults.filter(r => !r.success).length;
          
          if (failed === 0) {
            results.summary.passed++;
          } else {
            results.summary.failed++;
          }
          
          results.scenarios.push({
            name: scenario.name,
            results: scenarioResults,
            passed: failed === 0
          });
        } catch (error) {
          results.summary.failed++;
          results.scenarios.push({
            name: scenario.name,
            error: error.message,
            passed: false
          });
        }

        // Wait between scenarios
        await new Promise(resolve => setTimeout(resolve, 5000));
      }
    }

    results.summary.endTime = new Date().toISOString();

    // Save results
    const resultsPath = `/app/test-results/simulation-${Date.now()}.json`;
    await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));
    logger.info(`Results saved to ${resultsPath}`);

    // Log summary
    logger.info('Simulation completed:', results.summary);

    // Keep running if long-running session is configured
    if (this.config.sessionDuration > 0) {
      logger.info(`Continuing session for ${this.config.sessionDuration} seconds`);
      await new Promise(resolve => setTimeout(resolve, this.config.sessionDuration * 1000));
    }

    // Clean shutdown
    if (this.ws && this.isConnected) {
      this.ws.close(1000, 'Simulation completed');
    }
    process.exit(results.summary.failed > 0 ? 1 : 0);
  }

  // Failure injection methods
  async injectRandomFailure() {
    if (!this.config.failureInjection) return;

    const failures = [
      () => this.simulateNetworkLatency(),
      () => this.simulatePacketLoss(),
      () => this.simulateDisconnect(),
      () => this.simulateHighLoad(),
      () => this.simulateSlowResponse()
    ];

    const failure = failures[Math.floor(Math.random() * failures.length)];
    await failure();
  }

  async simulateNetworkLatency() {
    logger.info('Simulating network latency');
    // Add delay to all requests
    const originalSend = this.ws.send;
    this.ws.send = (data) => {
      setTimeout(() => originalSend.call(this.ws, data), Math.random() * 500 + 100);
    };
    
    setTimeout(() => {
      this.ws.send = originalSend;
    }, 30000);
  }

  async simulatePacketLoss() {
    logger.info('Simulating packet loss');
    // Randomly drop some messages
    const originalSend = this.ws.send;
    this.ws.send = (data) => {
      if (Math.random() > 0.1) { // 10% packet loss
        originalSend.call(this.ws, data);
      } else {
        logger.debug('Packet dropped');
      }
    };
    
    setTimeout(() => {
      this.ws.send = originalSend;
    }, 20000);
  }

  async simulateDisconnect() {
    logger.info('Simulating unexpected disconnect');
    if (this.ws) {
      this.ws.terminate();
    }
  }

  async simulateHighLoad() {
    logger.info('Simulating high load');
    // Send many requests in parallel
    const promises = [];
    for (let i = 0; i < 50; i++) {
      promises.push(this.callTool('swarm_status').catch(() => {}));
    }
    await Promise.all(promises);
  }

  async simulateSlowResponse() {
    logger.info('Simulating slow server responses');
    // This would need server-side cooperation
    await this.callTool('debug_slow_mode', { enabled: true, delay: 5000 });
    
    setTimeout(async () => {
      await this.callTool('debug_slow_mode', { enabled: false });
    }, 30000);
  }
}

// Main execution
async function main() {
  const simulator = new ClaudeSimulator();
  
  try {
    await simulator.runSimulation();
  } catch (error) {
    logger.error('Simulation failed:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}