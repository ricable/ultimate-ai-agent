/**
 * MCP Server Implementation for Synaptic Neural Mesh
 * Implements MCP 2024.11.5 specification with JSON-RPC 2.0
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { z } from 'zod';
import { EventEmitter } from 'events';

export class McpServer extends EventEmitter {
  constructor({ tools, transport, auth, events, config }) {
    super();
    this.tools = tools;
    this.transport = transport;
    this.auth = auth;
    this.events = events;
    this.config = config;
    
    this.server = null;
    this.serverTransport = null;
    this.isInitialized = false;
    this.isRunning = false;
    
    this.requestCount = 0;
    this.errorCount = 0;
    this.startTime = null;
  }

  async initialize() {
    try {
      // Create MCP server instance
      this.server = new Server(
        {
          name: 'synaptic-neural-mesh',
          version: '1.0.0',
          description: 'Distributed neural fabric with DAG consensus and swarm intelligence'
        },
        {
          capabilities: {
            tools: {
              listChanged: true
            },
            resources: {
              subscribe: true,
              listChanged: true
            },
            prompts: {
              listChanged: true
            },
            logging: {},
            experimental: {
              sampling: true
            }
          }
        }
      );

      // Setup transport
      await this.setupTransport();
      
      // Register tool handlers
      await this.registerToolHandlers();
      
      // Register resource handlers
      await this.registerResourceHandlers();
      
      // Register prompt handlers
      await this.registerPromptHandlers();

      // Setup error handling
      this.setupErrorHandling();

      this.isInitialized = true;
      this.emit('initialized');
      
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async setupTransport() {
    switch (this.config.transport) {
      case 'stdio':
        this.serverTransport = new StdioServerTransport();
        break;
      case 'http':
        // HTTP transport implementation would go here
        throw new Error('HTTP transport not yet implemented');
      default:
        throw new Error(`Unsupported transport: ${this.config.transport}`);
    }
  }

  async registerToolHandlers() {
    // List all available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => {
      return {
        tools: await this.tools.listTools()
      };
    });

    // Handle tool execution
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      const { name, arguments: args } = request.params;
      
      try {
        this.requestCount++;
        
        // Authentication check if enabled
        if (this.config.enableAuth) {
          await this.auth.authorize(request);
        }

        // Rate limiting
        await this.checkRateLimit(request);

        // Execute tool
        const result = await this.tools.executeTool(name, args);
        
        // Emit execution event
        this.events.emit('toolExecuted', {
          tool: name,
          args,
          result,
          timestamp: Date.now()
        });

        return {
          content: [
            {
              type: 'text',
              text: typeof result === 'string' ? result : JSON.stringify(result, null, 2)
            }
          ]
        };
        
      } catch (error) {
        this.errorCount++;
        this.emit('toolError', { name, args, error });
        
        return {
          content: [
            {
              type: 'text',
              text: `Error executing tool '${name}': ${error.message}`
            }
          ],
          isError: true
        };
      }
    });
  }

  async registerResourceHandlers() {
    // List available resources
    this.server.setRequestHandler('resources/list', async () => {
      return {
        resources: [
          {
            uri: 'neural-mesh://status',
            mimeType: 'application/json',
            name: 'Neural Mesh Status',
            description: 'Current status of the neural mesh network'
          },
          {
            uri: 'neural-mesh://dag',
            mimeType: 'application/json',
            name: 'DAG State',
            description: 'Current state of the DAG consensus'
          },
          {
            uri: 'neural-mesh://agents',
            mimeType: 'application/json',
            name: 'Active Agents',
            description: 'List of active neural agents in the mesh'
          },
          {
            uri: 'neural-mesh://metrics',
            mimeType: 'application/json',
            name: 'Performance Metrics',
            description: 'Real-time performance metrics'
          }
        ]
      };
    });

    // Read specific resources
    this.server.setRequestHandler('resources/read', async (request) => {
      const { uri } = request.params;
      
      try {
        const content = await this.getResourceContent(uri);
        
        return {
          contents: [
            {
              uri,
              mimeType: 'application/json',
              text: JSON.stringify(content, null, 2)
            }
          ]
        };
      } catch (error) {
        throw new Error(`Failed to read resource ${uri}: ${error.message}`);
      }
    });
  }

  async registerPromptHandlers() {
    // List available prompts
    this.server.setRequestHandler('prompts/list', async () => {
      return {
        prompts: [
          {
            name: 'neural-mesh-analysis',
            description: 'Analyze neural mesh performance and suggest optimizations',
            arguments: [
              {
                name: 'timeframe',
                description: 'Analysis timeframe (1h, 24h, 7d)',
                required: false
              }
            ]
          },
          {
            name: 'swarm-coordination',
            description: 'Generate coordination strategies for neural swarms',
            arguments: [
              {
                name: 'swarmSize',
                description: 'Number of agents in the swarm',
                required: true
              },
              {
                name: 'task',
                description: 'Primary task for the swarm',
                required: true
              }
            ]
          }
        ]
      };
    });

    // Get specific prompts
    this.server.setRequestHandler('prompts/get', async (request) => {
      const { name, arguments: args } = request.params;
      
      switch (name) {
        case 'neural-mesh-analysis':
          return {
            description: 'Neural mesh analysis prompt',
            messages: [
              {
                role: 'user',
                content: {
                  type: 'text',
                  text: await this.generateAnalysisPrompt(args?.timeframe || '24h')
                }
              }
            ]
          };
          
        case 'swarm-coordination':
          return {
            description: 'Swarm coordination prompt',
            messages: [
              {
                role: 'user',
                content: {
                  type: 'text',
                  text: await this.generateSwarmPrompt(args.swarmSize, args.task)
                }
              }
            ]
          };
          
        default:
          throw new Error(`Unknown prompt: ${name}`);
      }
    });
  }

  setupErrorHandling() {
    this.server.onerror = (error) => {
      console.error('MCP Server error:', error);
      this.emit('error', error);
    };

    process.on('SIGINT', async () => {
      await this.stop();
      process.exit(0);
    });
  }

  async checkRateLimit(request) {
    // Simple rate limiting implementation
    // In production, use Redis or similar for distributed rate limiting
    const now = Date.now();
    const windowMs = 60000; // 1 minute
    const maxRequests = 100;
    
    // This is a simplified implementation
    if (this.requestCount > maxRequests) {
      const timeSinceStart = now - (this.startTime || now);
      if (timeSinceStart < windowMs) {
        throw new Error('Rate limit exceeded');
      } else {
        this.requestCount = 0;
        this.startTime = now;
      }
    }
  }

  async getResourceContent(uri) {
    const [, , resource] = uri.split('://');
    const [type] = resource.split('/');
    
    switch (type) {
      case 'status':
        return await this.tools.getNeuralMeshStatus();
      case 'dag':
        return await this.tools.getDAGState();
      case 'agents':
        return await this.tools.getActiveAgents();
      case 'metrics':
        return await this.tools.getPerformanceMetrics();
      default:
        throw new Error(`Unknown resource type: ${type}`);
    }
  }

  async generateAnalysisPrompt(timeframe) {
    const metrics = await this.tools.getPerformanceMetrics(timeframe);
    return `Analyze the following neural mesh performance data and provide optimization recommendations:\n\n${JSON.stringify(metrics, null, 2)}`;
  }

  async generateSwarmPrompt(swarmSize, task) {
    return `Design an optimal coordination strategy for a neural swarm with ${swarmSize} agents to accomplish the following task: ${task}\n\nConsider:\n- Agent specialization\n- Communication patterns\n- Load distribution\n- Fault tolerance\n- Consensus mechanisms`;
  }

  async start() {
    if (!this.isInitialized) {
      await this.initialize();
    }

    try {
      await this.server.connect(this.serverTransport);
      this.isRunning = true;
      this.startTime = Date.now();
      this.emit('started');
      
      console.log('🌐 MCP Server started and listening for connections');
      
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async stop() {
    if (this.isRunning) {
      try {
        await this.server.close();
        this.isRunning = false;
        this.emit('stopped');
        
      } catch (error) {
        this.emit('error', error);
        throw error;
      }
    }
  }

  isRunning() {
    return this.isRunning;
  }

  getStats() {
    return {
      requestCount: this.requestCount,
      errorCount: this.errorCount,
      uptime: this.startTime ? Date.now() - this.startTime : 0,
      isRunning: this.isRunning
    };
  }
}

export default McpServer;