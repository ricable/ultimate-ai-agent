/**
 * DAA-MCP Bridge
 * Connects the Rust DAA (Distributed Autonomous Agents) system with MCP tools
 */

import { spawn, ChildProcess } from 'child_process';
import { EventEmitter } from 'events';
import path from 'path';

interface DAAAgent {
  id: string;
  type: string;
  capabilities: string[];
  status: 'active' | 'inactive' | 'error';
  mesh_connection: boolean;
}

interface DAAMessage {
  type: 'command' | 'query' | 'status' | 'data';
  agent_id?: string;
  payload: any;
  timestamp: number;
}

export class DAAMCPBridge extends EventEmitter {
  private daaProcess: ChildProcess | null = null;
  private agents: Map<string, DAAAgent> = new Map();
  private messageQueue: DAAMessage[] = [];
  private isConnected: boolean = false;
  
  constructor(private daaExecutablePath?: string) {
    super();
    this.daaExecutablePath = daaExecutablePath || this.findDAAExecutable();
  }
  
  private findDAAExecutable(): string {
    // Look for DAA executable in the Rust build directory
    const possiblePaths = [
      '../rs/daa/daa-main/target/release/daa-orchestrator',
      '../rs/daa/daa-main/target/debug/daa-orchestrator',
      'daa-orchestrator'
    ];
    
    // For now, return the release path
    return path.join(__dirname, possiblePaths[0]);
  }
  
  async connect(): Promise<void> {
    if (this.isConnected) {
      return;
    }
    
    console.log(`🔗 Connecting to DAA system at: ${this.daaExecutablePath}`);
    
    try {
      // Start the DAA orchestrator process
      this.daaProcess = spawn(this.daaExecutablePath, ['--mcp-mode'], {
        stdio: ['pipe', 'pipe', 'pipe'],
        cwd: path.dirname(this.daaExecutablePath)
      });
      
      this.setupProcessHandlers();
      this.isConnected = true;
      
      // Initialize with MCP bridge mode
      await this.sendCommand('init', {
        mode: 'mcp_bridge',
        capabilities: ['mesh_control', 'agent_management', 'streaming']
      });
      
      this.emit('connected');
      console.log('✅ DAA-MCP Bridge connected');
      
    } catch (error) {
      console.error('❌ Failed to connect to DAA system:', error);
      throw error;
    }
  }
  
  private setupProcessHandlers() {
    if (!this.daaProcess) return;
    
    // Handle stdout (DAA responses)
    this.daaProcess.stdout?.on('data', (data: Buffer) => {
      const lines = data.toString().split('\n').filter(line => line.trim());
      
      for (const line of lines) {
        try {
          const message: DAAMessage = JSON.parse(line);
          this.handleDAAMessage(message);
        } catch (error) {
          // Non-JSON output (logs, etc.)
          console.log('[DAA]', line);
        }
      }
    });
    
    // Handle stderr (DAA logs)
    this.daaProcess.stderr?.on('data', (data: Buffer) => {
      console.error('[DAA Error]', data.toString());
    });
    
    // Handle process exit
    this.daaProcess.on('exit', (code) => {
      console.log(`DAA process exited with code: ${code}`);
      this.isConnected = false;
      this.emit('disconnected', code);
    });
    
    // Handle process error
    this.daaProcess.on('error', (error) => {
      console.error('DAA process error:', error);
      this.emit('error', error);
    });
  }
  
  private handleDAAMessage(message: DAAMessage) {
    switch (message.type) {
      case 'status':
        this.handleStatusMessage(message);
        break;
      case 'data':
        this.emit('data', message);
        break;
      default:
        this.emit('message', message);
    }
  }
  
  private handleStatusMessage(message: DAAMessage) {
    if (message.payload.agents) {
      // Update agent registry
      for (const agentData of message.payload.agents) {
        this.agents.set(agentData.id, agentData);
      }
      this.emit('agents_updated', Array.from(this.agents.values()));
    }
  }
  
  async sendCommand(command: string, params: any = {}): Promise<any> {
    if (!this.isConnected || !this.daaProcess) {
      throw new Error('DAA bridge not connected');
    }
    
    const message: DAAMessage = {
      type: 'command',
      payload: { command, params },
      timestamp: Date.now()
    };
    
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error('DAA command timeout'));
      }, 10000);
      
      const handler = (response: DAAMessage) => {
        if (response.type === 'data' && response.payload.command === command) {
          clearTimeout(timeout);
          this.off('message', handler);
          resolve(response.payload.result);
        }
      };
      
      this.on('message', handler);
      this.daaProcess!.stdin?.write(JSON.stringify(message) + '\n');
    });
  }
  
  // MCP Tool Implementations that use DAA
  async createDAAAgent(type: string, capabilities: string[]): Promise<DAAAgent> {
    const result = await this.sendCommand('create_agent', {
      type,
      capabilities
    });
    
    const agent: DAAAgent = {
      id: result.agent_id,
      type,
      capabilities,
      status: 'active',
      mesh_connection: false
    };
    
    this.agents.set(agent.id, agent);
    return agent;
  }
  
  async connectAgentToMesh(agentId: string, meshId: string): Promise<boolean> {
    const result = await this.sendCommand('connect_to_mesh', {
      agent_id: agentId,
      mesh_id: meshId
    });
    
    if (result.success) {
      const agent = this.agents.get(agentId);
      if (agent) {
        agent.mesh_connection = true;
        this.agents.set(agentId, agent);
      }
    }
    
    return result.success;
  }
  
  async allocateResources(resources: any, agents: string[]): Promise<any> {
    return await this.sendCommand('allocate_resources', {
      resources,
      agents
    });
  }
  
  async enableCapabilityMatching(taskRequirements: string[]): Promise<string[]> {
    const result = await this.sendCommand('capability_match', {
      task_requirements: taskRequirements,
      available_agents: Array.from(this.agents.keys())
    });
    
    return result.matched_agents || [];
  }
  
  async establishCommunication(fromAgent: string, toAgent: string, message: any): Promise<boolean> {
    const result = await this.sendCommand('agent_communicate', {
      from: fromAgent,
      to: toAgent,
      message
    });
    
    return result.success;
  }
  
  async initiateConsensus(agents: string[], proposal: any): Promise<any> {
    return await this.sendCommand('consensus', {
      agents,
      proposal
    });
  }
  
  async enableFaultTolerance(agentId: string, strategy: string): Promise<boolean> {
    const result = await this.sendCommand('fault_tolerance', {
      agent_id: agentId,
      strategy
    });
    
    return result.success;
  }
  
  async optimizePerformance(target: string, metrics: string[]): Promise<any> {
    return await this.sendCommand('optimize', {
      target,
      metrics
    });
  }
  
  async manageLifecycle(agentId: string, action: string): Promise<boolean> {
    const result = await this.sendCommand('lifecycle', {
      agent_id: agentId,
      action
    });
    
    return result.success;
  }
  
  // Streaming support
  enableStreaming(streamType: 'activity' | 'metrics' | 'communication'): AsyncIterable<any> {
    const self = this;
    
    return {
      async *[Symbol.asyncIterator]() {
        // Set up streaming
        await self.sendCommand('enable_stream', { type: streamType });
        
        // Listen for streaming data
        const streamHandler = (message: DAAMessage) => {
          if (message.type === 'data' && message.payload.stream_type === streamType) {
            return message.payload.data;
          }
        };
        
        self.on('message', streamHandler);
        
        // Yield data as it comes
        while (self.isConnected) {
          yield new Promise((resolve) => {
            const handler = (message: DAAMessage) => {
              if (message.type === 'data' && message.payload.stream_type === streamType) {
                self.off('message', handler);
                resolve(message.payload.data);
              }
            };
            self.on('message', handler);
          });
        }
      }
    };
  }
  
  getAgents(): DAAAgent[] {
    return Array.from(this.agents.values());
  }
  
  getAgent(id: string): DAAAgent | undefined {
    return this.agents.get(id);
  }
  
  async disconnect(): Promise<void> {
    if (this.daaProcess) {
      this.daaProcess.kill('SIGTERM');
      this.daaProcess = null;
    }
    this.isConnected = false;
    this.agents.clear();
    this.emit('disconnected');
  }
}

// Export for use in MCP server
export default DAAMCPBridge;