/**
 * Test client for Synaptic Neural Mesh MCP Server
 * Demonstrates AI assistant orchestration of neural mesh
 */

import { spawn } from 'child_process';
import readline from 'readline';

interface MCPRequest {
  jsonrpc: string;
  id: string | number;
  method: string;
  params?: any;
}

interface MCPResponse {
  jsonrpc: string;
  id: string | number;
  result?: any;
  error?: any;
}

class SynapticMCPClient {
  private process: any;
  private requestId: number = 0;
  private pendingRequests: Map<string | number, (response: MCPResponse) => void> = new Map();
  
  async connect() {
    console.log('🔌 Connecting to Synaptic Neural Mesh MCP Server...');
    
    // Start the MCP server as a subprocess
    this.process = spawn('node', ['start-mcp-server.js'], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe']
    });
    
    // Handle server output
    const rl = readline.createInterface({
      input: this.process.stdout,
      crlfDelay: Infinity
    });
    
    rl.on('line', (line: string) => {
      try {
        const response: MCPResponse = JSON.parse(line);
        if (response.id && this.pendingRequests.has(response.id)) {
          const handler = this.pendingRequests.get(response.id)!;
          this.pendingRequests.delete(response.id);
          handler(response);
        }
      } catch (error) {
        console.error('Failed to parse response:', error);
      }
    });
    
    // Handle server errors
    this.process.stderr.on('data', (data: Buffer) => {
      console.error('[Server]', data.toString());
    });
    
    // Initialize connection
    await this.initialize();
  }
  
  private async initialize() {
    const response = await this.sendRequest('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {}
    });
    
    console.log('✅ Connected to Synaptic MCP Server');
    console.log('Server capabilities:', response.result?.capabilities);
  }
  
  private sendRequest(method: string, params?: any): Promise<MCPResponse> {
    return new Promise((resolve) => {
      const id = ++this.requestId;
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id,
        method,
        params
      };
      
      this.pendingRequests.set(id, resolve);
      this.process.stdin.write(JSON.stringify(request) + '\n');
    });
  }
  
  async callTool(name: string, args: any): Promise<any> {
    const response = await this.sendRequest('tools/call', {
      name,
      arguments: args
    });
    
    if (response.error) {
      throw new Error(`Tool error: ${response.error.message}`);
    }
    
    return JSON.parse(response.result?.content?.[0]?.text || '{}');
  }
  
  async listTools(): Promise<any[]> {
    const response = await this.sendRequest('tools/list');
    return response.result?.tools || [];
  }
  
  async demonstrateAIOrchestration() {
    console.log('\n🧠 AI Assistant Orchestrating Neural Mesh Demo\n');
    
    try {
      // Step 1: Initialize neural mesh
      console.log('1️⃣ Initializing synaptic neural mesh...');
      const mesh = await this.callTool('mesh_initialize', {
        topology: 'cortical',
        nodes: 1000,
        connectivity: 0.4,
        activation: 'relu'
      });
      console.log(`   ✅ Mesh created: ${mesh.meshId}`);
      console.log(`   • Topology: ${mesh.topology}`);
      console.log(`   • Nodes: ${mesh.nodes}`);
      
      // Step 2: Create neurons in batch
      console.log('\n2️⃣ Creating neurons in batch...');
      const neurons = await this.callTool('batch_neuron_create', {
        count: 500,
        distribution: 'gaussian',
        layer_distribution: [50, 100, 150, 100, 75, 25]
      });
      console.log(`   ✅ Created ${neurons.created_neurons} neurons`);
      console.log(`   • Distribution: ${neurons.distribution}`);
      console.log(`   • Creation time: ${neurons.creation_time_ms}ms`);
      
      // Step 3: Connect AI assistant
      console.log('\n3️⃣ Connecting AI assistant to mesh...');
      const connection = await this.callTool('assistant_connect', {
        assistant_type: 'claude',
        interface_layer: 3,
        bidirectional: true
      });
      console.log(`   ✅ Connected: ${connection.connection_id}`);
      console.log(`   • Latency: ${connection.latency_ms}ms`);
      console.log(`   • Capabilities:`, Object.keys(connection.capabilities).join(', '));
      
      // Step 4: Inject AI thought
      console.log('\n4️⃣ Injecting AI thought into mesh...');
      const thought = await this.callTool('thought_inject', {
        thought: 'How can we optimize neural network training efficiency?',
        encoding: 'embedding',
        target_layer: 3
      });
      console.log(`   ✅ Thought injected`);
      console.log(`   • Neurons activated: ${thought.injection_results.neurons_activated}`);
      console.log(`   • Resonance score: ${thought.injection_results.resonance_score.toFixed(2)}`);
      
      // Step 5: Train the mesh
      console.log('\n5️⃣ Training neural mesh...');
      const training = await this.callTool('mesh_train', {
        patterns: [
          [0.1, 0.8, 0.3, 0.9, 0.2],
          [0.9, 0.2, 0.7, 0.1, 0.8],
          [0.5, 0.5, 0.5, 0.5, 0.5]
        ],
        epochs: 50,
        learning_rate: 0.01,
        algorithm: 'spike-timing'
      });
      console.log(`   ✅ Training complete`);
      console.log(`   • Final accuracy: ${training.performance.final_accuracy.toFixed(2)}`);
      console.log(`   • Convergence epoch: ${training.performance.convergence_epoch}`);
      
      // Step 6: Query the mesh
      console.log('\n6️⃣ Querying mesh for insights...');
      const query = await this.callTool('mesh_query', {
        query: 'What patterns optimize training efficiency?',
        response_type: 'association',
        depth: 3
      });
      console.log(`   ✅ Query results:`);
      for (const assoc of query.associations) {
        console.log(`   • ${assoc.concept}: ${(assoc.strength * 100).toFixed(0)}%`);
      }
      
      // Step 7: Analyze connectivity
      console.log('\n7️⃣ Analyzing mesh connectivity...');
      const analysis = await this.callTool('connectivity_analyze', {
        meshId: mesh.meshId,
        analysis_type: 'clustering'
      });
      console.log(`   ✅ Analysis complete`);
      console.log(`   • Clustering coefficient: ${analysis.clustering_coefficient.toFixed(2)}`);
      console.log(`   • Clusters found: ${analysis.clusters_found}`);
      
      // Step 8: Optimize topology
      console.log('\n8️⃣ Optimizing mesh topology...');
      const optimization = await this.callTool('optimize_topology', {
        metric: 'efficiency',
        constraints: { max_connections: 10000 }
      });
      console.log(`   ✅ Optimization complete`);
      console.log(`   • Improvement: ${optimization.optimization_results.improvement}`);
      console.log(`   • Iterations: ${optimization.optimization_results.iterations}`);
      
      // Step 9: Stream real-time metrics
      console.log('\n9️⃣ Setting up metric streaming...');
      const stream = await this.callTool('stream_metrics', {
        metrics: ['activity_level', 'connectivity_score', 'efficiency_index'],
        interval: 100
      });
      console.log(`   ✅ Streaming configured`);
      console.log(`   • Stream ID: ${stream.stream_id}`);
      console.log(`   • Endpoint: ${stream.stream_config.endpoint}`);
      
      // Step 10: Save mesh state
      console.log('\n🔟 Saving mesh state...');
      const save = await this.callTool('mesh_save', {
        meshId: mesh.meshId,
        format: 'protobuf',
        compress: true
      });
      console.log(`   ✅ Mesh saved`);
      console.log(`   • Path: ${save.save_path}`);
      console.log(`   • Size: ${save.file_size_mb}MB`);
      
      console.log('\n✨ AI Orchestration Demo Complete!');
      console.log('The AI assistant has successfully:');
      console.log('- Created and configured a neural mesh');
      console.log('- Injected thoughts and trained patterns');
      console.log('- Analyzed and optimized the topology');
      console.log('- Set up real-time monitoring');
      console.log('- Persisted the mesh state');
      
    } catch (error) {
      console.error('❌ Demo error:', error);
    }
  }
  
  async close() {
    if (this.process) {
      this.process.kill();
    }
  }
}

// Run the demo
async function main() {
  const client = new SynapticMCPClient();
  
  try {
    await client.connect();
    
    // List available tools
    const tools = await client.listTools();
    console.log(`\n📋 Available tools: ${tools.length}`);
    console.log('Categories:');
    const categories = new Set(tools.map(t => t.name.split('_')[0]));
    categories.forEach(cat => {
      const count = tools.filter(t => t.name.startsWith(cat)).length;
      console.log(`  • ${cat}: ${count} tools`);
    });
    
    // Run the orchestration demo
    await client.demonstrateAIOrchestration();
    
  } catch (error) {
    console.error('Client error:', error);
  } finally {
    await client.close();
  }
}

// Execute if run directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { SynapticMCPClient };