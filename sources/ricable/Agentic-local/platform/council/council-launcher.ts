#!/usr/bin/env npx ts-node
/**
 * MLX Deep Council Launcher
 *
 * CLI tool for launching and managing distributed council sessions
 * across multiple Mac machines.
 *
 * Usage:
 *   npx ts-node council-launcher.ts launch --config council.json
 *   npx ts-node council-launcher.ts query "What is the meaning of life?"
 *   npx ts-node council-launcher.ts status
 *   npx ts-node council-launcher.ts configure --hosts mac1,mac2,mac3
 */

import { spawn, exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs';
import * as path from 'path';
import * as readline from 'readline';

import {
  MLXDeepCouncil,
  createLocalCouncil,
  createDistributedCouncil,
  CouncilConfig,
  CouncilSession,
  DistributedNode,
} from './mlx-deep-council';

const execAsync = promisify(exec);

// ============================================================================
// CONFIGURATION
// ============================================================================

interface LauncherConfig {
  name: string;
  nodes: NodeConfig[];
  defaultModel: string;
  chairmanModel?: string;
  backend: 'ring' | 'mpi' | 'auto';
  votingStrategy: 'weighted' | 'majority' | 'ranked-choice';
}

interface NodeConfig {
  hostname: string;
  ip: string;
  sshUser?: string;
  port: number;
  model?: string;
  gpuMemory: number;
  chip: string;
}

const DEFAULT_CONFIG: LauncherConfig = {
  name: 'MLX Deep Council',
  nodes: [
    {
      hostname: 'localhost',
      ip: '127.0.0.1',
      port: 8080,
      model: 'mlx-community/Llama-3.2-3B-Instruct-4bit',
      gpuMemory: 32,
      chip: 'M3 Max',
    },
  ],
  defaultModel: 'mlx-community/Llama-3.2-3B-Instruct-4bit',
  backend: 'ring',
  votingStrategy: 'weighted',
};

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

async function loadConfig(configPath: string): Promise<LauncherConfig> {
  try {
    const content = await fs.promises.readFile(configPath, 'utf-8');
    return { ...DEFAULT_CONFIG, ...JSON.parse(content) };
  } catch (error) {
    console.log('No config file found, using defaults');
    return DEFAULT_CONFIG;
  }
}

async function saveConfig(configPath: string, config: LauncherConfig): Promise<void> {
  await fs.promises.writeFile(configPath, JSON.stringify(config, null, 2));
  console.log(`Configuration saved to ${configPath}`);
}

async function detectMacs(): Promise<NodeConfig[]> {
  console.log('Detecting Macs on the network...');

  const nodes: NodeConfig[] = [];

  // First, add localhost
  try {
    const { stdout: chipInfo } = await execAsync('sysctl -n machdep.cpu.brand_string');
    const { stdout: memInfo } = await execAsync('sysctl -n hw.memsize');

    const memoryGB = Math.round(parseInt(memInfo) / (1024 * 1024 * 1024));

    nodes.push({
      hostname: 'localhost',
      ip: '127.0.0.1',
      port: 8080,
      gpuMemory: memoryGB,
      chip: chipInfo.trim().includes('Apple') ? 'M-series' : 'Intel',
    });
  } catch {
    // Not running on macOS
    nodes.push({
      hostname: 'localhost',
      ip: '127.0.0.1',
      port: 8080,
      gpuMemory: 16,
      chip: 'Unknown',
    });
  }

  // Check for Thunderbolt-connected Macs
  try {
    const { stdout } = await execAsync('system_profiler SPThunderboltDataType -json 2>/dev/null || echo "{}"');
    const data = JSON.parse(stdout);

    // Parse Thunderbolt devices (simplified)
    if (data.SPThunderboltDataType) {
      console.log('Found Thunderbolt connections, checking for Macs...');
    }
  } catch {
    // No Thunderbolt or not on macOS
  }

  return nodes;
}

async function testNodeConnectivity(node: NodeConfig): Promise<boolean> {
  if (node.hostname === 'localhost') {
    return true;
  }

  try {
    const ssh = node.sshUser ? `${node.sshUser}@${node.hostname}` : node.hostname;
    await execAsync(`ssh -o ConnectTimeout=5 ${ssh} "echo ok"`, { timeout: 10000 });
    return true;
  } catch {
    return false;
  }
}

async function startWorkerOnNode(node: NodeConfig, model: string): Promise<void> {
  const workerScript = path.join(__dirname, 'python', 'mlx_council_worker.py');

  if (node.hostname === 'localhost') {
    // Start locally
    const proc = spawn('python', [
      workerScript,
      '--model', model,
      '--port', String(node.port),
    ], {
      detached: true,
      stdio: 'ignore',
    });
    proc.unref();
    console.log(`Started worker on localhost:${node.port}`);
  } else {
    // Start remotely via SSH
    const ssh = node.sshUser ? `${node.sshUser}@${node.hostname}` : node.hostname;
    const cmd = `python ${workerScript} --model ${model} --port ${node.port} &`;

    await execAsync(`ssh ${ssh} "${cmd}"`);
    console.log(`Started worker on ${node.hostname}:${node.port}`);
  }

  // Wait for worker to be ready
  await waitForWorker(node);
}

async function waitForWorker(node: NodeConfig, timeout: number = 60000): Promise<void> {
  const startTime = Date.now();
  const endpoint = `http://${node.ip}:${node.port}/health`;

  while (Date.now() - startTime < timeout) {
    try {
      const response = await fetch(endpoint);
      if (response.ok) {
        console.log(`Worker at ${node.hostname}:${node.port} is ready`);
        return;
      }
    } catch {
      // Not ready yet
    }
    await new Promise(r => setTimeout(r, 1000));
  }

  throw new Error(`Worker at ${node.hostname}:${node.port} failed to start`);
}

function formatSession(session: CouncilSession): string {
  let output = '';

  output += '\n' + '='.repeat(80) + '\n';
  output += `Council Session: ${session.id}\n`;
  output += `Status: ${session.stage}\n`;
  output += `Consensus: ${session.consensusReached ? 'REACHED' : 'NOT REACHED'}\n`;
  output += '='.repeat(80) + '\n\n';

  // Individual responses
  output += 'INDIVIDUAL RESPONSES:\n';
  output += '-'.repeat(40) + '\n';
  for (const response of session.individualResponses) {
    const score = session.aggregatedScores.get(response.anonymousId) || 0;
    output += `\n[${response.anonymousId}] Score: ${score.toFixed(1)}/10\n`;
    output += `${response.content.slice(0, 500)}...\n`;
  }

  // Chairman synthesis
  if (session.chairmanSynthesis) {
    output += '\n' + '='.repeat(80) + '\n';
    output += 'CHAIRMAN SYNTHESIS:\n';
    output += '='.repeat(80) + '\n\n';
    output += session.chairmanSynthesis.finalResponse + '\n';

    if (session.chairmanSynthesis.reasoning) {
      output += '\nREASONING:\n';
      output += session.chairmanSynthesis.reasoning + '\n';
    }

    output += `\nConfidence: ${(session.chairmanSynthesis.confidenceScore * 100).toFixed(0)}%\n`;

    if (session.chairmanSynthesis.dissent) {
      output += `\nDissenting View: ${session.chairmanSynthesis.dissent}\n`;
    }
  }

  // Metrics
  output += '\nMETRICS:\n';
  output += `-  Total latency: ${session.metrics.totalLatency}ms\n`;
  output += `  Stage 1 (Responses): ${session.metrics.stage1Latency}ms\n`;
  output += `  Stage 2 (Peer Review): ${session.metrics.stage2Latency}ms\n`;
  output += `  Stage 3 (Synthesis): ${session.metrics.stage3Latency}ms\n`;
  output += `  Participating members: ${session.metrics.participatingMembers}\n`;
  output += `  Consensus strength: ${(session.metrics.consensusStrength * 100).toFixed(0)}%\n`;

  return output;
}

// ============================================================================
// INTERACTIVE MODE
// ============================================================================

async function runInteractive(council: MLXDeepCouncil): Promise<void> {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  console.log('\nMLX Deep Council Interactive Mode');
  console.log('Type your query and press Enter. Type "exit" to quit.\n');

  const askQuestion = (): void => {
    rl.question('Council> ', async (query) => {
      if (query.toLowerCase() === 'exit' || query.toLowerCase() === 'quit') {
        console.log('Shutting down council...');
        await council.shutdown();
        rl.close();
        return;
      }

      if (query.toLowerCase() === 'status') {
        const status = council.getStatus();
        console.log('\nCouncil Status:');
        console.log(`  Name: ${status.name}`);
        console.log(`  Members: ${status.members.length}`);
        console.log(`  Chairman: ${status.chairman?.name || 'None'}`);
        console.log(`  Active Sessions: ${status.activeSessions}`);
        console.log(`  Completed Sessions: ${status.completedSessions}\n`);
        askQuestion();
        return;
      }

      if (query.trim() === '') {
        askQuestion();
        return;
      }

      try {
        console.log('\nProcessing query through council...\n');

        const session = await council.query({
          content: query,
          requireConsensus: true,
          minAgreement: 0.6,
        });

        console.log(formatSession(session));
      } catch (error) {
        console.error('Error:', error);
      }

      askQuestion();
    });
  };

  askQuestion();
}

// ============================================================================
// COMMANDS
// ============================================================================

async function commandLaunch(args: string[]): Promise<void> {
  const configPath = args.find(a => a.startsWith('--config='))?.split('=')[1]
    || args[args.indexOf('--config') + 1]
    || 'council.json';

  const config = await loadConfig(configPath);

  console.log(`Launching MLX Deep Council: ${config.name}`);
  console.log(`Nodes: ${config.nodes.length}`);
  console.log(`Backend: ${config.backend}`);

  // Test connectivity to all nodes
  console.log('\nTesting node connectivity...');
  for (const node of config.nodes) {
    const connected = await testNodeConnectivity(node);
    console.log(`  ${node.hostname}: ${connected ? 'OK' : 'FAILED'}`);

    if (!connected && node.hostname !== 'localhost') {
      console.error(`Cannot connect to ${node.hostname}. Check SSH configuration.`);
      process.exit(1);
    }
  }

  // Start workers on all nodes
  console.log('\nStarting workers...');
  for (const node of config.nodes) {
    await startWorkerOnNode(node, node.model || config.defaultModel);
  }

  // Create and initialize council
  const council = createDistributedCouncil({
    name: config.name,
    nodes: config.nodes.map(n => ({
      hostname: n.hostname,
      ip: n.ip,
      model: n.model || config.defaultModel,
      gpuMemory: n.gpuMemory,
      chip: n.chip,
    })),
  });

  await council.initialize();

  // Enter interactive mode
  await runInteractive(council);
}

async function commandQuery(args: string[]): Promise<void> {
  const query = args.join(' ');

  if (!query) {
    console.error('Usage: council-launcher query "Your question here"');
    process.exit(1);
  }

  const configPath = 'council.json';
  const config = await loadConfig(configPath);

  // Create a quick local council for single queries
  const council = createLocalCouncil({
    name: 'Quick Council',
    models: [
      'mlx-community/Llama-3.2-3B-Instruct-4bit',
      'mlx-community/Mistral-7B-Instruct-v0.3-4bit',
      'mlx-community/Qwen2.5-7B-Instruct-4bit',
    ],
  });

  try {
    await council.initialize();

    console.log('Querying council...\n');

    const session = await council.query({
      content: query,
      requireConsensus: true,
    });

    console.log(formatSession(session));
  } finally {
    await council.shutdown();
  }
}

async function commandConfigure(args: string[]): Promise<void> {
  const hostsArg = args.find(a => a.startsWith('--hosts='))?.split('=')[1]
    || args[args.indexOf('--hosts') + 1];

  const outputPath = args.find(a => a.startsWith('--output='))?.split('=')[1]
    || args[args.indexOf('--output') + 1]
    || 'council.json';

  let nodes: NodeConfig[];

  if (hostsArg) {
    // Parse provided hosts
    const hosts = hostsArg.split(',').map(h => h.trim());

    nodes = await Promise.all(hosts.map(async (host, index) => {
      // Try to detect node info via SSH
      let gpuMemory = 32;
      let chip = 'Unknown';

      if (host !== 'localhost') {
        try {
          const { stdout: memInfo } = await execAsync(
            `ssh -o ConnectTimeout=5 ${host} "sysctl -n hw.memsize"`
          );
          gpuMemory = Math.round(parseInt(memInfo) / (1024 * 1024 * 1024));

          const { stdout: chipInfo } = await execAsync(
            `ssh -o ConnectTimeout=5 ${host} "sysctl -n machdep.cpu.brand_string"`
          );
          if (chipInfo.includes('Apple')) {
            chip = 'Apple Silicon';
          }
        } catch {
          console.warn(`Could not detect hardware info for ${host}`);
        }
      }

      return {
        hostname: host,
        ip: host === 'localhost' ? '127.0.0.1' : host,
        port: 8080 + index,
        gpuMemory,
        chip,
      };
    }));
  } else {
    // Auto-detect nodes
    nodes = await detectMacs();
  }

  const config: LauncherConfig = {
    ...DEFAULT_CONFIG,
    nodes,
  };

  await saveConfig(outputPath, config);

  console.log('\nGenerated configuration:');
  console.log(JSON.stringify(config, null, 2));
}

async function commandStatus(): Promise<void> {
  const configPath = 'council.json';
  const config = await loadConfig(configPath);

  console.log('Council Configuration Status');
  console.log('='.repeat(40));
  console.log(`Name: ${config.name}`);
  console.log(`Backend: ${config.backend}`);
  console.log(`Voting Strategy: ${config.votingStrategy}`);
  console.log(`\nNodes (${config.nodes.length}):`);

  for (const node of config.nodes) {
    const connected = await testNodeConnectivity(node);

    // Try to get worker status
    let workerStatus = 'not running';
    if (connected) {
      try {
        const response = await fetch(`http://${node.ip}:${node.port}/health`, {
          signal: AbortSignal.timeout(2000),
        });
        if (response.ok) {
          const data = await response.json();
          workerStatus = data.is_chairman ? 'running (chairman)' : 'running';
        }
      } catch {
        workerStatus = 'not running';
      }
    }

    console.log(`  - ${node.hostname}`);
    console.log(`    IP: ${node.ip}:${node.port}`);
    console.log(`    Chip: ${node.chip} (${node.gpuMemory}GB)`);
    console.log(`    SSH: ${connected ? 'OK' : 'FAILED'}`);
    console.log(`    Worker: ${workerStatus}`);
  }
}

async function commandStop(): Promise<void> {
  const configPath = 'council.json';
  const config = await loadConfig(configPath);

  console.log('Stopping council workers...');

  for (const node of config.nodes) {
    if (node.hostname === 'localhost') {
      // Kill local process
      try {
        await execAsync(`pkill -f "mlx_council_worker.*--port ${node.port}"`);
        console.log(`  Stopped worker on localhost:${node.port}`);
      } catch {
        // Process may not be running
      }
    } else {
      // Kill remote process
      const ssh = node.sshUser ? `${node.sshUser}@${node.hostname}` : node.hostname;
      try {
        await execAsync(`ssh ${ssh} "pkill -f 'mlx_council_worker.*--port ${node.port}'"`);
        console.log(`  Stopped worker on ${node.hostname}:${node.port}`);
      } catch {
        // Process may not be running
      }
    }
  }

  console.log('All workers stopped');
}

// ============================================================================
// MAIN
// ============================================================================

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0];

  console.log(`
 __  __ _   __  __   ___                   ___                      _ _
|  \\/  | |  \\ \\/ /  |   \\ ___ ___ _ __    / __|___ _  _ _ _  __ ___(_) |
| |\\/| | |__ >  <   | |) / -_) -_) '_ \\  | (__/ _ \\ || | ' \\/ _/ _ \\ | |
|_|  |_|____|/_/\\_\\ |___/\\___\\___| .__/   \\___\\___/\\_,_|_||_\\__\\___/_|_|
                                  |_|

Distributed Multi-Model Consensus System for Apple Silicon
`);

  switch (command) {
    case 'launch':
      await commandLaunch(args.slice(1));
      break;

    case 'query':
      await commandQuery(args.slice(1));
      break;

    case 'configure':
    case 'config':
      await commandConfigure(args.slice(1));
      break;

    case 'status':
      await commandStatus();
      break;

    case 'stop':
      await commandStop();
      break;

    case 'help':
    case '--help':
    case '-h':
    default:
      console.log(`
Usage: council-launcher <command> [options]

Commands:
  launch              Launch the council and enter interactive mode
    --config <file>   Path to configuration file (default: council.json)

  query <text>        Query the council with a single question
    Example: council-launcher query "What is the best sorting algorithm?"

  configure           Configure the council
    --hosts <list>    Comma-separated list of hostnames
    --output <file>   Output configuration file (default: council.json)

  status              Show council status and node connectivity

  stop                Stop all running council workers

  help                Show this help message

Examples:
  # Configure a 3-node council
  council-launcher configure --hosts mac-mini,macbook-pro,mac-studio

  # Launch the council
  council-launcher launch --config council.json

  # Quick query (uses local council)
  council-launcher query "Explain quantum computing"

Environment Variables:
  COUNCIL_DEFAULT_MODEL    Default model to use
  COUNCIL_BACKEND          Distributed backend (ring or mpi)

For more information, see: https://ml-explore.github.io/mlx/build/html/usage/distributed.html
`);
      break;
  }
}

// Run main
main().catch((error) => {
  console.error('Error:', error);
  process.exit(1);
});
