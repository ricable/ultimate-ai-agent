import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import Table from 'cli-table3';
import { loadConfig } from '../config/loader.js';
import { MeshClient } from '../core/mesh-client.js';

export function statusCommand() {
  const cmd = new Command('status');
  
  cmd
    .description('Show Synaptic Neural Mesh status')
    .option('-w, --watch', 'Watch status updates')
    .option('-r, --refresh <seconds>', 'Refresh interval for watch mode', '5')
    .action(async (options) => {
      if (options.watch) {
        await watchStatus(parseInt(options.refresh));
      } else {
        await showStatus();
      }
    });
    
  return cmd;
}

async function showStatus() {
  const spinner = ora('Fetching mesh status...').start();
  
  try {
    // Load configuration
    const config = await loadConfig();
    if (!config) {
      spinner.fail('No configuration found. Run "synaptic init" first.');
      return;
    }
    
    // Get mesh status
    const meshClient = new MeshClient('localhost', 7070);
    const meshStatus = await meshClient.getStatus();
    
    spinner.stop();
    
    // Display overall status
    console.log(chalk.cyan('\n🧠 Synaptic Neural Mesh Status\n'));
    
    const statusTable = new Table({
      style: { head: ['cyan'] }
    });
    
    statusTable.push(
      ['Project', config.project.name],
      ['Mesh Status', meshStatus.running ? chalk.green('Running') : chalk.red('Stopped')],
      ['Topology', config.mesh.topology],
      ['Nodes', `${meshStatus.activeNodes}/${meshStatus.totalNodes}`],
      ['Uptime', formatDuration(meshStatus.uptime)],
      ['Version', '1.0.0-alpha.1']
    );
    
    console.log(statusTable.toString());
    
    // Services status
    console.log('\n' + chalk.cyan('Services:'));
    const servicesTable = new Table({
      head: ['Service', 'Status', 'Port', 'Connections'],
      style: { head: ['cyan'] }
    });
    
    const services = [
      {
        name: 'Mesh Coordination',
        status: meshStatus.running ? 'Running' : 'Stopped',
        port: 7070,
        connections: meshStatus.connections || 0
      },
      {
        name: 'Neural Network',
        status: meshStatus.neural?.running ? 'Running' : 'Stopped',
        port: 7071,
        connections: meshStatus.neural?.connections || 0
      },
      {
        name: 'DAG Workflows',
        status: meshStatus.dag?.running ? 'Running' : 'Stopped',
        port: 7072,
        connections: meshStatus.dag?.connections || 0
      },
      {
        name: 'P2P Network',
        status: meshStatus.p2p?.running ? 'Running' : 'Stopped',
        port: 7073,
        connections: meshStatus.p2p?.peers || 0
      }
    ];
    
    if (config.features.mcp) {
      services.push({
        name: 'MCP Server',
        status: meshStatus.mcp?.running ? 'Running' : 'Stopped',
        port: 3000,
        connections: meshStatus.mcp?.connections || 0
      });
    }
    
    services.forEach(service => {
      const statusColor = service.status === 'Running' ? 'green' : 'red';
      servicesTable.push([
        service.name,
        chalk[statusColor](service.status),
        service.port,
        service.connections
      ]);
    });
    
    console.log(servicesTable.toString());
    
    // Performance metrics
    if (meshStatus.metrics) {
      console.log('\n' + chalk.cyan('Performance:'));
      const metricsTable = new Table({
        head: ['Metric', 'Value'],
        style: { head: ['cyan'] }
      });
      
      metricsTable.push(
        ['Tasks Processed', meshStatus.metrics.tasksProcessed || 0],
        ['Average Latency', `${meshStatus.metrics.avgLatency || 0}ms`],
        ['Memory Usage', formatBytes(meshStatus.metrics.memoryUsage || 0)],
        ['CPU Usage', `${meshStatus.metrics.cpuUsage || 0}%`],
        ['Network I/O', formatBytes(meshStatus.metrics.networkIO || 0)]
      );
      
      console.log(metricsTable.toString());
    }
    
    // Recent activity
    if (meshStatus.activity && meshStatus.activity.length > 0) {
      console.log('\n' + chalk.cyan('Recent Activity:'));
      meshStatus.activity.slice(0, 5).forEach(activity => {
        const time = new Date(activity.timestamp).toLocaleTimeString();
        console.log(`  ${chalk.gray(time)} ${activity.message}`);
      });
    }
    
  } catch (error) {
    spinner.fail(chalk.red('Failed to fetch status: ' + error.message));
  }
}

async function watchStatus(refreshInterval) {
  console.log(chalk.cyan('Watching mesh status... (Press Ctrl+C to stop)\n'));
  
  const watchLoop = async () => {
    // Clear screen
    process.stdout.write('\x1Bc');
    
    // Show timestamp
    console.log(chalk.gray(`Last updated: ${new Date().toLocaleString()}\n`));
    
    // Show status
    await showStatus();
    
    // Schedule next update
    setTimeout(watchLoop, refreshInterval * 1000);
  };
  
  // Handle Ctrl+C
  process.on('SIGINT', () => {
    console.log('\n' + chalk.yellow('Stopping status watch...'));
    process.exit(0);
  });
  
  // Start watching
  await watchLoop();
}

function formatDuration(ms) {
  if (!ms) return 'N/A';
  
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) {
    return `${days}d ${hours % 24}h ${minutes % 60}m`;
  } else if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unit = 0;
  
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit++;
  }
  
  return `${size.toFixed(2)} ${units[unit]}`;
}