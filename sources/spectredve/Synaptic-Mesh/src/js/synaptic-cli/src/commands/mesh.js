import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import Table from 'cli-table3';
import { MeshClient } from '../core/mesh-client.js';

export function meshCommand() {
  const cmd = new Command('mesh');
  
  cmd
    .description('Manage neural mesh topology and nodes')
    .option('-p, --port <port>', 'Mesh coordination port', '7070')
    .option('-h, --host <host>', 'Mesh coordination host', 'localhost');
  
  // List nodes
  cmd
    .command('list')
    .alias('ls')
    .description('List all nodes in the mesh')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Fetching mesh nodes...').start();
      
      try {
        const client = new MeshClient(parentOpts.host, parentOpts.port);
        const nodes = await client.getNodes();
        
        spinner.stop();
        
        if (nodes.length === 0) {
          console.log(chalk.yellow('No nodes in the mesh'));
          return;
        }
        
        const table = new Table({
          head: ['ID', 'Type', 'Status', 'Connections', 'Tasks', 'Created'],
          style: { head: ['cyan'] }
        });
        
        nodes.forEach(node => {
          table.push([
            node.id.substring(0, 8),
            node.type,
            node.status === 'active' ? chalk.green(node.status) : chalk.yellow(node.status),
            node.connections.length,
            node.taskCount || 0,
            new Date(node.created).toLocaleString()
          ]);
        });
        
        console.log(table.toString());
        console.log(chalk.gray(`Total nodes: ${nodes.length}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to fetch nodes: ' + error.message));
      }
    });
  
  // Add node
  cmd
    .command('add')
    .description('Add a new node to the mesh')
    .option('-t, --type <type>', 'Node type (agent, coordinator, storage)', 'agent')
    .option('-n, --name <name>', 'Node name')
    .option('--role <role>', 'Node role (researcher, coder, analyst, etc.)')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Adding node to mesh...').start();
      
      try {
        const client = new MeshClient(parentOpts.host, parentOpts.port);
        const node = await client.addNode({
          type: options.type,
          name: options.name,
          role: options.role
        });
        
        spinner.succeed(chalk.green(`Node added: ${node.id}`));
        
        console.log('\n' + chalk.cyan('Node Details:'));
        console.log(chalk.gray('  ID:') + ' ' + node.id);
        console.log(chalk.gray('  Type:') + ' ' + node.type);
        if (node.name) console.log(chalk.gray('  Name:') + ' ' + node.name);
        if (node.role) console.log(chalk.gray('  Role:') + ' ' + node.role);
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to add node: ' + error.message));
      }
    });
  
  // Remove node
  cmd
    .command('remove <nodeId>')
    .alias('rm')
    .description('Remove a node from the mesh')
    .option('-f, --force', 'Force removal without confirmation')
    .action(async (nodeId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Removing node from mesh...').start();
      
      try {
        const client = new MeshClient(parentOpts.host, parentOpts.port);
        
        // TODO: Add confirmation prompt if not forced
        
        await client.removeNode(nodeId);
        spinner.succeed(chalk.green(`Node removed: ${nodeId}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to remove node: ' + error.message));
      }
    });
  
  // Connect nodes
  cmd
    .command('connect <sourceId> <targetId>')
    .description('Create a connection between two nodes')
    .option('-w, --weight <weight>', 'Connection weight (0-1)', '1.0')
    .option('-b, --bidirectional', 'Create bidirectional connection')
    .action(async (sourceId, targetId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Creating connection...').start();
      
      try {
        const client = new MeshClient(parentOpts.host, parentOpts.port);
        
        await client.connectNodes(sourceId, targetId, {
          weight: parseFloat(options.weight),
          bidirectional: options.bidirectional
        });
        
        spinner.succeed(chalk.green('Connection created'));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to create connection: ' + error.message));
      }
    });
  
  // Disconnect nodes
  cmd
    .command('disconnect <sourceId> <targetId>')
    .description('Remove connection between two nodes')
    .action(async (sourceId, targetId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Removing connection...').start();
      
      try {
        const client = new MeshClient(parentOpts.host, parentOpts.port);
        await client.disconnectNodes(sourceId, targetId);
        
        spinner.succeed(chalk.green('Connection removed'));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to remove connection: ' + error.message));
      }
    });
  
  // Show topology
  cmd
    .command('topology')
    .alias('topo')
    .description('Display mesh topology visualization')
    .option('-f, --format <format>', 'Output format (ascii, json)', 'ascii')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Fetching topology...').start();
      
      try {
        const client = new MeshClient(parentOpts.host, parentOpts.port);
        const topology = await client.getTopology();
        
        spinner.stop();
        
        if (options.format === 'json') {
          console.log(JSON.stringify(topology, null, 2));
        } else {
          // ASCII visualization
          console.log(chalk.cyan('\nMesh Topology:'));
          console.log(chalk.gray('─'.repeat(50)));
          
          // Display based on topology type
          displayTopology(topology);
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to fetch topology: ' + error.message));
      }
    });
  
  // Optimize topology
  cmd
    .command('optimize')
    .description('Optimize mesh topology for performance')
    .option('--strategy <strategy>', 'Optimization strategy (balanced, latency, throughput)', 'balanced')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Optimizing mesh topology...').start();
      
      try {
        const client = new MeshClient(parentOpts.host, parentOpts.port);
        const result = await client.optimizeTopology(options.strategy);
        
        spinner.succeed(chalk.green('Topology optimized'));
        
        console.log('\n' + chalk.cyan('Optimization Results:'));
        console.log(chalk.gray('  Strategy:') + ' ' + result.strategy);
        console.log(chalk.gray('  Nodes affected:') + ' ' + result.nodesAffected);
        console.log(chalk.gray('  Connections changed:') + ' ' + result.connectionsChanged);
        console.log(chalk.gray('  Performance gain:') + ' ' + chalk.green(`+${result.performanceGain}%`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to optimize topology: ' + error.message));
      }
    });
    
  return cmd;
}

function displayTopology(topology) {
  switch (topology.type) {
    case 'mesh':
      displayMeshTopology(topology);
      break;
    case 'hierarchical':
      displayHierarchicalTopology(topology);
      break;
    case 'ring':
      displayRingTopology(topology);
      break;
    case 'star':
      displayStarTopology(topology);
      break;
    default:
      console.log(chalk.yellow('Unknown topology type: ' + topology.type));
  }
}

function displayMeshTopology(topology) {
  console.log(chalk.white('Type: Full Mesh'));
  console.log(chalk.gray(`Nodes: ${topology.nodes.length}`));
  console.log(chalk.gray(`Connections: ${topology.connections.length}`));
  
  // Simple ASCII representation
  console.log('\n' + chalk.gray('  [Node] ←→ [Node] ←→ [Node]'));
  console.log(chalk.gray('    ↑  ↘    ↗  ↘    ↗  ↑'));
  console.log(chalk.gray('    ↓    ↘↗      ↘↗    ↓'));
  console.log(chalk.gray('  [Node] ←→ [Node] ←→ [Node]'));
}

function displayHierarchicalTopology(topology) {
  console.log(chalk.white('Type: Hierarchical'));
  console.log(chalk.gray(`Levels: ${topology.levels}`));
  console.log(chalk.gray(`Root nodes: ${topology.roots.length}`));
  
  // ASCII tree representation
  console.log('\n' + chalk.gray('       [Root]'));
  console.log(chalk.gray('      ↙  ↓  ↘'));
  console.log(chalk.gray('  [Node] [Node] [Node]'));
  console.log(chalk.gray('   ↙ ↓     ↓     ↓ ↘'));
  console.log(chalk.gray('[Leaf] [Leaf] [Leaf] [Leaf]'));
}

function displayRingTopology(topology) {
  console.log(chalk.white('Type: Ring'));
  console.log(chalk.gray(`Nodes: ${topology.nodes.length}`));
  
  // ASCII ring representation
  console.log('\n' + chalk.gray('    [Node] → [Node]'));
  console.log(chalk.gray('      ↑         ↓'));
  console.log(chalk.gray('   [Node]    [Node]'));
  console.log(chalk.gray('      ↑         ↓'));
  console.log(chalk.gray('    [Node] ← [Node]'));
}

function displayStarTopology(topology) {
  console.log(chalk.white('Type: Star'));
  console.log(chalk.gray(`Center: ${topology.center.id}`));
  console.log(chalk.gray(`Spokes: ${topology.spokes.length}`));
  
  // ASCII star representation
  console.log('\n' + chalk.gray('    [Node]   [Node]'));
  console.log(chalk.gray('        ↘   ↙'));
  console.log(chalk.gray('[Node] → [Hub] ← [Node]'));
  console.log(chalk.gray('        ↗   ↖'));
  console.log(chalk.gray('    [Node]   [Node]'));
}