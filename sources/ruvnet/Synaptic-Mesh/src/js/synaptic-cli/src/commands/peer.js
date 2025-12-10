import { Command } from 'commander';
import chalk from 'chalk';
import ora from 'ora';
import Table from 'cli-table3';
import { PeerClient } from '../core/peer-client.js';

export function peerCommand() {
  const cmd = new Command('peer');
  
  cmd
    .description('Manage peer-to-peer connections')
    .option('-p, --port <port>', 'P2P service port', '7073')
    .option('-h, --host <host>', 'P2P service host', 'localhost');
  
  // List peers
  cmd
    .command('list')
    .alias('ls')
    .description('List connected peers')
    .option('--status <status>', 'Filter by status (connected, disconnected, pending)')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Fetching peers...').start();
      
      try {
        const client = new PeerClient(parentOpts.host, parentOpts.port);
        const peers = await client.getPeers(options.status);
        
        spinner.stop();
        
        if (peers.length === 0) {
          console.log(chalk.yellow('No peers found'));
          return;
        }
        
        const table = new Table({
          head: ['ID', 'Address', 'Status', 'Latency', 'Data Shared', 'Connected'],
          style: { head: ['cyan'] }
        });
        
        peers.forEach(peer => {
          const statusColor = {
            connected: 'green',
            disconnected: 'red',
            pending: 'yellow'
          }[peer.status] || 'gray';
          
          table.push([
            peer.id.substring(0, 12),
            peer.address,
            chalk[statusColor](peer.status),
            peer.latency ? `${peer.latency}ms` : 'N/A',
            formatBytes(peer.dataShared || 0),
            peer.connectedAt ? new Date(peer.connectedAt).toLocaleString() : 'N/A'
          ]);
        });
        
        console.log(table.toString());
        console.log(chalk.gray(`Total peers: ${peers.length}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to fetch peers: ' + error.message));
      }
    });
  
  // Connect to peer
  cmd
    .command('connect <address>')
    .description('Connect to a peer')
    .option('-t, --timeout <ms>', 'Connection timeout', '10000')
    .action(async (address, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Connecting to peer...').start();
      
      try {
        const client = new PeerClient(parentOpts.host, parentOpts.port);
        const peer = await client.connectToPeer(address, {
          timeout: parseInt(options.timeout)
        });
        
        spinner.succeed(chalk.green(`Connected to peer: ${peer.id}`));
        
        console.log('\n' + chalk.cyan('Peer Details:'));
        console.log(chalk.gray('  ID:') + ' ' + peer.id);
        console.log(chalk.gray('  Address:') + ' ' + peer.address);
        console.log(chalk.gray('  Protocol:') + ' ' + peer.protocol);
        console.log(chalk.gray('  Latency:') + ' ' + `${peer.latency}ms`);
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to connect: ' + error.message));
      }
    });
  
  // Disconnect from peer
  cmd
    .command('disconnect <peerId>')
    .description('Disconnect from a peer')
    .action(async (peerId, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Disconnecting from peer...').start();
      
      try {
        const client = new PeerClient(parentOpts.host, parentOpts.port);
        await client.disconnectFromPeer(peerId);
        
        spinner.succeed(chalk.green(`Disconnected from peer: ${peerId}`));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to disconnect: ' + error.message));
      }
    });
  
  // Discover peers
  cmd
    .command('discover')
    .description('Discover nearby peers')
    .option('-r, --radius <hops>', 'Discovery radius in hops', '3')
    .option('-t, --timeout <ms>', 'Discovery timeout', '30000')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Discovering peers...').start();
      
      try {
        const client = new PeerClient(parentOpts.host, parentOpts.port);
        const discovered = await client.discoverPeers({
          radius: parseInt(options.radius),
          timeout: parseInt(options.timeout)
        });
        
        spinner.succeed(chalk.green(`Discovered ${discovered.length} peers`));
        
        if (discovered.length > 0) {
          const table = new Table({
            head: ['ID', 'Address', 'Distance', 'Protocol', 'Capabilities'],
            style: { head: ['cyan'] }
          });
          
          discovered.forEach(peer => {
            table.push([
              peer.id.substring(0, 12),
              peer.address,
              `${peer.distance} hops`,
              peer.protocol,
              peer.capabilities.join(', ')
            ]);
          });
          
          console.log('\n' + table.toString());
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Discovery failed: ' + error.message));
      }
    });
  
  // Share data with peers
  cmd
    .command('share <file>')
    .description('Share a file with the peer network')
    .option('-p, --peers <peers>', 'Specific peer IDs (comma-separated)')
    .option('--replicas <count>', 'Number of replicas', '3')
    .action(async (file, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Sharing file...').start();
      
      try {
        const client = new PeerClient(parentOpts.host, parentOpts.port);
        
        const targetPeers = options.peers ? options.peers.split(',') : undefined;
        
        const sharing = await client.shareFile(file, {
          targetPeers,
          replicas: parseInt(options.replicas)
        });
        
        spinner.succeed(chalk.green('File shared successfully'));
        
        console.log('\n' + chalk.cyan('Sharing Details:'));
        console.log(chalk.gray('  File hash:') + ' ' + sharing.hash);
        console.log(chalk.gray('  Size:') + ' ' + formatBytes(sharing.size));
        console.log(chalk.gray('  Replicas:') + ' ' + sharing.replicas.length);
        console.log(chalk.gray('  Peers:') + ' ' + sharing.replicas.map(r => r.peerId.substring(0, 8)).join(', '));
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to share file: ' + error.message));
      }
    });
  
  // Download from peers
  cmd
    .command('download <hash>')
    .description('Download a file from the peer network')
    .option('-o, --output <file>', 'Output file path')
    .action(async (hash, options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Downloading file...').start();
      
      try {
        const client = new PeerClient(parentOpts.host, parentOpts.port);
        
        const download = await client.downloadFile(hash, {
          outputPath: options.output
        });
        
        spinner.succeed(chalk.green('File downloaded successfully'));
        
        console.log('\n' + chalk.cyan('Download Details:'));
        console.log(chalk.gray('  File:') + ' ' + download.path);
        console.log(chalk.gray('  Size:') + ' ' + formatBytes(download.size));
        console.log(chalk.gray('  Source peer:') + ' ' + download.sourcePeer.substring(0, 8));
        console.log(chalk.gray('  Duration:') + ' ' + `${download.duration}ms`);
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to download file: ' + error.message));
      }
    });
  
  // Show network status
  cmd
    .command('status')
    .description('Show peer network status')
    .action(async (options, command) => {
      const parentOpts = command.parent.opts();
      const spinner = ora('Fetching network status...').start();
      
      try {
        const client = new PeerClient(parentOpts.host, parentOpts.port);
        const status = await client.getNetworkStatus();
        
        spinner.stop();
        
        console.log(chalk.cyan('\nNetwork Status:'));
        console.log(chalk.gray('  Node ID:') + ' ' + status.nodeId);
        console.log(chalk.gray('  Connected peers:') + ' ' + status.connectedPeers);
        console.log(chalk.gray('  Network size:') + ' ' + status.networkSize);
        console.log(chalk.gray('  Data shared:') + ' ' + formatBytes(status.dataShared));
        console.log(chalk.gray('  Data received:') + ' ' + formatBytes(status.dataReceived));
        console.log(chalk.gray('  Uptime:') + ' ' + formatDuration(status.uptime));
        
        if (status.protocols) {
          console.log('\n' + chalk.cyan('Supported Protocols:'));
          status.protocols.forEach(protocol => {
            console.log(`  ${chalk.green('✓')} ${protocol.name} (${protocol.version})`);
          });
        }
        
      } catch (error) {
        spinner.fail(chalk.red('Failed to fetch status: ' + error.message));
      }
    });
    
  return cmd;
}

function formatBytes(bytes) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let size = bytes;
  let unit = 0;
  
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit++;
  }
  
  return `${size.toFixed(2)} ${units[unit]}`;
}

function formatDuration(ms) {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  if (days > 0) {
    return `${days}d ${hours % 24}h`;
  } else if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  } else if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  } else {
    return `${seconds}s`;
  }
}