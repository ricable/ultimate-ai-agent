#!/usr/bin/env node
/**
 * Synaptic Neural Mesh CLI
 * Main command-line interface for the distributed neural mesh framework
 */

const { Command } = require('commander');
const chalk = require('chalk');
const figlet = require('figlet');
const gradient = require('gradient-string');
const ora = require('ora');
const fs = require('fs-extra');
const path = require('path');
const { version } = require('../package.json');

// Import command modules
const initCommand = require('./commands/init');
const startCommand = require('./commands/start');
const meshCommand = require('./commands/mesh');
const neuralCommand = require('./commands/neural');
const dagCommand = require('./commands/dag');
const peerCommand = require('./commands/peer');
const statusCommand = require('./commands/status');

// Create main program
const program = new Command();

// Display banner
function displayBanner() {
  try {
    console.log(
      gradient.rainbow(
        figlet.textSync('Synaptic', {
          font: 'ANSI Shadow',
          horizontalLayout: 'default',
          verticalLayout: 'default'
        })
      )
    );
  } catch (error) {
    // Fallback if figlet fails
    console.log(chalk.cyan.bold('\n🧠 SYNAPTIC NEURAL MESH'));
  }
  console.log(
    chalk.cyan('  Distributed Neural Mesh Framework v' + version)
  );
  console.log(
    chalk.gray('  Decentralized AI orchestration at scale\n')
  );
}

// Configure program
program
  .name('synaptic')
  .description('Distributed neural mesh framework for decentralized AI orchestration')
  .version(version)
  .hook('preAction', () => {
    if (process.argv.length === 3 && process.argv[2] === '--help') {
      displayBanner();
    }
  });

// Initialize command
program
  .command('init [project-name]')
  .description('Initialize a new Synaptic Neural Mesh project')
  .option('-t, --template <template>', 'Use a specific template', 'default')
  .option('--no-install', 'Skip dependency installation')
  .option('--docker', 'Include Docker configuration')
  .option('--k8s', 'Include Kubernetes manifests')
  .action(async (projectName, options) => {
    displayBanner();
    await initCommand.execute(projectName, options);
  });

// Start command
program
  .command('start')
  .description('Start the Synaptic Neural Mesh node')
  .option('-c, --config <path>', 'Path to configuration file')
  .option('-p, --port <port>', 'Port to listen on', '7890')
  .option('-d, --daemon', 'Run as daemon')
  .option('--ui', 'Start with web UI')
  .action(async (options) => {
    await startCommand.execute(options);
  });

// Mesh command
program
  .command('mesh <action>')
  .description('Manage mesh topology and connections')
  .option('-t, --topology <type>', 'Mesh topology (star, ring, mesh, hierarchical)')
  .option('--peers <peers>', 'Comma-separated list of peer addresses')
  .action(async (action, options) => {
    await meshCommand.execute(action, options);
  });

// Neural command
program
  .command('neural <action>')
  .description('Manage neural network components')
  .option('-m, --model <model>', 'Neural model to use')
  .option('--train', 'Enable training mode')
  .option('--inference', 'Enable inference mode')
  .action(async (action, options) => {
    await neuralCommand.execute(action, options);
  });

// DAG command
program
  .command('dag <action>')
  .description('Manage DAG operations and consensus')
  .option('--validate', 'Validate DAG integrity')
  .option('--sync', 'Synchronize with peers')
  .action(async (action, options) => {
    await dagCommand.execute(action, options);
  });

// Peer command
program
  .command('peer <action>')
  .description('Manage peer connections and discovery')
  .option('--add <address>', 'Add a peer')
  .option('--remove <id>', 'Remove a peer')
  .option('--list', 'List all peers')
  .action(async (action, options) => {
    await peerCommand.execute(action, options);
  });

// Status command
program
  .command('status')
  .description('Display node and network status')
  .option('-j, --json', 'Output as JSON')
  .option('--detailed', 'Show detailed information')
  .action(async (options) => {
    await statusCommand.execute(options);
  });

// Error handling
program.exitOverride();

try {
  program.parse(process.argv);
} catch (error) {
  if (error.code === 'commander.unknownCommand') {
    console.error(chalk.red('\nUnknown command:', error.message));
    console.log(chalk.yellow('\nRun "synaptic --help" to see available commands'));
  } else if (error.code === 'commander.help') {
    // Help was requested, exit gracefully
    process.exit(0);
  } else {
    console.error(chalk.red('\nError:'), error.message);
  }
  process.exit(1);
}

// Show help if no command provided
if (!process.argv.slice(2).length) {
  displayBanner();
  program.outputHelp();
}

module.exports = { program };