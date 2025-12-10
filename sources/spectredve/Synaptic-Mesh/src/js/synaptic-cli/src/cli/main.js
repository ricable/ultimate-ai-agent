#!/usr/bin/env node

import { Command } from 'commander';
import chalk from 'chalk';
import figlet from 'figlet';
import gradient from 'gradient-string';
import { initCommand } from '../commands/init.js';
import { startCommand } from '../commands/start.js';
import { meshCommand } from '../commands/mesh.js';
import { neuralCommand } from '../commands/neural.js';
import { dagCommand } from '../commands/dag.js';
import { peerCommand } from '../commands/peer.js';
import { statusCommand } from '../commands/status.js';
import { configCommand } from '../commands/config.js';
import { version } from '../utils/version.js';

// Display banner
function displayBanner() {
  const banner = figlet.textSync('Synaptic', {
    font: 'ANSI Shadow',
    horizontalLayout: 'default',
    verticalLayout: 'default'
  });
  
  console.log(gradient.rainbow(banner));
  console.log(chalk.cyan('Neural Mesh CLI - Revolutionary AI Orchestration\n'));
}

// Create main program
const program = new Command();

// Display banner on help
program.configureHelp({
  helpWidth: 100,
  sortSubcommands: true,
  subcommandTerm: (cmd) => cmd.name() + ' ' + cmd.alias()
});

program.hook('preAction', () => {
  if (process.argv.includes('--help') || process.argv.includes('-h')) {
    displayBanner();
  }
});

// Configure program
program
  .name('synaptic')
  .description('Synaptic Neural Mesh CLI - Revolutionary AI orchestration with neural mesh topology')
  .version(version, '-v, --version', 'Display version')
  .option('-d, --debug', 'Enable debug mode')
  .option('-q, --quiet', 'Suppress non-essential output')
  .option('--no-color', 'Disable colored output');

// Add commands
program.addCommand(initCommand());
program.addCommand(startCommand());
program.addCommand(meshCommand());
program.addCommand(neuralCommand());
program.addCommand(dagCommand());
program.addCommand(peerCommand());
program.addCommand(statusCommand());
program.addCommand(configCommand());

// MCP server command
program
  .command('mcp')
  .description('Start MCP server for Model Context Protocol integration')
  .option('-p, --port <port>', 'Port to run MCP server on', '3000')
  .option('-h, --host <host>', 'Host to bind to', 'localhost')
  .option('--stdio', 'Use stdio transport instead of HTTP')
  .action(async (options) => {
    const { startMCPServer } = await import('../commands/mcp.js');
    await startMCPServer(options);
  });

// Swarm command
program
  .command('swarm')
  .description('Manage neural swarm coordination')
  .option('-t, --topology <type>', 'Swarm topology (mesh, hierarchical, ring, star)', 'mesh')
  .option('-a, --agents <count>', 'Number of agents', '5')
  .action(async (options) => {
    const { swarmCommand } = await import('../commands/swarm.js');
    await swarmCommand(options);
  });

// Export command
program
  .command('export')
  .description('Export mesh configuration or data')
  .option('-f, --format <format>', 'Export format (json, yaml, toml)', 'json')
  .option('-o, --output <file>', 'Output file path')
  .action(async (options) => {
    const { exportCommand } = await import('../commands/export.js');
    await exportCommand(options);
  });

// Import command
program
  .command('import')
  .description('Import mesh configuration or data')
  .argument('<file>', 'File to import')
  .option('-f, --format <format>', 'Import format (json, yaml, toml)', 'json')
  .action(async (file, options) => {
    const { importCommand } = await import('../commands/import.js');
    await importCommand(file, options);
  });

// Parse arguments
async function main() {
  try {
    // Show banner for main command without args
    if (process.argv.length === 2) {
      displayBanner();
      program.outputHelp();
      process.exit(0);
    }

    await program.parseAsync(process.argv);
  } catch (error) {
    console.error(chalk.red('Error:'), error.message);
    if (program.opts().debug) {
      console.error(error.stack);
    }
    process.exit(1);
  }
}

// Handle uncaught errors
process.on('uncaughtException', (error) => {
  console.error(chalk.red('Uncaught Exception:'), error.message);
  if (program.opts().debug) {
    console.error(error.stack);
  }
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('Unhandled Rejection at:'), promise, chalk.red('reason:'), reason);
  process.exit(1);
});

// Run main
main();