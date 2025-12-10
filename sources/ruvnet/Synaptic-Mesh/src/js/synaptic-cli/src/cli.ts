#!/usr/bin/env node

import { Command } from 'commander';
import chalk from 'chalk';
import { version } from '../package.json';
import { initCommand } from './commands/init';
import { startCommand } from './commands/start';
import { meshCommand } from './commands/mesh';
import { neuralCommand } from './commands/neural';
import { dagCommand } from './commands/dag';
import { peerCommand } from './commands/peer';
import { statusCommand } from './commands/status';
import { stopCommand } from './commands/stop';
import { configCommand } from './commands/config';
import { kimiCommand } from './commands/kimi';

// ASCII Art Logo
const logo = chalk.cyan(`
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   ███████╗██╗   ██╗███╗   ██╗ █████╗ ██████╗ ████████╗██╗   ║
║   ██╔════╝╚██╗ ██╔╝████╗  ██║██╔══██╗██╔══██╗╚══██╔══╝██║   ║
║   ███████╗ ╚████╔╝ ██╔██╗ ██║███████║██████╔╝   ██║   ██║   ║
║   ╚════██║  ╚██╔╝  ██║╚██╗██║██╔══██║██╔═══╝    ██║   ██║   ║
║   ███████║   ██║   ██║ ╚████║██║  ██║██║        ██║   ██║   ║
║   ╚══════╝   ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═╝   ║
║                                                               ║
║              🧠 Neural Mesh - Distributed Intelligence 🧠      ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
`);

// Create main program
const program = new Command();

// Configure program
program
  .name('synaptic-mesh')
  .description('Self-evolving distributed neural fabric with quantum-resistant DAG networking')
  .version(version, '-v, --version', 'Display version information')
  .helpOption('-h, --help', 'Display help information')
  .addHelpCommand('help [command]', 'Display help for command')
  .showHelpAfterError('(add --help for additional information)')
  .configureOutput({
    writeOut: (str) => process.stdout.write(str),
    writeErr: (str) => process.stderr.write(chalk.red(str))
  });

// Global options
program
  .option('-d, --debug', 'Enable debug mode with verbose output')
  .option('-q, --quiet', 'Suppress non-essential output')
  .option('--no-color', 'Disable colored output')
  .option('--config <path>', 'Specify custom config file path', '.synaptic/config.json');

// Custom help
program.on('--help', () => {
  console.log('');
  console.log('Examples:');
  console.log('  $ npx synaptic-mesh init                    # Initialize a new neural mesh node');
  console.log('  $ synaptic-mesh start --port 8080          # Start mesh node on port 8080');
  console.log('  $ synaptic-mesh mesh join peer.address     # Join existing mesh network');
  console.log('  $ synaptic-mesh neural spawn --type mlp    # Spawn a neural agent');
  console.log('  $ synaptic-mesh kimi chat "Hello AI!"      # Chat with Kimi-K2 AI model');
  console.log('  $ synaptic-mesh status                     # Check mesh status');
  console.log('');
  console.log('Documentation:');
  console.log('  https://github.com/ruvnet/Synaptic-Neural-Mesh');
});

// Show logo on help
program.hook('preAction', (thisCommand) => {
  if (thisCommand.args.length === 0 || thisCommand.args.includes('help')) {
    console.log(logo);
  }
});

// Register commands
program.addCommand(initCommand());
program.addCommand(startCommand());
program.addCommand(meshCommand());
program.addCommand(neuralCommand());
program.addCommand(dagCommand());
program.addCommand(peerCommand());
program.addCommand(statusCommand());
program.addCommand(stopCommand());
program.addCommand(configCommand());
program.addCommand(kimiCommand());

// Error handling
program.exitOverride((err) => {
  if (err.code === 'commander.version') {
    console.log(chalk.cyan(`Synaptic Neural Mesh v${version}`));
    process.exit(0);
  }
  if (err.code === 'commander.help') {
    process.exit(0);
  }
  console.error(chalk.red('Error:'), err.message);
  process.exit(1);
});

// Parse arguments
async function main() {
  try {
    await program.parseAsync(process.argv);
    
    // If no command specified, show help
    if (process.argv.length === 2) {
      console.log(logo);
      program.outputHelp();
    }
  } catch (error: any) {
    console.error(chalk.red('Fatal error:'), error?.message || error);
    if (program.opts().debug) {
      console.error(error?.stack);
    }
    process.exit(1);
  }
}

// Run CLI
main();