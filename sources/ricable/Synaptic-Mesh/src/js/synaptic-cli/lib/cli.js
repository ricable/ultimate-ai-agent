#!/usr/bin/env node
"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const package_json_1 = require("../package.json");
const init_1 = require("./commands/init");
const start_1 = require("./commands/start");
const mesh_1 = require("./commands/mesh");
const neural_1 = require("./commands/neural");
const dag_1 = require("./commands/dag");
const peer_1 = require("./commands/peer");
const status_1 = require("./commands/status");
const stop_1 = require("./commands/stop");
const config_1 = require("./commands/config");
const kimi_1 = require("./commands/kimi");
// ASCII Art Logo
const logo = chalk_1.default.cyan(`
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
const program = new commander_1.Command();
// Configure program
program
    .name('synaptic-mesh')
    .description('Self-evolving distributed neural fabric with quantum-resistant DAG networking')
    .version(package_json_1.version, '-v, --version', 'Display version information')
    .helpOption('-h, --help', 'Display help information')
    .addHelpCommand('help [command]', 'Display help for command')
    .showHelpAfterError('(add --help for additional information)')
    .configureOutput({
    writeOut: (str) => process.stdout.write(str),
    writeErr: (str) => process.stderr.write(chalk_1.default.red(str))
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
program.addCommand((0, init_1.initCommand)());
program.addCommand((0, start_1.startCommand)());
program.addCommand((0, mesh_1.meshCommand)());
program.addCommand((0, neural_1.neuralCommand)());
program.addCommand((0, dag_1.dagCommand)());
program.addCommand((0, peer_1.peerCommand)());
program.addCommand((0, status_1.statusCommand)());
program.addCommand((0, stop_1.stopCommand)());
program.addCommand((0, config_1.configCommand)());
program.addCommand((0, kimi_1.kimiCommand)());
// Error handling
program.exitOverride((err) => {
    if (err.code === 'commander.version') {
        console.log(chalk_1.default.cyan(`Synaptic Neural Mesh v${package_json_1.version}`));
        process.exit(0);
    }
    if (err.code === 'commander.help') {
        process.exit(0);
    }
    console.error(chalk_1.default.red('Error:'), err.message);
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
    }
    catch (error) {
        console.error(chalk_1.default.red('Fatal error:'), error?.message || error);
        if (program.opts().debug) {
            console.error(error?.stack);
        }
        process.exit(1);
    }
}
// Run CLI
main();
//# sourceMappingURL=cli.js.map