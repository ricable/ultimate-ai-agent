#!/usr/bin/env node
"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const figlet_1 = __importDefault(require("figlet"));
const gradient_string_1 = __importDefault(require("gradient-string"));
const init_js_1 = require("../commands/init.js");
const start_js_1 = require("../commands/start.js");
const mesh_js_1 = require("../commands/mesh.js");
const neural_js_1 = require("../commands/neural.js");
const dag_js_1 = require("../commands/dag.js");
const peer_js_1 = require("../commands/peer.js");
const status_js_1 = require("../commands/status.js");
const config_js_1 = require("../commands/config.js");
const version_js_1 = require("../utils/version.js");
// Display banner
function displayBanner() {
    const banner = figlet_1.default.textSync('Synaptic', {
        font: 'ANSI Shadow',
        horizontalLayout: 'default',
        verticalLayout: 'default'
    });
    console.log(gradient_string_1.default.rainbow(banner));
    console.log(chalk_1.default.cyan('Neural Mesh CLI - Revolutionary AI Orchestration\n'));
}
// Create main program
const program = new commander_1.Command();
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
    .version(version_js_1.version, '-v, --version', 'Display version')
    .option('-d, --debug', 'Enable debug mode')
    .option('-q, --quiet', 'Suppress non-essential output')
    .option('--no-color', 'Disable colored output');
// Add commands
program.addCommand((0, init_js_1.initCommand)());
program.addCommand((0, start_js_1.startCommand)());
program.addCommand((0, mesh_js_1.meshCommand)());
program.addCommand((0, neural_js_1.neuralCommand)());
program.addCommand((0, dag_js_1.dagCommand)());
program.addCommand((0, peer_js_1.peerCommand)());
program.addCommand((0, status_js_1.statusCommand)());
program.addCommand((0, config_js_1.configCommand)());
// MCP server command
program
    .command('mcp')
    .description('Start MCP server for Model Context Protocol integration')
    .option('-p, --port <port>', 'Port to run MCP server on', '3000')
    .option('-h, --host <host>', 'Host to bind to', 'localhost')
    .option('--stdio', 'Use stdio transport instead of HTTP')
    .action(async (options) => {
    const { startMCPServer } = await Promise.resolve().then(() => __importStar(require('../commands/mcp.js')));
    await startMCPServer(options);
});
// Swarm command
program
    .command('swarm')
    .description('Manage neural swarm coordination')
    .option('-t, --topology <type>', 'Swarm topology (mesh, hierarchical, ring, star)', 'mesh')
    .option('-a, --agents <count>', 'Number of agents', '5')
    .action(async (options) => {
    const { swarmCommand } = await Promise.resolve().then(() => __importStar(require('../commands/swarm.js')));
    await swarmCommand(options);
});
// Export command
program
    .command('export')
    .description('Export mesh configuration or data')
    .option('-f, --format <format>', 'Export format (json, yaml, toml)', 'json')
    .option('-o, --output <file>', 'Output file path')
    .action(async (options) => {
    const { exportCommand } = await Promise.resolve().then(() => __importStar(require('../commands/export.js')));
    await exportCommand(options);
});
// Import command
program
    .command('import')
    .description('Import mesh configuration or data')
    .argument('<file>', 'File to import')
    .option('-f, --format <format>', 'Import format (json, yaml, toml)', 'json')
    .action(async (file, options) => {
    const { importCommand } = await Promise.resolve().then(() => __importStar(require('../commands/import.js')));
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
    }
    catch (error) {
        console.error(chalk_1.default.red('Error:'), error.message);
        if (program.opts().debug) {
            console.error(error.stack);
        }
        process.exit(1);
    }
}
// Handle uncaught errors
process.on('uncaughtException', (error) => {
    console.error(chalk_1.default.red('Uncaught Exception:'), error.message);
    if (program.opts().debug) {
        console.error(error.stack);
    }
    process.exit(1);
});
process.on('unhandledRejection', (reason, promise) => {
    console.error(chalk_1.default.red('Unhandled Rejection at:'), promise, chalk_1.default.red('reason:'), reason);
    process.exit(1);
});
// Run main
main();
//# sourceMappingURL=main.js.map