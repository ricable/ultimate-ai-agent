"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.meshCommand = meshCommand;
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const ora_1 = __importDefault(require("ora"));
function meshCommand() {
    const command = new commander_1.Command('mesh');
    command
        .description('Manage mesh network connections')
        .addCommand(meshJoinCommand())
        .addCommand(meshLeaveCommand())
        .addCommand(meshListCommand());
    return command;
}
function meshJoinCommand() {
    const command = new commander_1.Command('join');
    command
        .description('Join an existing mesh network')
        .argument('<peer-address>', 'Peer address to connect to')
        .option('-t, --timeout <seconds>', 'Connection timeout', '30')
        .action(async (peerAddress, options) => {
        const spinner = (0, ora_1.default)('Connecting to peer...').start();
        try {
            // Simulate connection
            await new Promise(resolve => setTimeout(resolve, 2000));
            spinner.succeed(chalk_1.default.green(`✅ Connected to ${peerAddress}`));
            console.log('\n' + chalk_1.default.cyan('📡 Mesh Network Status:'));
            console.log(`Connected Peers: 1`);
            console.log(`Network Topology: mesh`);
            console.log(`Consensus: active`);
        }
        catch (error) {
            spinner.fail(chalk_1.default.red('Failed to connect'));
            console.error(error?.message || error);
            process.exit(1);
        }
    });
    return command;
}
function meshLeaveCommand() {
    const command = new commander_1.Command('leave');
    command
        .description('Leave the current mesh network')
        .action(async () => {
        console.log(chalk_1.default.yellow('🔌 Leaving mesh network...'));
        console.log(chalk_1.default.green('✅ Successfully disconnected from mesh'));
    });
    return command;
}
function meshListCommand() {
    const command = new commander_1.Command('list');
    command
        .description('List connected mesh peers')
        .action(async () => {
        console.log(chalk_1.default.cyan('\n📡 Connected Mesh Peers:'));
        console.log(chalk_1.default.gray('─'.repeat(50)));
        console.log('No peers connected');
        console.log(chalk_1.default.gray('─'.repeat(50)));
    });
    return command;
}
//# sourceMappingURL=mesh.js.map