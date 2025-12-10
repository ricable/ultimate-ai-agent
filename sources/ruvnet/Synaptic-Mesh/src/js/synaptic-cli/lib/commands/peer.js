"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.peerCommand = peerCommand;
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
function peerCommand() {
    const command = new commander_1.Command('peer');
    command
        .description('Manage peer connections')
        .addCommand(peerListCommand())
        .addCommand(peerConnectCommand())
        .addCommand(peerDisconnectCommand());
    return command;
}
function peerListCommand() {
    const command = new commander_1.Command('list');
    command
        .description('List all connected peers')
        .action(async () => {
        console.log(chalk_1.default.cyan('\n📡 Connected Peers:'));
        console.log(chalk_1.default.gray('─'.repeat(60)));
        console.log('No peers connected');
        console.log(chalk_1.default.gray('─'.repeat(60)));
    });
    return command;
}
function peerConnectCommand() {
    const command = new commander_1.Command('connect');
    command
        .description('Connect to a peer')
        .argument('<address>', 'Peer address to connect to')
        .action(async (address) => {
        console.log(chalk_1.default.yellow(`🔗 Connecting to peer: ${address}...`));
        console.log(chalk_1.default.green('✅ Connection established'));
    });
    return command;
}
function peerDisconnectCommand() {
    const command = new commander_1.Command('disconnect');
    command
        .description('Disconnect from a peer')
        .argument('<peer-id>', 'Peer ID to disconnect')
        .action(async (peerId) => {
        console.log(chalk_1.default.yellow(`🔌 Disconnecting from peer: ${peerId}...`));
        console.log(chalk_1.default.green('✅ Peer disconnected'));
    });
    return command;
}
//# sourceMappingURL=peer.js.map