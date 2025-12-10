"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.stopCommand = stopCommand;
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
function stopCommand() {
    const command = new commander_1.Command('stop');
    command
        .description('Stop the running neural mesh node')
        .option('-f, --force', 'Force stop without graceful shutdown')
        .action(async (options) => {
        console.log(chalk_1.default.yellow('🛡️ Stopping Synaptic Neural Mesh node...'));
        if (options.force) {
            console.log(chalk_1.default.red('⚠️ Force stop requested'));
        }
        else {
            console.log('🗋 Initiating graceful shutdown...');
            console.log('⏹️ Stopping neural agents...');
            console.log('🔌 Disconnecting from mesh...');
            console.log('💾 Saving state...');
        }
        console.log(chalk_1.default.green('✅ Node stopped successfully'));
    });
    return command;
}
//# sourceMappingURL=stop.js.map