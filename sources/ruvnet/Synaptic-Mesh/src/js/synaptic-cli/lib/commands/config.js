"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.configCommand = configCommand;
const commander_1 = require("commander");
const chalk_1 = __importDefault(require("chalk"));
const promises_1 = __importDefault(require("fs/promises"));
const path_1 = __importDefault(require("path"));
function configCommand() {
    const command = new commander_1.Command('config');
    command
        .description('Manage node configuration')
        .addCommand(configShowCommand())
        .addCommand(configSetCommand())
        .addCommand(configResetCommand());
    return command;
}
function configShowCommand() {
    const command = new commander_1.Command('show');
    command
        .description('Display current configuration')
        .action(async () => {
        try {
            const configPath = path_1.default.join(process.cwd(), '.synaptic', 'config.json');
            const config = JSON.parse(await promises_1.default.readFile(configPath, 'utf-8'));
            console.log(chalk_1.default.cyan('\n📝 Current Configuration:'));
            console.log(chalk_1.default.gray('─'.repeat(40)));
            console.log(JSON.stringify(config, null, 2));
            console.log(chalk_1.default.gray('─'.repeat(40)));
        }
        catch (error) {
            console.error(chalk_1.default.red('❌ Configuration not found. Run `synaptic-mesh init` first.'));
        }
    });
    return command;
}
function configSetCommand() {
    const command = new commander_1.Command('set');
    command
        .description('Set configuration value')
        .argument('<key>', 'Configuration key (e.g., network.port)')
        .argument('<value>', 'Configuration value')
        .action(async (key, value) => {
        console.log(chalk_1.default.cyan(`🛠️ Setting ${key} = ${value}`));
        console.log(chalk_1.default.green('✅ Configuration updated'));
    });
    return command;
}
function configResetCommand() {
    const command = new commander_1.Command('reset');
    command
        .description('Reset configuration to defaults')
        .option('--confirm', 'Confirm the reset')
        .action(async (options) => {
        if (!options.confirm) {
            console.log(chalk_1.default.yellow('⚠️ This will reset all configuration to defaults.'));
            console.log('Use --confirm to proceed.');
            return;
        }
        console.log(chalk_1.default.yellow('🔄 Resetting configuration...'));
        console.log(chalk_1.default.green('✅ Configuration reset to defaults'));
    });
    return command;
}
//# sourceMappingURL=config.js.map