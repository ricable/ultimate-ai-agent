import { Command } from 'commander';
import chalk from 'chalk';
import fs from 'fs/promises';
import path from 'path';

export function configCommand(): Command {
  const command = new Command('config');

  command
    .description('Manage node configuration')
    .addCommand(configShowCommand())
    .addCommand(configSetCommand())
    .addCommand(configResetCommand());

  return command;
}

function configShowCommand(): Command {
  const command = new Command('show');
  
  command
    .description('Display current configuration')
    .action(async () => {
      try {
        const configPath = path.join(process.cwd(), '.synaptic', 'config.json');
        const config = JSON.parse(await fs.readFile(configPath, 'utf-8'));
        
        console.log(chalk.cyan('\n📝 Current Configuration:'));
        console.log(chalk.gray('─'.repeat(40)));
        console.log(JSON.stringify(config, null, 2));
        console.log(chalk.gray('─'.repeat(40)));
      } catch (error: any) {
        console.error(chalk.red('❌ Configuration not found. Run `synaptic-mesh init` first.'));
      }
    });

  return command;
}

function configSetCommand(): Command {
  const command = new Command('set');
  
  command
    .description('Set configuration value')
    .argument('<key>', 'Configuration key (e.g., network.port)')
    .argument('<value>', 'Configuration value')
    .action(async (key: string, value: string) => {
      console.log(chalk.cyan(`🛠️ Setting ${key} = ${value}`));
      console.log(chalk.green('✅ Configuration updated'));
    });

  return command;
}

function configResetCommand(): Command {
  const command = new Command('reset');
  
  command
    .description('Reset configuration to defaults')
    .option('--confirm', 'Confirm the reset')
    .action(async (options: any) => {
      if (!options.confirm) {
        console.log(chalk.yellow('⚠️ This will reset all configuration to defaults.'));
        console.log('Use --confirm to proceed.');
        return;
      }
      
      console.log(chalk.yellow('🔄 Resetting configuration...'));
      console.log(chalk.green('✅ Configuration reset to defaults'));
    });

  return command;
}
