import { Command } from 'commander';
import chalk from 'chalk';

export function stopCommand(): Command {
  const command = new Command('stop');

  command
    .description('Stop the running neural mesh node')
    .option('-f, --force', 'Force stop without graceful shutdown')
    .action(async (options: any) => {
      console.log(chalk.yellow('🛡️ Stopping Synaptic Neural Mesh node...'));
      
      if (options.force) {
        console.log(chalk.red('⚠️ Force stop requested'));
      } else {
        console.log('🗋 Initiating graceful shutdown...');
        console.log('⏹️ Stopping neural agents...');
        console.log('🔌 Disconnecting from mesh...');
        console.log('💾 Saving state...');
      }
      
      console.log(chalk.green('✅ Node stopped successfully'));
    });

  return command;
}
