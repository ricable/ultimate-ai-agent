#!/usr/bin/env node

/**
 * Agent Booster CLI
 * Command-line interface for prompt optimization
 */

const { program } = require('commander');
const chalk = require('chalk');
const packageJson = require('../package.json');

// Import commands
const applyCommand = require('../commands/apply');
const batchCommand = require('../commands/batch');

// Configure CLI
program
  .name('agent-booster')
  .description('High-performance prompt optimization tool')
  .version(packageJson.version);

// Apply command - optimize a single prompt
program
  .command('apply')
  .description('Optimize a single prompt')
  .argument('<prompt>', 'The prompt to optimize')
  .option('-s, --strategy <type>', 'Optimization strategy (aggressive|balanced|conservative)', 'balanced')
  .option('-m, --max-tokens <number>', 'Maximum tokens in output', parseInt)
  .option('-t, --target-model <model>', 'Target model for optimization')
  .option('-p, --preserve <keywords...>', 'Keywords to preserve')
  .option('-a, --analyze', 'Show analysis before optimization')
  .option('-o, --output <file>', 'Write output to file')
  .option('--json', 'Output in JSON format')
  .action(applyCommand);

// Batch command - optimize multiple prompts
program
  .command('batch')
  .description('Optimize multiple prompts from a file')
  .argument('<input>', 'Input file with prompts (one per line or JSONL)')
  .option('-s, --strategy <type>', 'Optimization strategy (aggressive|balanced|conservative)', 'balanced')
  .option('-m, --max-tokens <number>', 'Maximum tokens in output', parseInt)
  .option('-t, --target-model <model>', 'Target model for optimization')
  .option('-o, --output <file>', 'Output file (default: stdout)')
  .option('--json', 'Input/output in JSONL format')
  .option('--parallel <number>', 'Number of parallel operations', parseInt, 4)
  .option('--progress', 'Show progress bar')
  .action(batchCommand);

// Analyze command - analyze without optimizing
program
  .command('analyze')
  .description('Analyze a prompt without optimizing')
  .argument('<prompt>', 'The prompt to analyze')
  .option('--json', 'Output in JSON format')
  .action(async (prompt, options) => {
    try {
      const agentBooster = require('agent-booster');
      const analysis = await agentBooster.analyze(prompt);

      if (options.json) {
        console.log(JSON.stringify(analysis, null, 2));
      } else {
        console.log(chalk.bold('\n📊 Prompt Analysis\n'));
        console.log(chalk.gray('━'.repeat(50)));
        console.log(chalk.cyan('Original Tokens:'), analysis.originalTokens);
        console.log(chalk.green('Estimated Optimized:'), analysis.optimizedTokens);
        console.log(chalk.yellow('Potential Savings:'), `${analysis.savings} tokens (${analysis.savingsPercent.toFixed(1)}%)`);
        console.log(chalk.blue('Complexity Score:'), `${analysis.complexity}/100`);

        if (analysis.patterns && analysis.patterns.length > 0) {
          console.log(chalk.bold('\n🔍 Detected Patterns:'));
          analysis.patterns.forEach(pattern => {
            console.log(chalk.gray('  •'), pattern);
          });
        }

        if (analysis.suggestions && analysis.suggestions.length > 0) {
          console.log(chalk.bold('\n💡 Suggestions:'));
          analysis.suggestions.forEach(suggestion => {
            console.log(chalk.gray('  •'), suggestion);
          });
        }
        console.log();
      }
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
      process.exit(1);
    }
  });

// Version info command
program
  .command('version')
  .description('Show version and runtime information')
  .action(() => {
    try {
      const agentBooster = require('agent-booster');
      const info = agentBooster.getVersion();

      console.log(chalk.bold('\n🚀 Agent Booster\n'));
      console.log(chalk.gray('━'.repeat(50)));
      console.log(chalk.cyan('CLI Version:'), packageJson.version);
      console.log(chalk.cyan('SDK Version:'), info.version);
      console.log(chalk.cyan('Core Version:'), info.coreVersion);
      console.log(chalk.green('Runtime:'), info.runtime || 'not loaded');
      console.log();
    } catch (error) {
      console.error(chalk.red('Error:'), error.message);
      process.exit(1);
    }
  });

// Parse arguments
program.parse();

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}
