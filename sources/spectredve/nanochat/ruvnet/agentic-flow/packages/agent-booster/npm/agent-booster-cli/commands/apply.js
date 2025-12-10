/**
 * Apply command - optimize a single prompt
 */

const fs = require('fs').promises;
const chalk = require('chalk');
const ora = require('ora');

async function applyCommand(prompt, options) {
  const spinner = ora('Initializing Agent Booster...').start();

  try {
    // Load Agent Booster
    const agentBooster = require('agent-booster');

    // Show runtime info
    const runtime = agentBooster.getRuntime();
    if (runtime) {
      spinner.text = `Using ${runtime} runtime...`;
    }

    // Show analysis if requested
    if (options.analyze) {
      spinner.text = 'Analyzing prompt...';
      const analysis = await agentBooster.analyze(prompt);

      spinner.stop();
      console.log(chalk.bold('\n📊 Analysis:\n'));
      console.log(chalk.cyan('Original Tokens:'), analysis.originalTokens);
      console.log(chalk.green('Estimated Optimized:'), analysis.optimizedTokens);
      console.log(chalk.yellow('Savings:'), `${analysis.savings} tokens (${analysis.savingsPercent.toFixed(1)}%)`);
      console.log();

      spinner.start('Optimizing prompt...');
    } else {
      spinner.text = 'Optimizing prompt...';
    }

    // Build optimization options
    const optimizationOptions = {
      strategy: options.strategy || 'balanced'
    };

    if (options.maxTokens) {
      optimizationOptions.maxTokens = options.maxTokens;
    }

    if (options.targetModel) {
      optimizationOptions.targetModel = options.targetModel;
    }

    if (options.preserve) {
      optimizationOptions.preserve = options.preserve;
    }

    // Apply optimization
    const optimized = await agentBooster.apply(prompt, optimizationOptions);

    spinner.succeed('Optimization complete!');

    // Output results
    if (options.json) {
      const result = {
        original: prompt,
        optimized: optimized,
        options: optimizationOptions,
        runtime: agentBooster.getRuntime()
      };
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log(chalk.bold('\n✨ Optimized Prompt:\n'));
      console.log(chalk.gray('━'.repeat(50)));
      console.log(optimized);
      console.log(chalk.gray('━'.repeat(50)));
      console.log();
    }

    // Write to file if requested
    if (options.output) {
      await fs.writeFile(options.output, optimized, 'utf8');
      console.log(chalk.green(`✓ Output written to ${options.output}`));
    }

  } catch (error) {
    spinner.fail('Optimization failed');
    console.error(chalk.red('\nError:'), error.message);

    // Show helpful error messages
    if (error.message.includes('No runtime available')) {
      console.log(chalk.yellow('\n💡 Tip:'), 'Install WASM package: npm install @agent-booster/wasm');
    }

    process.exit(1);
  }
}

module.exports = applyCommand;
