/**
 * Batch command - optimize multiple prompts from a file
 */

const fs = require('fs').promises;
const path = require('path');
const chalk = require('chalk');
const ora = require('ora');

/**
 * Reads prompts from a file (text or JSONL format)
 */
async function readPrompts(filePath, isJson) {
  const content = await fs.readFile(filePath, 'utf8');

  if (isJson) {
    // JSONL format - one JSON object per line
    return content
      .split('\n')
      .filter(line => line.trim())
      .map((line, index) => {
        try {
          const obj = JSON.parse(line);
          return obj.prompt || obj.text || obj.content || line;
        } catch (error) {
          throw new Error(`Invalid JSON on line ${index + 1}: ${error.message}`);
        }
      });
  } else {
    // Plain text - one prompt per line
    return content.split('\n').filter(line => line.trim());
  }
}

/**
 * Writes optimized prompts to a file
 */
async function writePrompts(filePath, prompts, originals, isJson) {
  if (isJson) {
    // JSONL format
    const lines = prompts.map((prompt, index) => {
      return JSON.stringify({
        original: originals[index],
        optimized: prompt
      });
    });
    await fs.writeFile(filePath, lines.join('\n') + '\n', 'utf8');
  } else {
    // Plain text
    await fs.writeFile(filePath, prompts.join('\n') + '\n', 'utf8');
  }
}

/**
 * Process prompts in parallel batches
 */
async function processBatches(agentBooster, prompts, options, spinner) {
  const results = [];
  const batchSize = options.parallel || 4;

  for (let i = 0; i < prompts.length; i += batchSize) {
    const batch = prompts.slice(i, i + batchSize);
    const batchNum = Math.floor(i / batchSize) + 1;
    const totalBatches = Math.ceil(prompts.length / batchSize);

    if (options.progress) {
      spinner.text = `Processing batch ${batchNum}/${totalBatches} (${i + 1}-${Math.min(i + batchSize, prompts.length)}/${prompts.length} prompts)`;
    }

    try {
      // Use batchApply if available for better performance
      const batchResults = await agentBooster.batchApply(batch, {
        strategy: options.strategy || 'balanced',
        maxTokens: options.maxTokens,
        targetModel: options.targetModel
      });

      results.push(...batchResults);
    } catch (error) {
      // Fall back to sequential processing on error
      for (const prompt of batch) {
        const result = await agentBooster.apply(prompt, {
          strategy: options.strategy || 'balanced',
          maxTokens: options.maxTokens,
          targetModel: options.targetModel
        });
        results.push(result);
      }
    }
  }

  return results;
}

async function batchCommand(inputFile, options) {
  const spinner = options.progress ? ora('Reading input file...').start() : null;

  try {
    // Verify input file exists
    try {
      await fs.access(inputFile);
    } catch (error) {
      throw new Error(`Input file not found: ${inputFile}`);
    }

    if (spinner) {
      spinner.text = 'Reading prompts...';
    }

    // Read prompts
    const prompts = await readPrompts(inputFile, options.json);

    if (prompts.length === 0) {
      throw new Error('No prompts found in input file');
    }

    if (spinner) {
      spinner.text = `Loaded ${prompts.length} prompt(s)`;
      spinner.succeed();
      spinner.start('Initializing Agent Booster...');
    } else {
      console.log(chalk.cyan(`📄 Loaded ${prompts.length} prompt(s) from ${inputFile}`));
    }

    // Load Agent Booster
    const agentBooster = require('agent-booster');
    const runtime = agentBooster.getRuntime();

    if (spinner) {
      spinner.succeed(`Using ${runtime || 'auto-detected'} runtime`);
      spinner.start('Optimizing prompts...');
    } else {
      console.log(chalk.green(`✓ Using ${runtime || 'auto-detected'} runtime`));
      console.log(chalk.cyan('⚡ Optimizing prompts...'));
    }

    const startTime = Date.now();

    // Process prompts
    const optimized = await processBatches(agentBooster, prompts, options, spinner);

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);

    if (spinner) {
      spinner.succeed(`Optimized ${optimized.length} prompts in ${duration}s`);
    } else {
      console.log(chalk.green(`✓ Optimized ${optimized.length} prompts in ${duration}s`));
    }

    // Output results
    if (options.output) {
      await writePrompts(options.output, optimized, prompts, options.json);
      console.log(chalk.green(`✓ Results written to ${options.output}`));
    } else {
      // Output to stdout
      if (options.json) {
        const output = optimized.map((opt, idx) => ({
          original: prompts[idx],
          optimized: opt
        }));
        console.log(JSON.stringify(output, null, 2));
      } else {
        console.log(chalk.bold('\n✨ Optimized Prompts:\n'));
        console.log(chalk.gray('━'.repeat(50)));
        optimized.forEach((prompt, idx) => {
          console.log(chalk.cyan(`\n[${idx + 1}/${optimized.length}]`));
          console.log(prompt);
        });
        console.log(chalk.gray('\n' + '━'.repeat(50)));
      }
    }

    // Show statistics
    if (!options.json) {
      console.log(chalk.bold('\n📊 Statistics:\n'));

      const originalTokens = prompts.reduce((sum, p) => sum + p.length / 4, 0); // Rough estimate
      const optimizedTokens = optimized.reduce((sum, p) => sum + p.length / 4, 0);
      const savings = originalTokens - optimizedTokens;
      const savingsPercent = (savings / originalTokens * 100).toFixed(1);

      console.log(chalk.cyan('Total Prompts:'), prompts.length);
      console.log(chalk.cyan('Processing Time:'), `${duration}s`);
      console.log(chalk.cyan('Avg Time per Prompt:'), `${(duration / prompts.length).toFixed(3)}s`);
      console.log(chalk.yellow('Estimated Token Savings:'), `~${Math.round(savings)} tokens (${savingsPercent}%)`);
      console.log();
    }

  } catch (error) {
    if (spinner) {
      spinner.fail('Batch optimization failed');
    }

    console.error(chalk.red('\nError:'), error.message);

    // Show helpful error messages
    if (error.message.includes('No runtime available')) {
      console.log(chalk.yellow('\n💡 Tip:'), 'Install WASM package: npm install @agent-booster/wasm');
    }

    process.exit(1);
  }
}

module.exports = batchCommand;
