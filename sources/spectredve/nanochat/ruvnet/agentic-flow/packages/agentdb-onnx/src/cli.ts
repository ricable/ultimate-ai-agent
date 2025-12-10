#!/usr/bin/env node
/**
 * CLI for AgentDB + ONNX
 */

import { Command } from 'commander';
import chalk from 'chalk';
import { createONNXAgentDB } from './index.js';
import { unlink } from 'fs/promises';

const program = new Command();

program
  .name('agentdb-onnx')
  .description('AgentDB with optimized ONNX embeddings')
  .version('1.0.0');

// Init command
program
  .command('init')
  .description('Initialize a new AgentDB with ONNX embeddings')
  .argument('<db-path>', 'Path to database file')
  .option('-m, --model <name>', 'Model name', 'Xenova/all-MiniLM-L6-v2')
  .option('--gpu', 'Enable GPU acceleration')
  .option('-b, --batch-size <size>', 'Batch size', '32')
  .option('-c, --cache-size <size>', 'Cache size', '10000')
  .action(async (dbPath, options) => {
    try {
      console.log(chalk.blue('Initializing AgentDB + ONNX...'));
      console.log(chalk.gray(`  Database: ${dbPath}`));
      console.log(chalk.gray(`  Model:    ${options.model}`));
      console.log(chalk.gray(`  GPU:      ${options.gpu ? 'enabled' : 'disabled'}`));

      const agentdb = await createONNXAgentDB({
        dbPath,
        modelName: options.model,
        useGPU: options.gpu,
        batchSize: parseInt(options.batchSize),
        cacheSize: parseInt(options.cacheSize)
      });

      console.log(chalk.green('✅ Initialized successfully'));

      await agentdb.close();
    } catch (error) {
      console.error(chalk.red('❌ Error:'), error);
      process.exit(1);
    }
  });

// Store pattern command
program
  .command('store-pattern')
  .description('Store a reasoning pattern')
  .argument('<db-path>', 'Database path')
  .requiredOption('-t, --task-type <type>', 'Task type')
  .requiredOption('-a, --approach <approach>', 'Approach description')
  .requiredOption('-s, --success-rate <rate>', 'Success rate (0-1)')
  .option('--tags <tags>', 'Comma-separated tags')
  .action(async (dbPath, options) => {
    try {
      const agentdb = await createONNXAgentDB({ dbPath, useGPU: false });

      const id = await agentdb.reasoningBank.storePattern({
        taskType: options.taskType,
        approach: options.approach,
        successRate: parseFloat(options.successRate),
        tags: options.tags?.split(',')
      });

      console.log(chalk.green(`✅ Pattern stored with ID: ${id}`));

      await agentdb.close();
    } catch (error) {
      console.error(chalk.red('❌ Error:'), error);
      process.exit(1);
    }
  });

// Search patterns command
program
  .command('search-patterns')
  .description('Search for similar patterns')
  .argument('<db-path>', 'Database path')
  .argument('<query>', 'Search query')
  .option('-k, --top-k <k>', 'Number of results', '10')
  .option('--threshold <threshold>', 'Similarity threshold', '0.7')
  .option('--task-type <type>', 'Filter by task type')
  .action(async (dbPath, query, options) => {
    try {
      const agentdb = await createONNXAgentDB({ dbPath, useGPU: false });

      const results = await agentdb.reasoningBank.searchPatterns({
        task: query,
        k: parseInt(options.topK),
        threshold: parseFloat(options.threshold),
        filters: options.taskType ? { taskType: options.taskType } : undefined
      });

      console.log(chalk.blue(`\nFound ${results.length} patterns:\n`));

      results.forEach((r, i) => {
        console.log(chalk.white(`${i + 1}. ${r.approach}`));
        console.log(chalk.gray(`   Type:       ${r.taskType}`));
        console.log(chalk.gray(`   Success:    ${(r.successRate * 100).toFixed(1)}%`));
        console.log(chalk.gray(`   Similarity: ${(r.similarity * 100).toFixed(1)}%`));
        if (r.tags && r.tags.length > 0) {
          console.log(chalk.gray(`   Tags:       ${r.tags.join(', ')}`));
        }
        console.log();
      });

      await agentdb.close();
    } catch (error) {
      console.error(chalk.red('❌ Error:'), error);
      process.exit(1);
    }
  });

// Store episode command
program
  .command('store-episode')
  .description('Store a reflexion episode')
  .argument('<db-path>', 'Database path')
  .requiredOption('-s, --session <id>', 'Session ID')
  .requiredOption('-t, --task <task>', 'Task description')
  .requiredOption('-r, --reward <reward>', 'Reward (0-1)')
  .requiredOption('--success', 'Task succeeded')
  .option('--critique <critique>', 'Self-critique')
  .action(async (dbPath, options) => {
    try {
      const agentdb = await createONNXAgentDB({ dbPath, useGPU: false });

      const id = await agentdb.reflexionMemory.storeEpisode({
        sessionId: options.session,
        task: options.task,
        reward: parseFloat(options.reward),
        success: true,
        critique: options.critique
      });

      console.log(chalk.green(`✅ Episode stored with ID: ${id}`));

      await agentdb.close();
    } catch (error) {
      console.error(chalk.red('❌ Error:'), error);
      process.exit(1);
    }
  });

// Search episodes command
program
  .command('search-episodes')
  .description('Search for similar episodes')
  .argument('<db-path>', 'Database path')
  .argument('<query>', 'Search query')
  .option('-k, --top-k <k>', 'Number of results', '10')
  .option('--only-successes', 'Only successful episodes')
  .option('--min-reward <reward>', 'Minimum reward threshold')
  .action(async (dbPath, query, options) => {
    try {
      const agentdb = await createONNXAgentDB({ dbPath, useGPU: false });

      const results = await agentdb.reflexionMemory.retrieveRelevant({
        task: query,
        k: parseInt(options.topK),
        onlySuccesses: options.onlySuccesses,
        minReward: options.minReward ? parseFloat(options.minReward) : undefined
      });

      console.log(chalk.blue(`\nFound ${results.length} episodes:\n`));

      results.forEach((r, i) => {
        console.log(chalk.white(`${i + 1}. ${r.task}`));
        console.log(chalk.gray(`   Session:    ${r.sessionId}`));
        console.log(chalk.gray(`   Reward:     ${(r.reward * 100).toFixed(1)}%`));
        console.log(chalk.gray(`   Success:    ${r.success ? 'Yes' : 'No'}`));
        console.log(chalk.gray(`   Similarity: ${(r.similarity * 100).toFixed(1)}%`));
        if (r.critique) {
          console.log(chalk.gray(`   Critique:   ${r.critique.substring(0, 80)}${r.critique.length > 80 ? '...' : ''}`));
        }
        console.log();
      });

      await agentdb.close();
    } catch (error) {
      console.error(chalk.red('❌ Error:'), error);
      process.exit(1);
    }
  });

// Stats command
program
  .command('stats')
  .description('Show database statistics')
  .argument('<db-path>', 'Database path')
  .action(async (dbPath) => {
    try {
      const agentdb = await createONNXAgentDB({ dbPath, useGPU: false });

      const stats = agentdb.getStats();

      console.log(chalk.blue('\n📊 AgentDB + ONNX Statistics\n'));

      console.log(chalk.white('Embeddings:'));
      console.log(chalk.gray(`  Model:          ${stats.embedder.model}`));
      console.log(chalk.gray(`  Total:          ${stats.embedder.totalEmbeddings}`));
      console.log(chalk.gray(`  Avg latency:    ${stats.embedder.avgLatency.toFixed(2)}ms`));
      console.log(chalk.gray(`  Cache hit rate: ${(stats.embedder.cache.hitRate * 100).toFixed(1)}%`));
      console.log(chalk.gray(`  Cache size:     ${stats.embedder.cache.size}/${stats.embedder.cache.maxSize}`));

      console.log(chalk.white('\nDatabase:'));
      if (stats.database) {
        Object.entries(stats.database).forEach(([key, value]) => {
          console.log(chalk.gray(`  ${key}: ${value}`));
        });
      }

      console.log();

      await agentdb.close();
    } catch (error) {
      console.error(chalk.red('❌ Error:'), error);
      process.exit(1);
    }
  });

// Benchmark command
program
  .command('benchmark')
  .description('Run performance benchmarks')
  .option('--operations <n>', 'Number of operations per test', '100')
  .action(async (options) => {
    console.log(chalk.blue('Running benchmarks...\n'));

    try {
      // Import and run benchmark dynamically
      await import('./benchmarks/benchmark-runner.js');
    } catch (error) {
      console.error(chalk.red('❌ Benchmark failed:'), error);
      process.exit(1);
    }
  });

program.parse();
