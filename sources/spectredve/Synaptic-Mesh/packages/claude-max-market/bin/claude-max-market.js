#!/usr/bin/env node

/**
 * Claude Max Market NPX Wrapper
 * 
 * Compliance-focused Docker orchestration for Claude Max capacity sharing
 * Implements user consent, control mechanisms, and legal compliance
 */

import { program } from 'commander';
import chalk from 'chalk';
import { ClaudeMaxMarket } from '../src/index.js';
import { LegalNotice } from '../src/legal/notice.js';
import { ComplianceChecker } from '../src/compliance/checker.js';

const market = new ClaudeMaxMarket();
const legal = new LegalNotice();
const compliance = new ComplianceChecker();

program
  .name('claude-max-market')
  .description('Claude Max Docker orchestration and market integration')
  .version('1.0.0');

// Legal compliance commands
program
  .command('terms')
  .description('Display legal notice and usage policy')
  .action(async () => {
    await legal.displayTerms();
  });

program
  .command('compliance-check')
  .description('Verify compliance with Anthropic ToS')
  .action(async () => {
    const isCompliant = await compliance.checkCompliance();
    if (isCompliant) {
      console.log(chalk.green('✅ System is compliant with Anthropic Terms of Service'));
    } else {
      console.log(chalk.red('❌ Compliance issues detected'));
      process.exit(1);
    }
  });

// Opt-in and consent management
program
  .command('opt-in')
  .description('Opt into Claude-backed job types with user consent')
  .option('--claude-jobs', 'Allow Claude job processing')
  .option('--max-daily <number>', 'Maximum daily tasks (default: 5)', '5')
  .option('--max-tokens <number>', 'Maximum tokens per task (default: 1000)', '1000')
  .action(async (options) => {
    await market.setupOptIn(options);
  });

program
  .command('opt-out')
  .description('Opt out of Claude job processing')
  .action(async () => {
    await market.optOut();
  });

// Usage tracking and limits
program
  .command('status')
  .description('Show usage statistics and limits')
  .action(async () => {
    await market.showStatus();
  });

program
  .command('limits')
  .description('Configure usage limits')
  .option('--daily <number>', 'Daily task limit')
  .option('--tokens <number>', 'Token limit per task')
  .option('--timeout <number>', 'Task timeout in seconds')
  .action(async (options) => {
    await market.setLimits(options);
  });

// Docker orchestration commands
program
  .command('docker:build')
  .description('Build Claude container image')
  .option('--tag <name>', 'Image tag (default: synaptic-mesh/claude-max)', 'synaptic-mesh/claude-max')
  .option('--no-cache', 'Build without cache')
  .action(async (options) => {
    await market.buildImage(options);
  });

program
  .command('docker:pull')
  .description('Pull Claude container image from registry')
  .option('--tag <name>', 'Image tag', 'synaptic-mesh/claude-max:latest')
  .action(async (options) => {
    await market.pullImage(options);
  });

// Job execution commands
program
  .command('execute')
  .description('Execute Claude job with user approval')
  .option('--prompt <text>', 'Claude prompt')
  .option('--file <path>', 'Input file path')
  .option('--model <name>', 'Claude model (default: claude-3-sonnet-20240229)', 'claude-3-sonnet-20240229')
  .option('--max-tokens <number>', 'Maximum tokens', '1000')
  .option('--approve-all', 'Auto-approve all tasks (not recommended)')
  .action(async (options) => {
    await market.executeJob(options);
  });

// Market integration commands
program
  .command('advertise')
  .description('Advertise available Claude capacity')
  .option('--slots <number>', 'Available execution slots', '1')
  .option('--price <number>', 'Price per task in RUV tokens', '5')
  .action(async (options) => {
    await market.advertiseCapacity(options);
  });

program
  .command('bid')
  .description('Bid for Claude task execution')
  .option('--task-id <id>', 'Task ID to bid on')
  .option('--max-price <number>', 'Maximum price to pay')
  .action(async (options) => {
    await market.placeBid(options);
  });

// Monitoring and logging commands
program
  .command('logs')
  .description('View execution logs and audit trail')
  .option('--follow', 'Follow log output')
  .option('--tail <number>', 'Number of lines to show', '50')
  .action(async (options) => {
    await market.showLogs(options);
  });

program
  .command('audit')
  .description('Generate compliance audit report')
  .option('--format <type>', 'Report format (json|text)', 'text')
  .option('--output <file>', 'Output file path')
  .action(async (options) => {
    await market.generateAuditReport(options);
  });

// Network and security commands
program
  .command('encrypt')
  .description('Encrypt job payload for secure transmission')
  .option('--input <file>', 'Input file to encrypt')
  .option('--output <file>', 'Output encrypted file')
  .action(async (options) => {
    await market.encryptPayload(options);
  });

program
  .command('decrypt')
  .description('Decrypt job payload')
  .option('--input <file>', 'Encrypted input file')
  .option('--output <file>', 'Output decrypted file')
  .action(async (options) => {
    await market.decryptPayload(options);
  });

// Health check and diagnostics
program
  .command('health')
  .description('Check system health and Docker connectivity')
  .action(async () => {
    await market.healthCheck();
  });

program
  .command('clean')
  .description('Clean up containers and temporary files')
  .option('--force', 'Force cleanup without confirmation')
  .action(async (options) => {
    await market.cleanup(options);
  });

// Configuration management
program
  .command('config')
  .description('Manage configuration settings')
  .option('--set <key=value>', 'Set configuration value')
  .option('--get <key>', 'Get configuration value')
  .option('--list', 'List all configuration')
  .action(async (options) => {
    await market.manageConfig(options);
  });

// Error handling
process.on('uncaughtException', (error) => {
  console.error(chalk.red('Uncaught Exception:'), error.message);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  console.error(chalk.red('Unhandled Rejection at:'), promise, 'reason:', reason);
  process.exit(1);
});

// Parse command line arguments
program.parse(process.argv);

// Show help if no command provided
if (!process.argv.slice(2).length) {
  program.outputHelp();
}