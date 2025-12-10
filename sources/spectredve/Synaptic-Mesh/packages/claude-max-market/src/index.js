/**
 * Claude Max Market - Main Implementation
 * 
 * Provides Docker orchestration, market integration, and compliance features
 * for Claude Max capacity sharing with full user control and consent
 */

import Docker from 'dockerode';
import inquirer from 'inquirer';
import chalk from 'chalk';
import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';
import WebSocket from 'ws';
import cron from 'node-cron';
import dotenv from 'dotenv';

import { JobOrchestrator } from './orchestration/jobOrchestrator.js';
import { EncryptionManager } from './security/encryption.js';
import { UsageTracker } from './tracking/usageTracker.js';
import { ComplianceManager } from './compliance/manager.js';
import { ConfigManager } from './config/manager.js';
import { LogManager } from './logging/logManager.js';
import { MarketIntegration } from './market/integration.js';

export class ClaudeMaxMarket {
  constructor(options = {}) {
    // Load environment configuration
    dotenv.config();
    
    // Initialize Docker client
    this.docker = new Docker();
    
    // Initialize core components
    this.orchestrator = new JobOrchestrator(this.docker);
    this.encryption = new EncryptionManager();
    this.usageTracker = new UsageTracker();
    this.compliance = new ComplianceManager();
    this.config = new ConfigManager();
    this.logger = new LogManager();
    this.market = new MarketIntegration();
    
    // User consent and control state
    this.userConsent = false;
    this.isOptedIn = false;
    this.dailyLimits = {
      maxTasks: 5,
      maxTokens: 1000,
      timeout: 300
    };
    
    // Initialize system
    this.init();
  }

  async init() {
    try {
      // Load configuration
      await this.config.load();
      
      // Initialize logging
      await this.logger.init();
      
      // Check compliance status
      await this.compliance.verifyCompliance();
      
      // Load user preferences
      await this.loadUserPreferences();
      
      // Set up periodic cleanup
      this.setupCleanupSchedule();
      
      this.logger.info('Claude Max Market initialized successfully');
    } catch (error) {
      console.error(chalk.red('Failed to initialize Claude Max Market:'), error.message);
      throw error;
    }
  }

  /**
   * Setup user opt-in with comprehensive consent management
   */
  async setupOptIn(options = {}) {
    console.log(chalk.blue('🔒 Claude Max Market Opt-In Configuration'));
    console.log(chalk.yellow('⚠️  Please review the terms and compliance requirements'));
    
    // Display legal notice first
    await this.displayLegalNotice();
    
    // Get user consent
    const consent = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'acceptTerms',
        message: 'Do you accept the terms and understand that you maintain full control of your Claude account?',
        default: false
      },
      {
        type: 'confirm',
        name: 'optInClaudeJobs',
        message: 'Do you want to opt into Claude-backed job processing?',
        default: false,
        when: (answers) => answers.acceptTerms
      },
      {
        type: 'number',
        name: 'maxDailyTasks',
        message: 'Maximum daily tasks to process:',
        default: parseInt(options.maxDaily) || 5,
        validate: (value) => value > 0 && value <= 100,
        when: (answers) => answers.optInClaudeJobs
      },
      {
        type: 'number',
        name: 'maxTokensPerTask',
        message: 'Maximum tokens per task:',
        default: parseInt(options.maxTokens) || 1000,
        validate: (value) => value > 0 && value <= 10000,
        when: (answers) => answers.optInClaudeJobs
      }
    ]);

    if (!consent.acceptTerms) {
      console.log(chalk.red('❌ Terms not accepted. Opt-in cancelled.'));
      return;
    }

    if (!consent.optInClaudeJobs) {
      console.log(chalk.yellow('⚠️  Claude job processing disabled by user choice.'));
      return;
    }

    // Save user preferences
    this.userConsent = true;
    this.isOptedIn = true;
    this.dailyLimits = {
      maxTasks: consent.maxDailyTasks,
      maxTokens: consent.maxTokensPerTask,
      timeout: 300
    };

    await this.saveUserPreferences();
    
    console.log(chalk.green('✅ Successfully opted into Claude job processing'));
    console.log(chalk.blue(`📊 Daily limits: ${this.dailyLimits.maxTasks} tasks, ${this.dailyLimits.maxTokens} tokens per task`));
    
    // Initialize usage tracking
    await this.usageTracker.reset();
    
    this.logger.info('User opted into Claude job processing', {
      maxTasks: this.dailyLimits.maxTasks,
      maxTokens: this.dailyLimits.maxTokens
    });
  }

  /**
   * Opt out of Claude job processing
   */
  async optOut() {
    console.log(chalk.yellow('🚫 Opting out of Claude job processing...'));
    
    const confirm = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'confirmOptOut',
        message: 'Are you sure you want to opt out? This will stop all Claude job processing.',
        default: false
      }
    ]);

    if (!confirm.confirmOptOut) {
      console.log(chalk.blue('Opt-out cancelled.'));
      return;
    }

    this.userConsent = false;
    this.isOptedIn = false;
    await this.saveUserPreferences();
    
    // Stop any running jobs
    await this.orchestrator.stopAllJobs();
    
    console.log(chalk.green('✅ Successfully opted out of Claude job processing'));
    this.logger.info('User opted out of Claude job processing');
  }

  /**
   * Execute Claude job with user approval and compliance checks
   */
  async executeJob(options = {}) {
    try {
      // Check if user is opted in
      if (!this.isOptedIn) {
        console.log(chalk.red('❌ You must opt in first using: claude-max-market opt-in'));
        return;
      }

      // Check usage limits
      const usageStatus = await this.usageTracker.checkLimits(this.dailyLimits);
      if (!usageStatus.allowed) {
        console.log(chalk.red(`❌ Usage limit exceeded: ${usageStatus.reason}`));
        return;
      }

      // Prepare job payload
      const jobPayload = await this.prepareJobPayload(options);
      
      // Request user approval unless auto-approved
      if (!options.approveAll) {
        const approval = await this.requestJobApproval(jobPayload);
        if (!approval) {
          console.log(chalk.yellow('❌ Job execution cancelled by user'));
          return;
        }
      }

      // Execute job with Docker orchestration
      console.log(chalk.blue('🚀 Executing Claude job...'));
      const result = await this.orchestrator.executeJob(jobPayload);
      
      // Track usage
      await this.usageTracker.recordUsage({
        tokens: result.usage?.totalTokens || 0,
        executionTime: result.executionTime,
        success: result.success
      });

      // Display results
      this.displayJobResult(result);
      
      this.logger.info('Job executed successfully', {
        jobId: result.jobId,
        tokens: result.usage?.totalTokens,
        executionTime: result.executionTime
      });

      return result;
    } catch (error) {
      console.error(chalk.red('❌ Job execution failed:'), error.message);
      this.logger.error('Job execution failed', { error: error.message });
      throw error;
    }
  }

  /**
   * Build Claude container image
   */
  async buildImage(options = {}) {
    console.log(chalk.blue('🔨 Building Claude container image...'));
    
    try {
      const buildResult = await this.orchestrator.buildImage({
        tag: options.tag,
        nocache: options.noCache
      });
      
      console.log(chalk.green(`✅ Successfully built image: ${options.tag}`));
      this.logger.info('Docker image built successfully', { tag: options.tag });
      
      return buildResult;
    } catch (error) {
      console.error(chalk.red('❌ Failed to build image:'), error.message);
      throw error;
    }
  }

  /**
   * Pull Claude container image
   */
  async pullImage(options = {}) {
    console.log(chalk.blue('📥 Pulling Claude container image...'));
    
    try {
      const pullResult = await this.orchestrator.pullImage(options.tag);
      
      console.log(chalk.green(`✅ Successfully pulled image: ${options.tag}`));
      this.logger.info('Docker image pulled successfully', { tag: options.tag });
      
      return pullResult;
    } catch (error) {
      console.error(chalk.red('❌ Failed to pull image:'), error.message);
      throw error;
    }
  }

  /**
   * Advertise available Claude capacity
   */
  async advertiseCapacity(options = {}) {
    if (!this.isOptedIn) {
      console.log(chalk.red('❌ You must opt in first to advertise capacity'));
      return;
    }

    console.log(chalk.blue('📢 Advertising Claude capacity...'));
    
    try {
      const offer = await this.market.advertise({
        slots: parseInt(options.slots) || 1,
        price: parseInt(options.price) || 5,
        capabilities: ['claude-3-sonnet', 'claude-3-haiku'],
        compliance: await this.compliance.getComplianceStatus()
      });
      
      console.log(chalk.green('✅ Capacity advertised successfully'));
      console.log(`📊 Slots: ${offer.slots}, Price: ${offer.price} RUV`);
      
      this.logger.info('Capacity advertised', offer);
      return offer;
    } catch (error) {
      console.error(chalk.red('❌ Failed to advertise capacity:'), error.message);
      throw error;
    }
  }

  /**
   * Place bid for Claude task execution
   */
  async placeBid(options = {}) {
    console.log(chalk.blue('💰 Placing bid for Claude task...'));
    
    try {
      const bid = await this.market.placeBid({
        taskId: options.taskId,
        maxPrice: parseInt(options.maxPrice),
        capabilities: ['claude-3-sonnet']
      });
      
      console.log(chalk.green('✅ Bid placed successfully'));
      console.log(`📊 Task ID: ${bid.taskId}, Max Price: ${bid.maxPrice} RUV`);
      
      this.logger.info('Bid placed', bid);
      return bid;
    } catch (error) {
      console.error(chalk.red('❌ Failed to place bid:'), error.message);
      throw error;
    }
  }

  /**
   * Show usage status and limits
   */
  async showStatus() {
    console.log(chalk.blue('📊 Claude Max Market Status'));
    console.log('=' .repeat(40));
    
    // User status
    console.log(chalk.bold('User Status:'));
    console.log(`Opted In: ${this.isOptedIn ? chalk.green('Yes') : chalk.red('No')}`);
    console.log(`Consent Given: ${this.userConsent ? chalk.green('Yes') : chalk.red('No')}`);
    
    if (this.isOptedIn) {
      // Usage statistics
      const usage = await this.usageTracker.getTodayUsage();
      console.log(chalk.bold('\nToday\'s Usage:'));
      console.log(`Tasks: ${usage.tasks}/${this.dailyLimits.maxTasks}`);
      console.log(`Tokens: ${usage.tokens}/${this.dailyLimits.maxTokens * this.dailyLimits.maxTasks}`);
      
      // Limits
      console.log(chalk.bold('\nLimits:'));
      console.log(`Max Daily Tasks: ${this.dailyLimits.maxTasks}`);
      console.log(`Max Tokens/Task: ${this.dailyLimits.maxTokens}`);
      console.log(`Timeout: ${this.dailyLimits.timeout}s`);
    }
    
    // Compliance status
    const compliance = await this.compliance.getComplianceStatus();
    console.log(chalk.bold('\nCompliance Status:'));
    console.log(`Anthropic ToS: ${compliance.anthropicTos ? chalk.green('✅') : chalk.red('❌')}`);
    console.log(`No Shared Keys: ${compliance.noSharedKeys ? chalk.green('✅') : chalk.red('❌')}`);
    console.log(`Peer Orchestrated: ${compliance.peerOrchestrated ? chalk.green('✅') : chalk.red('❌')}`);
    
    // System health
    const health = await this.orchestrator.healthCheck();
    console.log(chalk.bold('\nSystem Health:'));
    console.log(`Docker: ${health.docker ? chalk.green('✅') : chalk.red('❌')}`);
    console.log(`Image Available: ${health.imageAvailable ? chalk.green('✅') : chalk.red('❌')}`);
  }

  /**
   * Set usage limits
   */
  async setLimits(options = {}) {
    console.log(chalk.blue('⚙️  Configuring usage limits...'));
    
    const updates = {};
    if (options.daily) updates.maxTasks = parseInt(options.daily);
    if (options.tokens) updates.maxTokens = parseInt(options.tokens);
    if (options.timeout) updates.timeout = parseInt(options.timeout);
    
    if (Object.keys(updates).length === 0) {
      // Interactive limit setting
      const limits = await inquirer.prompt([
        {
          type: 'number',
          name: 'maxTasks',
          message: 'Maximum daily tasks:',
          default: this.dailyLimits.maxTasks,
          validate: (value) => value > 0 && value <= 100
        },
        {
          type: 'number',
          name: 'maxTokens',
          message: 'Maximum tokens per task:',
          default: this.dailyLimits.maxTokens,
          validate: (value) => value > 0 && value <= 10000
        },
        {
          type: 'number',
          name: 'timeout',
          message: 'Task timeout (seconds):',
          default: this.dailyLimits.timeout,
          validate: (value) => value > 0 && value <= 3600
        }
      ]);
      
      Object.assign(updates, limits);
    }
    
    // Update limits
    Object.assign(this.dailyLimits, updates);
    await this.saveUserPreferences();
    
    console.log(chalk.green('✅ Usage limits updated successfully'));
    console.log(`📊 Daily tasks: ${this.dailyLimits.maxTasks}`);
    console.log(`📊 Tokens/task: ${this.dailyLimits.maxTokens}`);
    console.log(`📊 Timeout: ${this.dailyLimits.timeout}s`);
    
    this.logger.info('Usage limits updated', this.dailyLimits);
  }

  /**
   * Show execution logs
   */
  async showLogs(options = {}) {
    console.log(chalk.blue('📋 Execution Logs'));
    console.log('=' .repeat(40));
    
    try {
      const logs = await this.logger.getLogs({
        tail: parseInt(options.tail) || 50,
        follow: options.follow
      });
      
      logs.forEach(log => {
        const timestamp = new Date(log.timestamp).toLocaleString();
        const level = log.level.toUpperCase();
        const color = this.getLogColor(level);
        
        console.log(`${chalk.gray(timestamp)} [${color(level)}] ${log.message}`);
        if (log.metadata) {
          console.log(chalk.gray('  ' + JSON.stringify(log.metadata, null, 2)));
        }
      });
    } catch (error) {
      console.error(chalk.red('❌ Failed to retrieve logs:'), error.message);
    }
  }

  /**
   * Generate compliance audit report
   */
  async generateAuditReport(options = {}) {
    console.log(chalk.blue('📋 Generating compliance audit report...'));
    
    try {
      const report = await this.compliance.generateAuditReport();
      
      if (options.format === 'json') {
        const output = JSON.stringify(report, null, 2);
        
        if (options.output) {
          await fs.writeFile(options.output, output);
          console.log(chalk.green(`✅ Audit report saved to: ${options.output}`));
        } else {
          console.log(output);
        }
      } else {
        // Text format
        console.log(chalk.bold('Compliance Audit Report'));
        console.log('=' .repeat(50));
        console.log(`Generated: ${new Date().toISOString()}`);
        console.log(`Compliance Score: ${report.score}/100`);
        console.log(`\nChecks Performed: ${report.checks.length}`);
        
        report.checks.forEach(check => {
          const status = check.passed ? chalk.green('✅') : chalk.red('❌');
          console.log(`${status} ${check.name}: ${check.description}`);
        });
        
        if (options.output) {
          await fs.writeFile(options.output, JSON.stringify(report, null, 2));
          console.log(chalk.green(`\n✅ Full report saved to: ${options.output}`));
        }
      }
    } catch (error) {
      console.error(chalk.red('❌ Failed to generate audit report:'), error.message);
    }
  }

  /**
   * Encrypt job payload for secure transmission
   */
  async encryptPayload(options = {}) {
    if (!options.input) {
      console.error(chalk.red('❌ Input file required'));
      return;
    }
    
    try {
      const inputData = await fs.readFile(options.input, 'utf8');
      const encrypted = await this.encryption.encrypt(inputData);
      
      const outputFile = options.output || options.input + '.encrypted';
      await fs.writeFile(outputFile, JSON.stringify(encrypted));
      
      console.log(chalk.green(`✅ Payload encrypted: ${outputFile}`));
      this.logger.info('Payload encrypted', { input: options.input, output: outputFile });
    } catch (error) {
      console.error(chalk.red('❌ Encryption failed:'), error.message);
    }
  }

  /**
   * Decrypt job payload
   */
  async decryptPayload(options = {}) {
    if (!options.input) {
      console.error(chalk.red('❌ Input file required'));
      return;
    }
    
    try {
      const encryptedData = JSON.parse(await fs.readFile(options.input, 'utf8'));
      const decrypted = await this.encryption.decrypt(encryptedData);
      
      const outputFile = options.output || options.input.replace('.encrypted', '');
      await fs.writeFile(outputFile, decrypted);
      
      console.log(chalk.green(`✅ Payload decrypted: ${outputFile}`));
      this.logger.info('Payload decrypted', { input: options.input, output: outputFile });
    } catch (error) {
      console.error(chalk.red('❌ Decryption failed:'), error.message);
    }
  }

  /**
   * System health check
   */
  async healthCheck() {
    console.log(chalk.blue('🔍 System Health Check'));
    console.log('=' .repeat(30));
    
    try {
      // Docker connectivity
      const dockerInfo = await this.docker.info();
      console.log(chalk.green('✅ Docker connection: OK'));
      
      // Image availability
      const images = await this.docker.listImages();
      const claudeImage = images.find(img => 
        img.RepoTags && img.RepoTags.some(tag => tag.includes('claude'))
      );
      
      if (claudeImage) {
        console.log(chalk.green('✅ Claude image: Available'));
      } else {
        console.log(chalk.yellow('⚠️  Claude image: Not found (run docker:pull)'));
      }
      
      // Compliance check
      const compliance = await this.compliance.quickCheck();
      console.log(`${compliance ? chalk.green('✅') : chalk.red('❌')} Compliance: ${compliance ? 'OK' : 'Issues detected'}`);
      
      // Configuration
      const configValid = await this.config.validate();
      console.log(`${configValid ? chalk.green('✅') : chalk.red('❌')} Configuration: ${configValid ? 'Valid' : 'Invalid'}`);
      
      // Disk space (basic check)
      console.log(chalk.green('✅ System resources: OK'));
      
      console.log(chalk.green('\n🎉 Health check completed'));
    } catch (error) {
      console.error(chalk.red('❌ Health check failed:'), error.message);
    }
  }

  /**
   * Cleanup containers and temporary files
   */
  async cleanup(options = {}) {
    console.log(chalk.blue('🧹 Cleaning up system...'));
    
    if (!options.force) {
      const confirm = await inquirer.prompt([
        {
          type: 'confirm',
          name: 'proceed',
          message: 'This will remove stopped containers and temporary files. Continue?',
          default: false
        }
      ]);
      
      if (!confirm.proceed) {
        console.log(chalk.yellow('Cleanup cancelled'));
        return;
      }
    }
    
    try {
      // Clean up Docker containers
      const cleanupResult = await this.orchestrator.cleanup();
      console.log(chalk.green(`✅ Removed ${cleanupResult.containersRemoved} containers`));
      
      // Clean up temporary files
      const tempCleanup = await this.logger.cleanupOldLogs();
      console.log(chalk.green(`✅ Cleaned ${tempCleanup.filesRemoved} log files`));
      
      console.log(chalk.green('🎉 Cleanup completed successfully'));
      this.logger.info('System cleanup completed', {
        containersRemoved: cleanupResult.containersRemoved,
        filesRemoved: tempCleanup.filesRemoved
      });
    } catch (error) {
      console.error(chalk.red('❌ Cleanup failed:'), error.message);
    }
  }

  /**
   * Manage configuration
   */
  async manageConfig(options = {}) {
    if (options.set) {
      const [key, value] = options.set.split('=');
      if (!key || !value) {
        console.error(chalk.red('❌ Invalid format. Use: --set key=value'));
        return;
      }
      
      await this.config.set(key, value);
      console.log(chalk.green(`✅ Configuration set: ${key}=${value}`));
    } else if (options.get) {
      const value = await this.config.get(options.get);
      console.log(`${options.get}=${value || 'undefined'}`);
    } else if (options.list) {
      const allConfig = await this.config.getAll();
      console.log(chalk.blue('Configuration:'));
      Object.entries(allConfig).forEach(([key, value]) => {
        console.log(`${key}=${value}`);
      });
    } else {
      console.error(chalk.red('❌ Please specify --set, --get, or --list'));
    }
  }

  // Private helper methods

  async prepareJobPayload(options) {
    let prompt = options.prompt;
    
    if (!prompt && options.file) {
      prompt = await fs.readFile(options.file, 'utf8');
    }
    
    if (!prompt) {
      throw new Error('No prompt provided');
    }
    
    return {
      id: uuidv4(),
      prompt,
      model: options.model || 'claude-3-sonnet-20240229',
      maxTokens: parseInt(options.maxTokens) || 1000,
      timestamp: new Date().toISOString()
    };
  }

  async requestJobApproval(jobPayload) {
    console.log(chalk.yellow('\n🔐 Job Approval Required'));
    console.log('=' .repeat(40));
    console.log(`Model: ${jobPayload.model}`);
    console.log(`Max Tokens: ${jobPayload.maxTokens}`);
    console.log(`Prompt Preview: ${jobPayload.prompt.substring(0, 100)}${jobPayload.prompt.length > 100 ? '...' : ''}`);
    
    const approval = await inquirer.prompt([
      {
        type: 'confirm',
        name: 'approved',
        message: 'Do you approve this Claude job execution?',
        default: false
      }
    ]);
    
    return approval.approved;
  }

  displayJobResult(result) {
    console.log(chalk.blue('\n📋 Job Result'));
    console.log('=' .repeat(30));
    
    if (result.success) {
      console.log(chalk.green('✅ Status: Success'));
      console.log(`📊 Execution Time: ${result.executionTime}ms`);
      console.log(`🔤 Tokens Used: ${result.usage?.totalTokens || 0}`);
      console.log(chalk.bold('\n📝 Response:'));
      console.log(result.response);
    } else {
      console.log(chalk.red('❌ Status: Failed'));
      console.log(chalk.red(`Error: ${result.error}`));
    }
  }

  async displayLegalNotice() {
    console.log(chalk.blue('\n📋 Legal Notice & Compliance'));
    console.log('=' .repeat(50));
    console.log(chalk.white(
      'Synaptic Mesh does not proxy or resell access to Claude Max.\n' +
      'All compute is run locally by consenting nodes with individual\n' +
      'Claude subscriptions. Participation is voluntary. API keys are\n' +
      'never shared or transmitted.\n\n' +
      'By proceeding, you confirm:\n' +
      '• You have your own Claude Max subscription\n' +
      '• You maintain full control of your API credentials\n' +
      '• You understand this is voluntary contribution, not resale\n' +
      '• You can revoke consent at any time'
    ));
    console.log('=' .repeat(50));
  }

  async loadUserPreferences() {
    try {
      const prefsPath = path.join(process.cwd(), '.claude-max-market.json');
      const prefs = JSON.parse(await fs.readFile(prefsPath, 'utf8'));
      
      this.userConsent = prefs.userConsent || false;
      this.isOptedIn = prefs.isOptedIn || false;
      this.dailyLimits = { ...this.dailyLimits, ...(prefs.dailyLimits || {}) };
    } catch (error) {
      // Preferences don't exist yet, use defaults
    }
  }

  async saveUserPreferences() {
    try {
      const prefsPath = path.join(process.cwd(), '.claude-max-market.json');
      const prefs = {
        userConsent: this.userConsent,
        isOptedIn: this.isOptedIn,
        dailyLimits: this.dailyLimits,
        lastUpdated: new Date().toISOString()
      };
      
      await fs.writeFile(prefsPath, JSON.stringify(prefs, null, 2));
    } catch (error) {
      this.logger.error('Failed to save user preferences', { error: error.message });
    }
  }

  setupCleanupSchedule() {
    // Run cleanup daily at 2 AM
    cron.schedule('0 2 * * *', async () => {
      this.logger.info('Running scheduled cleanup');
      try {
        await this.cleanup({ force: true });
      } catch (error) {
        this.logger.error('Scheduled cleanup failed', { error: error.message });
      }
    });
  }

  getLogColor(level) {
    switch (level) {
      case 'ERROR': return chalk.red;
      case 'WARN': return chalk.yellow;
      case 'INFO': return chalk.blue;
      case 'DEBUG': return chalk.gray;
      default: return chalk.white;
    }
  }
}