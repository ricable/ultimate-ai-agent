/**
 * Configuration Manager
 * 
 * Manages system configuration with validation and security
 */

import fs from 'fs/promises';
import path from 'path';
import crypto from 'crypto';

export class ConfigManager {
  constructor() {
    this.configPath = path.join(process.cwd(), '.claude-max-config.json');
    this.config = {
      docker: {
        image: 'synaptic-mesh/claude-max:latest',
        memory: '512m',
        cpuShares: 512,
        timeout: 300000
      },
      security: {
        enableEncryption: true,
        keyRotationDays: 30,
        maxSessionAge: 3600000
      },
      limits: {
        dailyTasks: 5,
        dailyTokens: 5000,
        maxTokensPerTask: 1000,
        concurrentJobs: 1
      },
      market: {
        defaultPrice: 5,
        maxBidPrice: 100,
        bidTimeout: 300000,
        autoAcceptBids: false
      },
      compliance: {
        requireApproval: true,
        logAllActivity: true,
        enableAuditTrail: true,
        checkInterval: 3600000
      }
    };
  }

  /**
   * Load configuration from file
   */
  async load() {
    try {
      const content = await fs.readFile(this.configPath, 'utf8');
      const loadedConfig = JSON.parse(content);
      this.config = this.mergeConfig(this.config, loadedConfig);
      await this.validate();
    } catch (error) {
      // Config file doesn't exist, use defaults
      await this.save();
    }
  }

  /**
   * Save configuration to file
   */
  async save() {
    try {
      const configData = {
        ...this.config,
        lastUpdated: new Date().toISOString(),
        version: '1.0.0'
      };
      
      await fs.writeFile(this.configPath, JSON.stringify(configData, null, 2));
      await fs.chmod(this.configPath, 0o600); // Restrict access
    } catch (error) {
      console.warn('Failed to save configuration:', error.message);
    }
  }

  /**
   * Get configuration value
   */
  async get(key) {
    const keys = key.split('.');
    let value = this.config;
    
    for (const k of keys) {
      value = value?.[k];
      if (value === undefined) break;
    }
    
    return value;
  }

  /**
   * Set configuration value
   */
  async set(key, value) {
    const keys = key.split('.');
    let obj = this.config;
    
    for (let i = 0; i < keys.length - 1; i++) {
      const k = keys[i];
      if (!obj[k] || typeof obj[k] !== 'object') {
        obj[k] = {};
      }
      obj = obj[k];
    }
    
    obj[keys[keys.length - 1]] = value;
    await this.validate();
    await this.save();
  }

  /**
   * Get all configuration
   */
  async getAll() {
    return { ...this.config };
  }

  /**
   * Reset to defaults
   */
  async reset() {
    const defaults = {
      docker: {
        image: 'synaptic-mesh/claude-max:latest',
        memory: '512m',
        cpuShares: 512,
        timeout: 300000
      },
      security: {
        enableEncryption: true,
        keyRotationDays: 30,
        maxSessionAge: 3600000
      },
      limits: {
        dailyTasks: 5,
        dailyTokens: 5000,
        maxTokensPerTask: 1000,
        concurrentJobs: 1
      },
      market: {
        defaultPrice: 5,
        maxBidPrice: 100,
        bidTimeout: 300000,
        autoAcceptBids: false
      },
      compliance: {
        requireApproval: true,
        logAllActivity: true,
        enableAuditTrail: true,
        checkInterval: 3600000
      }
    };
    
    this.config = defaults;
    await this.save();
  }

  /**
   * Validate configuration
   */
  async validate() {
    const errors = [];
    
    // Docker validation
    if (!this.config.docker?.image) {
      errors.push('Docker image not specified');
    }
    
    if (this.config.docker?.timeout && this.config.docker.timeout < 10000) {
      errors.push('Docker timeout too low (minimum 10 seconds)');
    }
    
    // Limits validation
    if (this.config.limits?.dailyTasks && this.config.limits.dailyTasks <= 0) {
      errors.push('Daily task limit must be positive');
    }
    
    if (this.config.limits?.maxTokensPerTask && this.config.limits.maxTokensPerTask <= 0) {
      errors.push('Max tokens per task must be positive');
    }
    
    // Market validation
    if (this.config.market?.defaultPrice && this.config.market.defaultPrice < 0) {
      errors.push('Default price cannot be negative');
    }
    
    if (this.config.market?.maxBidPrice && this.config.market.maxBidPrice <= 0) {
      errors.push('Max bid price must be positive');
    }
    
    if (errors.length > 0) {
      throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
    }
    
    return true;
  }

  /**
   * Get configuration schema for UI/CLI
   */
  getSchema() {
    return {
      docker: {
        type: 'object',
        properties: {
          image: { type: 'string', description: 'Docker image name' },
          memory: { type: 'string', description: 'Memory limit (e.g., 512m)' },
          cpuShares: { type: 'number', description: 'CPU shares (1024 = 1 CPU)' },
          timeout: { type: 'number', description: 'Job timeout in milliseconds' }
        }
      },
      security: {
        type: 'object',
        properties: {
          enableEncryption: { type: 'boolean', description: 'Enable payload encryption' },
          keyRotationDays: { type: 'number', description: 'Key rotation interval (days)' },
          maxSessionAge: { type: 'number', description: 'Max session age (milliseconds)' }
        }
      },
      limits: {
        type: 'object',
        properties: {
          dailyTasks: { type: 'number', description: 'Maximum tasks per day' },
          dailyTokens: { type: 'number', description: 'Maximum tokens per day' },
          maxTokensPerTask: { type: 'number', description: 'Maximum tokens per task' },
          concurrentJobs: { type: 'number', description: 'Maximum concurrent jobs' }
        }
      },
      market: {
        type: 'object',
        properties: {
          defaultPrice: { type: 'number', description: 'Default price in RUV tokens' },
          maxBidPrice: { type: 'number', description: 'Maximum bid price' },
          bidTimeout: { type: 'number', description: 'Bid timeout (milliseconds)' },
          autoAcceptBids: { type: 'boolean', description: 'Auto-accept matching bids' }
        }
      },
      compliance: {
        type: 'object',
        properties: {
          requireApproval: { type: 'boolean', description: 'Require user approval for jobs' },
          logAllActivity: { type: 'boolean', description: 'Log all activity' },
          enableAuditTrail: { type: 'boolean', description: 'Enable audit trail' },
          checkInterval: { type: 'number', description: 'Compliance check interval (ms)' }
        }
      }
    };
  }

  /**
   * Import configuration from file
   */
  async import(filePath) {
    try {
      const content = await fs.readFile(filePath, 'utf8');
      const importedConfig = JSON.parse(content);
      
      // Validate imported config
      const tempConfig = this.mergeConfig(this.config, importedConfig);
      
      // Temporarily set config for validation
      const originalConfig = { ...this.config };
      this.config = tempConfig;
      
      try {
        await this.validate();
        await this.save();
        return true;
      } catch (error) {
        // Restore original config on validation failure
        this.config = originalConfig;
        throw error;
      }
    } catch (error) {
      throw new Error(`Configuration import failed: ${error.message}`);
    }
  }

  /**
   * Export configuration to file
   */
  async export(filePath) {
    try {
      const exportData = {
        ...this.config,
        exported: new Date().toISOString(),
        version: '1.0.0'
      };
      
      await fs.writeFile(filePath, JSON.stringify(exportData, null, 2));
      return true;
    } catch (error) {
      throw new Error(`Configuration export failed: ${error.message}`);
    }
  }

  /**
   * Generate configuration hash for integrity checking
   */
  generateHash() {
    const configStr = JSON.stringify(this.config, Object.keys(this.config).sort());
    return crypto.createHash('sha256').update(configStr).digest('hex');
  }

  /**
   * Backup current configuration
   */
  async backup() {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPath = path.join(process.cwd(), `.claude-max-config.backup.${timestamp}.json`);
    
    try {
      await this.export(backupPath);
      return backupPath;
    } catch (error) {
      throw new Error(`Configuration backup failed: ${error.message}`);
    }
  }

  /**
   * Restore from backup
   */
  async restore(backupPath) {
    try {
      await this.import(backupPath);
      return true;
    } catch (error) {
      throw new Error(`Configuration restore failed: ${error.message}`);
    }
  }

  // Private helper methods

  mergeConfig(target, source) {
    const result = { ...target };
    
    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        result[key] = this.mergeConfig(target[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }
    
    return result;
  }
}