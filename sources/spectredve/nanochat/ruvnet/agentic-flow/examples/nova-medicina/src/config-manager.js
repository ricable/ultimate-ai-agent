/**
 * Configuration Manager
 * Manage Nova Medicina settings and API keys
 *
 * @author ruv (github.com/ruvnet, ruv.io)
 */

import { promises as fs } from 'fs';
import path from 'path';
import os from 'os';

export default class ConfigManager {
  constructor() {
    this.configDir = path.join(os.homedir(), '.nova-medicina');
    this.configFile = path.join(this.configDir, 'config.json');
    this.defaults = {
      verificationLevel: 'moderate',
      maxCacheAge: 24,
      hallucinationThreshold: 0.95,
      enableTelemetry: true,
      defaultOutputFormat: 'json'
    };
  }

  /**
   * Load configuration
   * @returns {Promise<Object>} Configuration object
   */
  async load() {
    try {
      const data = await fs.readFile(this.configFile, 'utf-8');
      return { ...this.defaults, ...JSON.parse(data) };
    } catch (error) {
      return this.defaults;
    }
  }

  /**
   * Save configuration
   * @param {Object} config - Configuration object
   * @returns {Promise<void>}
   */
  async save(config) {
    await fs.mkdir(this.configDir, { recursive: true });
    await fs.writeFile(this.configFile, JSON.stringify(config, null, 2));
  }

  /**
   * Set configuration value
   * @param {string} key - Configuration key
   * @param {*} value - Configuration value
   * @returns {Promise<void>}
   */
  async set(key, value) {
    const config = await this.load();
    config[key] = value;
    await this.save(config);
  }

  /**
   * Get configuration value
   * @param {string} key - Configuration key
   * @returns {Promise<*>} Configuration value
   */
  async get(key) {
    const config = await this.load();
    return config[key];
  }

  /**
   * Reset to default configuration
   * @returns {Promise<void>}
   */
  async reset() {
    await this.save(this.defaults);
  }

  /**
   * List all configuration settings
   * @returns {Promise<Object>} All configuration settings
   */
  async list() {
    return await this.load();
  }
}
