/**
 * Wizard Flow Integration Tests
 * Tests the complete Hive Mind setup wizard experience
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper to interact with CLI process
class CLIInteraction {
  constructor(command, args = []) {
    this.process = spawn('node', [command, ...args], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, NO_COLOR: '1' }
    });
    
    this.output = '';
    this.errorOutput = '';
    this.exitCode = null;
    this.closed = false;

    this.process.stdout.on('data', (data) => {
      this.output += data.toString();
    });

    this.process.stderr.on('data', (data) => {
      this.errorOutput += data.toString();
    });

    this.process.on('close', (code) => {
      this.exitCode = code;
      this.closed = true;
    });
  }

  write(input) {
    if (!this.closed) {
      this.process.stdin.write(input + '\n');
    }
  }

  async waitForPrompt(prompt, timeout = 5000) {
    const start = Date.now();
    while (!this.output.includes(prompt)) {
      if (Date.now() - start > timeout) {
        throw new Error(`Timeout waiting for prompt: ${prompt}`);
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  async waitForClose(timeout = 10000) {
    const start = Date.now();
    while (!this.closed) {
      if (Date.now() - start > timeout) {
        this.process.kill();
        throw new Error('Process timeout');
      }
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  getOutput() {
    return this.output;
  }

  getErrorOutput() {
    return this.errorOutput;
  }

  getExitCode() {
    return this.exitCode;
  }

  kill() {
    if (!this.closed) {
      this.process.kill();
    }
  }
}

describe('Hive Mind Wizard Flow', () => {
  const cliPath = path.join(__dirname, '../../../src/cli/simple-cli.js');
  const testDbPath = path.join(__dirname, '../../../tmp/test-wizard.db');
  let cli;

  beforeEach(() => {
    // Ensure tmp directory exists
    const tmpDir = path.dirname(testDbPath);
    if (!fs.existsSync(tmpDir)) {
      fs.mkdirSync(tmpDir, { recursive: true });
    }

    // Clean up any existing test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  afterEach(() => {
    if (cli) {
      cli.kill();
    }
    
    // Clean up test database
    if (fs.existsSync(testDbPath)) {
      fs.unlinkSync(testDbPath);
    }
  });

  describe('Basic Wizard Flow', () => {
    it('should complete wizard with minimal inputs', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      // Wait for welcome message
      await cli.waitForPrompt('🧙 Hive Mind Setup Wizard');
      await cli.waitForPrompt('Welcome to the Hive Mind distributed AI coordination system');

      // Enter objective
      await cli.waitForPrompt('Enter your objective');
      cli.write('Build a simple REST API');

      // Choose number of agents
      await cli.waitForPrompt('How many agents should we spawn?');
      cli.write('5');

      // Select topology
      await cli.waitForPrompt('Select swarm topology');
      cli.write('hierarchical');

      // Choose task type
      await cli.waitForPrompt('What type of task is this?');
      cli.write('development');

      // Wait for completion
      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      const output = cli.getOutput();
      
      // Verify initialization
      expect(output).toContain('Initializing database');
      expect(output).toContain('Spawning 5 agents');
      expect(output).toContain('Objective: Build a simple REST API');
      expect(output).toContain('Swarm is ready!');
      
      // Verify database was created
      expect(fs.existsSync(testDbPath)).toBe(true);
    });

    it('should handle advanced options', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('Complex system architecture design');

      await cli.waitForPrompt('How many agents');
      cli.write('20');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('mesh');

      await cli.waitForPrompt('What type of task');
      cli.write('architecture');

      // Advanced options
      await cli.waitForPrompt('Configure advanced options?');
      cli.write('y');

      // Memory settings
      await cli.waitForPrompt('Enable persistent memory?');
      cli.write('y');

      await cli.waitForPrompt('Memory size limit (MB)');
      cli.write('512');

      // Communication settings
      await cli.waitForPrompt('Enable message encryption?');
      cli.write('y');

      await cli.waitForPrompt('Message queue size');
      cli.write('1000');

      // Consensus settings
      await cli.waitForPrompt('Consensus threshold (%)');
      cli.write('75');

      // Auto-scaling
      await cli.waitForPrompt('Enable auto-scaling?');
      cli.write('y');

      await cli.waitForPrompt('Minimum agents');
      cli.write('10');

      await cli.waitForPrompt('Maximum agents');
      cli.write('50');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      const output = cli.getOutput();
      
      expect(output).toContain('Advanced configuration applied');
      expect(output).toContain('Persistent memory: enabled (512MB)');
      expect(output).toContain('Message encryption: enabled');
      expect(output).toContain('Auto-scaling: 10-50 agents');
      expect(output).toContain('Consensus threshold: 75%');
    });

    it('should validate inputs', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write(''); // Empty objective

      await cli.waitForPrompt('Objective cannot be empty');
      cli.write('Valid objective');

      await cli.waitForPrompt('How many agents');
      cli.write('0'); // Invalid number

      await cli.waitForPrompt('Number of agents must be between 1 and 1000');
      cli.write('5');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('invalid'); // Invalid topology

      await cli.waitForPrompt('Invalid selection');
      cli.write('star');

      await cli.waitForPrompt('What type of task');
      cli.write('development');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      expect(cli.getExitCode()).toBe(0);
    });
  });

  describe('Agent Configuration', () => {
    it('should allow custom agent distribution', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('Full-stack application development');

      await cli.waitForPrompt('How many agents');
      cli.write('custom');

      // Custom agent configuration
      await cli.waitForPrompt('Configure agent types');
      
      await cli.waitForPrompt('Number of researcher agents');
      cli.write('2');

      await cli.waitForPrompt('Number of coder agents');
      cli.write('4');

      await cli.waitForPrompt('Number of analyst agents');
      cli.write('1');

      await cli.waitForPrompt('Number of architect agents');
      cli.write('1');

      await cli.waitForPrompt('Number of tester agents');
      cli.write('2');

      await cli.waitForPrompt('Number of coordinator agents');
      cli.write('1');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('hierarchical');

      await cli.waitForPrompt('What type of task');
      cli.write('development');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      const output = cli.getOutput();
      
      expect(output).toContain('Spawning custom agent distribution:');
      expect(output).toContain('2 researcher agents');
      expect(output).toContain('4 coder agents');
      expect(output).toContain('1 analyst agent');
      expect(output).toContain('1 architect agent');
      expect(output).toContain('2 tester agents');
      expect(output).toContain('1 coordinator agent');
      expect(output).toContain('Total: 11 agents');
    });

    it('should suggest agent distribution based on task', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('Conduct market research and analysis');

      await cli.waitForPrompt('How many agents');
      cli.write('auto');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('mesh');

      await cli.waitForPrompt('What type of task');
      cli.write('research');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      const output = cli.getOutput();
      
      expect(output).toContain('Auto-detecting optimal agent distribution');
      expect(output).toContain('Based on task type: research');
      expect(output).toContain('researcher agents'); // Should have more researchers
      expect(output).toContain('analyst agents'); // Should have analysts
    });
  });

  describe('Task Templates', () => {
    it('should offer task templates', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('template');

      await cli.waitForPrompt('Select a task template');
      await cli.waitForPrompt('1. Code Review');
      await cli.waitForPrompt('2. Bug Fix');
      await cli.waitForPrompt('3. Feature Development');
      await cli.waitForPrompt('4. System Design');
      await cli.waitForPrompt('5. Performance Optimization');
      await cli.waitForPrompt('6. Security Audit');
      await cli.waitForPrompt('7. Documentation');
      await cli.waitForPrompt('8. Custom');

      cli.write('3'); // Feature Development

      await cli.waitForPrompt('Feature name');
      cli.write('User Authentication');

      await cli.waitForPrompt('Feature description');
      cli.write('JWT-based auth with refresh tokens');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      const output = cli.getOutput();
      
      expect(output).toContain('Using template: Feature Development');
      expect(output).toContain('Feature: User Authentication');
      expect(output).toContain('Pre-configured for feature development workflow');
    });
  });

  describe('Interactive Features', () => {
    it('should show progress indicators', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('Quick test task');

      await cli.waitForPrompt('How many agents');
      cli.write('3');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('star');

      await cli.waitForPrompt('What type of task');
      cli.write('general');

      // Should show progress
      await cli.waitForPrompt('⠋ Initializing database');
      await cli.waitForPrompt('⠋ Spawning agents');
      await cli.waitForPrompt('⠋ Setting up communication channels');
      await cli.waitForPrompt('⠋ Configuring task distribution');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      const output = cli.getOutput();
      expect(output).toContain('✓ Database initialized');
      expect(output).toContain('✓ Agents spawned');
      expect(output).toContain('✓ Communication established');
      expect(output).toContain('✓ Task distribution configured');
    });

    it('should provide help during wizard', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('?'); // Request help

      await cli.waitForPrompt('The objective describes what you want');
      await cli.waitForPrompt('Examples:');
      await cli.waitForPrompt('- Refactor the authentication module');
      await cli.waitForPrompt('- Design a microservices architecture');
      
      cli.write('Design a REST API');

      await cli.waitForPrompt('How many agents');
      cli.write('?'); // Request help

      await cli.waitForPrompt('Agent recommendations:');
      await cli.waitForPrompt('- Small tasks (1-5 agents)');
      await cli.waitForPrompt('- Medium tasks (5-20 agents)');
      await cli.waitForPrompt('- Large tasks (20-100 agents)');

      cli.write('10');

      // Continue through wizard
      await cli.waitForPrompt('Select swarm topology');
      cli.write('hierarchical');

      await cli.waitForPrompt('What type of task');
      cli.write('development');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      expect(cli.getExitCode()).toBe(0);
    });
  });

  describe('Error Recovery', () => {
    it('should handle interruption gracefully', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('Test interruption');

      await cli.waitForPrompt('How many agents');
      
      // Simulate Ctrl+C
      cli.process.kill('SIGINT');
      
      await cli.waitForClose();

      const output = cli.getOutput();
      expect(output).toContain('Setup cancelled by user');
      
      // Database should be cleaned up
      expect(fs.existsSync(testDbPath)).toBe(false);
    });

    it('should offer to resume incomplete setup', async () => {
      // First attempt - incomplete
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('Incomplete setup test');

      await cli.waitForPrompt('How many agents');
      cli.write('5');

      // Kill process to simulate crash
      cli.process.kill('SIGKILL');
      await cli.waitForClose();

      // Second attempt - should detect incomplete setup
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      await cli.waitForPrompt('Incomplete setup detected');
      await cli.waitForPrompt('Resume previous setup?');
      cli.write('n');

      await cli.waitForPrompt('Enter your objective');
      cli.write('Fresh setup');

      // Complete the wizard
      await cli.waitForPrompt('How many agents');
      cli.write('3');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('ring');

      await cli.waitForPrompt('What type of task');
      cli.write('general');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForClose();

      expect(cli.getExitCode()).toBe(0);
    });
  });

  describe('Post-Setup Actions', () => {
    it('should offer next steps after setup', async () => {
      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath]);

      // Quick setup
      await cli.waitForPrompt('Enter your objective');
      cli.write('Test next steps');

      await cli.waitForPrompt('How many agents');
      cli.write('3');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('star');

      await cli.waitForPrompt('What type of task');
      cli.write('general');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');

      // Should show next steps
      await cli.waitForPrompt('What would you like to do next?');
      await cli.waitForPrompt('1. Start task execution');
      await cli.waitForPrompt('2. View swarm status');
      await cli.waitForPrompt('3. Configure agents');
      await cli.waitForPrompt('4. Exit');

      cli.write('2');

      await cli.waitForPrompt('🐝 Swarm Status');
      await cli.waitForPrompt('Agents: 3 active');
      await cli.waitForPrompt('Topology: star');
      await cli.waitForPrompt('Ready for tasks');

      cli.write('4'); // Exit

      await cli.waitForClose();

      expect(cli.getExitCode()).toBe(0);
    });

    it('should save configuration for future use', async () => {
      const configPath = path.join(__dirname, '../../../tmp/hive-config.json');

      cli = new CLIInteraction(cliPath, ['hive-mind', '--db', testDbPath, '--save-config', configPath]);

      await cli.waitForPrompt('Enter your objective');
      cli.write('Configuration test');

      await cli.waitForPrompt('How many agents');
      cli.write('5');

      await cli.waitForPrompt('Select swarm topology');
      cli.write('mesh');

      await cli.waitForPrompt('What type of task');
      cli.write('research');

      await cli.waitForPrompt('✅ Hive Mind initialized successfully!');
      await cli.waitForPrompt('Configuration saved to');
      
      await cli.waitForClose();

      // Verify config was saved
      expect(fs.existsSync(configPath)).toBe(true);
      
      const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
      expect(config.objective).toBe('Configuration test');
      expect(config.agents).toBe(5);
      expect(config.topology).toBe('mesh');
      expect(config.taskType).toBe('research');

      // Clean up
      fs.unlinkSync(configPath);
    });
  });
});