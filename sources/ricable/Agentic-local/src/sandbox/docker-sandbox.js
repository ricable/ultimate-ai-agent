/**
 * Docker Sandbox Manager
 * Provides secure code execution environment for AI agents
 */

import { spawn } from 'child_process';
import { randomBytes } from 'crypto';
import { mkdirSync, writeFileSync, readFileSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';

export class DockerSandbox {
  constructor(options = {}) {
    this.image = options.image || 'node:20-alpine';
    this.memoryLimit = options.memoryLimit || '2g';
    this.cpuLimit = options.cpuLimit || '2';
    this.timeout = options.timeout || 300000; // 5 minutes
    this.network = options.network || 'none';
    this.volumesDir = options.volumesDir || join(process.cwd(), 'sandbox-volumes');

    // Ensure volumes directory exists
    try {
      mkdirSync(this.volumesDir, { recursive: true });
    } catch (err) {
      // Directory may already exist
    }
  }

  /**
   * Execute code in isolated Docker container
   * @param {string} code - Code to execute
   * @param {string} language - Programming language (javascript, python, etc.)
   * @param {object} options - Additional execution options
   * @returns {Promise<object>} Execution result
   */
  async execute(code, language = 'javascript', options = {}) {
    const sessionId = randomBytes(16).toString('hex');
    const sessionDir = join(this.volumesDir, sessionId);

    try {
      // Create session directory
      mkdirSync(sessionDir, { recursive: true });

      // Prepare code file
      const { filename, command } = this._prepareExecution(code, language, sessionDir);

      // Build Docker command
      const dockerArgs = [
        'run',
        '--rm',
        '--name', `sandbox-${sessionId}`,
        '--memory', this.memoryLimit,
        '--cpus', this.cpuLimit,
        '--network', this.network,
        '-v', `${sessionDir}:/workspace:rw`,
        '-w', '/workspace',
        '--security-opt', 'no-new-privileges',
        '--cap-drop', 'ALL',
        '--read-only',
        '--tmpfs', '/tmp:rw,noexec,nosuid,size=100m',
        this.image,
        ...command
      ];

      // Execute with timeout
      const result = await this._runWithTimeout(dockerArgs, this.timeout);

      return {
        success: result.exitCode === 0,
        exitCode: result.exitCode,
        stdout: result.stdout,
        stderr: result.stderr,
        sessionId
      };

    } catch (error) {
      return {
        success: false,
        exitCode: -1,
        stdout: '',
        stderr: error.message,
        sessionId
      };
    } finally {
      // Cleanup session directory
      try {
        rmSync(sessionDir, { recursive: true, force: true });
      } catch (cleanupError) {
        console.error(`Failed to cleanup session ${sessionId}:`, cleanupError);
      }
    }
  }

  /**
   * Execute Python code
   * @param {string} code - Python code
   * @returns {Promise<object>} Execution result
   */
  async executePython(code) {
    return this.execute(code, 'python', { image: 'python:3.11-alpine' });
  }

  /**
   * Execute JavaScript code
   * @param {string} code - JavaScript code
   * @returns {Promise<object>} Execution result
   */
  async executeJavaScript(code) {
    return this.execute(code, 'javascript');
  }

  /**
   * Prepare execution environment based on language
   * @private
   */
  _prepareExecution(code, language, sessionDir) {
    switch (language.toLowerCase()) {
      case 'javascript':
      case 'js':
        const jsFile = join(sessionDir, 'code.js');
        writeFileSync(jsFile, code);
        return {
          filename: 'code.js',
          command: ['node', 'code.js']
        };

      case 'python':
      case 'py':
        const pyFile = join(sessionDir, 'code.py');
        writeFileSync(pyFile, code);
        return {
          filename: 'code.py',
          command: ['python', 'code.py']
        };

      case 'typescript':
      case 'ts':
        const tsFile = join(sessionDir, 'code.ts');
        writeFileSync(tsFile, code);
        return {
          filename: 'code.ts',
          command: ['npx', 'tsx', 'code.ts']
        };

      default:
        throw new Error(`Unsupported language: ${language}`);
    }
  }

  /**
   * Run Docker command with timeout
   * @private
   */
  _runWithTimeout(args, timeout) {
    return new Promise((resolve, reject) => {
      let stdout = '';
      let stderr = '';

      const child = spawn('docker', args);

      // Set timeout
      const timer = setTimeout(() => {
        child.kill('SIGTERM');
        reject(new Error(`Execution timeout after ${timeout}ms`));
      }, timeout);

      // Collect output
      child.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      // Handle completion
      child.on('close', (code) => {
        clearTimeout(timer);
        resolve({
          exitCode: code,
          stdout,
          stderr
        });
      });

      // Handle errors
      child.on('error', (error) => {
        clearTimeout(timer);
        reject(error);
      });
    });
  }

  /**
   * Test Docker availability
   * @returns {Promise<boolean>}
   */
  static async isDockerAvailable() {
    try {
      const result = await new Promise((resolve, reject) => {
        const child = spawn('docker', ['--version']);
        child.on('close', (code) => resolve(code === 0));
        child.on('error', () => resolve(false));
      });
      return result;
    } catch {
      return false;
    }
  }

  /**
   * Pull required Docker images
   * @returns {Promise<void>}
   */
  static async pullImages(images = ['node:20-alpine', 'python:3.11-alpine']) {
    for (const image of images) {
      console.log(`Pulling Docker image: ${image}`);
      await new Promise((resolve, reject) => {
        const child = spawn('docker', ['pull', image], { stdio: 'inherit' });
        child.on('close', (code) => {
          if (code === 0) resolve();
          else reject(new Error(`Failed to pull image: ${image}`));
        });
      });
    }
  }
}

export default DockerSandbox;
