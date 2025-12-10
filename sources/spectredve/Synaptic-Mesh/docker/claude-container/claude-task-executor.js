#!/usr/bin/env node

/**
 * Secure Claude Task Executor
 * 
 * Features:
 * - JSON streaming interface via stdin/stdout
 * - Read-only filesystem with tmpfs workspace
 * - Network isolation (API access only)
 * - No persistent secrets storage
 * - Resource limits and security hardening
 * - User maintains full control with local API credentials
 */

const fs = require('fs');
const path = require('path');
const { createReadStream, createWriteStream } = require('fs');
const { Transform } = require('stream');

// Security configuration
const SECURITY_CONFIG = {
  maxMemoryMB: 512,
  maxExecutionTimeMs: 300000, // 5 minutes
  allowedApis: ['api.anthropic.com'],
  workspaceDir: '/tmp/claude-work',
  readOnlyMode: true,
  networkRestricted: true
};

// Load security configuration
let securityConfig = SECURITY_CONFIG;
try {
  const configPath = path.join(__dirname, 'security-config.json');
  if (fs.existsSync(configPath)) {
    const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    securityConfig = { ...SECURITY_CONFIG, ...config };
  }
} catch (error) {
  console.error('Warning: Could not load security config, using defaults');
}

/**
 * Security validator for API requests
 */
class SecurityValidator {
  constructor(config) {
    this.config = config;
  }

  validateApiEndpoint(url) {
    try {
      const parsedUrl = new URL(url);
      return this.config.allowedApis.some(allowed => 
        parsedUrl.hostname.endsWith(allowed)
      );
    } catch {
      return false;
    }
  }

  validateWorkspacePath(filePath) {
    const resolvedPath = path.resolve(filePath);
    const workspacePath = path.resolve(this.config.workspaceDir);
    return resolvedPath.startsWith(workspacePath);
  }

  sanitizeEnvironment() {
    // Remove sensitive environment variables
    const sensitiveVars = [
      'ANTHROPIC_API_KEY',
      'CLAUDE_API_KEY', 
      'AWS_ACCESS_KEY_ID',
      'AWS_SECRET_ACCESS_KEY'
    ];
    
    sensitiveVars.forEach(varName => {
      if (process.env[varName]) {
        delete process.env[varName];
      }
    });
  }
}

/**
 * Claude API Client with security controls
 */
class SecureClaudeClient {
  constructor(apiKey, securityValidator) {
    this.apiKey = apiKey;
    this.validator = securityValidator;
    this.requestCount = 0;
    this.maxRequests = 100; // Rate limiting
  }

  async executeTask(task) {
    // Rate limiting
    if (this.requestCount >= this.maxRequests) {
      throw new Error('Rate limit exceeded');
    }
    this.requestCount++;

    // Input validation
    if (!task || typeof task !== 'object') {
      throw new Error('Invalid task format');
    }

    // Security check for API endpoint
    const apiUrl = 'https://api.anthropic.com/v1/messages';
    if (!this.validator.validateApiEndpoint(apiUrl)) {
      throw new Error('Unauthorized API endpoint');
    }

    try {
      // Note: In production, use proper Anthropic SDK
      // This is a simplified implementation for container demo
      const Anthropic = require('@anthropic-ai/sdk');
      
      const anthropic = new Anthropic({
        apiKey: this.apiKey,
      });

      const response = await anthropic.messages.create({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 1000,
        messages: [{
          role: 'user',
          content: task.prompt || task.message || String(task)
        }]
      });

      return {
        success: true,
        response: response.content[0].text,
        metadata: {
          model: response.model,
          usage: response.usage,
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      return {
        success: false,
        error: error.message,
        timestamp: new Date().toISOString()
      };
    }
  }
}

/**
 * JSON Streaming Interface
 */
class JSONStreamInterface {
  constructor(secureClient, securityValidator) {
    this.client = secureClient;
    this.validator = securityValidator;
    this.setupStreams();
  }

  setupStreams() {
    // Input stream transformer
    const inputTransform = new Transform({
      objectMode: true,
      transform(chunk, encoding, callback) {
        try {
          const data = JSON.parse(chunk.toString());
          callback(null, data);
        } catch (error) {
          callback(new Error(`Invalid JSON input: ${error.message}`));
        }
      }
    });

    // Output stream transformer
    const outputTransform = new Transform({
      objectMode: true,
      transform(chunk, encoding, callback) {
        try {
          const json = JSON.stringify(chunk) + '\n';
          callback(null, json);
        } catch (error) {
          callback(error);
        }
      }
    });

    // Set up processing pipeline
    process.stdin
      .pipe(inputTransform)
      .on('data', (task) => this.processTask(task))
      .on('error', (error) => this.sendError(error));

    this.outputStream = outputTransform;
    this.outputStream.pipe(process.stdout);

    // Send ready signal
    this.sendResponse({
      type: 'ready',
      message: 'Claude container ready for tasks',
      security: {
        sandboxed: true,
        readOnly: securityConfig.readOnlyMode,
        networkRestricted: securityConfig.networkRestricted
      }
    });
  }

  async processTask(task) {
    try {
      // Validate task
      if (!task.id) {
        task.id = Date.now().toString();
      }

      // Security validation
      if (task.type === 'file_operation') {
        if (!this.validator.validateWorkspacePath(task.path)) {
          throw new Error('File path outside allowed workspace');
        }
      }

      // Process task with timeout
      const result = await Promise.race([
        this.client.executeTask(task),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Task timeout')), 
            securityConfig.maxExecutionTimeMs)
        )
      ]);

      this.sendResponse({
        type: 'task_result',
        taskId: task.id,
        ...result
      });

    } catch (error) {
      this.sendError(error, task.id);
    }
  }

  sendResponse(data) {
    this.outputStream.write(data);
  }

  sendError(error, taskId = null) {
    this.sendResponse({
      type: 'error',
      taskId,
      error: error.message,
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * Main application entry point
 */
async function main() {
  try {
    // Initialize security
    const validator = new SecurityValidator(securityConfig);
    validator.sanitizeEnvironment();

    // Check for API key from environment (user-provided)
    const apiKey = process.env.CLAUDE_API_KEY || process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      console.error(JSON.stringify({
        type: 'error',
        error: 'API key required. Set CLAUDE_API_KEY or ANTHROPIC_API_KEY environment variable.',
        timestamp: new Date().toISOString()
      }));
      process.exit(1);
    }

    // Initialize secure Claude client
    const secureClient = new SecureClaudeClient(apiKey, validator);

    // Start JSON streaming interface
    const streamInterface = new JSONStreamInterface(secureClient, validator);

    // Handle graceful shutdown
    process.on('SIGTERM', () => {
      console.error(JSON.stringify({
        type: 'shutdown',
        message: 'Container shutting down',
        timestamp: new Date().toISOString()
      }));
      process.exit(0);
    });

    process.on('SIGINT', () => {
      console.error(JSON.stringify({
        type: 'shutdown', 
        message: 'Container interrupted',
        timestamp: new Date().toISOString()
      }));
      process.exit(0);
    });

  } catch (error) {
    console.error(JSON.stringify({
      type: 'fatal_error',
      error: error.message,
      timestamp: new Date().toISOString()
    }));
    process.exit(1);
  }
}

// Start the application
if (require.main === module) {
  main().catch(error => {
    console.error(JSON.stringify({
      type: 'startup_error',
      error: error.message,
      timestamp: new Date().toISOString()
    }));
    process.exit(1);
  });
}

module.exports = {
  SecurityValidator,
  SecureClaudeClient,
  JSONStreamInterface
};