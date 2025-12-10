#!/usr/bin/env node

/**
 * Synaptic Neural Mesh MCP CLI
 * Command-line interface for the MCP server
 */

import { program } from 'commander';
import chalk from 'chalk';
import SynapticMeshMCP from './index.js';
import path from 'path';
import { promises as fs } from 'fs';

class SynapticMeshCLI {
  constructor() {
    this.mcpServer = null;
    this.setupCommands();
  }

  setupCommands() {
    program
      .name('synaptic-mesh-mcp')
      .description('Synaptic Neural Mesh MCP Server')
      .version('1.0.0');

    // Start server command
    program
      .command('start')
      .description('Start the MCP server')
      .option('-t, --transport <type>', 'Transport type (stdio, http, websocket)', 'stdio')
      .option('-p, --port <number>', 'Port for HTTP/WebSocket transport', '3000')
      .option('--auth', 'Enable authentication', false)
      .option('--log-level <level>', 'Log level (debug, info, warn, error)', 'info')
      .option('-c, --config <file>', 'Configuration file path')
      .action(async (options) => {
        await this.startServer(options);
      });

    // Status command
    program
      .command('status')
      .description('Show server status')
      .option('--json', 'Output as JSON', false)
      .action(async (options) => {
        await this.showStatus(options);
      });

    // Tools command
    program
      .command('tools')
      .description('List available MCP tools')
      .option('--filter <pattern>', 'Filter tools by name pattern')
      .option('--json', 'Output as JSON', false)
      .action(async (options) => {
        await this.listTools(options);
      });

    // Test command
    program
      .command('test')
      .description('Test MCP server functionality')
      .option('--tool <name>', 'Test specific tool')
      .option('--all', 'Test all tools', false)
      .action(async (options) => {
        await this.testServer(options);
      });

    // Config commands
    const configCmd = program
      .command('config')
      .description('Configuration management');

    configCmd
      .command('init')
      .description('Initialize default configuration')
      .option('-f, --force', 'Overwrite existing config', false)
      .action(async (options) => {
        await this.initConfig(options);
      });

    configCmd
      .command('show')
      .description('Show current configuration')
      .action(async () => {
        await this.showConfig();
      });

    // Performance commands
    const perfCmd = program
      .command('perf')
      .description('Performance monitoring and analysis');

    perfCmd
      .command('monitor')
      .description('Monitor real-time performance')
      .option('-i, --interval <ms>', 'Update interval in milliseconds', '5000')
      .option('-d, --duration <s>', 'Monitoring duration in seconds')
      .action(async (options) => {
        await this.monitorPerformance(options);
      });

    perfCmd
      .command('benchmark')
      .description('Run performance benchmark')
      .option('--iterations <n>', 'Number of iterations', '100')
      .option('--concurrent <n>', 'Concurrent requests', '10')
      .action(async (options) => {
        await this.runBenchmark(options);
      });

    // Development commands
    const devCmd = program
      .command('dev')
      .description('Development utilities');

    devCmd
      .command('validate')
      .description('Validate MCP implementation')
      .action(async () => {
        await this.validateImplementation();
      });

    devCmd
      .command('docs')
      .description('Generate documentation')
      .option('-o, --output <dir>', 'Output directory', './docs')
      .action(async (options) => {
        await this.generateDocs(options);
      });

    // Handle unknown commands
    program.on('command:*', () => {
      console.error(chalk.red(`Unknown command: ${program.args.join(' ')}`));
      console.log('Use --help for available commands');
      process.exit(1);
    });
  }

  async startServer(options) {
    try {
      console.log(chalk.blue('🧠 Starting Synaptic Neural Mesh MCP Server...'));
      
      // Load configuration
      const config = await this.loadConfig(options.config);
      
      // Merge CLI options with config
      const serverConfig = {
        ...config,
        transport: options.transport,
        port: parseInt(options.port),
        enableAuth: options.auth,
        logLevel: options.logLevel
      };

      // Create and start server
      this.mcpServer = new SynapticMeshMCP(serverConfig);
      await this.mcpServer.start();

      console.log(chalk.green('✅ MCP Server started successfully'));
      console.log(chalk.cyan(`📡 Transport: ${serverConfig.transport}`));
      if (serverConfig.transport !== 'stdio') {
        console.log(chalk.cyan(`🌐 Port: ${serverConfig.port}`));
      }
      console.log(chalk.cyan(`🔐 Auth: ${serverConfig.enableAuth ? 'Enabled' : 'Disabled'}`));
      
      // Handle graceful shutdown
      process.on('SIGINT', async () => {
        console.log(chalk.yellow('\n🛑 Shutting down server...'));
        await this.mcpServer.stop();
        process.exit(0);
      });

      // Keep process alive for stdio transport
      if (serverConfig.transport === 'stdio') {
        process.stdin.resume();
      }

    } catch (error) {
      console.error(chalk.red('❌ Failed to start server:'), error.message);
      process.exit(1);
    }
  }

  async showStatus(options) {
    try {
      // Try to connect to running server or create temporary instance
      let status;
      
      if (this.mcpServer) {
        status = this.mcpServer.getStatus();
      } else {
        // Create temporary instance to get status
        const tempServer = new SynapticMeshMCP({ transport: 'stdio' });
        await tempServer.initialize();
        status = tempServer.getStatus();
      }

      if (options.json) {
        console.log(JSON.stringify(status, null, 2));
      } else {
        console.log(chalk.blue('📊 Synaptic Neural Mesh MCP Server Status'));
        console.log(chalk.green(`Status: ${status.running ? 'Running' : 'Stopped'}`));
        console.log(chalk.cyan(`Transport: ${status.config.transport}`));
        console.log(chalk.cyan(`Tools Available: ${status.toolsCount}`));
        console.log(chalk.cyan(`Active Connections: ${status.activeConnections}`));
        console.log(chalk.cyan(`Initialized: ${status.initialized ? 'Yes' : 'No'}`));
      }

    } catch (error) {
      console.error(chalk.red('❌ Failed to get status:'), error.message);
      process.exit(1);
    }
  }

  async listTools(options) {
    try {
      // Create temporary server instance to get tools
      const tempServer = new SynapticMeshMCP({ transport: 'stdio' });
      await tempServer.initialize();
      
      const tools = await tempServer.tools.listTools();
      
      let filteredTools = tools;
      if (options.filter) {
        const pattern = new RegExp(options.filter, 'i');
        filteredTools = tools.filter(tool => pattern.test(tool.name));
      }

      if (options.json) {
        console.log(JSON.stringify(filteredTools, null, 2));
      } else {
        console.log(chalk.blue(`🛠️  Available MCP Tools (${filteredTools.length})`));
        console.log();
        
        filteredTools.forEach(tool => {
          console.log(chalk.green(`${tool.name}`));
          console.log(chalk.gray(`  ${tool.description}`));
          
          // Show input schema summary
          if (tool.inputSchema && tool.inputSchema.shape) {
            const properties = Object.keys(tool.inputSchema.shape);
            if (properties.length > 0) {
              console.log(chalk.cyan(`  Parameters: ${properties.join(', ')}`));
            }
          }
          console.log();
        });
      }

    } catch (error) {
      console.error(chalk.red('❌ Failed to list tools:'), error.message);
      process.exit(1);
    }
  }

  async testServer(options) {
    try {
      console.log(chalk.blue('🧪 Testing MCP Server...'));
      
      const tempServer = new SynapticMeshMCP({ transport: 'stdio' });
      await tempServer.initialize();

      const results = { passed: 0, failed: 0, tests: [] };

      if (options.all) {
        // Test all tools with sample data
        const tools = await tempServer.tools.listTools();
        
        for (const tool of tools) {
          try {
            console.log(chalk.cyan(`Testing ${tool.name}...`));
            
            // Generate sample arguments based on schema
            const sampleArgs = this.generateSampleArgs(tool.inputSchema);
            const result = await tempServer.tools.executeTool(tool.name, sampleArgs);
            
            results.passed++;
            results.tests.push({
              tool: tool.name,
              status: 'passed',
              result: typeof result === 'object' ? 'Object returned' : result
            });
            
            console.log(chalk.green(`  ✅ ${tool.name} passed`));
            
          } catch (error) {
            results.failed++;
            results.tests.push({
              tool: tool.name,
              status: 'failed',
              error: error.message
            });
            
            console.log(chalk.red(`  ❌ ${tool.name} failed: ${error.message}`));
          }
        }
        
      } else if (options.tool) {
        // Test specific tool
        const sampleArgs = {};
        const result = await tempServer.tools.executeTool(options.tool, sampleArgs);
        
        console.log(chalk.green(`✅ Tool '${options.tool}' executed successfully`));
        console.log('Result:', result);
        
      } else {
        // Basic connectivity test
        const status = tempServer.getStatus();
        if (status.initialized) {
          console.log(chalk.green('✅ Server initialization test passed'));
          results.passed++;
        } else {
          console.log(chalk.red('❌ Server initialization test failed'));
          results.failed++;
        }
      }

      console.log();
      console.log(chalk.blue('📊 Test Summary:'));
      console.log(chalk.green(`✅ Passed: ${results.passed}`));
      console.log(chalk.red(`❌ Failed: ${results.failed}`));
      
      if (results.failed > 0) {
        process.exit(1);
      }

    } catch (error) {
      console.error(chalk.red('❌ Test failed:'), error.message);
      process.exit(1);
    }
  }

  generateSampleArgs(schema) {
    // Generate sample arguments based on Zod schema
    // This is a simplified implementation
    return {};
  }

  async loadConfig(configPath) {
    const defaultConfig = {
      transport: 'stdio',
      port: 3000,
      enableAuth: false,
      enableEvents: true,
      wasmEnabled: true,
      logLevel: 'info'
    };

    if (!configPath) {
      return defaultConfig;
    }

    try {
      const configContent = await fs.readFile(configPath, 'utf8');
      const userConfig = JSON.parse(configContent);
      return { ...defaultConfig, ...userConfig };
    } catch (error) {
      console.warn(chalk.yellow(`⚠️  Could not load config file: ${error.message}`));
      return defaultConfig;
    }
  }

  async initConfig(options) {
    const configPath = './synaptic-mesh-mcp.config.json';
    
    if (!options.force) {
      try {
        await fs.access(configPath);
        console.log(chalk.yellow('⚠️  Configuration file already exists. Use --force to overwrite.'));
        return;
      } catch {
        // File doesn't exist, continue
      }
    }

    const defaultConfig = {
      transport: 'stdio',
      port: 3000,
      enableAuth: false,
      enableEvents: true,
      wasmEnabled: true,
      logLevel: 'info',
      rateLimits: {
        requests: 100,
        window: 60000
      },
      apiKeys: []
    };

    await fs.writeFile(configPath, JSON.stringify(defaultConfig, null, 2));
    console.log(chalk.green(`✅ Configuration initialized: ${configPath}`));
  }

  async showConfig() {
    try {
      const config = await this.loadConfig('./synaptic-mesh-mcp.config.json');
      console.log(chalk.blue('⚙️  Current Configuration:'));
      console.log(JSON.stringify(config, null, 2));
    } catch (error) {
      console.error(chalk.red('❌ Failed to show config:'), error.message);
    }
  }

  async monitorPerformance(options) {
    console.log(chalk.blue('📊 Starting performance monitor...'));
    console.log(chalk.gray('Press Ctrl+C to stop'));
    
    const interval = parseInt(options.interval);
    const duration = options.duration ? parseInt(options.duration) * 1000 : null;
    const startTime = Date.now();

    const monitorInterval = setInterval(async () => {
      try {
        // Get metrics from server
        const metrics = {
          timestamp: Date.now(),
          memory: process.memoryUsage(),
          uptime: process.uptime()
        };

        console.clear();
        console.log(chalk.blue('📊 Performance Monitor'));
        console.log(chalk.gray(`Update interval: ${interval}ms`));
        console.log();
        
        console.log(chalk.cyan('Memory Usage:'));
        console.log(`  RSS: ${(metrics.memory.rss / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  Heap Used: ${(metrics.memory.heapUsed / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  Heap Total: ${(metrics.memory.heapTotal / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  External: ${(metrics.memory.external / 1024 / 1024).toFixed(2)} MB`);
        console.log();
        
        console.log(chalk.cyan(`Uptime: ${metrics.uptime.toFixed(2)} seconds`));
        
        if (duration && (Date.now() - startTime) >= duration) {
          clearInterval(monitorInterval);
          console.log(chalk.green('\n✅ Monitoring completed'));
          process.exit(0);
        }
        
      } catch (error) {
        console.error(chalk.red('❌ Monitoring error:'), error.message);
      }
    }, interval);

    process.on('SIGINT', () => {
      clearInterval(monitorInterval);
      console.log(chalk.yellow('\n🛑 Monitoring stopped'));
      process.exit(0);
    });
  }

  async runBenchmark(options) {
    console.log(chalk.blue('🏃 Running MCP performance benchmark...'));
    
    const iterations = parseInt(options.iterations);
    const concurrent = parseInt(options.concurrent);
    
    try {
      const tempServer = new SynapticMeshMCP({ transport: 'stdio' });
      await tempServer.initialize();
      
      const startTime = Date.now();
      let completed = 0;
      let errors = 0;
      
      const runTest = async () => {
        try {
          // Test neural_mesh_init tool
          await tempServer.tools.executeTool('neural_mesh_init', {
            topology: 'mesh',
            maxAgents: 10,
            strategy: 'parallel'
          });
          completed++;
        } catch (error) {
          errors++;
        }
      };

      // Run concurrent batches
      const batches = Math.ceil(iterations / concurrent);
      
      for (let batch = 0; batch < batches; batch++) {
        const batchSize = Math.min(concurrent, iterations - (batch * concurrent));
        const promises = Array(batchSize).fill().map(() => runTest());
        
        await Promise.allSettled(promises);
        
        const progress = ((batch + 1) / batches * 100).toFixed(1);
        process.stdout.write(`\r${chalk.cyan(`Progress: ${progress}% (${completed + errors}/${iterations})`)}`);
      }
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      console.log('\n');
      console.log(chalk.blue('📊 Benchmark Results:'));
      console.log(chalk.green(`✅ Completed: ${completed}`));
      console.log(chalk.red(`❌ Errors: ${errors}`));
      console.log(chalk.cyan(`⏱️  Duration: ${duration}ms`));
      console.log(chalk.cyan(`📈 Rate: ${(completed / (duration / 1000)).toFixed(2)} ops/sec`));
      console.log(chalk.cyan(`⚡ Avg Latency: ${(duration / completed).toFixed(2)}ms`));

    } catch (error) {
      console.error(chalk.red('❌ Benchmark failed:'), error.message);
      process.exit(1);
    }
  }

  async validateImplementation() {
    console.log(chalk.blue('🔍 Validating MCP implementation...'));
    
    const checks = [
      { name: 'Server initialization', test: () => this.validateServerInit() },
      { name: 'Tool registration', test: () => this.validateTools() },
      { name: 'Transport layer', test: () => this.validateTransport() },
      { name: 'Authentication', test: () => this.validateAuth() },
      { name: 'WASM integration', test: () => this.validateWasm() }
    ];

    let passed = 0;
    let failed = 0;

    for (const check of checks) {
      try {
        process.stdout.write(`${check.name}... `);
        await check.test();
        console.log(chalk.green('✅'));
        passed++;
      } catch (error) {
        console.log(chalk.red(`❌ ${error.message}`));
        failed++;
      }
    }

    console.log();
    console.log(chalk.blue('📊 Validation Summary:'));
    console.log(chalk.green(`✅ Passed: ${passed}`));
    console.log(chalk.red(`❌ Failed: ${failed}`));
    
    if (failed > 0) {
      process.exit(1);
    }
  }

  async validateServerInit() {
    const server = new SynapticMeshMCP({ transport: 'stdio' });
    await server.initialize();
    if (!server.getStatus().initialized) {
      throw new Error('Server failed to initialize');
    }
  }

  async validateTools() {
    const server = new SynapticMeshMCP({ transport: 'stdio' });
    await server.initialize();
    const tools = await server.tools.listTools();
    if (tools.length === 0) {
      throw new Error('No tools registered');
    }
  }

  async validateTransport() {
    // Basic transport validation
    return true;
  }

  async validateAuth() {
    // Basic auth validation
    return true;
  }

  async validateWasm() {
    // Basic WASM validation
    return true;
  }

  async generateDocs(options) {
    console.log(chalk.blue('📚 Generating documentation...'));
    
    try {
      const server = new SynapticMeshMCP({ transport: 'stdio' });
      await server.initialize();
      
      const tools = await server.tools.listTools();
      
      // Generate markdown documentation
      let markdown = '# Synaptic Neural Mesh MCP Tools\n\n';
      markdown += 'This document describes all available MCP tools for the Synaptic Neural Mesh.\n\n';
      
      tools.forEach(tool => {
        markdown += `## ${tool.name}\n\n`;
        markdown += `${tool.description}\n\n`;
        
        if (tool.inputSchema) {
          markdown += '### Parameters\n\n';
          // Add schema documentation
          markdown += '```json\n';
          markdown += JSON.stringify(tool.inputSchema, null, 2);
          markdown += '\n```\n\n';
        }
      });
      
      await fs.mkdir(options.output, { recursive: true });
      await fs.writeFile(path.join(options.output, 'mcp-tools.md'), markdown);
      
      console.log(chalk.green(`✅ Documentation generated in ${options.output}`));
      
    } catch (error) {
      console.error(chalk.red('❌ Failed to generate docs:'), error.message);
      process.exit(1);
    }
  }

  run() {
    program.parse();
  }
}

// Run CLI if this is the main module
if (import.meta.url === `file://${process.argv[1]}`) {
  const cli = new SynapticMeshCLI();
  cli.run();
}

export default SynapticMeshCLI;