#!/usr/bin/env node

const axios = require('axios');
const chalk = require('chalk');
const yargs = require('yargs');
const winston = require('winston');
const { table } = require('table');
const yaml = require('js-yaml');
const fs = require('fs').promises;
const path = require('path');
const { spawn } = require('child_process');

// Logger setup
const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.printf(({ timestamp, level, message, ...meta }) => {
      const color = {
        error: chalk.red,
        warn: chalk.yellow,
        info: chalk.blue,
        debug: chalk.gray
      }[level] || chalk.white;
      
      return `${chalk.gray(timestamp)} ${color(level.toUpperCase())} ${message} ${
        Object.keys(meta).length ? JSON.stringify(meta, null, 2) : ''
      }`;
    })
  ),
  transports: [
    new winston.transports.Console(),
    new winston.transports.File({ 
      filename: '/app/test-results/test-runner.log',
      format: winston.format.json()
    })
  ]
});

// Test Runner Class
class MCPTestRunner {
  constructor(options) {
    this.options = {
      suite: options.suite || 'mcp-reliability',
      duration: options.duration || 3600,
      collectMetrics: options.collectMetrics !== false,
      generateReport: options.generateReport !== false,
      ...options
    };
    
    this.results = {
      suite: this.options.suite,
      startTime: new Date().toISOString(),
      endTime: null,
      tests: [],
      metrics: {},
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        skipped: 0
      }
    };
  }

  async runCommand(command, args = []) {
    return new Promise((resolve, reject) => {
      const proc = spawn(command, args, { shell: true });
      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      proc.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      proc.on('close', (code) => {
        if (code === 0) {
          resolve({ stdout, stderr, code });
        } else {
          reject(new Error(`Command failed with code ${code}: ${stderr}`));
        }
      });
    });
  }

  async checkServiceHealth(service, url) {
    try {
      const response = await axios.get(url, { timeout: 5000 });
      return {
        service,
        healthy: response.status === 200,
        status: response.status,
        responseTime: response.headers['x-response-time'] || 'N/A'
      };
    } catch (error) {
      return {
        service,
        healthy: false,
        error: error.message
      };
    }
  }

  async waitForServices() {
    logger.info('Waiting for services to be ready...');
    
    const services = [
      { name: 'mcp-server', url: 'http://mcp-server:3000/health' },
      { name: 'prometheus', url: 'http://prometheus:9090/-/ready' },
      { name: 'grafana', url: 'http://grafana:3000/api/health' },
      { name: 'loki', url: 'http://loki:3100/ready' }
    ];

    const maxAttempts = 30;
    let attempts = 0;

    while (attempts < maxAttempts) {
      const checks = await Promise.all(
        services.map(s => this.checkServiceHealth(s.name, s.url))
      );

      const unhealthy = checks.filter(c => !c.healthy);
      
      if (unhealthy.length === 0) {
        logger.info(chalk.green('All services are ready!'));
        return true;
      }

      logger.warn(`Waiting for ${unhealthy.length} services: ${unhealthy.map(s => s.name).join(', ')}`);
      await new Promise(resolve => setTimeout(resolve, 2000));
      attempts++;
    }

    logger.error('Services failed to become ready');
    return false;
  }

  async runTest(test) {
    logger.info(chalk.cyan(`Running test: ${test.name}`));
    const testResult = {
      name: test.name,
      startTime: new Date().toISOString(),
      endTime: null,
      duration: 0,
      status: 'running',
      steps: [],
      errors: []
    };

    const startTime = Date.now();

    try {
      for (const step of test.steps) {
        const stepResult = await this.executeStep(step);
        testResult.steps.push(stepResult);
        
        if (!stepResult.success) {
          throw new Error(`Step failed: ${step.name || step.action}`);
        }
      }

      testResult.status = 'passed';
      this.results.summary.passed++;
    } catch (error) {
      logger.error(`Test failed: ${test.name}`, error);
      testResult.status = 'failed';
      testResult.errors.push(error.message);
      this.results.summary.failed++;
    } finally {
      testResult.endTime = new Date().toISOString();
      testResult.duration = Date.now() - startTime;
      this.results.tests.push(testResult);
      this.results.summary.total++;
    }

    return testResult;
  }

  async executeStep(step) {
    logger.debug(`Executing step: ${step.action}`);
    const stepResult = {
      action: step.action,
      startTime: new Date().toISOString(),
      success: false,
      result: null,
      error: null
    };

    try {
      switch (step.action) {
        case 'http-request':
          const response = await axios({
            method: step.method || 'GET',
            url: step.url,
            data: step.data,
            headers: step.headers,
            timeout: step.timeout || 30000
          });
          stepResult.result = {
            status: response.status,
            data: response.data
          };
          stepResult.success = true;
          break;

        case 'docker-exec':
          const execResult = await this.runCommand(
            `docker exec ${step.container} ${step.command}`
          );
          stepResult.result = execResult;
          stepResult.success = true;
          break;

        case 'wait':
          await new Promise(resolve => setTimeout(resolve, step.duration || 1000));
          stepResult.success = true;
          break;

        case 'check-metrics':
          const metrics = await this.collectMetrics(step.query);
          stepResult.result = metrics;
          stepResult.success = this.evaluateMetrics(metrics, step.expected);
          break;

        case 'chaos-inject':
          await this.injectChaos(step.type, step.target, step.duration);
          stepResult.success = true;
          break;

        case 'verify-logs':
          const logs = await this.fetchLogs(step.container, step.filter);
          stepResult.result = { logCount: logs.length };
          stepResult.success = this.verifyLogs(logs, step.expected);
          break;

        default:
          throw new Error(`Unknown step action: ${step.action}`);
      }
    } catch (error) {
      stepResult.error = error.message;
      stepResult.success = false;
    }

    return stepResult;
  }

  async collectMetrics(query) {
    try {
      const response = await axios.get('http://prometheus:9090/api/v1/query', {
        params: { query }
      });
      
      return response.data.data.result;
    } catch (error) {
      logger.error('Failed to collect metrics:', error);
      return [];
    }
  }

  evaluateMetrics(metrics, expected) {
    // Simple evaluation logic - can be extended
    if (!expected) return true;
    
    for (const expectation of expected) {
      const metric = metrics.find(m => m.metric.__name__ === expectation.metric);
      if (!metric) return false;
      
      const value = parseFloat(metric.value[1]);
      if (expectation.min !== undefined && value < expectation.min) return false;
      if (expectation.max !== undefined && value > expectation.max) return false;
    }
    
    return true;
  }

  async injectChaos(type, target, duration) {
    logger.info(`Injecting chaos: ${type} on ${target} for ${duration}ms`);
    
    const chaosCommands = {
      'network-delay': `pumba netem --duration ${duration}ms delay --time 100 ${target}`,
      'network-loss': `pumba netem --duration ${duration}ms loss --percent 10 ${target}`,
      'container-pause': `pumba pause --duration ${duration}ms ${target}`,
      'container-kill': `pumba kill ${target}`,
      'container-stop': `pumba stop --duration ${duration}ms ${target}`
    };

    const command = chaosCommands[type];
    if (!command) {
      throw new Error(`Unknown chaos type: ${type}`);
    }

    try {
      await this.runCommand(command);
    } catch (error) {
      // Chaos commands might fail if container is already affected
      logger.warn(`Chaos injection warning: ${error.message}`);
    }
  }

  async fetchLogs(container, filter) {
    try {
      const { stdout } = await this.runCommand(
        `docker logs ${container} --since 5m ${filter ? `2>&1 | grep "${filter}"` : ''}`
      );
      
      return stdout.split('\n').filter(line => line.trim());
    } catch (error) {
      logger.error('Failed to fetch logs:', error);
      return [];
    }
  }

  verifyLogs(logs, expected) {
    if (!expected) return true;
    
    for (const expectation of expected) {
      if (expectation.contains) {
        const found = logs.some(log => log.includes(expectation.contains));
        if (!found) return false;
      }
      
      if (expectation.notContains) {
        const found = logs.some(log => log.includes(expectation.notContains));
        if (found) return false;
      }
      
      if (expectation.count !== undefined) {
        const matches = logs.filter(log => 
          log.includes(expectation.pattern || expectation.contains)
        );
        if (matches.length !== expectation.count) return false;
      }
    }
    
    return true;
  }

  async loadTestSuite() {
    const suitePath = path.join('/app/scenarios', `${this.options.suite}.yaml`);
    
    try {
      const content = await fs.readFile(suitePath, 'utf8');
      return yaml.load(content);
    } catch (error) {
      logger.warn(`Suite file not found: ${suitePath}, using default tests`);
      return this.getDefaultTestSuite();
    }
  }

  getDefaultTestSuite() {
    return {
      name: 'MCP Reliability Test Suite',
      tests: [
        {
          name: 'Basic Connectivity',
          steps: [
            {
              action: 'http-request',
              url: 'http://mcp-server:3000/health',
              expected: { status: 200 }
            }
          ]
        },
        {
          name: 'Connection Stability',
          steps: [
            {
              action: 'http-request',
              url: 'http://mcp-server:3000/health'
            },
            { action: 'wait', duration: 5000 },
            {
              action: 'http-request',
              url: 'http://mcp-server:3000/health'
            },
            {
              action: 'check-metrics',
              query: 'mcp_connections_total',
              expected: [{ metric: 'mcp_connections_total', min: 1 }]
            }
          ]
        },
        {
          name: 'Reconnection After Network Issues',
          steps: [
            {
              action: 'chaos-inject',
              type: 'network-delay',
              target: 'ruv-swarm-mcp-server',
              duration: 10000
            },
            { action: 'wait', duration: 15000 },
            {
              action: 'check-metrics',
              query: 'mcp_reconnect_attempts_total',
              expected: [{ metric: 'mcp_reconnect_attempts_total', min: 1 }]
            }
          ]
        },
        {
          name: 'Container Restart Recovery',
          steps: [
            {
              action: 'docker-exec',
              container: 'ruv-swarm-mcp-server',
              command: 'kill 1'
            },
            { action: 'wait', duration: 10000 },
            {
              action: 'http-request',
              url: 'http://mcp-server:3000/health',
              expected: { status: 200 }
            }
          ]
        },
        {
          name: 'High Load Performance',
          steps: [
            {
              action: 'check-metrics',
              query: 'rate(mcp_requests_total[5m])',
              expected: [{ metric: 'mcp_requests_total', min: 10 }]
            },
            {
              action: 'verify-logs',
              container: 'ruv-swarm-mcp-server',
              filter: 'error',
              expected: [{ notContains: 'OOM' }, { notContains: 'panic' }]
            }
          ]
        }
      ]
    };
  }

  async generateReport() {
    logger.info('Generating test report...');
    
    const report = {
      ...this.results,
      endTime: new Date().toISOString(),
      duration: Date.now() - new Date(this.results.startTime).getTime(),
      environment: {
        suite: this.options.suite,
        docker: await this.getDockerInfo(),
        services: await this.getServiceVersions()
      }
    };

    // Create summary table
    const summaryTable = table([
      ['Metric', 'Value'],
      ['Total Tests', report.summary.total],
      ['Passed', chalk.green(report.summary.passed)],
      ['Failed', chalk.red(report.summary.failed)],
      ['Skipped', chalk.yellow(report.summary.skipped)],
      ['Success Rate', `${((report.summary.passed / report.summary.total) * 100).toFixed(2)}%`],
      ['Duration', `${(report.duration / 1000).toFixed(2)}s`]
    ]);

    // Create detailed results table
    const resultsData = [['Test Name', 'Status', 'Duration', 'Errors']];
    
    for (const test of report.tests) {
      const status = test.status === 'passed' 
        ? chalk.green('✓ PASSED') 
        : chalk.red('✗ FAILED');
      
      resultsData.push([
        test.name,
        status,
        `${(test.duration / 1000).toFixed(2)}s`,
        test.errors.join(', ') || '-'
      ]);
    }
    
    const resultsTable = table(resultsData);

    // Print report to console
    console.log('\n' + chalk.bold.cyan('MCP Reliability Test Report'));
    console.log(chalk.gray('='.repeat(50)));
    console.log('\n' + chalk.bold('Summary:'));
    console.log(summaryTable);
    console.log('\n' + chalk.bold('Test Results:'));
    console.log(resultsTable);

    // Save detailed report
    const reportPath = `/app/reports/test-report-${Date.now()}.json`;
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    logger.info(`Detailed report saved to: ${reportPath}`);

    // Save markdown report
    const markdownReport = this.generateMarkdownReport(report);
    const mdPath = `/app/reports/test-report-${Date.now()}.md`;
    await fs.writeFile(mdPath, markdownReport);
    logger.info(`Markdown report saved to: ${mdPath}`);

    return report;
  }

  async getDockerInfo() {
    try {
      const { stdout } = await this.runCommand('docker version --format json');
      return JSON.parse(stdout);
    } catch (error) {
      return { error: error.message };
    }
  }

  async getServiceVersions() {
    const services = {};
    
    try {
      // Get MCP server version
      const mcpResponse = await axios.get('http://mcp-server:3000/version').catch(() => null);
      if (mcpResponse) {
        services.mcpServer = mcpResponse.data;
      }

      // Get other service info
      services.prometheus = await this.runCommand('docker exec ruv-swarm-prometheus prometheus --version')
        .then(r => r.stdout.trim())
        .catch(() => 'unknown');
      
      services.grafana = await axios.get('http://grafana:3000/api/health')
        .then(r => r.data)
        .catch(() => 'unknown');
      
    } catch (error) {
      logger.error('Failed to get service versions:', error);
    }

    return services;
  }

  generateMarkdownReport(report) {
    const successRate = ((report.summary.passed / report.summary.total) * 100).toFixed(2);
    
    return `# MCP Reliability Test Report

## Summary
- **Suite**: ${report.suite}
- **Start Time**: ${report.startTime}
- **End Time**: ${report.endTime}
- **Duration**: ${(report.duration / 1000).toFixed(2)}s
- **Success Rate**: ${successRate}%

## Test Results

| Test | Status | Duration | Errors |
|------|--------|----------|--------|
${report.tests.map(test => 
  `| ${test.name} | ${test.status.toUpperCase()} | ${(test.duration / 1000).toFixed(2)}s | ${test.errors.join(', ') || '-'} |`
).join('\n')}

## Environment

### Docker
\`\`\`json
${JSON.stringify(report.environment.docker, null, 2)}
\`\`\`

### Services
\`\`\`json
${JSON.stringify(report.environment.services, null, 2)}
\`\`\`

## Detailed Test Results

${report.tests.map(test => `
### ${test.name}
- **Status**: ${test.status}
- **Duration**: ${(test.duration / 1000).toFixed(2)}s
- **Steps**: ${test.steps.length}

${test.errors.length > 0 ? `
#### Errors
${test.errors.map(e => `- ${e}`).join('\n')}
` : ''}

#### Steps
${test.steps.map((step, i) => 
  `${i + 1}. **${step.action}**: ${step.success ? '✓' : '✗'} ${step.error || ''}`
).join('\n')}
`).join('\n')}
`;
  }

  async run() {
    logger.info(chalk.bold.cyan('MCP Reliability Test Runner'));
    logger.info(`Suite: ${this.options.suite}`);
    logger.info(`Duration: ${this.options.duration}s`);

    // Wait for services
    const servicesReady = await this.waitForServices();
    if (!servicesReady) {
      logger.error('Aborting: Services not ready');
      process.exit(1);
    }

    // Load test suite
    const suite = await this.loadTestSuite();
    logger.info(`Loaded test suite: ${suite.name}`);

    // Run tests
    const endTime = Date.now() + (this.options.duration * 1000);
    let iteration = 0;

    while (Date.now() < endTime) {
      iteration++;
      logger.info(chalk.cyan(`\nStarting iteration ${iteration}`));

      for (const test of suite.tests) {
        await this.runTest(test);
        
        // Check if we've exceeded duration
        if (Date.now() >= endTime) break;
        
        // Wait between tests
        await new Promise(resolve => setTimeout(resolve, 5000));
      }

      // Wait between iterations
      if (Date.now() < endTime) {
        logger.info('Waiting before next iteration...');
        await new Promise(resolve => setTimeout(resolve, 30000));
      }
    }

    // Generate report
    if (this.options.generateReport) {
      await this.generateReport();
    }

    // Exit with appropriate code
    const exitCode = this.results.summary.failed > 0 ? 1 : 0;
    logger.info(`Test run completed. Exit code: ${exitCode}`);
    process.exit(exitCode);
  }
}

// CLI setup
const argv = yargs
  .option('suite', {
    alias: 's',
    description: 'Test suite to run',
    default: 'mcp-reliability'
  })
  .option('duration', {
    alias: 'd',
    description: 'Test duration in seconds',
    type: 'number',
    default: 3600
  })
  .option('collect-metrics', {
    alias: 'm',
    description: 'Collect metrics during test',
    type: 'boolean',
    default: true
  })
  .option('generate-report', {
    alias: 'r',
    description: 'Generate test report',
    type: 'boolean',
    default: true
  })
  .help()
  .argv;

// Main execution
async function main() {
  const runner = new MCPTestRunner(argv);
  await runner.run();
}

if (require.main === module) {
  main().catch(error => {
    logger.error('Fatal error:', error);
    process.exit(1);
  });
}