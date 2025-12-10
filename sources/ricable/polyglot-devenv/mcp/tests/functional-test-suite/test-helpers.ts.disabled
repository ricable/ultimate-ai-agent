import { spawn } from 'child_process';
import { promisify } from 'util';
import path from 'path';

/**
 * Test Helpers and Utilities for DevPod Management
 * 
 * Comprehensive utility functions for:
 * - DevPod workspace lifecycle management
 * - Environment detection and validation
 * - Resource monitoring and cleanup
 * - Test data generation and setup
 * - Performance measurement utilities
 */

export interface DevPodWorkspace {
  name: string;
  environment: string;
  status: 'creating' | 'running' | 'stopped' | 'failed';
  created: Date;
  lastAccessed?: Date;
  resourceUsage?: ResourceMetrics;
}

export interface TestEnvironment {
  name: string;
  type: 'standard' | 'agentic';
  baseImage: string;
  requiredTools: string[];
  optionalFeatures: string[];
  healthChecks: HealthCheck[];
}

export interface HealthCheck {
  name: string;
  command: string;
  expectedOutput?: RegExp;
  timeout: number;
  critical: boolean;
}

export interface ResourceMetrics {
  cpuUsage: number;
  memoryUsageMB: number;
  diskUsageMB: number;
  networkBytesTransferred?: number;
  containerCount?: number;
}

export interface TestContext {
  workspaces: DevPodWorkspace[];
  environment: string;
  testId: string;
  startTime: Date;
  metadata: Record<string, any>;
}

// Environment Configurations
export const TEST_ENVIRONMENTS: Record<string, TestEnvironment> = {
  python: {
    name: 'python',
    type: 'standard',
    baseImage: 'python:3.12',
    requiredTools: ['python', 'uv', 'ruff', 'mypy', 'pytest'],
    optionalFeatures: ['fastapi', 'uvicorn', 'pydantic'],
    healthChecks: [
      {
        name: 'Python Version',
        command: 'python --version',
        expectedOutput: /Python 3\.12/,
        timeout: 5000,
        critical: true
      },
      {
        name: 'UV Package Manager',
        command: 'uv --version',
        timeout: 5000,
        critical: true
      },
      {
        name: 'Ruff Linter',
        command: 'ruff --version',
        timeout: 5000,
        critical: false
      }
    ]
  },
  typescript: {
    name: 'typescript',
    type: 'standard',
    baseImage: 'node:20',
    requiredTools: ['node', 'npm', 'tsc', 'eslint'],
    optionalFeatures: ['jest', 'prettier', 'next'],
    healthChecks: [
      {
        name: 'Node.js Version',
        command: 'node --version',
        expectedOutput: /v20\./,
        timeout: 5000,
        critical: true
      },
      {
        name: 'TypeScript Compiler',
        command: 'tsc --version',
        expectedOutput: /Version/,
        timeout: 5000,
        critical: true
      }
    ]
  },
  'agentic-python': {
    name: 'agentic-python',
    type: 'agentic',
    baseImage: 'python:3.12',
    requiredTools: ['python', 'uv', 'ruff', 'mypy', 'pytest', 'npx'],
    optionalFeatures: ['fastapi', 'uvicorn', 'pydantic', 'claude-flow'],
    healthChecks: [
      {
        name: 'Python Base',
        command: 'python --version',
        expectedOutput: /Python 3\.12/,
        timeout: 5000,
        critical: true
      },
      {
        name: 'Claude-Flow Integration',
        command: 'npx --yes claude-flow@alpha --version',
        timeout: 15000,
        critical: false
      }
    ]
  }
};

// Workspace Management Utilities
export class DevPodManager {
  private workspaces: Map<string, DevPodWorkspace> = new Map();
  private readonly workspacePrefix: string;

  constructor(workspacePrefix: string = 'test-workspace') {
    this.workspacePrefix = workspacePrefix;
  }

  /**
   * Provision a new DevPod workspace
   */
  async provisionWorkspace(
    environment: string,
    options: {
      count?: number;
      namePrefix?: string;
      timeout?: number;
      features?: string[];
    } = {}
  ): Promise<DevPodWorkspace[]> {
    const {
      count = 1,
      namePrefix = this.workspacePrefix,
      timeout = 600000,
      features = []
    } = options;

    const workspaces: DevPodWorkspace[] = [];

    for (let i = 0; i < count; i++) {
      const workspaceName = `${namePrefix}-${environment}-${Date.now()}-${i}`;
      
      const workspace: DevPodWorkspace = {
        name: workspaceName,
        environment,
        status: 'creating',
        created: new Date()
      };

      this.workspaces.set(workspaceName, workspace);
      workspaces.push(workspace);

      try {
        console.log(`📦 Provisioning workspace: ${workspaceName}`);
        
        // Use centralized DevPod management
        const command = `nu ../../host-tooling/devpod-management/manage-devpod.nu provision ${environment}`;
        const result = await this.executeHostCommand(command, timeout);

        if (result.success) {
          workspace.status = 'running';
          workspace.lastAccessed = new Date();
          console.log(`✅ Successfully provisioned: ${workspaceName}`);
        } else {
          workspace.status = 'failed';
          console.error(`❌ Failed to provision: ${workspaceName}`);
        }
      } catch (error) {
        workspace.status = 'failed';
        console.error(`❌ Error provisioning ${workspaceName}:`, error);
      }
    }

    return workspaces;
  }

  /**
   * List all active workspaces
   */
  async listWorkspaces(
    filter: {
      environment?: string;
      status?: DevPodWorkspace['status'];
      namePattern?: RegExp;
    } = {}
  ): Promise<DevPodWorkspace[]> {
    try {
      // Get actual DevPod workspace list
      const result = await this.executeHostCommand('devpod list --output json');
      
      if (result.success) {
        const devpodWorkspaces = JSON.parse(result.output);
        
        // Update internal workspace tracking
        for (const dpWs of devpodWorkspaces) {
          if (this.workspaces.has(dpWs.name)) {
            const workspace = this.workspaces.get(dpWs.name)!;
            workspace.status = this.mapDevPodStatus(dpWs.status);
            workspace.lastAccessed = new Date(dpWs.lastUsed || workspace.lastAccessed);
          }
        }
      }
    } catch (error) {
      console.warn('Failed to sync workspace list:', error);
    }

    // Apply filters
    let filteredWorkspaces = Array.from(this.workspaces.values());

    if (filter.environment) {
      filteredWorkspaces = filteredWorkspaces.filter(ws => ws.environment === filter.environment);
    }

    if (filter.status) {
      filteredWorkspaces = filteredWorkspaces.filter(ws => ws.status === filter.status);
    }

    if (filter.namePattern) {
      filteredWorkspaces = filteredWorkspaces.filter(ws => filter.namePattern!.test(ws.name));
    }

    return filteredWorkspaces;
  }

  /**
   * Execute command in a specific workspace
   */
  async executeInWorkspace(
    workspaceName: string,
    command: string,
    timeout: number = 30000
  ): Promise<{ success: boolean; output: string; error?: string }> {
    const workspace = this.workspaces.get(workspaceName);
    
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceName}`);
    }

    if (workspace.status !== 'running') {
      throw new Error(`Workspace not running: ${workspaceName} (status: ${workspace.status})`);
    }

    const fullCommand = `devpod ssh ${workspaceName} -- ${command}`;
    const result = await this.executeHostCommand(fullCommand, timeout);

    // Update last accessed time
    workspace.lastAccessed = new Date();

    return result;
  }

  /**
   * Run health checks on a workspace
   */
  async runHealthChecks(workspaceName: string): Promise<{
    overall: boolean;
    checks: Array<{ name: string; passed: boolean; output?: string; error?: string }>;
  }> {
    const workspace = this.workspaces.get(workspaceName);
    
    if (!workspace) {
      throw new Error(`Workspace not found: ${workspaceName}`);
    }

    const environment = TEST_ENVIRONMENTS[workspace.environment];
    if (!environment) {
      throw new Error(`Unknown environment: ${workspace.environment}`);
    }

    const checkResults = [];
    let overallSuccess = true;

    for (const healthCheck of environment.healthChecks) {
      try {
        console.log(`  🔍 Running health check: ${healthCheck.name}`);
        
        const result = await this.executeInWorkspace(
          workspaceName,
          healthCheck.command,
          healthCheck.timeout
        );

        const passed = result.success && (
          !healthCheck.expectedOutput || 
          healthCheck.expectedOutput.test(result.output)
        );

        checkResults.push({
          name: healthCheck.name,
          passed,
          output: result.output,
          error: result.error
        });

        if (!passed && healthCheck.critical) {
          overallSuccess = false;
        }

        console.log(`    ${passed ? '✅' : '❌'} ${healthCheck.name}`);
      } catch (error) {
        checkResults.push({
          name: healthCheck.name,
          passed: false,
          error: String(error)
        });

        if (healthCheck.critical) {
          overallSuccess = false;
        }

        console.log(`    ❌ ${healthCheck.name}: ${error}`);
      }
    }

    return {
      overall: overallSuccess,
      checks: checkResults
    };
  }

  /**
   * Measure resource usage for a workspace
   */
  async measureResourceUsage(workspaceName: string): Promise<ResourceMetrics> {
    try {
      // Get memory usage
      const memResult = await this.executeInWorkspace(
        workspaceName,
        "free -m | grep Mem | awk '{print $3}'"
      );

      // Get CPU info
      const cpuResult = await this.executeInWorkspace(
        workspaceName,
        "nproc"
      );

      // Get disk usage
      const diskResult = await this.executeInWorkspace(
        workspaceName,
        "df -m / | tail -1 | awk '{print $3}'"
      );

      const resourceMetrics: ResourceMetrics = {
        cpuUsage: parseInt(cpuResult.output) || 0,
        memoryUsageMB: parseInt(memResult.output) || 0,
        diskUsageMB: parseInt(diskResult.output) || 0
      };

      // Update workspace resource tracking
      const workspace = this.workspaces.get(workspaceName);
      if (workspace) {
        workspace.resourceUsage = resourceMetrics;
      }

      return resourceMetrics;
    } catch (error) {
      console.warn(`Failed to measure resources for ${workspaceName}:`, error);
      return {
        cpuUsage: 0,
        memoryUsageMB: 0,
        diskUsageMB: 0
      };
    }
  }

  /**
   * Clean up workspace
   */
  async cleanupWorkspace(workspaceName: string, force: boolean = false): Promise<boolean> {
    try {
      console.log(`🗑️ Cleaning up workspace: ${workspaceName}`);
      
      const command = `devpod delete ${workspaceName} ${force ? '--force' : ''}`;
      const result = await this.executeHostCommand(command);

      if (result.success) {
        this.workspaces.delete(workspaceName);
        console.log(`✅ Successfully cleaned up: ${workspaceName}`);
        return true;
      } else {
        console.error(`❌ Failed to cleanup: ${workspaceName}`);
        return false;
      }
    } catch (error) {
      console.error(`❌ Error cleaning up ${workspaceName}:`, error);
      return false;
    }
  }

  /**
   * Clean up all tracked workspaces
   */
  async cleanupAllWorkspaces(force: boolean = true): Promise<void> {
    const workspaceNames = Array.from(this.workspaces.keys());
    
    console.log(`🧹 Cleaning up ${workspaceNames.length} workspaces...`);
    
    const cleanupPromises = workspaceNames.map(name => 
      this.cleanupWorkspace(name, force)
    );

    await Promise.allSettled(cleanupPromises);
    
    console.log(`✅ Cleanup completed`);
  }

  /**
   * Get workspace statistics
   */
  getStatistics(): {
    total: number;
    byStatus: Record<string, number>;
    byEnvironment: Record<string, number>;
    totalResourceUsage: ResourceMetrics;
  } {
    const workspaces = Array.from(this.workspaces.values());
    
    const byStatus: Record<string, number> = {};
    const byEnvironment: Record<string, number> = {};
    const totalResourceUsage: ResourceMetrics = {
      cpuUsage: 0,
      memoryUsageMB: 0,
      diskUsageMB: 0
    };

    for (const ws of workspaces) {
      // Count by status
      byStatus[ws.status] = (byStatus[ws.status] || 0) + 1;
      
      // Count by environment
      byEnvironment[ws.environment] = (byEnvironment[ws.environment] || 0) + 1;
      
      // Aggregate resource usage
      if (ws.resourceUsage) {
        totalResourceUsage.cpuUsage += ws.resourceUsage.cpuUsage;
        totalResourceUsage.memoryUsageMB += ws.resourceUsage.memoryUsageMB;
        totalResourceUsage.diskUsageMB += ws.resourceUsage.diskUsageMB;
      }
    }

    totalResourceUsage.containerCount = workspaces.length;

    return {
      total: workspaces.length,
      byStatus,
      byEnvironment,
      totalResourceUsage
    };
  }

  private async executeHostCommand(
    command: string,
    timeout: number = 60000
  ): Promise<{ success: boolean; output: string; error?: string }> {
    return new Promise((resolve) => {
      const child = spawn('bash', ['-c', command], {
        stdio: ['pipe', 'pipe', 'pipe']
      });

      let output = '';
      let error = '';

      child.stdout?.on('data', (data) => {
        output += data.toString();
      });

      child.stderr?.on('data', (data) => {
        error += data.toString();
      });

      child.on('close', (code) => {
        resolve({
          success: code === 0,
          output: output.trim(),
          error: error.trim() || undefined
        });
      });

      // Timeout handling
      setTimeout(() => {
        child.kill();
        resolve({
          success: false,
          output: '',
          error: 'Command timeout'
        });
      }, timeout);
    });
  }

  private mapDevPodStatus(devpodStatus: string): DevPodWorkspace['status'] {
    switch (devpodStatus?.toLowerCase()) {
      case 'running':
        return 'running';
      case 'stopped':
        return 'stopped';
      case 'creating':
        return 'creating';
      default:
        return 'failed';
    }
  }
}

// Test Data Generation Utilities
export class TestDataGenerator {
  
  /**
   * Generate test workspace configurations
   */
  static generateWorkspaceConfigs(
    environments: string[],
    count: number = 1
  ): Array<{ environment: string; name: string; features: string[] }> {
    const configs = [];
    
    for (const env of environments) {
      for (let i = 0; i < count; i++) {
        configs.push({
          environment: env,
          name: `test-${env}-${Date.now()}-${i}`,
          features: this.getEnvironmentFeatures(env)
        });
      }
    }
    
    return configs;
  }

  /**
   * Generate test commands for environment validation
   */
  static generateValidationCommands(environment: string): string[] {
    const commandSets: Record<string, string[]> = {
      python: [
        'python --version',
        'uv --version',
        'python -c "import sys; print(f\'Python {sys.version_info.major}.{sys.version_info.minor}\')"',
        'python -c "import json; print(json.dumps({\'test\': \'data\'}))"'
      ],
      typescript: [
        'node --version',
        'npm --version',
        'node -e "console.log(\'Node.js test successful\')"',
        'npx tsc --version'
      ],
      rust: [
        'rustc --version',
        'cargo --version',
        'echo \'fn main() { println!("Rust test"); }\' > /tmp/test.rs && rustc /tmp/test.rs -o /tmp/test && /tmp/test'
      ],
      go: [
        'go version',
        'echo \'package main; import "fmt"; func main() { fmt.Println("Go test") }\' > /tmp/test.go && cd /tmp && go run test.go'
      ],
      nushell: [
        'nu --version',
        'echo "print \\"Nushell test\\"" | nu'
      ]
    };

    return commandSets[environment] || ['echo "No validation commands for environment"'];
  }

  /**
   * Generate test files for environment validation
   */
  static generateTestFiles(environment: string): Array<{ path: string; content: string }> {
    const fileSets: Record<string, Array<{ path: string; content: string }>> = {
      python: [
        {
          path: '/tmp/test_app.py',
          content: `
def hello():
    return "Hello from Python test"

if __name__ == "__main__":
    print(hello())
`
        },
        {
          path: '/tmp/test_requirements.txt',
          content: 'fastapi>=0.100.0\nuvicorn>=0.20.0\npydantic>=2.0.0'
        }
      ],
      typescript: [
        {
          path: '/tmp/test_app.ts',
          content: `
function hello(): string {
    return "Hello from TypeScript test";
}

console.log(hello());
`
        },
        {
          path: '/tmp/test_package.json',
          content: JSON.stringify({
            name: 'test-app',
            version: '1.0.0',
            scripts: {
              test: 'echo "Test script"'
            }
          }, null, 2)
        }
      ],
      rust: [
        {
          path: '/tmp/test_app.rs',
          content: `
fn main() {
    println!("Hello from Rust test");
}
`
        },
        {
          path: '/tmp/Cargo.toml',
          content: `
[package]
name = "test-app"
version = "0.1.0"
edition = "2021"
`
        }
      ]
    };

    return fileSets[environment] || [];
  }

  private static getEnvironmentFeatures(environment: string): string[] {
    const features: Record<string, string[]> = {
      python: ['fastapi', 'async', 'type-hints'],
      typescript: ['strict-mode', 'es-modules', 'jest'],
      rust: ['tokio', 'serde', 'clippy'],
      go: ['modules', 'generics', 'testing'],
      nushell: ['pipelines', 'structured-data', 'scripting'],
      'agentic-python': ['fastapi', 'async', 'agents', 'claude-flow'],
      'agentic-typescript': ['next.js', 'copilotkit', 'agents', 'claude-flow'],
      'agentic-rust': ['tokio', 'async-traits', 'agents', 'claude-flow'],
      'agentic-go': ['gin', 'microservices', 'agents', 'claude-flow'],
      'agentic-nushell': ['pipelines', 'automation', 'agents', 'claude-flow']
    };

    return features[environment] || [];
  }
}

// Performance Measurement Utilities
export class PerformanceMeasurer {
  private measurements: Map<string, { startTime: number; endTime?: number; metadata?: any }> = new Map();

  /**
   * Start measuring an operation
   */
  startMeasurement(operationId: string, metadata?: any): void {
    this.measurements.set(operationId, {
      startTime: Date.now(),
      metadata
    });
  }

  /**
   * End measuring an operation
   */
  endMeasurement(operationId: string): number | null {
    const measurement = this.measurements.get(operationId);
    
    if (!measurement) {
      return null;
    }

    measurement.endTime = Date.now();
    return measurement.endTime - measurement.startTime;
  }

  /**
   * Get measurement duration
   */
  getDuration(operationId: string): number | null {
    const measurement = this.measurements.get(operationId);
    
    if (!measurement || !measurement.endTime) {
      return null;
    }

    return measurement.endTime - measurement.startTime;
  }

  /**
   * Get all measurements
   */
  getAllMeasurements(): Record<string, { duration: number; metadata?: any }> {
    const results: Record<string, { duration: number; metadata?: any }> = {};
    
    for (const [id, measurement] of this.measurements) {
      if (measurement.endTime) {
        results[id] = {
          duration: measurement.endTime - measurement.startTime,
          metadata: measurement.metadata
        };
      }
    }

    return results;
  }

  /**
   * Generate performance report
   */
  generateReport(): string {
    const measurements = this.getAllMeasurements();
    const operations = Object.keys(measurements);
    
    if (operations.length === 0) {
      return 'No performance measurements recorded.';
    }

    let report = '\n📊 PERFORMANCE MEASUREMENT REPORT\n';
    report += '=' .repeat(50) + '\n';

    const durations = operations.map(op => measurements[op].duration);
    const avgDuration = durations.reduce((sum, d) => sum + d, 0) / durations.length;
    const maxDuration = Math.max(...durations);
    const minDuration = Math.min(...durations);

    report += `Total operations: ${operations.length}\n`;
    report += `Average duration: ${avgDuration.toFixed(2)}ms\n`;
    report += `Maximum duration: ${maxDuration}ms\n`;
    report += `Minimum duration: ${minDuration}ms\n\n`;

    report += 'Individual measurements:\n';
    for (const [operation, data] of Object.entries(measurements)) {
      report += `  ${operation}: ${data.duration}ms\n`;
    }

    report += '=' .repeat(50);
    return report;
  }

  /**
   * Clear all measurements
   */
  clear(): void {
    this.measurements.clear();
  }
}

// Test Context Management
export class TestContextManager {
  private contexts: Map<string, TestContext> = new Map();

  /**
   * Create a new test context
   */
  createContext(
    testId: string,
    environment: string,
    metadata: Record<string, any> = {}
  ): TestContext {
    const context: TestContext = {
      workspaces: [],
      environment,
      testId,
      startTime: new Date(),
      metadata
    };

    this.contexts.set(testId, context);
    return context;
  }

  /**
   * Get test context
   */
  getContext(testId: string): TestContext | undefined {
    return this.contexts.get(testId);
  }

  /**
   * Add workspace to context
   */
  addWorkspaceToContext(testId: string, workspace: DevPodWorkspace): void {
    const context = this.contexts.get(testId);
    if (context) {
      context.workspaces.push(workspace);
    }
  }

  /**
   * Clean up test context
   */
  async cleanupContext(testId: string, devpodManager: DevPodManager): Promise<void> {
    const context = this.contexts.get(testId);
    
    if (context) {
      // Clean up all workspaces in this context
      for (const workspace of context.workspaces) {
        await devpodManager.cleanupWorkspace(workspace.name, true);
      }
      
      this.contexts.delete(testId);
    }
  }

  /**
   * Get all active contexts
   */
  getAllContexts(): TestContext[] {
    return Array.from(this.contexts.values());
  }
}

// Utility Functions
export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function generateRandomId(prefix: string = 'test'): string {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function formatDuration(milliseconds: number): string {
  if (milliseconds < 1000) {
    return `${milliseconds}ms`;
  } else if (milliseconds < 60000) {
    return `${(milliseconds / 1000).toFixed(1)}s`;
  } else {
    return `${(milliseconds / 60000).toFixed(1)}m`;
  }
}

export function formatBytes(bytes: number): string {
  const units = ['B', 'KB', 'MB', 'GB'];
  let size = bytes;
  let unitIndex = 0;
  
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex++;
  }
  
  return `${size.toFixed(1)}${units[unitIndex]}`;
}

export function validateEnvironmentName(environment: string): boolean {
  const validEnvironments = [
    'python', 'typescript', 'rust', 'go', 'nushell',
    'agentic-python', 'agentic-typescript', 'agentic-rust', 
    'agentic-go', 'agentic-nushell'
  ];
  
  return validEnvironments.includes(environment);
}

export function isAgenticEnvironment(environment: string): boolean {
  return environment.startsWith('agentic-');
}

export function getBaseEnvironment(environment: string): string {
  if (isAgenticEnvironment(environment)) {
    return environment.replace('agentic-', '');
  }
  return environment;
}