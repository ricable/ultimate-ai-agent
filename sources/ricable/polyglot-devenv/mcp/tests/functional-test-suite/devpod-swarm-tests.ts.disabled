import { describe, test, expect, beforeAll, afterAll, jest } from '@jest/globals';
import { spawn } from 'child_process';
import { promisify } from 'util';
import path from 'path';

/**
 * DevPod Swarm Tests
 * 
 * Comprehensive testing of DevPod environment provisioning and management
 * across all 10 environments (5 standard + 5 agentic variants).
 */

const execAsync = promisify(spawn);

interface DevPodEnvironment {
  name: string;
  type: 'standard' | 'agentic';
  expectedTools: string[];
  expectedLibraries: string[];
  validationCommands: string[];
}

const TEST_ENVIRONMENTS: DevPodEnvironment[] = [
  // Standard Environments
  {
    name: 'python',
    type: 'standard',
    expectedTools: ['python', 'uv', 'ruff', 'mypy', 'pytest'],
    expectedLibraries: ['fastapi', 'uvicorn', 'pydantic'],
    validationCommands: [
      'python --version',
      'uv --version',
      'ruff --version',
      'mypy --version',
      'pytest --version'
    ]
  },
  {
    name: 'typescript',
    type: 'standard',
    expectedTools: ['node', 'npm', 'tsc', 'eslint', 'jest'],
    expectedLibraries: ['typescript', '@types/node', 'next'],
    validationCommands: [
      'node --version',
      'npm --version',
      'tsc --version',
      'npx eslint --version',
      'npx jest --version'
    ]
  },
  {
    name: 'rust',
    type: 'standard',
    expectedTools: ['rustc', 'cargo', 'rustfmt', 'clippy'],
    expectedLibraries: ['tokio', 'serde', 'clap'],
    validationCommands: [
      'rustc --version',
      'cargo --version',
      'rustfmt --version',
      'cargo clippy --version'
    ]
  },
  {
    name: 'go',
    type: 'standard',
    expectedTools: ['go', 'gofmt', 'golangci-lint'],
    expectedLibraries: ['gin', 'gorm', 'testify'],
    validationCommands: [
      'go version',
      'gofmt -h',
      'golangci-lint --version'
    ]
  },
  {
    name: 'nushell',
    type: 'standard',
    expectedTools: ['nu', 'git', 'teller'],
    expectedLibraries: [],
    validationCommands: [
      'nu --version',
      'git --version',
      'teller --version'
    ]
  },
  // Agentic Environments
  {
    name: 'agentic-python',
    type: 'agentic',
    expectedTools: ['python', 'uv', 'ruff', 'mypy', 'pytest', 'npx'],
    expectedLibraries: ['fastapi', 'uvicorn', 'pydantic', 'copilotkit'],
    validationCommands: [
      'python --version',
      'uv --version',
      'npx --yes claude-flow@alpha --version',
      'pip list | grep fastapi'
    ]
  },
  {
    name: 'agentic-typescript',
    type: 'agentic',
    expectedTools: ['node', 'npm', 'tsc', 'eslint', 'jest', 'next'],
    expectedLibraries: ['typescript', '@types/node', 'next', '@copilotkit/react-core'],
    validationCommands: [
      'node --version',
      'npm --version',
      'npx --yes claude-flow@alpha --version',
      'npm list @copilotkit/react-core'
    ]
  },
  {
    name: 'agentic-rust',
    type: 'agentic',
    expectedTools: ['rustc', 'cargo', 'rustfmt', 'clippy', 'npx'],
    expectedLibraries: ['tokio', 'serde', 'clap', 'async-trait'],
    validationCommands: [
      'rustc --version',
      'cargo --version',
      'npx --yes claude-flow@alpha --version',
      'cargo tree | grep tokio'
    ]
  },
  {
    name: 'agentic-go',
    type: 'agentic',
    expectedTools: ['go', 'gofmt', 'golangci-lint', 'npx'],
    expectedLibraries: ['gin', 'gorm', 'testify'],
    validationCommands: [
      'go version',
      'npx --yes claude-flow@alpha --version',
      'go list -m all | grep gin'
    ]
  },
  {
    name: 'agentic-nushell',
    type: 'agentic',
    expectedTools: ['nu', 'git', 'teller', 'npx'],
    expectedLibraries: [],
    validationCommands: [
      'nu --version',
      'npx --yes claude-flow@alpha --version',
      'nu --help | grep version'
    ]
  }
];

const WORKSPACE_PREFIX = 'polyglot-test';
const PROVISION_TIMEOUT = 600000; // 10 minutes
const VALIDATION_TIMEOUT = 120000; // 2 minutes

describe('DevPod Swarm Tests', () => {
  const provisionedWorkspaces: string[] = [];
  
  beforeAll(async () => {
    console.log('🚀 Starting DevPod swarm test suite...');
    console.log(`📋 Testing ${TEST_ENVIRONMENTS.length} environments`);
  }, 30000);

  afterAll(async () => {
    console.log('🧹 Cleaning up test workspaces...');
    await cleanupWorkspaces();
  }, 60000);

  describe('DevPod Environment Provisioning', () => {
    test.each(TEST_ENVIRONMENTS)(
      'should provision $name environment successfully',
      async (environment) => {
        console.log(`🚀 Provisioning ${environment.name} environment...`);
        
        const workspaceName = await provisionEnvironment(environment.name);
        provisionedWorkspaces.push(workspaceName);
        
        expect(workspaceName).toBeDefined();
        expect(workspaceName).toMatch(new RegExp(`^${WORKSPACE_PREFIX}-${environment.name}-\\d+$`));
        
        // Verify workspace appears in DevPod list
        const workspaces = await listWorkspaces();
        expect(workspaces).toContain(workspaceName);
        
        console.log(`✅ Successfully provisioned ${environment.name}: ${workspaceName}`);
      },
      PROVISION_TIMEOUT
    );
  });

  describe('Environment Tool Validation', () => {
    test.each(TEST_ENVIRONMENTS)(
      'should validate tools in $name environment',
      async (environment) => {
        const workspaceName = getWorkspaceName(environment.name);
        
        if (!workspaceName) {
          throw new Error(`No workspace found for environment: ${environment.name}`);
        }

        console.log(`🔍 Validating tools in ${environment.name}...`);
        
        // Test connectivity
        await testWorkspaceConnectivity(workspaceName);
        
        // Validate expected tools
        for (const tool of environment.expectedTools) {
          await validateToolAvailability(workspaceName, tool);
        }
        
        console.log(`✅ Tool validation passed for ${environment.name}`);
      },
      VALIDATION_TIMEOUT
    );
  });

  describe('Environment-Specific Validation', () => {
    test.each(TEST_ENVIRONMENTS)(
      'should run validation commands in $name environment',
      async (environment) => {
        const workspaceName = getWorkspaceName(environment.name);
        
        if (!workspaceName) {
          throw new Error(`No workspace found for environment: ${environment.name}`);
        }

        console.log(`🧪 Running validation commands in ${environment.name}...`);
        
        // Run environment-specific validation commands
        for (const command of environment.validationCommands) {
          try {
            const result = await executeInWorkspace(workspaceName, command);
            expect(result.success).toBe(true);
            console.log(`  ✅ ${command}: OK`);
          } catch (error) {
            console.error(`  ❌ ${command}: Failed`);
            throw error;
          }
        }
        
        console.log(`✅ Validation commands passed for ${environment.name}`);
      },
      VALIDATION_TIMEOUT
    );
  });

  describe('Claude-Flow Integration', () => {
    test.each(TEST_ENVIRONMENTS.filter(env => env.type === 'agentic'))(
      'should validate Claude-Flow integration in $name',
      async (environment) => {
        const workspaceName = getWorkspaceName(environment.name);
        
        if (!workspaceName) {
          throw new Error(`No workspace found for environment: ${environment.name}`);
        }

        console.log(`🤖 Testing Claude-Flow integration in ${environment.name}...`);
        
        // Test Claude-Flow initialization
        const initResult = await executeInWorkspace(
          workspaceName,
          'npx --yes claude-flow@alpha init --force'
        );
        expect(initResult.success).toBe(true);
        
        // Test Claude-Flow status
        const statusResult = await executeInWorkspace(
          workspaceName,
          'npx --yes claude-flow@alpha status'
        );
        expect(statusResult.success).toBe(true);
        
        console.log(`✅ Claude-Flow integration validated for ${environment.name}`);
      },
      VALIDATION_TIMEOUT
    );
  });

  describe('DevPod .claude/ Auto-Installation', () => {
    test.each(TEST_ENVIRONMENTS)(
      'should validate .claude/ auto-installation in $name',
      async (environment) => {
        const workspaceName = getWorkspaceName(environment.name);
        
        if (!workspaceName) {
          throw new Error(`No workspace found for environment: ${environment.name}`);
        }

        console.log(`⚙️ Testing .claude/ auto-installation in ${environment.name}...`);
        
        // Check if .claude directory exists
        const claudeDirResult = await executeInWorkspace(
          workspaceName,
          'test -d /workspace/.claude && echo "exists" || echo "missing"'
        );
        expect(claudeDirResult.output).toContain('exists');
        
        // Check essential .claude components
        const essentialComponents = [
          '/workspace/.claude/settings.json',
          '/workspace/.claude/hooks/',
          '/workspace/.claude/commands/'
        ];
        
        for (const component of essentialComponents) {
          const result = await executeInWorkspace(
            workspaceName,
            `test -e ${component} && echo "exists" || echo "missing"`
          );
          expect(result.output).toContain('exists');
          console.log(`  ✅ ${component}: Found`);
        }
        
        console.log(`✅ .claude/ auto-installation validated for ${environment.name}`);
      },
      VALIDATION_TIMEOUT
    );
  });

  describe('Environment Health Monitoring', () => {
    test('should monitor resource usage across all environments', async () => {
      console.log('📊 Monitoring resource usage across environments...');
      
      const resourceMetrics = await gatherResourceMetrics();
      
      // Validate resource constraints
      for (const workspace of provisionedWorkspaces) {
        const metrics = resourceMetrics[workspace];
        if (metrics) {
          // Memory usage should be reasonable (< 2GB per container)
          expect(metrics.memoryUsageMB).toBeLessThan(2048);
          
          // CPU usage should be reasonable (< 2 cores)
          expect(metrics.cpuCores).toBeLessThan(2);
          
          console.log(`  📊 ${workspace}: ${metrics.memoryUsageMB}MB, ${metrics.cpuCores} CPU`);
        }
      }
      
      console.log('✅ Resource monitoring validation completed');
    }, VALIDATION_TIMEOUT);
  });

  // Helper functions
  async function provisionEnvironment(envName: string): Promise<string> {
    const timestamp = Date.now();
    const workspaceName = `${WORKSPACE_PREFIX}-${envName}-${timestamp}`;
    
    try {
      // Use centralized DevPod management script
      const command = `nu ../../host-tooling/devpod-management/manage-devpod.nu provision ${envName}`;
      const result = await executeHostCommand(command);
      
      if (!result.success) {
        throw new Error(`Failed to provision ${envName}: ${result.error}`);
      }
      
      return workspaceName;
    } catch (error) {
      throw new Error(`Provisioning failed for ${envName}: ${error}`);
    }
  }

  async function listWorkspaces(): Promise<string[]> {
    try {
      const result = await executeHostCommand('devpod list --output json');
      if (result.success) {
        const workspaces = JSON.parse(result.output);
        return workspaces.map((ws: any) => ws.name);
      }
      return [];
    } catch (error) {
      console.warn('Failed to list workspaces:', error);
      return [];
    }
  }

  async function testWorkspaceConnectivity(workspaceName: string): Promise<void> {
    const result = await executeInWorkspace(workspaceName, 'echo "Connection test"');
    if (!result.success) {
      throw new Error(`Cannot connect to workspace: ${workspaceName}`);
    }
  }

  async function validateToolAvailability(workspaceName: string, tool: string): Promise<void> {
    const result = await executeInWorkspace(workspaceName, `which ${tool}`);
    if (!result.success) {
      throw new Error(`Tool not found in ${workspaceName}: ${tool}`);
    }
  }

  async function executeInWorkspace(
    workspaceName: string, 
    command: string
  ): Promise<{ success: boolean; output: string; error?: string }> {
    try {
      const fullCommand = `devpod ssh ${workspaceName} -- ${command}`;
      return await executeHostCommand(fullCommand);
    } catch (error) {
      return {
        success: false,
        output: '',
        error: String(error)
      };
    }
  }

  async function executeHostCommand(
    command: string
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
      }, 30000);
    });
  }

  function getWorkspaceName(envName: string): string | undefined {
    return provisionedWorkspaces.find(ws => ws.includes(envName));
  }

  async function gatherResourceMetrics(): Promise<Record<string, any>> {
    const metrics: Record<string, any> = {};
    
    for (const workspace of provisionedWorkspaces) {
      try {
        // Get memory usage
        const memResult = await executeInWorkspace(
          workspace,
          "free -m | grep Mem | awk '{print $3}'"
        );
        
        // Get CPU info
        const cpuResult = await executeInWorkspace(
          workspace,
          "nproc"
        );
        
        if (memResult.success && cpuResult.success) {
          metrics[workspace] = {
            memoryUsageMB: parseInt(memResult.output) || 0,
            cpuCores: parseInt(cpuResult.output) || 0
          };
        }
      } catch (error) {
        console.warn(`Failed to gather metrics for ${workspace}:`, error);
      }
    }
    
    return metrics;
  }

  async function cleanupWorkspaces(): Promise<void> {
    for (const workspace of provisionedWorkspaces) {
      try {
        console.log(`🗑️ Cleaning up workspace: ${workspace}`);
        await executeHostCommand(`devpod delete ${workspace} --force`);
      } catch (error) {
        console.warn(`Failed to cleanup workspace ${workspace}:`, error);
      }
    }
  }
});