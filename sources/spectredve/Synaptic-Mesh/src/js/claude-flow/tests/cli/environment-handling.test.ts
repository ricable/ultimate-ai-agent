/**
 * Comprehensive tests for environment detection and handling
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { 
  detectExecutionEnvironment, 
  applySmartDefaults, 
  getEnvironmentDescription,
  shouldUseNonInteractiveMode 
} from '../../src/cli/utils/environment-detector';
import { TaskExecutorV2 } from '../../src/swarm/executor-v2';
import { PromptDefaultsManager } from '../../src/cli/utils/prompt-defaults';

describe('Environment Detection', () => {
  let originalEnv: NodeJS.ProcessEnv;
  let originalStdout: any;
  let originalStdin: any;

  beforeEach(() => {
    originalEnv = { ...process.env };
    originalStdout = process.stdout.isTTY;
    originalStdin = process.stdin.isTTY;
  });

  afterEach(() => {
    process.env = originalEnv;
    Object.defineProperty(process.stdout, 'isTTY', { value: originalStdout, writable: true });
    Object.defineProperty(process.stdin, 'isTTY', { value: originalStdin, writable: true });
  });

  describe('VS Code Detection', () => {
    it('should detect VS Code integrated terminal', () => {
      process.env.TERM_PROGRAM = 'vscode';
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.isVSCode).toBe(true);
      expect(env.recommendedFlags).toContain('--dangerously-skip-permissions');
      expect(env.recommendedFlags).toContain('--non-interactive');
    });

    it('should detect VS Code without TTY', () => {
      process.env.TERM_PROGRAM = 'vscode';
      Object.defineProperty(process.stdout, 'isTTY', { value: false, writable: true });
      
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.isInteractive).toBe(false);
      expect(env.isVSCode).toBe(true);
      expect(env.warnings).toContain('VS Code integrated terminal detected - interactive features may be limited');
    });
  });

  describe('CI/CD Environment Detection', () => {
    const ciEnvironments = [
      { name: 'GitHub Actions', env: { CI: 'true', GITHUB_ACTIONS: 'true' } },
      { name: 'GitLab CI', env: { CI: 'true', GITLAB_CI: 'true' } },
      { name: 'Jenkins', env: { JENKINS_URL: 'http://jenkins.example.com' } },
      { name: 'CircleCI', env: { CIRCLECI: 'true' } },
      { name: 'Travis CI', env: { TRAVIS: 'true' } },
      { name: 'Buildkite', env: { BUILDKITE: 'true' } },
      { name: 'Drone', env: { DRONE: 'true' } }
    ];

    ciEnvironments.forEach(({ name, env: envVars }) => {
      it(`should detect ${name}`, () => {
        Object.assign(process.env, envVars);
        const env = detectExecutionEnvironment({ skipWarnings: true });
        
        expect(env.isCI).toBe(true);
        expect(env.recommendedFlags).toContain('--dangerously-skip-permissions');
        expect(env.recommendedFlags).toContain('--non-interactive');
        expect(env.recommendedFlags).toContain('--json');
      });
    });
  });

  describe('Docker Detection', () => {
    it('should detect Docker container', () => {
      process.env.DOCKER_CONTAINER = 'true';
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.isDocker).toBe(true);
    });

    it('should recommend flags for Docker without TTY', () => {
      process.env.DOCKER_CONTAINER = 'true';
      Object.defineProperty(process.stdout, 'isTTY', { value: false, writable: true });
      
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.recommendedFlags).toContain('--dangerously-skip-permissions');
      expect(env.recommendedFlags).toContain('--non-interactive');
    });
  });

  describe('SSH Detection', () => {
    it('should detect SSH session', () => {
      process.env.SSH_CLIENT = '192.168.1.1 12345 22';
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.isSSH).toBe(true);
    });

    it('should warn about SSH without TTY', () => {
      process.env.SSH_CLIENT = '192.168.1.1 12345 22';
      Object.defineProperty(process.stdout, 'isTTY', { value: false, writable: true });
      
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.warnings).toContain('SSH session without TTY - consider using ssh -t');
    });
  });

  describe('Git Bash Detection', () => {
    it('should detect Git Bash on Windows', () => {
      process.env.TERM_PROGRAM = 'mintty';
      process.env.MSYSTEM = 'MINGW64';
      Object.defineProperty(process, 'platform', { value: 'win32', writable: true });
      
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.isGitBash).toBe(true);
      expect(env.warnings).toContain('Git Bash detected - some interactive features may not work correctly');
    });
  });

  describe('Windows Terminal Detection', () => {
    it('should detect Windows Terminal', () => {
      process.env.WT_SESSION = 'some-session-id';
      
      const env = detectExecutionEnvironment({ skipWarnings: true });
      
      expect(env.isWindowsTerminal).toBe(true);
    });
  });
});

describe('Smart Defaults Application', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should apply defaults for VS Code environment', () => {
    process.env.TERM_PROGRAM = 'vscode';
    const options = { verbose: false };
    
    const enhanced = applySmartDefaults(options);
    
    expect(enhanced.dangerouslySkipPermissions).toBe(true);
    expect(enhanced.nonInteractive).toBe(true);
    expect(enhanced.appliedDefaults).toContain('--dangerously-skip-permissions');
    expect(enhanced.appliedDefaults).toContain('--non-interactive');
  });

  it('should apply defaults for CI environment', () => {
    process.env.CI = 'true';
    const options = {};
    
    const enhanced = applySmartDefaults(options);
    
    expect(enhanced.dangerouslySkipPermissions).toBe(true);
    expect(enhanced.nonInteractive).toBe(true);
    expect(enhanced.json).toBe(true);
  });

  it('should not override existing options', () => {
    process.env.CI = 'true';
    const options = { 
      dangerouslySkipPermissions: false,
      nonInteractive: false,
      json: false 
    };
    
    const enhanced = applySmartDefaults(options);
    
    expect(enhanced.dangerouslySkipPermissions).toBe(false);
    expect(enhanced.nonInteractive).toBe(false);
    expect(enhanced.json).toBe(false);
    expect(enhanced.appliedDefaults).toHaveLength(0);
  });

  it('should disable color for non-color terminals', () => {
    process.env.NO_COLOR = '1';
    const options = {};
    
    const enhanced = applySmartDefaults(options);
    
    expect(enhanced.noColor).toBe(true);
    expect(enhanced.appliedDefaults).toContain('--no-color');
  });
});

describe('Task Executor v2 Environment Handling', () => {
  let executor: TaskExecutorV2;

  beforeEach(() => {
    executor = new TaskExecutorV2();
  });

  it('should retry with non-interactive mode on interactive error', async () => {
    const mockTask = {
      id: { id: 'test-task' },
      description: 'Test task',
      requirements: { tools: [] }
    };
    
    const mockAgent = {
      id: { id: 'test-agent' },
      type: 'coder',
      status: 'idle'
    };

    // Mock the execution to fail with interactive error first
    vi.spyOn(executor as any, 'executeClaudeWithTimeoutV2')
      .mockRejectedValueOnce(new Error('stdin is not a tty'))
      .mockResolvedValueOnce({
        success: true,
        output: 'Success',
        error: '',
        exitCode: 0,
        duration: 1000,
        resourcesUsed: {},
        artifacts: {}
      });

    const result = await executor.executeClaudeTask(mockTask as any, mockAgent as any, {
      retryOnInteractiveError: true
    });

    expect(result.success).toBe(true);
    expect(executor['executeClaudeWithTimeoutV2']).toHaveBeenCalledTimes(2);
  });

  it('should include environment metadata in execution', async () => {
    const mockTask = {
      id: { id: 'test-task' },
      description: 'Test task',
      requirements: { tools: [] }
    };
    
    const mockAgent = {
      id: { id: 'test-agent' },
      type: 'coder',
      status: 'idle'
    };

    const buildCommandSpy = vi.spyOn(executor as any, 'buildClaudeCommandV2');

    await executor.executeClaudeTask(mockTask as any, mockAgent as any, {
      nonInteractive: true
    });

    expect(buildCommandSpy).toHaveBeenCalled();
    const result = buildCommandSpy.mock.results[0].value;
    expect(result.args).toContain('--non-interactive');
    expect(result.args).toContain('--dangerously-skip-permissions');
  });
});

describe('Prompt Defaults System', () => {
  let manager: PromptDefaultsManager;

  beforeEach(() => {
    manager = new PromptDefaultsManager('/tmp/test-prompt-defaults.json');
  });

  it('should return environment defaults with highest priority', () => {
    process.env.CLAUDE_AUTO_APPROVE = '1';
    manager = new PromptDefaultsManager('/tmp/test-prompt-defaults.json');
    
    const confirmDefault = manager.getDefault('deleteFile', 'task', 'confirm');
    expect(confirmDefault).toBe(true);
  });

  it('should apply non-interactive defaults', () => {
    manager.applyNonInteractiveDefaults(true);
    
    const modelDefault = manager.getDefault('model', 'init', 'select');
    expect(modelDefault).toBe('claude-3-opus-20240229');
    
    const confirmDefault = manager.getDefault('continue', 'task', 'confirm');
    expect(confirmDefault).toBe(true);
  });

  it('should respect safety defaults', () => {
    manager.applyNonInteractiveDefaults(true);
    
    // Should not auto-confirm dangerous operations
    const deleteDefault = manager.getDefault('delete', 'file', 'confirm');
    expect(deleteDefault).toBe(false);
    
    const deployDefault = manager.getDefault('deploy', 'production', 'confirm');
    expect(deployDefault).toBe(false);
  });

  it('should handle pattern matching', () => {
    manager.setDefault('confirm-*', true, {
      pattern: 'confirm-*',
      scope: 'global'
    });
    
    const result = manager.getDefault('confirm-action', 'test');
    expect(result).toBe(true);
  });
});

describe('Environment Description', () => {
  it('should provide readable environment description', () => {
    process.env.TERM_PROGRAM = 'vscode';
    process.env.CI = 'true';
    
    const description = getEnvironmentDescription();
    
    expect(description).toContain('VS Code');
    expect(description).toContain('CI');
  });

  it('should indicate feature support', () => {
    Object.defineProperty(process.stdout, 'isTTY', { value: true, writable: true });
    
    const description = getEnvironmentDescription();
    
    expect(description).toContain('interactive');
    expect(description).toContain('color');
  });
});

describe('Non-Interactive Mode Detection', () => {
  it('should recommend non-interactive for VS Code', () => {
    process.env.TERM_PROGRAM = 'vscode';
    
    const shouldUse = shouldUseNonInteractiveMode();
    
    expect(shouldUse).toBe(true);
  });

  it('should recommend non-interactive for CI', () => {
    process.env.CI = 'true';
    
    const shouldUse = shouldUseNonInteractiveMode();
    
    expect(shouldUse).toBe(true);
  });

  it('should allow interactive for standard terminal', () => {
    Object.defineProperty(process.stdout, 'isTTY', { value: true, writable: true });
    Object.defineProperty(process.stdin, 'isTTY', { value: true, writable: true });
    process.env = {}; // Clear all env vars
    
    const shouldUse = shouldUseNonInteractiveMode();
    
    expect(shouldUse).toBe(false);
  });

  it('should force non-interactive when requested', () => {
    const shouldUse = shouldUseNonInteractiveMode({ force: true });
    
    expect(shouldUse).toBe(true);
  });
});

describe('Edge Cases', () => {
  it('should handle missing fs module gracefully', () => {
    // Mock fs.existsSync to throw
    vi.mock('fs', () => ({
      existsSync: () => { throw new Error('FS not available'); }
    }));
    
    const env = detectExecutionEnvironment({ skipWarnings: true });
    
    expect(env).toBeDefined();
    expect(env.isDocker).toBe(false);
  });

  it('should handle undefined environment variables', () => {
    process.env = {};
    
    const env = detectExecutionEnvironment({ skipWarnings: true });
    
    expect(env).toBeDefined();
    expect(env.terminalType).toBe('unknown');
  });

  it('should handle raw mode detection failure', () => {
    vi.spyOn(process.stdin, 'setRawMode').mockImplementation(() => {
      throw new Error('Raw mode not supported');
    });
    
    const env = detectExecutionEnvironment({ skipWarnings: true });
    
    expect(env.supportsRawMode).toBe(false);
    expect(env.recommendedFlags).toContain('--compatible-ui');
  });
});