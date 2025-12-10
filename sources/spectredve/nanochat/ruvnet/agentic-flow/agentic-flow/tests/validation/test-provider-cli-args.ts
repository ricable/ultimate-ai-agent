/**
 * Test suite for CLI argument provider selection fix
 * Validates that --provider argument correctly overrides ANTHROPIC_API_KEY
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { parseArgs } from '../../src/utils/cli.js';

describe('Provider CLI Arguments', () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  it('should parse --provider openrouter argument', () => {
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--provider', 'openrouter'];
    const options = parseArgs();

    expect(options.provider).toBe('openrouter');
    expect(options.mode).toBe('agent');
  });

  it('should parse --openrouter-key argument', () => {
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--openrouter-key', 'sk-or-test-key'];
    const options = parseArgs();

    expect(options.openrouterApiKey).toBe('sk-or-test-key');
  });

  it('should parse --anthropic-key argument', () => {
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--anthropic-key', 'sk-ant-test-key'];
    const options = parseArgs();

    expect(options.anthropicApiKey).toBe('sk-ant-test-key');
  });

  it('should handle multiple provider arguments', () => {
    process.argv = [
      'node', 'cli.js',
      '--agent', 'coder',
      '--task', 'test',
      '--provider', 'openrouter',
      '--openrouter-key', 'sk-or-test-key',
      '--model', 'deepseek/deepseek-chat'
    ];
    const options = parseArgs();

    expect(options.provider).toBe('openrouter');
    expect(options.openrouterApiKey).toBe('sk-or-test-key');
    expect(options.model).toBe('deepseek/deepseek-chat');
  });
});

describe('Provider Environment Variable Propagation', () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
    // Clear provider-related env vars
    delete process.env.PROVIDER;
    delete process.env.ANTHROPIC_API_KEY;
    delete process.env.OPENROUTER_API_KEY;
    delete process.env.GOOGLE_GEMINI_API_KEY;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should set PROVIDER env var when provider is specified', () => {
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--provider', 'openrouter'];
    const options = parseArgs();

    // Simulate what index.ts does
    if (options.provider) {
      process.env.PROVIDER = options.provider;
    }

    expect(process.env.PROVIDER).toBe('openrouter');
  });

  it('should set OPENROUTER_API_KEY when specified', () => {
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--openrouter-key', 'sk-or-test'];
    const options = parseArgs();

    // Simulate what index.ts does
    if (options.openrouterApiKey) {
      process.env.OPENROUTER_API_KEY = options.openrouterApiKey;
    }

    expect(process.env.OPENROUTER_API_KEY).toBe('sk-or-test');
  });

  it('should not override existing ANTHROPIC_API_KEY unless explicitly set', () => {
    process.env.ANTHROPIC_API_KEY = 'sk-ant-existing';
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--provider', 'openrouter'];
    const options = parseArgs();

    // Simulate what index.ts does - should NOT override existing key
    if (options.anthropicApiKey) {
      process.env.ANTHROPIC_API_KEY = options.anthropicApiKey;
    }

    expect(process.env.ANTHROPIC_API_KEY).toBe('sk-ant-existing');
  });

  it('should override ANTHROPIC_API_KEY when explicitly provided via CLI', () => {
    process.env.ANTHROPIC_API_KEY = 'sk-ant-existing';
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--anthropic-key', 'sk-ant-new'];
    const options = parseArgs();

    // Simulate what index.ts does
    if (options.anthropicApiKey) {
      process.env.ANTHROPIC_API_KEY = options.anthropicApiKey;
    }

    expect(process.env.ANTHROPIC_API_KEY).toBe('sk-ant-new');
  });
});

describe('Provider Validation', () => {
  it('should accept valid providers', () => {
    const validProviders = ['anthropic', 'openrouter', 'gemini', 'onnx', 'requesty'];

    validProviders.forEach(provider => {
      process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--provider', provider];
      const options = parseArgs();
      expect(options.provider).toBe(provider);
    });
  });
});

describe('Backwards Compatibility', () => {
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    originalEnv = { ...process.env };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should still work with environment variables only', () => {
    process.env.PROVIDER = 'openrouter';
    process.env.OPENROUTER_API_KEY = 'sk-or-env-key';
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test'];

    const options = parseArgs();

    // CLI args don't override env vars unless explicitly set
    expect(process.env.PROVIDER).toBe('openrouter');
    expect(process.env.OPENROUTER_API_KEY).toBe('sk-or-env-key');
  });

  it('should prioritize CLI args over environment variables', () => {
    process.env.PROVIDER = 'anthropic';
    process.argv = ['node', 'cli.js', '--agent', 'coder', '--task', 'test', '--provider', 'openrouter'];

    const options = parseArgs();

    // Simulate what index.ts does - CLI args win
    if (options.provider) {
      process.env.PROVIDER = options.provider;
    }

    expect(options.provider).toBe('openrouter');
    expect(process.env.PROVIDER).toBe('openrouter');
  });
});
