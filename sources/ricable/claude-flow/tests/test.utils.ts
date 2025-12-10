/**
 * Test utilities for Claude-Flow
 */

import { describe, it, beforeEach, afterEach, beforeAll, afterAll, expect, jest } from '@jest/globals';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { spawn } from 'child_process';

// Jest-compatible assertion helpers
export function assertEquals<T>(actual: T, expected: T, message?: string): void {
  expect(actual).toBe(expected);
}

export function assertExists<T>(value: T, message?: string): void {
  expect(value).toBeDefined();
}

export function assertRejects(promise: Promise<any>, message?: string): void {
  expect(promise).rejects.toThrow();
}

export function assertThrows(fn: () => any, message?: string): void {
  expect(fn).toThrow();
}

// Re-export Jest testing utilities
export {
  describe,
  it,
  beforeEach,
  afterEach,
  beforeAll,
  afterAll,
  expect,
  jest,
};

// Mock Jest spy utilities for compatibility
export const spy = jest.fn;
export const stub = jest.fn;
export function assertSpyCall(mockFn: jest.Mock, callIndex: number, expectedArgs?: any[]): void {
  expect(mockFn).toHaveBeenNthCalledWith(callIndex + 1, ...(expectedArgs || []));
}
export function assertSpyCalls(mockFn: jest.Mock, expectedCalls: number): void {
  expect(mockFn).toHaveBeenCalledTimes(expectedCalls);
}

// Simple FakeTime implementation for Jest
export class FakeTime {
  private originalNow = Date.now;
  private currentTime: number;

  constructor(time?: number | Date) {
    this.currentTime = time instanceof Date ? time.getTime() : time || Date.now();
    Date.now = () => this.currentTime;
  }

  tick(ms: number): void {
    this.currentTime += ms;
  }

  restore(): void {
    Date.now = this.originalNow;
  }
}

/**
 * Creates a test fixture
 */
export function createFixture<T>(factory: () => T): {
  get(): T;
  reset(): void;
} {
  let instance: T;

  return {
    get(): T {
      if (!instance) {
        instance = factory();
      }
      return instance;
    },
    reset(): void {
      instance = factory();
    },
  };
}

/**
 * Waits for a condition to be true
 */
export async function waitFor(
  condition: () => boolean | Promise<boolean>,
  options: { timeout?: number; interval?: number } = {},
): Promise<void> {
  const { timeout = 5000, interval = 100 } = options;
  const start = Date.now();

  while (Date.now() - start < timeout) {
    if (await condition()) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, interval));
  }

  throw new Error('Timeout waiting for condition');
}

/**
 * Creates a deferred promise for testing
 */
export function createDeferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: any) => void;
} {
  let resolve: (value: T) => void;
  let reject: (reason?: any) => void;

  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });

  return { promise, resolve: resolve!, reject: reject! };
}

/**
 * Captures console output during test
 */
export function captureConsole(): {
  getOutput(): string[];
  getErrors(): string[];
  restore(): void;
} {
  const output: string[] = [];
  const errors: string[] = [];

  const originalLog = console.log;
  const originalError = console.error;
  const originalDebug = console.debug;
  const originalInfo = console.info;
  const originalWarn = console.warn;

  console.log = (...args: any[]) => output.push(args.join(' '));
  console.error = (...args: any[]) => errors.push(args.join(' '));
  console.debug = (...args: any[]) => output.push(args.join(' '));
  console.info = (...args: any[]) => output.push(args.join(' '));
  console.warn = (...args: any[]) => output.push(args.join(' '));

  return {
    getOutput: () => [...output],
    getErrors: () => [...errors],
    restore: () => {
      console.log = originalLog;
      console.error = originalError;
      console.debug = originalDebug;
      console.info = originalInfo;
      console.warn = originalWarn;
    },
  };
}

/**
 * Creates a test file in a temporary directory
 */
export function createTestFile(
  filePath: string,
  content: string,
): string {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'claude-flow-'));
  const fullPath = path.join(tempDir, filePath);
  const dir = path.dirname(fullPath);
  
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(fullPath, content);
  
  return fullPath;
}

/**
 * Runs a CLI command and captures output
 */
export function runCommand(
  args: string[],
  options: { stdin?: string; env?: Record<string, string> } = {},
): Promise<{ stdout: string; stderr: string; code: number }> {
  return new Promise((resolve, reject) => {
    const env = { ...process.env, ...options.env };
    const child = spawn('node', ['src/cli/index.ts', ...args], {
      stdio: ['pipe', 'pipe', 'pipe'],
      env,
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    child.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    child.on('close', (code) => {
      resolve({ stdout, stderr, code: code || 0 });
    });

    child.on('error', (error) => {
      reject(error);
    });

    if (options.stdin) {
      child.stdin.write(options.stdin);
      child.stdin.end();
    }
  });
}

/**
 * Test data builder for common types
 */
export class TestDataBuilder {
  static agentProfile(overrides = {}) {
    return {
      id: 'agent-1',
      name: 'Test Agent',
      type: 'coordinator' as const,
      capabilities: ['task-management', 'coordination'],
      systemPrompt: 'You are a test agent',
      maxConcurrentTasks: 5,
      priority: 10,
      environment: {},
      workingDirectory: '/tmp',
      shell: '/bin/bash',
      metadata: {},
      ...overrides,
    };
  }

  static task(overrides = {}) {
    return {
      id: 'task-1',
      type: 'test',
      description: 'Test task',
      priority: 50,
      dependencies: [],
      status: 'pending' as const,
      input: { test: true },
      createdAt: new Date(),
      metadata: {},
      ...overrides,
    };
  }

  static config(overrides = {}) {
    return {
      orchestrator: {
        maxConcurrentAgents: 10,
        taskQueueSize: 100,
        healthCheckInterval: 30000,
        shutdownTimeout: 30000,
        maintenanceInterval: 300000,
        metricsInterval: 60000,
        persistSessions: false,
        dataDir: './tests/data',
        sessionRetentionMs: 3600000,
        taskHistoryRetentionMs: 86400000,
        taskMaxRetries: 3,
      },
      terminal: {
        type: 'native' as const,
        poolSize: 5,
        recycleAfter: 10,
        healthCheckInterval: 60000,
        commandTimeout: 300000,
      },
      memory: {
        backend: 'sqlite' as const,
        cacheSizeMB: 10,
        syncInterval: 5000,
        conflictResolution: 'last-write' as const,
        retentionDays: 1,
        sqlitePath: ':memory:',
        markdownDir: './tests/data/memory',
      },
      coordination: {
        maxRetries: 3,
        retryDelay: 100,
        deadlockDetection: true,
        resourceTimeout: 60000,
        messageTimeout: 30000,
      },
      mcp: {
        transport: 'stdio' as const,
        port: 8081,
        tlsEnabled: false,
      },
      logging: {
        level: 'error' as const,
        format: 'json' as const,
        destination: 'console' as const,
      },
      ...overrides,
    };
  }
}

/**
 * Assertion helpers
 */
export function assertEventEmitted(
  events: Array<{ event: string; data: any }>,
  eventName: string,
  matcher?: (data: any) => boolean,
): void {
  const emitted = events.find((e) => e.event === eventName);
  expect(emitted).toBeDefined();
  
  if (matcher && emitted && !matcher(emitted.data)) {
    throw new Error(`Event '${eventName}' data did not match expected criteria`);
  }
}

export function assertNoEventEmitted(
  events: Array<{ event: string; data: any }>,
  eventName: string,
): void {
  const emitted = events.find((e) => e.event === eventName);
  expect(emitted).toBeUndefined();
}