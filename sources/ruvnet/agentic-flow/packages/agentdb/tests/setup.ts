/**
 * Global test setup for AgentDB test suite
 */

import { beforeAll, afterAll, beforeEach, afterEach } from 'vitest';
import * as fs from 'fs';
import * as path from 'path';

// Test database path
export const TEST_DB_DIR = path.join(process.cwd(), 'tests', 'fixtures');
export const TEST_DB_PATH = path.join(TEST_DB_DIR, 'test.db');

// Ensure test fixtures directory exists
beforeAll(() => {
  if (!fs.existsSync(TEST_DB_DIR)) {
    fs.mkdirSync(TEST_DB_DIR, { recursive: true });
  }
});

// Clean up test databases before each test
beforeEach(() => {
  const testFiles = [
    TEST_DB_PATH,
    `${TEST_DB_PATH}-wal`,
    `${TEST_DB_PATH}-shm`,
  ];

  testFiles.forEach(file => {
    if (fs.existsSync(file)) {
      try {
        fs.unlinkSync(file);
      } catch (e) {
        // Ignore errors
      }
    }
  });
});

// Clean up after all tests
afterAll(() => {
  const testFiles = [
    TEST_DB_PATH,
    `${TEST_DB_PATH}-wal`,
    `${TEST_DB_PATH}-shm`,
  ];

  testFiles.forEach(file => {
    if (fs.existsSync(file)) {
      try {
        fs.unlinkSync(file);
      } catch (e) {
        // Ignore errors
      }
    }
  });
});
