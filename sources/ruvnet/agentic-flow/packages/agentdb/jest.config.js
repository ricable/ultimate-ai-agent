/**
 * Jest Configuration for AgentDB v2.0.0
 *
 * Test coverage targets:
 * - CLI: >90%
 * - Simulation logic: >80%
 */

export default {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/simulation'],
  testMatch: [
    '**/__tests__/**/*.ts',
    '**/?(*.)+(spec|test).ts'
  ],
  collectCoverageFrom: [
    'src/cli/**/*.ts',
    'simulation/scenarios/**/*.ts',
    '!**/*.d.ts',
    '!**/node_modules/**',
    '!**/dist/**'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    },
    './src/cli/': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    }
  },
  coverageDirectory: '<rootDir>/coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: {
        esModuleInterop: true,
        allowSyntheticDefaultImports: true,
        module: 'commonjs'
      }
    }]
  },
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@simulation/(.*)$': '<rootDir>/simulation/$1'
  },
  testTimeout: 30000, // 30s default timeout
  verbose: true,
  bail: false, // Continue running tests after failures
  maxWorkers: '50%', // Use half of available CPU cores
};
