/**
 * Jest Configuration for Phase 5 Tests
 * Comprehensive configuration for Pydantic Schema Generation, Validation Engine, Template Export, Pipeline Integration, and Production Deployment tests
 */

module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests'],
  testMatch: [
    'tests/pydantic/**/*.test.ts',
    'tests/validation/**/*.test.ts',
    'tests/export/**/*.test.ts',
    'tests/pipeline/**/*.test.ts',
    'tests/deployment/**/*.test.ts'
  ],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest'
  },
  collectCoverageFrom: [
    'src/pydantic/**/*.ts',
    'src/validation/**/*.ts',
    'src/export/**/*.ts',
    'src/pipeline/**/*.ts',
    'src/deployment/**/*.ts',
    '!src/**/*.d.ts',
    '!src/index.ts'
  ],
  coverageDirectory: 'coverage/phase5',
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json',
    'clover'
  ],
  coverageThresholds: {
    global: {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    './src/pydantic/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    './src/validation/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    './src/export/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    './src/pipeline/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    },
    './src/deployment/': {
      branches: 100,
      functions: 100,
      lines: 100,
      statements: 100
    }
  },
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  testTimeout: 60000, // 60 seconds for comprehensive integration tests
  verbose: true,
  maxWorkers: 4, // Limit workers for resource-intensive tests
  maxConcurrency: 3,
  detectOpenHandles: true,
  forceExit: true,
  errorOnDeprecated: true,
  testSequencer: '<rootDir>/scripts/phase5-test-sequencer.js',
  reporters: [
    'default',
    [
      'jest-junit',
      {
        outputDirectory: 'test-results',
        outputName: 'phase5-test-results.xml',
        ancestorSeparator: ' › ',
        uniqueOutputName: 'false',
        suiteNameTemplate: '{filepath}',
        classNameTemplate: '{classname}',
        titleTemplate: '{title}'
      }
    ],
    [
      'jest-html-reporters',
      {
        publicPath: './test-results',
        filename: 'phase5-test-report.html',
        expand: true,
        hideIcon: false,
        pageTitle: 'Phase 5 Test Report',
        logoImgPath: undefined,
        inlineSource: false
      }
    ]
  ],
  globalSetup: '<rootDir>/scripts/phase5-global-setup.js',
  globalTeardown: '<rootDir>/scripts/phase5-global-teardown.js'
};