/**
 * Jest Configuration for Kimi-K2 Integration Tests
 * Comprehensive testing setup for Synaptic Neural Mesh CLI with Kimi-K2
 */

module.exports = {
  // Test environment
  testEnvironment: 'node',
  
  // Root directory for tests
  rootDir: './',
  
  // Test file patterns
  testMatch: [
    '**/tests/**/*.test.js',
    '**/tests/**/*.test.ts',
    '**/__tests__/**/*.js',
    '**/__tests__/**/*.ts'
  ],
  
  // Coverage configuration
  collectCoverage: true,
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    'lib/**/*.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/index.js',
    '!src/**/index.ts',
    '!**/node_modules/**',
    '!**/tests/**',
    '!**/coverage/**'
  ],
  
  coverageDirectory: 'coverage',
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json'
  ],
  
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    },
    './src/lib/kimi-k2-client.js': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    },
    './src/lib/kimi-k2-mcp-bridge.js': {
      branches: 85,
      functions: 85,
      lines: 85,
      statements: 85
    }
  },
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.js'
  ],
  
  // Test timeout
  testTimeout: 30000,
  
  // Module name mapping for absolute imports
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@lib/(.*)$': '<rootDir>/lib/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1'
  },
  
  // Transform configuration
  transform: {
    '^.+\\.(js|jsx|ts|tsx)$': ['babel-jest', {
      presets: [
        ['@babel/preset-env', {
          targets: {
            node: '18'
          }
        }],
        ['@babel/preset-typescript', {
          allowDeclareFields: true
        }]
      ],
      plugins: [
        '@babel/plugin-proposal-class-properties',
        '@babel/plugin-proposal-object-rest-spread'
      ]
    }]
  },
  
  // File extensions to consider
  moduleFileExtensions: ['js', 'ts', 'json', 'node'],
  
  // Global variables
  globals: {
    'process.env.NODE_ENV': 'test',
    'process.env.JEST_WORKER_ID': true
  },
  
  // Test suites organization
  projects: [
    {
      displayName: 'Unit Tests',
      testMatch: ['<rootDir>/tests/unit/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js']
    },
    {
      displayName: 'CLI Tests',
      testMatch: ['<rootDir>/tests/cli/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js']
    },
    {
      displayName: 'MCP Integration Tests', 
      testMatch: ['<rootDir>/tests/mcp/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js']
    },
    {
      displayName: 'Integration Tests',
      testMatch: ['<rootDir>/tests/integration/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
      testTimeout: 60000 // Longer timeout for integration tests
    },
    {
      displayName: 'Performance Tests',
      testMatch: ['<rootDir>/tests/performance/**/*.test.js'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
      testTimeout: 120000 // Extended timeout for performance tests
    }
  ],
  
  // Reporter configuration
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: 'test-results',
      outputName: 'junit.xml',
      ancestorSeparator: ' › ',
      uniqueOutputName: 'false',
      suiteNameTemplate: '{filepath}',
      classNameTemplate: '{classname}',
      titleTemplate: '{title}'
    }],
    ['jest-html-reporters', {
      publicPath: './test-results',
      filename: 'report.html',
      expand: true,
      hideIcon: false,
      pageTitle: 'Kimi-K2 Integration Test Report'
    }]
  ],
  
  // Verbose output for debugging
  verbose: true,
  
  // Automatically clear mock calls and instances between tests
  clearMocks: true,
  
  // Automatically restore mock state between tests
  restoreMocks: true,
  
  // Force exit after tests complete
  forceExit: true,
  
  // Detect open handles
  detectOpenHandles: true,
  
  // Error handling
  errorOnDeprecated: true,
  
  // Mock configuration
  __mocks__: {
    'fs-extra': '<rootDir>/tests/__mocks__/fs-extra.js',
    'child_process': '<rootDir>/tests/__mocks__/child_process.js'
  },
  
  // Test environment options
  testEnvironmentOptions: {
    NODE_ENV: 'test'
  },
  
  // Custom test sequence
  testSequencer: '<rootDir>/tests/test-sequencer.js',
  
  // Watch plugins for development
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname'
  ],
  
  // Notify on test completion
  notify: true,
  notifyMode: 'failure-change',
  
  // Bail configuration
  bail: false, // Continue running tests even if some fail
  
  // Max workers for parallel execution
  maxWorkers: '50%',
  
  // Cache configuration
  cache: true,
  cacheDirectory: '<rootDir>/.jest-cache',
  
  // Snapshot configuration
  snapshotSerializers: [
    '<rootDir>/tests/serializers/error-serializer.js'
  ],
  
  // Custom matchers
  setupFiles: [
    '<rootDir>/tests/jest-matchers.js'
  ]
};