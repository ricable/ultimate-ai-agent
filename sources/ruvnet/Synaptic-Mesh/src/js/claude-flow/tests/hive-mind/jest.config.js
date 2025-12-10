/**
 * Jest configuration for Hive Mind tests
 */

export default {
  // Test environment
  testEnvironment: 'node',
  
  // Root directory
  rootDir: '../..',
  
  // Test match patterns
  testMatch: [
    '<rootDir>/tests/hive-mind/**/*.test.js'
  ],
  
  // Module paths
  moduleNameMapper: {
    '^@/(.*)$': '<rootDir>/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/$1'
  },
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/tests/hive-mind/setup.js'
  ],
  
  // Coverage configuration
  collectCoverage: true,
  coverageDirectory: '<rootDir>/coverage/hive-mind',
  collectCoverageFrom: [
    'src/cli/wizards/hive-mind/**/*.js',
    'src/cli/simple-commands/hive-mind/**/*.js',
    'src/swarm/hive-mind/**/*.js',
    '!**/*.test.js',
    '!**/node_modules/**'
  ],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  
  // Test timeouts
  testTimeout: 30000, // 30 seconds for integration tests
  
  // Parallel execution
  maxWorkers: '50%',
  
  // Reporter configuration
  reporters: [
    'default',
    ['jest-junit', {
      outputDirectory: '<rootDir>/test-results/hive-mind',
      outputName: 'junit.xml',
      suiteName: 'Hive Mind Tests',
      ancestorSeparator: ' › ',
      usePathForSuiteName: true
    }],
    ['jest-html-reporter', {
      pageTitle: 'Hive Mind Test Report',
      outputPath: '<rootDir>/test-results/hive-mind/index.html',
      includeFailureMsg: true,
      includeSuiteFailure: true,
      includeConsoleLog: true,
      dateFormat: 'yyyy-mm-dd HH:MM:ss'
    }]
  ],
  
  // Transform configuration
  transform: {
    '^.+\\.jsx?$': ['babel-jest', {
      presets: [
        ['@babel/preset-env', {
          targets: {
            node: '20'
          }
        }]
      ]
    }]
  },
  
  // Global variables
  globals: {
    __DEV__: true,
    __TEST__: true
  },
  
  // Projects for different test suites
  projects: [
    {
      displayName: 'Unit Tests',
      testMatch: ['<rootDir>/tests/hive-mind/unit/**/*.test.js'],
      testEnvironment: 'node'
    },
    {
      displayName: 'Integration Tests',
      testMatch: ['<rootDir>/tests/hive-mind/integration/**/*.test.js'],
      testEnvironment: 'node',
      testTimeout: 60000 // 1 minute
    },
    {
      displayName: 'Performance Tests',
      testMatch: ['<rootDir>/tests/hive-mind/performance/**/*.test.js'],
      testEnvironment: 'node',
      testTimeout: 300000, // 5 minutes
      maxWorkers: 1 // Run serially for accurate measurements
    },
    {
      displayName: 'E2E Tests',
      testMatch: ['<rootDir>/tests/hive-mind/e2e/**/*.test.js'],
      testEnvironment: 'node',
      testTimeout: 600000, // 10 minutes
      maxWorkers: 1 // Run serially
    }
  ],
  
  // Watch plugins
  watchPlugins: [
    'jest-watch-typeahead/filename',
    'jest-watch-typeahead/testname'
  ],
  
  // Verbose output for CI
  verbose: process.env.CI === 'true'
};