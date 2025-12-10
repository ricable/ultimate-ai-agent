/**
 * Jest configuration for Synaptic Neural Mesh test suite
 * Supports ES modules, WASM, and comprehensive coverage
 */

export default {
  // Test environment
  testEnvironment: 'node',
  
  // Enable ES modules
  extensionsToTreatAsEsm: ['.js'],
  globals: {
    'ts-jest': {
      useESM: true
    }
  },
  
  // Module resolution
  moduleNameMapping: {
    '^(\\.{1,2}/.*)\\.js$': '$1'
  },
  
  // Test file patterns
  testMatch: [
    '**/unit/**/*.test.js',
    '**/integration/**/*.test.js',
    '**/e2e/**/*.test.js'
  ],
  
  // Coverage configuration
  collectCoverageFrom: [
    'src/**/*.js',
    'src/**/*.ts',
    '!src/**/*.test.js',
    '!src/**/*.spec.js',
    '!src/**/node_modules/**',
    '!src/**/coverage/**',
    '!src/**/dist/**',
    '!src/**/build/**'
  ],
  
  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    // Component-specific thresholds
    'src/js/claude-flow/': {
      branches: 95,
      functions: 95,
      lines: 95,
      statements: 95
    },
    'src/js/ruv-swarm/': {
      branches: 90,
      functions: 90,
      lines: 90,
      statements: 90
    }
  },
  
  // Coverage reporters
  coverageReporters: [
    'text',
    'text-summary',
    'html',
    'lcov',
    'json',
    'clover'
  ],
  
  // Coverage directory
  coverageDirectory: 'coverage',
  
  // Setup files
  setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
  
  // Test timeout
  testTimeout: 30000,
  
  // Parallel testing
  maxWorkers: '50%',
  
  // Transform configuration for WASM and other assets
  transform: {
    '^.+\\.js$': ['babel-jest', {
      presets: [
        ['@babel/preset-env', {
          targets: { node: 'current' },
          modules: false
        }]
      ]
    }]
  },
  
  // Module file extensions
  moduleFileExtensions: ['js', 'json', 'wasm'],
  
  // Test result processors
  reporters: [
    'default',
    ['jest-html-reporters', {
      publicPath: './tests/reports/html',
      filename: 'test-report.html',
      expand: true,
      hideIcon: false,
      pageTitle: 'Synaptic Neural Mesh Test Report'
    }],
    ['jest-junit', {
      outputDirectory: './tests/reports',
      outputName: 'junit.xml',
      suiteName: 'Synaptic Neural Mesh Tests',
      classNameTemplate: '{classname}',
      titleTemplate: '{title}',
      ancestorSeparator: ' › ',
      usePathForSuiteName: true
    }]
  ],
  
  // Performance budgets
  slowTestThreshold: 5,
  
  // Error handling
  errorOnDeprecated: true,
  
  // Verbose output for debugging
  verbose: false,
  
  // Watch mode configuration
  watchman: true,
  watchPathIgnorePatterns: [
    '<rootDir>/node_modules/',
    '<rootDir>/coverage/',
    '<rootDir>/dist/',
    '<rootDir>/build/',
    '<rootDir>/.git/'
  ],
  
  // Mock configuration
  clearMocks: true,
  restoreMocks: true,
  
  // Global setup and teardown
  globalSetup: '<rootDir>/tests/global-setup.js',
  globalTeardown: '<rootDir>/tests/global-teardown.js'
};