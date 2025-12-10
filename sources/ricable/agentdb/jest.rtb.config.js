module.exports = {
  displayName: 'RTB Hierarchical Template System',
  testMatch: [
    '<rootDir>/tests/rtb/**/*.test.ts'
  ],
  setupFilesAfterEnv: [
    '<rootDir>/tests/setup.ts'
  ],
  collectCoverageFrom: [
    'src/rtb/hierarchical-template-system/**/*.ts',
    'src/types/rtb-types.ts',
    '!src/**/*.d.ts',
    '!src/**/index.ts'
  ],
  coverageDirectory: 'coverage/rtb',
  coverageReporters: [
    'text',
    'lcov',
    'html',
    'json'
  ],
  coverageThresholds: {
    global: {
      branches: 85,
      functions: 85,
      lines: 85,
      statements: 85
    }
  },
  testEnvironment: 'node',
  transform: {
    '^.+\\.tsx?$': ['ts-jest', {
      tsconfig: '<rootDir>/tsconfig.json'
    }]
  },
  moduleFileExtensions: [
    'ts',
    'tsx',
    'js',
    'jsx',
    'json',
    'node'
  ],
  testTimeout: 30000, // 30 seconds for performance tests
  verbose: true,
  // Performance test specific configuration
  projects: [
    {
      displayName: 'Unit Tests',
      testMatch: [
        '<rootDir>/tests/rtb/hierarchical-template-system/priority-engine.test.ts',
        '<rootDir>/tests/rtb/hierarchical-template-system/template-merger.test.ts',
        '<rootDir>/tests/rtb/hierarchical-template-system/base-generator.test.ts',
        '<rootDir>/tests/rtb/hierarchical-template-system/variant-generators.test.ts',
        '<rootDir>/tests/rtb/hierarchical-template-system/frequency-relations.test.ts'
      ],
      testTimeout: 10000,
      maxWorkers: 4
    },
    {
      displayName: 'Integration Tests',
      testMatch: [
        '<rootDir>/tests/rtb/hierarchical-template-system/integration.test.ts'
      ],
      testTimeout: 60000,
      maxWorkers: 2
    },
    {
      displayName: 'Performance Tests',
      testMatch: [
        '<rootDir>/tests/rtb/hierarchical-template-system/performance.test.ts'
      ],
      testTimeout: 120000, // 2 minutes for performance tests
      maxWorkers: 1,
      setupFilesAfterEnv: [
        '<rootDir>/tests/rtb/performance-setup.ts'
      ]
    }
  ]
};