module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src', '<rootDir>/tests', '<rootDir>/test'],
  testMatch: [
    '**/__tests__/**/*.+(ts|tsx|js)',
    '**/?(*.)+(spec|test).+(ts|tsx|js)',
    '**/test/**/*.test.js',
    '**/test/**/*.spec.js'
  ],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest',
    '^.+\\.js$': 'babel-jest'
  },
  collectCoverageFrom: [
    'src/**/*.{js,ts}',
    '!src/**/*.d.ts',
    '!src/**/index.ts',
    '!src/**/*.test.js',
    '!src/**/*.spec.js'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'text-summary', 'lcov', 'html'],
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 80,
      lines: 80,
      statements: 80
    }
  },
  moduleNameMapper: {
    // Handle WASM imports in tests
    '\\.wasm$': '<rootDir>/tests/__mocks__/wasmMock.js'
  },
  setupFilesAfterEnv: ['<rootDir>/tests/setup.ts'],
  testTimeout: 30000,
  verbose: true,
  forceExit: true,
  detectOpenHandles: true,
  maxWorkers: process.env.CI ? 2 : '50%',
  globals: {
    'ts-jest': {
      tsconfig: {
        // Allow JS in tests
        allowJs: true
      }
    }
  }
};