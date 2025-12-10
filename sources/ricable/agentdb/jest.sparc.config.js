module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/tests/sparc', '<rootDir>/src/sparc'],
  testMatch: [
    '**/sparc/**/*.+(test|spec).+(ts|tsx|js)'
  ],
  transform: {
    '^.+\\.(ts|tsx)$': 'ts-jest'
  },
  collectCoverageFrom: [
    'src/sparc/**/*.{ts,tsx}',
    '!src/sparc/**/*.d.ts'
  ],
  coverageDirectory: 'coverage/sparc',
  coverageReporters: ['text', 'lcov'],
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  testTimeout: 30000,
  verbose: true
};