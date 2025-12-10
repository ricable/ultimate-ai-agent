/**
 * Node.js Test Runner Configuration
 *
 * Configuration for Node.js built-in test runner
 */

module.exports = {
  // Test file patterns
  testMatch: [
    'tests/**/*.test.js',
    'tests/**/*.bench.js'
  ],

  // Test environment
  testEnvironment: 'node',

  // Coverage settings (when using c8)
  coverage: {
    enabled: process.env.COVERAGE === 'true',
    provider: 'c8',
    reporter: ['text', 'html', 'lcov'],
    all: true,
    include: [
      'qudag/qudag-napi/**/*.rs',
      'packages/daa-sdk/src/**/*.{ts,js}'
    ],
    exclude: [
      'tests/**',
      'node_modules/**',
      '**/dist/**',
      '**/target/**'
    ]
  },

  // Test execution settings
  timeout: 30000, // 30 seconds per test
  concurrency: 1, // Run tests sequentially by default

  // Reporter settings
  reporter: process.env.CI ? 'tap' : 'spec',

  // Retry failed tests
  retries: process.env.CI ? 2 : 0,

  // Watch mode settings
  watch: process.env.WATCH === 'true',
  watchIgnore: [
    '**/node_modules/**',
    '**/dist/**',
    '**/target/**',
    '**/coverage/**'
  ]
};
