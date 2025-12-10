import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html', 'lcov'],
      exclude: [
        'node_modules/**',
        'dist/**',
        '**/*.test.ts',
        '**/*.spec.ts',
        '**/test-*.ts',
        'benchmarks/**',
        'scripts/**',
        'coverage/**',
        '**/*.config.ts',
        '**/db-fallback.ts', // Fallback type definitions
      ],
      include: ['src/**/*.ts'],
      statements: 80,
      branches: 75,
      functions: 80,
      lines: 80,
    },
    testTimeout: 30000,
    hookTimeout: 30000,
    setupFiles: ['./tests/setup.ts'],
    mockReset: true,
    restoreMocks: true,
    clearMocks: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
