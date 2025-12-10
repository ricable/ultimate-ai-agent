/**
 * Vitest Quick Benchmark Configuration
 * Reduced iterations for fast smoke testing
 */

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 60000, // 1 minute for quick tests
    hookTimeout: 30000,
    benchmark: {
      include: ['**/*.bench.ts'],
      exclude: ['node_modules', 'dist'],
      outputFile: './benchmark-results-quick.json'
    }
  },
  resolve: {
    alias: {
      '@': './src'
    }
  },
  define: {
    // Override default iterations for quick testing
    'DEFAULT_CONFIG.warmupIterations': 3,
    'DEFAULT_CONFIG.iterations': 10
  }
});
