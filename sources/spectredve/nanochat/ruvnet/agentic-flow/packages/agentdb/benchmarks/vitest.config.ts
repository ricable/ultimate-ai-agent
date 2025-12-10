/**
 * Vitest Configuration for AgentDB v2 Benchmarks
 */

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    testTimeout: 300000, // 5 minutes for long-running benchmarks
    hookTimeout: 60000,
    benchmark: {
      include: ['**/*.bench.ts'],
      exclude: ['node_modules', 'dist'],
      outputFile: './benchmark-results.json'
    }
  },
  resolve: {
    alias: {
      '@': './src'
    }
  }
});
