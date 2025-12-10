/**
 * ruvector - High-performance vector database for Node.js
 *
 * This package automatically detects and uses the best available implementation:
 * 1. Native (Rust-based, fastest) - if available for your platform
 * 2. WASM (WebAssembly, universal fallback) - works everywhere
 *
 * Also provides safe wrappers for GNN and Attention modules that handle
 * array type conversions automatically.
 */

export * from './types';

// Export core wrappers (safe interfaces with automatic type conversion)
export * from './core';
export * from './services';

let implementation: any;
let implementationType: 'native' | 'wasm' = 'wasm';

try {
  // Try to load native module first
  implementation = require('@ruvector/core');
  implementationType = 'native';

  // Verify it's actually working
  if (typeof implementation.VectorDB !== 'function') {
    throw new Error('Native module loaded but VectorDB not found');
  }
} catch (e: any) {
  // No WASM fallback available yet
  throw new Error(
    `Failed to load ruvector native module.\n` +
    `Error: ${e.message}\n` +
    `\nSupported platforms:\n` +
    `- Linux x64/ARM64\n` +
    `- macOS Intel/Apple Silicon\n` +
    `- Windows x64\n` +
    `\nIf you're on a supported platform, try:\n` +
    `  npm install --force @ruvector/core`
  );
}

/**
 * Get the current implementation type
 */
export function getImplementationType(): 'native' | 'wasm' {
  return implementationType;
}

/**
 * Check if native implementation is being used
 */
export function isNative(): boolean {
  return implementationType === 'native';
}

/**
 * Check if WASM implementation is being used
 */
export function isWasm(): boolean {
  return implementationType === 'wasm';
}

/**
 * Get version information
 */
export function getVersion(): { version: string; implementation: string } {
  const pkg = require('../package.json');
  return {
    version: pkg.version,
    implementation: implementationType
  };
}

// Export the VectorDB class
export const VectorDB = implementation.VectorDB;

// Export everything from the implementation
export default implementation;
