/**
 * Platform detection and dynamic loading for DAA SDK
 *
 * Automatically detects the runtime environment and loads the appropriate
 * bindings (native NAPI-rs for Node.js, WASM for browsers).
 *
 * @module platform
 */

/**
 * Detect the current platform runtime
 *
 * @returns 'native' for Node.js with native bindings, 'wasm' for browsers or fallback
 */
export function detectPlatform(): 'native' | 'wasm' {
  // Check if running in Node.js
  if (typeof process !== 'undefined' && process.versions?.node) {
    try {
      // Try loading native addon to verify it's available
      require.resolve('@daa/qudag-native');
      return 'native';
    } catch {
      console.warn('⚠️  Native bindings not found, falling back to WASM');
      return 'wasm';
    }
  }

  // Browser or other environment
  return 'wasm';
}

/**
 * Get platform-specific performance characteristics
 */
export function getPlatformInfo() {
  const platform = detectPlatform();

  if (platform === 'native') {
    return {
      platform: 'native',
      runtime: 'Node.js',
      performance: 'high',
      threadingSupport: true,
      relativeSpeed: 1.0,
      features: [
        'Multi-threading',
        'Direct memory access',
        'Zero-copy operations',
        'Full async/await support',
      ],
    };
  } else {
    return {
      platform: 'wasm',
      runtime: 'WebAssembly',
      performance: 'good',
      threadingSupport: false,
      relativeSpeed: 0.4, // WASM typically 40% of native speed
      features: [
        'Cross-platform compatibility',
        'Browser support',
        'Sandboxed execution',
        'Memory safety',
      ],
    };
  }
}

/**
 * Load QuDAG bindings based on platform
 */
export async function loadQuDAG(platform?: 'native' | 'wasm') {
  const targetPlatform = platform || detectPlatform();

  if (targetPlatform === 'native') {
    try {
      console.log('📦 Loading QuDAG native bindings (@daa/qudag-native)...');
      // Import the wrapper module which loads the NAPI bindings
      const qudag = await import('./qudag');

      // Initialize the native module
      const initMsg = qudag.init();
      console.log(`✅ ${initMsg}`);

      return qudag;
    } catch (error) {
      console.warn('⚠️  Failed to load native bindings, falling back to WASM:', error);
      return loadQuDAGWasm();
    }
  } else {
    return loadQuDAGWasm();
  }
}

/**
 * Load QuDAG WASM bindings
 */
async function loadQuDAGWasm(): Promise<any> {
  console.log('📦 Loading QuDAG WASM bindings...');
  try {
    // Dynamic import to avoid TypeScript errors when package isn't installed
    const wasm = await import('qudag-wasm' as any);
    if (wasm.default) {
      await wasm.default(); // Initialize WASM module
    }
    return wasm;
  } catch (error) {
    throw new Error(`WASM bindings not available: ${error}`);
  }
}

/**
 * Load Orchestrator bindings based on platform
 *
 * TODO: Implement orchestrator bindings
 */
export async function loadOrchestrator(platform?: 'native' | 'wasm'): Promise<any> {
  const targetPlatform = platform || detectPlatform();

  if (targetPlatform === 'native') {
    try {
      console.log('📦 Loading Orchestrator native bindings...');
      // const native = await import('@daa/orchestrator-native');
      // return native;
      throw new Error('Orchestrator native bindings not yet implemented');
    } catch (error) {
      console.warn('⚠️  Orchestrator native bindings not available');
      throw new Error('Orchestrator WASM bindings not yet implemented');
    }
  } else {
    throw new Error('Orchestrator WASM bindings not yet implemented');
  }
}

/**
 * Load Prime ML bindings based on platform
 *
 * TODO: Implement Prime ML bindings
 */
export async function loadPrime(platform?: 'native' | 'wasm'): Promise<any> {
  const targetPlatform = platform || detectPlatform();

  if (targetPlatform === 'native') {
    try {
      console.log('📦 Loading Prime ML native bindings...');
      // const native = await import('@daa/prime-native');
      // return native;
      throw new Error('Prime ML native bindings not yet implemented');
    } catch (error) {
      console.warn('⚠️  Prime ML native bindings not available');
      throw new Error('Prime ML WASM bindings not yet implemented');
    }
  } else {
    throw new Error('Prime ML WASM bindings not yet implemented');
  }
}

/**
 * Check if a specific binding is available
 */
export async function isBindingAvailable(
  binding: 'qudag' | 'orchestrator' | 'prime',
  platform: 'native' | 'wasm'
): Promise<boolean> {
  try {
    if (binding === 'qudag') {
      await loadQuDAG(platform);
      return true;
    } else if (binding === 'orchestrator') {
      await loadOrchestrator(platform);
      return true;
    } else if (binding === 'prime') {
      await loadPrime(platform);
      return true;
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Get list of available bindings for current platform
 */
export async function getAvailableBindings(): Promise<{
  platform: 'native' | 'wasm';
  available: string[];
  unavailable: string[];
}> {
  const platform = detectPlatform();
  const bindings = ['qudag', 'orchestrator', 'prime'] as const;

  const results = await Promise.all(
    bindings.map(async (binding) => ({
      binding,
      available: await isBindingAvailable(binding, platform),
    }))
  );

  return {
    platform,
    available: results.filter((r) => r.available).map((r) => r.binding),
    unavailable: results.filter((r) => !r.available).map((r) => r.binding),
  };
}
