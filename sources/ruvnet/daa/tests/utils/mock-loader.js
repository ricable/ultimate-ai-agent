/**
 * Mock Module Loader
 *
 * Dynamically loads native or WASM bindings, with fallback to mocks for testing
 */

/**
 * Load QuDAG module (native or WASM)
 * Falls back to mock if neither is available
 */
async function loadQuDAG() {
  // Try native first
  try {
    const native = require('../../qudag/qudag-napi');
    console.log('Loaded QuDAG native bindings');
    return native;
  } catch (nativeError) {
    console.log('Native bindings not available:', nativeError.message);
  }

  // Try WASM
  try {
    const wasm = await import('qudag-wasm');
    console.log('Loaded QuDAG WASM bindings');
    return wasm;
  } catch (wasmError) {
    console.log('WASM bindings not available:', wasmError.message);
  }

  // Fall back to mock
  console.log('Using QuDAG mocks for testing');
  const { createMockQuDAG } = require('./test-helpers');
  return createMockQuDAG();
}

/**
 * Load orchestrator module
 * Falls back to mock if not available
 */
async function loadOrchestrator() {
  try {
    const orchestrator = require('../../packages/daa-sdk/dist/orchestrator');
    console.log('Loaded orchestrator bindings');
    return orchestrator;
  } catch (error) {
    console.log('Orchestrator bindings not available:', error.message);
    console.log('Using orchestrator mocks for testing');

    return {
      Orchestrator: class {
        constructor() {}
        async start() { return { status: 'running' }; }
        async stop() { return { status: 'stopped' }; }
        async monitor() {
          return {
            status: 'healthy',
            agents: 0,
            tasks: 0,
            uptime: 0
          };
        }
      }
    };
  }
}

/**
 * Load Prime ML module
 * Falls back to mock if not available
 */
async function loadPrime() {
  try {
    const prime = require('../../prime-rust/bindings');
    console.log('Loaded Prime ML bindings');
    return prime;
  } catch (error) {
    console.log('Prime ML bindings not available:', error.message);
    console.log('Using Prime ML mocks for testing');

    return {
      TrainingNode: class {
        constructor() {}
        async initTraining() { return { sessionId: 'mock-session' }; }
        async trainEpoch() { return { loss: 0.5, accuracy: 0.8 }; }
      },
      Coordinator: class {
        constructor() {}
        async registerNode() {}
        async startTraining() { return 'mock-training-id'; }
      }
    };
  }
}

/**
 * Detect which bindings are available
 */
function detectAvailableBindings() {
  const available = {
    native: false,
    wasm: false,
    orchestrator: false,
    prime: false
  };

  // Check native
  try {
    require('../../qudag/qudag-napi');
    available.native = true;
  } catch {}

  // Check WASM (synchronous check)
  try {
    require.resolve('qudag-wasm');
    available.wasm = true;
  } catch {}

  // Check orchestrator
  try {
    require('../../packages/daa-sdk/dist/orchestrator');
    available.orchestrator = true;
  } catch {}

  // Check Prime ML
  try {
    require('../../prime-rust/bindings');
    available.prime = true;
  } catch {}

  return available;
}

/**
 * Get recommended platform based on environment
 */
function getRecommendedPlatform() {
  if (typeof process !== 'undefined' && process.versions?.node) {
    // Node.js environment - prefer native
    return 'native';
  }

  if (typeof window !== 'undefined') {
    // Browser environment - use WASM
    return 'wasm';
  }

  return 'mock';
}

module.exports = {
  loadQuDAG,
  loadOrchestrator,
  loadPrime,
  detectAvailableBindings,
  getRecommendedPlatform
};
