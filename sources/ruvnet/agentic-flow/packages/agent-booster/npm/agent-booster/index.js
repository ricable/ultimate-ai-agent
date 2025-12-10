/**
 * Agent Booster - Auto-detecting Native/WASM Loader
 * Attempts to load native bindings first, falls back to WASM
 */

const os = require('os');
const path = require('path');

let agentBooster = null;
let loadedRuntime = null;

/**
 * Attempts to load native bindings for the current platform
 * @returns {Object|null} Native bindings or null if not available
 */
function tryLoadNative() {
  const platform = os.platform();
  const arch = os.arch();

  // Map Node.js platform/arch to package names
  const platformMap = {
    'darwin-arm64': '@agent-booster/native-darwin-arm64',
    'darwin-x64': '@agent-booster/native-darwin-x64',
    'linux-arm64': '@agent-booster/native-linux-arm64',
    'linux-x64': '@agent-booster/native-linux-x64',
    'win32-x64': '@agent-booster/native-win32-x64'
  };

  const packageName = platformMap[`${platform}-${arch}`];

  if (!packageName) {
    return null;
  }

  try {
    const nativeModule = require(packageName);
    console.log(`[Agent Booster] Loaded native bindings: ${packageName}`);
    return nativeModule;
  } catch (error) {
    // Native bindings not available or failed to load
    return null;
  }
}

/**
 * Attempts to load WASM bindings as fallback
 * @returns {Object|null} WASM bindings or null if not available
 */
function tryLoadWasm() {
  try {
    const wasmModule = require('@agent-booster/wasm');
    console.log('[Agent Booster] Loaded WASM bindings');
    return wasmModule;
  } catch (error) {
    return null;
  }
}

/**
 * Initializes Agent Booster with auto-detection
 * @returns {Object} Agent Booster instance
 * @throws {Error} If no runtime is available
 */
function initialize() {
  if (agentBooster) {
    return agentBooster;
  }

  // Try native first (best performance)
  agentBooster = tryLoadNative();
  if (agentBooster) {
    loadedRuntime = 'native';
    return agentBooster;
  }

  // Fall back to WASM (universal compatibility)
  agentBooster = tryLoadWasm();
  if (agentBooster) {
    loadedRuntime = 'wasm';
    return agentBooster;
  }

  // No runtime available
  throw new Error(
    'Agent Booster: No runtime available. Please ensure either native bindings ' +
    'or WASM package is installed. Run: npm install @agent-booster/wasm'
  );
}

/**
 * Gets the currently loaded runtime type
 * @returns {string|null} 'native', 'wasm', or null
 */
function getRuntime() {
  return loadedRuntime;
}

/**
 * Applies prompt optimization transformations
 * @param {string} prompt - The input prompt to optimize
 * @param {Object} options - Optimization options
 * @returns {Promise<string>} Optimized prompt
 */
async function apply(prompt, options = {}) {
  const instance = initialize();

  if (typeof instance.apply !== 'function') {
    throw new Error('Agent Booster: apply method not available in loaded runtime');
  }

  return instance.apply(prompt, options);
}

/**
 * Applies prompt optimization in batch mode
 * @param {string[]} prompts - Array of prompts to optimize
 * @param {Object} options - Optimization options
 * @returns {Promise<string[]>} Array of optimized prompts
 */
async function batchApply(prompts, options = {}) {
  const instance = initialize();

  if (typeof instance.batchApply !== 'function') {
    // Fall back to sequential processing if batch not available
    const results = [];
    for (const prompt of prompts) {
      results.push(await apply(prompt, options));
    }
    return results;
  }

  return instance.batchApply(prompts, options);
}

/**
 * Analyzes a prompt and returns optimization metrics
 * @param {string} prompt - The prompt to analyze
 * @returns {Promise<Object>} Analysis results
 */
async function analyze(prompt) {
  const instance = initialize();

  if (typeof instance.analyze !== 'function') {
    throw new Error('Agent Booster: analyze method not available in loaded runtime');
  }

  return instance.analyze(prompt);
}

/**
 * Gets version information
 * @returns {Object} Version info including runtime type
 */
function getVersion() {
  const instance = initialize();
  return {
    version: require('./package.json').version,
    runtime: loadedRuntime,
    coreVersion: instance.getVersion ? instance.getVersion() : 'unknown'
  };
}

// Export public API
module.exports = {
  apply,
  batchApply,
  analyze,
  getVersion,
  getRuntime,
  initialize
};
