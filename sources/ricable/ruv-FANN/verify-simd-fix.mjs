#!/usr/bin/env node

/**
 * Verify SIMD fix works by running fresh benchmark
 */

import { RuvSwarm } from './ruv-swarm/npm/src/index-enhanced.js';
import { EnhancedMCPTools } from './ruv-swarm/npm/src/mcp-tools-enhanced.js';

console.log('🧪 Verifying SIMD Fix with Fresh MCP Tools...\n');

// Clear any global cache
delete global._ruvSwarmInstance;
global._ruvSwarmInitialized = 0;

try {
    // Create fresh instance with SIMD enabled
    const ruvSwarm = await RuvSwarm.initialize({
        useSIMD: true,
        debug: false
    });
    
    console.log('✅ Fresh RuvSwarm instance created');
    console.log('   SIMD Support:', ruvSwarm.features.simd_support);
    
    // Create fresh MCP tools instance
    const mcpTools = new EnhancedMCPTools(ruvSwarm);
    
    // Run features detection
    console.log('\n🔍 Running fresh features detection...');
    const features = await mcpTools.features_detect({ category: 'all' });
    
    console.log('   WASM SIMD Support:', features.wasm.simd_support);
    console.log('   RuvSwarm SIMD Support:', features.ruv_swarm.simd_support);
    console.log('   Runtime SIMD Support:', features.runtime.simd);
    
    // Run a quick benchmark
    console.log('\n⚡ Running fresh benchmark...');
    const benchmark = await mcpTools.benchmark_run({ type: 'all', iterations: 3 });
    
    console.log('   Environment SIMD Support:', benchmark.environment.features.simd_support);
    console.log('   Runtime SIMD Support:', benchmark.environment.runtime_features.simd);
    
    if (benchmark.environment.features.simd_support) {
        console.log('\n🎉 SUCCESS: SIMD support is now properly detected!');
    } else {
        console.log('\n⚠️  ISSUE: SIMD support still showing as false in benchmark');
    }
    
} catch (error) {
    console.error('❌ Error:', error.message);
}