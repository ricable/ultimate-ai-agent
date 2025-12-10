#!/usr/bin/env node

/**
 * SIMD Fix Verification Test
 * Tests that SIMD detection and operations are working correctly
 */

// Add package.json type: module or use .mjs extension
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function testSIMD() {
    console.log('🔬 RUV-SWARM SIMD Fix Verification');
    console.log('===================================\n');

    try {
        // Test 1: Load WASM module
        console.log('1️⃣ Loading WASM module...');
        
        const wasmPath = path.join(__dirname, 'crates/ruv-swarm-wasm/pkg-test');
        const wasmBinary = fs.readFileSync(path.join(wasmPath, 'ruv_swarm_wasm_bg.wasm'));
        
        // Dynamic import to handle ES modules
        const { default: init, ...wasm } = await import(path.join(wasmPath, 'ruv_swarm_wasm.js'));
        await init(wasmBinary);
        
        console.log('✅ WASM module loaded successfully\n');

        // Test 2: Check SIMD detection
        console.log('2️⃣ Testing SIMD detection...');
        
        if (!wasm.get_features) {
            throw new Error('get_features function not found');
        }
        
        const features = JSON.parse(wasm.get_features());
        console.log('SIMD support detected:', features.simd_support ? '✅ YES' : '❌ NO');
        
        if (features.simd_capabilities) {
            const simdCaps = JSON.parse(features.simd_capabilities);
            console.log('SIMD capabilities:', simdCaps);
        }
        console.log();

        // Test 3: Test SIMD operations
        console.log('3️⃣ Testing SIMD operations...');
        
        if (!wasm.SimdVectorOps) {
            console.log('❌ SimdVectorOps not available');
            return false;
        }
        
        const simdOps = new wasm.SimdVectorOps();
        
        // Test dot product
        const testA = [1, 2, 3, 4, 5, 6, 7, 8];
        const testB = [2, 3, 4, 5, 6, 7, 8, 9];
        const expected = 240; // 1*2 + 2*3 + 3*4 + 4*5 + 5*6 + 6*7 + 7*8 + 8*9
        
        const dotProduct = simdOps.dot_product(testA, testB);
        const dotProductCorrect = Math.abs(dotProduct - expected) < 0.001;
        
        console.log(`Dot product: ${dotProduct} (expected: ${expected}) ${dotProductCorrect ? '✅' : '❌'}`);
        
        // Test vector addition
        const addResult = simdOps.vector_add([1, 2, 3, 4], [5, 6, 7, 8]);
        const addExpected = [6, 8, 10, 12];
        const addCorrect = addResult.length === addExpected.length && 
                          addResult.every((v, i) => Math.abs(v - addExpected[i]) < 0.001);
        
        console.log(`Vector add: [${addResult.join(', ')}] ${addCorrect ? '✅' : '❌'}`);
        
        // Test activation functions
        const reluResult = simdOps.apply_activation([-2, -1, 0, 1, 2], 'relu');
        const reluExpected = [0, 0, 0, 1, 2];
        const reluCorrect = reluResult.length === reluExpected.length && 
                           reluResult.every((v, i) => Math.abs(v - reluExpected[i]) < 0.001);
        
        console.log(`ReLU activation: [${reluResult.join(', ')}] ${reluCorrect ? '✅' : '❌'}`);
        
        console.log();

        // Test 4: Performance benchmark
        console.log('4️⃣ Running performance benchmark...');
        
        if (wasm.SimdBenchmark) {
            const benchmark = new wasm.SimdBenchmark();
            const perfResult = benchmark.benchmark_dot_product(1000, 100);
            const perfData = JSON.parse(perfResult);
            
            console.log(`Benchmark results:`);
            console.log(`  SIMD time: ${perfData.simd_time?.toFixed(2)}ms`);
            console.log(`  Scalar time: ${perfData.scalar_time?.toFixed(2)}ms`);
            console.log(`  Speedup: ${perfData.speedup?.toFixed(2)}x ${perfData.speedup > 1.5 ? '✅' : '⚠️'}`);
        }
        
        console.log();

        // Summary
        const allTestsPassed = features.simd_support && dotProductCorrect && addCorrect && reluCorrect;
        
        console.log('📊 SUMMARY');
        console.log('==========');
        console.log(`SIMD Detection: ${features.simd_support ? '✅ WORKING' : '❌ BROKEN'}`);
        console.log(`SIMD Operations: ${(dotProductCorrect && addCorrect && reluCorrect) ? '✅ WORKING' : '❌ BROKEN'}`);
        console.log(`Overall Status: ${allTestsPassed ? '✅ SUCCESS' : '❌ NEEDS WORK'}`);
        
        return allTestsPassed;
        
    } catch (error) {
        console.error('❌ Test failed:', error.message);
        console.error(error.stack);
        return false;
    }
}

// Run the test
testSIMD().then(success => {
    process.exit(success ? 0 : 1);
}).catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
});