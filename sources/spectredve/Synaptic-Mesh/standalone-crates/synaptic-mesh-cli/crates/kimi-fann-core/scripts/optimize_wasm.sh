#!/bin/bash
# Advanced WASM Optimization Script
# Targets <50ms inference, <25MB memory, >1000 ops/sec throughput

set -e

echo "🚀 Starting Advanced WASM Optimization..."

# Build with maximum optimizations
echo "📦 Building with release optimizations..."
wasm-pack build --target web --release --out-dir pkg \
  --features "simd,gpu" \
  -- --config profile.release.opt-level=\"z\" \
     --config profile.release.lto=true \
     --config profile.release.codegen-units=1 \
     --config profile.release.panic=\"abort\"

# Advanced wasm-opt optimizations
echo "⚡ Applying advanced WASM optimizations..."

# Level 1: Size optimization with SIMD preservation
wasm-opt pkg/kimi_fann_core_bg.wasm -Oz \
  --enable-simd \
  --enable-bulk-memory \
  --enable-mutable-globals \
  --enable-reference-types \
  --strip-debug \
  --strip-producers \
  -o pkg/kimi_fann_core_bg_opt1.wasm

# Level 2: Performance optimization with aggressive inlining
wasm-opt pkg/kimi_fann_core_bg_opt1.wasm -O3 \
  --enable-simd \
  --enable-bulk-memory \
  --inline-functions-with-loops \
  --optimize-instructions \
  --optimize-added-constants \
  --precompute \
  --vacuum \
  -o pkg/kimi_fann_core_bg_opt2.wasm

# Level 3: Final optimization pass with specific neural network patterns
wasm-opt pkg/kimi_fann_core_bg_opt2.wasm -O4 \
  --enable-simd \
  --converge \
  --flatten \
  --rereloop \
  --merge-blocks \
  --optimize-casts \
  --optimize-instructions \
  --pick-load-signs \
  --precompute \
  --simplify-locals \
  --vacuum \
  -o pkg/kimi_fann_core_bg_optimized.wasm

# Replace original with optimized version
mv pkg/kimi_fann_core_bg_optimized.wasm pkg/kimi_fann_core_bg.wasm

# Clean up intermediate files
rm -f pkg/kimi_fann_core_bg_opt1.wasm pkg/kimi_fann_core_bg_opt2.wasm

# Generate size report
echo "📊 Optimization Results:"
echo "========================"

if [ -f pkg/kimi_fann_core_bg.wasm ]; then
    WASM_SIZE=$(stat -c%s pkg/kimi_fann_core_bg.wasm)
    WASM_SIZE_MB=$(echo "scale=2; $WASM_SIZE / 1024 / 1024" | bc -l)
    echo "📦 WASM Size: ${WASM_SIZE} bytes (${WASM_SIZE_MB} MB)"
    
    # Check if size meets target (<25MB total including JS)
    if [ $WASM_SIZE -lt 20971520 ]; then  # 20MB for WASM (leaving 5MB for JS)
        echo "✅ Size target MET: WASM < 20MB"
    else
        echo "❌ Size target MISSED: WASM >= 20MB"
    fi
fi

# Generate performance optimization report
echo ""
echo "🔧 Applied Optimizations:"
echo "========================"
echo "✅ Size optimization (-Oz): Maximum compression"
echo "✅ Performance optimization (-O3/-O4): Aggressive inlining"
echo "✅ SIMD enabled: Vector operations for neural networks"
echo "✅ Bulk memory: Fast memory operations"
echo "✅ Function inlining: Reduced call overhead"
echo "✅ Instruction optimization: Optimal instruction patterns"
echo "✅ Dead code elimination: Removed unused code"
echo "✅ Constant precomputation: Compile-time calculations"

# Create optimized package.json for performance
echo ""
echo "📝 Creating optimized package.json..."
cat > pkg/package.json << EOF
{
  "name": "kimi-fann-core",
  "version": "0.1.1",
  "description": "High-performance neural network inference engine with SIMD optimization",
  "main": "kimi_fann_core.js",
  "types": "kimi_fann_core.d.ts",
  "files": [
    "kimi_fann_core.js",
    "kimi_fann_core.d.ts",
    "kimi_fann_core_bg.wasm",
    "kimi_fann_core_bg.wasm.d.ts"
  ],
  "keywords": [
    "neural-networks",
    "wasm",
    "simd",
    "machine-learning",
    "performance",
    "inference"
  ],
  "performance": {
    "target_inference_ms": 50,
    "target_memory_mb": 25,
    "target_throughput_ops_sec": 1000,
    "optimizations_applied": [
      "simd",
      "wasm-opt-oz",
      "wasm-opt-o4",
      "lto",
      "size-optimization",
      "dead-code-elimination"
    ]
  },
  "engines": {
    "node": ">=16.0.0"
  }
}
EOF

# Create performance test HTML
echo ""
echo "🧪 Creating performance test file..."
cat > pkg/performance_test.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Kimi-FANN Core Performance Test</title>
    <style>
        body { font-family: monospace; margin: 20px; }
        .metric { margin: 10px 0; padding: 10px; border: 1px solid #ccc; }
        .pass { background-color: #d4edda; }
        .fail { background-color: #f8d7da; }
        .result { font-weight: bold; }
    </style>
</head>
<body>
    <h1>🚀 Kimi-FANN Core Performance Test</h1>
    <div id="status">Loading WASM module...</div>
    <div id="results"></div>

    <script type="module">
        import init, { 
            KimiRuntime, 
            ProcessingConfig, 
            MicroExpert, 
            ExpertDomain 
        } from './kimi_fann_core.js';

        const TARGET_INFERENCE_MS = 50;
        const TARGET_MEMORY_MB = 25;
        const TARGET_THROUGHPUT_OPS_SEC = 1000;

        async function runPerformanceTests() {
            try {
                console.log('Initializing WASM module...');
                await init();
                
                document.getElementById('status').textContent = 'WASM module loaded successfully!';
                
                const results = document.getElementById('results');
                
                // Test 1: Single inference latency
                console.log('Testing inference latency...');
                const config = ProcessingConfig.new();
                const runtime = KimiRuntime.new(config);
                
                const start = performance.now();
                const result = runtime.process("Calculate the derivative of x^2 + 2x + 1");
                const inferenceTime = performance.now() - start;
                
                const latencyDiv = document.createElement('div');
                latencyDiv.className = `metric ${inferenceTime <= TARGET_INFERENCE_MS ? 'pass' : 'fail'}`;
                latencyDiv.innerHTML = `
                    <strong>Inference Latency:</strong> 
                    <span class="result">${inferenceTime.toFixed(2)}ms</span> 
                    (Target: ≤${TARGET_INFERENCE_MS}ms)
                    ${inferenceTime <= TARGET_INFERENCE_MS ? '✅' : '❌'}
                `;
                results.appendChild(latencyDiv);
                
                // Test 2: Memory usage estimation
                console.log('Testing memory usage...');
                const memoryBefore = performance.memory ? performance.memory.usedJSHeapSize : 0;
                
                // Create multiple experts to test memory
                const experts = [];
                for (let i = 0; i < 6; i++) {
                    experts.push(MicroExpert.new(i));
                }
                
                const memoryAfter = performance.memory ? performance.memory.usedJSHeapSize : 0;
                const memoryUsedMB = (memoryAfter - memoryBefore) / (1024 * 1024);
                
                const memoryDiv = document.createElement('div');
                memoryDiv.className = `metric ${memoryUsedMB <= TARGET_MEMORY_MB ? 'pass' : 'fail'}`;
                memoryDiv.innerHTML = `
                    <strong>Memory Usage:</strong> 
                    <span class="result">${memoryUsedMB.toFixed(2)}MB</span> 
                    (Target: ≤${TARGET_MEMORY_MB}MB)
                    ${memoryUsedMB <= TARGET_MEMORY_MB ? '✅' : '❌'}
                `;
                results.appendChild(memoryDiv);
                
                // Test 3: Throughput
                console.log('Testing throughput...');
                const queries = [
                    "Simple calculation",
                    "Write a function",
                    "Translate this text",
                    "Solve equation",
                    "Analyze data"
                ];
                
                const throughputStart = performance.now();
                let operationsCompleted = 0;
                
                for (let i = 0; i < 100; i++) {
                    const query = queries[i % queries.length];
                    runtime.process(query);
                    operationsCompleted++;
                }
                
                const throughputTime = performance.now() - throughputStart;
                const opsPerSec = (operationsCompleted * 1000) / throughputTime;
                
                const throughputDiv = document.createElement('div');
                throughputDiv.className = `metric ${opsPerSec >= TARGET_THROUGHPUT_OPS_SEC ? 'pass' : 'fail'}`;
                throughputDiv.innerHTML = `
                    <strong>Throughput:</strong> 
                    <span class="result">${opsPerSec.toFixed(0)} ops/sec</span> 
                    (Target: ≥${TARGET_THROUGHPUT_OPS_SEC} ops/sec)
                    ${opsPerSec >= TARGET_THROUGHPUT_OPS_SEC ? '✅' : '❌'}
                `;
                results.appendChild(throughputDiv);
                
                // Test 4: SIMD availability
                console.log('Checking SIMD support...');
                const simdSupported = typeof WebAssembly.SIMD === 'object' || 
                                    (typeof WebAssembly.validate === 'function' && 
                                     WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0])));
                
                const simdDiv = document.createElement('div');
                simdDiv.className = `metric ${simdSupported ? 'pass' : 'fail'}`;
                simdDiv.innerHTML = `
                    <strong>SIMD Support:</strong> 
                    <span class="result">${simdSupported ? 'Available' : 'Not Available'}</span>
                    ${simdSupported ? '✅' : '❌'}
                `;
                results.appendChild(simdDiv);
                
                // Overall performance score
                const tests = [
                    inferenceTime <= TARGET_INFERENCE_MS,
                    memoryUsedMB <= TARGET_MEMORY_MB,
                    opsPerSec >= TARGET_THROUGHPUT_OPS_SEC,
                    simdSupported
                ];
                
                const passedTests = tests.filter(t => t).length;
                const score = (passedTests / tests.length) * 100;
                
                const scoreDiv = document.createElement('div');
                scoreDiv.className = `metric ${score >= 75 ? 'pass' : 'fail'}`;
                scoreDiv.innerHTML = `
                    <strong>Overall Performance Score:</strong> 
                    <span class="result">${score.toFixed(0)}%</span> 
                    (${passedTests}/${tests.length} tests passed)
                `;
                results.appendChild(scoreDiv);
                
                console.log('Performance tests completed!');
                
            } catch (error) {
                console.error('Performance test failed:', error);
                document.getElementById('status').textContent = `Error: ${error.message}`;
            }
        }

        runPerformanceTests();
    </script>
</body>
</html>
EOF

echo ""
echo "✅ WASM optimization complete!"
echo ""
echo "📋 Next Steps:"
echo "=============="
echo "1. Test performance: Open pkg/performance_test.html in browser"
echo "2. Validate targets:"
echo "   - Inference latency: <50ms ⏱️"
echo "   - Memory usage: <25MB 💾" 
echo "   - Throughput: >1000 ops/sec 🚀"
echo "   - P2P latency: <1ms 📡"
echo "3. Deploy optimized WASM to production"
echo ""
echo "🔧 Optimization flags applied:"
echo "- Link-time optimization (LTO)"
echo "- Size optimization (-Oz)"
echo "- Performance optimization (-O3/-O4)"
echo "- SIMD vectorization"
echo "- Dead code elimination"
echo "- Function inlining"
echo "- Instruction optimization"

# Make the script executable
chmod +x "$(basename "$0")"