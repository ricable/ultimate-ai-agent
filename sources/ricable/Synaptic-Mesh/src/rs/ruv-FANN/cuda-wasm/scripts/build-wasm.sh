#!/bin/bash

# Optimized CUDA-Rust-WASM Build Script
# Advanced WebAssembly compilation with performance optimizations and caching

set -e

echo "🚀 Building CUDA-Rust-WASM for WebAssembly with optimizations..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
BUILD_MODE="${BUILD_MODE:-release}"
ENABLE_SIMD="${ENABLE_SIMD:-true}"
ENABLE_THREADS="${ENABLE_THREADS:-false}"
OPTIMIZE_SIZE="${OPTIMIZE_SIZE:-true}"
ENABLE_CACHE="${ENABLE_CACHE:-true}"
PARALLEL_JOBS="${PARALLEL_JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Performance timing
START_TIME=$(date +%s)

# Check for required tools with versions
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        echo "Please install $1 and try again"
        exit 1
    else
        local version=$($1 --version 2>/dev/null | head -n1 || echo "version unknown")
        echo -e "${GREEN}✅ $1 found: ${version}${NC}"
    fi
}

echo -e "${BLUE}📋 Checking dependencies and versions...${NC}"
check_tool cargo
check_tool wasm-pack
check_tool wasm-opt

# Check for optional tools
check_optional_tool() {
    if command -v $1 &> /dev/null; then
        local version=$($1 --version 2>/dev/null | head -n1 || echo "version unknown")
        echo -e "${GREEN}✅ $1 found: ${version}${NC}"
        return 0
    else
        echo -e "${YELLOW}⚠️  $1 not found (optional)${NC}"
        return 1
    fi
}

echo -e "${BLUE}📋 Checking optional tools...${NC}"
HAS_BINARYEN=$(check_optional_tool wasm2wat && echo "true" || echo "false")
HAS_WABT=$(check_optional_tool wat2wasm && echo "true" || echo "false")
HAS_TWIGGY=$(check_optional_tool twiggy && echo "true" || echo "false")

# Set up build environment
echo -e "${BLUE}🔧 Configuring build environment...${NC}"
export RUSTFLAGS="-C target-feature=+bulk-memory,+mutable-globals"
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER="wasm-ld"

if [ "$ENABLE_SIMD" = "true" ]; then
    echo -e "${BLUE}🧮 Enabling WASM SIMD...${NC}"
    export RUSTFLAGS="$RUSTFLAGS -C target-feature=+simd128"
fi

if [ "$ENABLE_CACHE" = "true" ]; then
    export CARGO_INCREMENTAL=1
    echo -e "${BLUE}💾 Build caching enabled${NC}"
fi

# Clean previous builds (with cache preservation option)
if [ "$ENABLE_CACHE" = "false" ] || [ "$1" = "clean" ]; then
    echo -e "${BLUE}🧹 Cleaning previous builds...${NC}"
    rm -rf pkg dist target/wasm32-unknown-unknown
    cargo clean
else
    echo -e "${BLUE}🧹 Selective cleanup (preserving cache)...${NC}"
    rm -rf pkg dist
fi

# Create output directories
mkdir -p pkg dist

# Build Rust project for WASM with profile selection
echo -e "${BLUE}🔨 Building Rust project for WASM (${BUILD_MODE} mode)...${NC}"
if [ "$BUILD_MODE" = "release" ]; then
    if [ "$OPTIMIZE_SIZE" = "true" ]; then
        cargo build --profile wasm-size --target wasm32-unknown-unknown --features wasm-simd -j $PARALLEL_JOBS
    else
        cargo build --release --target wasm32-unknown-unknown --features wasm-simd -j $PARALLEL_JOBS
    fi
else
    cargo build --profile wasm-dev --target wasm32-unknown-unknown --features wasm-simd -j $PARALLEL_JOBS
fi

# Use wasm-pack with optimized settings
echo -e "${BLUE}📦 Generating WASM bindings with wasm-pack...${NC}"
WASM_PACK_ARGS="--target web --out-dir pkg"

if [ "$BUILD_MODE" = "release" ]; then
    if [ "$OPTIMIZE_SIZE" = "true" ]; then
        WASM_PACK_ARGS="$WASM_PACK_ARGS --profiling"
    else
        WASM_PACK_ARGS="$WASM_PACK_ARGS --release"
    fi
else
    WASM_PACK_ARGS="$WASM_PACK_ARGS --dev"
fi

# Add SIMD support if enabled
if [ "$ENABLE_SIMD" = "true" ]; then
    WASM_PACK_ARGS="$WASM_PACK_ARGS -- --features wasm-simd"
fi

wasm-pack build $WASM_PACK_ARGS

# Multi-stage WASM optimization
echo -e "${BLUE}⚡ Optimizing WASM binary with multiple passes...${NC}"
WASM_FILE="pkg/cuda_rust_wasm_bg.wasm"
OPTIMIZED_FILE="pkg/cuda_rust_wasm_bg_optimized.wasm"

# Get original size
ORIGINAL_SIZE=$(stat -f%z "$WASM_FILE" 2>/dev/null || stat -c%s "$WASM_FILE" 2>/dev/null || echo "0")

# Optimization passes
echo -e "${PURPLE}🔧 Pass 1: Size optimization...${NC}"
wasm-opt -Oz --enable-bulk-memory --enable-mutable-globals \
    $( [ "$ENABLE_SIMD" = "true" ] && echo "--enable-simd" || echo "" ) \
    -o "$OPTIMIZED_FILE" "$WASM_FILE"

if [ "$BUILD_MODE" = "release" ]; then
    echo -e "${PURPLE}🔧 Pass 2: Speed optimization...${NC}"
    wasm-opt -O3 --enable-bulk-memory --enable-mutable-globals \
        $( [ "$ENABLE_SIMD" = "true" ] && echo "--enable-simd" || echo "" ) \
        --fast-math --closed-world \
        -o "${OPTIMIZED_FILE}.tmp" "$OPTIMIZED_FILE"
    mv "${OPTIMIZED_FILE}.tmp" "$OPTIMIZED_FILE"

    echo -e "${PURPLE}🔧 Pass 3: Final cleanup...${NC}"
    wasm-opt -Oz --enable-bulk-memory --enable-mutable-globals \
        $( [ "$ENABLE_SIMD" = "true" ] && echo "--enable-simd" || echo "" ) \
        --vacuum --remove-unused-names --remove-unused-nonfunction-module-elements \
        -o "${OPTIMIZED_FILE}.final" "$OPTIMIZED_FILE"
    mv "${OPTIMIZED_FILE}.final" "$OPTIMIZED_FILE"
fi

# Replace original with optimized
mv "$OPTIMIZED_FILE" "$WASM_FILE"

# Get optimized size and calculate improvement
OPTIMIZED_SIZE=$(stat -f%z "$WASM_FILE" 2>/dev/null || stat -c%s "$WASM_FILE" 2>/dev/null || echo "0")
if [ "$ORIGINAL_SIZE" -gt 0 ]; then
    SAVINGS=$((ORIGINAL_SIZE - OPTIMIZED_SIZE))
    PERCENT_SAVINGS=$((SAVINGS * 100 / ORIGINAL_SIZE))
    echo -e "${GREEN}📊 Size reduction: $SAVINGS bytes (${PERCENT_SAVINGS}%)${NC}"
fi

# Analysis and profiling (if tools available)
if [ "$HAS_TWIGGY" = "true" ] && [ "$BUILD_MODE" = "release" ]; then
    echo -e "${BLUE}🔍 Analyzing WASM binary with twiggy...${NC}"
    twiggy top "$WASM_FILE" -n 10 > dist/size-analysis.txt
    echo -e "${GREEN}📊 Size analysis saved to dist/size-analysis.txt${NC}"
fi

# WASM validation and testing
echo -e "${BLUE}🧪 Validating WASM binary...${NC}"
if command -v wasm-validate &> /dev/null; then
    if wasm-validate "$WASM_FILE"; then
        echo -e "${GREEN}✅ WASM binary is valid${NC}"
    else
        echo -e "${RED}❌ WASM binary validation failed${NC}"
        exit 1
    fi
fi

# Create dist directory
mkdir -p dist

# Enhanced TypeScript definitions with performance APIs
echo -e "${BLUE}📝 Generating enhanced TypeScript definitions...${NC}"
cat > dist/index.d.ts << 'EOF'
// Enhanced TypeScript definitions for CUDA-Rust-WASM with performance monitoring

export interface TranspileOptions {
  target?: 'wasm' | 'webgpu';
  optimize?: boolean;
  profile?: boolean;
  simdEnabled?: boolean;
  parallelism?: number;
  cacheEnabled?: boolean;
}

export interface TranspileResult {
  code: string;
  wasmBinary?: Uint8Array;
  profile?: ProfileData;
  metrics?: CompilationMetrics;
}

export interface ProfileData {
  parseTime: number;
  transpileTime: number;
  optimizeTime: number;
  totalTime: number;
  memoryUsage: number;
  cacheHits: number;
}

export interface CompilationMetrics {
  originalSize: number;
  optimizedSize: number;
  compressionRatio: number;
  optimizationPasses: number;
  simdInstructions: number;
}

export interface KernelAnalysis {
  memoryPattern: string;
  threadUtilization: number;
  sharedMemoryUsage: number;
  registerUsage: number;
  suggestions: string[];
  performanceScore: number;
  bottlenecks: string[];
}

export interface BenchmarkResult {
  avgTime: number;
  minTime: number;
  maxTime: number;
  throughput: number;
  memoryBandwidth: number;
  efficiency: number;
}

export interface PerformanceMonitor {
  startTiming(operation: string): void;
  endTiming(operation: string): number;
  getMetrics(): Record<string, number>;
  reset(): void;
}

export interface CacheManager {
  get(key: string): Promise<any>;
  set(key: string, value: any, ttl?: number): Promise<void>;
  invalidate(pattern?: string): Promise<void>;
  getStats(): { hits: number; misses: number; size: number };
}

// Core API functions
export function transpileCuda(code: string, options?: TranspileOptions): Promise<TranspileResult>;
export function analyzeKernel(code: string): Promise<KernelAnalysis>;
export function benchmark(code: string, options?: { iterations?: number; warmup?: number }): Promise<BenchmarkResult>;

// Performance and monitoring
export function createPerformanceMonitor(): PerformanceMonitor;
export function getCacheManager(): CacheManager;
export function getSystemCapabilities(): Promise<{
  hasWebGPU: boolean;
  hasWebAssembly: boolean;
  hasSIMD: boolean;
  hasThreads: boolean;
  maxMemory: number;
  cores: number;
}>;

// WebGPU specific exports
export interface WebGPUKernel {
  dispatch(x: number, y?: number, z?: number): Promise<void>;
  setBuffer(index: number, buffer: GPUBuffer): void;
  readBuffer(index: number): Promise<ArrayBuffer>;
  getPerformanceStats(): Promise<{
    executionTime: number;
    memoryUsage: number;
    throughput: number;
  }>;
  optimize(): Promise<void>;
  dispose(): void;
}

export interface WebGPUContext {
  device: GPUDevice;
  adapter: GPUAdapter;
  createKernel(code: string, options?: TranspileOptions): Promise<WebGPUKernel>;
  createBuffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer;
  dispose(): void;
}

export function createWebGPUContext(): Promise<WebGPUContext>;
export function createWebGPUKernel(code: string): Promise<WebGPUKernel>;

// Utility functions
export function validateCudaCode(code: string): { valid: boolean; errors: string[] };
export function estimatePerformance(code: string): Promise<{
  estimatedTime: number;
  memoryRequirement: number;
  optimizationHints: string[];
}>;
EOF

# Create enhanced main entry point
echo -e "${BLUE}📄 Creating enhanced main entry point...${NC}"
cat > dist/index.js << 'EOF'
// Enhanced CUDA-Rust-WASM runtime with performance monitoring and caching

let native, wasm;
let performanceMonitor;
let cacheManager;

// Lazy loading for better startup performance
function loadNative() {
  if (!native) {
    try {
      native = require('../build/Release/cuda_rust_wasm.node');
    } catch (e) {
      console.warn('Native module not available, using WASM fallback');
    }
  }
  return native;
}

function loadWasm() {
  if (!wasm) {
    wasm = require('../pkg/cuda_rust_wasm.js');
  }
  return wasm;
}

// Enhanced promisify with error handling and timeout
function promisify(fn, timeout = 30000) {
  return (...args) => new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(`Operation timed out after ${timeout}ms`));
    }, timeout);
    
    fn(...args, (err, result) => {
      clearTimeout(timer);
      if (err) reject(err);
      else resolve(result);
    });
  });
}

// Performance monitor implementation
class PerformanceMonitor {
  constructor() {
    this.timings = new Map();
    this.metrics = new Map();
  }

  startTiming(operation) {
    this.timings.set(operation, performance.now());
  }

  endTiming(operation) {
    const start = this.timings.get(operation);
    if (!start) throw new Error(`No timing started for operation: ${operation}`);
    
    const duration = performance.now() - start;
    this.timings.delete(operation);
    
    // Update metrics
    if (!this.metrics.has(operation)) {
      this.metrics.set(operation, { count: 0, total: 0, min: Infinity, max: 0 });
    }
    
    const metric = this.metrics.get(operation);
    metric.count++;
    metric.total += duration;
    metric.min = Math.min(metric.min, duration);
    metric.max = Math.max(metric.max, duration);
    
    return duration;
  }

  getMetrics() {
    const result = {};
    for (const [operation, data] of this.metrics) {
      result[operation] = {
        ...data,
        average: data.total / data.count
      };
    }
    return result;
  }

  reset() {
    this.timings.clear();
    this.metrics.clear();
  }
}

// Cache manager with LRU eviction
class CacheManager {
  constructor(maxSize = 100, defaultTtl = 3600000) { // 1 hour default TTL
    this.cache = new Map();
    this.maxSize = maxSize;
    this.defaultTtl = defaultTtl;
    this.stats = { hits: 0, misses: 0 };
  }

  async get(key) {
    const entry = this.cache.get(key);
    if (!entry) {
      this.stats.misses++;
      return null;
    }

    if (Date.now() > entry.expires) {
      this.cache.delete(key);
      this.stats.misses++;
      return null;
    }

    // Move to end (LRU)
    this.cache.delete(key);
    this.cache.set(key, entry);
    this.stats.hits++;
    return entry.value;
  }

  async set(key, value, ttl = this.defaultTtl) {
    // Evict oldest if at capacity
    if (this.cache.size >= this.maxSize && !this.cache.has(key)) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }

    this.cache.set(key, {
      value,
      expires: Date.now() + ttl
    });
  }

  async invalidate(pattern) {
    if (!pattern) {
      this.cache.clear();
      return;
    }

    const regex = new RegExp(pattern);
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key);
      }
    }
  }

  getStats() {
    return { ...this.stats, size: this.cache.size };
  }
}

// Initialize singletons
performanceMonitor = new PerformanceMonitor();
cacheManager = new CacheManager();

// Enhanced core functions
async function transpileCuda(code, options = {}) {
  const cacheKey = `transpile:${btoa(code)}:${JSON.stringify(options)}`;
  const cached = await cacheManager.get(cacheKey);
  if (cached && options.cacheEnabled !== false) {
    return cached;
  }

  performanceMonitor.startTiming('transpile');
  
  try {
    const nativeModule = loadNative();
    let result;
    
    if (nativeModule && options.target !== 'wasm') {
      const transpileNative = promisify(nativeModule.transpileCuda);
      result = await transpileNative(code, options);
    } else {
      const wasmModule = loadWasm();
      result = await wasmModule.transpile_cuda(code, options);
    }

    const duration = performanceMonitor.endTiming('transpile');
    
    // Add performance metadata
    result.profile = {
      ...result.profile,
      transpileTime: duration,
      cacheEnabled: options.cacheEnabled !== false
    };

    if (options.cacheEnabled !== false) {
      await cacheManager.set(cacheKey, result);
    }

    return result;
  } catch (error) {
    performanceMonitor.endTiming('transpile');
    throw error;
  }
}

async function analyzeKernel(code) {
  const cacheKey = `analyze:${btoa(code)}`;
  const cached = await cacheManager.get(cacheKey);
  if (cached) return cached;

  performanceMonitor.startTiming('analyze');
  
  try {
    const nativeModule = loadNative();
    let result;
    
    if (nativeModule) {
      const analyzeNative = promisify(nativeModule.analyzeKernel);
      result = await analyzeNative(code);
    } else {
      const wasmModule = loadWasm();
      result = await wasmModule.analyze_kernel(code);
    }

    performanceMonitor.endTiming('analyze');
    await cacheManager.set(cacheKey, result);
    return result;
  } catch (error) {
    performanceMonitor.endTiming('analyze');
    throw error;
  }
}

// Enhanced benchmark function
async function benchmark(code, options = {}) {
  const iterations = options.iterations || 100;
  const warmup = options.warmup || 10;
  const times = [];
  
  // Warmup phase
  for (let i = 0; i < warmup; i++) {
    await transpileCuda(code, { target: 'wasm', cacheEnabled: false });
  }
  
  // Actual benchmark
  for (let i = 0; i < iterations; i++) {
    const start = performance.now();
    await transpileCuda(code, { target: 'wasm', cacheEnabled: false });
    const end = performance.now();
    times.push(end - start);
  }
  
  times.sort((a, b) => a - b);
  const sum = times.reduce((a, b) => a + b, 0);
  const avg = sum / iterations;
  
  // Calculate percentiles
  const p50 = times[Math.floor(iterations * 0.5)];
  const p95 = times[Math.floor(iterations * 0.95)];
  const p99 = times[Math.floor(iterations * 0.99)];
  
  return {
    avgTime: avg,
    minTime: times[0],
    maxTime: times[times.length - 1],
    p50Time: p50,
    p95Time: p95,
    p99Time: p99,
    throughput: 1000 / avg,
    memoryBandwidth: 0, // Would need to be calculated based on actual memory usage
    efficiency: Math.min(times[0] / avg, 1.0) // How close average is to minimum
  };
}

// System capabilities detection
async function getSystemCapabilities() {
  const capabilities = {
    hasWebGPU: !!navigator.gpu,
    hasWebAssembly: typeof WebAssembly !== 'undefined',
    hasSIMD: false,
    hasThreads: typeof SharedArrayBuffer !== 'undefined',
    maxMemory: 0,
    cores: navigator.hardwareConcurrency || 1
  };

  // Test for WASM SIMD support
  try {
    const wasmModule = loadWasm();
    capabilities.hasSIMD = await wasmModule.has_simd_support();
  } catch (e) {
    capabilities.hasSIMD = false;
  }

  // Estimate max memory (rough heuristic)
  if (typeof performance !== 'undefined' && performance.memory) {
    capabilities.maxMemory = performance.memory.jsHeapSizeLimit || 0;
  }

  return capabilities;
}

// Enhanced WebGPU kernel with performance monitoring
async function createWebGPUKernel(code, options = {}) {
  if (!navigator.gpu) {
    throw new Error('WebGPU is not supported in this browser');
  }
  
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error('Failed to get WebGPU adapter');
  }
  
  const device = await adapter.requestDevice();
  const transpiled = await transpileCuda(code, { ...options, target: 'webgpu' });
  const shaderModule = device.createShaderModule({ code: transpiled.code });
  
  return {
    device,
    shaderModule,
    buffers: new Map(),
    performanceStats: { executionTime: 0, memoryUsage: 0, throughput: 0 },
    
    async dispatch(x, y = 1, z = 1) {
      const start = performance.now();
      
      const commandEncoder = device.createCommandEncoder();
      const computePass = commandEncoder.beginComputePass();
      
      // Implementation would depend on the specific kernel
      // This is a framework for the dispatch logic
      
      computePass.end();
      device.queue.submit([commandEncoder.finish()]);
      
      const end = performance.now();
      this.performanceStats.executionTime = end - start;
    },
    
    setBuffer(index, buffer) {
      this.buffers.set(index, buffer);
    },
    
    async readBuffer(index) {
      const buffer = this.buffers.get(index);
      if (!buffer) throw new Error(`No buffer at index ${index}`);
      
      await buffer.mapAsync(GPUMapMode.READ);
      const arrayBuffer = buffer.getMappedRange().slice();
      buffer.unmap();
      return arrayBuffer;
    },
    
    async getPerformanceStats() {
      return { ...this.performanceStats };
    },
    
    async optimize() {
      // Placeholder for kernel optimization logic
    },
    
    dispose() {
      for (const buffer of this.buffers.values()) {
        buffer.destroy();
      }
      this.buffers.clear();
      device.destroy();
    }
  };
}

// Factory functions
function createPerformanceMonitor() {
  return new PerformanceMonitor();
}

function getCacheManager() {
  return cacheManager;
}

// Validation utilities
function validateCudaCode(code) {
  const errors = [];
  
  // Basic syntax validation
  if (!code.includes('__global__') && !code.includes('__device__')) {
    errors.push('No CUDA kernel or device functions found');
  }
  
  // Check for balanced braces
  const openBraces = (code.match(/{/g) || []).length;
  const closeBraces = (code.match(/}/g) || []).length;
  if (openBraces !== closeBraces) {
    errors.push('Mismatched braces in code');
  }
  
  return {
    valid: errors.length === 0,
    errors
  };
}

async function estimatePerformance(code) {
  const analysis = await analyzeKernel(code);
  
  // Simple heuristics for performance estimation
  const complexity = code.length / 1000; // Basic complexity metric
  const estimatedTime = complexity * 10; // ms
  const memoryRequirement = analysis.sharedMemoryUsage * 1024; // bytes
  
  const hints = [];
  if (analysis.threadUtilization < 0.5) {
    hints.push('Consider increasing thread utilization');
  }
  if (analysis.sharedMemoryUsage > 0.8) {
    hints.push('High shared memory usage may cause bank conflicts');
  }
  
  return {
    estimatedTime,
    memoryRequirement,
    optimizationHints: hints
  };
}

// Public API
module.exports = {
  // Core functions
  transpileCuda,
  analyzeKernel,
  benchmark,
  
  // Performance and monitoring
  createPerformanceMonitor,
  getCacheManager,
  getSystemCapabilities,
  
  // WebGPU
  createWebGPUKernel,
  
  // Utilities
  validateCudaCode,
  estimatePerformance
};
EOF

# Copy WASM files to dist
echo -e "${BLUE}📋 Copying WASM files and assets...${NC}"
cp pkg/cuda_rust_wasm_bg.wasm dist/
cp pkg/cuda_rust_wasm.js dist/cuda_rust_wasm_wasm.js

# Copy TypeScript definitions from pkg if they exist
if [ -f "pkg/cuda_rust_wasm.d.ts" ]; then
    cp pkg/cuda_rust_wasm.d.ts dist/cuda_rust_wasm_wasm.d.ts
fi

# Create optimized package.json for dist
cat > dist/package.json << EOF
{
  "name": "cuda-rust-wasm",
  "version": "0.1.0",
  "description": "High-performance CUDA to WebAssembly transpiler with optimization",
  "main": "index.js",
  "types": "index.d.ts",
  "files": [
    "*.js",
    "*.d.ts",
    "*.wasm"
  ],
  "keywords": ["cuda", "webassembly", "gpu", "performance", "transpiler"],
  "engines": {
    "node": ">=16.0.0"
  },
  "browser": {
    "fs": false,
    "path": false
  }
}
EOF

# Create performance benchmark report
echo -e "${BLUE}📊 Generating build performance report...${NC}"
END_TIME=$(date +%s)
BUILD_DURATION=$((END_TIME - START_TIME))

cat > dist/build-report.json << EOF
{
  "buildTime": "${BUILD_DURATION}s",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "configuration": {
    "buildMode": "${BUILD_MODE}",
    "simdEnabled": ${ENABLE_SIMD},
    "threadsEnabled": ${ENABLE_THREADS},
    "optimizeSize": ${OPTIMIZE_SIZE},
    "cacheEnabled": ${ENABLE_CACHE},
    "parallelJobs": ${PARALLEL_JOBS}
  },
  "sizes": {
    "wasmBinary": $(stat -f%z "dist/cuda_rust_wasm_bg.wasm" 2>/dev/null || stat -c%s "dist/cuda_rust_wasm_bg.wasm" 2>/dev/null || echo 0),
    "jsWrapper": $(stat -f%z "dist/index.js" 2>/dev/null || stat -c%s "dist/index.js" 2>/dev/null || echo 0),
    "typeDefinitions": $(stat -f%z "dist/index.d.ts" 2>/dev/null || stat -c%s "dist/index.d.ts" 2>/dev/null || echo 0)
  },
  "optimizations": {
    "originalSize": ${ORIGINAL_SIZE},
    "optimizedSize": ${OPTIMIZED_SIZE},
    "compressionRatio": $(echo "scale=2; ${OPTIMIZED_SIZE} / ${ORIGINAL_SIZE}" | bc -l 2>/dev/null || echo "1.0")
  }
}
EOF

# Comprehensive build size report
echo -e "${GREEN}✅ Build complete!${NC}"
echo -e "${BLUE}📊 Comprehensive build report:${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# File sizes
WASM_SIZE=$(stat -f%z "dist/cuda_rust_wasm_bg.wasm" 2>/dev/null || stat -c%s "dist/cuda_rust_wasm_bg.wasm" 2>/dev/null || echo 0)
JS_SIZE=$(stat -f%z "dist/index.js" 2>/dev/null || stat -c%s "dist/index.js" 2>/dev/null || echo 0)
TS_SIZE=$(stat -f%z "dist/index.d.ts" 2>/dev/null || stat -c%s "dist/index.d.ts" 2>/dev/null || echo 0)

echo -e "${BLUE}📁 Output files:${NC}"
echo -e "  🔧 WASM binary:        $(numfmt --to=iec-i --suffix=B $WASM_SIZE 2>/dev/null || echo "${WASM_SIZE} bytes")"
echo -e "  📄 JavaScript wrapper: $(numfmt --to=iec-i --suffix=B $JS_SIZE 2>/dev/null || echo "${JS_SIZE} bytes")"
echo -e "  📝 TypeScript defs:    $(numfmt --to=iec-i --suffix=B $TS_SIZE 2>/dev/null || echo "${TS_SIZE} bytes")"

if [ "$ORIGINAL_SIZE" -gt 0 ] && [ "$OPTIMIZED_SIZE" -gt 0 ]; then
    COMPRESSION_RATIO=$(echo "scale=1; $OPTIMIZED_SIZE * 100 / $ORIGINAL_SIZE" | bc -l 2>/dev/null || echo "100.0")
    echo -e "  📉 Size reduction:      ${PERCENT_SAVINGS}% (now ${COMPRESSION_RATIO}% of original)"
fi

echo ""
echo -e "${BLUE}⚙️  Build configuration:${NC}"
echo -e "  🎯 Build mode:          ${BUILD_MODE}"
echo -e "  🧮 SIMD enabled:        ${ENABLE_SIMD}"
echo -e "  🧵 Threads enabled:     ${ENABLE_THREADS}"
echo -e "  📦 Size optimization:   ${OPTIMIZE_SIZE}"
echo -e "  💾 Build caching:       ${ENABLE_CACHE}"
echo -e "  ⚡ Parallel jobs:       ${PARALLEL_JOBS}"
echo -e "  ⏱️  Build time:          ${BUILD_DURATION}s"

echo ""
echo -e "${BLUE}🔧 Tools used:${NC}"
echo -e "  ✅ wasm-pack, wasm-opt, cargo"
if [ "$HAS_TWIGGY" = "true" ]; then
    echo -e "  ✅ twiggy (size analysis available in dist/size-analysis.txt)"
fi
if [ "$HAS_BINARYEN" = "true" ]; then
    echo -e "  ✅ binaryen tools available"
fi

echo ""
echo -e "${GREEN}🎉 WASM build completed successfully!${NC}"
echo -e "${BLUE}📁 Output directory: $(pwd)/dist${NC}"
echo -e "${BLUE}🚀 Ready for deployment or testing${NC}"

# Final validation
if [ ! -f "dist/cuda_rust_wasm_bg.wasm" ] || [ ! -f "dist/index.js" ] || [ ! -f "dist/index.d.ts" ]; then
    echo -e "${RED}❌ Build validation failed: Missing required output files${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Build validation passed${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"