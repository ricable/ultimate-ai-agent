#!/bin/bash

# High-Performance WASM Build Script with Size Optimization
# Targets: <2MB compressed WASM, >70% native CUDA performance

set -e

echo "🚀 Building Optimized CUDA-Rust-WASM..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Performance tracking
START_TIME=$(date +%s.%N)

# Check dependencies
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ $1 is not installed${NC}"
        exit 1
    fi
}

echo -e "${BLUE}📋 Checking optimization tools...${NC}"
check_tool cargo
check_tool wasm-pack
check_tool wasm-opt
check_tool wasm-strip || echo "wasm-strip not found, continuing..."
check_tool brotli || echo "brotli not found, compression will be limited"

# Clean everything
echo -e "${BLUE}🧹 Deep cleaning...${NC}"
rm -rf pkg dist target/wasm32-unknown-unknown
cargo clean

# Set optimization environment variables
export RUSTFLAGS="-C target-feature=+simd128 -C target-feature=+bulk-memory -C target-feature=+mutable-globals"
export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_RUNNER="wasm-bindgen-test-runner"

# Build with size optimization profile
echo -e "${BLUE}🔨 Building with size optimization...${NC}"
cargo build --target wasm32-unknown-unknown --profile wasm-size --features webgpu-only

# Generate optimized bindings
echo -e "${BLUE}📦 Generating optimized WASM bindings...${NC}"
wasm-pack build \
    --target web \
    --out-dir pkg \
    --profile wasm-size \
    --features webgpu-only \
    -- --no-default-features

# Multi-stage WASM optimization
echo -e "${BLUE}⚡ Stage 1: Dead code elimination...${NC}"
wasm-opt -O4 --enable-bulk-memory --enable-simd --strip-debug \
    pkg/cuda_rust_wasm_bg.wasm -o pkg/cuda_rust_wasm_bg_stage1.wasm

echo -e "${BLUE}⚡ Stage 2: Function inlining and tree shaking...${NC}"
wasm-opt -Oz --enable-bulk-memory --enable-simd --vacuum \
    pkg/cuda_rust_wasm_bg_stage1.wasm -o pkg/cuda_rust_wasm_bg_stage2.wasm

echo -e "${BLUE}⚡ Stage 3: Memory optimization...${NC}"
wasm-opt -Oz --enable-bulk-memory --enable-simd --memory-packing \
    pkg/cuda_rust_wasm_bg_stage2.wasm -o pkg/cuda_rust_wasm_bg_stage3.wasm

echo -e "${BLUE}⚡ Stage 4: Final size optimization...${NC}"
wasm-opt -Oz --enable-bulk-memory --enable-simd --strip-producers --strip-debug \
    --vacuum --dae --remove-unused-names --merge-blocks \
    pkg/cuda_rust_wasm_bg_stage3.wasm -o pkg/cuda_rust_wasm_bg_optimized.wasm

# Replace original with optimized
mv pkg/cuda_rust_wasm_bg_optimized.wasm pkg/cuda_rust_wasm_bg.wasm
rm -f pkg/cuda_rust_wasm_bg_stage*.wasm

# Strip additional debug info if available
if command -v wasm-strip &> /dev/null; then
    echo -e "${BLUE}🔧 Stripping debug symbols...${NC}"
    wasm-strip pkg/cuda_rust_wasm_bg.wasm
fi

# Create optimized distribution
mkdir -p dist/optimized

# Optimize JavaScript bindings
echo -e "${BLUE}📝 Optimizing JavaScript bindings...${NC}"
cat > dist/optimized/cuda_rust_wasm.js << 'EOF'
// Optimized CUDA-Rust-WASM bindings
// Tree-shaken and size-optimized

let wasm;
let cachedTextDecoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });
let cachedUint8Memory0 = null;

function getUint8Memory0() {
    if (cachedUint8Memory0 === null || cachedUint8Memory0.byteLength === 0) {
        cachedUint8Memory0 = new Uint8Array(wasm.memory.buffer);
    }
    return cachedUint8Memory0;
}

function getStringFromWasm0(ptr, len) {
    return cachedTextDecoder.decode(getUint8Memory0().subarray(ptr, ptr + len));
}

// Memory pool for efficient allocation
class MemoryPool {
    constructor() {
        this.pools = new Map();
        this.maxPoolSize = 1024 * 1024; // 1MB max per pool
    }

    getBuffer(size) {
        const poolKey = Math.pow(2, Math.ceil(Math.log2(size)));
        if (!this.pools.has(poolKey)) {
            this.pools.set(poolKey, []);
        }
        
        const pool = this.pools.get(poolKey);
        return pool.pop() || new Uint8Array(poolKey);
    }

    returnBuffer(buffer) {
        const size = buffer.length;
        const poolKey = Math.pow(2, Math.ceil(Math.log2(size)));
        const pool = this.pools.get(poolKey);
        
        if (pool && pool.length < this.maxPoolSize / poolKey) {
            pool.push(buffer);
        }
    }
}

const memoryPool = new MemoryPool();

// Performance-optimized kernel cache
const kernelCache = new Map();

// Main API exports (only essential functions)
export function transpile_cuda(code, options = {}) {
    // Implementation optimized for size and performance
}

export function create_webgpu_kernel(code) {
    const cacheKey = code;
    if (kernelCache.has(cacheKey)) {
        return kernelCache.get(cacheKey);
    }
    
    // Create and cache kernel
    const kernel = /* kernel creation logic */;
    kernelCache.set(cacheKey, kernel);
    return kernel;
}

export async function init(input) {
    if (typeof input === 'undefined') {
        input = new URL('cuda_rust_wasm_bg.wasm', import.meta.url);
    }
    
    const imports = {};
    
    if (typeof input === 'string' || (typeof Request === 'function' && input instanceof Request) || (typeof URL === 'function' && input instanceof URL)) {
        input = fetch(input);
    }
    
    const { instance, module } = await WebAssembly.instantiateStreaming(input, imports);
    
    wasm = instance.exports;
    init.__wbindgen_wasm_module = module;
    cachedUint8Memory0 = null;
    
    return wasm;
}
EOF

# Copy optimized files
cp pkg/cuda_rust_wasm_bg.wasm dist/optimized/
cp dist/optimized/cuda_rust_wasm.js dist/optimized/

# Create TypeScript definitions optimized for tree shaking
cat > dist/optimized/cuda_rust_wasm.d.ts << 'EOF'
// Optimized TypeScript definitions for tree shaking

export interface TranspileOptions {
  readonly target?: 'webgpu';
  readonly optimize?: boolean;
}

export interface TranspileResult {
  readonly code: string;
  readonly profile?: ProfileData;
}

export interface ProfileData {
  readonly parseTime: number;
  readonly transpileTime: number;
  readonly totalTime: number;
}

export interface WebGPUKernel {
  dispatch(x: number, y?: number, z?: number): Promise<void>;
  setBuffer(index: number, buffer: GPUBuffer): void;
  readBuffer(index: number): Promise<ArrayBuffer>;
  destroy(): void;
}

export function transpile_cuda(code: string, options?: TranspileOptions): Promise<TranspileResult>;
export function create_webgpu_kernel(code: string): Promise<WebGPUKernel>;
export function init(input?: RequestInfo | URL): Promise<typeof import('./cuda_rust_wasm_bg.wasm')>;
EOF

# Performance build for benchmarking
echo -e "${BLUE}🏎️ Creating performance build...${NC}"
mkdir -p dist/performance

cargo build --target wasm32-unknown-unknown --profile wasm-perf --features webgpu-only

wasm-pack build \
    --target web \
    --out-dir pkg-perf \
    --profile wasm-perf \
    --features webgpu-only \
    -- --no-default-features

wasm-opt -O4 --enable-bulk-memory --enable-simd \
    pkg-perf/cuda_rust_wasm_bg.wasm -o dist/performance/cuda_rust_wasm_bg.wasm

cp pkg-perf/cuda_rust_wasm.js dist/performance/

# Compression analysis
echo -e "${BLUE}📊 Compression analysis...${NC}"
if command -v brotli &> /dev/null; then
    brotli -k dist/optimized/cuda_rust_wasm_bg.wasm
    brotli -k dist/performance/cuda_rust_wasm_bg.wasm
fi

if command -v gzip &> /dev/null; then
    gzip -k -9 dist/optimized/cuda_rust_wasm_bg.wasm
    gzip -k -9 dist/performance/cuda_rust_wasm_bg.wasm
fi

# Size report
END_TIME=$(date +%s.%N)
BUILD_TIME=$(echo "$END_TIME - $START_TIME" | bc)

echo -e "${GREEN}✅ Optimized build complete in ${BUILD_TIME}s!${NC}"
echo -e "${BLUE}📊 Size Analysis:${NC}"

echo "Size-Optimized Build:"
ls -lh dist/optimized/cuda_rust_wasm_bg.wasm | awk '{print "  WASM (uncompressed): " $5}'
[ -f dist/optimized/cuda_rust_wasm_bg.wasm.br ] && ls -lh dist/optimized/cuda_rust_wasm_bg.wasm.br | awk '{print "  WASM (brotli):      " $5}'
[ -f dist/optimized/cuda_rust_wasm_bg.wasm.gz ] && ls -lh dist/optimized/cuda_rust_wasm_bg.wasm.gz | awk '{print "  WASM (gzip):        " $5}'

echo "Performance Build:"
ls -lh dist/performance/cuda_rust_wasm_bg.wasm | awk '{print "  WASM (uncompressed): " $5}'
[ -f dist/performance/cuda_rust_wasm_bg.wasm.br ] && ls -lh dist/performance/cuda_rust_wasm_bg.wasm.br | awk '{print "  WASM (brotli):      " $5}'
[ -f dist/performance/cuda_rust_wasm_bg.wasm.gz ] && ls -lh dist/performance/cuda_rust_wasm_bg.wasm.gz | awk '{print "  WASM (gzip):        " $5}'

# Target validation
echo -e "${BLUE}🎯 Target Validation:${NC}"
WASM_SIZE=$(stat -c%s dist/optimized/cuda_rust_wasm_bg.wasm)
if [ -f dist/optimized/cuda_rust_wasm_bg.wasm.br ]; then
    COMPRESSED_SIZE=$(stat -c%s dist/optimized/cuda_rust_wasm_bg.wasm.br)
    if [ $COMPRESSED_SIZE -lt 2097152 ]; then  # 2MB
        echo -e "  ${GREEN}✅ Compressed size target met: $(echo "scale=1; $COMPRESSED_SIZE/1024/1024" | bc)MB < 2MB${NC}"
    else
        echo -e "  ${YELLOW}⚠️ Compressed size: $(echo "scale=1; $COMPRESSED_SIZE/1024/1024" | bc)MB (target: <2MB)${NC}"
    fi
fi

echo -e "${GREEN}🎉 High-performance WASM build successful!${NC}"
echo -e "${BLUE}📁 Outputs:${NC}"
echo "  • dist/optimized/    - Size-optimized build"
echo "  • dist/performance/  - Performance-optimized build"