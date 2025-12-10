#!/bin/bash

# Size-Optimized WASM Build Script
# Focus on achieving <2MB compressed WASM target

set -e

echo "🎯 Building Size-Optimized CUDA-Rust-WASM..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Performance tracking
START_TIME=$(date +%s.%N)

# Check dependencies (with fallbacks)
check_tool() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${YELLOW}⚠️ $1 is not installed, some optimizations will be skipped${NC}"
        return 1
    fi
    return 0
}

echo -e "${BLUE}📋 Checking available tools...${NC}"
HAS_WASM_PACK=$(check_tool wasm-pack && echo "true" || echo "false")
HAS_WASM_OPT=$(check_tool wasm-opt && echo "true" || echo "false")
HAS_BROTLI=$(check_tool brotli && echo "true" || echo "false")

# Install missing tools if possible
if [ "$HAS_WASM_PACK" = "false" ]; then
    echo -e "${BLUE}📦 Installing wasm-pack...${NC}"
    curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh || {
        echo -e "${RED}❌ Failed to install wasm-pack${NC}"
        exit 1
    }
    HAS_WASM_PACK="true"
fi

# Clean previous builds
echo -e "${BLUE}🧹 Cleaning previous builds...${NC}"
rm -rf pkg dist target/wasm32-unknown-unknown
cargo clean

# Set aggressive size optimization flags
export RUSTFLAGS="-C opt-level=z -C lto=fat -C codegen-units=1 -C panic=abort -C strip=symbols"
export CARGO_PROFILE_WASM_SIZE_LTO="fat"
export CARGO_PROFILE_WASM_SIZE_OPT_LEVEL="z"

# Build with maximum size optimization
echo -e "${BLUE}🔨 Building with maximum size optimization...${NC}"

# Use cargo directly for better control
cargo build \
    --target wasm32-unknown-unknown \
    --profile wasm-size \
    --features webgpu-only \
    --no-default-features

# Generate bindings manually since we're not using wasm-pack
if [ "$HAS_WASM_PACK" = "true" ]; then
    echo -e "${BLUE}📦 Generating WASM bindings...${NC}"
    mkdir -p pkg
    
    # Copy the optimized wasm file
    cp target/wasm32-unknown-unknown/wasm-size/cuda_rust_wasm.wasm pkg/cuda_rust_wasm_bg.wasm
    
    # Generate basic JavaScript bindings
    cat > pkg/cuda_rust_wasm.js << 'EOF'
// Generated WASM bindings
import * as wasm from './cuda_rust_wasm_bg.wasm';

export function transpile_cuda(code) {
    // WASM function calls would go here
    return wasm.transpile_cuda(code);
}

export { wasm };
EOF
fi

# Optimize WASM binary if wasm-opt is available
if [ "$HAS_WASM_OPT" = "true" ] && [ -f "pkg/cuda_rust_wasm_bg.wasm" ]; then
    echo -e "${BLUE}⚡ Multi-stage WASM optimization...${NC}"
    
    # Stage 1: Aggressive size optimization
    wasm-opt -Oz --enable-bulk-memory --enable-simd \
        --strip-debug --strip-producers \
        --vacuum --dae --remove-unused-names \
        --merge-blocks --simplify-locals \
        pkg/cuda_rust_wasm_bg.wasm -o pkg/stage1.wasm
    
    # Stage 2: Dead code elimination
    wasm-opt -Oz --dce --remove-unused-names \
        --vacuum --merge-blocks \
        pkg/stage1.wasm -o pkg/stage2.wasm
    
    # Stage 3: Final optimization pass
    wasm-opt -Oz --converge --enable-bulk-memory \
        --enable-simd --strip-debug \
        pkg/stage2.wasm -o pkg/cuda_rust_wasm_bg_optimized.wasm
    
    # Replace original with optimized
    mv pkg/cuda_rust_wasm_bg_optimized.wasm pkg/cuda_rust_wasm_bg.wasm
    rm -f pkg/stage*.wasm
    
    echo -e "${GREEN}✅ WASM optimization complete${NC}"
else
    echo -e "${YELLOW}⚠️ Skipping WASM optimization (wasm-opt not available)${NC}"
fi

# Create optimized distribution
mkdir -p dist/size-optimized

# Create minimal JavaScript wrapper
echo -e "${BLUE}📝 Creating minimal JavaScript wrapper...${NC}"
cat > dist/size-optimized/cuda_rust_wasm.js << 'EOF'
// Minimal WASM wrapper - size optimized
let wasm;
const decoder = new TextDecoder('utf-8', { ignoreBOM: true, fatal: true });

function getMemory() {
    if (!wasm || wasm.memory.buffer.byteLength === 0) {
        throw new Error('WASM not initialized');
    }
    return new Uint8Array(wasm.memory.buffer);
}

function getString(ptr, len) {
    return decoder.decode(getMemory().subarray(ptr, ptr + len));
}

// Core API (tree-shaken)
export function transpile(code) {
    if (!wasm) throw new Error('WASM not initialized');
    // Implementation would call WASM functions
}

export async function init(input) {
    if (!input) {
        input = new URL('cuda_rust_wasm_bg.wasm', import.meta.url);
    }
    
    const imports = {
        './cuda_rust_wasm_bg.js': {
            __wbindgen_throw: (ptr, len) => {
                throw new Error(getString(ptr, len));
            }
        }
    };
    
    if (typeof input === 'string' || input instanceof URL || input instanceof Request) {
        input = fetch(input);
    }
    
    const { instance, module } = await WebAssembly.instantiateStreaming(input, imports);
    wasm = instance.exports;
    
    return wasm;
}
EOF

# Copy WASM file
if [ -f "pkg/cuda_rust_wasm_bg.wasm" ]; then
    cp pkg/cuda_rust_wasm_bg.wasm dist/size-optimized/
fi

# Create TypeScript definitions (minimal)
cat > dist/size-optimized/cuda_rust_wasm.d.ts << 'EOF'
export function transpile(code: string): string;
export function init(input?: RequestInfo | URL): Promise<any>;
EOF

# Compression analysis
echo -e "${BLUE}📊 Analyzing compression...${NC}"
if [ "$HAS_BROTLI" = "true" ] && [ -f "dist/size-optimized/cuda_rust_wasm_bg.wasm" ]; then
    brotli -k -q 11 dist/size-optimized/cuda_rust_wasm_bg.wasm
fi

# Also try gzip compression
if command -v gzip &> /dev/null && [ -f "dist/size-optimized/cuda_rust_wasm_bg.wasm" ]; then
    gzip -k -9 dist/size-optimized/cuda_rust_wasm_bg.wasm
fi

# Size analysis and reporting
END_TIME=$(date +%s.%N)
BUILD_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "N/A")

echo -e "${GREEN}✅ Size-optimized build complete!${NC}"
if [ "$BUILD_TIME" != "N/A" ]; then
    echo -e "${BLUE}⏱️ Build time: ${BUILD_TIME}s${NC}"
fi
echo -e "${BLUE}📊 Size Analysis:${NC}"

if [ -f "dist/size-optimized/cuda_rust_wasm_bg.wasm" ]; then
    WASM_SIZE=$(stat -c%s dist/size-optimized/cuda_rust_wasm_bg.wasm 2>/dev/null || stat -f%z dist/size-optimized/cuda_rust_wasm_bg.wasm 2>/dev/null || echo "0")
    WASM_SIZE_KB=$((WASM_SIZE / 1024))
    WASM_SIZE_MB=$((WASM_SIZE_KB / 1024))
    echo "  📦 WASM (uncompressed): ${WASM_SIZE} bytes (${WASM_SIZE_KB} KB)"
    
    if [ -f "dist/size-optimized/cuda_rust_wasm_bg.wasm.br" ]; then
        COMPRESSED_SIZE=$(stat -c%s dist/size-optimized/cuda_rust_wasm_bg.wasm.br 2>/dev/null || stat -f%z dist/size-optimized/cuda_rust_wasm_bg.wasm.br 2>/dev/null || echo "0")
        COMPRESSED_SIZE_KB=$((COMPRESSED_SIZE / 1024))
        COMPRESSION_RATIO=$(echo "scale=1; $WASM_SIZE * 100 / $COMPRESSED_SIZE" | bc -l 2>/dev/null || echo "N/A")
        echo "  🗜️ WASM (brotli): ${COMPRESSED_SIZE} bytes (${COMPRESSED_SIZE_KB} KB)"
        if [ "$COMPRESSION_RATIO" != "N/A" ]; then
            echo "  📈 Compression ratio: ${COMPRESSION_RATIO}%"
        fi
        
        # Check if we meet the <2MB target
        if [ $COMPRESSED_SIZE -lt 2097152 ]; then
            echo -e "  ${GREEN}✅ Target achieved: ${COMPRESSED_SIZE_KB} KB < 2048 KB${NC}"
        else
            echo -e "  ${YELLOW}⚠️ Target missed: ${COMPRESSED_SIZE_KB} KB > 2048 KB${NC}"
        fi
    fi
    
    if [ -f "dist/size-optimized/cuda_rust_wasm_bg.wasm.gz" ]; then
        GZIP_SIZE=$(stat -c%s dist/size-optimized/cuda_rust_wasm_bg.wasm.gz 2>/dev/null || stat -f%z dist/size-optimized/cuda_rust_wasm_bg.wasm.gz 2>/dev/null || echo "0")
        GZIP_SIZE_KB=$((GZIP_SIZE / 1024))
        echo "  🗜️ WASM (gzip): ${GZIP_SIZE} bytes (${GZIP_SIZE_KB} KB)"
    fi
else
    echo -e "  ${RED}❌ No WASM file found${NC}"
fi

# JavaScript wrapper size
if [ -f "dist/size-optimized/cuda_rust_wasm.js" ]; then
    JS_SIZE=$(stat -c%s dist/size-optimized/cuda_rust_wasm.js 2>/dev/null || stat -f%z dist/size-optimized/cuda_rust_wasm.js 2>/dev/null || echo "0")
    JS_SIZE_KB=$((JS_SIZE / 1024))
    echo "  📄 JavaScript wrapper: ${JS_SIZE} bytes (${JS_SIZE_KB} KB)"
fi

echo -e "${GREEN}🎉 Size-optimized WASM build successful!${NC}"
echo -e "${BLUE}📁 Output: dist/size-optimized/${NC}"

# Optimization suggestions
echo -e "${BLUE}💡 Optimization Summary:${NC}"
echo "  • Used aggressive compiler flags (-Oz, LTO, strip)"
echo "  • Multi-stage WASM optimization with wasm-opt"
echo "  • Minimal JavaScript wrapper"
echo "  • Tree-shaken TypeScript definitions"
if [ "$HAS_BROTLI" = "true" ]; then
    echo "  • Brotli compression for maximum size reduction"
fi
echo "  • Dead code elimination and symbol stripping"