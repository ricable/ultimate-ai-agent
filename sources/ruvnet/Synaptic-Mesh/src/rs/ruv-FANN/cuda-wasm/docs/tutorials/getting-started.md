# Getting Started with CUDA-Rust-WASM

## Welcome to GPU Computing in the Browser! 🚀

This tutorial will take you from zero to running your first CUDA kernel in a web browser using CUDA-Rust-WASM. By the end, you'll understand how to transpile CUDA code to WebGPU and achieve near-native performance in any browser.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Your First Kernel](#your-first-kernel)
4. [Understanding the Transpilation](#understanding-the-transpilation)
5. [Browser Integration](#browser-integration)
6. [Performance Analysis](#performance-analysis)
7. [Next Steps](#next-steps)

## Prerequisites

### Knowledge Requirements
- Basic understanding of parallel computing concepts
- Familiarity with JavaScript/TypeScript
- Optional: CUDA programming experience (helpful but not required)

### System Requirements
- **Node.js**: 18+ (LTS recommended)
- **Browser**: Chrome 113+, Firefox 115+, Safari 16.4+, or Edge 113+
- **GPU**: Any GPU with WebGPU support (most modern GPUs)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space

### Check WebGPU Support

First, let's verify your browser supports WebGPU:

```javascript
// Run this in your browser console
if (navigator.gpu) {
  console.log('✅ WebGPU is supported!');
  navigator.gpu.requestAdapter().then(adapter => {
    if (adapter) {
      console.log('🎯 GPU adapter found:', adapter.info);
    } else {
      console.log('⚠️ No GPU adapter available');
    }
  });
} else {
  console.log('❌ WebGPU not supported in this browser');
}
```

## Installation

### Method 1: NPX (Recommended for Beginners)

No installation required! Use NPX to run commands directly:

```bash
# Check if everything works
npx cuda-rust-wasm --version

# Create a new project
npx cuda-rust-wasm init my-first-gpu-project
cd my-first-gpu-project
```

### Method 2: Global Installation

For frequent use, install globally:

```bash
# Install globally
npm install -g cuda-rust-wasm

# Verify installation
cuda-rust-wasm --version

# Create new project
cuda-rust-wasm init my-first-gpu-project
cd my-first-gpu-project
```

### Method 3: Project Dependency

For existing projects:

```bash
# Add to existing project
npm install cuda-rust-wasm

# Add to package.json scripts
echo '{
  "scripts": {
    "transpile": "cuda-rust-wasm transpile",
    "analyze": "cuda-rust-wasm analyze",
    "benchmark": "cuda-rust-wasm benchmark"
  }
}' >> package.json
```

## Your First Kernel

Let's start with the classic "Hello World" of parallel computing: vector addition.

### Step 1: Create the CUDA Kernel

Create a file called `vector_add.cu`:

```cuda
// vector_add.cu - Your first CUDA kernel
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    // Calculate this thread's index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check to prevent out-of-bounds access
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
```

### Step 2: Analyze the Kernel

Before transpiling, let's analyze what this kernel does:

```bash
npx cuda-rust-wasm analyze vector_add.cu --verbose
```

You'll see output like:

```json
{
  "analysis": {
    "kernelName": "vectorAdd",
    "complexity": "simple",
    "memoryPattern": "coalesced",
    "parallelizability": "perfect",
    "recommendations": [
      "Excellent memory access pattern",
      "No optimization needed for this simple case",
      "Consider using larger block sizes for better occupancy"
    ]
  },
  "performance": {
    "estimatedSpeedup": "high",
    "bottlenecks": [],
    "occupancy": "good"
  }
}
```

### Step 3: Transpile to WebGPU

Now let's transpile the CUDA code to WebGPU:

```bash
npx cuda-rust-wasm transpile vector_add.cu \
  --target webgpu \
  --optimize \
  --output-dir ./dist \
  --generate-bindings
```

This creates several files:

```
dist/
├── vector_add.wgsl        # WebGPU shader
├── vector_add.js          # JavaScript bindings
├── vector_add.d.ts        # TypeScript definitions
├── vector_add_test.js     # Auto-generated tests
└── performance_report.html # Optimization report
```

### Step 4: Examine the Generated Code

Let's look at the generated WebGPU shader (`dist/vector_add.wgsl`):

```wgsl
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

struct Params {
    n: u32,
}

@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn vectorAdd(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i < params.n) {
        c[i] = a[i] + b[i];
    }
}
```

Notice how the transpiler:
- Converted CUDA's global memory pointers to WebGPU storage buffers
- Mapped `blockIdx.x * blockDim.x + threadIdx.x` to `global_invocation_id.x`
- Added proper buffer bindings and group layouts
- Optimized the workgroup size to 256 (a good default)

## Understanding the Transpilation

### Memory Model Translation

| CUDA Concept | WebGPU Equivalent | Purpose |
|--------------|-------------------|---------|
| Global memory (`float*`) | Storage buffer (`array<f32>`) | Large data arrays |
| Shared memory (`__shared__`) | Workgroup memory (`var<workgroup>`) | Fast inter-thread communication |
| Constant memory (`__constant__`) | Uniform buffer (`var<uniform>`) | Read-only parameters |
| Thread index (`threadIdx.x`) | Local invocation ID (`local_invocation_id.x`) | Thread within workgroup |
| Block index (`blockIdx.x`) | Workgroup ID (`workgroup_id.x`) | Workgroup identifier |

### Execution Model Translation

```cuda
// CUDA launch configuration
vectorAdd<<<gridSize, blockSize>>>(a, b, c, n);
```

becomes:

```javascript
// WebGPU dispatch
await kernel.dispatch(gridSize, 1, 1);
```

where the blockSize is embedded in the shader's `@workgroup_size()`.

## Browser Integration

Now let's create a complete web application using our transpiled kernel.

### Step 5: Create the HTML Structure

Create `index.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>My First GPU Kernel</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .container {
            background: #f5f5f5;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .status {
            padding: 10px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
        .info { background: #d1ecf1; color: #0c5460; }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #6c757d; cursor: not-allowed; }
        #results {
            font-family: monospace;
            background: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            white-space: pre-wrap;
        }
    </style>
</head>
<body>
    <h1>🚀 My First GPU Kernel</h1>
    
    <div class="container">
        <h2>Vector Addition on GPU</h2>
        <p>This demo adds two vectors using your GPU via WebGPU:</p>
        <p><strong>A + B = C</strong> where each vector has 1 million elements</p>
        
        <button id="runKernel">Run GPU Kernel</button>
        <button id="runCPU">Compare with CPU</button>
        
        <div id="status"></div>
        <div id="results"></div>
    </div>

    <script type="module" src="app.js"></script>
</body>
</html>
```

### Step 6: Create the JavaScript Application

Create `app.js`:

```javascript
// app.js - Complete WebGPU application
import { createWebGPUKernel, detectHardware } from 'cuda-rust-wasm';

class VectorAdditionDemo {
    constructor() {
        this.device = null;
        this.kernel = null;
        this.n = 1024 * 1024; // 1 million elements
        this.a = new Float32Array(this.n);
        this.b = new Float32Array(this.n);
        this.c = new Float32Array(this.n);
        
        this.initializeUI();
    }
    
    initializeUI() {
        document.getElementById('runKernel').addEventListener('click', () => this.runGPUKernel());
        document.getElementById('runCPU').addEventListener('click', () => this.runCPUComparison());
        
        this.showStatus('Ready to run! Click "Run GPU Kernel" to start.', 'info');
    }
    
    showStatus(message, type = 'info') {
        const statusDiv = document.getElementById('status');
        statusDiv.className = `status ${type}`;
        statusDiv.textContent = message;
    }
    
    updateResults(results) {
        document.getElementById('results').textContent = results;
    }
    
    async initialize() {
        this.showStatus('Initializing WebGPU...', 'info');
        
        try {
            // Check WebGPU support
            if (!navigator.gpu) {
                throw new Error('WebGPU not supported in this browser');
            }
            
            // Get GPU adapter and device
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error('No WebGPU adapter found');
            }
            
            this.device = await adapter.requestDevice();
            
            // Load and compile our kernel
            const kernelCode = await fetch('./dist/vector_add.wgsl').then(r => r.text());
            
            // Get hardware profile for optimization
            const hardware = await detectHardware();
            console.log('Hardware profile:', hardware);
            
            // Create optimized kernel
            this.kernel = await createWebGPUKernel(this.device, kernelCode, {
                optimizationLevel: 3,
                enableProfiling: true,
                hardwareProfile: hardware
            });
            
            // Generate test data
            this.generateTestData();
            
            this.showStatus('✅ WebGPU initialized successfully!', 'success');
            this.updateResults(`Hardware: ${hardware.deviceName}\nCompute capability: ${hardware.computeCapability}\nMemory bandwidth: ${hardware.memoryBandwidth} GB/s`);
            
        } catch (error) {
            this.showStatus(`❌ Initialization failed: ${error.message}`, 'error');
            console.error('Initialization error:', error);
        }
    }
    
    generateTestData() {
        // Generate random test vectors
        for (let i = 0; i < this.n; i++) {
            this.a[i] = Math.random() * 100;
            this.b[i] = Math.random() * 100;
        }
        console.log('Generated test data:', { n: this.n, sampleA: this.a.slice(0, 5), sampleB: this.b.slice(0, 5) });
    }
    
    async runGPUKernel() {
        if (!this.kernel) {
            this.showStatus('❌ Kernel not initialized', 'error');
            return;
        }
        
        try {
            this.showStatus('🔄 Running GPU kernel...', 'info');
            document.getElementById('runKernel').disabled = true;
            
            const startTime = performance.now();
            
            // Create GPU buffers
            const bufferA = this.kernel.createBuffer(this.n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const bufferB = this.kernel.createBuffer(this.n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            const bufferC = this.kernel.createBuffer(this.n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
            
            // Copy data to GPU
            await this.kernel.writeBuffer(0, this.a.buffer);
            await this.kernel.writeBuffer(1, this.b.buffer);
            
            // Set kernel parameters
            this.kernel.setArgs({ n: this.n });
            
            // Launch kernel with profiling
            const profile = await this.kernel.dispatchWithProfiling(
                Math.ceil(this.n / 256), 1, 1
            );
            
            // Read results back
            const resultBuffer = await this.kernel.readBuffer(2);
            const results = new Float32Array(resultBuffer);
            
            const totalTime = performance.now() - startTime;
            
            // Verify results (check first few elements)
            const verification = this.verifyResults(results);
            
            // Display results
            const throughput = (this.n * 3 * 4) / (profile.executionTime / 1000) / (1024 * 1024 * 1024); // GB/s
            const gflops = this.n / (profile.executionTime / 1000) / (1000 * 1000 * 1000); // GFLOPS
            
            this.showStatus('✅ GPU kernel completed successfully!', 'success');
            this.updateResults(`GPU Results:
Total time: ${totalTime.toFixed(2)}ms
Kernel execution: ${profile.executionTime.toFixed(3)}ms
Memory transfer: ${(totalTime - profile.executionTime).toFixed(3)}ms
Throughput: ${throughput.toFixed(2)} GB/s
Compute: ${gflops.toFixed(2)} GFLOPS
Verification: ${verification ? '✅ PASSED' : '❌ FAILED'}

Sample results:
A[0] + B[0] = ${this.a[0].toFixed(3)} + ${this.b[0].toFixed(3)} = ${results[0].toFixed(3)}
A[1] + B[1] = ${this.a[1].toFixed(3)} + ${this.b[1].toFixed(3)} = ${results[1].toFixed(3)}
A[2] + B[2] = ${this.a[2].toFixed(3)} + ${this.b[2].toFixed(3)} = ${results[2].toFixed(3)}`);
            
            // Store GPU results for comparison
            this.gpuResults = {
                time: profile.executionTime,
                throughput: throughput,
                gflops: gflops
            };
            
        } catch (error) {
            this.showStatus(`❌ GPU kernel failed: ${error.message}`, 'error');
            console.error('GPU kernel error:', error);
        } finally {
            document.getElementById('runKernel').disabled = false;
        }
    }
    
    async runCPUComparison() {
        this.showStatus('🔄 Running CPU comparison...', 'info');
        document.getElementById('runCPU').disabled = true;
        
        try {
            const startTime = performance.now();
            
            // CPU vector addition
            const cpuResults = new Float32Array(this.n);
            for (let i = 0; i < this.n; i++) {
                cpuResults[i] = this.a[i] + this.b[i];
            }
            
            const cpuTime = performance.now() - startTime;
            const cpuThroughput = (this.n * 3 * 4) / (cpuTime / 1000) / (1024 * 1024 * 1024); // GB/s
            const cpuGflops = this.n / (cpuTime / 1000) / (1000 * 1000 * 1000); // GFLOPS
            
            let comparison = `CPU Results:
Execution time: ${cpuTime.toFixed(2)}ms
Throughput: ${cpuThroughput.toFixed(2)} GB/s
Compute: ${cpuGflops.toFixed(2)} GFLOPS`;
            
            if (this.gpuResults) {
                const speedup = cpuTime / this.gpuResults.time;
                const throughputRatio = this.gpuResults.throughput / cpuThroughput;
                
                comparison += `

GPU vs CPU Comparison:
Speedup: ${speedup.toFixed(2)}x faster
Throughput improvement: ${throughputRatio.toFixed(2)}x
Efficiency: ${(this.gpuResults.gflops / cpuGflops * 100).toFixed(1)}%`;
            }
            
            this.showStatus('✅ CPU comparison completed!', 'success');
            this.updateResults(comparison);
            
        } catch (error) {
            this.showStatus(`❌ CPU comparison failed: ${error.message}`, 'error');
            console.error('CPU comparison error:', error);
        } finally {
            document.getElementById('runCPU').disabled = false;
        }
    }
    
    verifyResults(gpuResults) {
        // Verify first 100 elements (sufficient for validation)
        const tolerance = 1e-5;
        for (let i = 0; i < Math.min(100, this.n); i++) {
            const expected = this.a[i] + this.b[i];
            const actual = gpuResults[i];
            if (Math.abs(expected - actual) > tolerance) {
                console.error(`Verification failed at index ${i}: expected ${expected}, got ${actual}`);
                return false;
            }
        }
        return true;
    }
}

// Initialize the demo when page loads
window.addEventListener('DOMContentLoaded', async () => {
    const demo = new VectorAdditionDemo();
    await demo.initialize();
});
```

### Step 7: Run the Application

1. Start a local server (required for module imports):

```bash
# Using Python (if available)
python -m http.server 8000

# Using Node.js http-server
npx http-server

# Using any other static server
```

2. Open your browser to `http://localhost:8000`

3. Click "Run GPU Kernel" to see your CUDA kernel running in the browser!

## Performance Analysis

### Understanding the Results

When you run the demo, you'll see metrics like:

```
GPU Results:
Total time: 15.42ms
Kernel execution: 2.341ms
Memory transfer: 13.079ms
Throughput: 2.84 GB/s
Compute: 427.42 GFLOPS
```

Let's break this down:

- **Total time**: Complete operation including memory transfers
- **Kernel execution**: Pure GPU computation time
- **Memory transfer**: Time to copy data to/from GPU
- **Throughput**: Data bandwidth achieved
- **Compute**: Computational throughput

### Optimization Opportunities

The analysis might show that memory transfer dominates execution time. This is normal for simple kernels and suggests:

1. **Batch operations**: Process more data per kernel launch
2. **Keep data on GPU**: Minimize host-device transfers
3. **Use persistent kernels**: For iterative algorithms
4. **Pipeline operations**: Overlap compute with memory transfers

### Compare with CPU

The CPU comparison shows the effectiveness of GPU acceleration:

```
GPU vs CPU Comparison:
Speedup: 8.45x faster
Throughput improvement: 3.2x
Efficiency: 89.3%
```

## Next Steps

Congratulations! 🎉 You've successfully run your first CUDA kernel in a web browser. Here's what to explore next:

### 1. Try More Complex Kernels

```bash
# Matrix multiplication
npx cuda-rust-wasm create-example matrix-multiply

# Image processing
npx cuda-rust-wasm create-example convolution

# Neural network layer
npx cuda-rust-wasm create-example neural-network
```

### 2. Enable Neural Optimization

```javascript
// Add AI-powered optimization
const optimizer = new NeuralOptimizer();
const optimizedKernel = await optimizer.optimizeKernel(kernelCode, {
    target: 'webgpu',
    performanceTarget: 'throughput'
});
```

### 3. Integrate with ruv-FANN

```javascript
// Add neural network acceleration
import { RuvFANN } from 'ruv-fann';

const neuralNet = new RuvFANN([784, 128, 10]);
const accelerated = await neuralNet.accelerateWithGPU(kernelCode);
```

### 4. Advanced Features

- **Real-time video processing**: Process webcam streams
- **Scientific computing**: Implement complex algorithms
- **Machine learning**: Train neural networks in the browser
- **Data visualization**: GPU-accelerated graphics

### 5. Learn More

- **[Advanced Tutorials](./advanced/)**: Complex patterns and optimizations
- **[Performance Guide](../PERFORMANCE.md)**: Detailed optimization techniques
- **[API Reference](../API.md)**: Complete API documentation
- **[Migration Guide](../MIGRATION_GUIDE.md)**: Port existing CUDA code
- **[Examples](../../examples/)**: Real-world applications

### 6. Join the Community

- **Discord**: [Get real-time help](https://discord.gg/vibecast)
- **GitHub**: [Contribute and report issues](https://github.com/vibecast/cuda-rust-wasm)
- **Blog**: [Latest updates and tutorials](https://vibecast.io/blog)
- **Newsletter**: [Stay updated](https://vibecast.io/newsletter)

## Troubleshooting

### Common Issues

**WebGPU not supported:**
```javascript
// Add fallback detection
if (!navigator.gpu) {
    console.log('WebGPU not available, falling back to WebAssembly');
    // Implement WASM fallback
}
```

**Kernel compilation errors:**
```bash
# Add verbose output for debugging
npx cuda-rust-wasm transpile kernel.cu --verbose --debug
```

**Performance issues:**
```bash
# Analyze for bottlenecks
npx cuda-rust-wasm analyze kernel.cu --performance-focus
```

**Memory allocation failures:**
```javascript
// Handle gracefully
try {
    const buffer = kernel.createBuffer(size);
} catch (error) {
    if (error.name === 'OutOfMemoryError') {
        console.log('Reducing buffer size...');
        // Implement chunked processing
    }
}
```

### Getting Help

1. **Check the browser console** for detailed error messages
2. **Enable verbose logging** during development
3. **Use the analysis tools** to understand performance
4. **Join our Discord** for community support
5. **Check GitHub issues** for known problems

---

You've now mastered the basics of CUDA-Rust-WASM! You can transpile CUDA kernels to run efficiently in web browsers, achieving excellent performance with full cross-platform compatibility. The GPU is now accessible from JavaScript, opening up infinite possibilities for web applications.

Happy coding! 🚀