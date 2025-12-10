#!/bin/bash
# Build script for WASM compilation of ranML components

set -e

echo "🚀 Building ranML for WebAssembly"
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Check if wasm-pack is installed
if ! command -v wasm-pack &> /dev/null; then
    echo -e "${RED}❌ wasm-pack is not installed${NC}"
    echo "Install with: curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh"
    exit 1
fi

# Check if cargo is installed with wasm32 target
if ! rustup target list --installed | grep -q "wasm32-unknown-unknown"; then
    echo -e "${YELLOW}⚠️  Adding wasm32-unknown-unknown target...${NC}"
    rustup target add wasm32-unknown-unknown
fi

echo -e "${GREEN}✅ Prerequisites satisfied${NC}"

# Create output directory
OUTPUT_DIR="wasm-dist"
mkdir -p "$OUTPUT_DIR"

# Build configuration
CARGO_CONFIG="Cargo-wasm.toml"
FEATURES="wasm"
OPTIMIZATION="--release"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            OPTIMIZATION="--dev"
            echo -e "${YELLOW}🔧 Development build enabled${NC}"
            shift
            ;;
        --webgpu)
            FEATURES="wasm,webgpu"
            echo -e "${BLUE}🎮 WebGPU acceleration enabled${NC}"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dev         Build in development mode"
            echo "  --webgpu      Enable WebGPU acceleration"
            echo "  --output DIR  Set output directory (default: wasm-dist)"
            echo "  --help        Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}🔧 Build configuration:${NC}"
echo "  - Features: $FEATURES"
echo "  - Optimization: $OPTIMIZATION"
echo "  - Output: $OUTPUT_DIR"

# Build core components for WASM
echo -e "${BLUE}🔨 Building ran-core for WASM...${NC}"
cd crates/ran-core
wasm-pack build \
    --target web \
    --out-dir "../../$OUTPUT_DIR/ran-core" \
    --features "$FEATURES" \
    $OPTIMIZATION \
    --scope ranml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ ran-core built successfully${NC}"
else
    echo -e "${RED}❌ ran-core build failed${NC}"
    exit 1
fi

cd ../..

# Build neural network components for WASM  
echo -e "${BLUE}🔨 Building ran-neural for WASM...${NC}"
cd crates/ran-neural
wasm-pack build \
    --target web \
    --out-dir "../../$OUTPUT_DIR/ran-neural" \
    --features "$FEATURES" \
    $OPTIMIZATION \
    --scope ranml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ ran-neural built successfully${NC}"
else
    echo -e "${RED}❌ ran-neural build failed${NC}"
    exit 1
fi

cd ../..

# Build forecasting components for WASM
echo -e "${BLUE}🔨 Building ran-forecasting for WASM...${NC}"
cd crates/ran-forecasting
wasm-pack build \
    --target web \
    --out-dir "../../$OUTPUT_DIR/ran-forecasting" \
    --features "$FEATURES" \
    $OPTIMIZATION \
    --scope ranml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ ran-forecasting built successfully${NC}"
else
    echo -e "${RED}❌ ran-forecasting build failed${NC}"
    exit 1
fi

cd ../..

# Build edge components for WASM
echo -e "${BLUE}🔨 Building ran-edge for WASM...${NC}"
cd crates/ran-edge
wasm-pack build \
    --target web \
    --out-dir "../../$OUTPUT_DIR/ran-edge" \
    --features "$FEATURES" \
    $OPTIMIZATION \
    --scope ranml

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ ran-edge built successfully${NC}"
else
    echo -e "${RED}❌ ran-edge build failed${NC}"
    exit 1
fi

cd ../..

# Create unified package.json for the entire WASM distribution
echo -e "${BLUE}📦 Creating unified package.json...${NC}"
cat > "$OUTPUT_DIR/package.json" << EOF
{
  "name": "@ranml/wasm",
  "version": "0.1.0",
  "description": "WebAssembly build of RAN ML optimization suite",
  "main": "index.js",
  "types": "index.d.ts",
  "files": [
    "ran-core/",
    "ran-neural/", 
    "ran-forecasting/",
    "ran-edge/",
    "index.js",
    "index.d.ts"
  ],
  "scripts": {
    "test": "wasm-pack test --headless --firefox",
    "test-chrome": "wasm-pack test --headless --chrome",
    "test-node": "wasm-pack test --node"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/your-org/ranML.git"
  },
  "keywords": ["wasm", "neural-networks", "5g", "ran", "optimization"],
  "author": "RAN ML Team",
  "license": "MIT OR Apache-2.0",
  "dependencies": {},
  "devDependencies": {
    "@types/node": "^20.0.0"
  }
}
EOF

# Create unified TypeScript definitions
echo -e "${BLUE}📝 Creating TypeScript definitions...${NC}"
cat > "$OUTPUT_DIR/index.d.ts" << 'EOF'
/**
 * RAN ML WebAssembly Module
 * 
 * This module provides WebAssembly bindings for the RAN ML optimization suite,
 * enabling neural network training and inference in web browsers and edge devices.
 */

declare module '@ranml/wasm' {
  // Core types from ran-core
  export interface GeoCoordinate {
    latitude: number;
    longitude: number;
    altitude?: number;
  }
  
  export interface TimeSeriesPoint<T> {
    timestamp: Date;
    value: T;
    metadata?: Record<string, string>;
  }
  
  export interface PerformanceMetrics {
    throughput: number;
    latency: number;
    error_rate: number;
    timestamp: Date;
  }
  
  // Neural network types from ran-neural
  export enum ModelType {
    ThroughputPredictor = "ThroughputPredictor",
    HandoverDecision = "HandoverDecision", 
    LoadBalancer = "LoadBalancer",
    InterferenceClassifier = "InterferenceClassifier"
  }
  
  export interface NetworkConfig {
    layers: number[];
    activation: string;
    learning_rate: number;
    training_algorithm: string;
  }
  
  export class RanNeuralNetwork {
    constructor(model_type: ModelType);
    static with_config(model_type: ModelType, config: NetworkConfig): RanNeuralNetwork;
    
    predict(features: Float64Array): Float64Array;
    predict_batch(inputs: Float64Array[]): Float64Array[];
    train(training_data: TrainingData): TrainingMetrics;
    
    save_model(): Uint8Array;
    load_model(data: Uint8Array): void;
    
    get_performance_metrics(): PerformanceReport;
    reset_stats(): void;
    is_ready(): boolean;
  }
  
  export interface TrainingData {
    inputs: Float64Array[];
    outputs: Float64Array[];
  }
  
  export interface TrainingMetrics {
    epochs: number;
    final_error: number;
    training_time_ms: number;
    convergence_achieved: boolean;
  }
  
  export interface PerformanceReport {
    model_type: ModelType;
    throughput_ops_per_sec: number;
    memory_usage_mb: number;
    accuracy?: number;
    last_updated: Date;
  }
  
  // Forecasting types from ran-forecasting
  export enum ForecastHorizon {
    Minutes = "Minutes",
    Hours = "Hours", 
    Days = "Days"
  }
  
  export class RanForecaster {
    constructor(model: ForecastingModel);
    
    fit(timeseries: RanTimeSeries): Promise<void>;
    predict(): Promise<RanForecast>;
    predict_with_data(input_data: RanTimeSeries): Promise<RanForecast>;
    update(new_data: RanTimeSeries): Promise<void>;
    
    get_performance(): ForecastAccuracy | null;
    is_fitted(): boolean;
    reset(): Promise<void>;
  }
  
  export interface RanTimeSeries {
    name: string;
    points: TimeSeriesPoint<number>[];
    metadata: Record<string, string>;
  }
  
  export interface RanForecast {
    values: number[];
    timestamps: Date[];
    confidence_intervals?: [number, number][];
    metadata: Record<string, string>;
  }
  
  export interface ForecastAccuracy {
    mape: number;
    rmse: number;
    mae: number;
    r_squared: number;
  }
  
  // Edge deployment types from ran-edge
  export class EdgeInferenceEngine {
    constructor();
    
    load_model(model_data: Uint8Array): Promise<void>;
    predict(features: Float64Array): Promise<Float64Array>;
    predict_batch(inputs: Float64Array[]): Promise<Float64Array[]>;
    
    get_status(): EdgeStatus;
    get_metrics(): EdgeMetrics;
  }
  
  export interface EdgeStatus {
    loaded_models: number;
    memory_usage_mb: number;
    is_ready: boolean;
  }
  
  export interface EdgeMetrics {
    total_inferences: number;
    avg_inference_time_ms: number;
    throughput_per_sec: number;
    memory_peak_mb: number;
  }
  
  // Utility functions
  export function init(): Promise<void>;
  export function set_panic_hook(): void;
  export function get_version(): string;
  export function get_build_info(): BuildInfo;
  
  export interface BuildInfo {
    version: string;
    git_hash: string;
    build_date: string;
    features: string[];
    target: string;
  }
  
  // Error types
  export class RanError extends Error {
    constructor(message: string);
  }
  
  export class NeuralError extends RanError {
    constructor(message: string);
  }
  
  export class ForecastError extends RanError {
    constructor(message: string);
  }
}
EOF

# Create unified JavaScript entry point
echo -e "${BLUE}📄 Creating JavaScript entry point...${NC}"
cat > "$OUTPUT_DIR/index.js" << 'EOF'
/**
 * RAN ML WebAssembly Module Entry Point
 */

// Import all WASM modules
import init_core, * as core from './ran-core/ran_core.js';
import init_neural, * as neural from './ran-neural/ran_neural.js';
import init_forecasting, * as forecasting from './ran-forecasting/ran_forecasting.js';
import init_edge, * as edge from './ran-edge/ran_edge.js';

let initialized = false;

/**
 * Initialize all WASM modules
 */
export async function init() {
  if (initialized) return;
  
  // Initialize all modules in parallel
  await Promise.all([
    init_core(),
    init_neural(), 
    init_forecasting(),
    init_edge()
  ]);
  
  // Set panic hook for better error reporting
  if (core.set_panic_hook) {
    core.set_panic_hook();
  }
  
  initialized = true;
  console.log('RAN ML WASM modules initialized successfully');
}

// Re-export all components
export * from './ran-core/ran_core.js';
export * from './ran-neural/ran_neural.js';
export * from './ran-forecasting/ran_forecasting.js';
export * from './ran-edge/ran_edge.js';

// Convenience exports
export { 
  RanNeuralNetwork,
  ModelType,
  NetworkConfig,
  TrainingData,
  TrainingMetrics,
  PerformanceReport
} from './ran-neural/ran_neural.js';

export {
  RanForecaster,
  ForecastHorizon,
  RanTimeSeries,
  RanForecast,
  ForecastAccuracy
} from './ran-forecasting/ran_forecasting.js';

export {
  EdgeInferenceEngine,
  EdgeStatus,
  EdgeMetrics
} from './ran-edge/ran_edge.js';

export {
  GeoCoordinate,
  TimeSeriesPoint,
  PerformanceMetrics
} from './ran-core/ran_core.js';

/**
 * Get version information
 */
export function get_version() {
  return "0.1.0";
}

/**
 * Get build information
 */
export function get_build_info() {
  return {
    version: "0.1.0",
    git_hash: process.env.GIT_HASH || "unknown",
    build_date: new Date().toISOString(),
    features: ["wasm", "neural", "forecasting", "edge"],
    target: "wasm32-unknown-unknown"
  };
}

// Auto-initialize when possible
if (typeof window !== 'undefined' || typeof importScripts !== 'undefined') {
  // Browser or Web Worker environment
  init().catch(console.error);
}
EOF

# Create example usage file
echo -e "${BLUE}📖 Creating usage examples...${NC}"
cat > "$OUTPUT_DIR/examples.js" << 'EOF'
/**
 * RAN ML WASM Usage Examples
 */

import { init, RanNeuralNetwork, ModelType, RanForecaster, EdgeInferenceEngine } from './index.js';

// Example 1: Neural Network Inference
async function neuralNetworkExample() {
  await init();
  
  console.log('🧠 Neural Network Example');
  
  // Create a throughput predictor
  const predictor = new RanNeuralNetwork(ModelType.ThroughputPredictor);
  
  // Prepare input features (cell load, power, SINR, UEs, frequency)
  const features = new Float64Array([0.6, 0.8, 0.7, 25, 2.4]);
  
  // Run prediction
  const prediction = predictor.predict(features);
  console.log('Predicted throughput:', prediction[0], 'Mbps');
  
  // Get performance metrics
  const metrics = predictor.get_performance_metrics();
  console.log('Inference time:', metrics.throughput_ops_per_sec, 'ops/sec');
}

// Example 2: Time Series Forecasting
async function forecastingExample() {
  await init();
  
  console.log('📈 Forecasting Example');
  
  // Create time series data
  const timeseries = {
    name: "cell_traffic",
    points: [],
    metadata: { cell_id: "001", type: "throughput" }
  };
  
  // Add sample data points (24 hours of traffic)
  const now = new Date();
  for (let i = 0; i < 24; i++) {
    const timestamp = new Date(now.getTime() + i * 3600000); // +1 hour each
    const value = 100 + 20 * Math.sin(i * Math.PI / 12); // Daily pattern
    
    timeseries.points.push({
      timestamp: timestamp,
      value: value,
      metadata: {}
    });
  }
  
  // Note: In real implementation, you would create and fit the forecaster
  console.log('Traffic data prepared:', timeseries.points.length, 'points');
}

// Example 3: Edge Inference
async function edgeInferenceExample() {
  await init();
  
  console.log('⚡ Edge Inference Example');
  
  // Create edge inference engine
  const engine = new EdgeInferenceEngine();
  
  // Check status
  const status = engine.get_status();
  console.log('Edge engine ready:', status.is_ready);
  console.log('Memory usage:', status.memory_usage_mb, 'MB');
  
  // Simulate batch inference
  const batchInputs = [
    new Float64Array([0.5, 0.6, 0.7, 20, 2.4]),
    new Float64Array([0.7, 0.8, 0.5, 30, 5.0]),
    new Float64Array([0.3, 0.4, 0.9, 15, 2.4])
  ];
  
  console.log('Prepared batch of', batchInputs.length, 'inputs');
  // Note: Actual prediction would require a loaded model
}

// Example 4: Performance Monitoring
async function performanceExample() {
  await init();
  
  console.log('📊 Performance Monitoring Example');
  
  const predictor = new RanNeuralNetwork(ModelType.ThroughputPredictor);
  
  // Run multiple predictions to collect performance data
  const features = new Float64Array([0.5, 0.7, 0.6, 25, 2.4]);
  const iterations = 1000;
  
  console.time('batch_inference');
  for (let i = 0; i < iterations; i++) {
    predictor.predict(features);
  }
  console.timeEnd('batch_inference');
  
  // Get detailed performance metrics
  const report = predictor.get_performance_metrics();
  console.log('Performance Report:');
  console.log('- Throughput:', report.throughput_ops_per_sec.toFixed(1), 'ops/sec');
  console.log('- Memory usage:', report.memory_usage_mb.toFixed(2), 'MB');
  console.log('- Model type:', report.model_type);
}

// Run examples
async function runAllExamples() {
  try {
    await neuralNetworkExample();
    await forecastingExample();
    await edgeInferenceExample();
    await performanceExample();
    console.log('✅ All examples completed successfully');
  } catch (error) {
    console.error('❌ Example failed:', error);
  }
}

// Export for use in browsers or Node.js
if (typeof window !== 'undefined') {
  window.runRanMLExamples = runAllExamples;
} else {
  runAllExamples();
}
EOF

# Generate size report
echo -e "${BLUE}📊 Generating size report...${NC}"
SIZE_REPORT="$OUTPUT_DIR/size_report.txt"
echo "RAN ML WASM Build Size Report" > "$SIZE_REPORT"
echo "=============================" >> "$SIZE_REPORT"
echo "Generated: $(date)" >> "$SIZE_REPORT"
echo "" >> "$SIZE_REPORT"

for module in ran-core ran-neural ran-forecasting ran-edge; do
    if [ -d "$OUTPUT_DIR/$module" ]; then
        echo "Module: $module" >> "$SIZE_REPORT"
        echo "----------------" >> "$SIZE_REPORT"
        
        if [ -f "$OUTPUT_DIR/$module/${module//-/_}_bg.wasm" ]; then
            wasm_file="$OUTPUT_DIR/$module/${module//-/_}_bg.wasm"
            wasm_size=$(wc -c < "$wasm_file")
            echo "WASM size: $(numfmt --to=iec --suffix=B $wasm_size)" >> "$SIZE_REPORT"
        fi
        
        if [ -f "$OUTPUT_DIR/$module/${module//-/_}.js" ]; then
            js_file="$OUTPUT_DIR/$module/${module//-/_}.js"
            js_size=$(wc -c < "$js_file")
            echo "JS size: $(numfmt --to=iec --suffix=B $js_size)" >> "$SIZE_REPORT"
        fi
        
        echo "" >> "$SIZE_REPORT"
    fi
done

# Calculate total size
total_size=$(find "$OUTPUT_DIR" -name "*.wasm" -o -name "*.js" | xargs wc -c | tail -1 | awk '{print $1}')
echo "Total size: $(numfmt --to=iec --suffix=B $total_size)" >> "$SIZE_REPORT"

echo -e "${GREEN}📊 Size report generated: $SIZE_REPORT${NC}"

# Run tests if requested
if [ "$1" = "--test" ]; then
    echo -e "${BLUE}🧪 Running WASM tests...${NC}"
    cd crates/ran-core && wasm-pack test --headless --firefox && cd ../..
    cd crates/ran-neural && wasm-pack test --headless --firefox && cd ../..
    cd crates/ran-forecasting && wasm-pack test --headless --firefox && cd ../..
    cd crates/ran-edge && wasm-pack test --headless --firefox && cd ../..
fi

echo -e "${GREEN}🎉 WASM build completed successfully!${NC}"
echo -e "${BLUE}📁 Output directory: $OUTPUT_DIR${NC}"
echo -e "${BLUE}📦 Package: $OUTPUT_DIR/package.json${NC}"
echo -e "${BLUE}🔧 TypeScript: $OUTPUT_DIR/index.d.ts${NC}"
echo -e "${BLUE}📖 Examples: $OUTPUT_DIR/examples.js${NC}"
echo -e "${BLUE}📊 Size report: $SIZE_REPORT${NC}"

echo ""
echo -e "${YELLOW}💡 Next steps:${NC}"
echo "  1. Test in browser: Open examples.js in a web page"
echo "  2. Publish to npm: cd $OUTPUT_DIR && npm publish"
echo "  3. Use in projects: npm install @ranml/wasm"