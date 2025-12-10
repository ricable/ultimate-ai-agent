# DTM Traffic Prediction System

## Overview

The DTM (Dynamic Traffic Management) Traffic Prediction System is a comprehensive neural network-based solution for predicting cellular network traffic patterns and generating automated load balancing scripts. It combines LSTM, GRU, and Temporal Convolutional Networks (TCN) to provide accurate multi-horizon forecasting with QoS-aware predictions.

## Features

### 🧠 Neural Network Models
- **LSTM with Custom Gates**: Long Short-Term Memory networks with peephole connections for capturing long-term dependencies
- **GRU Cells**: Gated Recurrent Units for efficient sequence modeling
- **Temporal Convolutional Networks**: TCN with dilated convolutions for parallel processing of temporal sequences
- **Ensemble Learning**: Combines predictions from multiple models with optimized weights

### 📊 Traffic Pattern Modeling
- **PRB Utilization Forecasting**: Predicts Physical Resource Block usage across network layers
- **Layer-Specific Demand**: Separate predictions for L2100 (LTE), N78 (3.5GHz 5G), and N258 (mmWave) layers
- **Service Type Classification**: Automatic classification of eMBB, VoNR, URLLC, and MIoT services
- **QoS-Aware Predictions**: Incorporates 5QI indicators for Quality of Service management

### 🚀 CUDA Acceleration
- **Custom CUDA Kernels**: Optimized GPU kernels for LSTM/GRU operations
- **Memory Management**: Efficient GPU memory allocation and data transfer
- **Fallback Support**: Automatic fallback to CPU implementation when CUDA is unavailable

### 📜 AMOS Script Generation
- **Automated Load Balancing**: Generates AMOS (Automated Network Management Operations System) scripts
- **Multi-Action Support**: Handover triggers, antenna tilt adjustments, power control, QoS updates
- **Safety Features**: Includes validation checks and rollback scripts
- **Emergency Scripts**: Immediate response scripts for critical network conditions

## Architecture

### Core Components

```
TrafficPredictor
├── LSTMPredictor (LSTM cells with custom gates)
├── GRUPredictor (GRU cells)
├── TCNPredictor (Temporal Convolutional Networks)
└── AmosScriptGenerator (Load balancing script generation)
```

### Data Flow

1. **Input**: Historical traffic patterns with PRB utilization, QoS indicators, service types
2. **Feature Extraction**: Time series features, layer encoding, service type encoding
3. **Model Inference**: Parallel prediction using LSTM, GRU, and TCN models
4. **Ensemble**: Weighted combination of model predictions
5. **Output**: Multi-horizon forecasts with confidence intervals
6. **Action Generation**: AMOS scripts for proactive load balancing

## Usage

### Basic Usage

```rust
use ran_opt::dtm_traffic::{TrafficPredictor, PredictorConfig, TrafficPattern};

// Create configuration
let config = PredictorConfig::default();

// Initialize predictor
let mut predictor = TrafficPredictor::new(config);

// Train on historical data
predictor.train(&training_patterns, 100)?;

// Make predictions
let predictions = predictor.predict(&recent_patterns)?;

// Generate AMOS scripts
let scripts = predictor.generate_amos_scripts(&predictions);
```

### Advanced Configuration

```rust
let config = PredictorConfig {
    sequence_length: 96,  // 24 hours with 15-min intervals
    forecast_horizons: vec![4, 8, 16, 24], // 1h, 2h, 4h, 6h
    lstm_hidden_size: 256,
    gru_hidden_size: 256,
    tcn_filters: 64,
    tcn_kernel_size: 3,
    tcn_dilations: vec![1, 2, 4, 8, 16, 32],
    dropout_rate: 0.2,
    learning_rate: 0.001,
    batch_size: 32,
};
```

## Network Layers

### L2100 (LTE 2100 MHz)
- **Characteristics**: Wide coverage, moderate capacity
- **Service Types**: Voice (VoNR), basic data services
- **Load Balancing**: Offload to 5G layers when congested

### N78 (5G NR 3.5 GHz)
- **Characteristics**: Good coverage-capacity balance
- **Service Types**: eMBB, URLLC, enhanced data services
- **Load Balancing**: Primary 5G layer, carrier aggregation with mmWave

### N258 (5G NR 26 GHz mmWave)
- **Characteristics**: High capacity, limited coverage
- **Service Types**: Ultra-high throughput eMBB
- **Load Balancing**: Hotspot coverage, carrier aggregation secondary

## Service Types

### eMBB (Enhanced Mobile Broadband)
- **Priority**: Medium
- **Bandwidth**: High
- **Latency**: Moderate
- **Typical Use**: Video streaming, web browsing

### VoNR (Voice over New Radio)
- **Priority**: High
- **Bandwidth**: Low
- **Latency**: Low
- **Typical Use**: Voice calls, real-time communication

### URLLC (Ultra-Reliable Low-Latency Communications)
- **Priority**: Highest
- **Bandwidth**: Variable
- **Latency**: Ultra-low (<1ms)
- **Typical Use**: Industrial automation, autonomous vehicles

### MIoT (Massive IoT)
- **Priority**: Low
- **Bandwidth**: Very low
- **Latency**: High tolerance
- **Typical Use**: Sensors, smart meters, monitoring devices

## QoS Indicators (5QI)

The system supports multiple 5QI values for different service requirements:

- **QI1**: Conversational Voice
- **QI2**: Conversational Video (Live Streaming)
- **QI3**: Real Time Gaming
- **QI4**: Non-Conversational Video (Buffered Streaming)
- **QI5**: IMS Signaling
- **QI65-70**: Mission Critical Services
- **QI79**: V2X Messages
- **QI80**: Low Latency eMBB Applications

## AMOS Script Actions

### Load Balancing Actions

1. **Trigger Handover**
   ```bash
   amos_cli handover --source-freq 2100 --target-freq 3500 --percentage 0.3
   ```

2. **Adjust Antenna Tilt**
   ```bash
   amos_cli antenna --freq 2100 --tilt-adjustment 2.0 --apply
   ```

3. **Modify Transmission Power**
   ```bash
   amos_cli power --freq 3500 --adjustment 3.0 --apply
   ```

4. **Update QoS Parameters**
   ```bash
   amos_cli qos --freq 3500 --service URLLC --priority 1 --bandwidth 0.9
   ```

5. **Configure Carrier Aggregation**
   ```bash
   amos_cli carrier-aggregation --primary 3500 --secondary 26000 --enable true
   ```

### Safety Features

- **Validation Checks**: Verify parameter ranges and network state
- **Rollback Scripts**: Automatic generation of rollback commands
- **Emergency Scripts**: Immediate response for critical conditions
- **Gradual Implementation**: Phased deployment to minimize disruption

## Performance Optimizations

### CUDA Acceleration

The system includes custom CUDA kernels for:
- Matrix operations (GEMM) using CUBLAS
- Element-wise operations (sigmoid, tanh, relu)
- Memory-efficient batch processing
- Automatic fallback to CPU when CUDA unavailable

### Memory Management

- Efficient GPU memory allocation
- Batch processing to maximize GPU utilization
- Streaming for large datasets
- Memory pooling for reduced allocation overhead

## Monitoring and Validation

### Model Performance Metrics
- **MAE (Mean Absolute Error)**: Average prediction error
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **MAPE (Mean Absolute Percentage Error)**: Relative error percentage
- **R² Score**: Coefficient of determination

### Network Impact Metrics
- **PRB Utilization Accuracy**: Prediction vs. actual utilization
- **Handover Success Rate**: Percentage of successful handovers
- **QoS Improvement**: Before/after QoS metrics
- **User Experience**: Throughput and latency improvements

## Configuration Files

### Threshold Configuration
```rust
ThresholdConfig {
    prb_high_threshold: 0.8,      // 80% PRB utilization
    prb_medium_threshold: 0.6,    // 60% PRB utilization
    prb_low_threshold: 0.4,       // 40% PRB utilization
    user_count_threshold: 100,    // Maximum users per cell
    throughput_threshold: 500.0,  // Mbps threshold
    qos_degradation_threshold: 0.1, // 10% QoS degradation
}
```

### Template Configuration
```rust
TemplateConfig {
    cell_id_prefix: "CELL".to_string(),
    frequency_bands: HashMap::from([
        (NetworkLayer::L2100, "2100".to_string()),
        (NetworkLayer::N78, "3500".to_string()),
        (NetworkLayer::N258, "26000".to_string()),
    ]),
    service_priorities: HashMap::from([
        (ServiceType::URLLC, 1),  // Highest priority
        (ServiceType::VoNR, 2),
        (ServiceType::EMBB, 3),
        (ServiceType::MIoT, 4),   // Lowest priority
    ]),
    handover_hysteresis: 3.0,     // dB
    handover_time_to_trigger: 160, // ms
}
```

## File Structure

```
src/dtm_traffic/
├── mod.rs                  # Main module with TrafficPredictor
├── models.rs              # LSTM, GRU, TCN implementations
├── cuda_kernels.rs        # CUDA acceleration kernels
└── amos_generator.rs      # Load balancing script generation

examples/
└── dtm_traffic_prediction.rs  # Usage example
```

## Dependencies

### Core Dependencies
- `ndarray`: Multi-dimensional array operations
- `ndarray-rand`: Random number generation for arrays
- `chrono`: Date and time handling
- `serde`: Serialization/deserialization

### CUDA Dependencies
- `libloading`: Dynamic library loading for CUDA runtime
- CUDA runtime library (libcudart.so)
- CUBLAS library (libcublas.so)

### Optional Dependencies
- `plotters`: Visualization of predictions and metrics
- `criterion`: Performance benchmarking

## Testing

Run the comprehensive test suite:
```bash
cargo test dtm_traffic
```

Run specific test categories:
```bash
cargo test dtm_traffic::models  # Model tests
cargo test dtm_traffic::cuda    # CUDA tests
cargo test dtm_traffic::amos    # AMOS script tests
```

## Benchmarking

Performance benchmarks are available:
```bash
cargo bench dtm_traffic_bench
```

## Future Enhancements

### Short-term
- Support for additional 5QI indicators
- Enhanced CUDA kernel optimization
- Real-time streaming prediction API
- Integration with network management systems

### Long-term
- Federated learning across multiple sites
- Reinforcement learning for dynamic optimization
- Integration with network slicing management
- Advanced anomaly detection and prediction