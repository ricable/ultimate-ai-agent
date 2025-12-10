# RIC-TSA: QoE-aware Traffic Steering Application

## Overview

The RIC Traffic Steering Application (RIC-TSA) is a sophisticated implementation of QoE-aware traffic steering for Near-RT RIC (RAN Intelligent Controller) environments. This system provides sub-millisecond inference capabilities for real-time traffic steering decisions while optimizing user Quality of Experience (QoE).

## Key Features

### 🎯 QoE-Aware Intelligence
- **QoE Prediction Networks**: Multi-modal neural networks for predicting user experience metrics
- **User Group Classification**: ML-based classification for service differentiation
- **MAC Scheduler Optimization**: Neural network-driven resource allocation
- **A1 Policy Generation**: Automated policy creation for Near-RT RIC

### ⚡ Sub-Millisecond Performance
- **Streaming Inference Engine**: High-throughput batch processing
- **Knowledge Distillation**: Model compression for edge deployment
- **Pipeline Optimization**: Multi-stage processing for maximum efficiency
- **Result Caching**: Intelligent caching to avoid redundant computations

### 🌐 5G Network Optimization
- **Multi-Band Support**: 700MHz, 1800MHz, 2600MHz, 3500MHz, 28GHz
- **Carrier Aggregation**: Intelligent load balancing across carriers
- **MIMO Optimization**: Spatial multiplexing and beamforming
- **Interference Coordination**: Cross-cell interference mitigation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RIC-TSA Engine                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   QoE Predictor │  │ User Classifier │  │  MAC Scheduler  │  │
│  │                 │  │                 │  │                 │  │
│  │ • Multi-service │  │ • Behavior      │  │ • Resource      │  │
│  │   prediction    │  │   analysis      │  │   allocation    │  │
│  │ • Attention     │  │ • Service       │  │ • Power control │  │
│  │   mechanisms    │  │   preferences   │  │ • Beamforming   │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ A1 Policy Gen   │  │ Knowledge Dist  │  │ Streaming Inf   │  │
│  │                 │  │                 │  │                 │  │
│  │ • Dynamic       │  │ • Teacher-      │  │ • Batch         │  │
│  │   policies      │  │   student       │  │   processing    │  │
│  │ • Condition     │  │   training      │  │ • Cache         │  │
│  │   optimization  │  │ • Edge deploy   │  │   management    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Core Modules

#### 1. QoE Prediction (`qoe_prediction.rs`)
- **Purpose**: Predict Quality of Experience metrics for traffic steering decisions
- **Key Components**:
  - Primary prediction network with attention mechanisms
  - Service-specific networks (Video, Voice, Gaming, AR/VR, IoT, Emergency)
  - Temporal prediction for future QoE states
  - Multi-cell QoE scoring for steering decisions

```rust
pub struct QoEPredictor {
    primary_network: Arc<RwLock<NeuralNetwork>>,
    attention_network: Arc<RwLock<NeuralNetwork>>,
    service_networks: HashMap<ServiceType, Arc<RwLock<NeuralNetwork>>>,
    temporal_network: Arc<RwLock<NeuralNetwork>>,
    // ...
}
```

#### 2. User Classification (`user_classification.rs`)
- **Purpose**: Classify users into service groups based on behavior and requirements
- **Key Components**:
  - Multi-model classification (Primary, Behavior, Service, Temporal)
  - Feature extraction from traffic patterns and QoE sensitivity
  - User history tracking for temporal analysis
  - Behavior change detection

```rust
pub struct UserClassifier {
    primary_classifier: Arc<RwLock<NeuralNetwork>>,
    behavior_analyzer: Arc<RwLock<NeuralNetwork>>,
    service_predictor: Arc<RwLock<NeuralNetwork>>,
    temporal_classifier: Arc<RwLock<NeuralNetwork>>,
    // ...
}
```

#### 3. MAC Scheduler (`mac_scheduler.rs`)
- **Purpose**: Optimize resource allocation based on QoE predictions and user classifications
- **Key Components**:
  - Resource allocation network for PRB assignment
  - QoS parameter optimization
  - Power control and beamforming
  - Carrier aggregation management

```rust
pub struct MacScheduler {
    allocation_network: Arc<RwLock<NeuralNetwork>>,
    qos_network: Arc<RwLock<NeuralNetwork>>,
    power_network: Arc<RwLock<NeuralNetwork>>,
    beam_network: Arc<RwLock<NeuralNetwork>>,
    // ...
}
```

#### 4. A1 Policy Generator (`a1_policy.rs`)
- **Purpose**: Generate A1 policies for Near-RT RIC based on steering decisions
- **Key Components**:
  - Dynamic policy generation from ML outputs
  - Condition and action optimization
  - Policy template management
  - Performance tracking and adaptation

```rust
pub struct A1PolicyGenerator {
    policy_network: Arc<RwLock<NeuralNetwork>>,
    condition_network: Arc<RwLock<NeuralNetwork>>,
    action_network: Arc<RwLock<NeuralNetwork>>,
    policy_templates: Arc<RwLock<HashMap<A1PolicyType, PolicyTemplate>>>,
    // ...
}
```

#### 5. Knowledge Distillation (`knowledge_distillation.rs`)
- **Purpose**: Compress large teacher models into smaller student models for edge deployment
- **Key Components**:
  - Teacher-student training with temperature scaling
  - Model compression with configurable ratios
  - Edge model creation and validation
  - Performance benchmarking

```rust
pub struct KnowledgeDistillation {
    teacher_qoe_predictor: Arc<QoEPredictor>,
    student_qoe_predictor: Arc<RwLock<NeuralNetwork>>,
    distillation_loss: DistillationLoss,
    // ...
}
```

#### 6. Streaming Inference (`streaming_inference.rs`)
- **Purpose**: Provide high-throughput, low-latency inference for real-time operations
- **Key Components**:
  - Batch processing with configurable sizes
  - Result caching with TTL management
  - Performance monitoring and optimization
  - Pipeline processing for maximum throughput

```rust
pub struct StreamingInferenceEngine {
    qoe_batch_processor: Arc<RwLock<BatchProcessor>>,
    result_cache: Arc<RwLock<ResultCache>>,
    performance_monitor: Arc<RwLock<StreamingPerformanceMonitor>>,
    // ...
}
```

## User Groups and Service Types

### User Groups
- **Premium**: High QoE requirements, priority resource allocation
- **Standard**: Balanced QoE and efficiency
- **Basic**: Cost-optimized with basic QoE guarantees
- **IoT**: Low-power, high-reliability requirements
- **Emergency**: Critical priority with maximum reliability

### Service Types
- **Video Streaming**: High throughput, moderate latency tolerance
- **Voice Call**: Low latency, jitter-sensitive
- **Gaming**: Ultra-low latency, reliability-critical
- **File Transfer**: High throughput, latency-tolerant
- **Web Browsing**: Moderate requirements, bursty traffic
- **IoT Sensor**: Low data, high reliability
- **Emergency**: Critical priority, maximum reliability
- **AR/VR**: Ultra-high throughput, ultra-low latency

## Frequency Bands and Carriers

### Supported Bands
- **700MHz**: Long range, good penetration, rural coverage
- **1800MHz**: Balanced coverage and capacity
- **2600MHz**: High capacity, urban coverage
- **3500MHz**: 5G mid-band, high capacity
- **28GHz**: 5G mmWave, ultra-high capacity, short range

### Carrier Features
- **Bandwidth**: 10MHz to 200MHz depending on band
- **Load Balancing**: Dynamic distribution based on QoE predictions
- **Interference Coordination**: Cross-carrier optimization
- **Aggregation**: Multi-carrier bonding for high-throughput users

## Performance Targets

### Latency Requirements
- **Sub-millisecond Inference**: <1ms for real-time decisions
- **Batch Processing**: <0.1ms per UE in batch mode
- **End-to-End**: <5ms from measurement to steering action

### Throughput Targets
- **Single Decisions**: >1000 decisions/second
- **Batch Processing**: >10,000 UEs/second
- **Streaming Mode**: Continuous processing with <100μs gaps

### Accuracy Targets
- **QoE Prediction**: >90% accuracy vs ground truth
- **User Classification**: >85% classification accuracy
- **Steering Success**: >95% improvement in target QoE metrics

## Usage Examples

### Basic Traffic Steering

```rust
use ran_opt::ric_tsa::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize RIC-TSA engine
    let mut ric_engine = RicTsaEngine::new()?;
    
    // Add network topology
    let carrier = CellCarrier {
        carrier_id: 1,
        band: FrequencyBand::Band1800MHz,
        bandwidth: 20,
        current_load: 50.0,
        max_capacity: 150.0,
        coverage_area: 10.0,
    };
    ric_engine.add_cell_carrier(carrier);
    
    // Register user equipment
    let ue_context = UEContext {
        ue_id: 1,
        user_group: UserGroup::Premium,
        service_type: ServiceType::VideoStreaming,
        current_qoe: QoEMetrics {
            throughput: 25.0,
            latency: 15.0,
            jitter: 3.0,
            packet_loss: 0.05,
            video_quality: 4.5,
            audio_quality: 4.8,
            reliability: 99.5,
            availability: 99.9,
        },
        // ... other fields
    };
    ric_engine.register_ue(ue_context);
    
    // Make steering decision
    let decision = ric_engine.make_steering_decision(1).await?;
    println!("Steering decision: {:?}", decision);
    
    // Generate A1 policy
    let policy = ric_engine.generate_a1_policy(&decision).await?;
    println!("Generated policy: {}", policy.policy_id);
    
    Ok(())
}
```

### Knowledge Distillation for Edge Deployment

```rust
use ran_opt::ric_tsa::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create teacher models
    let teacher_qoe = Arc::new(QoEPredictor::new()?);
    let teacher_classifier = Arc::new(UserClassifier::new()?);
    let teacher_scheduler = Arc::new(MacScheduler::new()?);
    
    // Initialize knowledge distillation
    let distillation = KnowledgeDistillation::new(
        teacher_qoe,
        teacher_classifier, 
        teacher_scheduler,
    )?;
    
    // Collect training data
    let ue_contexts = vec![/* ... UE contexts ... */];
    distillation.collect_training_data(&ue_contexts).await?;
    
    // Train student models
    distillation.train_student_models().await?;
    
    // Create edge model
    let edge_model = distillation.create_edge_model().await?;
    
    // Validate performance
    let validation = distillation.validate_student_models(&ue_contexts).await?;
    println!("Edge model accuracy: {:.1}%", validation.qoe_accuracy * 100.0);
    
    Ok(())
}
```

### Streaming Inference

```rust
use ran_opt::ric_tsa::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize streaming engine
    let streaming_engine = StreamingInferenceEngine::new()?;
    
    // Register edge model
    let edge_model = Arc::new(/* ... create edge model ... */);
    streaming_engine.register_edge_model("primary".to_string(), edge_model).await?;
    
    // Process batch of UEs
    let ue_ids = vec![1, 2, 3, 4, 5];
    let ue_contexts = HashMap::new(); // ... populate with UE contexts
    
    let results = streaming_engine.process_batch(&ue_ids, &ue_contexts).await?;
    
    // Display results
    for (ue_id, result) in results {
        println!("UE {}: Cell {} ({:.0}% confidence)", 
            ue_id, result.target_cell, result.confidence * 100.0);
    }
    
    // Check performance
    let metrics = streaming_engine.get_performance_metrics().await;
    println!("Throughput: {:.0} req/sec", metrics.throughput_requests_per_sec);
    
    Ok(())
}
```

## Configuration

### QoE Prediction Configuration

```rust
pub struct QoEPredictorConfig {
    pub input_size: usize,           // 32
    pub hidden_sizes: Vec<usize>,    // [256, 128, 64]
    pub output_size: usize,          // 8 (QoE metrics)
    pub attention_heads: usize,      // 8
    pub dropout_rate: f32,           // 0.1
    pub learning_rate: f32,          // 0.001
    pub batch_size: usize,           // 32
    pub sequence_length: usize,      // 10
    pub prediction_horizon: u32,     // 30 seconds
}
```

### Knowledge Distillation Configuration

```rust
pub struct KnowledgeDistillationConfig {
    pub temperature: f32,            // 4.0 (softmax temperature)
    pub alpha: f32,                  // 0.7 (distillation loss weight)
    pub beta: f32,                   // 0.3 (student loss weight)
    pub learning_rate: f32,          // 0.001
    pub batch_size: usize,           // 32
    pub distillation_epochs: usize,  // 50
    pub compression_ratio: f32,      // 0.1 (10x compression)
    pub inference_time_target: f32,  // 0.5ms
}
```

### Streaming Inference Configuration

```rust
pub struct StreamingInferenceConfig {
    pub max_batch_size: usize,       // 64
    pub batch_timeout_ms: u64,       // 1ms
    pub max_concurrent_batches: usize, // 4
    pub cache_ttl_ms: u64,           // 100ms
    pub enable_prefetching: bool,    // true
    pub enable_result_caching: bool, // true
    pub pipeline_stages: usize,      // 5
    pub thread_pool_size: usize,     // 8
}
```

## Performance Optimization

### Sub-Millisecond Inference
1. **Model Compression**: Use knowledge distillation to create 10x smaller models
2. **Batch Processing**: Process multiple UEs simultaneously
3. **Result Caching**: Cache frequently-used predictions
4. **Pipeline Optimization**: Overlap computation and I/O operations
5. **Memory Management**: Pre-allocate tensors and reuse memory pools

### Throughput Optimization
1. **Parallel Processing**: Use multiple CPU cores and GPU acceleration
2. **Streaming Architecture**: Continuous processing without blocking
3. **Load Balancing**: Distribute work across multiple instances
4. **Priority Queuing**: Process emergency traffic first
5. **Adaptive Batching**: Adjust batch sizes based on load

### Accuracy Optimization
1. **Ensemble Methods**: Combine multiple model predictions
2. **Online Learning**: Continuously update models with feedback
3. **Feature Engineering**: Extract relevant features for predictions
4. **Attention Mechanisms**: Focus on important input features
5. **Temporal Modeling**: Use historical data for better predictions

## Integration Points

### Near-RT RIC Integration
- **A1 Interface**: Policy management and configuration
- **E2 Interface**: UE measurements and KPIs
- **O1 Interface**: Performance monitoring and alarms
- **xApp Framework**: Deployment as RIC xApp

### 5G Core Integration
- **SMF**: Session management and QoS policies
- **PCF**: Policy control and charging rules
- **NSSF**: Network slice selection
- **UDM**: User data management

### RAN Integration
- **gNB**: Base station integration via E2 interface
- **CU/DU Split**: Central and distributed unit coordination
- **RIC Platform**: Integration with O-RAN compliant RIC

## Testing and Validation

### Unit Tests
- Individual component testing for each module
- Mock data generation for reproducible tests
- Performance benchmarks for latency and throughput

### Integration Tests
- End-to-end workflow testing
- Multi-UE scenario validation
- Network topology stress testing

### Performance Tests
- Latency measurement under various loads
- Throughput testing with different batch sizes
- Memory usage and optimization validation

## Deployment

### Edge Deployment
1. **Model Optimization**: Use knowledge distillation for compression
2. **Container Packaging**: Docker containers with optimized runtimes
3. **Resource Management**: CPU/GPU resource allocation
4. **Monitoring**: Real-time performance monitoring
5. **Auto-scaling**: Dynamic scaling based on load

### Cloud Deployment
1. **Kubernetes**: Container orchestration
2. **Service Mesh**: Inter-service communication
3. **Load Balancing**: Traffic distribution
4. **High Availability**: Multi-zone deployment
5. **Observability**: Logging, metrics, and tracing

## Monitoring and Observability

### Key Metrics
- **Inference Latency**: p50, p95, p99 response times
- **Throughput**: Requests per second
- **Accuracy**: Prediction vs actual QoE
- **Resource Utilization**: CPU, memory, GPU usage
- **Error Rates**: Failed predictions and timeouts

### Alerting
- **SLA Violations**: Latency or accuracy below thresholds
- **Resource Exhaustion**: Memory or CPU limits reached
- **Model Drift**: Accuracy degradation over time
- **System Failures**: Component crashes or timeouts

### Dashboards
- **Real-time Performance**: Live metrics and charts
- **Historical Trends**: Long-term performance analysis
- **Capacity Planning**: Resource utilization forecasts
- **QoE Analytics**: User experience improvement tracking

## Future Enhancements

### Advanced ML Techniques
- **Transformer Models**: Attention-based architectures
- **Federated Learning**: Distributed model training
- **Reinforcement Learning**: Policy optimization
- **Multi-Agent Systems**: Collaborative decision making

### 6G Preparation
- **AI-Native Architecture**: Built-in AI capabilities
- **Semantic Communications**: Content-aware networking
- **Digital Twin**: Virtual network representation
- **Autonomous Networks**: Self-optimizing systems

### Performance Improvements
- **Quantum Computing**: Quantum-enhanced optimization
- **Neuromorphic Computing**: Brain-inspired architectures
- **In-Memory Computing**: Reduced memory access latency
- **Optical Computing**: Light-based processing

---

## Contributing

Please refer to the main project README for contribution guidelines and development setup instructions.

## License

This project is licensed under the same terms as the main ran-opt project.