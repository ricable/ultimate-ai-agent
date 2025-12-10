# AFM-Correlate: Autonomous Fault Management Correlation Engine

## Overview

AFM-Correlate is an advanced correlation engine implementing hierarchical attention networks for multi-scale evidence synthesis across domains. It provides comprehensive correlation capabilities for telecom network fault management, integrating KPI deviations, alarm sequences, configuration changes, and topology impacts into coherent evidence bundles with confidence scores.

## Architecture

### Core Components

#### 1. Cross-Attention Mechanisms (`cross_attention.rs`)
- **Multi-head attention** for evidence correlation
- **Cross-domain attention** between different evidence sources
- **Sinusoidal position encoding** for temporal relationships
- **Scaled dot-product attention** with dropout regularization

**Key Features:**
- Supports 8 attention heads with configurable hidden dimensions
- Xavier initialization for stable training
- Cosine similarity computation for feature relationships
- Source-specific attention weighting

#### 2. Multi-Source Fusion Networks (`fusion_network.rs`)
- **Source-specific encoders** for each evidence type
- **Cross-source attention** for inter-domain relationships
- **Hierarchical fusion** with residual connections
- **Layer normalization** for stable feature learning

**Evidence Sources Supported:**
- KPI Deviations
- Alarm Sequences  
- Configuration Changes
- Topology Impacts
- Performance Metrics
- Log Patterns

#### 3. Temporal Alignment Algorithms (`temporal_alignment.rs`)
- **Dynamic Time Warping (DTW)** for sequence alignment
- **Temporal pattern detection** (burst, cascade, periodic, cross-source)
- **Multi-scale windowing** for different time granularities
- **Position-aware alignment** with confidence scoring

**Pattern Types:**
- **Burst**: Multiple events in short time windows
- **Cascade**: Escalating severity patterns
- **Periodic**: Recurring temporal sequences
- **Cross-source**: Multi-domain temporal correlations

#### 4. Evidence Scoring Networks (`evidence_scoring.rs`)
- **Neural network scoring** with 3-layer architecture
- **Pairwise correlation assessment** between evidence items
- **Group coherence analysis** for evidence clusters
- **Impact assessment** with propagation risk evaluation

**Scoring Dimensions:**
- Feature similarity (cosine similarity)
- Temporal proximity (exponential decay)
- Severity correlation
- Confidence correlation
- Source relationship strength
- Metadata similarity

#### 5. Hierarchical Attention Networks (`hierarchical_attention.rs`)
- **Multi-scale processing** (fine, medium, coarse granularity)
- **Scale-specific attention layers** with position encodings
- **Cross-scale attention** for feature integration
- **Patch-based processing** for scalability

**Scale Levels:**
- **Fine-grained**: Individual metric deviations (4-item patches)
- **Medium-scale**: Component-level correlations (8-item patches)
- **Coarse-scale**: System-wide patterns (16-item patches)

## Data Structures

### Evidence Item
```rust
pub struct EvidenceItem {
    pub id: String,
    pub source: EvidenceSource,
    pub timestamp: DateTime<Utc>,
    pub severity: f32,
    pub confidence: f32,
    pub features: Vec<f32>,
    pub metadata: HashMap<String, String>,
}
```

### Correlation Result
```rust
pub struct CorrelationResult {
    pub correlation_id: String,
    pub evidence_items: Vec<EvidenceItem>,
    pub correlation_score: f32,
    pub confidence: f32,
    pub temporal_alignment: f32,
    pub cross_domain_score: f32,
    pub impact_assessment: ImpactAssessment,
}
```

### Evidence Bundle
```rust
pub struct EvidenceBundle {
    pub bundle_id: String,
    pub timestamp: DateTime<Utc>,
    pub correlations: Vec<CorrelationResult>,
    pub hierarchical_score: f32,
    pub multi_scale_features: HashMap<String, Vec<f32>>,
    pub synthesis_report: SynthesisReport,
}
```

## Configuration

### Correlation Configuration
```rust
pub struct CorrelationConfig {
    pub temporal_window: Duration,        // Default: 15 minutes
    pub min_correlation_score: f32,       // Default: 0.7
    pub max_evidence_items: usize,        // Default: 100
    pub attention_heads: usize,           // Default: 8
    pub hidden_dim: usize,                // Default: 256
    pub dropout_rate: f32,                // Default: 0.1
}
```

## Usage Examples

### Basic Correlation Engine Setup
```rust
use afm_correlate::{CorrelationEngine, CorrelationConfig, EvidenceItem, EvidenceSource};
use chrono::{Utc, Duration};
use std::collections::HashMap;

#[tokio::main]
async fn main() {
    // Configure correlation engine
    let config = CorrelationConfig {
        temporal_window: Duration::minutes(15),
        min_correlation_score: 0.7,
        max_evidence_items: 100,
        attention_heads: 8,
        hidden_dim: 256,
        dropout_rate: 0.1,
    };
    
    let engine = CorrelationEngine::new(config);
    
    // Create evidence item
    let evidence = EvidenceItem {
        id: "kpi_cpu_spike_001".to_string(),
        source: EvidenceSource::KpiDeviation,
        timestamp: Utc::now(),
        severity: 0.85,
        confidence: 0.92,
        features: vec![0.85, 0.12, 0.67, 0.34, 0.89],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("metric".to_string(), "cpu_utilization".to_string());
            meta.insert("component".to_string(), "compute_node_1".to_string());
            meta.insert("threshold".to_string(), "80%".to_string());
            meta.insert("actual".to_string(), "94%".to_string());
            meta
        },
    };
    
    // Add evidence and correlate
    engine.add_evidence(evidence).await;
    let bundles = engine.correlate().await;
    
    // Process results
    for bundle in bundles {
        println!("Bundle: {} (score: {:.3})", 
                bundle.bundle_id, bundle.hierarchical_score);
        
        for correlation in bundle.correlations {
            println!("  Correlation: {} (score: {:.3})", 
                    correlation.correlation_id, correlation.correlation_score);
            println!("    Cross-domain: {:.3}", correlation.cross_domain_score);
            println!("    Impact severity: {:.3}", 
                    correlation.impact_assessment.severity);
        }
    }
}
```

### Cross-Domain Correlation Example
```rust
// Create evidence from different domains
let kpi_evidence = EvidenceItem {
    id: "kpi_001".to_string(),
    source: EvidenceSource::KpiDeviation,
    // ... other fields
};

let alarm_evidence = EvidenceItem {
    id: "alarm_001".to_string(),
    source: EvidenceSource::AlarmSequence,
    // ... other fields
};

let config_evidence = EvidenceItem {
    id: "config_001".to_string(),
    source: EvidenceSource::ConfigurationChange,
    // ... other fields
};

// Add all evidence items
engine.add_evidence(kpi_evidence).await;
engine.add_evidence(alarm_evidence).await;
engine.add_evidence(config_evidence).await;

// Perform cross-domain correlation
let bundles = engine.correlate().await;

for bundle in bundles {
    // Check for cross-domain correlations
    for correlation in bundle.correlations {
        if correlation.cross_domain_score > 0.8 {
            println!("Strong cross-domain correlation detected!");
            println!("Sources involved: {:?}", 
                correlation.evidence_items.iter()
                .map(|e| e.source).collect::<Vec<_>>());
        }
    }
}
```

## Evidence Sources and Correlation Types

### 1. KPI Deviations
- **CPU utilization spikes**
- **Memory pressure events**  
- **Network latency increases**
- **Throughput degradation**
- **Error rate anomalies**

### 2. Alarm Sequences
- **Hardware failures**
- **Service degradation alerts**
- **Resource exhaustion warnings**
- **Connectivity issues**
- **Performance threshold breaches**

### 3. Configuration Changes
- **Parameter modifications**
- **Software updates**
- **Policy changes**
- **Automatic rollbacks**
- **Optimization adjustments**

### 4. Topology Impacts
- **Neighbor relation updates**
- **Cell configuration changes**
- **Handover parameter modifications**
- **Coverage optimization**
- **Capacity adjustments**

### 5. Performance Metrics
- **Service quality indicators**
- **User experience metrics**
- **Network efficiency measures**
- **Resource utilization stats**
- **Availability metrics**

### 6. Log Patterns
- **Error message sequences**
- **Warning patterns**
- **Diagnostic information**
- **Operational events**
- **System state changes**

## Correlation Algorithms

### Cross-Attention Mechanism
1. **Input Processing**: Convert evidence features to embeddings
2. **Multi-Head Attention**: Compute attention across evidence items
3. **Cross-Domain Weighting**: Apply source-specific attention weights
4. **Feature Integration**: Combine attended features with residuals

### Temporal Alignment
1. **Pattern Detection**: Identify burst, cascade, periodic, and cross-source patterns
2. **Window Grouping**: Organize evidence by temporal windows
3. **DTW Alignment**: Apply Dynamic Time Warping for sequence matching
4. **Confidence Scoring**: Assess temporal correlation strength

### Evidence Scoring
1. **Feature Similarity**: Compute cosine similarity between evidence features
2. **Temporal Proximity**: Apply exponential decay based on time differences
3. **Source Relationships**: Use predefined source correlation matrix
4. **Impact Assessment**: Evaluate severity, scope, and propagation risk

### Hierarchical Attention
1. **Multi-Scale Patches**: Create patches at different granularities
2. **Scale-Specific Processing**: Apply attention within each scale
3. **Cross-Scale Integration**: Combine features across scales
4. **Hierarchical Scoring**: Compute overall correlation confidence

## Performance Characteristics

### Scalability
- **Evidence Buffer**: Configurable maximum items (default: 100)
- **Temporal Windows**: Sliding window approach for real-time processing
- **Attention Complexity**: O(n²) within patches, linear across scales
- **Memory Usage**: Bounded by evidence buffer size and window duration

### Accuracy
- **Correlation Threshold**: Configurable minimum score (default: 0.7)
- **Cross-Domain Weighting**: Source relationship matrix tuning
- **Temporal Tolerance**: Alignment window configuration
- **Confidence Assessment**: Multi-factor confidence computation

### Real-Time Performance
- **Streaming Processing**: Continuous evidence ingestion
- **Incremental Correlation**: Window-based correlation updates
- **Asynchronous Operation**: Non-blocking evidence addition
- **Parallel Processing**: Multi-threaded attention computation

## Integration with RAN-OPT

AFM-Correlate integrates seamlessly with other RAN-OPT components:

### AFM-Detect Integration
- **Evidence Input**: Receives anomaly detections as evidence items
- **Confidence Propagation**: Uses AFM-Detect confidence scores
- **Feature Sharing**: Leverages autoencoder features for correlation

### DTM Integration
- **Traffic Pattern Correlation**: Integrates traffic predictions with KPI deviations
- **Power State Correlation**: Correlates power events with performance impacts
- **Mobility Correlation**: Integrates handover events with network performance

### PFS Integration
- **Log Pattern Evidence**: Processes PFS log anomalies as evidence items
- **Performance Correlation**: Integrates system performance with network KPIs
- **Data Pipeline Integration**: Receives streaming evidence from PFS data pipelines

## Testing and Validation

### Unit Tests
- **Component Testing**: Individual module validation
- **Integration Testing**: Cross-component functionality
- **Performance Testing**: Scalability and throughput validation
- **Accuracy Testing**: Correlation quality assessment

### Demonstration Scripts
- **Basic Demo**: `afm_correlate_demo.rs` - Shows core functionality
- **Integration Examples**: `examples.rs` - Comprehensive use cases
- **Test Suite**: `integration_test.rs` - Validation framework

### Benchmarking
- **Correlation Latency**: Time to process evidence and generate correlations
- **Memory Usage**: Evidence buffer and processing memory requirements
- **Throughput**: Evidence items processed per second
- **Accuracy Metrics**: Precision and recall for correlation detection

## Future Enhancements

### Machine Learning Integration
- **Adaptive Thresholds**: Learning-based correlation score adjustment
- **Feature Learning**: Automated feature extraction from raw evidence
- **Pattern Recognition**: Advanced temporal pattern detection
- **Causal Inference**: Root cause analysis enhancement

### Advanced Algorithms
- **Graph Neural Networks**: Network topology-aware correlation
- **Transformer Models**: Advanced attention mechanisms
- **Reinforcement Learning**: Adaptive correlation strategies
- **Federated Learning**: Distributed correlation across network domains

### Operational Features
- **Real-Time Dashboards**: Correlation visualization
- **Alert Integration**: Automated incident creation
- **API Extensions**: RESTful correlation services
- **Configuration Management**: Dynamic parameter tuning

## Conclusion

AFM-Correlate provides a comprehensive correlation engine for autonomous fault management in RAN environments. Its hierarchical attention architecture enables multi-scale evidence synthesis across domains, providing high-confidence correlation results with detailed impact assessments and actionable recommendations.

The implementation combines state-of-the-art attention mechanisms with domain-specific correlation logic, enabling effective correlation of complex multi-source evidence patterns in real-time telecom network operations.