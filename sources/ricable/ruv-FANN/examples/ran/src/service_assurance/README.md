# RAN Intelligence Platform - Service Assurance

This directory contains the service assurance components of the RAN Intelligence Platform, implementing proactive issue detection and mitigation systems for cellular networks.

## Components

### ASA-INT-01 - Uplink Interference Classifier
**Target: >95% classification accuracy**

Located in `asa-int/`

**Features:**
- Real-time uplink interference classification using machine learning
- Supports multiple interference types: External Jammer, PIM, Adjacent Channel, Thermal Noise, Legitimate Traffic
- Advanced noise floor analysis with spectral, temporal, and statistical features
- Ensemble classification using Random Forest, SVM, and Neural Network models
- Automated mitigation strategy recommendations
- Performance monitoring with 95% accuracy requirement validation

**Key Capabilities:**
- Multi-dimensional feature extraction from noise floor measurements
- Signal processing with FFT analysis and spectral feature computation
- Configurable classification thresholds and model architectures
- Real-time prediction with sub-millisecond response times
- Comprehensive mitigation recommendations based on interference type

### ASA-5G-01 - ENDC Setup Failure Predictor
**Target: >80% failure prediction accuracy**

Located in `asa-5g/`

**Features:**
- Predictive analysis of EN-DC (E-UTRAN New Radio - Dual Connectivity) setup failures
- Real-time signal quality assessment for LTE and NR components
- Bearer configuration optimization recommendations
- Proactive failure prevention through early warning systems
- 5G NSA/SA service health monitoring

**Key Capabilities:**
- Multi-factor prediction using signal quality, network congestion, and UE capability
- Support for all ENDC failure types: Initial Setup, Bearer Setup, Bearer Modification, Release
- Advanced feature engineering with signal correlation analysis
- Gradient boosting prediction model with 80% accuracy validation
- Automated mitigation strategies including bearer reconfiguration and cell reselection

### ASA-QOS-01 - Predictive VoLTE Jitter Forecaster
**Target: Accuracy within 10ms of actual jitter**

Located in `asa-qos/`

**Features:**
- Advanced VoLTE jitter prediction using time-series forecasting
- Multiple ML models: ARIMA, LSTM, GRU, Transformer, Ensemble
- Real-time quality degradation alerting
- Voice quality optimization recommendations
- Service assurance dashboard integration

**Key Capabilities:**
- Sub-10ms jitter prediction accuracy
- Real-time forecasting with configurable horizon
- Quality trend analysis and pattern recognition
- Automated QoS parameter optimization
- Integration with network performance monitoring

## Architecture

```
Service Assurance
├── ASA-INT-01 (Interference Classifier)
│   ├── Feature Extraction Engine
│   ├── Ensemble ML Models
│   ├── Mitigation Strategy Engine
│   └── Performance Monitor
├── ASA-5G-01 (ENDC Failure Predictor)
│   ├── Signal Quality Analyzer
│   ├── Failure Prediction Engine
│   ├── Bearer Optimization Engine
│   └── Health Monitor
└── ASA-QOS-01 (VoLTE Jitter Forecaster)
    ├── Time-Series Models
    ├── Quality Analyzer
    ├── Trend Predictor
    └── Optimization Engine
```

## Performance Requirements

| Component | Metric | Target | Validation |
|-----------|--------|--------|------------|
| ASA-INT-01 | Classification Accuracy | >95% | Ensemble validation with confusion matrix |
| ASA-5G-01 | Prediction Accuracy | >80% | Cross-validation on failure events |
| ASA-QOS-01 | Jitter Accuracy | ±10ms | Time-series validation with RMSE |

## Integration

All service assurance components integrate with:
- **Platform Foundation Services**: Data ingestion, feature engineering, model registry
- **Network Intelligence**: Capacity prediction, slice management, clustering
- **Predictive Optimization**: Resource allocation, mobility optimization
- **Common Services**: Configuration, monitoring, alerting

## Configuration

Each component supports configuration through:
- TOML configuration files
- Environment variables
- gRPC configuration endpoints

Example configuration locations:
- `asa-int/config.toml` - Interference classifier settings
- `asa-5g/config.toml` - ENDC predictor configuration  
- `asa-qos/config.toml` - VoLTE forecaster parameters

## Deployment

### Docker Deployment
```bash
# Build all service assurance components
docker-compose -f docker/docker-compose.yml up service-assurance

# Individual services
docker-compose up asa-int-service
docker-compose up asa-5g-service  
docker-compose up asa-qos-service
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/service-assurance/
```

### Development
```bash
# Start individual services
cd asa-int && cargo run --bin asa-int-server
cd asa-5g && cargo run --bin asa-5g-server
cd asa-qos && cargo run --bin asa-qos-server
```

## APIs

### gRPC Services

#### Interference Classification
```protobuf
service InterferenceClassifier {
    rpc ClassifyUlInterference(ClassifyRequest) returns (ClassifyResponse);
    rpc GetMitigationRecommendations(MitigationRequest) returns (MitigationResponse);
    rpc TrainModel(TrainRequest) returns (TrainResponse);
}
```

#### ENDC Failure Prediction
```protobuf
service ServiceAssuranceService {
    rpc PredictEndcFailure(PredictEndcFailureRequest) returns (PredictEndcFailureResponse);
    rpc Get5GServiceHealth(Get5GServiceHealthRequest) returns (Get5GServiceHealthResponse);
}
```

#### VoLTE Jitter Forecasting
```protobuf
service ServiceAssuranceService {
    rpc ForecastVoLTEJitter(ForecastVoLTEJitterRequest) returns (ForecastVoLTEJitterResponse);
    rpc GetQosAnalysis(GetQosAnalysisRequest) returns (GetQosAnalysisResponse);
}
```

## Monitoring

Service assurance components provide comprehensive monitoring:

- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Operational Metrics**: Prediction latency, throughput, error rates
- **Business Metrics**: Issue prevention rate, MTTR reduction, service quality improvement

## Testing

```bash
# Run component tests
cd asa-int && cargo test
cd asa-5g && cargo test
cd asa-qos && cargo test

# Integration tests
cargo test --workspace integration

# Performance benchmarks
cargo bench --workspace
```

## Troubleshooting

### Common Issues

1. **Low Accuracy**: Check training data quality and feature engineering
2. **High Latency**: Optimize model complexity and feature extraction
3. **Memory Usage**: Tune model parameters and batch sizes
4. **Convergence Issues**: Adjust learning rates and regularization

### Logs

Service logs are available at:
- `logs/asa-int.log` - Interference classifier logs
- `logs/asa-5g.log` - ENDC predictor logs
- `logs/asa-qos.log` - VoLTE forecaster logs

## Contributing

When contributing to service assurance components:

1. Maintain accuracy requirements (95%, 80%, 10ms respectively)
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Follow performance optimization guidelines
5. Ensure integration compatibility with other platform components

## License

Part of the RAN Intelligence Platform - see main project license.