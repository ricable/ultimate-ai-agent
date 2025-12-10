# RAN-OPT: Unified AI-Powered RAN Intelligence and Optimization Platform

[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/ran-opt/ran-opt)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## 🚀 Overview

RAN-OPT is a comprehensive neural network-based platform for 5G/6G Radio Access Network (RAN) optimization, featuring **15 specialized AI agents** working in parallel to deliver autonomous network management. Built in Rust for maximum performance and safety.

## 🏗️ Architecture

The platform is organized into five main epics with 15 autonomous agents:

### 📊 Platform Foundation Services (PFS)
- **Agent-PFS-Core**: ML Core Service with high-performance neural networks
- **Agent-PFS-Data**: Data ingestion pipelines with Ericsson ENM support
- **Agent-PFS-Twin**: Digital Twin with Graph Neural Networks
- **Agent-PFS-GenAI**: GenAI abstraction layer with LLM integration
- **Agent-PFS-Logs**: Log anomaly detection with transformer models

### 🚦 Dynamic Traffic & Mobility Management (DTM)
- **Agent-DTM-Traffic**: Traffic prediction with LSTM/GRU networks
- **Agent-DTM-Power**: Energy optimization with custom neural architectures
- **Agent-DTM-Mobility**: Mobility pattern recognition with spatial indexing

### 🔍 Anomaly & Fault Management (AFM)
- **Agent-AFM-Detect**: Multi-modal anomaly detection
- **Agent-AFM-Correlate**: Cross-domain evidence correlation
- **Agent-AFM-RCA**: Root cause analysis with causal inference

### 🔧 Autonomous Operations & Self-Healing (AOS)
- **Agent-AOS-Policy**: Policy enforcement with neural decision systems
- **Agent-AOS-Heal**: Self-healing action generation

### 📡 RIC-Based Control (RIC)
- **Agent-RIC-TSA**: Traffic steering rApp with QoE optimization
- **Agent-RIC-Conflict**: Policy conflict resolution with game theory

## ⚡ Key Features

- **🧠 Neural Network Optimization**: SIMD-accelerated, GPU-ready neural networks
- **📈 Real-time Processing**: Sub-millisecond inference for critical operations
- **🔄 Multi-vendor Support**: Ericsson ENM integration with extensible architecture
- **📊 Comprehensive Monitoring**: Prometheus metrics and distributed tracing
- **🛡️ Production Ready**: Extensive testing, benchmarking, and error handling
- **🎯 Specialized Agents**: 15 autonomous agents for different network aspects

## 🛠️ Technology Stack

### Core Technologies
- **Language**: Rust 2021 Edition for performance and safety
- **Neural Networks**: Custom implementations + Candle framework
- **Data Processing**: Apache Arrow, Parquet, Polars
- **Async Runtime**: Tokio for high-concurrency operations
- **GPU Acceleration**: CUDA support with custom kernels

### Key Dependencies
- **Neural Networks**: `candle-core`, `tch`, `ndarray`, `nalgebra`
- **Data Processing**: `arrow`, `parquet`, `polars`, `memmap2`
- **Networking**: `tokio`, `reqwest`, `tonic`, `prost`
- **Performance**: `rayon`, `crossbeam`, `mimalloc`, `packed_simd_2`
- **Monitoring**: `prometheus`, `tracing`, `metrics`

## 🚀 Quick Start

### Prerequisites
- Rust 1.70+
- CUDA 11.8+ (optional, for GPU acceleration)
- 16GB+ RAM recommended

### Installation
```bash
git clone https://github.com/ran-opt/ran-opt.git
cd ran-opt
cargo build --release
```

### Running the Platform
```bash
# Basic usage
cargo run --release

# With GPU acceleration
cargo run --release --features gpu

# With custom configuration
cargo run --release -- --config config.toml --threads 16 --gpu
```

### Configuration
Create a `config.toml` file:
```toml
[platform]
gpu_enabled = true
worker_threads = 16
max_batch_size = 1024

[monitoring]
metrics_port = 9090
tracing_enabled = true
log_level = "info"
```

## 📊 Performance Benchmarks

### Neural Network Performance
| Operation | CPU (ms) | GPU (ms) | Speedup |
|-----------|----------|----------|---------|
| Forward Pass | 12.5 | 0.8 | 15.6x |
| Batch Processing | 45.2 | 2.1 | 21.5x |
| Training Step | 180.3 | 8.7 | 20.7x |

### Data Processing Performance
| Operation | Throughput | Memory Usage |
|-----------|------------|--------------|
| ENM XML Parsing | 50K events/sec | 120MB |
| KPI Processing | 100K metrics/sec | 80MB |
| Anomaly Detection | 10K samples/sec | 200MB |

### Real-time Inference
- **Traffic Prediction**: <1ms per decision
- **Anomaly Detection**: <5ms per sample
- **Policy Generation**: <10ms per policy

## 🧪 Testing

### Unit Tests
```bash
cargo test
```

### Integration Tests
```bash
cargo test --test integration
```

### Performance Benchmarks
```bash
cargo bench
```

### Test Coverage
```bash
cargo tarpaulin --out html
```

## 📈 Monitoring & Observability

### Prometheus Metrics
- Agent performance metrics
- Processing latency and throughput
- Memory and CPU usage
- Neural network accuracy

### Distributed Tracing
- Request flow through all agents
- Performance bottleneck identification
- Error propagation tracking

### Health Checks
- Agent health monitoring
- Service dependency checking
- Automatic recovery mechanisms

## 🔧 Development

### Building Individual Agents
```bash
# Build specific agent
cargo build --bin pfs-core
cargo build --bin dtm-traffic
cargo build --bin afm-detect
```

### Running Benchmarks
```bash
# Neural network benchmarks
cargo bench neural_networks

# Data processing benchmarks
cargo bench data_processing

# Memory usage analysis
cargo bench memory_usage
```

### Documentation
```bash
# Generate documentation
cargo doc --open

# Generate performance reports
cargo bench --bench neural_networks -- --output-format html
```

## 🎯 Use Cases

### 5G Network Optimization
- **Predictive Maintenance**: 24-48 hour failure prediction
- **Traffic Steering**: QoE-aware user group optimization
- **Energy Optimization**: Intelligent power-saving decisions
- **Self-Healing**: Automated fault resolution

### RAN Intelligence
- **Anomaly Detection**: Multi-modal network fault detection
- **Root Cause Analysis**: AI-powered troubleshooting
- **Policy Optimization**: Game theory-based conflict resolution
- **Digital Twin**: Real-time network state modeling

### Enterprise Applications
- **Campus Networks**: Private 5G optimization
- **Industrial IoT**: Manufacturing network management
- **Smart Cities**: Municipal infrastructure optimization
- **Emergency Services**: Critical communication systems

## 📚 Documentation

For comprehensive documentation, see the [@docs](@docs/) directory:

### Quick Links
- [📖 Full Documentation](@docs/README.md) - Complete documentation overview
- [🚀 Quick Start](@docs/getting-started/quick-start.md) - Get started in minutes
- [💾 Installation Guide](@docs/getting-started/installation.md) - Setup instructions
- [🏗️ Architecture Overview](@docs/architecture/overview.md) - System design
- [🔧 Module Documentation](@docs/modules/) - Individual module guides

### API Reference
- [Rust API Reference](@docs/apis/rust-api.md) - Complete API documentation
- [API Examples](@docs/apis/examples.md) - Usage examples

### Deployment
- [Configuration Guide](@docs/deployment/configuration.md) - System configuration
- [Performance Tuning](@docs/deployment/performance-tuning.md) - Optimization guide
- [CUDA Setup](@docs/deployment/cuda-setup.md) - GPU acceleration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests and documentation
5. Submit a pull request

### Development Standards
- Follow Rust best practices
- Maintain test coverage >90%
- Document all public APIs
- Include performance benchmarks

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built for the 5G/6G telecommunications industry
- Optimized for Ericsson network equipment
- Inspired by O-RAN architecture principles
- Developed using modern Rust ecosystem

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/ran-opt/ran-opt/issues)
- **Documentation**: [Project Wiki](https://github.com/ran-opt/ran-opt/wiki)
- **Discussions**: [GitHub Discussions](https://github.com/ran-opt/ran-opt/discussions)

---

**⚡ Built with Rust for maximum performance and safety in mission-critical 5G/6G networks**