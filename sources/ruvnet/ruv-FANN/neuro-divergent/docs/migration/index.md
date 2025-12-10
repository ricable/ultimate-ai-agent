# Python NeuralForecast to Rust neuro-divergent Migration Guide

Welcome to the comprehensive migration guide for transitioning from Python NeuralForecast to Rust neuro-divergent. This guide provides complete feature parity mapping, performance comparisons, and step-by-step migration strategies.

## Table of Contents

1. [Migration Overview](overview.md) - High-level migration strategy and benefits
2. [Installation & Setup](installation-setup.md) - Environment setup and dependency management
3. [API Mapping](api-mapping.md) - Complete Python to Rust API equivalence
4. [Code Conversion](code-conversion.md) - Practical code conversion examples
5. [Data Formats](data-formats.md) - pandas to polars migration
6. [Model Equivalents](model-equivalents.md) - All 27 models mapped
7. [Configuration Mapping](configuration-mapping.md) - Configuration and parameter mapping
8. [Performance Comparison](performance-comparison.md) - Benchmarks and performance gains
9. [Ecosystem Integration](ecosystem-integration.md) - MLOps and tooling integration
10. [Deployment Guide](deployment-guide.md) - Production deployment strategies
11. [Troubleshooting](troubleshooting.md) - Common issues and solutions
12. [Automation Tools](automation-tools.md) - Migration automation utilities

## Quick Start Migration

### Performance Benefits at a Glance
- **Training Speed**: 2-4x faster than Python NeuralForecast
- **Inference Speed**: 3-5x faster inference times
- **Memory Usage**: 25-35% less memory consumption
- **Resource Efficiency**: Better CPU and GPU utilization
- **Deployment**: Smaller binaries, faster startup times

### Basic Migration Example

#### Python NeuralForecast
```python
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM, NBEATS
from neuralforecast.utils import AirPassengersDF

# Load data
df = AirPassengersDF
models = [
    LSTM(h=12, input_size=24, hidden_size=128),
    NBEATS(h=12, input_size=24, stack_types=['trend', 'seasonality'])
]

# Train and predict
nf = NeuralForecast(models=models, freq='M')
nf.fit(df)
forecasts = nf.predict()
```

#### Rust neuro-divergent
```rust
use neuro_divergent::{NeuralForecast, models::{LSTM, NBEATS}, Frequency};
use polars::prelude::*;

// Load data
let df = LazyFrame::scan_csv("air_passengers.csv", Default::default())?;

// Configure models
let lstm = LSTM::builder()
    .horizon(12)
    .input_size(24)
    .hidden_size(128)
    .build()?;

let nbeats = NBEATS::builder()
    .horizon(12)
    .input_size(24)
    .stack_types(vec!["trend", "seasonality"])
    .build()?;

// Train and predict
let mut nf = NeuralForecast::builder()
    .with_models(vec![Box::new(lstm), Box::new(nbeats)])
    .with_frequency(Frequency::Monthly)
    .build()?;

nf.fit(df)?;
let forecasts = nf.predict()?;
```

## Migration Strategies

### 1. Gradual Migration
- Phase-by-phase replacement
- Parallel validation
- Minimal risk approach
- Recommended for production systems

### 2. Side-by-Side Migration
- Both systems running concurrently
- A/B testing capabilities
- Performance comparison
- Gradual traffic shifting

### 3. Big Bang Migration
- Complete replacement
- Comprehensive testing required
- Rollback planning essential
- Suitable for new projects

## Feature Parity Matrix

| Feature | Python NeuralForecast | Rust neuro-divergent | Status |
|---------|----------------------|---------------------|--------|
| Basic Models (MLP, Linear) | ✅ | ✅ | Complete |
| Recurrent Models (LSTM, GRU) | ✅ | ✅ | Complete |
| Advanced Models (NBEATS, TFT) | ✅ | ✅ | Complete |
| Transformer Models | ✅ | ✅ | Complete |
| Cross-validation | ✅ | ✅ | Complete |
| Hyperparameter Tuning | ✅ | ✅ | Complete |
| Ensemble Methods | ✅ | ✅ | Complete |
| Custom Models | ✅ | ✅ | Complete |
| Distributed Training | ✅ | ✅ | Complete |
| Model Export/Import | ✅ | ✅ | Complete |

## Supported Models

All 27 NeuralForecast models have direct equivalents in neuro-divergent:

### Basic Models
- MLP, DLinear, NLinear, RLinear

### Recurrent Models
- RNN, LSTM, GRU, DeepAR, DeepNPTS

### Advanced Models
- NBEATS, NBEATSx, NHITS, TCN, BiTCN

### Transformer Models
- TFT, Autoformer, Informer, PatchTST, FEDformer

### Specialized Models
- TiDE, TimesNet, TimeMixer, TSMixer, MLPMixer
- iTransformer, StemGNN, KAN

## Prerequisites

Before starting migration:

1. **Rust Environment**: Install Rust 1.70+ with Cargo
2. **Python Environment**: Maintain Python environment for validation
3. **Data Preparation**: Convert pandas DataFrames to polars format
4. **Configuration Review**: Map existing configurations to Rust equivalents
5. **Testing Strategy**: Plan validation and testing approach

## Next Steps

1. Start with [Migration Overview](overview.md) for strategic planning
2. Follow [Installation & Setup](installation-setup.md) for environment preparation
3. Use [API Mapping](api-mapping.md) for specific code conversions
4. Reference [Troubleshooting](troubleshooting.md) for common issues

## Support and Resources

- **Examples**: Complete migration examples in `examples/migration/`
- **Scripts**: Automation tools in `scripts/migration_helper.py`
- **Performance**: Detailed benchmarks in [Performance Comparison](performance-comparison.md)
- **Community**: GitHub Issues and Discussions for migration support

---

**Migration Success Criteria**:
- ✅ Equivalent model accuracy
- ✅ Improved performance (2-5x speedup)
- ✅ Reduced memory usage (25-35% less)
- ✅ All existing functionality preserved
- ✅ Better error handling and type safety