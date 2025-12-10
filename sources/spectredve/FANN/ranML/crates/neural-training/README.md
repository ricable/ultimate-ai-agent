# Neural Network Training System

A comprehensive swarm-based neural network training system for telecom data analysis using Rust and WASM.

## Features

### 🧠 Multiple Neural Network Architectures
- **Shallow Network**: Simple 3-layer architecture for baseline performance
- **Deep Network**: 6-layer deep architecture with multiple activation functions
- **Wide Network**: Broader hidden layers for complex feature extraction
- **Residual-like Network**: Skip connections inspired by ResNet
- **Bottleneck Network**: Encoder-decoder architecture for feature compression

### 🐝 Swarm-Based Training Coordination
- **5 Specialized Agents**:
  - Data Processing Expert
  - Neural Architecture Designer
  - Training Implementation Specialist
  - Activation Function Researcher
  - Evaluation Metrics Specialist
- **Parallel Training**: Train multiple models simultaneously
- **Load Balancing**: Dynamic task distribution across agents
- **Fault Tolerance**: Automatic recovery and retry mechanisms

### 📊 Enhanced Model Evaluation
- **Comprehensive Metrics**: Extended evaluation including AIC, BIC, directional accuracy, and prediction intervals
- **Advanced Cross-Validation**: Enhanced k-fold with statistical significance testing and confidence intervals
- **Statistical Analysis**: Comprehensive hypothesis testing suite (normality, homoscedasticity, independence)
- **Residual Diagnostics**: Advanced diagnostic tools for model assumption validation
- **Performance Benchmarking**: Scalability analysis, memory profiling, and bottleneck identification
- **Model Comparison**: Systematic ranking with trade-off analysis and optimization recommendations

### ⚡ Advanced Features
- **WASM Compatibility**: Run in browsers and edge devices
- **GPU Acceleration**: Optional Candle backend for GPU training
- **Hyperparameter Tuning**: Automated grid search
- **Real-time Monitoring**: Training progress and swarm coordination metrics
- **Configurable Pipeline**: JSON/YAML configuration files

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/ruvnet/ruv-FANN.git
cd ruv-FANN/ranML/crates/neural-training

# Build the project
cargo build --release
```

### Basic Usage

1. **Preprocess Data**:
```bash
cargo run --bin neural-trainer preprocess \
  --input ../../data/pm/fanndata.csv \
  --output processed_data \
  --split-ratio 0.8 \
  --normalize
```

2. **Train Models**:
```bash
cargo run --bin neural-trainer train \
  --data ../../data/pm/fanndata.csv \
  --output results \
  --max-epochs 1000 \
  --learning-rate 0.01
```

3. **Hyperparameter Tuning**:
```bash
cargo run --bin neural-trainer tune \
  --data ../../data/pm/fanndata.csv \
  --architecture deep \
  --max-combinations 50
```

4. **Evaluate Models**:
```bash
# Basic evaluation
cargo run --bin evaluation_demo -- \
  --data ../../data/pm/fanndata.csv \
  --models 5 \
  --cv-folds 10 \
  --confidence 0.95

# Comprehensive evaluation with benchmarking
cargo run --bin evaluation_demo -- \
  --data ../../data/pm/fanndata.csv \
  --comprehensive \
  --parallel \
  --benchmark
```

### Configuration

Generate a default configuration file:
```bash
cargo run --bin neural-trainer generate-config \
  --output training_config.json \
  --format json
```

Then customize the configuration:
```json
{
  "data": {
    "input_file": "data/pm/fanndata.csv",
    "train_test_split": 0.8,
    "normalize_features": true,
    "target_column": "cell_availability"
  },
  "models": {
    "architectures": [
      {
        "name": "shallow",
        "layer_sizes": [22, 32, 1],
        "activations": ["linear", "relu", "linear"],
        "enabled": true
      }
    ]
  },
  "training": {
    "learning_rate": 0.01,
    "momentum": 0.9,
    "max_epochs": 1000,
    "target_error": 0.001
  },
  "swarm": {
    "enabled": true,
    "max_parallel_agents": 5
  }
}
```

## Architecture

### Data Flow
```
Data Input → Preprocessing → Architecture Design → Parallel Training → Evaluation → Results
     ↓              ↓              ↓                    ↓              ↓
CSV Files → Normalization → Network Creation → Swarm Coordination → Metrics → Reports
```

### Swarm Coordination
```
┌─────────────────────────────────────────────┐
│          Swarm Orchestrator                 │
├─────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐          │
│  │ Data Agent  │  │ Arch Agent  │          │
│  └─────────────┘  └─────────────┘          │
│  ┌─────────────┐  ┌─────────────┐          │
│  │Train Agent  │  │ Eval Agent  │          │
│  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────┤
│        Neural Network Models               │
│   ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐    │
│   │ MLP  │ │ Deep │ │ Wide │ │ResNet│    │
│   └──────┘ └──────┘ └──────┘ └──────┘    │
└─────────────────────────────────────────────┘
```

## Data Format

The system expects telecom data in CSV format with columns including:
- `cell_availability_%`: Target variable
- `volte_traffic`: VoLTE traffic metrics
- `sinr_pusch_avg`: Signal quality indicators
- `ul_volume_gb`, `dl_volume_gb`: Data volume metrics
- And many more telecom KPIs...

## Model Architectures

### Shallow Network
- **Layers**: [Input, Hidden(32), Output]
- **Activations**: [Linear, ReLU, Linear]
- **Use Case**: Baseline performance, fast training

### Deep Network
- **Layers**: [Input, 64, 32, 16, 8, Output]
- **Activations**: [Linear, ReLU, ReLU, ReLU, Sigmoid, Linear]
- **Use Case**: Complex pattern recognition

### Wide Network
- **Layers**: [Input, Hidden(128), Hidden(64), Output]
- **Activations**: [Linear, ReLU, ReLU, Linear]
- **Use Case**: Feature-rich datasets

### Residual-like Network
- **Layers**: [Input, 64, 64, 32, Output]
- **Activations**: [Linear, ReLU, LeakyReLU, Tanh, Linear]
- **Use Case**: Deep learning with gradient flow

### Bottleneck Network
- **Layers**: [Input, 16, 8, 16, Output]
- **Activations**: [Linear, Tanh, Sigmoid, ReLU, Linear]
- **Use Case**: Feature compression and reconstruction

## Enhanced Evaluation System

The system provides a comprehensive evaluation framework with advanced statistical analysis:

### Core Metrics
- **Traditional Metrics**: MSE, MAE, RMSE, R², MAPE, Correlation
- **Enhanced Metrics**: AIC, BIC, Adjusted R², SMAPE, Theil's U
- **Directional Accuracy**: Percentage of correct trend predictions
- **Prediction Intervals**: Coverage probability analysis

### Statistical Analysis
- **Normality Tests**: Shapiro-Wilk, Jarque-Bera, Kolmogorov-Smirnov, Anderson-Darling
- **Heteroscedasticity Tests**: Breusch-Pagan, White test, Goldfeld-Quandt
- **Independence Tests**: Durbin-Watson, Ljung-Box for autocorrelation
- **Hypothesis Testing**: Paired t-tests, Wilcoxon signed-rank tests
- **Confidence Intervals**: Bootstrap and analytical methods

### Performance Benchmarking
- **Scalability Analysis**: Performance across data sizes and model complexity
- **Memory Profiling**: Peak usage, allocations, efficiency analysis
- **CPU Profiling**: Instruction-level analysis, cache performance
- **Throughput Analysis**: Sustained and peak performance measurements
- **Bottleneck Identification**: Automated performance limitation detection

### Model Comparison
- **Multi-Model Ranking**: Systematic comparison across accuracy, speed, efficiency
- **Trade-off Analysis**: Speed vs accuracy, memory vs performance
- **Pareto Optimization**: Identification of optimal configurations
- **Recommendation Engine**: Automated optimization suggestions

### Example Output
```
📊 COMPREHENSIVE EVALUATION RESULTS
════════════════════════════════════
Model: complex_model_2
Evaluation Time: 1.45s

📈 Core Metrics:
  MSE:              0.001234
  RMSE:             0.035128
  MAE:              0.028456
  R²:               0.945678
  Adjusted R²:      0.943210
  Correlation:      0.972345

📉 Error Analysis:
  MAPE:             4.23%
  SMAPE:            4.12%
  Max Error:        0.156789
  Directional Acc:  87.5%

🎯 Model Selection Criteria:
  AIC:              -2456.78
  BIC:              -2398.45
  Theil's U:        0.234567

📊 Cross-Validation:
  CV Score:         0.941234 ± 0.012345
  Best Fold:        #3

💡 Recommendations:
  1. Model shows excellent performance with R² > 0.94
  2. Consider ensemble methods for further improvement
```

## Swarm Agent Capabilities

### Data Processing Expert
- Data cleaning and preprocessing
- Feature normalization and scaling
- Outlier detection and removal
- Train/test splitting

### Neural Architecture Designer
- Network topology design
- Activation function selection
- Layer size optimization
- Architecture comparison

### Training Implementation Specialist
- Backpropagation algorithms
- Gradient optimization
- Convergence monitoring
- Weight updates

### Activation Function Researcher
- Function analysis and selection
- Gradient computation
- Mathematical optimization
- Performance comparison

### Evaluation Metrics Specialist
- Model performance analysis
- Statistical validation
- Cross-validation
- Metric computation

## Advanced Features

### Hyperparameter Tuning
Automated grid search across:
- Learning rates: [0.001, 0.01, 0.1]
- Momentum values: [0.0, 0.5, 0.9]
- Batch sizes: [None, 32, 64]
- Weight decay: [0.0, 0.0001, 0.001]

### Cross-Validation
- K-fold cross-validation
- Stratified sampling
- Statistical significance testing
- Confidence intervals

### Parallel Processing
- Multi-threaded training using Rayon
- Async coordination with Tokio
- WASM-compatible execution
- Resource utilization monitoring

## Output Files

Training produces several output files:
- `training_results.json`: Complete training results
- `model_comparison.json`: Performance comparison
- `training_config.json`: Configuration used
- `predictions.csv`: Model predictions
- `training_history.csv`: Epoch-by-epoch progress

## Integration

### With ruv-FANN Core
```rust
use neural_training::*;
use ruv_fann::Network;

let mut system = NeuralTrainingSystem::new();
let results = system.run_training_pipeline("data.csv").await?;
```

### With ruv-swarm MCP
The system integrates with the ruv-swarm MCP server for Claude Code:
```bash
npx ruv-swarm neural-train --data fanndata.csv --agents 5
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## License

Dual-licensed under MIT or Apache-2.0, same as the ruv-FANN project.

## Citation

If you use this system in your research, please cite:
```bibtex
@software{ruv_fann_neural_training,
  title = {Neural Network Training System for Telecom Data},
  author = {ruv team},
  year = {2025},
  url = {https://github.com/ruvnet/ruv-FANN}
}
```