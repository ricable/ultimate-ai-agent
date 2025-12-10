# Telecom Performance Data Preprocessing Setup

## Overview

This preprocessing pipeline handles telecom performance data for neural network training, supporting:

- **54K lines CSV data** with 93 telecom performance features
- **1M+ JSON training samples** with numerical feature arrays
- **Advanced feature engineering** for network quality, traffic, latency, handover metrics
- **GPU-optimized data loading** with PyTorch DataLoaders
- **Comprehensive statistical analysis** and visualization

## Quick Start

### 1. Install Dependencies

```bash
# Using pip
pip install -r requirements_preprocessing.txt

# Or using uv (recommended)
uv pip install -r requirements_preprocessing.txt
```

### 2. Verify Data Structure

Ensure your data directory contains:
```
data/pm/
├── fanndata.csv      # 54K lines, 93 telecom features
├── train.json        # 1M+ training samples
└── test.json         # 260K+ test samples
```

### 3. Run Basic Test

```bash
python3 test_preprocessing.py
```

### 4. Run Full Preprocessing Pipeline

```bash
# Basic execution
python3 src/preprocessing/run_preprocessing.py

# With custom configuration
python3 src/preprocessing/run_preprocessing.py --config config.json

# Custom batch size and output directory
python3 src/preprocessing/run_preprocessing.py --batch-size 2048 --output-dir data/custom_processed
```

## Data Pipeline Components

### 1. Data Pipeline (`data_pipeline.py`)
- **CSV Processing**: Handles 93-feature telecom performance data
- **JSON Processing**: Processes large-scale numerical feature arrays
- **Missing Value Handling**: KNN imputation, mean/median strategies
- **Feature Scaling**: Standard, Robust, MinMax normalization
- **Time Series Features**: Cyclical encoding, business hours indicators

### 2. Feature Engineering (`feature_engineering.py`)
- **Network Quality Features**: SINR, RSSI composite scores
- **Traffic Features**: Volume ratios, per-user metrics, VoLTE performance
- **Latency Features**: Multi-dimensional latency analysis
- **Handover Features**: Success rates, attempt patterns
- **Band Features**: LTE band performance indicators
- **Statistical Features**: Mean, variance, range across feature groups
- **Interaction Features**: Polynomial feature combinations
- **Clustering Features**: KMeans-based user behavior clustering
- **PCA Features**: Dimensionality reduction components
- **Anomaly Features**: Isolation Forest outlier detection

### 3. Data Loading (`data_loader.py`)
- **GPU-Optimized Loading**: CUDA memory management, pin memory
- **Multiple Dataset Types**: Standard, Time Series, Multi-Modal
- **Efficient Batching**: Configurable batch sizes, workers
- **Mixed Precision Support**: FP16/FP32 optimization
- **Memory Management**: Caching, prefetching strategies

### 4. Statistical Analysis (`data_statistics.py`)
- **Comprehensive Statistics**: Basic stats, correlations, distributions
- **Telecom Metrics Analysis**: KPI-specific analysis
- **Static Visualizations**: Matplotlib/Seaborn plots
- **Interactive Visualizations**: Plotly dashboards
- **Quality Assessment**: Missing values, outliers, data quality issues

## Configuration

### Default Configuration
```json
{
  "data": {
    "data_dir": "data/pm",
    "output_dir": "data/processed",
    "normalize_method": "standard",
    "handle_missing": "knn",
    "feature_selection": true,
    "correlation_threshold": 0.95
  },
  "loader": {
    "batch_size": 1024,
    "num_workers": 4,
    "train_split": 0.8,
    "val_split": 0.1,
    "test_split": 0.1
  },
  "feature_engineering": {
    "apply_all": true,
    "max_interaction_features": 50,
    "n_clusters": 5,
    "pca_components": 10
  }
}
```

### Custom Configuration
```bash
# Create custom config
cat > custom_config.json << EOF
{
  "data": {
    "batch_size": 2048,
    "normalize_method": "robust",
    "handle_missing": "median"
  },
  "create_visualizations": false,
  "generate_report": true
}
EOF

# Run with custom config
python3 src/preprocessing/run_preprocessing.py --config custom_config.json
```

## Output Structure

After processing, you'll have:

```
data/processed/
├── X_train.npy           # Processed training features
├── y_train.npy           # Training targets
├── X_test.npy            # Test features
├── y_test.npy            # Test targets
├── scalers.pkl           # Fitted scalers
├── encoders.pkl          # Fitted encoders
├── statistics.json       # Data statistics
├── feature_names.json    # Feature names mapping
└── data_loader_info.json # DataLoader configuration

data/analysis/
├── plots/                # Static visualizations
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── telecom_metrics_dashboard.png
│   └── time_series_analysis.png
├── interactive/          # Interactive visualizations
│   ├── correlation_heatmap.html
│   ├── scatter_matrix.html
│   └── time_series_dashboard.html
└── comprehensive_analysis_report.json
```

## Key Features

### 1. Telecom-Specific Feature Engineering
- **Network Quality Scores**: Composite SINR/RSSI metrics
- **Traffic Efficiency**: Volume ratios, per-user throughput
- **Service Quality**: VoLTE, latency, drop rate analysis
- **Band Performance**: LTE frequency band optimization
- **Handover Analysis**: Inter/intra frequency patterns

### 2. GPU Optimization
- **Memory Management**: Efficient CUDA memory usage
- **Parallel Loading**: Multi-worker data loading
- **Mixed Precision**: FP16/FP32 for memory efficiency
- **Batch Optimization**: Configurable batch sizes

### 3. Comprehensive Analysis
- **54+ Visualizations**: Distribution, correlation, time series
- **Quality Assessment**: Missing values, outliers, duplicates
- **Performance Metrics**: Loading times, memory usage
- **Interactive Dashboards**: Plotly-based exploration

## Performance Expectations

### Processing Times (Approximate)
- **CSV Loading (54K rows)**: 5-15 seconds
- **JSON Loading (1M samples)**: 30-60 seconds
- **Feature Engineering**: 60-120 seconds
- **Visualization Generation**: 30-90 seconds
- **Total Pipeline**: 3-5 minutes

### Memory Usage
- **CSV Data**: ~400MB loaded, ~800MB processed
- **JSON Data**: ~2GB training, ~500MB test
- **Feature Engineering**: +50-100% memory overhead
- **GPU Loading**: Configurable batch sizing

## Troubleshooting

### Common Issues

1. **Memory Errors**
   ```bash
   # Reduce batch size
   python3 src/preprocessing/run_preprocessing.py --batch-size 512
   
   # Disable visualizations
   python3 src/preprocessing/run_preprocessing.py --no-visualizations
   ```

2. **Missing Dependencies**
   ```bash
   # Install all requirements
   pip install -r requirements_preprocessing.txt
   
   # Check specific modules
   python3 -c "import pandas, numpy, sklearn, torch; print('All dependencies OK')"
   ```

3. **GPU Issues**
   ```bash
   # Force CPU mode
   export CUDA_VISIBLE_DEVICES=""
   python3 src/preprocessing/run_preprocessing.py
   ```

4. **Large Dataset Handling**
   ```bash
   # Use memory mapping
   python3 -c "
   from src.preprocessing.run_preprocessing import *
   config = {'loader': {'use_memory_mapping': True, 'cache_data': False}}
   # Run with config
   "
   ```

## Integration with Neural Networks

The preprocessed data is ready for training with:

### 1. Standard Feed-Forward Networks
```python
from src.preprocessing import create_data_loading_pipeline

# Load preprocessed data
pipeline = create_data_loading_pipeline("data/processed")
train_loader = pipeline['dataloaders']['train']

# Training loop
for batch_features, batch_targets in train_loader:
    # Your training code here
    pass
```

### 2. Time Series Networks
```python
from src.preprocessing import TimeSeriesDataset

# Create time series dataset
ts_dataset = TimeSeriesDataset(features, targets, sequence_length=24)
```

### 3. Multi-Modal Networks
```python
from src.preprocessing import MultiModalDataset

# Combine CSV and JSON data
multimodal_dataset = MultiModalDataset(csv_features, json_features)
```

## Advanced Usage

### Custom Feature Engineering
```python
from src.preprocessing import TelecomFeatureEngineering

# Custom feature engineering
fe = TelecomFeatureEngineering()
df = fe.create_network_quality_features(df)
df = fe.create_custom_features(df)  # Your custom method
```

### Streaming Data Processing
```python
# For real-time data streams
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor(config)
for data_chunk in data_stream:
    processed_chunk = preprocessor.transform_new_data(data_chunk)
    # Send to model
```

### Distributed Processing
```python
# For large-scale processing
import torch.distributed as dist
from src.preprocessing import GPUOptimizedDataLoader

# Multi-GPU data loading
gpu_loader = GPUOptimizedDataLoader(config)
distributed_loader = gpu_loader.create_distributed_loader(dataset)
```

## Contributing

To extend the preprocessing pipeline:

1. **Add new feature engineering methods** in `feature_engineering.py`
2. **Add new data sources** in `data_pipeline.py`
3. **Add new visualizations** in `data_statistics.py`
4. **Add new dataset types** in `data_loader.py`

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the comprehensive analysis report in `data/analysis/`
3. Examine the processing logs for detailed error information
4. Test with the basic test script: `python3 test_preprocessing.py`