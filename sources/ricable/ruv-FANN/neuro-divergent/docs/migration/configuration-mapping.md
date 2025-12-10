# Configuration Mapping Guide: Python to Rust Configuration Migration

This guide covers the complete migration of configuration files, environment variables, and parameter settings from Python NeuralForecast to Rust neuro-divergent.

## Table of Contents

1. [Configuration File Formats](#configuration-file-formats)
2. [Environment Variables](#environment-variables)
3. [Model Configuration](#model-configuration)
4. [Training Configuration](#training-configuration)
5. [Data Configuration](#data-configuration)
6. [Logging and Monitoring](#logging-and-monitoring)
7. [Performance Tuning](#performance-tuning)
8. [Deployment Configuration](#deployment-configuration)

## Configuration File Formats

### Python Configuration (YAML/JSON)

**Python config.yaml**:
```yaml
# Python NeuralForecast Configuration
data:
  path: "./data/timeseries.csv"
  freq: "D"
  static_features: ["category", "region"]
  future_features: ["holidays", "weather"]
  
models:
  LSTM:
    h: 12
    input_size: 24
    hidden_size: 128
    num_layers: 2
    dropout: 0.1
    learning_rate: 0.001
    max_steps: 1000
    batch_size: 32
    
  NBEATS:
    h: 12
    input_size: 24
    stack_types: ["trend", "seasonality"]
    n_blocks: [3, 3]
    
training:
  early_stopping:
    patience: 50
    min_delta: 0.001
  validation:
    split: 0.2
    shuffle: false
    
logging:
  level: "INFO"
  file: "./logs/training.log"
```

**Rust config.toml**:
```toml
# Rust neuro-divergent Configuration
[data]
path = "./data/timeseries.csv"
frequency = "Daily"
static_features = ["category", "region"]
future_features = ["holidays", "weather"]

[models.lstm]
horizon = 12
input_size = 24
hidden_size = 128
num_layers = 2
dropout = 0.1
learning_rate = 0.001
max_steps = 1000
batch_size = 32

[models.nbeats]
horizon = 12
input_size = 24
stack_types = ["trend", "seasonality"]
n_blocks = [3, 3]

[training]
early_stopping_patience = 50
early_stopping_min_delta = 0.001
validation_split = 0.2
validation_shuffle = false

[logging]
level = "info"
file = "./logs/training.log"
```

### Configuration Loading Code

**Python Configuration Loading**:
```python
import yaml
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ModelConfig:
    h: int
    input_size: int
    hidden_size: int
    learning_rate: float = 0.001
    max_steps: int = 1000

@dataclass
class Config:
    data_path: str
    freq: str
    models: Dict[str, ModelConfig]
    
def load_config(config_path: str) -> Config:
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    models = {}
    for model_name, model_params in config_dict['models'].items():
        models[model_name] = ModelConfig(**model_params)
    
    return Config(
        data_path=config_dict['data']['path'],
        freq=config_dict['data']['freq'],
        models=models
    )

# Usage
config = load_config('config.yaml')
```

**Rust Configuration Loading**:
```rust
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use anyhow::Result;

#[derive(Debug, Deserialize, Serialize)]
pub struct DataConfig {
    pub path: String,
    pub frequency: String,
    pub static_features: Option<Vec<String>>,
    pub future_features: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelConfig {
    pub horizon: usize,
    pub input_size: usize,
    pub hidden_size: usize,
    pub learning_rate: f64,
    pub max_steps: usize,
    pub dropout: Option<f64>,
    pub num_layers: Option<usize>,
    pub batch_size: Option<usize>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TrainingConfig {
    pub early_stopping_patience: Option<usize>,
    pub early_stopping_min_delta: Option<f64>,
    pub validation_split: Option<f64>,
    pub validation_shuffle: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    pub data: DataConfig,
    pub models: HashMap<String, ModelConfig>,
    pub training: Option<TrainingConfig>,
    pub logging: Option<LoggingConfig>,
}

impl Config {
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
    
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

// Usage
let config = Config::from_file("config.toml")?;
```

## Environment Variables

### Python Environment Variables

**Python .env**:
```bash
# Data paths
NEURALFORECAST_DATA_PATH=./data
NEURALFORECAST_MODEL_PATH=./models
NEURALFORECAST_LOG_PATH=./logs

# Training settings
NEURALFORECAST_MAX_EPOCHS=1000
NEURALFORECAST_BATCH_SIZE=32
NEURALFORECAST_LEARNING_RATE=0.001

# Performance settings
NEURALFORECAST_NUM_WORKERS=4
NEURALFORECAST_CUDA_VISIBLE_DEVICES=0

# Logging
NEURALFORECAST_LOG_LEVEL=INFO
NEURALFORECAST_VERBOSE=true
```

**Rust Environment Variables**:
```bash
# Data paths
NEURO_DIVERGENT_DATA_PATH=./data
NEURO_DIVERGENT_MODEL_PATH=./models
NEURO_DIVERGENT_LOG_PATH=./logs

# Training settings  
NEURO_DIVERGENT_MAX_EPOCHS=1000
NEURO_DIVERGENT_BATCH_SIZE=32
NEURO_DIVERGENT_LEARNING_RATE=0.001

# Performance settings
NEURO_DIVERGENT_NUM_THREADS=4
NEURO_DIVERGENT_GPU_DEVICE=0

# Logging
RUST_LOG=info
NEURO_DIVERGENT_VERBOSE=true
```

### Environment Variable Loading

**Python Environment Loading**:
```python
import os
from dotenv import load_dotenv

load_dotenv()

class EnvConfig:
    DATA_PATH = os.getenv('NEURALFORECAST_DATA_PATH', './data')
    MODEL_PATH = os.getenv('NEURALFORECAST_MODEL_PATH', './models')
    LOG_PATH = os.getenv('NEURALFORECAST_LOG_PATH', './logs')
    
    MAX_EPOCHS = int(os.getenv('NEURALFORECAST_MAX_EPOCHS', '1000'))
    BATCH_SIZE = int(os.getenv('NEURALFORECAST_BATCH_SIZE', '32'))
    LEARNING_RATE = float(os.getenv('NEURALFORECAST_LEARNING_RATE', '0.001'))
    
    NUM_WORKERS = int(os.getenv('NEURALFORECAST_NUM_WORKERS', '4'))
    CUDA_DEVICE = os.getenv('NEURALFORECAST_CUDA_VISIBLE_DEVICES', '0')
    
    LOG_LEVEL = os.getenv('NEURALFORECAST_LOG_LEVEL', 'INFO')
    VERBOSE = os.getenv('NEURALFORECAST_VERBOSE', 'false').lower() == 'true'
```

**Rust Environment Loading**:
```rust
use std::env;
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct EnvConfig {
    pub data_path: PathBuf,
    pub model_path: PathBuf,
    pub log_path: PathBuf,
    
    pub max_epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    
    pub num_threads: usize,
    pub gpu_device: usize,
    
    pub log_level: String,
    pub verbose: bool,
}

impl EnvConfig {
    pub fn from_env() -> Self {
        Self {
            data_path: env::var("NEURO_DIVERGENT_DATA_PATH")
                .unwrap_or_else(|_| "./data".to_string())
                .into(),
            model_path: env::var("NEURO_DIVERGENT_MODEL_PATH")
                .unwrap_or_else(|_| "./models".to_string())
                .into(),
            log_path: env::var("NEURO_DIVERGENT_LOG_PATH")
                .unwrap_or_else(|_| "./logs".to_string())
                .into(),
                
            max_epochs: env::var("NEURO_DIVERGENT_MAX_EPOCHS")
                .unwrap_or_else(|_| "1000".to_string())
                .parse()
                .unwrap_or(1000),
            batch_size: env::var("NEURO_DIVERGENT_BATCH_SIZE")
                .unwrap_or_else(|_| "32".to_string())
                .parse()
                .unwrap_or(32),
            learning_rate: env::var("NEURO_DIVERGENT_LEARNING_RATE")
                .unwrap_or_else(|_| "0.001".to_string())
                .parse()
                .unwrap_or(0.001),
                
            num_threads: env::var("NEURO_DIVERGENT_NUM_THREADS")
                .unwrap_or_else(|_| "4".to_string())
                .parse()
                .unwrap_or(4),
            gpu_device: env::var("NEURO_DIVERGENT_GPU_DEVICE")
                .unwrap_or_else(|_| "0".to_string())
                .parse()
                .unwrap_or(0),
                
            log_level: env::var("RUST_LOG")
                .unwrap_or_else(|_| "info".to_string()),
            verbose: env::var("NEURO_DIVERGENT_VERBOSE")
                .unwrap_or_else(|_| "false".to_string())
                .parse()
                .unwrap_or(false),
        }
    }
}
```

## Model Configuration

### Parameter Mapping

| Python Parameter | Rust Parameter | Type | Default | Notes |
|------------------|----------------|------|---------|-------|
| `h` | `horizon` | `usize` | 1 | Forecast horizon |
| `input_size` | `input_size` | `usize` | 1 | Input sequence length |
| `hidden_size` | `hidden_size` | `usize` | 128 | Hidden layer size |
| `num_layers` | `num_layers` | `usize` | 1 | Number of layers |
| `dropout` | `dropout` | `f64` | 0.0 | Dropout rate |
| `learning_rate` | `learning_rate` | `f64` | 0.001 | Learning rate |
| `max_steps` | `max_steps` | `usize` | 1000 | Training steps |
| `batch_size` | `batch_size` | `usize` | 32 | Batch size |
| `random_state` | `random_seed` | `Option<u64>` | None | Random seed |

### Model-Specific Configuration

**Python LSTM Config**:
```python
lstm_config = {
    'h': 12,
    'input_size': 24,
    'hidden_size': 128,
    'num_layers': 2,
    'dropout': 0.1,
    'bidirectional': False,
    'learning_rate': 0.001,
    'max_steps': 1000,
    'batch_size': 32,
    'early_stop_patience_steps': 50,
    'val_check_steps': 10,
    'random_state': 42
}
```

**Rust LSTM Config**:
```rust
use neuro_divergent::models::lstm::LSTMConfig;
use neuro_divergent::training::TrainingConfig;

let lstm_config = LSTMConfig {
    horizon: 12,
    input_size: 24,
    hidden_size: 128,
    num_layers: 2,
    dropout: 0.1,
    bidirectional: false,
    random_seed: Some(42),
};

let training_config = TrainingConfig {
    learning_rate: 0.001,
    max_steps: 1000,
    batch_size: 32,
    early_stopping: Some(EarlyStoppingConfig {
        patience: 50,
        validation_check_steps: 10,
        ..Default::default()
    }),
    ..Default::default()
};
```

## Training Configuration

### Training Parameters

**Python Training Config**:
```python
training_config = {
    'max_steps': 1000,
    'learning_rate': 0.001,
    'batch_size': 32,
    'val_size': 0.2,
    'test_size': 0.1,
    'early_stop_patience_steps': 50,
    'val_check_steps': 10,
    'step_size': 1,
    'num_lr_decays': 3,
    'scaler_type': 'robust',
    'random_state': 42,
    'verbose': True
}
```

**Rust Training Config**:
```rust
use neuro_divergent::training::{TrainingConfig, EarlyStoppingConfig, ScalerType};

let training_config = TrainingConfig {
    max_steps: 1000,
    learning_rate: 0.001,
    batch_size: 32,
    validation_split: Some(0.2),
    test_split: Some(0.1),
    early_stopping: Some(EarlyStoppingConfig {
        patience: 50,
        validation_check_steps: 10,
        min_delta: 0.0001,
        ..Default::default()
    }),
    step_size: 1,
    num_lr_decays: 3,
    scaler_type: ScalerType::Robust,
    random_seed: Some(42),
    verbose: true,
    ..Default::default()
};
```

### Optimizer Configuration

**Python Optimizer Config**:
```python
optimizer_config = {
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'betas': [0.9, 0.999],
    'eps': 1e-8,
    'lr_scheduler': {
        'type': 'StepLR',
        'step_size': 100,
        'gamma': 0.5
    }
}
```

**Rust Optimizer Config**:
```rust
use neuro_divergent::training::{OptimizerConfig, OptimizerType, SchedulerConfig, SchedulerType};

let optimizer_config = OptimizerConfig {
    optimizer_type: OptimizerType::Adam,
    learning_rate: 0.001,
    weight_decay: Some(0.0001),
    momentum: Some(0.9),
    beta1: Some(0.9),
    beta2: Some(0.999),
    epsilon: Some(1e-8),
    scheduler: Some(SchedulerConfig {
        scheduler_type: SchedulerType::StepLR,
        step_size: Some(100),
        gamma: Some(0.5),
        ..Default::default()
    }),
    ..Default::default()
};
```

## Data Configuration

### Data Pipeline Configuration

**Python Data Config**:
```python
data_config = {
    'data_path': './data/timeseries.csv',
    'freq': 'D',
    'date_col': 'ds',
    'target_col': 'y',
    'unique_id_col': 'unique_id',
    'static_features': ['category', 'region'],
    'dynamic_features': ['price', 'promotion'],
    'future_features': ['holidays', 'weather'],
    'scaler_type': 'standard',
    'fill_strategy': 'forward',
    'outlier_detection': True,
    'outlier_threshold': 3.0
}
```

**Rust Data Config**:
```rust
use neuro_divergent::data::{DataConfig, ScalerType, FillStrategy};

let data_config = DataConfig {
    data_path: "./data/timeseries.csv".into(),
    frequency: Frequency::Daily,
    date_column: "ds".to_string(),
    target_column: "y".to_string(),
    unique_id_column: "unique_id".to_string(),
    static_features: Some(vec!["category".to_string(), "region".to_string()]),
    dynamic_features: Some(vec!["price".to_string(), "promotion".to_string()]),
    future_features: Some(vec!["holidays".to_string(), "weather".to_string()]),
    scaler_type: ScalerType::Standard,
    fill_strategy: FillStrategy::Forward,
    outlier_detection: true,
    outlier_threshold: 3.0,
    ..Default::default()
};
```

## Logging and Monitoring

### Logging Configuration

**Python Logging**:
```python
import logging

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': './logs/training.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
```

**Rust Logging**:
```rust
use tracing::{info, debug, warn, error};
use tracing_subscriber::{fmt, EnvFilter};

// Initialize logging
pub fn init_logging(level: &str, log_file: Option<&str>) -> Result<()> {
    let filter = EnvFilter::new(level);
    
    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_thread_ids(true)
        .with_target(true);
    
    if let Some(file_path) = log_file {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)?;
        
        subscriber
            .with_writer(file)
            .init();
    } else {
        subscriber.init();
    }
    
    Ok(())
}

// Usage
init_logging("info", Some("./logs/training.log"))?;
info!("Training started");
```

## Performance Tuning

### Performance Configuration

**Python Performance Config**:
```python
performance_config = {
    'num_workers': 4,
    'prefetch_factor': 2,
    'pin_memory': True,
    'persistent_workers': True,
    'multiprocessing_context': 'spawn',
    'cuda_device': 0,
    'mixed_precision': True,
    'compile_model': True,
    'memory_efficient': True
}
```

**Rust Performance Config**:
```rust
use neuro_divergent::config::PerformanceConfig;

let performance_config = PerformanceConfig {
    num_threads: 4,
    prefetch_batches: 2,
    pin_memory: true,
    persistent_workers: true,
    gpu_device: Some(0),
    mixed_precision: true,
    compile_optimizations: true,
    memory_efficient: true,
    parallel_data_loading: true,
    ..Default::default()
};

// Set global thread pool
rayon::ThreadPoolBuilder::new()
    .num_threads(performance_config.num_threads)
    .build_global()?;

// Set polars threads
std::env::set_var("POLARS_MAX_THREADS", performance_config.num_threads.to_string());
```

## Deployment Configuration

### Production Configuration

**Python Production Config**:
```python
production_config = {
    'model_path': '/app/models',
    'data_path': '/app/data',
    'log_path': '/app/logs',
    'api_port': 8000,
    'api_host': '0.0.0.0',
    'workers': 4,
    'timeout': 30,
    'keep_alive': 2,
    'max_requests': 1000,
    'memory_limit': '2G',
    'cpu_limit': '1.0',
    'health_check_interval': 30,
    'metrics_enabled': True,
    'debug': False
}
```

**Rust Production Config**:
```rust
use neuro_divergent::config::DeploymentConfig;
use std::time::Duration;

let deployment_config = DeploymentConfig {
    model_path: "/app/models".into(),
    data_path: "/app/data".into(),
    log_path: "/app/logs".into(),
    api_port: 8000,
    api_host: "0.0.0.0".to_string(),
    workers: 4,
    timeout: Duration::from_secs(30),
    keep_alive: Duration::from_secs(2),
    max_requests: 1000,
    memory_limit_mb: 2048,
    cpu_limit: 1.0,
    health_check_interval: Duration::from_secs(30),
    metrics_enabled: true,
    debug: false,
    ..Default::default()
};
```

### Configuration Validation

```rust
use neuro_divergent::config::{Config, ValidationError};

impl Config {
    pub fn validate(&self) -> Result<(), ValidationError> {
        // Validate required fields
        if self.data.path.is_empty() {
            return Err(ValidationError::MissingField("data.path".to_string()));
        }
        
        // Validate model configurations
        for (model_name, model_config) in &self.models {
            if model_config.horizon == 0 {
                return Err(ValidationError::InvalidValue(
                    format!("models.{}.horizon must be > 0", model_name)
                ));
            }
            
            if model_config.input_size == 0 {
                return Err(ValidationError::InvalidValue(
                    format!("models.{}.input_size must be > 0", model_name)
                ));
            }
            
            if model_config.learning_rate <= 0.0 || model_config.learning_rate > 1.0 {
                return Err(ValidationError::InvalidValue(
                    format!("models.{}.learning_rate must be in (0, 1]", model_name)
                ));
            }
        }
        
        // Validate training configuration
        if let Some(training) = &self.training {
            if let Some(split) = training.validation_split {
                if split <= 0.0 || split >= 1.0 {
                    return Err(ValidationError::InvalidValue(
                        "training.validation_split must be in (0, 1)".to_string()
                    ));
                }
            }
        }
        
        Ok(())
    }
}
```

---

**Next**: Continue to [Performance Comparison](performance-comparison.md) for detailed benchmarks and performance benefits.