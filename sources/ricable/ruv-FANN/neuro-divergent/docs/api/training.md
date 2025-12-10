# Training System

The training system provides comprehensive infrastructure for optimizing neural forecasting models, including optimizers, loss functions, learning rate schedulers, callbacks, and training metrics.

## Overview

The training system consists of several key components:

- **Optimizers**: Gradient-based optimization algorithms
- **Loss Functions**: Objective functions for model training
- **Schedulers**: Learning rate adaptation strategies
- **Callbacks**: Training monitoring and control mechanisms
- **Metrics**: Performance evaluation during training

## Optimizers

### Adam Optimizer

The most commonly used optimizer, combining momentum with adaptive learning rates.

```rust
#[derive(Debug, Clone)]
pub struct Adam<T: Float> {
    pub learning_rate: T,
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub weight_decay: T,
    pub amsgrad: bool,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::{Adam, OptimizerConfig};

// Basic Adam optimizer
let adam = Adam::new(0.001)?;  // Default parameters

// Customized Adam
let custom_adam = Adam::builder()
    .learning_rate(0.001)
    .beta1(0.9)
    .beta2(0.999)
    .epsilon(1e-8)
    .weight_decay(0.01)  // L2 regularization
    .amsgrad(false)
    .build()?;

// Using optimizer config
let optimizer_config = OptimizerConfig::Adam {
    learning_rate: 0.001,
    beta1: 0.9,
    beta2: 0.999,
    weight_decay: 0.01,
};
```

### SGD (Stochastic Gradient Descent)

Simple yet effective optimizer, especially with momentum.

```rust
#[derive(Debug, Clone)]
pub struct SGD<T: Float> {
    pub learning_rate: T,
    pub momentum: T,
    pub dampening: T,
    pub weight_decay: T,
    pub nesterov: bool,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::SGD;

// Basic SGD
let sgd = SGD::new(0.01)?;

// SGD with momentum
let sgd_momentum = SGD::builder()
    .learning_rate(0.01)
    .momentum(0.9)
    .weight_decay(0.0001)
    .nesterov(true)  // Nesterov momentum
    .build()?;
```

### AdamW

Adam with decoupled weight decay, often better for transformer models.

```rust
#[derive(Debug, Clone)]
pub struct AdamW<T: Float> {
    pub learning_rate: T,
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub weight_decay: T,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::AdamW;

// AdamW for transformer training
let adamw = AdamW::builder()
    .learning_rate(0.0001)  // Lower LR for transformers
    .beta1(0.9)
    .beta2(0.999)
    .weight_decay(0.01)     // Important for AdamW
    .build()?;
```

## Loss Functions

### Mean Squared Error (MSE)

Standard loss function for regression tasks.

```rust
#[derive(Debug, Clone)]
pub struct MSE<T: Float> {
    pub reduction: ReductionType,
}

#[derive(Debug, Clone, Copy)]
pub enum ReductionType {
    Mean,
    Sum,
    None,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::{MSE, ReductionType};

// Standard MSE
let mse = MSE::new();

// MSE with sum reduction
let mse_sum = MSE::builder()
    .reduction(ReductionType::Sum)
    .build();

// Compute loss
let predictions = vec![1.0, 2.0, 3.0];
let targets = vec![1.1, 1.9, 3.2];
let loss = mse.compute(&predictions, &targets)?;
```

### Mean Absolute Error (MAE)

Robust loss function, less sensitive to outliers.

```rust
#[derive(Debug, Clone)]
pub struct MAE<T: Float> {
    pub reduction: ReductionType,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::MAE;

// MAE for robust training
let mae = MAE::new();

// Better for data with outliers
if data.has_outliers() {
    let loss_fn = MAE::new();
} else {
    let loss_fn = MSE::new();
}
```

### Huber Loss

Combines MSE and MAE benefits, quadratic for small errors, linear for large errors.

```rust
#[derive(Debug, Clone)]
pub struct Huber<T: Float> {
    pub delta: T,
    pub reduction: ReductionType,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::Huber;

// Huber loss with delta = 1.0
let huber = Huber::builder()
    .delta(1.0)
    .reduction(ReductionType::Mean)
    .build();

// Adaptive delta based on data
let delta = estimate_noise_level(&data) * 1.35;  // 1.35 is common multiplier
let adaptive_huber = Huber::builder()
    .delta(delta)
    .build();
```

### Quantile Loss

For probabilistic forecasting and prediction intervals.

```rust
#[derive(Debug, Clone)]
pub struct QuantileLoss<T: Float> {
    pub quantiles: Vec<T>,
    pub reduction: ReductionType,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::QuantileLoss;

// Multi-quantile loss
let quantile_loss = QuantileLoss::builder()
    .quantiles(vec![0.1, 0.5, 0.9])
    .reduction(ReductionType::Mean)
    .build();

// Single quantile (median)
let median_loss = QuantileLoss::builder()
    .quantiles(vec![0.5])
    .build();
```

### MAPE (Mean Absolute Percentage Error)

Percentage-based error, good for interpretability.

```rust
#[derive(Debug, Clone)]
pub struct MAPE<T: Float> {
    pub epsilon: T,  // Small value to avoid division by zero
    pub reduction: ReductionType,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::MAPE;

// MAPE with epsilon for numerical stability
let mape = MAPE::builder()
    .epsilon(1e-8)
    .reduction(ReductionType::Mean)
    .build();

// Only use MAPE when targets are not zero or very small
if data.min_value() > 0.01 {
    let loss_fn = MAPE::new();
} else {
    let loss_fn = MAE::new();  // Fallback to MAE
}
```

## Learning Rate Schedulers

### StepLR

Decay learning rate by a factor every few epochs.

```rust
#[derive(Debug, Clone)]
pub struct StepLR<T: Float> {
    pub step_size: usize,
    pub gamma: T,
    pub last_epoch: i32,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::StepLR;

// Decay by 0.1 every 30 epochs
let step_lr = StepLR::builder()
    .step_size(30)
    .gamma(0.1)
    .build();

// Conservative decay
let conservative_step = StepLR::builder()
    .step_size(50)
    .gamma(0.5)  // Halve every 50 epochs
    .build();
```

### ExponentialLR

Exponential decay of learning rate.

```rust
#[derive(Debug, Clone)]
pub struct ExponentialLR<T: Float> {
    pub gamma: T,
    pub last_epoch: i32,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::ExponentialLR;

// Exponential decay
let exp_lr = ExponentialLR::builder()
    .gamma(0.99)  // 1% decay per epoch
    .build();

// Faster decay
let fast_exp_lr = ExponentialLR::builder()
    .gamma(0.95)  // 5% decay per epoch
    .build();
```

### CosineAnnealingLR

Cosine annealing with restarts.

```rust
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR<T: Float> {
    pub t_max: usize,
    pub eta_min: T,
    pub last_epoch: i32,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::CosineAnnealingLR;

// Cosine annealing over 100 epochs
let cosine_lr = CosineAnnealingLR::builder()
    .t_max(100)
    .eta_min(1e-6)
    .build();

// For transformer training
let transformer_cosine = CosineAnnealingLR::builder()
    .t_max(50)
    .eta_min(1e-7)
    .build();
```

### ReduceLROnPlateau

Reduce learning rate when metric plateaus.

```rust
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau<T: Float> {
    pub mode: PlateauMode,
    pub factor: T,
    pub patience: usize,
    pub threshold: T,
    pub threshold_mode: ThresholdMode,
    pub cooldown: usize,
    pub min_lr: T,
}

#[derive(Debug, Clone, Copy)]
pub enum PlateauMode {
    Min,  // For loss
    Max,  // For accuracy
}
```

#### Usage Examples

```rust
use neuro_divergent::training::{ReduceLROnPlateau, PlateauMode};

// Reduce LR when validation loss plateaus
let plateau_lr = ReduceLROnPlateau::builder()
    .mode(PlateauMode::Min)
    .factor(0.5)
    .patience(10)
    .threshold(0.01)
    .min_lr(1e-6)
    .build();
```

## Training Callbacks

### EarlyStopping

Stop training when metric stops improving.

```rust
#[derive(Debug, Clone)]
pub struct EarlyStopping<T: Float> {
    pub monitor: String,
    pub min_delta: T,
    pub patience: usize,
    pub mode: StoppingMode,
    pub restore_best_weights: bool,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::{EarlyStopping, StoppingMode};

// Early stopping on validation loss
let early_stopping = EarlyStopping::builder()
    .monitor("val_loss".to_string())
    .min_delta(0.001)
    .patience(15)
    .mode(StoppingMode::Min)
    .restore_best_weights(true)
    .build();

// Early stopping on accuracy
let accuracy_stopping = EarlyStopping::builder()
    .monitor("val_accuracy".to_string())
    .patience(20)
    .mode(StoppingMode::Max)
    .build();
```

### ModelCheckpoint

Save model at best performance.

```rust
#[derive(Debug, Clone)]
pub struct ModelCheckpoint {
    pub filepath: String,
    pub monitor: String,
    pub mode: CheckpointMode,
    pub save_best_only: bool,
    pub save_weights_only: bool,
    pub period: usize,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::{ModelCheckpoint, CheckpointMode};

// Save best model based on validation loss
let checkpoint = ModelCheckpoint::builder()
    .filepath("best_model.json".to_string())
    .monitor("val_loss".to_string())
    .mode(CheckpointMode::Min)
    .save_best_only(true)
    .build();

// Save every 10 epochs
let periodic_checkpoint = ModelCheckpoint::builder()
    .filepath("checkpoint_epoch_{epoch}.json".to_string())
    .period(10)
    .save_best_only(false)
    .build();
```

### LearningRateMonitor

Monitor and log learning rate changes.

```rust
#[derive(Debug, Clone)]
pub struct LearningRateMonitor {
    pub logging_interval: usize,
    pub log_momentum: bool,
}
```

## Training Configuration

### Complete Training Setup

```rust
use neuro_divergent::training::TrainingConfig;

let training_config = TrainingConfig::builder()
    .max_epochs(100)
    .batch_size(32)
    .learning_rate(0.001)
    .optimizer(OptimizerConfig::Adam {
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
    })
    .loss_function(LossFunctionConfig::MSE)
    .scheduler(SchedulerConfig::ReduceLROnPlateau {
        patience: 10,
        factor: 0.5,
        min_lr: 1e-6,
    })
    .early_stopping(EarlyStoppingConfig {
        monitor: "val_loss".to_string(),
        patience: 15,
        min_delta: 0.001,
        mode: StoppingMode::Min,
    })
    .gradient_clipping(1.0)
    .validation_split(0.2)
    .build()?;
```

## Training Metrics

### Tracking Training Progress

```rust
#[derive(Debug, Clone)]
pub struct TrainingMetrics<T: Float> {
    pub epoch: usize,
    pub train_loss: T,
    pub val_loss: Option<T>,
    pub learning_rate: T,
    pub epoch_time: f64,
    pub custom_metrics: HashMap<String, T>,
}
```

#### Usage Examples

```rust
use neuro_divergent::training::TrainingMetrics;

// Access training metrics during training
fn on_epoch_end(metrics: &TrainingMetrics<f64>) {
    println!("Epoch {}: Loss = {:.4}, LR = {:.6}", 
             metrics.epoch, metrics.train_loss, metrics.learning_rate);
    
    if let Some(val_loss) = metrics.val_loss {
        println!("  Validation Loss = {:.4}", val_loss);
    }
    
    // Log custom metrics
    for (name, value) in &metrics.custom_metrics {
        println!("  {}: {:.4}", name, value);
    }
}
```

### Custom Metrics

```rust
use neuro_divergent::training::Metric;

// Define custom metric
struct MAPE;

impl<T: Float> Metric<T> for MAPE {
    fn name(&self) -> &str {
        "mape"
    }
    
    fn compute(&self, predictions: &[T], targets: &[T]) -> NeuroDivergentResult<T> {
        let mut total_error = T::zero();
        let mut count = 0;
        
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if target.abs() > T::from(1e-8).unwrap() {
                let percentage_error = (pred - target).abs() / target.abs();
                total_error = total_error + percentage_error;
                count += 1;
            }
        }
        
        if count > 0 {
            Ok(total_error / T::from(count).unwrap() * T::from(100.0).unwrap())
        } else {
            Err(NeuroDivergentError::math("Cannot compute MAPE: all targets are zero"))
        }
    }
}
```

## Advanced Training Techniques

### Gradient Clipping

```rust
use neuro_divergent::training::{GradientClipping, ClippingType};

// Gradient norm clipping
let gradient_clipping = GradientClipping::builder()
    .max_norm(1.0)
    .norm_type(ClippingType::L2Norm)
    .build();

// Gradient value clipping
let value_clipping = GradientClipping::builder()
    .max_norm(0.5)
    .norm_type(ClippingType::Value)
    .build();
```

### Mixed Precision Training

```rust
use neuro_divergent::training::MixedPrecisionConfig;

// Enable mixed precision for faster training
let mixed_precision = MixedPrecisionConfig::builder()
    .enabled(true)
    .loss_scale(128.0)
    .scale_window(2000)
    .build();
```

### Learning Rate Warmup

```rust
use neuro_divergent::training::WarmupConfig;

// Linear warmup for first 1000 steps
let warmup = WarmupConfig::builder()
    .warmup_steps(1000)
    .warmup_method(WarmupMethod::Linear)
    .start_factor(0.1)
    .build();
```

## Training Best Practices

### Model-Specific Training Strategies

```rust
// LSTM training configuration
fn lstm_training_config() -> TrainingConfig {
    TrainingConfig::builder()
        .max_epochs(200)
        .batch_size(32)
        .learning_rate(0.001)
        .optimizer(OptimizerConfig::Adam {
            weight_decay: 0.01
        })
        .gradient_clipping(1.0)  // Important for RNNs
        .scheduler(SchedulerConfig::ReduceLROnPlateau {
            patience: 10,
            factor: 0.5,
        })
        .build()
        .unwrap()
}

// Transformer training configuration
fn transformer_training_config() -> TrainingConfig {
    TrainingConfig::builder()
        .max_epochs(100)
        .batch_size(16)  // Smaller batch for memory
        .learning_rate(0.0001)  // Lower LR
        .optimizer(OptimizerConfig::AdamW {
            weight_decay: 0.01
        })
        .scheduler(SchedulerConfig::CosineAnnealingLR {
            t_max: 100,
            eta_min: 1e-6,
        })
        .warmup_steps(1000)
        .build()
        .unwrap()
}
```

### Hyperparameter Tuning

```rust
use neuro_divergent::training::HyperparameterTuner;

// Define search space
let search_space = SearchSpace::builder()
    .add_param("learning_rate", LogUniform::new(1e-5, 1e-1))
    .add_param("batch_size", Choice::new(vec![16, 32, 64, 128]))
    .add_param("hidden_size", Choice::new(vec![64, 128, 256, 512]))
    .add_param("num_layers", IntUniform::new(1, 4))
    .add_param("dropout", Uniform::new(0.0, 0.5))
    .build();

// Run hyperparameter optimization
let tuner = HyperparameterTuner::new(search_space);
let best_params = tuner.optimize(&data, &validation_data, 50)?;  // 50 trials
```

### Training Pipeline

```rust
// Complete training pipeline
fn train_model(
    mut model: Box<dyn BaseModel<f64>>,
    train_data: &TimeSeriesDataFrame<f64>,
    val_data: Option<&TimeSeriesDataFrame<f64>>,
    config: TrainingConfig,
) -> NeuroDivergentResult<TrainingHistory<f64>> {
    
    // Setup data loaders
    let train_dataset = TimeSeriesDataset::from_dataframe(train_data)?;
    let val_dataset = val_data.map(TimeSeriesDataset::from_dataframe).transpose()?;
    
    // Initialize optimizer
    let mut optimizer = create_optimizer(&config.optimizer, &model)?;
    
    // Initialize scheduler
    let mut scheduler = create_scheduler(&config.scheduler, &optimizer)?;
    
    // Initialize callbacks
    let mut callbacks = create_callbacks(&config)?;
    
    let mut history = TrainingHistory::new();
    
    // Training loop
    for epoch in 0..config.max_epochs {
        // Training phase
        model.train_mode();
        let train_loss = train_epoch(&mut model, &train_dataset, &mut optimizer, &config)?;
        
        // Validation phase
        let val_loss = if let Some(val_data) = &val_dataset {
            model.eval_mode();
            Some(validate_epoch(&model, val_data, &config)?)
        } else {
            None
        };
        
        // Update learning rate
        scheduler.step(val_loss.unwrap_or(train_loss))?;
        
        // Record metrics
        let metrics = TrainingMetrics {
            epoch,
            train_loss,
            val_loss,
            learning_rate: optimizer.learning_rate(),
            epoch_time: epoch_start_time.elapsed().as_secs_f64(),
            custom_metrics: HashMap::new(),
        };
        
        history.add_epoch(metrics.clone());
        
        // Run callbacks
        let should_stop = callbacks.on_epoch_end(&metrics)?;
        if should_stop {
            println!("Early stopping at epoch {}", epoch);
            break;
        }
    }
    
    Ok(history)
}
```

The training system provides comprehensive tools for optimizing neural forecasting models with state-of-the-art algorithms, adaptive strategies, and monitoring capabilities to ensure efficient and effective model training.