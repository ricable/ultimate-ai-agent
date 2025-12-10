# Training Guide

This comprehensive guide covers training neural forecasting models in Neuro-Divergent, from basic training loops to advanced optimization strategies and distributed training.

## Training Overview

Training in Neuro-Divergent follows a structured approach:

```
Data Preparation → Model Configuration → Training Loop → Validation → Model Selection
```

The training system is designed to be both powerful and easy to use, with sensible defaults that work well for most use cases.

## Basic Training

### Simple Training Example

```rust
use neuro_divergent::prelude::*;

async fn basic_training_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load and prepare data
    let data = TimeSeriesDataFrame::from_csv("data.csv")?;
    
    // Create model
    let mut model = LSTM::new(LSTMConfig::new()
        .with_input_size(24)
        .with_hidden_size(128)
        .with_horizon(12))?;
    
    // Basic training
    model.fit(&data).await?;
    
    println!("Training completed!");
    Ok(())
}
```

### Training Configuration

```rust
// Detailed training configuration
let training_config = TrainingConfig::new()
    .with_max_epochs(100)
    .with_learning_rate(0.001)
    .with_batch_size(32)
    .with_validation_split(0.2)
    .with_early_stopping(patience: 15)
    .with_checkpoint_every(10)  // Save model every 10 epochs
    .with_verbose(true);

model.fit_with_config(&data, &training_config).await?;
```

## Training Configuration Deep Dive

### Learning Rate Strategies

```rust
use neuro_divergent::training::{LearningRateScheduler, SchedulerType};

// Fixed learning rate
let fixed_lr = LearningRateScheduler::fixed(0.001);

// Step decay
let step_lr = LearningRateScheduler::step_decay()
    .with_initial_lr(0.01)
    .with_step_size(30)     // Decay every 30 epochs
    .with_gamma(0.1);       // Multiply by 0.1

// Exponential decay
let exp_lr = LearningRateScheduler::exponential_decay()
    .with_initial_lr(0.01)
    .with_decay_rate(0.95)
    .with_decay_steps(10);

// Cosine annealing
let cosine_lr = LearningRateScheduler::cosine_annealing()
    .with_initial_lr(0.01)
    .with_min_lr(1e-6)
    .with_t_max(100);       // Full cycle length

// Warm restart
let restart_lr = LearningRateScheduler::cosine_with_warm_restarts()
    .with_initial_lr(0.01)
    .with_t_0(10)           // Initial cycle length
    .with_t_mult(2);        // Cycle length multiplier

let training_config = TrainingConfig::new()
    .with_learning_rate_scheduler(cosine_lr);
```

### Optimizers

```rust
use neuro_divergent::training::{Optimizer, OptimizerConfig};

// Adam (default and recommended)
let adam = Optimizer::Adam(AdamConfig::new()
    .with_learning_rate(0.001)
    .with_beta1(0.9)
    .with_beta2(0.999)
    .with_weight_decay(1e-5)
    .with_amsgrad(false));

// AdamW (better weight decay)
let adamw = Optimizer::AdamW(AdamWConfig::new()
    .with_learning_rate(0.001)
    .with_weight_decay(0.01)
    .with_decoupled_weight_decay(true));

// SGD with momentum
let sgd = Optimizer::SGD(SGDConfig::new()
    .with_learning_rate(0.01)
    .with_momentum(0.9)
    .with_nesterov(true)
    .with_weight_decay(1e-4));

// RMSprop
let rmsprop = Optimizer::RMSprop(RMSpropConfig::new()
    .with_learning_rate(0.001)
    .with_alpha(0.99)
    .with_eps(1e-8));

let training_config = TrainingConfig::new()
    .with_optimizer(adamw);
```

### Loss Functions

```rust
use neuro_divergent::training::{LossFunction, LossConfig};

// Mean Absolute Error (robust to outliers)
let mae_loss = LossFunction::MAE;

// Mean Squared Error (standard choice)
let mse_loss = LossFunction::MSE;

// Huber Loss (combination of MAE and MSE)
let huber_loss = LossFunction::Huber(HuberConfig::new()
    .with_delta(1.0));

// Quantile Loss (for probabilistic forecasting)
let quantile_loss = LossFunction::Quantile(QuantileConfig::new()
    .with_quantiles(vec![0.1, 0.5, 0.9])
    .with_weights(vec![1.0, 2.0, 1.0])); // Weight median more

// Custom weighted loss
let weighted_loss = LossFunction::Weighted(WeightedConfig::new()
    .with_base_loss(LossFunction::MAE)
    .with_weight_function(|timestamp, value| {
        // More recent observations get higher weight
        let days_ago = (Utc::now() - timestamp).num_days();
        (-days_ago as f64 * 0.01).exp()
    }));

let training_config = TrainingConfig::new()
    .with_loss_function(quantile_loss);
```

## Advanced Training Techniques

### Early Stopping

```rust
use neuro_divergent::training::{EarlyStopping, EarlyStoppingConfig};

let early_stopping = EarlyStopping::new()
    .with_patience(20)              // Wait 20 epochs before stopping
    .with_min_delta(1e-4)           // Minimum improvement threshold
    .with_restore_best_weights(true) // Restore best model weights
    .with_monitor_metric(Metric::ValidationLoss)
    .with_mode(Mode::Minimize);

let training_config = TrainingConfig::new()
    .with_early_stopping(early_stopping);
```

### Gradient Clipping

```rust
// Prevent exploding gradients
let training_config = TrainingConfig::new()
    .with_gradient_clipping(GradientClipping::ByNorm(1.0))  // Clip by norm
    .with_gradient_clipping(GradientClipping::ByValue(-1.0, 1.0)); // Clip by value
```

### Regularization

```rust
// L1 and L2 regularization
let training_config = TrainingConfig::new()
    .with_l1_regularization(1e-5)   // L1 penalty
    .with_l2_regularization(1e-4)   // L2 penalty
    .with_dropout(0.2)              // Dropout in model layers
    .with_label_smoothing(0.1);     // Label smoothing for classification
```

### Mixed Precision Training

```rust
// Enable mixed precision for faster training on modern GPUs
let training_config = TrainingConfig::new()
    .with_mixed_precision(true)
    .with_loss_scaling(LossScaling::Dynamic)
    .with_fp16_opt_level(OptLevel::O1); // Conservative mixed precision
```

## Training Monitoring

### Progress Tracking

```rust
use neuro_divergent::training::{TrainingMonitor, MetricTracker};

let monitor = TrainingMonitor::new()
    .with_metrics(vec![
        Metric::TrainingLoss,
        Metric::ValidationLoss,
        Metric::MAE,
        Metric::MAPE,
        Metric::LearningRate,
    ])
    .with_log_interval(10)          // Log every 10 epochs
    .with_plot_interval(50)         // Plot every 50 epochs
    .with_save_plots(true);

let training_config = TrainingConfig::new()
    .with_monitor(monitor);
```

### Custom Callbacks

```rust
use neuro_divergent::training::{Callback, CallbackConfig};

// Custom callback for learning rate scheduling
struct CustomLRScheduler {
    initial_lr: f64,
    patience: usize,
    factor: f64,
    best_loss: f64,
    wait: usize,
}

impl Callback for CustomLRScheduler {
    fn on_epoch_end(&mut self, epoch: usize, logs: &TrainingLogs) -> CallbackResult {
        let current_loss = logs.get_metric("validation_loss").unwrap_or(f64::INFINITY);
        
        if current_loss < self.best_loss {
            self.best_loss = current_loss;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                let new_lr = logs.get_metric("learning_rate").unwrap() * self.factor;
                logs.set_learning_rate(new_lr)?;
                self.wait = 0;
                println!("Reduced learning rate to {:.6}", new_lr);
            }
        }
        
        CallbackResult::Continue
    }
}
```

### Visualization

```rust
// Real-time training visualization
let visualizer = TrainingVisualizer::new()
    .with_live_plotting(true)
    .with_metrics_dashboard(true)
    .with_model_architecture_plot(true)
    .with_save_directory("training_plots/");

// Enable visualization
let training_config = TrainingConfig::new()
    .with_visualizer(visualizer);
```

## Hyperparameter Optimization

### Grid Search

```rust
use neuro_divergent::optimization::{GridSearch, HyperparameterSpace};

let param_space = HyperparameterSpace::new()
    .add_categorical("model_type", vec!["LSTM", "GRU", "Transformer"])
    .add_discrete("hidden_size", vec![64, 128, 256, 512])
    .add_discrete("num_layers", vec![1, 2, 3, 4])
    .add_continuous("learning_rate", 1e-5, 1e-1, log_scale: true)
    .add_continuous("dropout", 0.0, 0.5);

let grid_search = GridSearch::new()
    .with_parameter_space(param_space)
    .with_cv_folds(5)
    .with_scoring_metric(Metric::MAE)
    .with_n_jobs(-1)  // Use all available cores
    .with_timeout(Duration::from_hours(6));

let best_config = grid_search.fit(&data).await?;
println!("Best configuration: {:?}", best_config);
```

### Random Search

```rust
use neuro_divergent::optimization::RandomSearch;

let random_search = RandomSearch::new()
    .with_parameter_space(param_space)
    .with_n_trials(100)
    .with_cv_folds(3)
    .with_scoring_metric(Metric::MAE)
    .with_early_stopping(20);  // Stop if no improvement for 20 trials

let best_config = random_search.fit(&data).await?;
```

### Bayesian Optimization

```rust
use neuro_divergent::optimization::BayesianOptimization;

let bayesian_opt = BayesianOptimization::new()
    .with_parameter_space(param_space)
    .with_acquisition_function(AcquisitionFunction::ExpectedImprovement)
    .with_n_initial_points(10)
    .with_n_calls(50)
    .with_gaussian_process_kernel(Kernel::Matern52)
    .with_alpha(1e-6);

let best_config = bayesian_opt.fit(&data).await?;

// Plot optimization history
bayesian_opt.plot_convergence("optimization_convergence.png")?;
bayesian_opt.plot_partial_dependence("partial_dependence.png")?;
```

### Multi-Objective Optimization

```rust
use neuro_divergent::optimization::MultiObjectiveOptimization;

let multi_objective = MultiObjectiveOptimization::new()
    .add_objective("accuracy", weight: 0.7, direction: Direction::Maximize)
    .add_objective("training_time", weight: 0.2, direction: Direction::Minimize)
    .add_objective("model_size", weight: 0.1, direction: Direction::Minimize)
    .with_algorithm(Algorithm::NSGA2)
    .with_population_size(50)
    .with_generations(100);

let pareto_front = multi_objective.optimize(&data).await?;
```

## Cross-Validation

### Time Series Cross-Validation

```rust
use neuro_divergent::validation::{TimeSeriesCrossValidator, CVConfig};

let cv = TimeSeriesCrossValidator::new()
    .with_n_splits(5)
    .with_test_size(0.2)
    .with_gap_size(7)           // 7-day gap between train and test
    .with_expanding_window(true) // Use expanding window
    .with_purge_size(3);        // Purge 3 observations to prevent leakage

let cv_results = cv.cross_validate(&model, &data).await?;

println!("Cross-validation results:");
println!("Mean MAE: {:.4} ± {:.4}", cv_results.mean_mae(), cv_results.std_mae());
println!("Mean MAPE: {:.4} ± {:.4}", cv_results.mean_mape(), cv_results.std_mape());
```

### Blocked Cross-Validation

```rust
// For multiple time series
let blocked_cv = BlockedTimeSeriesCrossValidator::new()
    .with_n_splits(5)
    .with_block_size(30)        // 30-day blocks
    .with_test_ratio(0.2)
    .with_series_splits(true);  // Split by series, not just time

let blocked_results = blocked_cv.cross_validate(&global_model, &multi_series_data).await?;
```

### Custom Cross-Validation

```rust
use neuro_divergent::validation::CustomCrossValidator;

// Define custom split strategy
struct SeasonalSplitter {
    seasons_per_fold: usize,
}

impl CrossValidationSplitter for SeasonalSplitter {
    fn split(&self, data: &TimeSeriesDataFrame) -> Vec<(TrainIndex, TestIndex)> {
        // Custom logic to split by seasons
        // Return vector of (train_indices, test_indices)
        unimplemented!()
    }
}

let custom_cv = CustomCrossValidator::new()
    .with_splitter(Box::new(SeasonalSplitter { seasons_per_fold: 4 }))
    .with_scoring_metrics(vec![Metric::MAE, Metric::RMSE, Metric::MAPE]);
```

## Distributed Training

### Multi-GPU Training

```rust
use neuro_divergent::distributed::{DistributedTraining, GPUConfig};

// Single machine, multiple GPUs
let gpu_config = GPUConfig::new()
    .with_devices(vec![0, 1, 2, 3])     // Use GPUs 0-3
    .with_strategy(DistributionStrategy::DataParallel)
    .with_batch_size_per_gpu(8)         // 8 samples per GPU
    .with_gradient_synchronization(SyncMethod::AllReduce);

let distributed_training = DistributedTraining::new()
    .with_gpu_config(gpu_config)
    .with_mixed_precision(true);

let training_config = TrainingConfig::new()
    .with_distributed_training(distributed_training);

model.fit_with_config(&data, &training_config).await?;
```

### Multi-Node Training

```rust
use neuro_divergent::distributed::{ClusterConfig, NodeConfig};

// Multiple machines training
let cluster_config = ClusterConfig::new()
    .add_node(NodeConfig::new()
        .with_address("192.168.1.10:8000")
        .with_gpus(vec![0, 1])
        .with_role(NodeRole::Master))
    .add_node(NodeConfig::new()
        .with_address("192.168.1.11:8000")
        .with_gpus(vec![0, 1])
        .with_role(NodeRole::Worker))
    .with_communication_backend(Backend::NCCL)
    .with_distributed_optimizer(DistributedOptimizer::ZeRO);

let distributed_training = DistributedTraining::new()
    .with_cluster_config(cluster_config);
```

### Federated Learning

```rust
use neuro_divergent::distributed::FederatedLearning;

// Train on multiple datasets without sharing raw data
let federated_config = FederatedLearning::new()
    .with_aggregation_strategy(AggregationStrategy::FedAvg)
    .with_local_epochs(5)
    .with_participation_ratio(0.8)      // 80% of clients participate
    .with_differential_privacy(true)
    .with_privacy_budget(1.0);

let federated_trainer = FederatedTrainer::new()
    .with_config(federated_config)
    .add_client_data("client_1", client_1_data)
    .add_client_data("client_2", client_2_data);

let global_model = federated_trainer.train(base_model).await?;
```

## Model Checkpointing and Recovery

### Automatic Checkpointing

```rust
use neuro_divergent::checkpointing::{CheckpointConfig, CheckpointStrategy};

let checkpoint_config = CheckpointConfig::new()
    .with_save_directory("checkpoints/")
    .with_strategy(CheckpointStrategy::BestValidationLoss)
    .with_save_frequency(SaveFrequency::EveryNEpochs(10))
    .with_keep_last_n(5)            // Keep only last 5 checkpoints
    .with_compression(true)
    .with_metadata(true);           // Save training metadata

let training_config = TrainingConfig::new()
    .with_checkpointing(checkpoint_config);
```

### Manual Checkpointing

```rust
// Save model during training
model.save_checkpoint("checkpoint_epoch_50.model")?;

// Resume training from checkpoint
let mut resumed_model = LSTM::load_checkpoint("checkpoint_epoch_50.model")?;
resumed_model.resume_training(&data, &training_config).await?;
```

### Fault Tolerance

```rust
use neuro_divergent::training::FaultTolerantTraining;

let fault_tolerant = FaultTolerantTraining::new()
    .with_auto_checkpoint(true)
    .with_retry_on_failure(3)       // Retry 3 times on failure
    .with_health_check_interval(Duration::from_secs(60))
    .with_graceful_shutdown(true);

let training_config = TrainingConfig::new()
    .with_fault_tolerance(fault_tolerant);
```

## Training for Different Model Types

### Recurrent Models (LSTM, GRU)

```rust
// LSTM-specific training considerations
let lstm_training = TrainingConfig::new()
    .with_max_epochs(200)           // RNNs often need more epochs
    .with_learning_rate(0.001)      // Conservative learning rate
    .with_gradient_clipping(GradientClipping::ByNorm(1.0)) // Prevent exploding gradients
    .with_batch_size(32)            // Moderate batch size
    .with_sequence_length(50)       // Longer sequences for better patterns
    .with_teacher_forcing_ratio(0.5); // For sequence-to-sequence models

lstm_model.fit_with_config(&data, &lstm_training).await?;
```

### Transformer Models

```rust
// Transformer-specific training
let transformer_training = TrainingConfig::new()
    .with_max_epochs(100)           // Transformers train faster
    .with_learning_rate(0.0001)     // Lower learning rate
    .with_warmup_steps(4000)        // Learning rate warmup
    .with_batch_size(64)            // Larger batch sizes
    .with_label_smoothing(0.1)      // Regularization
    .with_attention_dropout(0.1);   // Attention-specific dropout

transformer_model.fit_with_config(&data, &transformer_training).await?;
```

### Probabilistic Models (DeepAR)

```rust
// DeepAR-specific training
let deepar_training = TrainingConfig::new()
    .with_loss_function(LossFunction::NegativeLogLikelihood)
    .with_sample_size(100)          // Number of samples during training
    .with_likelihood(Likelihood::Gaussian) // Or StudentT, NegativeBinomial
    .with_quantile_levels(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]);

deepar_model.fit_with_config(&data, &deepar_training).await?;
```

## Training Diagnostics

### Learning Curves

```rust
use neuro_divergent::diagnostics::LearningCurveAnalyzer;

let analyzer = LearningCurveAnalyzer::new()
    .with_train_sizes(vec![0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    .with_cv_folds(5)
    .with_scoring_metric(Metric::MAE);

let learning_curves = analyzer.analyze(&model, &data).await?;
learning_curves.plot("learning_curves.png")?;

// Diagnose issues
if learning_curves.is_underfitting() {
    println!("Model is underfitting - try increasing complexity");
} else if learning_curves.is_overfitting() {
    println!("Model is overfitting - try regularization or more data");
}
```

### Gradient Analysis

```rust
use neuro_divergent::diagnostics::GradientAnalyzer;

let grad_analyzer = GradientAnalyzer::new()
    .with_layer_analysis(true)
    .with_gradient_flow_visualization(true);

// Analyze gradients during training
model.set_gradient_analyzer(grad_analyzer);
model.fit(&data).await?;

let gradient_report = model.get_gradient_analysis()?;
gradient_report.print_summary();

if gradient_report.has_vanishing_gradients() {
    println!("Vanishing gradients detected - try different activation functions");
}
if gradient_report.has_exploding_gradients() {
    println!("Exploding gradients detected - try gradient clipping");
}
```

### Weight Analysis

```rust
use neuro_divergent::diagnostics::WeightAnalyzer;

let weight_analyzer = WeightAnalyzer::new()
    .with_weight_distribution_analysis(true)
    .with_dead_neuron_detection(true)
    .with_layer_utilization_analysis(true);

let weight_report = weight_analyzer.analyze(&trained_model)?;
weight_report.plot_weight_distributions("weight_distributions.png")?;

println!("Dead neurons: {}", weight_report.dead_neuron_count());
println!("Average layer utilization: {:.2}%", weight_report.avg_utilization() * 100.0);
```

## Production Training Considerations

### Automated Training Pipeline

```rust
use neuro_divergent::pipeline::{TrainingPipeline, PipelineConfig};

let pipeline = TrainingPipeline::new()
    .add_stage("data_validation", DataValidationStage::new())
    .add_stage("preprocessing", PreprocessingStage::new())
    .add_stage("feature_engineering", FeatureEngineeringStage::new())
    .add_stage("hyperparameter_tuning", HyperparameterTuningStage::new())
    .add_stage("training", TrainingStage::new())
    .add_stage("evaluation", EvaluationStage::new())
    .add_stage("model_registration", ModelRegistrationStage::new())
    .with_error_handling(ErrorHandling::StopOnError)
    .with_logging(LogLevel::Info)
    .with_monitoring(true);

let pipeline_result = pipeline.execute(&raw_data).await?;
```

### Model Versioning

```rust
use neuro_divergent::versioning::{ModelVersioning, VersionConfig};

let versioning = ModelVersioning::new()
    .with_registry("model_registry/")
    .with_versioning_strategy(VersioningStrategy::Semantic)
    .with_metadata_tracking(true)
    .with_experiment_tracking(true);

// Train and version model
let model_version = versioning.train_and_register(
    &model_config,
    &training_config,
    &data,
    &metadata
).await?;

println!("Model registered as version: {}", model_version);
```

### Continuous Training

```rust
use neuro_divergent::continuous::{ContinuousTraining, RetrainingTrigger};

let continuous_training = ContinuousTraining::new()
    .with_data_source("streaming_data_source")
    .with_retrain_triggers(vec![
        RetrainingTrigger::DataDrift(threshold: 0.05),
        RetrainingTrigger::PerformanceDegradation(threshold: 0.1),
        RetrainingTrigger::TimeInterval(Duration::from_days(7)),
    ])
    .with_incremental_learning(true)
    .with_validation_strategy(ValidationStrategy::HoldoutLatest)
    .with_automatic_deployment(true);

continuous_training.start().await?;
```

## Best Practices

### Training Best Practices Checklist

- [ ] **Start with sensible defaults** and iterate
- [ ] **Use time-aware validation** splits
- [ ] **Monitor both training and validation metrics**
- [ ] **Implement early stopping** to prevent overfitting
- [ ] **Use gradient clipping** for RNNs
- [ ] **Save checkpoints** regularly
- [ ] **Log hyperparameters** and results
- [ ] **Validate on out-of-sample data**

### Common Training Issues

#### Overfitting
```rust
// Solutions for overfitting
let training_config = TrainingConfig::new()
    .with_dropout(0.3)              // Increase dropout
    .with_l2_regularization(1e-3)   // Add L2 regularization
    .with_early_stopping(patience: 10) // Stop training earlier
    .with_data_augmentation(true);  // Augment training data
```

#### Slow Convergence
```rust
// Solutions for slow convergence
let training_config = TrainingConfig::new()
    .with_learning_rate(0.01)       // Increase learning rate
    .with_batch_size(64)            // Increase batch size
    .with_gradient_clipping(None)   // Remove gradient clipping
    .with_optimizer(Optimizer::AdamW); // Try different optimizer
```

#### Unstable Training
```rust
// Solutions for unstable training
let training_config = TrainingConfig::new()
    .with_learning_rate(0.0001)     // Decrease learning rate
    .with_gradient_clipping(GradientClipping::ByNorm(0.5)) // Add gradient clipping
    .with_batch_norm(true)          // Add batch normalization
    .with_mixed_precision(false);   // Disable mixed precision
```

## Next Steps

Now that you understand training fundamentals:

1. **Make Predictions**: Learn [Prediction Techniques](prediction.md)
2. **Evaluate Performance**: Master [Evaluation Methods](evaluation.md)
3. **Optimize Models**: Explore [Performance Optimization](performance.md)
4. **Deploy Models**: Review [Best Practices](best-practices.md)

Remember: successful training requires patience, experimentation, and systematic evaluation. Start simple and gradually increase complexity as needed.