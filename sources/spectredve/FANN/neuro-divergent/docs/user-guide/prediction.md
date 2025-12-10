# Prediction Guide

This comprehensive guide covers making predictions with neural forecasting models in Neuro-Divergent, from basic point forecasts to advanced probabilistic predictions and real-time inference.

## Prediction Overview

Prediction in Neuro-Divergent supports multiple modes:

```
Point Forecasts → Single values for each time step
Probabilistic → Full distributions with uncertainty
Quantile → Specific percentiles (e.g., 10th, 90th)
Scenario → Multiple possible futures
Real-time → Streaming predictions
```

## Basic Prediction

### Simple Point Forecasts

```rust
use neuro_divergent::prelude::*;

async fn basic_prediction_example() -> Result<(), Box<dyn std::error::Error>> {
    // Load trained model
    let model = LSTM::load("trained_model.bin")?;
    
    // Load test data
    let test_data = TimeSeriesDataFrame::from_csv("test_data.csv")?;
    
    // Generate point forecasts
    let forecasts = model.predict(&test_data).await?;
    
    // Access results
    println!("Forecast horizon: {}", forecasts.horizon());
    for (i, value) in forecasts.values().iter().enumerate() {
        println!("Step {}: {:.2}", i + 1, value);
    }
    
    Ok(())
}
```

### Prediction Configuration

```rust
// Configure prediction behavior
let prediction_config = PredictionConfig::new()
    .with_horizon(24)               // 24 steps ahead
    .with_batch_size(32)            // Process 32 series at once
    .with_return_attention_weights(true) // For interpretability
    .with_return_hidden_states(true)     // For analysis
    .with_temperature(1.0)          // For probabilistic models
    .with_top_k(10);                // For sampling-based models

let forecasts = model.predict_with_config(&test_data, &prediction_config).await?;
```

## Probabilistic Forecasting

### Prediction Intervals

```rust
use neuro_divergent::prediction::{ProbabilisticForecaster, IntervalConfig};

// Generate prediction intervals
let interval_config = IntervalConfig::new()
    .with_confidence_levels(vec![0.8, 0.9, 0.95])  // 80%, 90%, 95% intervals
    .with_method(IntervalMethod::Quantile);

let probabilistic_forecasts = model.predict_with_intervals(
    &test_data, 
    &interval_config
).await?;

// Access intervals
for level in probabilistic_forecasts.confidence_levels() {
    let (lower, upper) = probabilistic_forecasts.get_interval(level)?;
    println!("{}% Interval: [{:.2}, {:.2}]", level * 100.0, lower[0], upper[0]);
}
```

### Quantile Forecasting

```rust
use neuro_divergent::prediction::QuantileForecaster;

// Predict specific quantiles
let quantiles = vec![0.1, 0.25, 0.5, 0.75, 0.9];
let quantile_forecasts = model.predict_quantiles(&test_data, &quantiles).await?;

// Access quantile predictions
for (i, q) in quantiles.iter().enumerate() {
    let q_forecast = quantile_forecasts.get_quantile(*q)?;
    println!("{}% quantile: {:.2}", q * 100.0, q_forecast[0]);
}

// Visualize quantile fan chart
quantile_forecasts.plot_fan_chart("quantile_fan.png")?;
```

### Distribution Sampling

```rust
use neuro_divergent::prediction::DistributionSampler;

// Sample from predictive distribution
let sampler = DistributionSampler::new()
    .with_num_samples(1000)
    .with_seed(42);

let samples = model.sample_predictions(&test_data, &sampler).await?;

// Analyze samples
let mean_forecast = samples.mean_across_samples();
let std_forecast = samples.std_across_samples();
let percentiles = samples.percentiles(&[5.0, 25.0, 50.0, 75.0, 95.0]);

println!("Mean forecast: {:.2} ± {:.2}", mean_forecast[0], std_forecast[0]);
```

## Multi-Step Forecasting Strategies

### Direct Multi-Step

```rust
// Train model to predict entire horizon directly
let direct_config = ModelConfig::new()
    .with_prediction_strategy(PredictionStrategy::Direct)
    .with_horizon(12);  // Predict all 12 steps simultaneously

let direct_model = LSTM::new(direct_config)?;
direct_model.fit(&train_data).await?;

let direct_forecasts = direct_model.predict(&test_data).await?;
```

### Recursive (Iterative)

```rust
// Use 1-step predictions recursively
let recursive_config = PredictionConfig::new()
    .with_strategy(PredictionStrategy::Recursive)
    .with_horizon(12);

let recursive_forecasts = one_step_model.predict_recursive(&test_data, &recursive_config).await?;

// Access recursive predictions with uncertainty propagation
for step in 0..recursive_forecasts.horizon() {
    let uncertainty = recursive_forecasts.uncertainty_at_step(step);
    println!("Step {}: uncertainty = {:.3}", step + 1, uncertainty);
}
```

### Multi-Output Direct

```rust
// Predict multiple horizons simultaneously
let multi_output_config = ModelConfig::new()
    .with_prediction_strategy(PredictionStrategy::MultiOutput)
    .with_horizons(vec![1, 3, 7, 14, 30]); // Multiple forecast horizons

let multi_output_model = Transformer::new(multi_output_config)?;
let multi_forecasts = multi_output_model.predict(&test_data).await?;

// Access forecasts for different horizons
for horizon in multi_forecasts.horizons() {
    let forecast = multi_forecasts.get_horizon_forecast(horizon)?;
    println!("{}-step forecast: {:.2}", horizon, forecast[0]);
}
```

### Ensemble Forecasting

```rust
use neuro_divergent::ensemble::{EnsemblePredictor, CombinationMethod};

// Combine multiple models
let ensemble = EnsemblePredictor::new()
    .add_model("lstm", lstm_model, weight: 0.4)
    .add_model("transformer", transformer_model, weight: 0.3)
    .add_model("nbeats", nbeats_model, weight: 0.3)
    .with_combination_method(CombinationMethod::WeightedAverage);

let ensemble_forecasts = ensemble.predict(&test_data).await?;

// Advanced ensemble methods
let advanced_ensemble = EnsemblePredictor::new()
    .add_models(models)
    .with_combination_method(CombinationMethod::StackedGeneralization)
    .with_meta_learner(Box::new(MLP::new(meta_config)?))
    .with_cross_validation(true);
```

## Real-Time Prediction

### Streaming Predictions

```rust
use neuro_divergent::streaming::{StreamingPredictor, StreamConfig};

// Set up streaming prediction
let stream_config = StreamConfig::new()
    .with_buffer_size(100)
    .with_prediction_interval(Duration::from_secs(60)) // Predict every minute
    .with_batch_processing(true)
    .with_auto_retrain_threshold(0.1); // Retrain if performance drops

let streaming_predictor = StreamingPredictor::new(model)
    .with_config(stream_config);

// Start streaming predictions
let mut prediction_stream = streaming_predictor.start_stream().await?;

while let Some(batch) = prediction_stream.next().await {
    match batch {
        StreamingBatch::Predictions(forecasts) => {
            // Handle new predictions
            for forecast in forecasts {
                println!("Real-time forecast: {:.2}", forecast.value);
                send_to_downstream_system(forecast).await?;
            }
        }
        StreamingBatch::Alert(alert) => {
            // Handle prediction alerts
            println!("Alert: {}", alert.message);
        }
    }
}
```

### Incremental Learning

```rust
use neuro_divergent::incremental::{IncrementalPredictor, UpdateStrategy};

// Predictor that learns from new data
let incremental_predictor = IncrementalPredictor::new(base_model)
    .with_update_strategy(UpdateStrategy::OnlineLearning)
    .with_learning_rate_decay(0.99)
    .with_memory_window(1000)     // Keep last 1000 observations
    .with_adaptation_rate(0.1);   // How quickly to adapt

// Process streaming data with continuous learning
for new_data_point in data_stream {
    // Make prediction
    let prediction = incremental_predictor.predict_one(&new_data_point).await?;
    
    // Update model with new observation (when ground truth becomes available)
    if let Some(actual_value) = new_data_point.actual_value {
        incremental_predictor.update(new_data_point.features, actual_value).await?;
    }
}
```

### Low-Latency Inference

```rust
use neuro_divergent::inference::{FastInference, OptimizationLevel};

// Optimize model for fast inference
let optimized_model = FastInference::optimize(model)
    .with_optimization_level(OptimizationLevel::Aggressive)
    .with_quantization(QuantizationType::Int8)
    .with_pruning(PruningRatio::Light)
    .with_tensorrt_optimization(true)   // NVIDIA TensorRT
    .with_onnx_export(true)            // Export to ONNX
    .build()?;

// Benchmark inference speed
let benchmark = InferenceBenchmark::new()
    .with_batch_sizes(vec![1, 8, 32])
    .with_sequence_lengths(vec![24, 48, 96])
    .with_iterations(1000);

let results = benchmark.run(&optimized_model).await?;
println!("Average inference time: {:.2}ms", results.mean_latency());
```

## Batch Prediction

### Large-Scale Prediction

```rust
use neuro_divergent::batch::{BatchPredictor, BatchConfig};

// Process large datasets efficiently
let batch_config = BatchConfig::new()
    .with_batch_size(128)
    .with_num_workers(8)
    .with_prefetch_factor(2)
    .with_memory_mapping(true)      // Memory-mapped files for large datasets
    .with_progress_tracking(true);

let batch_predictor = BatchPredictor::new(model)
    .with_config(batch_config);

// Predict on large dataset
let large_dataset = TimeSeriesDataset::from_parquet("large_dataset.parquet")?;
let batch_forecasts = batch_predictor.predict_dataset(&large_dataset).await?;

// Save results efficiently
batch_forecasts.save_to_parquet("batch_forecasts.parquet")?;
```

### Parallel Prediction

```rust
use neuro_divergent::parallel::{ParallelPredictor, ParallelConfig};

// Predict multiple series in parallel
let parallel_config = ParallelConfig::new()
    .with_num_workers(num_cpus::get())
    .with_chunk_size(100)
    .with_load_balancing(true);

let parallel_predictor = ParallelPredictor::new(models)
    .with_config(parallel_config);

let parallel_forecasts = parallel_predictor.predict_parallel(&multi_series_data).await?;
```

## Advanced Prediction Techniques

### Hierarchical Forecasting

```rust
use neuro_divergent::hierarchical::{HierarchicalPredictor, ReconciliationMethod};

// Predict at multiple aggregation levels
let hierarchical_predictor = HierarchicalPredictor::new()
    .with_hierarchy_structure(hierarchy_matrix)
    .with_base_forecasts(base_level_forecasts)
    .with_reconciliation_method(ReconciliationMethod::OptimalCombination)
    .with_covariance_estimation(CovarianceMethod::Shrinkage);

let reconciled_forecasts = hierarchical_predictor.reconcile().await?;

// Access forecasts at different levels
let country_forecast = reconciled_forecasts.get_level("country")?;
let state_forecasts = reconciled_forecasts.get_level("state")?;
```

### Cross-Series Forecasting

```rust
use neuro_divergent::cross_series::{CrossSeriesPredictor, DependencyGraph};

// Model dependencies between time series
let dependency_graph = DependencyGraph::new()
    .add_dependency("sales", "advertising", lag: 2)
    .add_dependency("sales", "weather", lag: 0)
    .add_dependency("inventory", "sales", lag: -1); // Leading indicator

let cross_series_predictor = CrossSeriesPredictor::new()
    .with_dependency_graph(dependency_graph)
    .with_base_model(transformer_model);

let cross_series_forecasts = cross_series_predictor.predict(&multi_series_data).await?;
```

### Scenario-Based Forecasting

```rust
use neuro_divergent::scenarios::{ScenarioPredictor, Scenario};

// Generate forecasts for different scenarios
let scenarios = vec![
    Scenario::new("optimistic")
        .with_external_factors(hashmap! {
            "economic_growth" => 0.05,
            "market_conditions" => 1.2,
        }),
    Scenario::new("pessimistic")
        .with_external_factors(hashmap! {
            "economic_growth" => -0.02,
            "market_conditions" => 0.8,
        }),
    Scenario::new("baseline")
        .with_external_factors(hashmap! {
            "economic_growth" => 0.02,
            "market_conditions" => 1.0,
        }),
];

let scenario_predictor = ScenarioPredictor::new(model);
let scenario_forecasts = scenario_predictor.predict_scenarios(&test_data, &scenarios).await?;

// Compare scenarios
for scenario in scenarios {
    let forecast = scenario_forecasts.get_scenario(&scenario.name)?;
    println!("{} scenario: {:.2}", scenario.name, forecast.mean());
}
```

## Prediction Monitoring and Alerts

### Performance Monitoring

```rust
use neuro_divergent::monitoring::{PredictionMonitor, AlertCondition};

// Monitor prediction quality in real-time
let monitor = PredictionMonitor::new()
    .add_alert(AlertCondition::AccuracyDrop(threshold: 0.1))
    .add_alert(AlertCondition::BiasDetection(threshold: 0.05))
    .add_alert(AlertCondition::DistributionShift(threshold: 0.2))
    .add_alert(AlertCondition::LatencyIncrease(threshold: Duration::from_millis(100)))
    .with_notification_channel("slack://alerts-channel")
    .with_dashboard_url("http://monitoring-dashboard/");

// Monitor predictions
monitor.start_monitoring(&streaming_predictor).await?;
```

### Prediction Confidence

```rust
use neuro_divergent::confidence::{ConfidenceEstimator, ConfidenceMethod};

// Estimate prediction confidence
let confidence_estimator = ConfidenceEstimator::new()
    .with_method(ConfidenceMethod::EnsembleVariance)
    .with_calibration_data(&validation_data)
    .with_temperature_scaling(true);

let predictions_with_confidence = confidence_estimator.predict_with_confidence(
    &model, 
    &test_data
).await?;

// Filter low-confidence predictions
let high_confidence_predictions = predictions_with_confidence
    .filter_by_confidence(threshold: 0.8)?;

println!("High confidence predictions: {}/{}", 
         high_confidence_predictions.len(),
         predictions_with_confidence.len());
```

## Prediction Explanation and Interpretability

### Feature Importance for Predictions

```rust
use neuro_divergent::explainability::{PredictionExplainer, ExplanationMethod};

// Explain individual predictions
let explainer = PredictionExplainer::new()
    .with_method(ExplanationMethod::SHAP)
    .with_baseline_data(&train_data)
    .with_feature_names(feature_names);

let explanation = explainer.explain_prediction(&model, &single_prediction_data).await?;

// Visualize feature contributions
explanation.plot_feature_importance("feature_importance.png")?;
explanation.plot_waterfall("prediction_waterfall.png")?;

// Get top contributing features
let top_features = explanation.get_top_features(n: 10);
for (feature, importance) in top_features {
    println!("{}: {:.4}", feature, importance);
}
```

### Attention Visualization

```rust
// For attention-based models (Transformers)
let attention_weights = transformer_model.get_attention_weights(&test_data)?;

// Visualize attention patterns
attention_weights.plot_attention_heatmap("attention_heatmap.png")?;
attention_weights.plot_attention_rollout("attention_rollout.png")?;

// Identify important time steps
let important_timesteps = attention_weights.get_high_attention_timesteps(threshold: 0.1);
println!("Important timesteps: {:?}", important_timesteps);
```

## Error Handling and Robustness

### Prediction Error Handling

```rust
use neuro_divergent::error_handling::{PredictionErrorHandler, FallbackStrategy};

// Robust prediction with error handling
let error_handler = PredictionErrorHandler::new()
    .with_fallback_strategy(FallbackStrategy::LastKnownValue)
    .with_retry_attempts(3)
    .with_timeout(Duration::from_secs(30))
    .with_input_validation(true)
    .with_output_validation(true);

let robust_predictor = RobustPredictor::new(model)
    .with_error_handler(error_handler);

match robust_predictor.predict(&test_data).await {
    Ok(forecasts) => {
        // Handle successful predictions
        process_forecasts(forecasts)?;
    }
    Err(PredictionError::InputValidationFailed(msg)) => {
        // Handle input validation errors
        log::error!("Input validation failed: {}", msg);
    }
    Err(PredictionError::ModelInferenceFailed(msg)) => {
        // Handle model inference errors
        log::error!("Model inference failed: {}", msg);
        use_fallback_model()?;
    }
}
```

### Input Validation

```rust
use neuro_divergent::validation::{PredictionInputValidator, ValidationRule};

// Validate prediction inputs
let input_validator = PredictionInputValidator::new()
    .add_rule(ValidationRule::RequiredColumns(vec!["timestamp", "value"]))
    .add_rule(ValidationRule::NoMissingValues)
    .add_rule(ValidationRule::ValidTimeRange(start_date, end_date))
    .add_rule(ValidationRule::FrequencyConsistency)
    .add_rule(ValidationRule::ValueRangeCheck(min: 0.0, max: 1000000.0));

if let Err(validation_errors) = input_validator.validate(&test_data) {
    for error in validation_errors {
        log::warn!("Input validation error: {}", error);
    }
    return Err("Invalid input data".into());
}
```

## Performance Optimization

### Prediction Caching

```rust
use neuro_divergent::caching::{PredictionCache, CacheStrategy};

// Cache predictions for repeated queries
let cache = PredictionCache::new()
    .with_strategy(CacheStrategy::LRU)
    .with_max_size(1000)
    .with_ttl(Duration::from_minutes(30))
    .with_invalidation_on_model_update(true);

let cached_predictor = CachedPredictor::new(model)
    .with_cache(cache);

// Predictions are automatically cached and retrieved
let forecasts = cached_predictor.predict(&test_data).await?;
```

### Model Quantization

```rust
use neuro_divergent::quantization::{ModelQuantizer, QuantizationConfig};

// Quantize model for faster inference
let quantization_config = QuantizationConfig::new()
    .with_precision(Precision::Int8)
    .with_calibration_data(&calibration_data)
    .with_preserve_accuracy(true);

let quantized_model = ModelQuantizer::quantize(model, &quantization_config)?;

// Benchmark quantized vs original model
let benchmark_results = compare_inference_speed(model, quantized_model).await?;
println!("Speedup: {:.2}x, Accuracy loss: {:.4}", 
         benchmark_results.speedup_factor,
         benchmark_results.accuracy_degradation);
```

## Production Deployment

### Prediction API

```rust
use neuro_divergent::api::{PredictionAPI, APIConfig};
use warp::Filter;

// Create prediction API
let api_config = APIConfig::new()
    .with_max_batch_size(100)
    .with_timeout(Duration::from_secs(30))
    .with_rate_limiting(100, Duration::from_secs(60)) // 100 requests per minute
    .with_authentication_required(true);

let prediction_api = PredictionAPI::new(model)
    .with_config(api_config);

// Define API routes
let predict_route = warp::path("predict")
    .and(warp::post())
    .and(warp::body::json())
    .and_then(move |data: TimeSeriesData| {
        let api = prediction_api.clone();
        async move {
            match api.predict(data).await {
                Ok(forecasts) => Ok(warp::reply::json(&forecasts)),
                Err(e) => Err(warp::reject::custom(e)),
            }
        }
    });

// Start server
warp::serve(predict_route)
    .run(([127, 0, 0, 1], 3030))
    .await;
```

### Model Serving Infrastructure

```rust
use neuro_divergent::serving::{ModelServer, ServingConfig};

// Production model serving
let serving_config = ServingConfig::new()
    .with_auto_scaling(true)
    .with_health_checks(true)
    .with_metrics_collection(true)
    .with_load_balancing(LoadBalancer::RoundRobin)
    .with_gpu_acceleration(true);

let model_server = ModelServer::new()
    .add_model("lstm_v1", lstm_model)
    .add_model("transformer_v2", transformer_model)
    .with_config(serving_config);

model_server.start().await?;
```

## Best Practices

### Prediction Best Practices Checklist

- [ ] **Validate inputs** before prediction
- [ ] **Handle missing values** appropriately
- [ ] **Monitor prediction quality** in real-time
- [ ] **Implement fallback strategies** for failures
- [ ] **Cache frequent predictions** for performance
- [ ] **Use appropriate prediction intervals** for uncertainty
- [ ] **Document prediction assumptions** and limitations
- [ ] **Test edge cases** and error conditions

### Common Prediction Pitfalls

1. **Extrapolation Beyond Training Range**: Models may fail on unseen data ranges
2. **Seasonal Misalignment**: Wrong seasonal assumptions in prediction horizon
3. **Feature Leakage**: Using future information inadvertently
4. **Overconfidence**: Not accounting for model uncertainty
5. **Batch Size Inconsistency**: Different behavior between training and prediction batch sizes

### Performance Considerations

```rust
// Optimize for different use cases
match use_case {
    UseCase::RealTimeTrading => {
        // Prioritize latency
        optimize_for_latency(&mut model)?;
    }
    UseCase::BatchReporting => {
        // Prioritize throughput
        optimize_for_throughput(&mut model)?;
    }
    UseCase::EdgeDevice => {
        // Prioritize memory usage
        optimize_for_memory(&mut model)?;
    }
    UseCase::HighAccuracy => {
        // Prioritize accuracy
        use_ensemble_prediction(&mut model)?;
    }
}
```

## Next Steps

Now that you understand prediction techniques:

1. **Evaluate Predictions**: Learn [Evaluation Methods](evaluation.md) to assess forecast quality
2. **Optimize Performance**: Explore [Performance Guide](performance.md) for speed and memory optimization
3. **Deploy to Production**: Review [Best Practices](best-practices.md) for production deployment
4. **Handle Problems**: Check [Troubleshooting](troubleshooting.md) for common issues

Remember: good predictions require understanding your data, choosing appropriate models, and properly handling uncertainty. Always validate your predictions against business requirements and real-world constraints.