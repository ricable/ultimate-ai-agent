# Evaluation and Metrics

The evaluation system provides comprehensive metrics and tools for assessing forecasting model performance, including point forecasts, probabilistic forecasts, and cross-validation strategies specifically designed for time series data.

## Overview

The evaluation system includes:

- **Point Forecast Metrics**: Traditional accuracy measures
- **Probabilistic Metrics**: Uncertainty quantification assessment
- **Cross-Validation**: Time series-aware validation strategies
- **Model Comparison**: Statistical tests and ranking methods
- **Custom Metrics**: Extensible metric framework

## Point Forecast Metrics

### Mean Absolute Error (MAE)

Measures average absolute difference between predictions and actual values.

```rust
use neuro_divergent::metrics::MAE;

let mae = MAE::new();
let predictions = vec![1.1, 2.2, 3.3];
let actuals = vec![1.0, 2.0, 3.0];
let error = mae.compute(&predictions, &actuals)?;
println!("MAE: {:.4}", error);  // 0.1000
```

### Mean Squared Error (MSE)

Squared differences, penalizes larger errors more heavily.

```rust
use neuro_divergent::metrics::MSE;

let mse = MSE::new();
let error = mse.compute(&predictions, &actuals)?;
println!("MSE: {:.4}", error);
```

### Root Mean Squared Error (RMSE)

Square root of MSE, in same units as the data.

```rust
use neuro_divergent::metrics::RMSE;

let rmse = RMSE::new();
let error = rmse.compute(&predictions, &actuals)?;
println!("RMSE: {:.4}", error);
```

### Mean Absolute Percentage Error (MAPE)

Percentage-based error, good for interpretability.

```rust
use neuro_divergent::metrics::MAPE;

let mape = MAPE::new();
let error = mape.compute(&predictions, &actuals)?;
println!("MAPE: {:.2}%", error);

// MAPE with minimum threshold to avoid division by very small numbers
let robust_mape = MAPE::builder()
    .min_threshold(0.01)
    .build();
```

### Symmetric Mean Absolute Percentage Error (SMAPE)

Symmetric version of MAPE, bounded between 0 and 200%.

```rust
use neuro_divergent::metrics::SMAPE;

let smape = SMAPE::new();
let error = smape.compute(&predictions, &actuals)?;
println!("SMAPE: {:.2}%", error);
```

### Mean Absolute Scaled Error (MASE)

Scale-free metric comparing against naive seasonal forecast.

```rust
use neuro_divergent::metrics::MASE;

// MASE requires training data for scaling
let mase = MASE::builder()
    .training_data(&train_actuals)
    .seasonality(12)  // Monthly data
    .build()?;

let error = mase.compute(&predictions, &actuals)?;
println!("MASE: {:.4}", error);  // <1 is better than seasonal naive
```

## Probabilistic Forecast Metrics

### Continuous Ranked Probability Score (CRPS)

Measures quality of probabilistic forecasts.

```rust
use neuro_divergent::metrics::CRPS;

// For ensemble forecasts
let ensemble_forecasts = vec![
    vec![1.0, 1.1, 0.9],  // Multiple samples for each time step
    vec![2.0, 2.2, 1.8],
    vec![3.0, 3.1, 2.9],
];
let actuals = vec![1.05, 2.1, 3.05];

let crps = CRPS::new();
let score = crps.compute_ensemble(&ensemble_forecasts, &actuals)?;
println!("CRPS: {:.4}", score);

// For quantile forecasts
let quantile_forecasts = vec![
    vec![0.8, 1.0, 1.2],  // [q10, q50, q90] for each time step
    vec![1.8, 2.0, 2.2],
    vec![2.8, 3.0, 3.2],
];
let quantiles = vec![0.1, 0.5, 0.9];

let score = crps.compute_quantiles(&quantile_forecasts, &quantiles, &actuals)?;
```

### Pinball Loss

Asymmetric loss function for quantile forecasts.

```rust
use neuro_divergent::metrics::PinballLoss;

// For 10th percentile
let pinball_10 = PinballLoss::new(0.1);
let quantile_predictions = vec![0.8, 1.8, 2.8];  // 10th percentile forecasts
let loss = pinball_10.compute(&quantile_predictions, &actuals)?;

// For multiple quantiles
let quantiles = vec![0.1, 0.5, 0.9];
let mut total_loss = 0.0;
for (i, &quantile) in quantiles.iter().enumerate() {
    let pinball = PinballLoss::new(quantile);
    let q_predictions: Vec<f64> = quantile_forecasts.iter()
        .map(|qf| qf[i])
        .collect();
    total_loss += pinball.compute(&q_predictions, &actuals)?;
}
println!("Average Pinball Loss: {:.4}", total_loss / quantiles.len() as f64);
```

### Coverage Rate

Measures how often actual values fall within prediction intervals.

```rust
use neuro_divergent::metrics::CoverageRate;

// 80% prediction intervals
let lower_bounds = vec![0.5, 1.5, 2.5];
let upper_bounds = vec![1.5, 2.5, 3.5];
let actuals = vec![1.05, 2.1, 3.05];

let coverage = CoverageRate::new(0.8);  // Target 80% coverage
let actual_coverage = coverage.compute(&lower_bounds, &upper_bounds, &actuals)?;
println!("Coverage Rate: {:.1}%", actual_coverage * 100.0);

// Should be close to 80% for well-calibrated intervals
```

### Interval Width

Measures average width of prediction intervals.

```rust
use neuro_divergent::metrics::IntervalWidth;

let width_metric = IntervalWidth::new();
let avg_width = width_metric.compute(&lower_bounds, &upper_bounds)?;
println!("Average Interval Width: {:.4}", avg_width);

// Narrower intervals are better if coverage is maintained
```

## Time Series Specific Metrics

### Directional Accuracy

Measures how often the forecast correctly predicts direction of change.

```rust
use neuro_divergent::metrics::DirectionalAccuracy;

let past_values = vec![1.0, 2.0, 3.0];
let predictions = vec![2.1, 3.1, 2.9];  // Predict up, up, down
let actuals = vec![2.05, 3.05, 2.95];   // Actual up, up, down

let dir_acc = DirectionalAccuracy::new();
let accuracy = dir_acc.compute(&past_values, &predictions, &actuals)?;
println!("Directional Accuracy: {:.1}%", accuracy * 100.0);
```

### Theil's U Statistic

Compares forecast accuracy against naive methods.

```rust
use neuro_divergent::metrics::TheilU;

let theil_u = TheilU::new();
let u_statistic = theil_u.compute(&predictions, &actuals, &past_values)?;
println!("Theil's U: {:.4}", u_statistic);  // <1 is better than naive
```

### Relative metrics

Compare model performance against benchmarks.

```rust
use neuro_divergent::metrics::{RelativeMAE, RelativeRMSE};

// Compare against seasonal naive
let seasonal_naive_predictions = generate_seasonal_naive(&past_values, 12)?;

let rel_mae = RelativeMAE::new();
let relative_error = rel_mae.compute(
    &predictions, 
    &actuals, 
    &seasonal_naive_predictions
)?;
println!("Relative MAE: {:.4}", relative_error);  // <1 is better than baseline
```

## Cross-Validation

### Time Series Cross-Validation

```rust
use neuro_divergent::evaluation::{TimeSeriesCrossValidator, CrossValidationConfig};

// Configure time series CV
let cv_config = CrossValidationConfig::builder()
    .n_splits(5)
    .horizon(12)
    .step_size(6)     // 50% overlap
    .min_train_size(100)
    .build()?;

let cv = TimeSeriesCrossValidator::new(cv_config);

// Perform cross-validation
let cv_results = cv.evaluate(&mut model, &data)?;

// Access results
for (fold, result) in cv_results.iter().enumerate() {
    println!("Fold {}: MAE = {:.4}, MAPE = {:.2}%", 
             fold, result.mae, result.mape);
}

// Overall statistics
let overall_mae = cv_results.mean_mae();
let mae_std = cv_results.std_mae();
println!("Overall MAE: {:.4} ± {:.4}", overall_mae, mae_std);
```

### Expanding Window Cross-Validation

```rust
use neuro_divergent::evaluation::ExpandingWindowCV;

// Expanding window (increasing training set size)
let expanding_cv = ExpandingWindowCV::builder()
    .initial_train_size(200)
    .horizon(12)
    .step_size(12)
    .max_splits(10)
    .build()?;

let results = expanding_cv.evaluate(&mut model, &data)?;
```

### Rolling Window Cross-Validation

```rust
use neuro_divergent::evaluation::RollingWindowCV;

// Rolling window (fixed training set size)
let rolling_cv = RollingWindowCV::builder()
    .window_size(300)
    .horizon(12)
    .step_size(6)
    .build()?;

let results = rolling_cv.evaluate(&mut model, &data)?;
```

## Model Comparison

### Statistical Significance Testing

```rust
use neuro_divergent::evaluation::{DieboldMarianoTest, WilcoxonTest};

// Diebold-Mariano test for forecast accuracy comparison
let dm_test = DieboldMarianoTest::new();
let dm_statistic = dm_test.compare(
    &model1_errors, 
    &model2_errors,
    1  // forecast horizon
)?;

if dm_statistic.p_value < 0.05 {
    println!("Significant difference between models (p = {:.4})", dm_statistic.p_value);
}

// Wilcoxon signed-rank test (non-parametric)
let wilcoxon = WilcoxonTest::new();
let w_statistic = wilcoxon.compare(&model1_errors, &model2_errors)?;
```

### Model Ranking

```rust
use neuro_divergent::evaluation::ModelRanking;

// Rank models by multiple metrics
let ranking = ModelRanking::builder()
    .add_metric("MAE", &mae_scores)
    .add_metric("MAPE", &mape_scores)
    .add_metric("MASE", &mase_scores)
    .weights(vec![0.4, 0.3, 0.3])  // Weighted combination
    .build()?;

let ranked_models = ranking.rank(&model_names)?;
for (rank, (model, score)) in ranked_models.iter().enumerate() {
    println!("{}. {}: {:.4}", rank + 1, model, score);
}
```

## Custom Metrics

### Implementing Custom Metrics

```rust
use neuro_divergent::metrics::{Metric, MetricResult};

// Custom metric example: Weighted MAPE
struct WeightedMAPE {
    weights: Vec<f64>,
}

impl Metric<f64> for WeightedMAPE {
    fn name(&self) -> &str {
        "weighted_mape"
    }
    
    fn compute(&self, predictions: &[f64], actuals: &[f64]) -> NeuroDivergentResult<f64> {
        if predictions.len() != actuals.len() || predictions.len() != self.weights.len() {
            return Err(NeuroDivergentError::data("Length mismatch"));
        }
        
        let mut weighted_error = 0.0;
        let mut total_weight = 0.0;
        
        for ((pred, actual), weight) in predictions.iter()
            .zip(actuals.iter())
            .zip(self.weights.iter()) {
            
            if actual.abs() > 1e-8 {
                let percentage_error = (pred - actual).abs() / actual.abs();
                weighted_error += percentage_error * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            Ok(weighted_error / total_weight * 100.0)
        } else {
            Err(NeuroDivergentError::math("All targets are zero"))
        }
    }
}

// Usage
let weights = vec![1.0, 2.0, 3.0];  // More weight on recent forecasts
let weighted_mape = WeightedMAPE { weights };
let error = weighted_mape.compute(&predictions, &actuals)?;
```

### Business-Specific Metrics

```rust
// Inventory-specific metric
struct ServiceLevel {
    stockout_cost: f64,
    holding_cost: f64,
}

impl Metric<f64> for ServiceLevel {
    fn name(&self) -> &str {
        "service_level"
    }
    
    fn compute(&self, predictions: &[f64], actuals: &[f64]) -> NeuroDivergentResult<f64> {
        let mut total_cost = 0.0;
        
        for (pred, actual) in predictions.iter().zip(actuals.iter()) {
            let error = pred - actual;
            if error < 0.0 {
                // Stockout (underforecast)
                total_cost += error.abs() * self.stockout_cost;
            } else {
                // Overstock (overforecast)
                total_cost += error * self.holding_cost;
            }
        }
        
        Ok(total_cost / predictions.len() as f64)
    }
}
```

## Evaluation Pipelines

### Comprehensive Model Evaluation

```rust
use neuro_divergent::evaluation::ModelEvaluator;

// Set up comprehensive evaluation
let evaluator = ModelEvaluator::builder()
    .add_metric(Box::new(MAE::new()))
    .add_metric(Box::new(RMSE::new()))
    .add_metric(Box::new(MAPE::new()))
    .add_metric(Box::new(MASE::builder().seasonality(12).build()?))
    .add_probabilistic_metric(Box::new(CRPS::new()))
    .cross_validation_config(cv_config)
    .significance_level(0.05)
    .build()?;

// Evaluate single model
let results = evaluator.evaluate_model(&mut model, &test_data)?;
println!("Evaluation Results:");
for (metric, value) in results.metrics {
    println!("  {}: {:.4}", metric, value);
}

// Compare multiple models
let models = vec![lstm_model, nbeats_model, transformer_model];
let comparison = evaluator.compare_models(models, &test_data)?;

// Best model by each metric
for (metric, best_model) in comparison.best_by_metric {
    println!("Best {} model: {}", metric, best_model);
}
```

### Backtesting Framework

```rust
use neuro_divergent::evaluation::Backtester;

// Set up backtesting
let backtester = Backtester::builder()
    .start_date("2020-01-01".parse()?)
    .end_date("2023-12-31".parse()?)
    .retrain_frequency(30)  // Retrain every 30 days
    .forecast_horizon(7)
    .metrics(vec![
        Box::new(MAE::new()),
        Box::new(MAPE::new()),
        Box::new(DirectionalAccuracy::new()),
    ])
    .build()?;

// Run backtest
let backtest_results = backtester.run(&mut model, &historical_data)?;

// Analyze results over time
for result in &backtest_results.period_results {
    println!("Period {}: MAE = {:.4}", result.period, result.mae);
}

// Overall performance
println!("Overall MAE: {:.4}", backtest_results.overall_mae());
println!("Best Period: {}", backtest_results.best_period());
println!("Worst Period: {}", backtest_results.worst_period());
```

### Residual Analysis

```rust
use neuro_divergent::evaluation::ResidualAnalysis;

// Analyze forecast residuals
let residuals: Vec<f64> = predictions.iter()
    .zip(actuals.iter())
    .map(|(pred, actual)| pred - actual)
    .collect();

let residual_analysis = ResidualAnalysis::new(&residuals)?;

// Statistical tests
println!("Normality test p-value: {:.4}", residual_analysis.normality_test_p_value());
println!("Autocorrelation (lag 1): {:.4}", residual_analysis.autocorrelation(1)?);
println!("Mean residual: {:.4}", residual_analysis.mean());
println!("Residual std: {:.4}", residual_analysis.std());

// Check for bias
if residual_analysis.mean().abs() > 0.01 {
    println!("Warning: Model appears to be biased");
}

// Check for heteroscedasticity
let bp_test = residual_analysis.breusch_pagan_test()?;
if bp_test.p_value < 0.05 {
    println!("Warning: Heteroscedasticity detected");
}
```

## Evaluation Best Practices

### Metric Selection Guidelines

```rust
fn select_metrics(data_characteristics: &DataCharacteristics) -> Vec<Box<dyn Metric<f64>>> {
    let mut metrics: Vec<Box<dyn Metric<f64>>> = Vec::new();
    
    // Always include MAE and RMSE
    metrics.push(Box::new(MAE::new()));
    metrics.push(Box::new(RMSE::new()));
    
    // Add percentage errors if appropriate
    if data_characteristics.min_value > 0.0 {
        metrics.push(Box::new(MAPE::new()));
    } else {
        metrics.push(Box::new(SMAPE::new()));
    }
    
    // Add scaled error if seasonal
    if data_characteristics.is_seasonal {
        metrics.push(Box::new(MASE::builder()
            .seasonality(data_characteristics.seasonal_period)
            .build().unwrap()));
    }
    
    // Add directional accuracy for trending data
    if data_characteristics.has_trend {
        metrics.push(Box::new(DirectionalAccuracy::new()));
    }
    
    // Add business-specific metrics
    if let Some(business_metric) = &data_characteristics.business_metric {
        metrics.push(business_metric.clone());
    }
    
    metrics
}
```

### Statistical Validity

```rust
// Ensure sufficient test set size
fn validate_test_size(test_size: usize, horizon: usize) -> NeuroDivergentResult<()> {
    let min_size = horizon * 10;  // At least 10 forecast periods
    if test_size < min_size {
        return Err(NeuroDivergentError::data(
            format!("Test set too small: {} < {}", test_size, min_size)
        ));
    }
    Ok(())
}

// Multiple forecast origins for robust evaluation
fn robust_evaluation(
    model: &mut dyn BaseModel<f64>,
    data: &TimeSeriesDataFrame<f64>,
    horizon: usize,
    n_origins: usize,
) -> NeuroDivergentResult<Vec<f64>> {
    let mut all_errors = Vec::new();
    
    let step_size = (data.shape().0 - horizon) / n_origins;
    
    for i in 0..n_origins {
        let split_point = data.shape().0 - horizon - i * step_size;
        let (train, test) = data.split_at_index(split_point)?;
        
        model.fit(&train)?;
        let forecasts = model.predict(&test)?;
        
        let mae = MAE::new().compute(&forecasts.values(), &test.target_values())?;
        all_errors.push(mae);
    }
    
    Ok(all_errors)
}
```

The evaluation system provides comprehensive tools for assessing forecasting model performance across multiple dimensions, ensuring robust and reliable model selection and deployment decisions.