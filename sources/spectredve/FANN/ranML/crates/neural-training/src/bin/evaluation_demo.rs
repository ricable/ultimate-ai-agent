//! Comprehensive model evaluation demonstration
//! 
//! This binary demonstrates the enhanced evaluation capabilities including:
//! - Advanced cross-validation techniques
//! - Statistical significance testing
//! - Residual analysis
//! - Performance profiling
//! - Model comparison and selection

use anyhow::Result;
use neural_training::{
    data::{TelecomDataLoader, TelecomDataset},
    evaluation::{ModelEvaluator, ModelComparisonReport, EvaluationMetrics},
    models::NeuralModel,
    training::ModelTrainingResult,
};
use std::time::Instant;
use clap::{Arg, Command};
use log::{info, warn, error};

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    let matches = Command::new("evaluation-demo")
        .version("1.0")
        .about("Neural Network Evaluation Framework Demo")
        .arg(Arg::new("data-path")
            .short('d')
            .long("data")
            .value_name("FILE")
            .help("Path to telecom dataset CSV file")
            .required(true))
        .arg(Arg::new("models")
            .short('m')
            .long("models")
            .value_name("NUMBER")
            .help("Number of models to evaluate")
            .default_value("3"))
        .arg(Arg::new("cross-validation")
            .short('k')
            .long("cv-folds")
            .value_name("FOLDS")
            .help("Number of cross-validation folds")
            .default_value("5"))
        .arg(Arg::new("confidence-level")
            .short('c')
            .long("confidence")
            .value_name("LEVEL")
            .help("Confidence level for statistical tests (0.0-1.0)")
            .default_value("0.95"))
        .arg(Arg::new("parallel")
            .short('p')
            .long("parallel")
            .help("Enable parallel evaluation")
            .action(clap::ArgAction::SetTrue))
        .arg(Arg::new("comprehensive")
            .long("comprehensive")
            .help("Run comprehensive evaluation with all advanced techniques")
            .action(clap::ArgAction::SetTrue))
        .arg(Arg::new("benchmark")
            .short('b')
            .long("benchmark")
            .help("Run performance benchmarks")
            .action(clap::ArgAction::SetTrue))
        .get_matches();

    // Parse arguments
    let data_path = matches.get_one::<String>("data-path").unwrap();
    let num_models: usize = matches.get_one::<String>("models").unwrap().parse()?;
    let cv_folds: usize = matches.get_one::<String>("cross-validation").unwrap().parse()?;
    let confidence_level: f32 = matches.get_one::<String>("confidence-level").unwrap().parse()?;
    let enable_parallel = matches.get_flag("parallel");
    let comprehensive = matches.get_flag("comprehensive");
    let benchmark = matches.get_flag("benchmark");

    info!("Starting Neural Network Evaluation Demo");
    info!("Data path: {}", data_path);
    info!("Number of models: {}", num_models);
    info!("CV folds: {}", cv_folds);
    info!("Confidence level: {}", confidence_level);
    info!("Parallel evaluation: {}", enable_parallel);
    info!("Comprehensive mode: {}", comprehensive);

    // Load and prepare data
    info!("Loading telecom dataset...");
    let dataset = TelecomDataLoader::load(data_path)?;
    info!("Dataset loaded: {} samples, {} features", 
          dataset.features.nrows(), dataset.features.ncols());

    // Split data
    let data_split = dataset.split_train_test(0.8)?;
    info!("Train samples: {}, Test samples: {}", 
          data_split.train.features.nrows(), 
          data_split.test.features.nrows());

    // Create evaluator with configuration
    let evaluator = ModelEvaluator::new()
        .with_config(enable_parallel, confidence_level, Some(42));

    // Create multiple models for comparison
    let models = create_demo_models(num_models)?;
    info!("Created {} models for evaluation", models.len());

    // Run evaluation based on mode
    if comprehensive {
        run_comprehensive_evaluation(&evaluator, models, &data_split, cv_folds).await?;
    } else {
        run_standard_evaluation(&evaluator, models, &data_split, cv_folds).await?;
    }

    if benchmark {
        run_performance_benchmarks(&evaluator, &data_split).await?;
    }

    info!("Evaluation demo completed successfully!");
    Ok(())
}

/// Create demonstration models with different architectures
fn create_demo_models(num_models: usize) -> Result<Vec<NeuralModel>> {
    let mut models = Vec::new();
    
    for i in 0..num_models {
        let model = match i % 3 {
            0 => {
                // Simple model
                let mut model = NeuralModel::new(format!("simple_model_{}", i));
                model.layers = vec![22, 16, 8, 1]; // Input -> Hidden -> Output
                model
            },
            1 => {
                // Medium complexity model
                let mut model = NeuralModel::new(format!("medium_model_{}", i));
                model.layers = vec![22, 32, 16, 8, 1];
                model
            },
            2 => {
                // Complex model
                let mut model = NeuralModel::new(format!("complex_model_{}", i));
                model.layers = vec![22, 64, 32, 16, 8, 1];
                model
            },
            _ => unreachable!(),
        };
        
        models.push(model);
    }
    
    Ok(models)
}

/// Run comprehensive evaluation with all advanced techniques
async fn run_comprehensive_evaluation(
    evaluator: &ModelEvaluator,
    mut models: Vec<NeuralModel>,
    data_split: &neural_training::data::DataSplit,
    cv_folds: usize,
) -> Result<()> {
    info!("=== COMPREHENSIVE EVALUATION ===");
    
    let mut training_results = Vec::new();
    
    for (i, model) in models.iter_mut().enumerate() {
        info!("\n--- Evaluating Model {}: {} ---", i + 1, model.name);
        
        let start_time = Instant::now();
        
        // Comprehensive evaluation
        let report = evaluator.evaluate_comprehensive(model, &data_split.test)?;
        
        let evaluation_time = start_time.elapsed();
        
        // Display detailed results
        display_comprehensive_results(&report, evaluation_time);
        
        // Create training result for comparison
        let training_result = ModelTrainingResult {
            model_name: model.name.clone(),
            final_loss: report.evaluation_metrics.mse,
            training_time: evaluation_time,
            epochs_completed: 100, // Placeholder
            convergence_achieved: true,
            validation_results: Some(neural_training::training::ValidationResults {
                predictions: vec![0.0; data_split.test.targets.len()], // Placeholder
                targets: data_split.test.targets.to_vec(),
                loss_history: vec![report.evaluation_metrics.mse],
            }),
        };
        
        training_results.push(training_result);
    }
    
    // Model comparison
    info!("\n=== MODEL COMPARISON REPORT ===");
    let comparison_report = ModelEvaluator::compare_models(&training_results)?;
    display_comparison_report(&comparison_report);
    
    // Statistical analysis
    perform_statistical_analysis(&training_results)?;
    
    Ok(())
}

/// Run standard evaluation
async fn run_standard_evaluation(
    evaluator: &ModelEvaluator,
    mut models: Vec<NeuralModel>,
    data_split: &neural_training::data::DataSplit,
    cv_folds: usize,
) -> Result<()> {
    info!("=== STANDARD EVALUATION ===");
    
    let mut training_results = Vec::new();
    
    for (i, model) in models.iter_mut().enumerate() {
        info!("\n--- Evaluating Model {}: {} ---", i + 1, model.name);
        
        let start_time = Instant::now();
        
        // Basic evaluation
        let metrics = evaluator.evaluate_model(model, &data_split.test)?;
        
        // Cross-validation
        let cv_results = evaluator.cross_validate_enhanced(model.clone(), &data_split.test, cv_folds)?;
        
        let evaluation_time = start_time.elapsed();
        
        // Display results
        display_standard_results(&metrics, &cv_results, evaluation_time);
        
        // Create training result for comparison
        let training_result = ModelTrainingResult {
            model_name: model.name.clone(),
            final_loss: metrics.mse,
            training_time: evaluation_time,
            epochs_completed: 100, // Placeholder
            convergence_achieved: true,
            validation_results: Some(neural_training::training::ValidationResults {
                predictions: vec![0.0; data_split.test.targets.len()], // Placeholder
                targets: data_split.test.targets.to_vec(),
                loss_history: vec![metrics.mse],
            }),
        };
        
        training_results.push(training_result);
    }
    
    // Model comparison
    info!("\n=== MODEL COMPARISON REPORT ===");
    let comparison_report = ModelEvaluator::compare_models(&training_results)?;
    display_comparison_report(&comparison_report);
    
    Ok(())
}

/// Display comprehensive evaluation results
fn display_comprehensive_results(report: &neural_training::evaluation::ModelEvaluationReport, evaluation_time: std::time::Duration) {
    let metrics = &report.evaluation_metrics;
    
    println!("\n📊 COMPREHENSIVE EVALUATION RESULTS");
    println!("════════════════════════════════════");
    println!("Model: {}", report.model_name);
    println!("Evaluation Time: {:.2}s", evaluation_time.as_secs_f32());
    
    println!("\n📈 Core Metrics:");
    println!("  MSE:              {:.6}", metrics.mse);
    println!("  RMSE:             {:.6}", metrics.rmse);
    println!("  MAE:              {:.6}", metrics.mae);
    println!("  R²:               {:.6}", metrics.r_squared);
    println!("  Adjusted R²:      {:.6}", metrics.adjusted_r_squared);
    println!("  Correlation:      {:.6}", metrics.correlation);
    
    println!("\n📉 Error Analysis:");
    println!("  MAPE:             {:.2}%", metrics.mape);
    println!("  SMAPE:            {:.2}%", metrics.smape);
    println!("  Max Error:        {:.6}", metrics.max_error);
    println!("  Median AE:        {:.6}", metrics.median_absolute_error);
    println!("  Directional Acc:  {:.2}%", metrics.directional_accuracy);
    
    println!("\n🎯 Model Selection Criteria:");
    println!("  AIC:              {:.2}", metrics.aic);
    println!("  BIC:              {:.2}", metrics.bic);
    println!("  Theil's U:        {:.6}", metrics.theil_u);
    println!("  Explained Var:    {:.6}", metrics.explained_variance);
    
    println!("\n🔬 Residual Analysis:");
    println!("  Mean Residuals:   {:.6}", metrics.mean_residuals);
    println!("  Std Residuals:    {:.6}", metrics.std_residuals);
    println!("  PI Coverage:      {:.2}%", metrics.prediction_interval_coverage);
    
    println!("\n📊 Cross-Validation:");
    let cv = &report.cross_validation;
    println!("  CV Score:         {:.6} ± {:.6}", cv.cv_score, cv.cv_score_std);
    println!("  Best Fold:        #{}", cv.best_fold + 1);
    println!("  Worst Fold:       #{}", cv.worst_fold + 1);
    
    println!("\n⚡ Performance Profile:");
    let perf = &report.performance_profile;
    println!("  Throughput:       {:.0} samples/sec", perf.throughput_samples_per_second);
    println!("  Per Sample:       {:.3}ms", perf.prediction_time_per_sample.as_millis());
    println!("  Memory Usage:     {} bytes", perf.memory_usage);
    
    if !report.recommendations.is_empty() {
        println!("\n💡 Recommendations:");
        for (i, rec) in report.recommendations.iter().enumerate() {
            println!("  {}. {}", i + 1, rec);
        }
    }
}

/// Display standard evaluation results
fn display_standard_results(
    metrics: &EvaluationMetrics,
    cv_results: &neural_training::evaluation::CrossValidationResults,
    evaluation_time: std::time::Duration,
) {
    println!("\n📊 STANDARD EVALUATION RESULTS");
    println!("═══════════════════════════════");
    println!("Model: {}", cv_results.model_name);
    println!("Evaluation Time: {:.2}s", evaluation_time.as_secs_f32());
    
    println!("\n📈 Metrics:");
    println!("  MSE:              {:.6}", metrics.mse);
    println!("  RMSE:             {:.6}", metrics.rmse);
    println!("  MAE:              {:.6}", metrics.mae);
    println!("  R²:               {:.6}", metrics.r_squared);
    println!("  MAPE:             {:.2}%", metrics.mape);
    println!("  Correlation:      {:.6}", metrics.correlation);
    
    println!("\n📊 Cross-Validation ({} folds):", cv_results.fold_results.len());
    println!("  CV Score:         {:.6} ± {:.6}", cv_results.cv_score, cv_results.cv_score_std);
    println!("  Best Fold:        #{} (R² = {:.6})", 
             cv_results.best_fold + 1, 
             cv_results.fold_results[cv_results.best_fold].metrics.r_squared);
    println!("  Worst Fold:       #{} (R² = {:.6})", 
             cv_results.worst_fold + 1, 
             cv_results.fold_results[cv_results.worst_fold].metrics.r_squared);
    
    println!("\n🔍 Confidence Intervals ({}% level):", (cv_results.confidence_intervals.confidence_level * 100.0) as u8);
    println!("  MSE:              [{:.6}, {:.6}]", 
             cv_results.confidence_intervals.mse_ci.0, 
             cv_results.confidence_intervals.mse_ci.1);
    println!("  R²:               [{:.6}, {:.6}]", 
             cv_results.confidence_intervals.r_squared_ci.0, 
             cv_results.confidence_intervals.r_squared_ci.1);
    println!("  MAE:              [{:.6}, {:.6}]", 
             cv_results.confidence_intervals.mae_ci.0, 
             cv_results.confidence_intervals.mae_ci.1);
}

/// Display model comparison report
fn display_comparison_report(report: &ModelComparisonReport) {
    println!("\n🏆 MODEL COMPARISON REPORT");
    println!("═══════════════════════════");
    
    println!("\n🥇 Best Models:");
    println!("  Overall:          {}", report.best_overall_model);
    println!("  Accuracy:         {}", report.best_accuracy_model);
    println!("  Speed:            {}", report.fastest_model);
    println!("  Efficiency:       {}", report.most_efficient_model);
    
    println!("\n📊 Performance Summary:");
    let summary = &report.performance_summary;
    println!("  Average MSE:      {:.6}", summary.average_mse);
    println!("  Average R²:       {:.6}", summary.average_r_squared);
    println!("  Best MSE:         {:.6}", summary.best_mse);
    println!("  Worst MSE:        {:.6}", summary.worst_mse);
    println!("  MSE Std Dev:      {:.6}", summary.mse_std_dev);
    println!("  Convergence Rate: {:.1}%", summary.convergence_rate);
    println!("  Avg Training:     {:.2}s", summary.average_training_time.as_secs_f32());
    
    println!("\n🏅 Rankings:");
    
    println!("\n  Accuracy Ranking:");
    for (i, entry) in report.model_rankings.accuracy_ranking.iter().enumerate() {
        println!("    {}. {} (R² = {:.6})", i + 1, entry.model_name, entry.score);
    }
    
    println!("\n  Speed Ranking:");
    for (i, entry) in report.model_rankings.speed_ranking.iter().enumerate() {
        println!("    {}. {} ({:.2}s)", i + 1, entry.model_name, entry.score);
    }
    
    println!("\n  Efficiency Ranking:");
    for (i, entry) in report.model_rankings.efficiency_ranking.iter().enumerate() {
        println!("    {}. {} (score = {:.6})", i + 1, entry.model_name, entry.score);
    }
}

/// Perform statistical analysis on results
fn perform_statistical_analysis(training_results: &[ModelTrainingResult]) -> Result<()> {
    println!("\n🔬 STATISTICAL ANALYSIS");
    println!("════════════════════════");
    
    if training_results.len() < 2 {
        println!("  Insufficient models for statistical comparison");
        return Ok(());
    }
    
    let losses: Vec<f32> = training_results.iter()
        .map(|r| r.final_loss)
        .collect();
    
    let mean_loss = losses.iter().sum::<f32>() / losses.len() as f32;
    let variance = losses.iter()
        .map(|&loss| (loss - mean_loss).powi(2))
        .sum::<f32>() / losses.len() as f32;
    let std_dev = variance.sqrt();
    
    println!("  Mean Loss:        {:.6}", mean_loss);
    println!("  Std Deviation:    {:.6}", std_dev);
    println!("  Coefficient of Variation: {:.2}%", (std_dev / mean_loss) * 100.0);
    
    // Find best and worst performers
    let best_idx = losses.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    
    let worst_idx = losses.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    
    println!("  Best Model:       {} (Loss: {:.6})", 
             training_results[best_idx].model_name, 
             training_results[best_idx].final_loss);
    println!("  Worst Model:      {} (Loss: {:.6})", 
             training_results[worst_idx].model_name, 
             training_results[worst_idx].final_loss);
    
    // Performance difference
    let performance_diff = (training_results[worst_idx].final_loss - training_results[best_idx].final_loss) 
        / training_results[best_idx].final_loss * 100.0;
    println!("  Performance Gap:  {:.1}%", performance_diff);
    
    Ok(())
}

/// Run performance benchmarks
async fn run_performance_benchmarks(
    evaluator: &ModelEvaluator,
    data_split: &neural_training::data::DataSplit,
) -> Result<()> {
    info!("\n=== PERFORMANCE BENCHMARKS ===");
    
    // Create a reference model for benchmarking
    let mut model = NeuralModel::new("benchmark_model".to_string());
    model.layers = vec![22, 32, 16, 1];
    
    let benchmark_sizes = vec![100, 500, 1000, 5000];
    
    println!("\n⏱️  SCALABILITY BENCHMARKS");
    println!("════════════════════════════");
    println!("{:>10} {:>15} {:>15} {:>15}", "Samples", "Time (ms)", "Throughput", "Memory (KB)");
    println!("{:-<60}", "");
    
    for &size in &benchmark_sizes {
        if size > data_split.test.features.nrows() {
            continue;
        }
        
        // Create subset of data
        let subset_features = data_split.test.features.slice(ndarray::s![0..size, ..]).to_owned();
        let subset_targets = data_split.test.targets.slice(ndarray::s![0..size]).to_owned();
        
        let subset_data = TelecomDataset {
            features: subset_features,
            targets: subset_targets,
            feature_names: data_split.test.feature_names.clone(),
            target_name: data_split.test.target_name.clone(),
            normalization_stats: data_split.test.normalization_stats.clone(),
        };
        
        // Benchmark evaluation
        let start_time = Instant::now();
        let _metrics = evaluator.evaluate_model(&mut model, &subset_data)?;
        let duration = start_time.elapsed();
        
        let throughput = size as f32 / duration.as_secs_f32();
        let memory_estimate = size * 4 * subset_data.features.ncols() / 1024; // Rough estimate in KB
        
        println!("{:>10} {:>15.2} {:>15.0} {:>15}", 
                 size, 
                 duration.as_millis(), 
                 throughput,
                 memory_estimate);
    }
    
    // Cross-validation benchmark
    println!("\n⏱️  CROSS-VALIDATION BENCHMARKS");
    println!("═══════════════════════════════════");
    
    let cv_folds_options = vec![3, 5, 10];
    
    println!("{:>10} {:>15} {:>20}", "CV Folds", "Time (s)", "Time per Fold (s)");
    println!("{:-<50}", "");
    
    for &folds in &cv_folds_options {
        let start_time = Instant::now();
        let _cv_results = evaluator.cross_validate_enhanced(model.clone(), &data_split.test, folds)?;
        let duration = start_time.elapsed();
        
        let time_per_fold = duration.as_secs_f32() / folds as f32;
        
        println!("{:>10} {:>15.2} {:>20.3}", 
                 folds, 
                 duration.as_secs_f32(), 
                 time_per_fold);
    }
    
    Ok(())
}