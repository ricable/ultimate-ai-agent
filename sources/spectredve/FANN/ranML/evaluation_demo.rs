//! Comprehensive Model Evaluation Demo
//! 
//! This demo showcases the advanced evaluation and benchmarking capabilities
//! for neural network models trained on telecom network data.

use neural_training::{
    TelecomDataset, NeuralModel, NetworkArchitectures, ModelEvaluator,
    DashboardGenerator, BenchmarkFramework, SimpleNeuralTrainingSystem,
    EvaluationMetrics, ModelComparisonReport
};
use anyhow::Result;
use ndarray::{Array2, Array1};
use rand::{thread_rng, Rng};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧠 Neural Network Model Evaluation Demo");
    println!("═══════════════════════════════════════════");
    println!("This demo showcases comprehensive evaluation capabilities for telecom network performance prediction models.\n");

    // Step 1: Create synthetic telecom dataset
    println!("📊 Step 1: Creating synthetic telecom network performance dataset");
    let dataset = create_telecom_dataset(1000)?;
    println!("  ✅ Dataset created: {} samples, {} features", 
             dataset.features.nrows(), 
             dataset.features.ncols());
    
    // Split into train/test
    let (train_data, test_data) = split_dataset(&dataset, 0.8)?;
    println!("  📈 Training set: {} samples", train_data.features.nrows());
    println!("  🔬 Test set: {} samples\n", test_data.features.nrows());

    // Step 2: Train multiple models with different architectures
    println!("🧠 Step 2: Training multiple neural network architectures");
    let architectures = vec!["shallow", "deep", "wide"];
    let mut trained_models = Vec::new();
    let mut training_results = Vec::new();

    for arch in &architectures {
        print!("  🔧 Training {} architecture... ", arch);
        let start_time = Instant::now();
        
        match SimpleNeuralTrainingSystem::train_basic_model(&train_data, Some(&test_data), arch) {
            Ok(result) => {
                let training_time = start_time.elapsed();
                println!("✅ Complete ({:.2}s, Loss: {:.6})", 
                        training_time.as_secs_f32(), 
                        result.final_training_loss);
                
                // Create model for evaluation
                let input_size = train_data.features.ncols();
                let architecture = match arch.as_str() {
                    "shallow" => NetworkArchitectures::shallow_network(input_size, 1),
                    "deep" => NetworkArchitectures::deep_network(input_size, 1),
                    "wide" => NetworkArchitectures::wide_network(input_size, 1),
                    _ => NetworkArchitectures::shallow_network(input_size, 1),
                };
                
                let model = NeuralModel::from_architecture(architecture)?;
                trained_models.push(model);
                training_results.push(result);
            }
            Err(e) => {
                println!("❌ Failed: {}", e);
            }
        }
    }

    if trained_models.is_empty() {
        println!("❌ No models were successfully trained!");
        return Ok(());
    }

    println!("");

    // Step 3: Comprehensive evaluation
    println!("📈 Step 3: Comprehensive Model Evaluation");
    let evaluator = create_evaluator();

    println!("  🔍 Evaluating individual models...");
    let mut all_metrics = Vec::new();
    
    for (i, model) in trained_models.iter_mut().enumerate() {
        let metrics = ModelEvaluator::evaluate_model(model, &test_data)?;
        all_metrics.push((model.name.clone(), metrics.clone()));
        
        println!("    📊 {}: R²={:.4}, RMSE={:.4}, MAPE={:.2}%", 
                model.name, metrics.r_squared, metrics.rmse, metrics.mape);
    }

    // Step 4: Model comparison analysis
    println!("\n🔍 Step 4: Comparative Analysis");
    let comparison_report = ModelEvaluator::compare_models(&training_results)?;
    
    println!("  🏆 Best overall model: {}", comparison_report.best_overall_model);
    println!("  🎯 Best accuracy model: {}", comparison_report.best_accuracy_model);
    println!("  ⚡ Fastest model: {}", comparison_report.fastest_model);
    println!("  💪 Most efficient model: {}", comparison_report.most_efficient_model);
    
    // Display performance summary
    let summary = &comparison_report.performance_summary;
    println!("  📊 Performance Summary:");
    println!("    - Average R²: {:.4}", summary.average_r_squared);
    println!("    - Average MSE: {:.6}", summary.average_mse);
    println!("    - Convergence rate: {:.1}%", summary.convergence_rate);
    println!("    - MSE std dev: {:.6}", summary.mse_std_dev);

    // Step 5: Cross-validation analysis
    println!("\n🔄 Step 5: Cross-Validation Analysis");
    for model in &trained_models {
        println!("  📊 Running 5-fold CV for {}...", model.name);
        match ModelEvaluator::cross_validate(model.clone(), &train_data, 5) {
            Ok(cv_results) => {
                println!("    ✅ CV R² = {:.4} ± {:.4}", 
                        cv_results.cv_score, cv_results.cv_score_std);
                println!("    📈 Best fold: {}, Worst fold: {}", 
                        cv_results.best_fold + 1, cv_results.worst_fold + 1);
            }
            Err(e) => {
                println!("    ❌ CV failed: {}", e);
            }
        }
    }

    // Step 6: Performance benchmarking
    println!("\n⚡ Step 6: Performance Benchmarking");
    println!("  🚀 Running comprehensive benchmarks...");
    
    let benchmark_framework = BenchmarkFramework::new();
    match benchmark_framework.run_comprehensive_benchmark(&evaluator, &test_data).await {
        Ok(benchmark_results) => {
            println!("  ✅ Benchmarking completed!");
            
            // Display key metrics
            println!("  📊 System Performance:");
            println!("    - CPU cores: {}", benchmark_results.system_info.cpu_cores);
            println!("    - Memory: {:.1}GB total, {:.1}GB available", 
                    benchmark_results.system_info.total_memory_gb,
                    benchmark_results.system_info.available_memory_gb);
            
            println!("  ⚡ Throughput Analysis:");
            println!("    - Peak throughput: {:.1} samples/sec", 
                    benchmark_results.throughput_analysis.peak_throughput);
            println!("    - Sustained throughput: {:.1} samples/sec", 
                    benchmark_results.throughput_analysis.sustained_throughput);
            println!("    - Throughput variance: {:.2}", 
                    benchmark_results.throughput_analysis.throughput_variance);
            
            println!("  🧠 Memory Analysis:");
            println!("    - Peak memory usage: {:.1}MB", 
                    benchmark_results.memory_profile.peak_memory_usage as f64 / 1_048_576.0);
            println!("    - Memory efficiency: {:.1}%", 
                    benchmark_results.memory_profile.memory_efficiency_score);
            
            // Display optimization recommendations
            if !benchmark_results.recommendations.is_empty() {
                println!("  💡 Optimization Recommendations:");
                for (i, rec) in benchmark_results.recommendations.iter().take(3).enumerate() {
                    println!("    {}. {}", i + 1, rec);
                }
            }
        }
        Err(e) => {
            println!("  ⚠️  Benchmarking failed: {}", e);
        }
    }

    // Step 7: Generate comprehensive dashboard
    println!("\n📊 Step 7: Generating Evaluation Dashboard");
    println!("  🎨 Creating interactive dashboard...");
    
    let dashboard_generator = DashboardGenerator::new("./evaluation_reports");
    match dashboard_generator.generate_dashboard(&trained_models, &test_data, None).await {
        Ok(dashboard) => {
            println!("  ✅ Dashboard generated successfully!");
            println!("  📁 Dashboard ID: {}", dashboard.dashboard_id);
            println!("  🌐 HTML report: ./evaluation_reports/dashboard_{}.html", dashboard.dashboard_id);
            println!("  📊 JSON data: ./evaluation_reports/dashboard_data_{}.json", dashboard.dashboard_id);
            
            // Display deployment readiness
            let readiness = &dashboard.deployment_readiness;
            println!("  🚀 Deployment Readiness: {:.1}% ({:?})", 
                    readiness.overall_score, readiness.readiness_status);
            
            // Show top recommendations
            if !dashboard.recommendations.development_recommendations.is_empty() {
                println!("  💡 Top Development Recommendations:");
                for (i, rec) in dashboard.recommendations.development_recommendations.iter().take(2).enumerate() {
                    println!("    {}. {} (Impact: {:.1}%)", 
                            i + 1, rec.title, rec.expected_impact * 100.0);
                }
            }
        }
        Err(e) => {
            println!("  ❌ Dashboard generation failed: {}", e);
        }
    }

    // Step 8: Statistical analysis summary
    println!("\n📈 Step 8: Statistical Analysis Summary");
    display_statistical_summary(&all_metrics)?;

    // Step 9: Model selection recommendations
    println!("\n🎯 Step 9: Model Selection Recommendations");
    provide_model_recommendations(&comparison_report, &all_metrics)?;

    println!("\n✅ Comprehensive evaluation completed!");
    println!("═══════════════════════════════════════");
    println!("📁 All results saved to: ./evaluation_reports/");
    println!("🌐 Open the HTML dashboard for interactive exploration");
    println!("📊 Use the JSON files for programmatic analysis");

    Ok(())
}

fn create_telecom_dataset(n_samples: usize) -> Result<TelecomDataset> {
    let mut rng = thread_rng();
    let n_features = 22; // Standard telecom feature set
    
    let mut features = Array2::zeros((n_samples, n_features));
    let mut targets = Array1::zeros(n_samples);
    
    // Generate realistic telecom network performance data
    for i in 0..n_samples {
        // Signal quality features (0-4)
        features[[i, 0]] = rng.gen_range(0.2..1.0); // Signal strength
        features[[i, 1]] = rng.gen_range(0.0..0.8); // Noise level
        features[[i, 2]] = rng.gen_range(0.5..1.0); // SNR
        features[[i, 3]] = rng.gen_range(0.0..0.3); // Interference
        features[[i, 4]] = rng.gen_range(0.6..1.0); // Channel quality
        
        // Network performance features (5-10)
        features[[i, 5]] = rng.gen_range(0.0..0.1); // Latency (normalized)
        features[[i, 6]] = rng.gen_range(0.3..1.0); // Throughput
        features[[i, 7]] = rng.gen_range(0.0..0.05); // Packet loss
        features[[i, 8]] = rng.gen_range(0.0..0.08); // Jitter
        features[[i, 9]] = rng.gen_range(0.7..1.0); // Bandwidth utilization
        features[[i, 10]] = rng.gen_range(0.85..1.0); // Connection success rate
        
        // System resource features (11-16)
        features[[i, 11]] = rng.gen_range(0.3..0.9); // CPU usage
        features[[i, 12]] = rng.gen_range(0.4..0.8); // Memory usage
        features[[i, 13]] = rng.gen_range(0.2..0.7); // Disk I/O
        features[[i, 14]] = rng.gen_range(0.1..0.6); // Network I/O
        features[[i, 15]] = rng.gen_range(0.5..1.0); // System health
        features[[i, 16]] = rng.gen_range(0.8..1.0); // Uptime percentage
        
        // Environmental features (17-21)
        features[[i, 17]] = rng.gen_range(0.0..1.0); // Time of day (normalized)
        features[[i, 18]] = rng.gen_range(0.0..1.0); // Day of week
        features[[i, 19]] = rng.gen_range(0.3..1.0); // Network load
        features[[i, 20]] = rng.gen_range(0.0..1.0); // Weather impact
        features[[i, 21]] = rng.gen_range(0.5..1.0); // Geographic factor
        
        // Create realistic target based on feature combinations
        let signal_quality = (features[[i, 0]] + features[[i, 2]] + features[[i, 4]]) / 3.0;
        let network_perf = (features[[i, 6]] + features[[i, 10]] - features[[i, 7]]) / 2.0;
        let system_health = (features[[i, 15]] + features[[i, 16]]) / 2.0;
        
        targets[i] = (0.4 * signal_quality + 0.4 * network_perf + 0.2 * system_health)
            .max(0.0).min(1.0) + rng.gen_range(-0.05..0.05);
    }
    
    let feature_names = vec![
        "signal_strength", "noise_level", "snr", "interference", "channel_quality",
        "latency", "throughput", "packet_loss", "jitter", "bandwidth_util", "connection_success",
        "cpu_usage", "memory_usage", "disk_io", "network_io", "system_health", "uptime",
        "time_of_day", "day_of_week", "network_load", "weather_impact", "geographic_factor"
    ].into_iter().map(|s| s.to_string()).collect();
    
    Ok(TelecomDataset {
        features,
        targets,
        feature_names,
        target_name: "network_performance_score".to_string(),
        normalization_stats: None,
    })
}

fn split_dataset(dataset: &TelecomDataset, train_ratio: f64) -> Result<(TelecomDataset, TelecomDataset)> {
    let n_samples = dataset.features.nrows();
    let n_train = (n_samples as f64 * train_ratio) as usize;
    
    let train_features = dataset.features.slice(ndarray::s![0..n_train, ..]).to_owned();
    let train_targets = dataset.targets.slice(ndarray::s![0..n_train]).to_owned();
    
    let test_features = dataset.features.slice(ndarray::s![n_train.., ..]).to_owned();
    let test_targets = dataset.targets.slice(ndarray::s![n_train..]).to_owned();
    
    let train_data = TelecomDataset {
        features: train_features,
        targets: train_targets,
        feature_names: dataset.feature_names.clone(),
        target_name: dataset.target_name.clone(),
        normalization_stats: dataset.normalization_stats.clone(),
    };
    
    let test_data = TelecomDataset {
        features: test_features,
        targets: test_targets,
        feature_names: dataset.feature_names.clone(),
        target_name: dataset.target_name.clone(),
        normalization_stats: dataset.normalization_stats.clone(),
    };
    
    Ok((train_data, test_data))
}

fn create_evaluator() -> ModelEvaluator {
    ModelEvaluator {
        enable_parallel: true,
        confidence_level: 0.95,
        random_seed: Some(42),
        performance_tracking: std::sync::Arc::new(std::sync::Mutex::new(
            neural_training::evaluation::PerformanceTracker {
                evaluation_times: Vec::new(),
                memory_usage: Vec::new(),
                operation_counts: std::collections::HashMap::new(),
                bottlenecks: Vec::new(),
            }
        )),
    }
}

fn display_statistical_summary(metrics: &[(String, EvaluationMetrics)]) -> Result<()> {
    if metrics.is_empty() {
        return Ok(());
    }
    
    // Calculate statistics across all models
    let r_squared_values: Vec<f32> = metrics.iter().map(|(_, m)| m.r_squared).collect();
    let rmse_values: Vec<f32> = metrics.iter().map(|(_, m)| m.rmse).collect();
    let mape_values: Vec<f32> = metrics.iter().map(|(_, m)| m.mape).collect();
    
    let avg_r_squared = r_squared_values.iter().sum::<f32>() / r_squared_values.len() as f32;
    let avg_rmse = rmse_values.iter().sum::<f32>() / rmse_values.len() as f32;
    let avg_mape = mape_values.iter().sum::<f32>() / mape_values.len() as f32;
    
    let std_r_squared = {
        let variance = r_squared_values.iter()
            .map(|&v| (v - avg_r_squared).powi(2))
            .sum::<f32>() / r_squared_values.len() as f32;
        variance.sqrt()
    };
    
    println!("  📊 Across {} models:", metrics.len());
    println!("    R² Score: {:.4} ± {:.4}", avg_r_squared, std_r_squared);
    println!("    RMSE: {:.4} ± {:.4}", avg_rmse, 
            (rmse_values.iter().map(|&v| (v - avg_rmse).powi(2)).sum::<f32>() / rmse_values.len() as f32).sqrt());
    println!("    MAPE: {:.2}% ± {:.2}%", avg_mape,
            (mape_values.iter().map(|&v| (v - avg_mape).powi(2)).sum::<f32>() / mape_values.len() as f32).sqrt());
    
    // Find best and worst performers
    let best_model = metrics.iter()
        .max_by(|(_, a), (_, b)| a.r_squared.partial_cmp(&b.r_squared).unwrap())
        .unwrap();
    let worst_model = metrics.iter()
        .min_by(|(_, a), (_, b)| a.r_squared.partial_cmp(&b.r_squared).unwrap())
        .unwrap();
    
    println!("  🏆 Best performer: {} (R² = {:.4})", best_model.0, best_model.1.r_squared);
    println!("  📉 Needs improvement: {} (R² = {:.4})", worst_model.0, worst_model.1.r_squared);
    
    Ok(())
}

fn provide_model_recommendations(
    comparison_report: &ModelComparisonReport,
    metrics: &[(String, EvaluationMetrics)]
) -> Result<()> {
    println!("  🎯 Production Deployment:");
    println!("    Primary: {} (Best overall performance)", comparison_report.best_overall_model);
    println!("    Backup: {} (Fastest inference)", comparison_report.fastest_model);
    
    println!("  🔧 Development Priorities:");
    
    // Analyze common issues
    let avg_r_squared = metrics.iter().map(|(_, m)| m.r_squared).sum::<f32>() / metrics.len() as f32;
    let avg_mape = metrics.iter().map(|(_, m)| m.mape).sum::<f32>() / metrics.len() as f32;
    
    if avg_r_squared < 0.85 {
        println!("    1. Improve model accuracy (current avg R² = {:.3})", avg_r_squared);
        println!("       - Consider ensemble methods");
        println!("       - Feature engineering");
        println!("       - Hyperparameter optimization");
    }
    
    if avg_mape > 15.0 {
        println!("    2. Reduce prediction error (current avg MAPE = {:.1}%)", avg_mape);
        println!("       - Better loss function selection");
        println!("       - Regularization techniques");
        println!("       - Data augmentation");
    }
    
    println!("    3. Monitor model performance in production");
    println!("       - Implement A/B testing");
    println!("       - Set up automated retraining");
    println!("       - Create performance alerts");
    
    Ok(())
}