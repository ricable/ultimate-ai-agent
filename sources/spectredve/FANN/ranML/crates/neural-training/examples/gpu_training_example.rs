//! Complete GPU Training Example
//! 
//! This example demonstrates how to use the GPU-accelerated training system
//! to train CNN, LSTM, and Dense MLP models on telecom data with Mac GPU acceleration.

use anyhow::Result;
use log::info;
use std::path::PathBuf;

use neural_training::{
    data::TelecomDataset,
    models::TrainingParameters,
    gpu_training::{GpuTrainingConfig, GpuTrainingPipeline, ModelArchitecture, MemoryOptimization},
    gpu_data_loader::{GpuDataLoader, PreprocessingConfig, NormalizationType, MissingValueStrategy},
    training_monitor::{TrainingMonitor, TrainingMonitorBuilder, AlertThresholds},
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Info)
        .init();

    info!("Starting GPU Training Example");

    // 1. Load and preprocess data
    let data_path = PathBuf::from("../../data/pm/fanndata.csv");
    let dataset = load_telecom_data(&data_path).await?;
    
    // 2. Split data into training and validation sets
    let (train_data, validation_data) = dataset.train_test_split(0.8, Some(42))?;
    
    info!("Dataset loaded: {} training samples, {} validation samples, {} features",
          train_data.features.nrows(),
          validation_data.features.nrows(),
          train_data.features.ncols());

    // 3. Setup preprocessing configuration
    let preprocessing_config = PreprocessingConfig {
        normalization: NormalizationType::StandardScore,
        missing_value_strategy: MissingValueStrategy::Mean,
        feature_selection_threshold: 0.01,
        apply_scaling: true,
        outlier_threshold: 3.0,
        polynomial_features: false,
        polynomial_degree: 2,
        time_window_size: 10,
        time_window_overlap: 5,
    };

    // 4. Setup GPU training configuration
    let gpu_config = GpuTrainingConfig {
        device: "auto".to_string(), // Auto-detect best device (MPS on Mac)
        batch_size: 32,
        num_workers: 4,
        memory_pool_size: 1024, // 1GB memory pool
        mixed_precision: true,
        memory_optimization: MemoryOptimization::Balanced,
        gradient_accumulation_steps: 1,
        auto_lr_schedule: true,
        early_stopping_patience: 30,
        checkpoint_interval: 10,
    };

    // 5. Setup training parameters
    let training_params = TrainingParameters {
        learning_rate: 0.001,
        momentum: 0.9,
        max_epochs: 200,
        target_error: 0.001,
        batch_size: Some(32),
        weight_decay: 0.0001,
        dropout_rate: 0.2,
    };

    // 6. Setup training monitor
    let alert_thresholds = AlertThresholds {
        max_training_loss: 5.0,
        max_gpu_memory: 6000.0, // 6GB
        max_epoch_time: 120.0,  // 2 minutes
        min_throughput: 50.0,   // 50 samples/sec
        min_learning_rate: 1e-7,
    };

    let monitor = TrainingMonitorBuilder::new()
        .update_frequency(1.0)
        .history_window(500)
        .enable_plotting(true)
        .log_file_path("gpu_training_logs.json".to_string())
        .alert_thresholds(alert_thresholds)
        .build()?;

    info!("Configuration complete. Starting GPU training pipeline...");

    // 7. Create and initialize training pipeline
    let pipeline = GpuTrainingPipeline::new(gpu_config)?;

    // 8. Start monitoring
    monitor.start_training("CNN_Spatial".to_string(), ModelArchitecture::CNN, training_params.max_epochs)?;
    monitor.start_training("LSTM_TimeSeries".to_string(), ModelArchitecture::LSTM, training_params.max_epochs)?;
    monitor.start_training("MLP_Regression".to_string(), ModelArchitecture::DenseMLP, training_params.max_epochs)?;

    // 9. Run parallel training of all three models
    info!("Starting parallel training of CNN, LSTM, and Dense MLP models...");
    let start_time = std::time::Instant::now();

    let results = pipeline.run_training_pipeline(
        train_data,
        Some(validation_data),
        training_params,
    ).await?;

    let total_time = start_time.elapsed();

    // 10. Complete monitoring and get final stats
    for result in &results.model_results {
        monitor.complete_training(result.model_name.clone(), result.clone())?;
    }

    let final_stats = monitor.get_stats();

    // 11. Display results
    println!("\n{}", "=".repeat(60));
    println!("GPU TRAINING COMPLETED");
    println!("{}", "=".repeat(60));
    
    println!("\n📊 OVERALL RESULTS:");
    println!("  • Total Training Time: {:.2} seconds", total_time.as_secs_f64());
    println!("  • Best Model: {} (Loss: {:.6})", results.best_model_name, results.best_loss);
    println!("  • Device: {} ({})", results.device_info.device_type, results.device_info.device_name);
    
    println!("\n🏆 MODEL PERFORMANCE:");
    for result in &results.model_results {
        println!("  • {}: Loss = {:.6}, Epochs = {}, Time = {:.1}s, Memory = {:.1}MB",
                 result.model_name,
                 result.best_loss,
                 result.total_epochs,
                 result.training_time.as_secs_f64(),
                 result.final_gpu_memory);
    }

    println!("\n⚡ PERFORMANCE METRICS:");
    println!("  • Average Throughput: {:.1} samples/sec", final_stats.performance.average_throughput);
    println!("  • Peak Throughput: {:.1} samples/sec", final_stats.performance.peak_throughput);
    println!("  • Average GPU Utilization: {:.1}%", final_stats.performance.average_gpu_utilization);
    println!("  • Peak GPU Memory: {:.1} MB", final_stats.gpu_stats.peak_memory_usage);

    // 12. Generate detailed report
    let detailed_report = results.generate_report();
    println!("\n📋 DETAILED REPORT:");
    println!("{}", detailed_report);

    // 13. Save results
    save_training_results(&results, &monitor).await?;

    // 14. Performance comparison
    analyze_model_performance(&results)?;

    info!("GPU training example completed successfully!");
    info!("Check 'gpu_training_results/' directory for saved outputs");

    Ok(())
}

/// Load telecom dataset with error handling
async fn load_telecom_data(data_path: &PathBuf) -> Result<TelecomDataset> {
    if !data_path.exists() {
        // Create synthetic data for demo if real data doesn't exist
        info!("Real data not found, generating synthetic telecom dataset...");
        return generate_synthetic_telecom_data(2000, 45);
    }

    TelecomDataset::from_csv(data_path)
}

/// Generate synthetic telecom data for demonstration
fn generate_synthetic_telecom_data(num_samples: usize, num_features: usize) -> Result<TelecomDataset> {
    use ndarray::{Array1, Array2};
    use rand::Rng;

    let mut rng = rand::thread_rng();

    // Generate realistic telecom features
    let features = Array2::from_shape_fn((num_samples, num_features), |(i, j)| {
        match j {
            // Cell load metrics (0-100%)
            0..=5 => rng.gen_range(0.0..100.0),
            // SINR values (-20 to 30 dB)
            6..=10 => rng.gen_range(-20.0..30.0),
            // Throughput metrics (0-1000 Mbps)
            11..=15 => rng.gen_range(0.0..1000.0),
            // Latency metrics (1-100 ms)
            16..=20 => rng.gen_range(1.0..100.0),
            // Power levels (0-50 dBm)
            21..=25 => rng.gen_range(0.0..50.0),
            // User counts (0-500)
            26..=30 => rng.gen_range(0.0..500.0),
            // Time-based features (0-24 hours, 0-7 days)
            31..=35 => rng.gen_range(0.0..24.0),
            36..=40 => rng.gen_range(0.0..7.0),
            // Random normalized features
            _ => rng.gen_range(-1.0..1.0),
        }
    });

    // Generate synthetic targets based on feature combinations
    let targets = Array1::from_shape_fn(num_samples, |i| {
        let row = features.row(i);
        // Synthetic target: weighted combination of key features with noise
        let throughput_factor = row[12] / 1000.0; // Normalized throughput
        let load_factor = 1.0 - (row[2] / 100.0); // Inverted load (lower is better)
        let sinr_factor = (row[8] + 20.0) / 50.0; // Normalized SINR
        let latency_factor = 1.0 - (row[18] / 100.0); // Inverted latency
        
        let target = (throughput_factor * 0.3 + load_factor * 0.25 + 
                     sinr_factor * 0.25 + latency_factor * 0.2) + 
                     rng.gen_range(-0.1..0.1); // Add noise
        
        target.max(0.0).min(1.0) // Clamp to [0, 1]
    });

    // Generate feature names
    let feature_names = (0..num_features).map(|i| {
        match i {
            0..=5 => format!("cell_load_{}", i),
            6..=10 => format!("sinr_{}", i - 6),
            11..=15 => format!("throughput_{}", i - 11),
            16..=20 => format!("latency_{}", i - 16),
            21..=25 => format!("power_{}", i - 21),
            26..=30 => format!("users_{}", i - 26),
            31..=35 => format!("hour_{}", i - 31),
            36..=40 => format!("day_{}", i - 36),
            _ => format!("feature_{}", i),
        }
    }).collect();

    Ok(TelecomDataset {
        features,
        targets,
        feature_names,
    })
}

/// Save training results and monitoring data
async fn save_training_results(
    results: &neural_training::gpu_training::TrainingPipelineResult,
    monitor: &TrainingMonitor,
) -> Result<()> {
    use std::fs;

    // Create results directory
    fs::create_dir_all("gpu_training_results")?;

    // Save main results
    let results_json = serde_json::to_string_pretty(results)?;
    fs::write("gpu_training_results/pipeline_results.json", results_json)?;

    // Save individual model results
    for result in &results.model_results {
        let filename = format!("gpu_training_results/{}_detailed.json", result.model_name);
        let model_json = serde_json::to_string_pretty(result)?;
        fs::write(filename, model_json)?;
    }

    // Save monitoring statistics
    let stats = monitor.get_stats();
    let stats_json = serde_json::to_string_pretty(&stats)?;
    fs::write("gpu_training_results/training_statistics.json", stats_json)?;

    // Save monitoring data for each model
    for result in &results.model_results {
        if let Some(metrics) = monitor.get_model_metrics(&result.model_name) {
            let filename = format!("gpu_training_results/{}_metrics_history.json", result.model_name);
            let metrics_json = serde_json::to_string_pretty(&metrics)?;
            fs::write(filename, metrics_json)?;
        }
    }

    // Generate and save comprehensive report
    let report = results.generate_report();
    fs::write("gpu_training_results/training_report.txt", report)?;

    // Generate monitoring report
    let monitor_report = monitor.generate_progress_report();
    fs::write("gpu_training_results/monitoring_report.txt", monitor_report)?;

    info!("All results saved to 'gpu_training_results/' directory");
    Ok(())
}

/// Analyze and compare model performance
fn analyze_model_performance(
    results: &neural_training::gpu_training::TrainingPipelineResult,
) -> Result<()> {
    println!("\n🔍 PERFORMANCE ANALYSIS:");
    
    // Find best and worst performers
    let mut sorted_results = results.model_results.clone();
    sorted_results.sort_by(|a, b| a.best_loss.partial_cmp(&b.best_loss).unwrap());
    
    let best = &sorted_results[0];
    let worst = &sorted_results[sorted_results.len() - 1];
    
    println!("  • Best Performer: {} (Loss: {:.6})", best.model_name, best.best_loss);
    println!("  • Worst Performer: {} (Loss: {:.6})", worst.model_name, worst.best_loss);
    
    // Performance ratios
    let improvement_ratio = worst.best_loss / best.best_loss;
    println!("  • Performance Gap: {:.2}x improvement from worst to best", improvement_ratio);
    
    // Training efficiency
    println!("\n⏱️ TRAINING EFFICIENCY:");
    for result in &results.model_results {
        let samples_per_second = 1000.0 / result.training_time.as_secs_f64(); // Assuming 1000 samples
        let loss_per_second = result.best_loss / result.training_time.as_secs_f64();
        
        println!("  • {}: {:.1} samples/sec, {:.2e} loss/sec",
                 result.model_name, samples_per_second, loss_per_second);
    }
    
    // Memory efficiency
    println!("\n💾 MEMORY EFFICIENCY:");
    for result in &results.model_results {
        println!("  • {}: {:.1} MB peak memory",
                 result.model_name, result.final_gpu_memory);
    }
    
    // Convergence analysis
    println!("\n📈 CONVERGENCE ANALYSIS:");
    for result in &results.model_results {
        let convergence_rate = if result.total_epochs > 0 {
            result.best_epoch as f64 / result.total_epochs as f64
        } else {
            0.0
        };
        
        println!("  • {}: converged at {:.1}% of training ({}/{})",
                 result.model_name,
                 convergence_rate * 100.0,
                 result.best_epoch,
                 result.total_epochs);
    }

    Ok(())
}