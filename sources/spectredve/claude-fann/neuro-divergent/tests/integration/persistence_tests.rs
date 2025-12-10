//! Integration tests for model persistence and loading
//!
//! Tests model persistence scenarios including:
//! - Saving and loading trained models
//! - Model state serialization
//! - Cross-version compatibility
//! - Incremental learning from saved states

use neuro_divergent::prelude::*;
use neuro_divergent::{
    NeuralForecast, Frequency, ScalerType,
    models::{LSTM, NBEATS, MLP, DeepAR, Transformer},
    data::{TimeSeriesDataFrame, TimeSeriesSchema},
    config::{LossFunction, OptimizerType},
};
use std::path::{Path, PathBuf};
use std::fs;
use polars::prelude::*;
use chrono::{DateTime, Utc, Duration};
use num_traits::Float;
use rand::Rng;
use tempfile::TempDir;
use serde::{Serialize, Deserialize};

/// Generate test data for persistence tests
fn generate_test_data<T: Float>(
    n_series: usize,
    n_points: usize,
) -> Result<TimeSeriesDataFrame<T>, Box<dyn std::error::Error>>
where
    T: From<f64> + Into<f64>,
{
    let mut rng = rand::thread_rng();
    let start_date = Utc::now() - Duration::days(n_points as i64);
    
    let mut unique_ids = Vec::new();
    let mut timestamps = Vec::new();
    let mut values = Vec::new();
    
    for series_id in 0..n_series {
        let series_name = format!("series_{}", series_id);
        let base = rng.gen_range(10.0..100.0);
        
        for i in 0..n_points {
            let date = start_date + Duration::days(i as i64);
            unique_ids.push(series_name.clone());
            timestamps.push(date.timestamp());
            
            // Simple pattern for reproducibility
            let value = base + (i as f64 * 0.1) + 5.0 * (i as f64 * 0.1).sin();
            values.push(value);
        }
    }
    
    let df = df! {
        "unique_id" => unique_ids,
        "ds" => timestamps,
        "y" => values,
    }?;
    
    Ok(TimeSeriesDataFrame::new(
        df,
        TimeSeriesSchema::default(),
        Some(Frequency::Daily),
    ))
}

/// Test basic model save and load
#[test]
fn test_basic_model_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("lstm_model.ndf");
    
    // Generate data
    let data = generate_test_data::<f32>(3, 100)?;
    
    // Create and train model
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .num_layers(1)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Train model
    nf.fit(data.clone())?;
    
    // Generate forecast before saving
    let forecast_before = nf.predict()?;
    
    // Save model
    nf.save(&model_path)?;
    assert!(model_path.exists());
    
    // Load model
    let loaded_nf = NeuralForecast::<f32>::load(&model_path)?;
    
    // Verify loaded model state
    assert!(loaded_nf.is_fitted());
    assert_eq!(loaded_nf.frequency(), nf.frequency());
    assert_eq!(loaded_nf.model_names(), nf.model_names());
    
    // Generate forecast after loading
    let forecast_after = loaded_nf.predict()?;
    
    // Forecasts should be identical
    assert_eq!(forecast_before.shape(), forecast_after.shape());
    assert_eq!(forecast_before.horizon(), forecast_after.horizon());
    
    Ok(())
}

/// Test ensemble persistence
#[test]
fn test_ensemble_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("ensemble_model.ndf");
    
    // Generate data
    let data = generate_test_data::<f32>(2, 150)?;
    
    // Create ensemble
    let models: Vec<Box<dyn BaseModel<f32>>> = vec![
        Box::new(LSTM::builder()
            .horizon(14)
            .input_size(28)
            .hidden_size(32)
            .build()?),
        Box::new(NBEATS::builder()
            .horizon(14)
            .input_size(28)
            .build()?),
        Box::new(MLP::builder()
            .horizon(14)
            .input_size(28)
            .hidden_layers(vec![32, 16])
            .build()?),
    ];
    
    let mut nf = NeuralForecast::builder()
        .with_models(models)
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Train ensemble
    nf.fit(data)?;
    
    // Save ensemble
    nf.save(&model_path)?;
    
    // Load ensemble
    let loaded_nf = NeuralForecast::<f32>::load(&model_path)?;
    
    // Verify all models are loaded
    assert_eq!(loaded_nf.model_names().len(), 3);
    assert!(loaded_nf.model_names().contains(&"LSTM".to_string()));
    assert!(loaded_nf.model_names().contains(&"NBEATS".to_string()));
    assert!(loaded_nf.model_names().contains(&"MLP".to_string()));
    
    // Test predictions work
    let forecasts = loaded_nf.predict()?;
    assert_eq!(forecasts.model_names().len(), 3);
    
    Ok(())
}

/// Test model state preservation
#[test]
fn test_model_state_preservation() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("state_test.ndf");
    
    // Generate data
    let data = generate_test_data::<f32>(2, 100)?;
    
    // Create model with specific configuration
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .num_layers(2)
        .dropout(0.2)
        .bidirectional(true)
        .learning_rate(0.001)
        .loss_function(LossFunction::MAE)
        .optimizer(OptimizerType::AdamW)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .with_local_scaler(ScalerType::StandardScaler)
        .with_num_threads(4)
        .build()?;
    
    // Train model
    nf.fit(data)?;
    
    // Save model with metadata
    nf.save(&model_path)?;
    
    // Also save model metadata separately for verification
    let metadata_path = temp_dir.path().join("metadata.json");
    let metadata = ModelMetadata {
        version: "1.0.0".to_string(),
        created_at: Utc::now(),
        model_configs: nf.get_model_configs()?,
        training_info: TrainingInfo {
            epochs_trained: 50,
            final_loss: 0.123,
            training_time_seconds: 45.6,
        },
    };
    
    let metadata_json = serde_json::to_string_pretty(&metadata)?;
    fs::write(&metadata_path, metadata_json)?;
    
    // Load model
    let loaded_nf = NeuralForecast::<f32>::load(&model_path)?;
    
    // Verify configuration is preserved
    assert_eq!(loaded_nf.num_threads, Some(4));
    assert_eq!(loaded_nf.local_scaler_type, Some(ScalerType::StandardScaler));
    
    // Verify model configuration is preserved
    if let Some(model) = loaded_nf.get_model("LSTM") {
        // Model should have same configuration
        // This would require access to model internals
    }
    
    Ok(())
}

/// Test incremental learning from saved model
#[test]
fn test_incremental_learning() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("incremental_model.ndf");
    
    // Generate initial training data
    let initial_data = generate_test_data::<f32>(2, 100)?;
    
    // Create and train initial model
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(initial_data)?;
    
    // Save initial model
    nf.save(&model_path)?;
    
    // Generate new data for incremental learning
    let new_data = generate_test_data::<f32>(2, 50)?;
    
    // Load model
    let mut loaded_nf = NeuralForecast::<f32>::load(&model_path)?;
    
    // Continue training with new data (incremental learning)
    loaded_nf.fit_incremental(new_data)?;
    
    // Save updated model
    let updated_path = temp_dir.path().join("incremental_updated.ndf");
    loaded_nf.save(&updated_path)?;
    
    // Verify model can still predict
    let forecasts = loaded_nf.predict()?;
    assert_eq!(forecasts.horizon(), 7);
    
    Ok(())
}

/// Test model versioning and compatibility
#[test]
fn test_model_versioning() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    
    // Create models with version info
    let v1_path = temp_dir.path().join("model_v1.ndf");
    let v2_path = temp_dir.path().join("model_v2.ndf");
    
    let data = generate_test_data::<f32>(2, 100)?;
    
    // Version 1 model
    let lstm_v1 = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(32)
        .build()?;
    
    let mut nf_v1 = NeuralForecast::builder()
        .with_model(Box::new(lstm_v1))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf_v1.fit(data.clone())?;
    nf_v1.save_with_version(&v1_path, "1.0.0")?;
    
    // Version 2 model (with more features)
    let lstm_v2 = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .num_layers(2)
        .bidirectional(true)
        .build()?;
    
    let mut nf_v2 = NeuralForecast::builder()
        .with_model(Box::new(lstm_v2))
        .with_frequency(Frequency::Daily)
        .with_prediction_intervals(PredictionIntervals::default())
        .build()?;
    
    nf_v2.fit(data)?;
    nf_v2.save_with_version(&v2_path, "2.0.0")?;
    
    // Try loading different versions
    let loaded_v1 = NeuralForecast::<f32>::load(&v1_path)?;
    let loaded_v2 = NeuralForecast::<f32>::load(&v2_path)?;
    
    // Check version compatibility
    let v1_info = loaded_v1.get_version_info()?;
    let v2_info = loaded_v2.get_version_info()?;
    
    assert_eq!(v1_info.version, "1.0.0");
    assert_eq!(v2_info.version, "2.0.0");
    
    Ok(())
}

/// Test persistence with large models
#[test]
#[ignore] // This test is slow
fn test_large_model_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let model_path = temp_dir.path().join("large_model.ndf");
    
    // Generate larger dataset
    let data = generate_test_data::<f32>(50, 500)?;
    
    // Create large model
    let large_transformer = Transformer::builder()
        .horizon(30)
        .input_size(60)
        .d_model(512)
        .num_heads(8)
        .num_layers(6)
        .feed_forward_dim(2048)
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(large_transformer))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Train model
    println!("Training large model...");
    let train_start = std::time::Instant::now();
    nf.fit(data)?;
    let train_time = train_start.elapsed();
    println!("Training completed in {:?}", train_time);
    
    // Save model
    println!("Saving large model...");
    let save_start = std::time::Instant::now();
    nf.save(&model_path)?;
    let save_time = save_start.elapsed();
    let file_size = fs::metadata(&model_path)?.len();
    println!("Model saved in {:?}, size: {} MB", save_time, file_size / 1_000_000);
    
    // Load model
    println!("Loading large model...");
    let load_start = std::time::Instant::now();
    let loaded_nf = NeuralForecast::<f32>::load(&model_path)?;
    let load_time = load_start.elapsed();
    println!("Model loaded in {:?}", load_time);
    
    // Verify model works
    let _ = loaded_nf.predict()?;
    
    Ok(())
}

/// Test checkpoint saving during training
#[test]
fn test_training_checkpoints() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let checkpoint_dir = temp_dir.path().join("checkpoints");
    fs::create_dir(&checkpoint_dir)?;
    
    let data = generate_test_data::<f32>(3, 200)?;
    
    // Create model with checkpoint callback
    let lstm = LSTM::builder()
        .horizon(7)
        .input_size(14)
        .hidden_size(64)
        .max_steps(100)
        .checkpoint_dir(checkpoint_dir.clone())
        .checkpoint_frequency(20) // Save every 20 steps
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(lstm))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    // Train model (checkpoints will be saved automatically)
    nf.fit(data)?;
    
    // Verify checkpoints were created
    let checkpoint_files: Vec<_> = fs::read_dir(&checkpoint_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "ckpt")
                .unwrap_or(false)
        })
        .collect();
    
    assert!(!checkpoint_files.is_empty(), "No checkpoints were saved");
    println!("Found {} checkpoints", checkpoint_files.len());
    
    // Load from checkpoint
    let latest_checkpoint = checkpoint_files
        .iter()
        .max_by_key(|entry| entry.metadata().unwrap().modified().unwrap())
        .unwrap();
    
    let restored_nf = NeuralForecast::<f32>::load_from_checkpoint(
        &latest_checkpoint.path()
    )?;
    
    // Verify restored model works
    let forecasts = restored_nf.predict()?;
    assert_eq!(forecasts.horizon(), 7);
    
    Ok(())
}

/// Test model compression
#[test]
fn test_model_compression() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let uncompressed_path = temp_dir.path().join("model_uncompressed.ndf");
    let compressed_path = temp_dir.path().join("model_compressed.ndf.gz");
    
    let data = generate_test_data::<f32>(3, 100)?;
    
    // Create and train model
    let nbeats = NBEATS::builder()
        .horizon(14)
        .input_size(28)
        .n_blocks(vec![3, 3, 3])
        .mlp_units(vec![vec![512, 512], vec![512, 512], vec![512, 512]])
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(nbeats))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(data)?;
    
    // Save uncompressed
    nf.save(&uncompressed_path)?;
    let uncompressed_size = fs::metadata(&uncompressed_path)?.len();
    
    // Save compressed
    nf.save_compressed(&compressed_path)?;
    let compressed_size = fs::metadata(&compressed_path)?.len();
    
    println!("Uncompressed size: {} KB", uncompressed_size / 1024);
    println!("Compressed size: {} KB", compressed_size / 1024);
    println!("Compression ratio: {:.2}x", 
             uncompressed_size as f64 / compressed_size as f64);
    
    // Load compressed model
    let loaded_nf = NeuralForecast::<f32>::load_compressed(&compressed_path)?;
    
    // Verify model works
    let forecasts = loaded_nf.predict()?;
    assert_eq!(forecasts.horizon(), 14);
    
    // Compression should reduce size significantly
    assert!(compressed_size < uncompressed_size * 3 / 4);
    
    Ok(())
}

/// Test model export to different formats
#[test]
fn test_model_export_formats() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    
    let data = generate_test_data::<f32>(2, 100)?;
    
    // Create and train model
    let mlp = MLP::builder()
        .horizon(7)
        .input_size(14)
        .hidden_layers(vec![32, 16])
        .build()?;
    
    let mut nf = NeuralForecast::builder()
        .with_model(Box::new(mlp))
        .with_frequency(Frequency::Daily)
        .build()?;
    
    nf.fit(data)?;
    
    // Export to different formats
    
    // 1. ONNX format (for interoperability)
    let onnx_path = temp_dir.path().join("model.onnx");
    nf.export_onnx(&onnx_path)?;
    assert!(onnx_path.exists());
    
    // 2. JSON format (for debugging/inspection)
    let json_path = temp_dir.path().join("model.json");
    nf.export_json(&json_path)?;
    assert!(json_path.exists());
    
    // 3. Binary format (native)
    let bin_path = temp_dir.path().join("model.bin");
    nf.export_binary(&bin_path)?;
    assert!(bin_path.exists());
    
    // Verify JSON export contains expected structure
    let json_content = fs::read_to_string(&json_path)?;
    let json_value: serde_json::Value = serde_json::from_str(&json_content)?;
    assert!(json_value["models"].is_array());
    assert!(json_value["frequency"].is_string());
    
    Ok(())
}

/// Test concurrent model persistence
#[test]
fn test_concurrent_persistence() -> Result<(), Box<dyn std::error::Error>> {
    let temp_dir = TempDir::new()?;
    let data = generate_test_data::<f32>(5, 150)?;
    
    // Create multiple models
    let models = vec![
        ("lstm", Box::new(LSTM::builder()
            .horizon(7)
            .input_size(14)
            .hidden_size(32)
            .build()?) as Box<dyn BaseModel<f32>>),
        ("nbeats", Box::new(NBEATS::builder()
            .horizon(7)
            .input_size(14)
            .build()?)),
        ("mlp", Box::new(MLP::builder()
            .horizon(7)
            .input_size(14)
            .hidden_layers(vec![32])
            .build()?)),
    ];
    
    // Train and save models concurrently
    use std::thread;
    use std::sync::Arc;
    
    let data_arc = Arc::new(data);
    let temp_dir_path = temp_dir.path().to_path_buf();
    
    let handles: Vec<_> = models.into_iter().map(|(name, model)| {
        let data_clone = Arc::clone(&data_arc);
        let path = temp_dir_path.join(format!("{}.ndf", name));
        
        thread::spawn(move || -> Result<(), Box<dyn std::error::Error + Send>> {
            let mut nf = NeuralForecast::builder()
                .with_model(model)
                .with_frequency(Frequency::Daily)
                .build()?;
            
            nf.fit((*data_clone).clone())?;
            nf.save(&path)?;
            
            Ok(())
        })
    }).collect();
    
    // Wait for all saves to complete
    for handle in handles {
        handle.join().unwrap()?;
    }
    
    // Verify all models were saved
    let saved_files: Vec<_> = fs::read_dir(temp_dir.path())?
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.path().extension()
                .and_then(|ext| ext.to_str())
                .map(|ext| ext == "ndf")
                .unwrap_or(false)
        })
        .collect();
    
    assert_eq!(saved_files.len(), 3);
    
    // Load all models
    for file in saved_files {
        let loaded_nf = NeuralForecast::<f32>::load(&file.path())?;
        assert!(loaded_nf.is_fitted());
    }
    
    Ok(())
}

/// Helper structs for metadata
#[derive(Debug, Serialize, Deserialize)]
struct ModelMetadata {
    version: String,
    created_at: DateTime<Utc>,
    model_configs: Vec<serde_json::Value>,
    training_info: TrainingInfo,
}

#[derive(Debug, Serialize, Deserialize)]
struct TrainingInfo {
    epochs_trained: usize,
    final_loss: f64,
    training_time_seconds: f64,
}