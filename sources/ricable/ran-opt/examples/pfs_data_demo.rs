//! PFS Data Processing Demo
//! 
//! Demonstrates usage of the high-performance data processing pipeline
//! for neural network training with Ericsson ENM data.

use ran_opt::pfs_data::{
    DataProcessor, DataChunk,
    parser::{EnmParser, CounterMatcher},
    pipeline::{FeatureExtractor, PipelineConfig, NormalizationMethod, StreamingPipeline},
    kpi::{KpiMappings, KpiCalculator},
    tensor::{TensorStorage, TensorMeta, TensorDataType, TensorBatch, TensorDataset},
};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Write};
use tempfile::tempdir;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("PFS Data Processing Demo");
    println!("=======================");
    
    // Demo 1: Basic data processor
    demo_data_processor()?;
    
    // Demo 2: ENM XML parsing
    demo_enm_parser()?;
    
    // Demo 3: KPI calculations
    demo_kpi_calculation()?;
    
    // Demo 4: Feature extraction pipeline
    demo_feature_pipeline()?;
    
    // Demo 5: Tensor storage and compression
    demo_tensor_storage()?;
    
    // Demo 6: End-to-end processing pipeline
    demo_end_to_end_pipeline()?;
    
    println!("\nDemo completed successfully!");
    Ok(())
}

/// Demonstrate basic data processor functionality
fn demo_data_processor() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n1. Basic Data Processor Demo");
    println!("----------------------------");
    
    let mut processor = DataProcessor::new(1024);
    
    // Create sample ENM XML data
    let xml_data = r#"<?xml version="1.0" encoding="UTF-8"?>
    <measData>
        <measInfo>
            <measType p="1">pmRrcConnEstabSucc</measType>
            <measType p="2">pmRrcConnEstabAtt</measType>
            <measValue measObjLdn="ERBS001/EUtranCellFDD=Cell1">
                <measResults>
                    <r p="1">150</r>
                    <r p="2">200</r>
                </measResults>
            </measValue>
        </measInfo>
    </measData>"#;
    
    // Create temporary file
    let temp_dir = tempdir()?;
    let xml_file = temp_dir.path().join("test.xml");
    let mut file = File::create(&xml_file)?;
    file.write_all(xml_data.as_bytes())?;
    
    // Process the XML file
    let records = processor.process_enm_xml(&xml_file)?;
    println!("Processed {} record batches", records.len());
    
    // Show statistics
    let stats = processor.stats();
    println!("Records processed: {}", stats.records_processed.load(std::sync::atomic::Ordering::Relaxed));
    println!("Bytes processed: {}", stats.bytes_processed.load(std::sync::atomic::Ordering::Relaxed));
    
    Ok(())
}

/// Demonstrate ENM XML parser
fn demo_enm_parser() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n2. ENM XML Parser Demo");
    println!("----------------------");
    
    let mut parser = EnmParser::new();
    let matcher = CounterMatcher::new();
    
    let xml_data = r#"<?xml version="1.0" encoding="UTF-8"?>
    <measData>
        <measInfo>
            <measType p="900">RRC</measType>
            <measValue>
                pmRrcConnEstabSucc=180
                pmRrcConnEstabAtt=220
                pmLteScellAddSucc=95
                pmLteScellAddAtt=100
                pmHoExeSucc=85
                pmHoExeAtt=90
            </measValue>
        </measInfo>
    </measData>"#;
    
    let cursor = std::io::Cursor::new(xml_data.as_bytes());
    let measurements = parser.parse(cursor)?;
    
    println!("Parsed {} measurements", measurements.len());
    
    for measurement in &measurements {
        println!("Measurement type: {}", measurement.measurement_type);
        println!("Granularity: {} seconds", measurement.granularity_period);
        println!("Values:");
        
        for (counter, value) in &measurement.values {
            let counter_type = matcher.match_counter(counter);
            println!("  {}: {:?} (type: {:?})", counter, value, counter_type);
        }
    }
    
    Ok(())
}

/// Demonstrate KPI calculations
fn demo_kpi_calculation() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n3. KPI Calculation Demo");
    println!("----------------------");
    
    let calculator = KpiCalculator::new();
    
    // Sample counter values
    let mut counters = HashMap::new();
    counters.insert("pmRrcConnEstabSucc".to_string(), 180.0);
    counters.insert("pmRrcConnEstabAtt".to_string(), 200.0);
    counters.insert("pmLteScellAddSucc".to_string(), 95.0);
    counters.insert("pmLteScellAddAtt".to_string(), 100.0);
    counters.insert("pmHoExeSucc".to_string(), 85.0);
    counters.insert("pmHoExeAtt".to_string(), 90.0);
    counters.insert("pmPdcpVolDlDrb".to_string(), 1_000_000_000.0); // 1 GB
    counters.insert("pmPdcpVolUlDrb".to_string(), 500_000_000.0);   // 500 MB
    
    // Calculate all KPIs
    let kpis = calculator.calculate_all_kpis(&counters);
    
    println!("Calculated KPIs:");
    for (kpi_name, value) in &kpis {
        println!("  {}: {:.2}", kpi_name, value);
    }
    
    // Show specific KPI calculations
    let mappings = KpiMappings::new();
    if let Some(rrc_success_rate) = mappings.calculate_kpi("rrc_conn_success_rate", &counters) {
        println!("\nRRC Connection Success Rate: {:.2}%", rrc_success_rate);
    }
    
    if let Some(dl_throughput) = mappings.calculate_kpi("dl_throughput", &counters) {
        println!("Downlink Throughput: {:.2} Mbps", dl_throughput);
    }
    
    Ok(())
}

/// Demonstrate feature extraction pipeline
fn demo_feature_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n4. Feature Extraction Pipeline Demo");
    println!("-----------------------------------");
    
    let config = PipelineConfig {
        normalization: NormalizationMethod::ZScore,
        feature_selection: vec![],
        window_size: 10,
        batch_size: 5,
        outlier_detection: true,
    };
    
    let mut pipeline = StreamingPipeline::new(config);
    
    // Create sample data batches
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("pmRrcConnEstabSucc", DataType::Float32, false),
        Field::new("pmRrcConnEstabAtt", DataType::Float32, false),
        Field::new("pmPdcpVolDlDrb", DataType::Float32, false),
    ]));
    
    // Generate sample data
    for i in 0..3 {
        let success_array = Arc::new(Float32Array::from(vec![
            150.0 + i as f32 * 10.0,
            160.0 + i as f32 * 10.0,
            170.0 + i as f32 * 10.0,
        ]));
        let attempt_array = Arc::new(Float32Array::from(vec![
            200.0 + i as f32 * 5.0,
            210.0 + i as f32 * 5.0,
            220.0 + i as f32 * 5.0,
        ]));
        let volume_array = Arc::new(Float32Array::from(vec![
            1000000.0 + i as f32 * 100000.0,
            1100000.0 + i as f32 * 100000.0,
            1200000.0 + i as f32 * 100000.0,
        ]));
        
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![success_array, attempt_array, volume_array],
        )?;
        
        pipeline.add_batch(batch)?;
    }
    
    // Get processed features
    let features = pipeline.flush()?;
    println!("Extracted {} feature vectors", features.len());
    
    if !features.is_empty() {
        println!("Feature vector dimensions: {}", features[0].len());
        println!("Sample features: {:?}", &features[0][..5.min(features[0].len())]);
    }
    
    Ok(())
}

/// Demonstrate tensor storage and compression
fn demo_tensor_storage() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n5. Tensor Storage and Compression Demo");
    println!("-------------------------------------");
    
    // Create tensor metadata
    let meta = TensorMeta::new(vec![10, 20], TensorDataType::Float32);
    let mut storage = TensorStorage::new(meta);
    
    // Generate sample data
    let data: Vec<f32> = (0..200).map(|i| i as f32 * 0.1).collect();
    println!("Original data size: {} elements", data.len());
    
    // Store with compression
    storage.store_compressed(&data, 6)?; // Compression level 6
    println!("Compressed size: {} bytes", storage.data.len());
    println!("Compression ratio: {:.2}x", storage.meta.compression_ratio);
    
    // Load and verify
    let loaded_data = storage.load_decompressed()?;
    println!("Loaded data size: {} elements", loaded_data.len());
    
    // Verify data integrity
    let max_error = data.iter()
        .zip(loaded_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    println!("Maximum error: {:.6}", max_error);
    
    // Test file I/O
    let temp_dir = tempdir()?;
    let tensor_file = temp_dir.path().join("tensor.bin");
    storage.store_to_file(&tensor_file)?;
    
    let loaded_storage = TensorStorage::load_from_file(&tensor_file)?;
    println!("Loaded tensor shape: {:?}", loaded_storage.meta.shape);
    
    Ok(())
}

/// Demonstrate end-to-end processing pipeline
fn demo_end_to_end_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    println!("\n6. End-to-End Processing Pipeline Demo");
    println!("--------------------------------------");
    
    // Create a complete dataset
    let mut dataset = TensorDataset::new(32, true);
    
    // Simulate processing multiple files
    for batch_idx in 0..3 {
        println!("Processing batch {}/3", batch_idx + 1);
        
        // Create tensor batch
        let mut tensor_batch = TensorBatch::new();
        
        // Add input tensor (features)
        let input_meta = TensorMeta::new(vec![32, 10], TensorDataType::Float32);
        let mut input_tensor = TensorStorage::new(input_meta);
        let input_data: Vec<f32> = (0..320).map(|i| (i as f32 + batch_idx as f32 * 320.0) * 0.01).collect();
        input_tensor.store_compressed(&input_data, 5)?;
        tensor_batch.add_input(input_tensor);
        
        // Add target tensor (labels)
        let target_meta = TensorMeta::new(vec![32, 1], TensorDataType::Float32);
        let mut target_tensor = TensorStorage::new(target_meta);
        let target_data: Vec<f32> = (0..32).map(|i| ((i + batch_idx * 32) % 2) as f32).collect();
        target_tensor.store_compressed(&target_data, 5)?;
        tensor_batch.add_target(target_tensor);
        
        // Add metadata
        tensor_batch.metadata.insert("batch_id".to_string(), batch_idx.to_string());
        tensor_batch.metadata.insert("timestamp".to_string(), "2024-01-01T00:00:00Z".to_string());
        
        dataset.add_batch(tensor_batch);
    }
    
    println!("Dataset created with {} batches", dataset.len() / 32);
    println!("Total samples: {}", dataset.len());
    
    // Save dataset
    let temp_dir = tempdir()?;
    let dataset_dir = temp_dir.path().join("dataset");
    
    for (i, batch) in dataset.batches.iter().enumerate() {
        let batch_dir = dataset_dir.join(format!("batch_{}", i));
        let mut batch_clone = batch.clone();
        batch_clone.save_to_dir(&batch_dir)?;
    }
    
    println!("Dataset saved to: {:?}", dataset_dir);
    
    // Load and verify
    let mut loaded_dataset = TensorDataset::new(32, true);
    for i in 0..dataset.batches.len() {
        let batch_dir = dataset_dir.join(format!("batch_{}", i));
        if batch_dir.exists() {
            let loaded_batch = TensorBatch::load_from_dir(&batch_dir)?;
            loaded_dataset.add_batch(loaded_batch);
        }
    }
    
    println!("Loaded dataset with {} batches", loaded_dataset.len() / 32);
    
    // Show compression statistics
    let total_compressed_size: usize = loaded_dataset.batches.iter()
        .flat_map(|batch| batch.inputs.iter().chain(batch.targets.iter()))
        .map(|tensor| tensor.data.len())
        .sum();
    
    println!("Total compressed size: {} bytes", total_compressed_size);
    
    Ok(())
}

impl Clone for TensorBatch {
    fn clone(&self) -> Self {
        Self {
            inputs: self.inputs.clone(),
            targets: self.targets.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

impl Clone for TensorStorage {
    fn clone(&self) -> Self {
        Self {
            meta: self.meta.clone(),
            data: self.data.clone(),
            mmap: None, // Don't clone memory map
        }
    }
}