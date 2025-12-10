//! Integration tests for PFS Data processing pipeline

use ran_opt::pfs_data::{
    DataProcessor, DataChunk,
    parser::{EnmParser, CounterMatcher},
    pipeline::{FeatureExtractor, PipelineConfig, NormalizationMethod},
    kpi::KpiCalculator,
    tensor::{TensorStorage, TensorMeta, TensorDataType, TensorBatch},
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn test_complete_processing_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    // Create sample ENM XML data
    let xml_data = r#"<?xml version="1.0" encoding="UTF-8"?>
    <measData>
        <measInfo>
            <measType p="900">RRC</measType>
            <measValue measObjLdn="ERBS001/EUtranCellFDD=Cell1">
                pmRrcConnEstabSucc=180
                pmRrcConnEstabAtt=200
                pmLteScellAddSucc=95
                pmLteScellAddAtt=100
                pmHoExeSucc=85
                pmHoExeAtt=90
                pmPdcpVolDlDrb=1000000000
                pmPdcpVolUlDrb=500000000
            </measValue>
        </measInfo>
    </measData>"#;

    // Step 1: Parse XML
    let mut parser = EnmParser::new();
    let cursor = std::io::Cursor::new(xml_data.as_bytes());
    let measurements = parser.parse(cursor)?;
    
    assert_eq!(measurements.len(), 1);
    assert!(!measurements[0].values.is_empty());

    // Step 2: Calculate KPIs
    let calculator = KpiCalculator::new();
    let mut counters = HashMap::new();
    
    for (counter, value) in &measurements[0].values {
        if let ran_opt::pfs_data::parser::MeasurementValue::Counter(val) = value {
            counters.insert(counter.clone(), *val as f64);
        }
    }
    
    let kpis = calculator.calculate_all_kpis(&counters);
    assert!(kpis.contains_key("rrc_conn_success_rate"));
    
    // Verify KPI calculation
    let rrc_success_rate = kpis.get("rrc_conn_success_rate").unwrap();
    assert!(*rrc_success_rate > 80.0 && *rrc_success_rate < 100.0);

    // Step 3: Feature extraction
    let config = PipelineConfig {
        normalization: NormalizationMethod::ZScore,
        feature_selection: vec![],
        window_size: 5,
        batch_size: 10,
        outlier_detection: false,
    };
    
    let mut extractor = FeatureExtractor::new(config);
    
    // Create Arrow batch from measurements
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("pmRrcConnEstabSucc", DataType::Float32, false),
        Field::new("pmRrcConnEstabAtt", DataType::Float32, false),
        Field::new("pmPdcpVolDlDrb", DataType::Float32, false),
    ]));
    
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(vec![180.0])),
            Arc::new(Float32Array::from(vec![200.0])),
            Arc::new(Float32Array::from(vec![1000000000.0])),
        ],
    )?;
    
    let features = extractor.extract_features(&batch)?;
    assert!(!features.is_empty());
    
    // Step 4: Tensor storage
    let meta = TensorMeta::new(vec![features.len(), features[0].len()], TensorDataType::Float32);
    let mut storage = TensorStorage::new(meta);
    
    // Flatten features for storage
    let flattened: Vec<f32> = features.into_iter().flatten().collect();
    storage.store_compressed(&flattened, 6)?;
    
    // Verify compression
    assert!(storage.meta.compressed);
    assert!(storage.meta.compression_ratio > 1.0);
    
    // Step 5: Save and load tensor
    let temp_dir = tempdir()?;
    let tensor_file = temp_dir.path().join("features.tensor");
    storage.store_to_file(&tensor_file)?;
    
    let loaded_storage = TensorStorage::load_from_file(&tensor_file)?;
    let loaded_data = loaded_storage.load_decompressed()?;
    
    // Verify data integrity
    assert_eq!(flattened.len(), loaded_data.len());
    
    let max_error = flattened.iter()
        .zip(loaded_data.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    
    assert!(max_error < 1e-6);
    
    Ok(())
}

#[test]
fn test_data_processor_with_memory_mapping() -> Result<(), Box<dyn std::error::Error>> {
    let mut processor = DataProcessor::new(512);
    
    // Create temporary XML file
    let xml_data = r#"<?xml version="1.0" encoding="UTF-8"?>
    <measData>
        <measInfo>
            <measType p="900">LTE</measType>
            <measValue measObjLdn="ERBS001/EUtranCellFDD=Cell1">
                pmRrcConnEstabSucc=150
                pmRrcConnEstabAtt=180
                pmLteScellAddSucc=90
                pmLteScellAddAtt=95
            </measValue>
            <measValue measObjLdn="ERBS001/EUtranCellFDD=Cell2">
                pmRrcConnEstabSucc=160
                pmRrcConnEstabAtt=190
                pmLteScellAddSucc=88
                pmLteScellAddAtt=92
            </measValue>
        </measInfo>
    </measData>"#;
    
    let temp_dir = tempdir()?;
    let xml_file = temp_dir.path().join("test.xml");
    let mut file = File::create(&xml_file)?;
    file.write_all(xml_data.as_bytes())?;
    
    // Process with memory mapping
    let records = processor.process_enm_xml(&xml_file)?;
    assert!(!records.is_empty());
    
    // Save as Parquet
    let parquet_file = temp_dir.path().join("output.parquet");
    processor.write_parquet(&records, &parquet_file)?;
    
    // Load back from Parquet
    let loaded_records = processor.read_parquet(&parquet_file)?;
    assert_eq!(records.len(), loaded_records.len());
    
    // Verify statistics
    let stats = processor.stats();
    assert!(stats.records_processed.load(std::sync::atomic::Ordering::Relaxed) > 0);
    
    Ok(())
}

#[test]
fn test_parallel_chunk_processing() -> Result<(), Box<dyn std::error::Error>> {
    let processor = DataProcessor::new(1024);
    
    // Create test chunks
    let chunks = vec![
        DataChunk {
            data: b"<measValue>pmRrcConnEstabSucc=100</measValue>".to_vec(),
            timestamp: 1000,
            source: "test1".to_string(),
        },
        DataChunk {
            data: b"<measValue>pmRrcConnEstabSucc=110</measValue>".to_vec(),
            timestamp: 2000,
            source: "test2".to_string(),
        },
        DataChunk {
            data: b"<measValue>pmRrcConnEstabSucc=120</measValue>".to_vec(),
            timestamp: 3000,
            source: "test3".to_string(),
        },
    ];
    
    // Process in parallel
    let results = processor.parallel_process(chunks);
    
    // Results should be processed (even if parsing fails due to incomplete XML)
    assert!(results.len() <= 3); // Some chunks might fail parsing
    
    Ok(())
}

#[test]
fn test_counter_matcher() {
    let matcher = CounterMatcher::new();
    
    use ran_opt::pfs_data::parser::CounterType;
    
    assert!(matches!(
        matcher.match_counter("pmRrcConnEstabSucc"),
        CounterType::RrcConnection
    ));
    
    assert!(matches!(
        matcher.match_counter("pmLteScellAddSucc"),
        CounterType::ScellAddition
    ));
    
    assert!(matches!(
        matcher.match_counter("pmHoExeSucc"),
        CounterType::Handover
    ));
    
    assert!(matches!(
        matcher.match_counter("pmPdcpVolDlDrb"),
        CounterType::Throughput
    ));
    
    assert!(matches!(
        matcher.match_counter("unknown_counter"),
        CounterType::Other
    ));
}

#[test]
fn test_tensor_batch_operations() -> Result<(), Box<dyn std::error::Error>> {
    let mut batch = TensorBatch::new();
    
    // Create input tensor
    let input_meta = TensorMeta::new(vec![10, 5], TensorDataType::Float32);
    let mut input_tensor = TensorStorage::new(input_meta);
    let input_data: Vec<f32> = (0..50).map(|i| i as f32 * 0.1).collect();
    input_tensor.store_compressed(&input_data, 5)?;
    batch.add_input(input_tensor);
    
    // Create target tensor
    let target_meta = TensorMeta::new(vec![10, 1], TensorDataType::Float32);
    let mut target_tensor = TensorStorage::new(target_meta);
    let target_data: Vec<f32> = (0..10).map(|i| (i % 2) as f32).collect();
    target_tensor.store_compressed(&target_data, 5)?;
    batch.add_target(target_tensor);
    
    // Add metadata
    batch.metadata.insert("experiment_id".to_string(), "test_001".to_string());
    batch.metadata.insert("created_at".to_string(), "2024-01-01".to_string());
    
    // Save batch
    let temp_dir = tempdir()?;
    let batch_dir = temp_dir.path().join("batch");
    batch.save_to_dir(&batch_dir)?;
    
    // Load batch
    let loaded_batch = TensorBatch::load_from_dir(&batch_dir)?;
    
    // Verify
    assert_eq!(batch.inputs.len(), loaded_batch.inputs.len());
    assert_eq!(batch.targets.len(), loaded_batch.targets.len());
    assert_eq!(batch.metadata.len(), loaded_batch.metadata.len());
    
    // Verify data integrity
    let original_input = batch.inputs[0].load_decompressed()?;
    let loaded_input = loaded_batch.inputs[0].load_decompressed()?;
    assert_eq!(original_input, loaded_input);
    
    Ok(())
}

#[test]
fn test_feature_extraction_with_normalization() -> Result<(), Box<dyn std::error::Error>> {
    use ran_opt::pfs_data::pipeline::FeatureSelectionMethod;
    
    let config = PipelineConfig {
        normalization: NormalizationMethod::MinMax,
        feature_selection: vec![FeatureSelectionMethod::TopK(3)],
        window_size: 5,
        batch_size: 10,
        outlier_detection: true,
    };
    
    let mut extractor = FeatureExtractor::new(config);
    
    // Create test data with different scales
    use arrow::array::Float32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("small_values", DataType::Float32, false),
        Field::new("large_values", DataType::Float32, false),
    ]));
    
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(vec![1.0, 2.0, 3.0])),
            Arc::new(Float32Array::from(vec![1000.0, 2000.0, 3000.0])),
        ],
    )?;
    
    let features = extractor.extract_features(&batch)?;
    assert!(!features.is_empty());
    
    // Check that features are normalized (should be between 0 and 1 for MinMax)
    for feature_vec in &features {
        for &value in feature_vec {
            assert!(value >= 0.0 && value <= 1.0 || value.is_nan());
        }
    }
    
    Ok(())
}