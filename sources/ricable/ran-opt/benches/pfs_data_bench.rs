//! PFS Data Processing Benchmarks
//! 
//! Performance benchmarks for the data processing pipeline

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ran_opt::pfs_data::{
    DataProcessor, DataChunk,
    parser::EnmParser,
    pipeline::{FeatureExtractor, PipelineConfig, NormalizationMethod},
    tensor::{TensorStorage, TensorMeta, TensorDataType},
};
use std::io::Cursor;
use arrow::array::Float32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

fn benchmark_xml_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("xml_parsing");
    
    // Generate XML data of different sizes
    let sizes = vec![1, 10, 100, 1000];
    
    for size in sizes {
        let xml_data = generate_enm_xml(size);
        
        group.bench_with_input(
            BenchmarkId::new("enm_parser", size),
            &xml_data,
            |b, data| {
                b.iter(|| {
                    let mut parser = EnmParser::new();
                    let cursor = Cursor::new(data.as_bytes());
                    black_box(parser.parse(cursor).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction");
    
    let config = PipelineConfig {
        normalization: NormalizationMethod::ZScore,
        feature_selection: vec![],
        window_size: 10,
        batch_size: 5,
        outlier_detection: false,
    };
    
    let batch_sizes = vec![10, 100, 1000, 10000];
    
    for batch_size in batch_sizes {
        let batch = create_test_batch(batch_size);
        
        group.bench_with_input(
            BenchmarkId::new("feature_extractor", batch_size),
            &batch,
            |b, batch| {
                b.iter(|| {
                    let mut extractor = FeatureExtractor::new(config.clone());
                    black_box(extractor.extract_features(batch).unwrap());
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_tensor_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_compression");
    
    let sizes = vec![1000, 10000, 100000];
    let compression_levels = vec![1, 6, 22];
    
    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.1).collect();
        let meta = TensorMeta::new(vec![size], TensorDataType::Float32);
        
        for level in &compression_levels {
            group.bench_with_input(
                BenchmarkId::new("compress", format!("{}_{}", size, level)),
                &(data.clone(), meta.clone(), *level),
                |b, (data, meta, level)| {
                    b.iter(|| {
                        let mut storage = TensorStorage::new(meta.clone());
                        black_box(storage.store_compressed(data, *level as u32).unwrap());
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn benchmark_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_processing");
    
    let chunk_counts = vec![1, 4, 16, 64];
    
    for chunk_count in chunk_counts {
        let chunks = create_test_chunks(chunk_count, 1000);
        
        group.bench_with_input(
            BenchmarkId::new("parallel_chunks", chunk_count),
            &chunks,
            |b, chunks| {
                b.iter(|| {
                    let processor = DataProcessor::new(1024);
                    black_box(processor.parallel_process(chunks.clone()));
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_memory_mapped_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_mapped_io");
    
    let file_sizes = vec![1024, 10240, 102400]; // 1KB, 10KB, 100KB
    
    for size in file_sizes {
        let data: Vec<f32> = (0..size/4).map(|i| i as f32).collect();
        let meta = TensorMeta::new(vec![size/4], TensorDataType::Float32);
        
        group.bench_with_input(
            BenchmarkId::new("mmap_write_read", size),
            &(data, meta),
            |b, (data, meta)| {
                b.iter(|| {
                    let mut storage = TensorStorage::new(meta.clone());
                    storage.store_compressed(data, 6).unwrap();
                    
                    let temp_file = tempfile::NamedTempFile::new().unwrap();
                    storage.store_to_file(temp_file.path()).unwrap();
                    
                    let loaded = TensorStorage::load_from_file(temp_file.path()).unwrap();
                    black_box(loaded);
                });
            },
        );
    }
    
    group.finish();
}

// Helper functions

fn generate_enm_xml(measurement_count: usize) -> String {
    let mut xml = String::from(r#"<?xml version="1.0" encoding="UTF-8"?>
<measData>
    <measInfo>
        <measType p="900">RRC</measType>"#);
    
    for i in 0..measurement_count {
        xml.push_str(&format!(r#"
        <measValue measObjLdn="ERBS{:03}/EUtranCellFDD=Cell{}">
            pmRrcConnEstabSucc={}
            pmRrcConnEstabAtt={}
            pmLteScellAddSucc={}
            pmLteScellAddAtt={}
            pmHoExeSucc={}
            pmHoExeAtt={}
            pmPdcpVolDlDrb={}
            pmPdcpVolUlDrb={}
        </measValue>"#,
            i % 100,
            i % 10,
            150 + i % 50,
            200 + i % 30,
            90 + i % 20,
            100 + i % 10,
            80 + i % 15,
            90 + i % 8,
            1000000 + i * 10000,
            500000 + i * 5000,
        ));
    }
    
    xml.push_str(r#"
    </measInfo>
</measData>"#);
    
    xml
}

fn create_test_batch(row_count: usize) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("pmRrcConnEstabSucc", DataType::Float32, false),
        Field::new("pmRrcConnEstabAtt", DataType::Float32, false),
        Field::new("pmLteScellAddSucc", DataType::Float32, false),
        Field::new("pmLteScellAddAtt", DataType::Float32, false),
        Field::new("pmPdcpVolDlDrb", DataType::Float32, false),
    ]));
    
    let success_data: Vec<f32> = (0..row_count).map(|i| 150.0 + (i % 50) as f32).collect();
    let attempt_data: Vec<f32> = (0..row_count).map(|i| 200.0 + (i % 30) as f32).collect();
    let scell_success_data: Vec<f32> = (0..row_count).map(|i| 90.0 + (i % 20) as f32).collect();
    let scell_attempt_data: Vec<f32> = (0..row_count).map(|i| 100.0 + (i % 10) as f32).collect();
    let volume_data: Vec<f32> = (0..row_count).map(|i| 1000000.0 + (i * 10000) as f32).collect();
    
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(success_data)),
            Arc::new(Float32Array::from(attempt_data)),
            Arc::new(Float32Array::from(scell_success_data)),
            Arc::new(Float32Array::from(scell_attempt_data)),
            Arc::new(Float32Array::from(volume_data)),
        ],
    ).unwrap()
}

fn create_test_chunks(chunk_count: usize, chunk_size: usize) -> Vec<DataChunk> {
    (0..chunk_count)
        .map(|i| {
            let xml_data = generate_enm_xml(chunk_size / 100);
            DataChunk {
                data: xml_data.into_bytes(),
                timestamp: i as i64,
                source: format!("test_source_{}", i),
            }
        })
        .collect()
}

criterion_group!(
    benches,
    benchmark_xml_parsing,
    benchmark_feature_extraction,
    benchmark_tensor_compression,
    benchmark_parallel_processing,
    benchmark_memory_mapped_io
);
criterion_main!(benches);