//! PFS Data Processing Module
//! 
//! High-performance data ingestion and processing pipeline for neural networks
//! with specialized support for Ericsson ENM XML parsing and columnar data storage.

use arrow::array::{ArrayRef, Float32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use memmap2::{Mmap, MmapMut, MmapOptions};
use parquet::arrow::{ArrowReader, ArrowWriter, ParquetFileArrowReader};
use parquet::file::reader::{FileReader, SerializedFileReader};
use quick_xml::events::Event;
use quick_xml::Reader;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use crossbeam::queue::ArrayQueue;

pub mod parser;
pub mod pipeline;
pub mod kpi;
pub mod tensor;

/// Main data processor with optimized memory management
pub struct DataProcessor {
    /// Lock-free queue for concurrent data processing
    queue: Arc<ArrayQueue<DataChunk>>,
    /// Memory-mapped file handles
    mmap_cache: HashMap<String, Mmap>,
    /// KPI counter mappings
    kpi_mappings: kpi::KpiMappings,
    /// Processing statistics
    stats: ProcessingStats,
}

/// Atomic statistics for monitoring
#[derive(Default)]
pub struct ProcessingStats {
    pub records_processed: AtomicU64,
    pub bytes_processed: AtomicU64,
    pub parse_errors: AtomicU64,
}

/// Data chunk for processing
#[derive(Clone)]
pub struct DataChunk {
    pub data: Vec<u8>,
    pub timestamp: i64,
    pub source: String,
}

impl DataProcessor {
    /// Create a new data processor with specified queue size
    pub fn new(queue_size: usize) -> Self {
        Self {
            queue: Arc::new(ArrayQueue::new(queue_size)),
            mmap_cache: HashMap::new(),
            kpi_mappings: kpi::KpiMappings::default(),
            stats: ProcessingStats::default(),
        }
    }

    /// Process ENM XML file using memory-mapped I/O
    pub fn process_enm_xml<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Store in cache for reuse
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.mmap_cache.insert(path_str.clone(), mmap);
        
        // Get reference to mmap data
        let mmap_ref = &self.mmap_cache[&path_str];
        
        // Parse XML using zero-copy reader
        let cursor = Cursor::new(&mmap_ref[..]);
        let mut reader = Reader::from_reader(cursor);
        reader.trim_text(true);
        
        let mut buf = Vec::new();
        let mut records = Vec::new();
        let mut current_record = HashMap::new();
        
        loop {
            match reader.read_event(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name() == b"measValue" {
                        current_record.clear();
                    }
                }
                Ok(Event::Text(e)) => {
                    // Extract KPI values
                    let text = e.unescape_and_decode(&reader)?;
                    if let Some((key, value)) = text.split_once('=') {
                        current_record.insert(key.to_string(), value.to_string());
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name() == b"measValue" && !current_record.is_empty() {
                        // Convert to columnar format
                        if let Ok(batch) = self.convert_to_record_batch(&current_record) {
                            records.push(batch);
                        }
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => {
                    self.stats.parse_errors.fetch_add(1, Ordering::Relaxed);
                    eprintln!("Error at position {}: {:?}", reader.buffer_position(), e);
                }
                _ => {}
            }
            buf.clear();
        }
        
        self.stats.records_processed.fetch_add(records.len() as u64, Ordering::Relaxed);
        Ok(records)
    }

    /// Convert parsed data to Arrow RecordBatch
    fn convert_to_record_batch(&self, data: &HashMap<String, String>) -> Result<RecordBatch, Box<dyn std::error::Error>> {
        let mut fields = vec![];
        let mut columns: Vec<ArrayRef> = vec![];
        
        // Map KPIs to columns
        for (kpi_name, kpi_info) in &self.kpi_mappings.mappings {
            fields.push(Field::new(kpi_name, kpi_info.data_type.clone(), true));
            
            if let Some(value_str) = data.get(kpi_name) {
                match &kpi_info.data_type {
                    DataType::Float32 => {
                        let value = value_str.parse::<f32>().unwrap_or(0.0);
                        columns.push(Arc::new(Float32Array::from(vec![value])));
                    }
                    DataType::Int64 => {
                        let value = value_str.parse::<i64>().unwrap_or(0);
                        columns.push(Arc::new(Int64Array::from(vec![value])));
                    }
                    _ => {
                        columns.push(Arc::new(StringArray::from(vec![value_str.as_str()])));
                    }
                }
            } else {
                // Insert null value
                match &kpi_info.data_type {
                    DataType::Float32 => {
                        columns.push(Arc::new(Float32Array::from(vec![None as Option<f32>])));
                    }
                    DataType::Int64 => {
                        columns.push(Arc::new(Int64Array::from(vec![None as Option<i64>])));
                    }
                    _ => {
                        columns.push(Arc::new(StringArray::from(vec![None as Option<&str>])));
                    }
                }
            }
        }
        
        let schema = Arc::new(Schema::new(fields));
        RecordBatch::try_new(schema, columns).map_err(|e| e.into())
    }

    /// Write data to Parquet with zero-copy optimization
    pub fn write_parquet<P: AsRef<Path>>(&self, batches: &[RecordBatch], path: P) -> Result<(), Box<dyn std::error::Error>> {
        if batches.is_empty() {
            return Ok(());
        }
        
        let file = File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, batches[0].schema(), None)?;
        
        for batch in batches {
            writer.write(batch)?;
        }
        
        writer.close()?;
        Ok(())
    }

    /// Read Parquet file using memory-mapped I/O
    pub fn read_parquet<P: AsRef<Path>>(&mut self, path: P) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
        let file = File::open(&path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        // Store in cache
        let path_str = path.as_ref().to_string_lossy().to_string();
        self.mmap_cache.insert(path_str.clone(), mmap);
        
        // Create reader from memory-mapped data
        let mmap_ref = &self.mmap_cache[&path_str];
        let cursor = Cursor::new(&mmap_ref[..]);
        let file_reader = SerializedFileReader::new(cursor)?;
        let mut arrow_reader = ParquetFileArrowReader::new(Arc::new(file_reader));
        
        let mut batches = vec![];
        let mut batch_reader = arrow_reader.get_record_reader(1024)?;
        
        while let Some(batch) = batch_reader.next() {
            batches.push(batch?);
        }
        
        Ok(batches)
    }

    /// Process data chunks in parallel using rayon
    pub fn parallel_process(&self, chunks: Vec<DataChunk>) -> Vec<RecordBatch> {
        chunks
            .par_iter()
            .filter_map(|chunk| {
                // Process each chunk independently
                self.process_chunk(chunk).ok()
            })
            .flatten()
            .collect()
    }

    /// Process a single data chunk
    fn process_chunk(&self, chunk: &DataChunk) -> Result<Vec<RecordBatch>, Box<dyn std::error::Error>> {
        let mut reader = Reader::from_reader(Cursor::new(&chunk.data));
        let mut records = vec![];
        let mut buf = Vec::new();
        let mut current_record = HashMap::new();
        
        // Similar XML parsing logic as process_enm_xml but for chunks
        loop {
            match reader.read_event(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    if e.name() == b"measValue" {
                        current_record.clear();
                    }
                }
                Ok(Event::End(ref e)) => {
                    if e.name() == b"measValue" && !current_record.is_empty() {
                        if let Ok(batch) = self.convert_to_record_batch(&current_record) {
                            records.push(batch);
                        }
                    }
                }
                Ok(Event::Eof) => break,
                _ => {}
            }
            buf.clear();
        }
        
        self.stats.bytes_processed.fetch_add(chunk.data.len() as u64, Ordering::Relaxed);
        Ok(records)
    }

    /// Get processing statistics
    pub fn stats(&self) -> &ProcessingStats {
        &self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_processor_creation() {
        let processor = DataProcessor::new(1024);
        assert_eq!(processor.stats.records_processed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_data_chunk() {
        let chunk = DataChunk {
            data: vec![1, 2, 3, 4],
            timestamp: 1234567890,
            source: "test".to_string(),
        };
        assert_eq!(chunk.data.len(), 4);
    }
}