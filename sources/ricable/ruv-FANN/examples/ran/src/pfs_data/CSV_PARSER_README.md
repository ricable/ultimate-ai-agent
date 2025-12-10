# Enhanced CSV Data Parser for RAN Intelligence

## Overview

The Enhanced CSV Data Parser is a robust, production-ready Rust module designed to parse and process `fanndata.csv` files containing real cellular network data. It provides comprehensive error handling, data validation, parallel processing capabilities, and detailed reporting.

## Key Features

### 🚀 Performance
- **Parallel Processing**: Utilizes Rayon for multi-threaded CSV parsing
- **Streaming**: Memory-efficient processing of large files
- **Configurable Batching**: Optimizable batch sizes for different file sizes
- **Performance Metrics**: Real-time tracking of processing speed and memory usage

### 🔍 Data Validation
- **Range Validation**: Automatic validation of all numeric fields against realistic ranges
- **Type Safety**: Strong type checking with comprehensive error handling
- **Missing Data Handling**: Graceful handling of empty or missing fields
- **Data Quality Scoring**: Automatic calculation of overall data quality metrics

### 🛡️ Error Handling
- **Custom Error Types**: Detailed error classification and reporting
- **Error Recovery**: Configurable error thresholds with graceful degradation
- **Validation Errors**: Separate tracking of validation vs. parsing errors
- **Production Ready**: Robust error handling suitable for production environments

### 📊 Reporting & Export
- **Multiple Export Formats**: JSON, CSV summary, feature vectors for ML
- **Quality Reports**: Comprehensive data quality and processing performance reports
- **Anomaly Detection**: Built-in anomaly detection with severity scoring
- **Statistics Tracking**: Detailed parsing and processing statistics

## Data Structure

The parser handles the 101-column `fanndata.csv` structure with the following key fields:

### Cell Identification
- `CODE_ELT_ENODEB`: eNodeB identifier
- `ENODEB`: eNodeB name
- `CODE_ELT_CELLULE`: Cell identifier
- `CELLULE`: Cell name
- `SYS.BANDE`: Frequency band
- `SYS.NB_BANDES`: Number of bands

### Key Performance Indicators
- `CELL_AVAILABILITY_%`: Cell availability percentage
- `VOLTE_TRAFFIC (ERL)`: VoLTE traffic in Erlangs
- `RRC_CONNECTED_USERS_AVERAGE`: Average connected users
- `UL_VOLUME_PDCP_GBYTES`: Uplink volume in GB
- `DL_VOLUME_PDCP_GBYTES`: Downlink volume in GB

### Signal Quality Metrics
- `SINR_PUSCH_AVG`: Average SINR for PUSCH
- `SINR_PUCCH_AVG`: Average SINR for PUCCH
- `UL RSSI PUCCH/PUSCH/TOTAL`: Uplink RSSI measurements
- `MAC_DL_BLER/MAC_UL_BLER`: Block error rates

### Performance Metrics
- `&_AVE_4G_LTE_DL_USER_THRPUT`: Downlink user throughput
- `&_AVE_4G_LTE_UL_USER_THRPUT`: Uplink user throughput
- `ERAB_DROP_RATE_QCI_5/8`: E-RAB drop rates by QCI
- `LTE_INTRA_FREQ_HO_SR/LTE_INTER_FREQ_HO_SR`: Handover success rates

### 5G/ENDC Metrics
- `ENDC_ESTABLISHMENT_ATT/SUCC`: ENDC establishment attempts/successes
- `ENDC_SETUP_SR`: ENDC setup success rate
- `ENDC_SCG_FAILURE_RATIO`: SCG failure ratio

## Usage Examples

### Basic Usage

```rust
use crate::pfs_data::csv_data_parser::CsvDataParser;

// Create parser with default configuration
let mut parser = CsvDataParser::new();

// Parse CSV file
let dataset = parser.parse_csv_file("fanndata.csv")?;

// Access parsed data
println!("Parsed {} rows", dataset.rows.len());
println!("Found {} unique cells", dataset.stats.unique_cells);
println!("Data quality score: {:.1}%", dataset.stats.data_quality_score);
```

### Production Configuration

```rust
use crate::pfs_data::csv_data_parser::{CsvDataParser, CsvParsingConfig, ValidationRules};

// Create production-ready configuration
let config = CsvParsingConfig {
    delimiter: b';',
    batch_size: 2000,
    max_errors_before_abort: 50,
    parallel_processing: true,
    validate_data_ranges: true,
    skip_empty_rows: true,
    strict_column_count: false,  // Be flexible with malformed data
    expected_column_count: 101,
    ..Default::default()
};

// Custom validation rules
let validation_rules = ValidationRules {
    availability_range: (0.0, 100.0),
    throughput_range: (0.0, 5000.0),
    sinr_range: (-30.0, 40.0),
    rssi_range: (-150.0, -30.0),
    error_rate_range: (0.0, 50.0),
    ..Default::default()
};

let mut parser = CsvDataParser::with_config(config);
parser.set_validation_rules(validation_rules);

let dataset = parser.parse_csv_file("fanndata.csv")?;
```

### Error Handling

```rust
use crate::pfs_data::csv_data_parser::{CsvDataParser, CsvParsingError};

let mut parser = CsvDataParser::new();

match parser.parse_csv_file("fanndata.csv") {
    Ok(dataset) => {
        println!("✅ Parsing successful!");
        println!("   📊 Total rows: {}", dataset.rows.len());
        println!("   🔍 Validation errors: {}", dataset.stats.validation_errors);
        println!("   🔢 Data type errors: {}", dataset.stats.data_type_errors);
    }
    Err(CsvParsingError::ValidationError(msg)) => {
        eprintln!("❌ Validation error: {}", msg);
    }
    Err(CsvParsingError::IoError(e)) => {
        eprintln!("❌ File error: {}", e);
    }
    Err(e) => {
        eprintln!("❌ Parsing error: {}", e);
    }
}
```

### Data Export

```rust
use crate::pfs_data::csv_data_parser::ExportFormat;

let parser = CsvDataParser::new();
let dataset = parser.parse_csv_file("fanndata.csv")?;

// Export full dataset as JSON
parser.export_data(&dataset, ExportFormat::Json, "full_data.json")?;

// Export summary as CSV
parser.export_data(&dataset, ExportFormat::CsvSummary, "summary.csv")?;

// Export feature vectors for ML
parser.export_data(&dataset, ExportFormat::FeatureVectors, "features.json")?;
```

### Quality Reporting

```rust
let parser = CsvDataParser::new();
let dataset = parser.parse_csv_file("fanndata.csv")?;

// Generate comprehensive quality report
let quality_report = parser.generate_quality_report(&dataset);

println!("📋 Data Quality Report:");
println!("   📊 Total records: {}", quality_report.total_records);
println!("   📈 Data completeness: {:.2}%", quality_report.data_completeness * 100.0);
println!("   🎯 Quality score: {:.1}%", quality_report.data_quality_score);
println!("   🚨 Critical issues: {}", quality_report.critical_issues);
println!("   ⚡ Processing speed: {:.0} rows/sec", quality_report.processing_performance.rows_per_second);
```

## Configuration Options

### CsvParsingConfig

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `delimiter` | `u8` | `b';'` | CSV field delimiter |
| `has_headers` | `bool` | `true` | Whether CSV has header row |
| `batch_size` | `usize` | `1000` | Batch size for processing |
| `max_errors_before_abort` | `usize` | `100` | Maximum errors before aborting |
| `parallel_processing` | `bool` | `true` | Enable parallel processing |
| `validate_data_ranges` | `bool` | `true` | Enable data validation |
| `skip_empty_rows` | `bool` | `true` | Skip empty rows |
| `strict_column_count` | `bool` | `true` | Enforce exact column count |
| `expected_column_count` | `usize` | `101` | Expected number of columns |

### ValidationRules

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `availability_range` | `(f64, f64)` | `(0.0, 100.0)` | Valid availability percentage range |
| `throughput_range` | `(f64, f64)` | `(0.0, 10000.0)` | Valid throughput range (Mbps) |
| `sinr_range` | `(f64, f64)` | `(-20.0, 50.0)` | Valid SINR range (dB) |
| `rssi_range` | `(f64, f64)` | `(-140.0, -40.0)` | Valid RSSI range (dBm) |
| `error_rate_range` | `(f64, f64)` | `(0.0, 100.0)` | Valid error rate range (%) |
| `user_count_range` | `(u32, u32)` | `(0, 10000)` | Valid user count range |
| `handover_rate_range` | `(f64, f64)` | `(0.0, 100.0)` | Valid handover success rate range |
| `latency_range` | `(f64, f64)` | `(0.0, 1000.0)` | Valid latency range (ms) |

## Performance Tuning

### Memory Usage
- **Small files (<1MB)**: `batch_size: 500`, `parallel_processing: false`
- **Medium files (1-100MB)**: `batch_size: 1000`, `parallel_processing: true`
- **Large files (>100MB)**: `batch_size: 2000`, `parallel_processing: true`
- **Memory constrained**: `batch_size: 200`, `parallel_processing: false`

### Error Tolerance
- **Strict validation**: `max_errors_before_abort: 10`, `validate_data_ranges: true`
- **Lenient parsing**: `max_errors_before_abort: 1000`, `validate_data_ranges: false`
- **Production**: `max_errors_before_abort: 50`, `validate_data_ranges: true`

## Output Data Structures

### ParsedCsvDataset
Contains the complete parsed dataset with:
- `rows`: Vector of parsed CSV rows
- `stats`: Parsing statistics and performance metrics
- `neural_results`: Neural processing results
- `anomaly_summary`: Summary of detected anomalies
- `feature_vectors`: ML-ready feature vectors

### DataQualityReport
Comprehensive quality assessment including:
- Data completeness and quality scores
- Error counts and processing performance
- Cell statistics and anomaly rates
- Memory usage and processing speed metrics

## Integration with Neural Processing

The parser integrates seamlessly with the existing neural data processor:

```rust
// Parse CSV data
let dataset = parser.parse_csv_file("fanndata.csv")?;

// Neural processing is automatically applied
println!("Neural results: {}", dataset.neural_results.len());

// Access feature vectors for ML training
let features = &dataset.feature_vectors;
println!("AFM features: {} vectors", features.afm_features.len());
println!("DTM features: {} vectors", features.dtm_features.len());
```

## Anomaly Detection

Built-in anomaly detection identifies:
- **Availability anomalies**: Cells with <95% availability
- **Throughput anomalies**: Low throughput performance
- **Quality anomalies**: Poor signal quality (low SINR)
- **Error rate anomalies**: High block error rates
- **Handover anomalies**: Poor handover performance

Each anomaly is scored and classified by severity.

## Testing

Comprehensive test suite includes:
- Unit tests for all parsing functions
- Integration tests with sample data
- Error handling validation
- Performance benchmarking
- Configuration validation

Run tests with:
```bash
cargo test csv_data_parser
```

## Dependencies

- `csv`: CSV parsing and reading
- `serde`: Serialization and deserialization
- `rayon`: Parallel processing
- `std::collections`: HashMap for column mapping
- `std::time`: Performance timing

## Future Enhancements

- [ ] Support for streaming very large files
- [ ] Advanced anomaly detection algorithms
- [ ] Real-time processing capabilities
- [ ] Additional export formats (Parquet, Arrow)
- [ ] Enhanced visualization support
- [ ] Automatic schema detection
- [ ] Data profiling and statistics

## License

This module is part of the ruv-FANN project and follows the same licensing terms.