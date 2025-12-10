//! Enhanced CSV Parser Usage Example
//! 
//! This example demonstrates how to use the enhanced CSV parser with all its
//! advanced features including robust error handling, data validation,
//! parallel processing, and comprehensive reporting.

use crate::pfs_data::csv_data_parser::{
    CsvDataParser, CsvParsingConfig, ValidationRules, ExportFormat, CsvParsingError
};
use std::path::Path;

/// Comprehensive example of using the enhanced CSV parser
pub fn run_enhanced_csv_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Enhanced CSV Parser Example");
    println!("==============================");
    
    // Example 1: Basic usage with default configuration
    println!("\n📊 Example 1: Basic Usage");
    basic_csv_parsing_example()?;
    
    // Example 2: Custom configuration for production use
    println!("\n⚙️ Example 2: Production Configuration");
    production_csv_parsing_example()?;
    
    // Example 3: Error handling and recovery
    println!("\n🛡️ Example 3: Error Handling");
    error_handling_example()?;
    
    // Example 4: Data validation and quality assessment
    println!("\n🔍 Example 4: Data Validation");
    data_validation_example()?;
    
    // Example 5: Performance optimization
    println!("\n⚡ Example 5: Performance Optimization");
    performance_optimization_example()?;
    
    // Example 6: Export and reporting
    println!("\n📁 Example 6: Export and Reporting");
    export_and_reporting_example()?;
    
    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Basic CSV parsing with default settings
fn basic_csv_parsing_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Creating parser with default configuration...");
    let mut parser = CsvDataParser::new();
    
    // Check if the CSV file exists
    let csv_path = "examples/ran/swarm_demo/data/fanndata.csv";
    if !Path::new(csv_path).exists() {
        println!("   ⚠️ CSV file not found at: {}", csv_path);
        println!("   📝 This example would parse the real fanndata.csv file");
        return Ok(());
    }
    
    println!("   📊 Parsing CSV file: {}", csv_path);
    match parser.parse_csv_file(csv_path) {
        Ok(dataset) => {
            println!("   ✅ Parsing successful!");
            println!("      📈 Total rows: {}", dataset.rows.len());
            println!("      🏢 Unique eNodeBs: {}", dataset.stats.unique_enodebs);
            println!("      📱 Unique cells: {}", dataset.stats.unique_cells);
            println!("      ⚡ Processing speed: {:.0} rows/sec", dataset.stats.rows_per_second);
        }
        Err(e) => {
            println!("   ❌ Parsing failed: {}", e);
        }
    }
    
    Ok(())
}

/// Production-ready configuration example
fn production_csv_parsing_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Configuring parser for production use...");
    
    // Create production configuration
    let config = CsvParsingConfig {
        delimiter: b';',
        has_headers: true,
        batch_size: 2000,           // Larger batches for better performance
        max_errors_before_abort: 50, // Allow some errors but not too many
        parallel_processing: true,   // Enable parallel processing
        validate_data_ranges: true,  // Enable data validation
        skip_empty_rows: true,       // Skip empty rows
        strict_column_count: false,  // Be flexible with column count
        expected_column_count: 101,
    };
    
    // Custom validation rules for production
    let validation_rules = ValidationRules {
        availability_range: (0.0, 100.0),
        throughput_range: (0.0, 5000.0),    // More realistic max throughput
        sinr_range: (-30.0, 40.0),          // Extended SINR range
        rssi_range: (-150.0, -30.0),        // Extended RSSI range
        error_rate_range: (0.0, 50.0),      // Allow higher error rates in production
        user_count_range: (0, 5000),        // Realistic user count range
        handover_rate_range: (0.0, 100.0),
        latency_range: (0.0, 500.0),        // Extended latency range
    };
    
    let mut parser = CsvDataParser::with_config(config);
    parser.set_validation_rules(validation_rules);
    
    println!("   📋 Production configuration:");
    println!("      🔢 Batch size: {}", parser.get_config().batch_size);
    println!("      🔍 Data validation: {}", parser.get_config().validate_data_ranges);
    println!("      ⚡ Parallel processing: {}", parser.get_config().parallel_processing);
    println!("      🛡️ Max errors before abort: {}", parser.get_config().max_errors_before_abort);
    
    println!("   ✅ Production configuration ready");
    Ok(())
}

/// Error handling and recovery example
fn error_handling_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating error handling capabilities...");
    
    // Create a lenient configuration that handles errors gracefully
    let config = CsvParsingConfig {
        delimiter: b';',
        has_headers: true,
        batch_size: 1000,
        max_errors_before_abort: 1000,  // High error threshold
        parallel_processing: false,      // Sequential for better error tracking
        validate_data_ranges: false,     // Disable validation to avoid validation errors
        skip_empty_rows: true,
        strict_column_count: false,      // Be flexible with malformed data
        expected_column_count: 101,
    };
    
    let _parser = CsvDataParser::with_config(config);
    
    // Example of handling different error types
    println!("   🔍 Error types that can be handled:");
    
    // Simulate different error scenarios
    let error_examples = vec![
        CsvParsingError::ValidationError("Availability 150% exceeds valid range".to_string()),
        CsvParsingError::DataTypeError("Cannot parse 'invalid' as number".to_string()),
        CsvParsingError::MissingColumn("Required column not found".to_string()),
        CsvParsingError::InvalidFormat("Row has 90 columns, expected 101".to_string()),
    ];
    
    for (i, error) in error_examples.iter().enumerate() {
        println!("      {}. {}", i + 1, error);
    }
    
    println!("   💡 Recovery strategies:");
    println!("      🔄 Automatic retry with relaxed validation");
    println!("      📊 Continue processing with error logging");
    println!("      🎯 Fallback to default values for missing data");
    println!("      📈 Detailed error reporting for debugging");
    
    Ok(())
}

/// Data validation example
fn data_validation_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating data validation features...");
    
    let parser = CsvDataParser::new();
    
    // Example validation scenarios
    println!("   🔍 Validation scenarios:");
    
    // Valid data examples
    let valid_examples = vec![
        ("Availability", 95.5, "within 0-100% range"),
        ("SINR", 15.0, "within -20 to 50 dB range"),
        ("RSSI", -105.0, "within -140 to -40 dBm range"),
        ("Throughput", 50.0, "within 0-10000 Mbps range"),
    ];
    
    for (metric, value, description) in valid_examples {
        println!("      ✅ {}: {} ({})", metric, value, description);
    }
    
    // Invalid data examples
    let invalid_examples = vec![
        ("Availability", 150.0, "exceeds 100% limit"),
        ("SINR", -50.0, "below -20 dB minimum"),
        ("RSSI", -20.0, "above -40 dBm maximum"),
        ("Throughput", 15000.0, "exceeds 10000 Mbps limit"),
    ];
    
    for (metric, value, issue) in invalid_examples {
        println!("      ❌ {}: {} ({})", metric, value, issue);
    }
    
    println!("   📋 Validation features:");
    println!("      🎯 Range validation for all numeric fields");
    println!("      🔢 Data type validation (f64, u32, string)");
    println!("      📊 NaN and infinite value detection");
    println!("      📝 Required field presence validation");
    println!("      🔍 Custom validation rules support");
    
    Ok(())
}

/// Performance optimization example
fn performance_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating performance optimization features...");
    
    println!("   ⚡ Performance features:");
    println!("      🔄 Parallel processing using rayon");
    println!("      📦 Configurable batch sizes for optimal memory usage");
    println!("      🚀 Streaming parsing for large files");
    println!("      💾 Memory-efficient data structures");
    println!("      📊 Real-time progress tracking");
    
    println!("   📈 Performance metrics tracked:");
    println!("      ⏱️ Total parsing time");
    println!("      🚀 Rows processed per second");
    println!("      💾 Memory usage estimation");
    println!("      📊 Data quality score calculation");
    println!("      🔍 Error rate monitoring");
    
    // Simulate performance configurations
    let performance_configs = vec![
        ("Small files (<1MB)", 500, false),
        ("Medium files (1-100MB)", 1000, true),
        ("Large files (>100MB)", 2000, true),
        ("Memory constrained", 200, false),
    ];
    
    println!("   🎯 Recommended configurations:");
    for (scenario, batch_size, parallel) in performance_configs {
        println!("      📋 {}: batch_size={}, parallel={}", scenario, batch_size, parallel);
    }
    
    Ok(())
}

/// Export and reporting example
fn export_and_reporting_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Demonstrating export and reporting features...");
    
    println!("   📁 Export formats available:");
    println!("      📄 JSON: Complete dataset with full metadata");
    println!("      📊 CSV Summary: Key metrics in spreadsheet format");
    println!("      🧠 Feature Vectors: ML-ready data for neural networks");
    
    println!("   📋 Reports generated:");
    println!("      📈 Data Quality Report");
    println!("         - Completeness ratio");
    println!("         - Validation error count");
    println!("         - Processing performance metrics");
    println!("         - Cell statistics summary");
    
    println!("      🚨 Anomaly Summary Report");
    println!("         - Total anomalies detected");
    println!("         - Critical fault identification");
    println!("         - Affected cell lists");
    println!("         - Top issue categories");
    
    println!("      ⚡ Performance Report");
    println!("         - Processing speed metrics");
    println!("         - Memory usage analysis");
    println!("         - Error rate statistics");
    println!("         - Optimization recommendations");
    
    // Example export operations (would work with real data)
    println!("   💡 Example export operations:");
    println!("      parser.export_data(&dataset, ExportFormat::Json, \"full_data.json\")?;");
    println!("      parser.export_data(&dataset, ExportFormat::CsvSummary, \"summary.csv\")?;");
    println!("      parser.export_data(&dataset, ExportFormat::FeatureVectors, \"features.json\")?;");
    println!("      let quality_report = parser.generate_quality_report(&dataset);");
    
    Ok(())
}

/// Utility function to demonstrate configuration examples
pub fn show_configuration_examples() {
    println!("🔧 CSV Parser Configuration Examples");
    println!("====================================");
    
    println!("\n📊 Basic Configuration:");
    println!("```rust");
    println!("let parser = CsvDataParser::new(); // Uses defaults");
    println!("```");
    
    println!("\n⚙️ Custom Configuration:");
    println!("```rust");
    println!("let config = CsvParsingConfig {{");
    println!("    delimiter: b';',");
    println!("    batch_size: 2000,");
    println!("    parallel_processing: true,");
    println!("    validate_data_ranges: true,");
    println!("    max_errors_before_abort: 100,");
    println!("    ..Default::default()");
    println!("}};");
    println!("let parser = CsvDataParser::with_config(config);");
    println!("```");
    
    println!("\n🔍 Custom Validation Rules:");
    println!("```rust");
    println!("let rules = ValidationRules {{");
    println!("    availability_range: (0.0, 100.0),");
    println!("    throughput_range: (0.0, 5000.0),");
    println!("    sinr_range: (-30.0, 40.0),");
    println!("    ..Default::default()");
    println!("}};");
    println!("parser.set_validation_rules(rules);");
    println!("```");
    
    println!("\n📈 Usage Example:");
    println!("```rust");
    println!("let mut parser = CsvDataParser::new();");
    println!("let dataset = parser.parse_csv_file(\"fanndata.csv\")?;");
    println!("let quality_report = parser.generate_quality_report(&dataset);");
    println!("parser.export_data(&dataset, ExportFormat::Json, \"output.json\")?;");
    println!("```");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_csv_example() {
        // Test that the example functions can be called without panicking
        assert!(show_configuration_examples() == ());
    }
    
    #[test]
    fn test_configuration_examples() {
        // Test that we can create various configurations
        let _basic_parser = CsvDataParser::new();
        
        let custom_config = CsvParsingConfig {
            batch_size: 500,
            parallel_processing: false,
            ..Default::default()
        };
        let _custom_parser = CsvDataParser::with_config(custom_config);
        
        let _custom_rules = ValidationRules {
            availability_range: (0.0, 100.0),
            throughput_range: (0.0, 1000.0),
            ..Default::default()
        };
    }
}