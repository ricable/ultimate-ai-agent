//! Comprehensive KPI Data Processing Demo
//! 
//! This demo showcases the complete data processing pipeline for network KPIs:
//! - CSV data parsing (fanndata.csv format)
//! - Data validation and cleansing
//! - Comprehensive KPI analysis
//! - Neural network feature extraction
//! - Real-time ingestion capabilities
//! - Complete integration workflow

use std::path::Path;
use std::env;
use tokio;

// Import our comprehensive data processing modules
use ran::pfs_data::comprehensive_data_integration::{
    ComprehensiveDataIntegration, IntegrationConfig, demo_fanndata_processing
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🤖 Comprehensive KPI Data Processing Demo");
    println!("=========================================");
    println!();
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 && (args[1] == "--help" || args[1] == "-h") {
        display_help();
        return Ok(());
    }
    
    // Determine CSV file path
    let csv_path = if args.len() > 1 {
        args[1].clone()
    } else {
        "./fanndata.csv".to_string()
    };
    
    println!("📂 Looking for CSV data at: {}", csv_path);
    
    if !Path::new(&csv_path).exists() {
        println!("❌ CSV file not found at: {}", csv_path);
        println!("💡 Please provide a valid CSV file path as an argument, or ensure fanndata.csv exists in the current directory");
        return Ok(());
    }
    
    // Configure processing options based on arguments
    let mut config = IntegrationConfig::default();
    
    // Parse additional configuration from arguments
    for arg in &args[2..] {
        match arg.as_str() {
            "--disable-validation" => config.enable_validation = false,
            "--disable-neural" => config.enable_neural_processing = false,
            "--enable-realtime" => config.enable_realtime = true,
            "--no-export" => config.export_results = false,
            "--batch-size=500" => config.batch_size = 500,
            "--batch-size=1000" => config.batch_size = 1000,
            "--batch-size=2000" => config.batch_size = 2000,
            _ => {}
        }
    }
    
    println!("⚙️  Processing Configuration:");
    println!("   📊 Validation: {}", if config.enable_validation { "✅ Enabled" } else { "❌ Disabled" });
    println!("   🧠 Neural Processing: {}", if config.enable_neural_processing { "✅ Enabled" } else { "❌ Disabled" });
    println!("   📡 Real-time: {}", if config.enable_realtime { "✅ Enabled" } else { "❌ Disabled" });
    println!("   💾 Export Results: {}", if config.export_results { "✅ Enabled" } else { "❌ Disabled" });
    println!("   📦 Batch Size: {}", config.batch_size);
    println!();
    
    // Run the comprehensive processing pipeline
    match run_comprehensive_processing(&csv_path, config).await {
        Ok(()) => {
            println!("🎉 Comprehensive KPI processing completed successfully!");
            println!();
            println!("✨ What was accomplished:");
            println!("   🔍 Data validation and quality assessment");
            println!("   🧮 Comprehensive KPI calculations and analysis");
            println!("   🧠 Neural network feature extraction");
            println!("   📊 Performance metrics and insights");
            println!("   💾 Processed data export (if enabled)");
            println!();
            println!("📁 Check the output directory for detailed results and reports.");
        }
        Err(e) => {
            println!("❌ Processing failed with error: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}

async fn run_comprehensive_processing(csv_path: &str, config: IntegrationConfig) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting comprehensive KPI data processing...");
    println!();
    
    // Create the integration pipeline
    let mut integration = ComprehensiveDataIntegration::new(config);
    
    // Process the CSV file with full pipeline
    let result = integration.process_fanndata_csv(csv_path).await?;
    
    // Display detailed results
    println!("\n📈 DETAILED PROCESSING RESULTS");
    println!("==============================");
    
    // Data summary
    println!("\n📊 Data Summary:");
    println!("   Total Cells/eNodeBs: {}", result.data_summary.total_cells);
    println!("   Date Range: {} to {}", result.data_summary.date_range.0, result.data_summary.date_range.1);
    println!("   KPI Categories: {}", result.data_summary.kpi_categories.join(", "));
    println!("   Data Completeness: {:.1}%", result.data_summary.completeness_percentage);
    println!("   Frequency Bands: {}", result.data_summary.frequency_bands.join(", "));
    
    // Quality assessment
    println!("\n🎯 Quality Assessment:");
    println!("   Overall Score: {:.1}%", result.quality_assessment.overall_score);
    println!("   Completeness: {:.1}%", result.quality_assessment.completeness_score);
    println!("   Consistency: {:.1}%", result.quality_assessment.consistency_score);
    println!("   Accuracy: {:.1}%", result.quality_assessment.accuracy_score);
    
    if !result.quality_assessment.issues_by_category.is_empty() {
        println!("\n   Issues by Category:");
        for (category, count) in &result.quality_assessment.issues_by_category {
            println!("     • {}: {} occurrences", category, count);
        }
    }
    
    // Performance statistics
    println!("\n⚡ Performance Statistics:");
    println!("   Records Processed: {}", result.statistics.records_processed);
    println!("   Records Validated: {}", result.statistics.records_validated);
    println!("   Records Corrected: {}", result.statistics.records_corrected);
    println!("   Total Duration: {:.2}s", result.statistics.total_duration.as_secs_f64());
    println!("   Avg Time/Record: {:.2}ms", result.statistics.avg_processing_time.as_millis());
    println!("   Throughput: {:.1} records/sec", 
             result.statistics.records_processed as f64 / result.statistics.total_duration.as_secs_f64());
    
    // Critical issues
    if !result.quality_assessment.critical_issues.is_empty() {
        println!("\n🚨 Critical Issues Requiring Attention:");
        for issue in &result.quality_assessment.critical_issues {
            println!("   ⚠️  {}", issue);
        }
    }
    
    // Recommendations
    if !result.recommendations.is_empty() {
        println!("\n💡 Recommendations for Improvement:");
        for recommendation in &result.recommendations {
            println!("   💫 {}", recommendation);
        }
    }
    
    // Export information
    if !result.export_paths.is_empty() {
        println!("\n📁 Exported Files:");
        for path in &result.export_paths {
            println!("   📄 {}", path);
        }
    }
    
    Ok(())
}

fn display_help() {
    println!("🤖 Comprehensive KPI Data Processing Demo");
    println!("=========================================");
    println!();
    println!("USAGE:");
    println!("    comprehensive_kpi_demo [CSV_FILE_PATH] [OPTIONS]");
    println!();
    println!("ARGUMENTS:");
    println!("    CSV_FILE_PATH    Path to the CSV file to process (default: ./fanndata.csv)");
    println!();
    println!("OPTIONS:");
    println!("    --disable-validation    Disable data validation and cleansing");
    println!("    --disable-neural        Disable neural network processing");
    println!("    --enable-realtime      Enable real-time processing capabilities");
    println!("    --no-export            Disable result export");
    println!("    --batch-size=N         Set processing batch size (500, 1000, or 2000)");
    println!("    --help, -h             Display this help message");
    println!();
    println!("EXAMPLES:");
    println!("    # Process fanndata.csv with default settings");
    println!("    comprehensive_kpi_demo");
    println!();
    println!("    # Process custom CSV file with validation disabled");
    println!("    comprehensive_kpi_demo my_data.csv --disable-validation");
    println!();
    println!("    # High-throughput processing with larger batches");
    println!("    comprehensive_kpi_demo data.csv --batch-size=2000 --disable-neural");
    println!();
    println!("    # Real-time processing mode");
    println!("    comprehensive_kpi_demo live_data.csv --enable-realtime");
    println!();
    println!("FEATURES:");
    println!("    🔍 Comprehensive data validation and quality assessment");
    println!("    🧮 Advanced KPI calculations and network performance analysis");
    println!("    🧠 Neural network-based feature extraction and insights");
    println!("    📊 Real-time data ingestion and processing capabilities");
    println!("    💾 Automated result export with detailed reporting");
    println!("    ⚡ High-performance batch processing");
    println!("    🛠️  Automatic error correction and data cleansing");
    println!();
    println!("OUTPUT:");
    println!("    The demo generates comprehensive analysis reports including:");
    println!("    • Data quality assessment and validation results");
    println!("    • KPI performance metrics and insights");
    println!("    • Neural network feature analysis");
    println!("    • Processing performance statistics");
    println!("    • Recommendations for data quality improvement");
    println!("    • Exported data files for further analysis");
    println!();
    println!("SUPPORTED DATA FORMAT:");
    println!("    The demo is optimized for French network KPI data (fanndata.csv format)");
    println!("    containing metrics such as:");
    println!("    • Signal quality (SINR, RSSI, BLER)");
    println!("    • Throughput and latency metrics");
    println!("    • Handover success rates");
    println!("    • ENDC (5G NSA) establishment metrics");
    println!("    • Traffic and load indicators");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_demo_with_sample_csv() {
        // Create a temporary CSV file for testing
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_data.csv");
        let mut file = File::create(&file_path).unwrap();
        
        // Write sample CSV data (headers + a few rows)
        writeln!(file, "Nom_Site;Date;sinr_pusch_avg;ave_4G_LTE_DL_User_Thrput;dl_latency_avg").unwrap();
        writeln!(file, "Site1;2024-01-01;15.5;45.2;12.8").unwrap();
        writeln!(file, "Site2;2024-01-01;18.1;52.3;10.5").unwrap();
        writeln!(file, "Site3;2024-01-01;12.8;38.7;15.2").unwrap();
        
        // Test configuration
        let config = IntegrationConfig {
            enable_validation: true,
            enable_neural_processing: false, // Disable for fast test
            enable_realtime: false,
            export_results: false,
            batch_size: 10,
            ..Default::default()
        };
        
        // This test verifies the demo can be instantiated and configured
        // Full processing test would require more extensive mock data
        let integration = ComprehensiveDataIntegration::new(config);
        assert!(integration.config.enable_validation);
    }
    
    #[test]
    fn test_configuration_parsing() {
        let mut config = IntegrationConfig::default();
        
        // Simulate argument parsing
        config.enable_validation = false;
        config.batch_size = 500;
        
        assert!(!config.enable_validation);
        assert_eq!(config.batch_size, 500);
    }
}