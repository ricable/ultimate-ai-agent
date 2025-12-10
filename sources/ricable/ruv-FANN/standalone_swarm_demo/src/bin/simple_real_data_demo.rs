//! Simple Real Data Integration Demo
//! 
//! A simplified demonstration of processing real network KPI data from fanndata.csv
//! with basic anomaly detection and performance analysis.

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Real network KPI data structure from fanndata.csv
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealNetworkData {
    pub timestamp: String,
    pub enodeb_name: String,
    pub cell_name: String,
    pub cell_availability: f64,
    pub dl_user_throughput: f64,
    pub ul_user_throughput: f64,
    pub sinr_pusch: f64,
    pub erab_drop_rate_qci5: f64,
    pub mac_dl_bler: f64,
    pub dl_latency_avg: f64,
    pub endc_setup_sr: f64,
    pub active_users_dl: u64,
}

impl RealNetworkData {
    /// Calculate a simple anomaly score based on KPI thresholds
    pub fn calculate_anomaly_score(&self) -> f64 {
        let mut score = 0.0;
        let mut factors = 0;
        
        // Low availability
        if self.cell_availability < 95.0 {
            score += (95.0 - self.cell_availability) / 95.0;
            factors += 1;
        }
        
        // High drop rate
        if self.erab_drop_rate_qci5 > 2.0 {
            score += self.erab_drop_rate_qci5 / 10.0;
            factors += 1;
        }
        
        // High error rate
        if self.mac_dl_bler > 5.0 {
            score += self.mac_dl_bler / 50.0;
            factors += 1;
        }
        
        // Poor signal quality
        if self.sinr_pusch < 0.0 {
            score += (-self.sinr_pusch) / 20.0;
            factors += 1;
        }
        
        // High latency
        if self.dl_latency_avg > 50.0 {
            score += (self.dl_latency_avg - 50.0) / 100.0;
            factors += 1;
        }
        
        if factors > 0 { score / factors as f64 } else { 0.0 }
    }
    
    /// Get performance category
    pub fn get_performance_category(&self) -> String {
        let anomaly_score = self.calculate_anomaly_score();
        
        match anomaly_score {
            score if score < 0.2 => "Excellent".to_string(),
            score if score < 0.4 => "Good".to_string(),
            score if score < 0.6 => "Fair".to_string(),
            score if score < 0.8 => "Poor".to_string(),
            _ => "Critical".to_string(),
        }
    }
}

/// Simple CSV parser for fanndata.csv
pub struct SimpleCsvParser {
    file_path: String,
}

impl SimpleCsvParser {
    pub fn new(file_path: String) -> Self {
        Self { file_path }
    }
    
    /// Parse the CSV file and return structured data
    pub fn parse_data(&self) -> Result<Vec<RealNetworkData>, Box<dyn std::error::Error>> {
        let mut records = Vec::new();
        let file = File::open(&self.file_path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header
        if let Some(_header) = lines.next() {
            // Header processed
        }
        
        for (line_num, line_result) in lines.enumerate() {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }
            
            match self.parse_line(&line) {
                Ok(record) => records.push(record),
                Err(e) => {
                    if line_num < 5 { // Only show first few errors
                        eprintln!("Parse error on line {}: {}", line_num + 2, e);
                    }
                }
            }
        }
        
        Ok(records)
    }
    
    /// Parse a single CSV line
    fn parse_line(&self, line: &str) -> Result<RealNetworkData, Box<dyn std::error::Error>> {
        let fields: Vec<&str> = line.split(';').collect();
        
        if fields.len() < 50 {
            return Err("Insufficient fields in CSV line".into());
        }
        
        let parse_f64 = |idx: usize| -> f64 {
            fields.get(idx)
                .and_then(|s| s.trim().parse::<f64>().ok())
                .unwrap_or(0.0)
        };
        
        let parse_u64 = |idx: usize| -> u64 {
            fields.get(idx)
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(0)
        };
        
        let parse_string = |idx: usize| -> String {
            fields.get(idx)
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        
        Ok(RealNetworkData {
            timestamp: parse_string(0),
            enodeb_name: parse_string(2),
            cell_name: parse_string(4),
            cell_availability: parse_f64(7),
            dl_user_throughput: parse_f64(31),
            ul_user_throughput: parse_f64(32),
            sinr_pusch: parse_f64(35),
            erab_drop_rate_qci5: parse_f64(14),
            mac_dl_bler: parse_f64(40),
            dl_latency_avg: parse_f64(50),
            endc_setup_sr: parse_f64(90),
            active_users_dl: parse_u64(78),
        })
    }
}

/// Analysis results structure
#[derive(Debug, Serialize)]
pub struct AnalysisResults {
    pub total_records: usize,
    pub anomaly_count: usize,
    pub anomaly_rate: f64,
    pub performance_categories: HashMap<String, usize>,
    pub average_availability: f64,
    pub average_throughput: f64,
    pub average_sinr: f64,
    pub processing_time_ms: f64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Simple Real Data Integration Demo");
    println!("===================================");
    
    // Get CSV file path
    let csv_path = env::args()
        .nth(1)
        .unwrap_or_else(|| "../data/fanndata.csv".to_string());
    
    println!("📂 Processing file: {}", csv_path);
    
    // Parse the data
    let start_time = Instant::now();
    let parser = SimpleCsvParser::new(csv_path);
    let data = parser.parse_data()?;
    let parsing_time = start_time.elapsed();
    
    println!("✅ Parsed {} records in {:.2}ms", 
        data.len(), parsing_time.as_millis());
    
    // Analyze the data
    let analysis_start = Instant::now();
    let mut anomaly_count = 0;
    let mut performance_categories = HashMap::new();
    let mut total_availability = 0.0;
    let mut total_throughput = 0.0;
    let mut total_sinr = 0.0;
    let mut valid_sinr_count = 0;
    
    for record in &data {
        // Count anomalies
        let anomaly_score = record.calculate_anomaly_score();
        if anomaly_score > 0.5 {
            anomaly_count += 1;
        }
        
        // Categorize performance
        let category = record.get_performance_category();
        *performance_categories.entry(category).or_insert(0) += 1;
        
        // Aggregate metrics
        total_availability += record.cell_availability;
        total_throughput += record.dl_user_throughput;
        if record.sinr_pusch > -50.0 && record.sinr_pusch < 50.0 {
            total_sinr += record.sinr_pusch;
            valid_sinr_count += 1;
        }
    }
    
    let analysis_time = analysis_start.elapsed();
    let total_time = start_time.elapsed();
    
    // Create results
    let results = AnalysisResults {
        total_records: data.len(),
        anomaly_count,
        anomaly_rate: (anomaly_count as f64 / data.len() as f64) * 100.0,
        performance_categories,
        average_availability: total_availability / data.len() as f64,
        average_throughput: total_throughput / data.len() as f64,
        average_sinr: if valid_sinr_count > 0 { total_sinr / valid_sinr_count as f64 } else { 0.0 },
        processing_time_ms: total_time.as_millis() as f64,
    };
    
    // Display results
    println!("\n📊 Analysis Results:");
    println!("===================");
    println!("📈 Total Records: {}", results.total_records);
    println!("⚠️  Anomalies Detected: {} ({:.1}%)", 
        results.anomaly_count, results.anomaly_rate);
    println!("📶 Average Availability: {:.1}%", results.average_availability);
    println!("🚀 Average DL Throughput: {:.0} Mbps", results.average_throughput);
    println!("📡 Average SINR: {:.1} dB", results.average_sinr);
    println!("⏱️  Processing Time: {:.0}ms", results.processing_time_ms);
    println!("🏃 Processing Rate: {:.0} records/sec", 
        results.total_records as f64 / (results.processing_time_ms / 1000.0));
    
    println!("\n📋 Performance Categories:");
    for (category, count) in &results.performance_categories {
        println!("   {}: {} ({:.1}%)", 
            category, count, (*count as f64 / results.total_records as f64) * 100.0);
    }
    
    // Show some example anomalies
    println!("\n🔍 Example Anomalies:");
    let mut anomaly_examples = 0;
    for record in &data {
        if record.calculate_anomaly_score() > 0.6 && anomaly_examples < 3 {
            println!("   Cell: {}_{}", record.enodeb_name, record.cell_name);
            println!("      Availability: {:.1}%", record.cell_availability);
            println!("      Drop Rate: {:.2}%", record.erab_drop_rate_qci5);
            println!("      SINR: {:.1} dB", record.sinr_pusch);
            println!("      Category: {}", record.get_performance_category());
            anomaly_examples += 1;
        }
    }
    
    // Export results
    let json_results = serde_json::to_string_pretty(&results)?;
    std::fs::write("simple_analysis_results.json", json_results)?;
    println!("\n💾 Results exported to: simple_analysis_results.json");
    
    // Generate summary report
    let summary = format!(
        "# Simple Real Data Analysis Summary\n\n\
        ## Overview\n\
        - **Records Processed**: {}\n\
        - **Anomaly Rate**: {:.1}%\n\
        - **Average Availability**: {:.1}%\n\
        - **Processing Speed**: {:.0} records/sec\n\n\
        ## Performance Distribution\n{}\n\n\
        Generated on: {}\n",
        results.total_records,
        results.anomaly_rate,
        results.average_availability,
        results.total_records as f64 / (results.processing_time_ms / 1000.0),
        results.performance_categories.iter()
            .map(|(k, v)| format!("- **{}**: {} ({:.1}%)", 
                k, v, (*v as f64 / results.total_records as f64) * 100.0))
            .collect::<Vec<_>>()
            .join("\n"),
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
    );
    
    std::fs::write("simple_analysis_summary.md", summary)?;
    println!("📋 Summary exported to: simple_analysis_summary.md");
    
    println!("\n🎉 Simple real data analysis complete!");
    println!("✅ Successfully processed {} network records", results.total_records);
    println!("✅ Detected {} anomalies ({:.1}%)", results.anomaly_count, results.anomaly_rate);
    println!("✅ Zero mock data used - all analysis from real KPIs");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_score_calculation() {
        let normal_data = RealNetworkData {
            timestamp: "2025-06-27 00".to_string(),
            enodeb_name: "TEST_ENODEB".to_string(),
            cell_name: "TEST_CELL".to_string(),
            cell_availability: 99.0,
            dl_user_throughput: 25000.0,
            ul_user_throughput: 5000.0,
            sinr_pusch: 10.0,
            erab_drop_rate_qci5: 1.0,
            mac_dl_bler: 3.0,
            dl_latency_avg: 20.0,
            endc_setup_sr: 95.0,
            active_users_dl: 45,
        };
        
        let score = normal_data.calculate_anomaly_score();
        assert!(score < 0.3, "Normal data should have low anomaly score");
        
        let anomalous_data = RealNetworkData {
            cell_availability: 70.0, // Low
            erab_drop_rate_qci5: 15.0, // High
            mac_dl_bler: 25.0, // High
            sinr_pusch: -15.0, // Poor
            dl_latency_avg: 150.0, // High
            ..normal_data
        };
        
        let anomaly_score = anomalous_data.calculate_anomaly_score();
        assert!(anomaly_score > 0.5, "Anomalous data should have high score");
    }
    
    #[test]
    fn test_performance_categorization() {
        let excellent_data = RealNetworkData {
            timestamp: "2025-06-27 00".to_string(),
            enodeb_name: "TEST".to_string(),
            cell_name: "TEST".to_string(),
            cell_availability: 99.9,
            dl_user_throughput: 30000.0,
            ul_user_throughput: 8000.0,
            sinr_pusch: 15.0,
            erab_drop_rate_qci5: 0.1,
            mac_dl_bler: 1.0,
            dl_latency_avg: 10.0,
            endc_setup_sr: 99.0,
            active_users_dl: 50,
        };
        
        assert_eq!(excellent_data.get_performance_category(), "Excellent");
    }
}