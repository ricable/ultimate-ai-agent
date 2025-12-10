//! Binary to create train/test datasets from fanndata.csv

use ran_training::data_splitter::{create_train_test_split, DataSplitConfig};
use ran_training::data::TargetType;
use std::env;
use std::path::PathBuf;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();
    
    // Default paths
    let csv_path = if args.len() > 1 {
        PathBuf::from(&args[1])
    } else {
        PathBuf::from("data/pm/fanndata.csv")
    };

    let train_path = if args.len() > 2 {
        PathBuf::from(&args[2])
    } else {
        PathBuf::from("data/pm/train.json")
    };

    let test_path = if args.len() > 3 {
        PathBuf::from(&args[3])
    } else {
        PathBuf::from("data/pm/test.json")
    };

    println!("=== FANN Data Preprocessing ===");
    println!("CSV input: {:?}", csv_path);
    println!("Train output: {:?}", train_path);
    println!("Test output: {:?}", test_path);

    // Configuration for data splitting
    let config = DataSplitConfig {
        train_ratio: 0.8,
        target_type: TargetType::CellAvailability,
        max_records: None, // Process all records
        seed: Some(42), // For reproducible results
        shuffle: true,
        stratify: false, // Simple random split for now
    };

    println!("\n=== Configuration ===");
    println!("Train ratio: {:.1}%", config.train_ratio * 100.0);
    println!("Target type: {:?}", config.target_type);
    println!("Shuffle: {}", config.shuffle);
    println!("Seed: {:?}", config.seed);

    println!("\n=== Processing CSV ===");
    match create_train_test_split(&csv_path, &train_path, &test_path, Some(config)) {
        Ok(split_info) => {
            println!("✅ Successfully created train/test datasets!");
            split_info.display();
            
            println!("\n=== Output Files ===");
            println!("📁 Training data: {:?}", train_path);
            println!("📁 Test data: {:?}", test_path);
            
            // Display sample counts and ratios
            let train_ratio = split_info.train_samples as f32 / split_info.total_samples as f32;
            let test_ratio = split_info.test_samples as f32 / split_info.total_samples as f32;
            
            println!("\n=== Split Summary ===");
            println!("📊 Training: {} samples ({:.1}%)", split_info.train_samples, train_ratio * 100.0);
            println!("📊 Testing: {} samples ({:.1}%)", split_info.test_samples, test_ratio * 100.0);
            println!("📊 Features: {}", split_info.feature_count);
            
            // Verify files were created
            if train_path.exists() && test_path.exists() {
                println!("\n✅ Files successfully created and verified!");
            } else {
                println!("\n❌ Warning: Output files may not exist");
            }
        }
        Err(e) => {
            eprintln!("❌ Error processing data: {}", e);
            std::process::exit(1);
        }
    }
}