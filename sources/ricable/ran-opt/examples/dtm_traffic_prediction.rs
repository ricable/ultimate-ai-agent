use ran_opt::dtm_traffic::{
    TrafficPredictor, TrafficPattern, PredictorConfig, QoSIndicator, ServiceType, NetworkLayer,
};
use std::collections::HashMap;
use std::time::SystemTime;
use chrono::{DateTime, Utc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("DTM Traffic Prediction System - Example");
    println!("=======================================");
    
    // Create predictor configuration
    let config = PredictorConfig {
        sequence_length: 48,  // 12 hours with 15-min intervals
        forecast_horizons: vec![4, 8, 16, 24], // 1h, 2h, 4h, 6h
        lstm_hidden_size: 128,
        gru_hidden_size: 128,
        tcn_filters: 32,
        tcn_kernel_size: 3,
        tcn_dilations: vec![1, 2, 4, 8],
        dropout_rate: 0.1,
        learning_rate: 0.001,
        batch_size: 16,
    };
    
    // Create traffic predictor
    let mut predictor = TrafficPredictor::new(config);
    
    // Generate synthetic training data
    println!("Generating synthetic traffic data...");
    let training_data = generate_synthetic_traffic_data(1000)?;
    
    // Train the predictor
    println!("Training traffic prediction models...");
    predictor.train(&training_data, 50)?;
    
    // Generate recent patterns for prediction
    let recent_patterns = generate_recent_patterns(48)?;
    
    // Make predictions
    println!("Making traffic predictions...");
    let predictions = predictor.predict(&recent_patterns)?;
    
    // Display predictions
    display_predictions(&predictions);
    
    // Generate AMOS load balancing scripts
    println!("\nGenerating AMOS load balancing scripts...");
    let amos_scripts = predictor.generate_amos_scripts(&predictions);
    
    for (i, script) in amos_scripts.iter().enumerate() {
        println!("Script {}: {} lines", i + 1, script.lines().count());
        if i < 2 {  // Show first 2 scripts
            println!("--- Script {} ---", i + 1);
            println!("{}", script.lines().take(20).collect::<Vec<_>>().join("\n"));
            println!("...");
        }
    }
    
    // Demonstrate service type classification
    println!("\nService Type Classification Examples:");
    demonstrate_service_classification(&predictor);
    
    // Demonstrate QoS-aware predictions
    println!("\nQoS-Aware Predictions:");
    demonstrate_qos_predictions(&predictions);
    
    // Show layer-specific predictions
    println!("\nLayer-Specific Demand Predictions:");
    demonstrate_layer_predictions(&predictions);
    
    Ok(())
}

fn generate_synthetic_traffic_data(count: usize) -> Result<Vec<TrafficPattern>, Box<dyn std::error::Error>> {
    let mut patterns = Vec::new();
    let start_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs() as i64;
    
    for i in 0..count {
        let timestamp = start_time - (count - i) as i64 * 15 * 60; // 15-minute intervals
        
        // Simulate daily patterns with some randomness
        let hour_of_day = ((timestamp / 3600) % 24) as f32;
        let day_factor = (hour_of_day * std::f32::consts::PI / 12.0).sin() * 0.3 + 0.7;
        
        for &layer in &[NetworkLayer::L2100, NetworkLayer::N78, NetworkLayer::N258] {
            let base_utilization = match layer {
                NetworkLayer::L2100 => 0.6,
                NetworkLayer::N78 => 0.4,
                NetworkLayer::N258 => 0.3,
            };
            
            let prb_utilization = (base_utilization * day_factor + 
                                  (rand::random::<f32>() - 0.5) * 0.2).max(0.1).min(0.95);
            
            let service_type = match layer {
                NetworkLayer::L2100 if prb_utilization > 0.7 => ServiceType::VoNR,
                NetworkLayer::N78 if prb_utilization > 0.8 => ServiceType::URLLC,
                NetworkLayer::N258 => ServiceType::EMBB,
                _ => ServiceType::MIoT,
            };
            
            let mut qos_indicators = HashMap::new();
            qos_indicators.insert(QoSIndicator::QI1, rand::random::<f32>() * 0.5 + 0.3);
            qos_indicators.insert(QoSIndicator::QI80, rand::random::<f32>() * 0.4 + 0.2);
            qos_indicators.insert(QoSIndicator::QI4, rand::random::<f32>() * 0.6 + 0.4);
            
            let user_count = (prb_utilization * 200.0) as u32;
            let throughput_mbps = prb_utilization * 1000.0 * match layer {
                NetworkLayer::L2100 => 0.5,
                NetworkLayer::N78 => 1.0,
                NetworkLayer::N258 => 2.0,
            };
            
            patterns.push(TrafficPattern {
                timestamp,
                prb_utilization,
                layer,
                service_type,
                qos_indicators,
                user_count,
                throughput_mbps,
            });
        }
    }
    
    Ok(patterns)
}

fn generate_recent_patterns(count: usize) -> Result<Vec<TrafficPattern>, Box<dyn std::error::Error>> {
    let mut patterns = Vec::new();
    let start_time = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)?
        .as_secs() as i64;
    
    for i in 0..count {
        let timestamp = start_time - (count - i) as i64 * 15 * 60;
        
        // Simulate current network state with increasing load
        let load_factor = 1.0 + (i as f32 / count as f32) * 0.4; // Increasing load
        
        for &layer in &[NetworkLayer::L2100, NetworkLayer::N78, NetworkLayer::N258] {
            let base_utilization = match layer {
                NetworkLayer::L2100 => 0.5,
                NetworkLayer::N78 => 0.6,
                NetworkLayer::N258 => 0.4,
            };
            
            let prb_utilization = (base_utilization * load_factor).min(0.95);
            
            let service_type = match layer {
                NetworkLayer::L2100 => ServiceType::VoNR,
                NetworkLayer::N78 => ServiceType::EMBB,
                NetworkLayer::N258 => ServiceType::URLLC,
            };
            
            let mut qos_indicators = HashMap::new();
            qos_indicators.insert(QoSIndicator::QI1, 0.8);
            qos_indicators.insert(QoSIndicator::QI80, 0.7);
            qos_indicators.insert(QoSIndicator::QI4, 0.6);
            
            let user_count = (prb_utilization * 150.0) as u32;
            let throughput_mbps = prb_utilization * 800.0;
            
            patterns.push(TrafficPattern {
                timestamp,
                prb_utilization,
                layer,
                service_type,
                qos_indicators,
                user_count,
                throughput_mbps,
            });
        }
    }
    
    Ok(patterns)
}

fn display_predictions(predictions: &ran_opt::dtm_traffic::ForecastResult) {
    println!("Traffic Predictions:");
    println!("==================");
    
    for (i, &horizon) in predictions.horizons.iter().enumerate() {
        let datetime = DateTime::from_timestamp(horizon, 0).unwrap_or_else(|| Utc::now());
        println!("Horizon {}: {} ({})", i + 1, 
                datetime.format("%Y-%m-%d %H:%M:%S UTC"),
                datetime.format("%H:%M"));
        
        println!("  PRB Utilization Predictions:");
        println!("    L2100: {:.1}% ± {:.1}%", 
                predictions.prb_predictions[[i, 0]] * 100.0,
                (predictions.confidence_intervals[[i, 0, 1]] - 
                 predictions.confidence_intervals[[i, 0, 0]]) * 50.0);
        println!("    N78:   {:.1}% ± {:.1}%", 
                predictions.prb_predictions[[i, 1]] * 100.0,
                (predictions.confidence_intervals[[i, 1, 1]] - 
                 predictions.confidence_intervals[[i, 1, 0]]) * 50.0);
        println!("    N258:  {:.1}% ± {:.1}%", 
                predictions.prb_predictions[[i, 2]] * 100.0,
                (predictions.confidence_intervals[[i, 2, 1]] - 
                 predictions.confidence_intervals[[i, 2, 0]]) * 50.0);
        
        println!("  Service Type Demands:");
        for layer_idx in 0..3 {
            let layer_name = match layer_idx {
                0 => "L2100",
                1 => "N78",
                2 => "N258",
                _ => "Unknown",
            };
            
            println!("    {}: eMBB={:.1}%, VoNR={:.1}%, URLLC={:.1}%, MIoT={:.1}%",
                    layer_name,
                    predictions.service_predictions[[i, layer_idx, 0]] * 100.0,
                    predictions.service_predictions[[i, layer_idx, 1]] * 100.0,
                    predictions.service_predictions[[i, layer_idx, 2]] * 100.0,
                    predictions.service_predictions[[i, layer_idx, 3]] * 100.0);
        }
        
        println!();
    }
}

fn demonstrate_service_classification(predictor: &TrafficPredictor) {
    let test_patterns = vec![
        // High-priority voice traffic
        TrafficPattern {
            timestamp: 1234567890,
            prb_utilization: 0.6,
            layer: NetworkLayer::L2100,
            service_type: ServiceType::VoNR,
            qos_indicators: {
                let mut qos = HashMap::new();
                qos.insert(QoSIndicator::QI1, 0.9);
                qos
            },
            user_count: 80,
            throughput_mbps: 50.0,
        },
        // High-throughput data traffic
        TrafficPattern {
            timestamp: 1234567890,
            prb_utilization: 0.8,
            layer: NetworkLayer::N78,
            service_type: ServiceType::EMBB,
            qos_indicators: {
                let mut qos = HashMap::new();
                qos.insert(QoSIndicator::QI80, 0.8);
                qos
            },
            user_count: 120,
            throughput_mbps: 800.0,
        },
        // Ultra-low latency traffic
        TrafficPattern {
            timestamp: 1234567890,
            prb_utilization: 0.4,
            layer: NetworkLayer::N258,
            service_type: ServiceType::URLLC,
            qos_indicators: {
                let mut qos = HashMap::new();
                qos.insert(QoSIndicator::QI80, 0.95);
                qos
            },
            user_count: 30,
            throughput_mbps: 200.0,
        },
    ];
    
    for pattern in test_patterns {
        let classified = predictor.classify_service_type(&pattern);
        println!("Layer: {:?}, Input: {:?}, Classified: {:?}", 
                pattern.layer, pattern.service_type, classified);
    }
}

fn demonstrate_qos_predictions(predictions: &ran_opt::dtm_traffic::ForecastResult) {
    println!("QoS Indicator Predictions (showing potential degradation):");
    
    for (i, &horizon) in predictions.horizons.iter().enumerate() {
        let datetime = DateTime::from_timestamp(horizon, 0).unwrap_or_else(|| Utc::now());
        println!("Horizon {}: {}", i + 1, datetime.format("%H:%M"));
        
        for layer_idx in 0..3 {
            let layer_name = match layer_idx {
                0 => "L2100",
                1 => "N78",
                2 => "N258",
                _ => "Unknown",
            };
            
            let qos_degradation = predictions.qos_predictions[[i, layer_idx, 0]];
            if qos_degradation > 0.1 {
                println!("  {} - QoS Alert: {:.1}% degradation expected", 
                        layer_name, qos_degradation * 100.0);
            }
        }
    }
}

fn demonstrate_layer_predictions(predictions: &ran_opt::dtm_traffic::ForecastResult) {
    println!("Layer-Specific Demand Analysis:");
    
    for (i, &horizon) in predictions.horizons.iter().enumerate() {
        let datetime = DateTime::from_timestamp(horizon, 0).unwrap_or_else(|| Utc::now());
        println!("Horizon {}: {}", i + 1, datetime.format("%H:%M"));
        
        // L2100 analysis
        let l2100_utilization = predictions.prb_predictions[[i, 0]];
        if l2100_utilization > 0.8 {
            println!("  L2100 - High load predicted: {:.1}% PRB utilization", 
                    l2100_utilization * 100.0);
            println!("         Recommended: Offload to 5G layers");
        }
        
        // N78 analysis
        let n78_utilization = predictions.prb_predictions[[i, 1]];
        if n78_utilization > 0.7 {
            println!("  N78 - Moderate load: {:.1}% PRB utilization", 
                    n78_utilization * 100.0);
            println!("        Recommended: Enable carrier aggregation");
        }
        
        // N258 analysis
        let n258_utilization = predictions.prb_predictions[[i, 2]];
        if n258_utilization > 0.6 {
            println!("  N258 - Capacity available: {:.1}% PRB utilization", 
                    n258_utilization * 100.0);
            println!("         Recommended: Increase power for better coverage");
        }
    }
}