use candle_core::{Device, Result, Tensor};
use ran_opt::afm_detect::{AFMDetector, DetectionMode, AnomalyResult};

fn main() -> Result<()> {
    println!("AFM Anomaly Detection Example");
    println!("=============================");
    
    // Initialize device
    let device = Device::Cpu;
    
    // Create AFM detector
    let input_dim = 64;
    let latent_dim = 16;
    let mut detector = AFMDetector::new(input_dim, latent_dim, device.clone())?;
    
    // Generate synthetic normal data for training
    let normal_data = generate_normal_data(100, input_dim, &device)?;
    println!("Generated {} normal samples", normal_data.dims()[0]);
    
    // Train the detector
    println!("\nTraining AFM detector...");
    detector.train_on_normal(&normal_data, 50, 0.001)?;
    
    // Generate test data with anomalies
    let test_data = generate_test_data(20, input_dim, &device)?;
    let history_data = generate_history_data(50, input_dim, &device)?;
    
    println!("\nDetecting anomalies...");
    println!("=====================");
    
    // Test different detection modes
    let modes = vec![
        DetectionMode::KpiKqi,
        DetectionMode::HardwareDegradation,
        DetectionMode::ThermalPower,
        DetectionMode::Combined,
    ];
    
    for mode in modes {
        println!("\nMode: {:?}", mode);
        println!("------------");
        
        let result = detector.detect(&test_data, mode, Some(&history_data))?;
        
        println!("Anomaly Score: {:.4}", result.score);
        println!("Method Scores:");
        for (method, score) in &result.method_scores {
            println!("  {}: {:.4}", method, score);
        }
        
        if let Some(failure_prob) = result.failure_probability {
            println!("Failure Probability (48h): {:.4}", failure_prob);
        }
        
        if let Some(anomaly_type) = &result.anomaly_type {
            println!("Anomaly Type: {:?}", anomaly_type);
        }
        
        println!("Confidence: [{:.4}, {:.4}]", result.confidence.0, result.confidence.1);
        
        // Interpretation
        if result.score > 0.8 {
            println!("⚠️  HIGH RISK: Immediate attention required!");
        } else if result.score > 0.6 {
            println!("⚠️  MEDIUM RISK: Monitor closely");
        } else if result.score > 0.4 {
            println!("ℹ️  LOW RISK: Normal operation with minor deviations");
        } else {
            println!("✅ NORMAL: System operating within expected parameters");
        }
    }
    
    // Demonstrate failure prediction
    println!("\n\nFailure Prediction Examples");
    println!("==========================");
    
    let kpi_data = generate_kpi_time_series(10, input_dim, &device)?;
    let kpi_history = generate_kpi_history(100, input_dim, &device)?;
    
    let result = detector.detect(&kpi_data, DetectionMode::KpiKqi, Some(&kpi_history))?;
    
    if let Some(failure_prob) = result.failure_probability {
        println!("KPI/KQI Failure Probability: {:.4}", failure_prob);
        
        if failure_prob > 0.7 {
            println!("🚨 Critical: Failure likely within 24-48 hours");
            println!("   Recommended actions:");
            println!("   - Increase monitoring frequency");
            println!("   - Prepare maintenance resources");
            println!("   - Consider preventive measures");
        } else if failure_prob > 0.4 {
            println!("⚠️  Warning: Elevated failure risk");
            println!("   Recommended actions:");
            println!("   - Schedule preventive maintenance");
            println!("   - Review system parameters");
        } else {
            println!("✅ Normal: Low failure risk");
        }
    }
    
    // Hardware degradation example
    println!("\nHardware Degradation Analysis");
    println!("-----------------------------");
    
    let hw_data = generate_degradation_pattern(10, input_dim, &device)?;
    let hw_history = generate_degradation_history(200, input_dim, &device)?;
    
    let result = detector.detect(&hw_data, DetectionMode::HardwareDegradation, Some(&hw_history))?;
    
    println!("Hardware Degradation Score: {:.4}", result.score);
    if let Some(failure_prob) = result.failure_probability {
        println!("Hardware Failure Probability: {:.4}", failure_prob);
    }
    
    // Thermal/Power anomaly example
    println!("\nThermal/Power Anomaly Detection");
    println!("------------------------------");
    
    let thermal_data = generate_thermal_anomaly(5, input_dim, &device)?;
    let thermal_history = generate_thermal_history(50, input_dim, &device)?;
    
    let result = detector.detect(&thermal_data, DetectionMode::ThermalPower, Some(&thermal_history))?;
    
    println!("Thermal/Power Anomaly Score: {:.4}", result.score);
    if let Some(failure_prob) = result.failure_probability {
        println!("Thermal Failure Probability: {:.4}", failure_prob);
    }
    
    if result.score > 0.5 {
        println!("🌡️  Temperature/Power anomaly detected!");
        println!("   Potential causes:");
        println!("   - Cooling system failure");
        println!("   - Power supply issues");
        println!("   - Component overheating");
    }
    
    println!("\nExample completed successfully!");
    Ok(())
}

/// Generate synthetic normal data
fn generate_normal_data(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    Tensor::randn(0.0, 1.0, &[n_samples, n_features], device)
}

/// Generate test data with some anomalies
fn generate_test_data(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Tensor::randn(0.0, 1.0, &[n_samples, n_features], device)?;
    
    // Add some anomalies
    for i in 0..n_samples {
        if i % 4 == 0 {
            // Spike anomaly
            let spike = Tensor::randn(0.0, 3.0, &[1, n_features], device)?;
            data = data.slice_assign(&[i..i+1, 0..n_features], &spike)?;
        }
    }
    
    Ok(data)
}

/// Generate history data for temporal analysis
fn generate_history_data(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Tensor::randn(0.0, 0.5, &[n_samples, n_features], device)?;
    
    // Add some trend
    for i in 0..n_samples {
        let trend = Tensor::full(i as f32 * 0.01, &[1, n_features], device)?;
        let sample = data.i(i)? + trend;
        data = data.slice_assign(&[i..i+1, 0..n_features], &sample)?;
    }
    
    Ok(data)
}

/// Generate KPI time series data
fn generate_kpi_time_series(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::new();
    
    for i in 0..n_samples {
        let base = Tensor::randn(0.5, 0.2, &[1, n_features], device)?;
        
        // Add periodic pattern
        let phase = (i as f32 * 0.1).sin();
        let periodic = Tensor::full(phase * 0.1, &[1, n_features], device)?;
        
        let sample = (base + periodic)?;
        data.push(sample);
    }
    
    Tensor::cat(&data, 0)
}

/// Generate KPI history with gradual degradation
fn generate_kpi_history(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::new();
    
    for i in 0..n_samples {
        let base = Tensor::randn(0.8, 0.1, &[1, n_features], device)?;
        
        // Add gradual degradation
        let degradation = Tensor::full(-(i as f32 * 0.002), &[1, n_features], device)?;
        
        let sample = (base + degradation)?;
        data.push(sample);
    }
    
    Tensor::cat(&data, 0)
}

/// Generate hardware degradation pattern
fn generate_degradation_pattern(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::new();
    
    for i in 0..n_samples {
        let base = Tensor::randn(0.6, 0.15, &[1, n_features], device)?;
        
        // Exponential decay pattern
        let decay = (-i as f32 * 0.1).exp();
        let decay_tensor = Tensor::full(decay * 0.3, &[1, n_features], device)?;
        
        let sample = (base + decay_tensor)?;
        data.push(sample);
    }
    
    Tensor::cat(&data, 0)
}

/// Generate hardware degradation history
fn generate_degradation_history(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::new();
    
    for i in 0..n_samples {
        let base = Tensor::randn(1.0, 0.1, &[1, n_features], device)?;
        
        // Monotonic degradation
        let degradation = -(i as f32 * 0.001);
        let degradation_tensor = Tensor::full(degradation, &[1, n_features], device)?;
        
        let sample = (base + degradation_tensor)?;
        data.push(sample);
    }
    
    Tensor::cat(&data, 0)
}

/// Generate thermal anomaly data
fn generate_thermal_anomaly(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::new();
    
    for i in 0..n_samples {
        let base = Tensor::randn(0.3, 0.1, &[1, n_features], device)?;
        
        // Thermal runaway pattern
        let thermal = (i as f32 * 0.2).exp() * 0.1;
        let thermal_tensor = Tensor::full(thermal, &[1, n_features], device)?;
        
        let sample = (base + thermal_tensor)?;
        data.push(sample);
    }
    
    Tensor::cat(&data, 0)
}

/// Generate thermal history
fn generate_thermal_history(n_samples: usize, n_features: usize, device: &Device) -> Result<Tensor> {
    let mut data = Vec::new();
    
    for i in 0..n_samples {
        let base = Tensor::randn(0.2, 0.05, &[1, n_features], device)?;
        
        // Gradual temperature increase
        let temp_increase = i as f32 * 0.001;
        let temp_tensor = Tensor::full(temp_increase, &[1, n_features], device)?;
        
        let sample = (base + temp_tensor)?;
        data.push(sample);
    }
    
    Tensor::cat(&data, 0)
}