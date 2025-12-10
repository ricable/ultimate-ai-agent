use asa_int::{Config, InterferenceService};
use tonic::transport::Server;
use tracing::{info, error};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting ASA-INT-01 Uplink Interference Classifier Service");
    
    // Load configuration
    let config_path = env::var("ASA_INT_CONFIG")
        .unwrap_or_else(|_| "config.toml".to_string());
    
    let config = if std::path::Path::new(&config_path).exists() {
        Config::from_file(&config_path)?
    } else {
        info!("Config file not found, using default configuration");
        Config::default()
    };
    
    // Validate configuration
    config.validate()?;
    info!("Configuration validated successfully");
    info!("Target accuracy: {:.1}%", config.performance.min_accuracy * 100.0);
    
    // Create service
    let service = InterferenceService::new(config.clone())?;
    
    // Initialize classifier
    service.initialize_classifier().await?;
    info!("Interference classifier initialized with architecture: {}", config.model.architecture);
    
    // Create gRPC server
    let addr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    
    info!("ASA-INT-01 server listening on {}", addr);
    info!("Service capabilities:");
    info!("  - Uplink interference classification with >95% accuracy");
    info!("  - Real-time noise floor analysis");
    info!("  - Automated mitigation recommendations");
    info!("  - Performance monitoring and alerts");
    
    // Build and start server
    Server::builder()
        .add_service(
            asa_int::proto::interference_classifier::interference_classifier_server::InterferenceClassifierServer::new(service)
        )
        .serve(addr)
        .await?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use asa_int::{NoiseFloorMeasurement, CellParameters};
    use std::collections::HashMap;
    
    #[tokio::test]
    async fn test_service_initialization() {
        let config = Config::default();
        let service = InterferenceService::new(config).unwrap();
        
        // Test that service can be initialized
        assert!(service.initialize_classifier().await.is_ok());
    }
    
    #[test]
    fn test_configuration_validation() {
        let mut config = Config::default();
        
        // Valid configuration should pass
        assert!(config.validate().is_ok());
        
        // Invalid accuracy requirement should fail
        config.model.target_accuracy = 0.8;
        assert!(config.validate().is_err());
        
        // Reset and test minimum accuracy
        config = Config::default();
        config.performance.min_accuracy = 0.9;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_mitigation_recommendations() {
        let config = Config::default();
        let service = InterferenceService::new(config).unwrap();
        
        let recommendations = service.build_mitigation_recommendations("EXTERNAL_JAMMER", 0.9);
        assert!(!recommendations.is_empty());
        assert!(recommendations.iter().any(|r| r.contains("frequency hopping")));
        
        let priority = service.calculate_priority_level("EXTERNAL_JAMMER", 0.9);
        assert_eq!(priority, 5); // Should be highest priority
        
        let impact = service.estimate_impact("EXTERNAL_JAMMER", 0.9);
        assert!(impact.contains("High"));
    }
    
    fn create_test_measurements() -> Vec<NoiseFloorMeasurement> {
        vec![
            NoiseFloorMeasurement {
                timestamp: "2024-01-01T00:00:00Z".to_string(),
                noise_floor_pusch: -110.0,
                noise_floor_pucch: -108.0,
                cell_ret: 5.2,
                rsrp: -85.0,
                sinr: 15.0,
                active_users: 50,
                prb_utilization: 0.6,
            },
            NoiseFloorMeasurement {
                timestamp: "2024-01-01T00:01:00Z".to_string(),
                noise_floor_pusch: -109.5,
                noise_floor_pucch: -107.8,
                cell_ret: 5.1,
                rsrp: -84.5,
                sinr: 15.2,
                active_users: 52,
                prb_utilization: 0.62,
            },
        ]
    }
    
    fn create_test_cell_params() -> CellParameters {
        CellParameters {
            cell_id: "TEST_CELL_001".to_string(),
            frequency_band: "B3".to_string(),
            tx_power: 46.0,
            antenna_count: 4,
            bandwidth_mhz: 20.0,
            technology: "LTE".to_string(),
        }
    }
}