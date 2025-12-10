use asa_5g::{Config, EndcService};
use tonic::transport::Server;
use tracing::{info, error};
use std::env;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("Starting ASA-5G-01 ENDC Setup Failure Predictor Service");
    
    // Load configuration
    let config_path = env::var("ASA_5G_CONFIG")
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
    info!("Target prediction accuracy: {:.1}%", config.performance.min_prediction_accuracy * 100.0);
    
    // Create service
    let service = EndcService::new(config.clone()).await?;
    info!("ENDC failure predictor initialized with model: {}", config.prediction.model_type);
    
    // Create gRPC server
    let addr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    
    info!("ASA-5G-01 server listening on {}", addr);
    info!("Service capabilities:");
    info!("  - ENDC setup failure prediction with >80% accuracy");
    info!("  - Real-time 5G NSA/SA health monitoring");
    info!("  - Proactive mitigation recommendations");
    info!("  - Signal quality analysis and optimization");
    
    // Build and start server
    Server::builder()
        .add_service(
            asa_5g::proto::service_assurance_service_server::ServiceAssuranceServiceServer::new(service)
        )
        .serve(addr)
        .await?;
    
    Ok(())
}