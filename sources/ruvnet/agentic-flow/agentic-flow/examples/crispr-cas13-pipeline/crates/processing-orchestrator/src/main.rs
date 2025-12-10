//! # Processing Orchestrator
//!
//! Job orchestration service for the CRISPR-Cas13 pipeline.

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod jobs;
mod scheduler;
mod workers;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "processing_orchestrator=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Processing orchestrator starting...");

    // TODO: Initialize Kafka consumers/producers
    // TODO: Start worker pool
    // TODO: Begin processing jobs

    tracing::info!("Processing orchestrator running");

    // Keep the service running
    tokio::signal::ctrl_c().await?;

    tracing::info!("Shutting down processing orchestrator");

    Ok(())
}
