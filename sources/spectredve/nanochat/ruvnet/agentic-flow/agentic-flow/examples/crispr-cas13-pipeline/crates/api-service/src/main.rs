//! # API Service
//!
//! REST API service for the CRISPR-Cas13 bioinformatics pipeline.

use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod error;
mod routes;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "api_service=info,tower_http=debug".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Build application router
    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/api/v1/targets", post(routes::create_target))
        .route("/api/v1/predict", post(routes::predict_offtargets));

    // Start server
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    tracing::info!("API service listening on {}", addr);

    // Axum 0.7+ uses tokio::net::TcpListener
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
