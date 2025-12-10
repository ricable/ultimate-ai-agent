use axum::{
    extract::State,
    middleware,
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use std::sync::Arc;
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod handlers;
mod middleware as mw;
mod routes;
mod state;

use state::AppState;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "climate_api=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    tracing::info!("🚀 Starting Climate Prediction API Server");

    // Initialize application state
    let state = AppState::new().await?;
    let app_state = Arc::new(state);

    // Initialize metrics
    let prometheus_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()?;

    // Build CORS layer
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    // Build API routes
    let api_routes = Router::new()
        .route("/predictions", get(routes::predictions::get_prediction))
        .route("/predictions", post(routes::predictions::create_prediction))
        .route("/predictions/:id", get(routes::predictions::get_prediction_by_id))
        .route("/health", get(routes::health::health_check))
        .route("/metrics", get(routes::metrics::metrics_handler))
        .layer(
            ServiceBuilder::new()
                .layer(middleware::from_fn_with_state(
                    app_state.clone(),
                    mw::auth::auth_middleware,
                ))
                .layer(middleware::from_fn(mw::rate_limit::rate_limit_middleware))
                .layer(TimeoutLayer::new(std::time::Duration::from_secs(30)))
                .layer(TraceLayer::new_for_http()),
        )
        .with_state(app_state.clone());

    // Build main router
    let app = Router::new()
        .nest("/api", api_routes)
        .route("/", get(root_handler))
        .layer(cors)
        .with_state(prometheus_handle);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("🎯 Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn root_handler() -> &'static str {
    "Climate Prediction API v0.1.0 - Visit /api/health for status"
}
