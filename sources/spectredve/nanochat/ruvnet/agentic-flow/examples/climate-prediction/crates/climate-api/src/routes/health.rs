use axum::{http::StatusCode, response::IntoResponse, Json};
use serde::Serialize;

#[derive(Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

static START_TIME: std::sync::LazyLock<std::time::Instant> =
    std::sync::LazyLock::new(std::time::Instant::now);

/// GET /api/health
pub async fn health_check() -> impl IntoResponse {
    let uptime = START_TIME.elapsed().as_secs();

    let response = HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: uptime,
        timestamp: chrono::Utc::now(),
    };

    (StatusCode::OK, Json(response))
}
