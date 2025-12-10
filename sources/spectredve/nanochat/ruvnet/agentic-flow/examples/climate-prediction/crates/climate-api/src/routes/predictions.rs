use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use uuid::Uuid;
use validator::Validate;

use crate::handlers::prediction_handler::{ApiError, ApiResponse};
use crate::state::AppState;

#[derive(Debug, Deserialize, Validate)]
pub struct PredictionQuery {
    #[validate(range(min = -90.0, max = 90.0))]
    pub lat: f64,

    #[validate(range(min = -180.0, max = 180.0))]
    pub lon: f64,

    #[validate(range(min = 1, max = 30))]
    #[serde(default = "default_days")]
    pub days: u32,

    #[serde(default)]
    pub include_confidence: bool,
}

fn default_days() -> u32 {
    7
}

#[derive(Debug, Deserialize, Validate)]
pub struct PredictionRequest {
    #[validate(range(min = -90.0, max = 90.0))]
    pub latitude: f64,

    #[validate(range(min = -180.0, max = 180.0))]
    pub longitude: f64,

    #[validate(range(min = 1, max = 30))]
    pub forecast_days: u32,

    #[serde(default)]
    pub parameters: Vec<String>,

    #[serde(default)]
    pub include_raw_data: bool,
}

#[derive(Debug, Serialize)]
pub struct PredictionResponse {
    pub id: Uuid,
    pub location: Location,
    pub forecast: Vec<ForecastDay>,
    pub metadata: PredictionMetadata,
}

#[derive(Debug, Serialize)]
pub struct Location {
    pub latitude: f64,
    pub longitude: f64,
    pub name: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ForecastDay {
    pub date: String,
    pub temperature_max: f64,
    pub temperature_min: f64,
    pub precipitation: f64,
    pub humidity: f64,
    pub wind_speed: f64,
    pub conditions: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct PredictionMetadata {
    pub model_version: String,
    pub generated_at: chrono::DateTime<chrono::Utc>,
    pub computation_time_ms: u64,
}

/// GET /api/predictions?lat=37.7749&lon=-122.4194&days=7
pub async fn get_prediction(
    Query(params): Query<PredictionQuery>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ApiError> {
    // Validate input
    params.validate()
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    tracing::info!(
        "Generating prediction for lat={}, lon={}, days={}",
        params.lat,
        params.lon,
        params.days
    );

    // Record metrics
    metrics::counter!("api.predictions.requests").increment(1);
    let start = std::time::Instant::now();

    // Generate prediction (mock data for now)
    let prediction = generate_mock_prediction(
        params.lat,
        params.lon,
        params.days,
        params.include_confidence,
    );

    let elapsed = start.elapsed().as_millis() as u64;
    metrics::histogram!("api.predictions.duration_ms").record(elapsed as f64);

    // Cache the prediction
    state.cache_prediction(
        prediction.id,
        serde_json::to_value(&prediction).unwrap(),
        3600, // 1 hour TTL
    );

    Ok(Json(ApiResponse::success(prediction)))
}

/// POST /api/predictions
pub async fn create_prediction(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<PredictionRequest>,
) -> Result<impl IntoResponse, ApiError> {
    // Validate input
    payload.validate()
        .map_err(|e| ApiError::ValidationError(e.to_string()))?;

    tracing::info!(
        "Creating prediction for lat={}, lon={}, days={}",
        payload.latitude,
        payload.longitude,
        payload.forecast_days
    );

    // Record metrics
    metrics::counter!("api.predictions.created").increment(1);
    let start = std::time::Instant::now();

    // Generate prediction
    let prediction = generate_mock_prediction(
        payload.latitude,
        payload.longitude,
        payload.forecast_days,
        true,
    );

    let elapsed = start.elapsed().as_millis() as u64;

    // Cache the prediction
    state.cache_prediction(
        prediction.id,
        serde_json::to_value(&prediction).unwrap(),
        3600,
    );

    Ok((
        StatusCode::CREATED,
        Json(ApiResponse::success(prediction)),
    ))
}

/// GET /api/predictions/:id
pub async fn get_prediction_by_id(
    Path(id): Path<Uuid>,
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ApiError> {
    tracing::info!("Fetching prediction with id={}", id);

    // Check cache
    if let Some(cached) = state.get_cached_prediction(&id) {
        metrics::counter!("api.predictions.cache_hits").increment(1);
        return Ok(Json(ApiResponse::success(cached)));
    }

    metrics::counter!("api.predictions.cache_misses").increment(1);
    Err(ApiError::NotFound(format!("Prediction {} not found", id)))
}

// Mock prediction generator (replace with actual prediction logic)
fn generate_mock_prediction(
    lat: f64,
    lon: f64,
    days: u32,
    include_confidence: bool,
) -> PredictionResponse {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let forecast: Vec<ForecastDay> = (0..days)
        .map(|day| {
            let date = chrono::Utc::now() + chrono::Duration::days(day as i64);
            ForecastDay {
                date: date.format("%Y-%m-%d").to_string(),
                temperature_max: 20.0 + rng.gen_range(-5.0..10.0),
                temperature_min: 10.0 + rng.gen_range(-5.0..5.0),
                precipitation: rng.gen_range(0.0..50.0),
                humidity: rng.gen_range(40.0..90.0),
                wind_speed: rng.gen_range(0.0..30.0),
                conditions: ["Sunny", "Cloudy", "Rainy", "Partly Cloudy"][rng.gen_range(0..4)].to_string(),
                confidence: if include_confidence {
                    Some(rng.gen_range(0.7..0.99))
                } else {
                    None
                },
            }
        })
        .collect();

    PredictionResponse {
        id: Uuid::new_v4(),
        location: Location {
            latitude: lat,
            longitude: lon,
            name: Some("Unknown Location".to_string()),
        },
        forecast,
        metadata: PredictionMetadata {
            model_version: "v1.0.0".to_string(),
            generated_at: chrono::Utc::now(),
            computation_time_ms: 42,
        },
    }
}
