//! API route handlers

use axum::Json;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
}

pub async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Deserialize)]
pub struct CreateTargetRequest {
    guide_rna: String,
    target_sequence: String,
}

#[derive(Serialize)]
pub struct CreateTargetResponse {
    id: String,
    message: String,
}

pub async fn create_target(
    Json(_payload): Json<CreateTargetRequest>,
) -> Json<CreateTargetResponse> {
    Json(CreateTargetResponse {
        id: uuid::Uuid::new_v4().to_string(),
        message: "Target created successfully".to_string(),
    })
}

#[derive(Deserialize)]
pub struct PredictRequest {
    target_id: String,
}

#[derive(Serialize)]
pub struct PredictResponse {
    prediction_id: String,
    status: String,
}

pub async fn predict_offtargets(Json(_payload): Json<PredictRequest>) -> Json<PredictResponse> {
    Json(PredictResponse {
        prediction_id: uuid::Uuid::new_v4().to_string(),
        status: "processing".to_string(),
    })
}
