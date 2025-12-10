use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde::Serialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ApiError {
    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Internal server error: {0}")]
    InternalError(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_message) = match self {
            ApiError::ValidationError(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::NotFound(msg) => (StatusCode::NOT_FOUND, msg),
            ApiError::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg),
            ApiError::RateLimitExceeded => {
                (StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded".to_string())
            }
            ApiError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        let body = Json(ErrorResponse {
            error: error_message,
            status_code: status.as_u16(),
        });

        (status, body).into_response()
    }
}

#[derive(Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub status_code: u16,
}

#[derive(Serialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: T,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T) -> Self {
        Self {
            success: true,
            data,
        }
    }
}
