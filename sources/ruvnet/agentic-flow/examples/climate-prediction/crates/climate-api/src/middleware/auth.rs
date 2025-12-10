use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::Arc;

use crate::state::AppState;

const API_KEY_HEADER: &str = "x-api-key";

pub async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<Response, AuthError> {
    // Extract API key from header
    let api_key = headers
        .get(API_KEY_HEADER)
        .and_then(|v| v.to_str().ok())
        .ok_or_else(|| AuthError::MissingApiKey)?;

    // Validate API key
    let key_info = state
        .validate_api_key(api_key)
        .ok_or_else(|| AuthError::InvalidApiKey)?;

    tracing::debug!("Authenticated request with key: {}", key_info.name);

    // Update last used timestamp
    state.update_api_key_usage(api_key);

    // Record authentication metric
    metrics::counter!("api.auth.success").increment(1);

    // Continue with request
    Ok(next.run(request).await)
}

#[derive(Debug)]
pub enum AuthError {
    MissingApiKey,
    InvalidApiKey,
}

impl IntoResponse for AuthError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            AuthError::MissingApiKey => {
                (StatusCode::UNAUTHORIZED, "Missing API key header")
            }
            AuthError::InvalidApiKey => {
                (StatusCode::UNAUTHORIZED, "Invalid API key")
            }
        };

        metrics::counter!("api.auth.failures").increment(1);

        (status, message).into_response()
    }
}
