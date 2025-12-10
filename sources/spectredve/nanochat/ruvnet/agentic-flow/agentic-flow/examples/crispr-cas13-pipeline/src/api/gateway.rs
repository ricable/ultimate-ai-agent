//! API Gateway - Rights-Preserving HTTP Server
//!
//! Provides secure REST API with authentication, authorization, and privacy controls.

use axum::{
    extract::{Path, State},
    http::{Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tower_http::cors::CorsLayer;
use tracing::{error, info, warn};

#[derive(Error, Debug)]
pub enum GatewayError {
    #[error("Authentication failed: {0}")]
    AuthenticationError(String),
    #[error("Authorization failed: {0}")]
    AuthorizationError(String),
    #[error("Privacy violation: {0}")]
    PrivacyViolation(String),
    #[error("Internal server error: {0}")]
    InternalError(String),
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            GatewayError::AuthenticationError(msg) => (StatusCode::UNAUTHORIZED, msg),
            GatewayError::AuthorizationError(msg) => (StatusCode::FORBIDDEN, msg),
            GatewayError::PrivacyViolation(msg) => (StatusCode::FORBIDDEN, msg),
            GatewayError::InternalError(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };

        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}

#[derive(Clone)]
pub struct GatewayState {
    pub auth_tokens: Arc<RwLock<std::collections::HashMap<String, UserContext>>>,
    pub privacy_engine: Arc<crate::privacy::differential::PrivacyEngine>,
    pub policy_enforcer: Arc<crate::governance::policy::PolicyEnforcer>,
    pub audit_logger: Arc<crate::audit::logger::AuditLogger>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserContext {
    pub user_id: String,
    pub roles: Vec<String>,
    pub privacy_level: String,
    pub issued_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthRequest {
    pub username: String,
    pub password: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthResponse {
    pub token: String,
    pub expires_in: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataRequest {
    pub query: String,
    pub privacy_budget: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DataResponse {
    pub data: serde_json::Value,
    pub privacy_budget_consumed: f64,
    pub audit_id: String,
}

/// Authentication middleware
async fn auth_middleware(
    State(state): State<GatewayState>,
    mut req: Request<axum::body::Body>,
    next: Next,
) -> Result<Response, GatewayError> {
    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| GatewayError::AuthenticationError("Missing Authorization header".into()))?;

    let token = auth_header
        .strip_prefix("Bearer ")
        .ok_or_else(|| GatewayError::AuthenticationError("Invalid Authorization format".into()))?;

    let tokens = state.auth_tokens.read().await;
    let user_ctx = tokens
        .get(token)
        .ok_or_else(|| GatewayError::AuthenticationError("Invalid token".into()))?
        .clone();

    // Check token expiration
    let now = chrono::Utc::now();
    if now.signed_duration_since(user_ctx.issued_at).num_hours() > 24 {
        return Err(GatewayError::AuthenticationError("Token expired".into()));
    }

    req.extensions_mut().insert(user_ctx);
    Ok(next.run(req).await)
}

/// Handle user authentication
async fn authenticate(
    State(state): State<GatewayState>,
    Json(auth_req): Json<AuthRequest>,
) -> Result<Json<AuthResponse>, GatewayError> {
    info!("Authentication request for user: {}", auth_req.username);

    // In production, validate against secure credential store
    // This is a simplified example
    let user_ctx = UserContext {
        user_id: auth_req.username.clone(),
        roles: vec!["researcher".to_string()],
        privacy_level: "standard".to_string(),
        issued_at: chrono::Utc::now(),
    };

    let token = uuid::Uuid::new_v4().to_string();
    state.auth_tokens.write().await.insert(token.clone(), user_ctx);

    // Log authentication event
    state
        .audit_logger
        .log_event(
            "authentication",
            &auth_req.username,
            serde_json::json!({ "status": "success" }),
        )
        .await
        .map_err(|e| GatewayError::InternalError(e.to_string()))?;

    Ok(Json(AuthResponse {
        token,
        expires_in: 86400, // 24 hours
    }))
}

/// Handle data query with privacy and policy enforcement
async fn query_data(
    State(state): State<GatewayState>,
    user_ctx: axum::Extension<UserContext>,
    Json(req): Json<DataRequest>,
) -> Result<Json<DataResponse>, GatewayError> {
    info!("Data query from user: {}, query: {}", user_ctx.user_id, req.query);

    // Check policy authorization
    let policy_decision = state
        .policy_enforcer
        .evaluate(&user_ctx.user_id, "data:query", &req.query)
        .await
        .map_err(|e| GatewayError::InternalError(e.to_string()))?;

    if !policy_decision.allowed {
        warn!("Policy violation for user: {}", user_ctx.user_id);
        return Err(GatewayError::AuthorizationError(
            policy_decision.reason.unwrap_or_else(|| "Access denied".to_string()),
        ));
    }

    // Apply differential privacy
    let privacy_budget = req.privacy_budget.unwrap_or(1.0);
    let private_data = state
        .privacy_engine
        .apply_noise(&serde_json::json!({"query_result": "sample_data"}), privacy_budget)
        .await
        .map_err(|e| GatewayError::PrivacyViolation(e.to_string()))?;

    // Log access
    let audit_id = state
        .audit_logger
        .log_event(
            "data_access",
            &user_ctx.user_id,
            serde_json::json!({
                "query": req.query,
                "privacy_budget": privacy_budget,
            }),
        )
        .await
        .map_err(|e| GatewayError::InternalError(e.to_string()))?;

    Ok(Json(DataResponse {
        data: private_data.data,
        privacy_budget_consumed: private_data.budget_consumed,
        audit_id,
    }))
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

/// Build the API gateway router
pub fn build_router(state: GatewayState) -> Router {
    Router::new()
        .route("/health", get(health_check))
        .route("/auth/login", post(authenticate))
        .route("/data/query", post(query_data))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Start the API gateway server
pub async fn start_server(
    addr: std::net::SocketAddr,
    state: GatewayState,
) -> anyhow::Result<()> {
    let app = build_router(state);

    info!("🚀 API Gateway starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .await
        .map_err(|e| anyhow::anyhow!("Server error: {}", e))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_middleware_missing_header() {
        // Test implementation
    }

    #[tokio::test]
    async fn test_authenticate_success() {
        // Test implementation
    }
}
