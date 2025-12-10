//! RUV-Swarm MCP (Model Context Protocol) Server
//!
//! Provides MCP server integration for RUV-Swarm, enabling Claude and other
//! MCP-compatible clients to interact with the swarm orchestration system.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    extract::{State, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, RwLock};
use tower_http::cors::CorsLayer;
use tracing::{debug, error, info};
use uuid::Uuid;

pub mod handlers;
pub mod orchestrator;
pub mod tools;
pub mod types;

use crate::orchestrator::SwarmOrchestrator;

use crate::handlers::RequestHandler;
use crate::tools::ToolRegistry;

/// MCP Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    /// Server bind address
    pub bind_addr: SocketAddr,
    /// Maximum concurrent connections
    pub max_connections: usize,
    /// Request timeout in seconds
    pub request_timeout_secs: u64,
    /// Enable debug logging
    pub debug: bool,
}

impl Default for McpConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1:3000".parse().unwrap(),
            max_connections: 100,
            request_timeout_secs: 300,
            debug: false,
        }
    }
}

/// MCP Server state
pub struct McpServerState {
    /// Swarm orchestrator instance
    orchestrator: Arc<SwarmOrchestrator>,
    /// Tool registry
    tools: Arc<ToolRegistry>,
    /// Active sessions
    sessions: Arc<DashMap<Uuid, Arc<Session>>>,
    /// Server configuration
    config: McpConfig,
}

/// Client session
pub struct Session {
    pub id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: RwLock<chrono::DateTime<chrono::Utc>>,
    pub metadata: DashMap<String, Value>,
}

/// MCP Server
pub struct McpServer {
    state: Arc<McpServerState>,
}

impl McpServer {
    /// Create a new MCP server
    pub fn new(orchestrator: Arc<SwarmOrchestrator>, config: McpConfig) -> Self {
        let tools = Arc::new(ToolRegistry::new());

        // Register all tools
        tools::register_tools(&tools);

        let state = Arc::new(McpServerState {
            orchestrator,
            tools,
            sessions: Arc::new(DashMap::new()),
            config,
        });

        Self { state }
    }

    /// Start the MCP server
    pub async fn start(&self) -> anyhow::Result<()> {
        let app = self.build_router();
        let addr = self.state.config.bind_addr;

        info!("Starting MCP server on {}", addr);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Build the router
    fn build_router(&self) -> Router {
        Router::new()
            .route("/", get(root_handler))
            .route("/mcp", get(websocket_handler))
            .route("/tools", get(list_tools_handler))
            .route("/health", get(health_handler))
            .layer(CorsLayer::permissive())
            .with_state(self.state.clone())
    }
}

/// Root handler
async fn root_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "name": "ruv-swarm-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "protocol": "mcp/1.0",
        "endpoints": {
            "websocket": "/mcp",
            "tools": "/tools",
            "health": "/health"
        }
    }))
}

/// Health check handler
async fn health_handler(State(state): State<Arc<McpServerState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "active_sessions": state.sessions.len(),
        "tools_count": state.tools.count(),
        "timestamp": chrono::Utc::now()
    }))
}

/// List available tools
async fn list_tools_handler(State(state): State<Arc<McpServerState>>) -> impl IntoResponse {
    let tools = state.tools.list_tools();
    Json(serde_json::json!({
        "tools": tools,
        "count": tools.len()
    }))
}

/// WebSocket handler for MCP protocol
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<McpServerState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle WebSocket connection
async fn handle_socket(socket: axum::extract::ws::WebSocket, state: Arc<McpServerState>) {
    let session_id = Uuid::new_v4();
    let session = Arc::new(Session {
        id: session_id,
        created_at: chrono::Utc::now(),
        last_activity: RwLock::new(chrono::Utc::now()),
        metadata: DashMap::new(),
    });

    state.sessions.insert(session_id, session.clone());
    info!("New MCP session: {}", session_id);

    let (mut sender, mut receiver) = socket.split();
    let (tx, mut rx) = mpsc::channel(100);

    // Spawn task to handle outgoing messages
    let tx_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sender.send(msg).await.is_err() {
                break;
            }
        }
    });

    // Create request handler
    let handler = RequestHandler::new(
        state.orchestrator.clone(),
        state.tools.clone(),
        session.clone(),
        tx.clone(),
    );

    // Handle incoming messages
    while let Some(Ok(msg)) = receiver.next().await {
        if let axum::extract::ws::Message::Text(text) = msg {
            match serde_json::from_str::<McpRequest>(&text) {
                Ok(request) => {
                    debug!("Received MCP request: {:?}", request.method);

                    // Update last activity
                    *session.last_activity.write().await = chrono::Utc::now();

                    // Handle request
                    match handler.handle_request(request).await {
                        Ok(response) => {
                            if let Ok(json) = serde_json::to_string(&response) {
                                let _ = tx.send(axum::extract::ws::Message::Text(json)).await;
                            }
                        }
                        Err(e) => {
                            error!("Error handling request: {}", e);
                            let error_response =
                                McpResponse::error(None, -32603, format!("Internal error: {}", e));
                            if let Ok(json) = serde_json::to_string(&error_response) {
                                let _ = tx.send(axum::extract::ws::Message::Text(json)).await;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to parse MCP request: {}", e);
                    let error_response =
                        McpResponse::error(None, -32700, "Parse error".to_string());
                    if let Ok(json) = serde_json::to_string(&error_response) {
                        let _ = tx.send(axum::extract::ws::Message::Text(json)).await;
                    }
                }
            }
        }
    }

    // Cleanup
    tx_task.abort();
    state.sessions.remove(&session_id);
    info!("MCP session closed: {}", session_id);
}

/// MCP Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: Option<Value>,
    pub id: Option<Value>,
}

/// MCP Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
}

impl McpResponse {
    pub fn success(id: Option<Value>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: Some(result),
            error: None,
            id,
        }
    }

    pub fn error(id: Option<Value>, code: i32, message: String) -> Self {
        Self {
            jsonrpc: "2.0".to_string(),
            result: None,
            error: Some(McpError {
                code,
                message,
                data: None,
            }),
            id,
        }
    }
}

/// MCP Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mcp_server_creation() {
        use ruv_swarm_core::SwarmConfig;
        let config = SwarmConfig::default();
        let orchestrator = Arc::new(SwarmOrchestrator::new(config));
        let mcp_config = McpConfig::default();

        let server = McpServer::new(orchestrator, mcp_config);
        assert!(server.state.tools.count() > 0);
    }
}
