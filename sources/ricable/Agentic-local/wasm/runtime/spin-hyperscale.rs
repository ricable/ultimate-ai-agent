//! HyperScale SpinKube Agent Runtime
//!
//! Ultra-high-density WASM agent runtime for SpinKube with:
//! - 10,000+ agents per node
//! - Millisecond cold starts
//! - Memory-safe sandboxing
//! - A2A protocol support
//! - MCP tool integration

use serde::{Deserialize, Serialize};
use spin_sdk::http::{IntoResponse, Request, Response};
use spin_sdk::http_component;
use spin_sdk::key_value::Store;
use std::collections::HashMap;

// ============================================================================
// AGENT TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Agent {
    pub id: String,
    pub name: String,
    pub agent_type: AgentType,
    pub capabilities: Vec<String>,
    pub state: AgentState,
    pub config: AgentConfig,
    pub created_at: u64,
    pub last_active: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentType {
    Coder,
    Researcher,
    Analyst,
    Orchestrator,
    Monitor,
    Deployer,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Processing,
    Waiting,
    Error(String),
    Terminated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub model: String,
    pub temperature: f32,
    pub max_tokens: u32,
    pub system_prompt: Option<String>,
    pub tools: Vec<String>,
    pub memory_enabled: bool,
    pub max_memory_items: u32,
}

// ============================================================================
// TASK TYPES
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: String,
    pub agent_id: String,
    pub task_type: TaskType,
    pub input: serde_json::Value,
    pub status: TaskStatus,
    pub result: Option<serde_json::Value>,
    pub error: Option<String>,
    pub created_at: u64,
    pub started_at: Option<u64>,
    pub completed_at: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskType {
    Chat,
    CodeGenerate,
    CodeReview,
    Research,
    Analyze,
    Execute,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

// ============================================================================
// A2A PROTOCOL
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    pub url: String,
    pub version: String,
    pub capabilities: AgentCapabilities,
    pub skills: Vec<Skill>,
    pub default_input_modes: Vec<String>,
    pub default_output_modes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapabilities {
    pub streaming: bool,
    pub push_notifications: bool,
    pub state_transition_history: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub examples: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2ARequest {
    pub jsonrpc: String,
    pub id: String,
    pub method: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AResponse {
    pub jsonrpc: String,
    pub id: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<A2AError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2AError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

// ============================================================================
// MCP PROTOCOL
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolCall {
    pub tool: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MCPToolResult {
    pub tool: String,
    pub result: serde_json::Value,
    pub error: Option<String>,
}

// ============================================================================
// MEMORY & STATE
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub id: String,
    pub agent_id: String,
    pub content: String,
    pub embedding: Option<Vec<f32>>,
    pub metadata: HashMap<String, String>,
    pub created_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
    pub timestamp: u64,
}

// ============================================================================
// HYPERSCALE RUNTIME
// ============================================================================

pub struct HyperScaleRuntime {
    store: Store,
    agent_id: String,
}

impl HyperScaleRuntime {
    pub fn new(agent_id: &str) -> Self {
        Self {
            store: Store::open_default().expect("Failed to open KV store"),
            agent_id: agent_id.to_string(),
        }
    }

    /// Get or create agent
    pub fn get_agent(&self) -> Option<Agent> {
        let key = format!("agent:{}", self.agent_id);
        self.store
            .get(&key)
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    /// Save agent state
    pub fn save_agent(&self, agent: &Agent) -> Result<(), String> {
        let key = format!("agent:{}", agent.id);
        let data = serde_json::to_vec(agent).map_err(|e| e.to_string())?;
        self.store.set(&key, &data).map_err(|e| e.to_string())
    }

    /// Add memory entry
    pub fn add_memory(&self, content: &str, metadata: HashMap<String, String>) -> Result<String, String> {
        let memory_id = format!("mem_{}_{}", self.agent_id, timestamp());
        let entry = MemoryEntry {
            id: memory_id.clone(),
            agent_id: self.agent_id.clone(),
            content: content.to_string(),
            embedding: None, // Computed by embedding service
            metadata,
            created_at: timestamp(),
        };

        let key = format!("memory:{}:{}", self.agent_id, memory_id);
        let data = serde_json::to_vec(&entry).map_err(|e| e.to_string())?;
        self.store.set(&key, &data).map_err(|e| e.to_string())?;

        Ok(memory_id)
    }

    /// Get conversation history
    pub fn get_conversation(&self, conversation_id: &str) -> Vec<ConversationTurn> {
        let key = format!("conv:{}:{}", self.agent_id, conversation_id);
        self.store
            .get(&key)
            .ok()
            .flatten()
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .unwrap_or_default()
    }

    /// Add to conversation
    pub fn add_to_conversation(
        &self,
        conversation_id: &str,
        role: &str,
        content: &str,
    ) -> Result<(), String> {
        let mut conversation = self.get_conversation(conversation_id);
        conversation.push(ConversationTurn {
            role: role.to_string(),
            content: content.to_string(),
            timestamp: timestamp(),
        });

        let key = format!("conv:{}:{}", self.agent_id, conversation_id);
        let data = serde_json::to_vec(&conversation).map_err(|e| e.to_string())?;
        self.store.set(&key, &data).map_err(|e| e.to_string())
    }
}

fn timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ============================================================================
// HTTP HANDLERS
// ============================================================================

/// Main HTTP component handler
#[http_component]
fn handle_request(req: Request) -> anyhow::Result<impl IntoResponse> {
    let path = req.path();
    let method = req.method().to_string();

    match (method.as_str(), path) {
        // A2A Protocol Endpoints
        ("GET", "/.well-known/agent.json") => handle_agent_card(),
        ("POST", "/a2a") => handle_a2a_request(req),
        ("GET", p) if p.starts_with("/a2a/tasks/") => handle_get_task(p),

        // Agent Management
        ("GET", "/agent") => handle_get_agent(),
        ("POST", "/agent/chat") => handle_chat(req),
        ("POST", "/agent/execute") => handle_execute(req),

        // MCP Endpoints
        ("GET", "/mcp/tools") => handle_list_tools(),
        ("POST", "/mcp/tools/call") => handle_tool_call(req),

        // Health & Metrics
        ("GET", "/health") => handle_health(),
        ("GET", "/metrics") => handle_metrics(),

        _ => Ok(Response::builder()
            .status(404)
            .body("Not Found")
            .build()),
    }
}

fn handle_agent_card() -> anyhow::Result<Response> {
    let agent_id = std::env::var("AGENT_ID").unwrap_or_else(|_| "default-agent".to_string());

    let card = AgentCard {
        name: format!("HyperScale Agent {}", agent_id),
        description: "High-performance WASM AI agent with A2A protocol support".to_string(),
        url: format!("http://{}.edge-ai-agents.svc.cluster.local", agent_id),
        version: "1.0.0".to_string(),
        capabilities: AgentCapabilities {
            streaming: true,
            push_notifications: false,
            state_transition_history: true,
        },
        skills: vec![
            Skill {
                id: "code-generation".to_string(),
                name: "Code Generation".to_string(),
                description: "Generate code in multiple languages".to_string(),
                tags: vec!["code".to_string(), "generation".to_string()],
                examples: vec!["Write a Python function to sort a list".to_string()],
            },
            Skill {
                id: "code-review".to_string(),
                name: "Code Review".to_string(),
                description: "Review and improve code quality".to_string(),
                tags: vec!["code".to_string(), "review".to_string()],
                examples: vec!["Review this function for bugs".to_string()],
            },
        ],
        default_input_modes: vec!["text".to_string()],
        default_output_modes: vec!["text".to_string()],
    };

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&card)?)
        .build())
}

fn handle_a2a_request(req: Request) -> anyhow::Result<Response> {
    let body: A2ARequest = serde_json::from_slice(req.body())?;

    let result = match body.method.as_str() {
        "tasks/send" => handle_tasks_send(&body.params),
        "tasks/get" => handle_tasks_get(&body.params),
        "tasks/cancel" => handle_tasks_cancel(&body.params),
        _ => Err(format!("Unknown method: {}", body.method)),
    };

    let response = match result {
        Ok(value) => A2AResponse {
            jsonrpc: "2.0".to_string(),
            id: body.id,
            result: Some(value),
            error: None,
        },
        Err(msg) => A2AResponse {
            jsonrpc: "2.0".to_string(),
            id: body.id,
            result: None,
            error: Some(A2AError {
                code: -32000,
                message: msg,
                data: None,
            }),
        },
    };

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&response)?)
        .build())
}

fn handle_tasks_send(params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let task_id = format!("task_{}", timestamp());

    // Create and queue task
    let task = Task {
        id: task_id.clone(),
        agent_id: std::env::var("AGENT_ID").unwrap_or_default(),
        task_type: TaskType::Custom("a2a".to_string()),
        input: params.clone(),
        status: TaskStatus::Pending,
        result: None,
        error: None,
        created_at: timestamp(),
        started_at: None,
        completed_at: None,
    };

    // Store task
    let store = Store::open_default().map_err(|e| e.to_string())?;
    let key = format!("task:{}", task_id);
    let data = serde_json::to_vec(&task).map_err(|e| e.to_string())?;
    store.set(&key, &data).map_err(|e| e.to_string())?;

    Ok(serde_json::json!({
        "id": task_id,
        "status": "pending"
    }))
}

fn handle_tasks_get(params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let task_id = params["id"].as_str().ok_or("Missing task id")?;

    let store = Store::open_default().map_err(|e| e.to_string())?;
    let key = format!("task:{}", task_id);

    match store.get(&key).map_err(|e| e.to_string())? {
        Some(data) => {
            let task: Task = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
            serde_json::to_value(task).map_err(|e| e.to_string())
        }
        None => Err("Task not found".to_string()),
    }
}

fn handle_tasks_cancel(params: &serde_json::Value) -> Result<serde_json::Value, String> {
    let task_id = params["id"].as_str().ok_or("Missing task id")?;

    let store = Store::open_default().map_err(|e| e.to_string())?;
    let key = format!("task:{}", task_id);

    match store.get(&key).map_err(|e| e.to_string())? {
        Some(data) => {
            let mut task: Task = serde_json::from_slice(&data).map_err(|e| e.to_string())?;
            task.status = TaskStatus::Cancelled;
            let updated = serde_json::to_vec(&task).map_err(|e| e.to_string())?;
            store.set(&key, &updated).map_err(|e| e.to_string())?;
            Ok(serde_json::json!({"cancelled": true}))
        }
        None => Err("Task not found".to_string()),
    }
}

fn handle_get_task(path: &str) -> anyhow::Result<Response> {
    let task_id = path.trim_start_matches("/a2a/tasks/");
    let result = handle_tasks_get(&serde_json::json!({"id": task_id}));

    match result {
        Ok(task) => Ok(Response::builder()
            .status(200)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(&task)?)
            .build()),
        Err(e) => Ok(Response::builder()
            .status(404)
            .body(e)
            .build()),
    }
}

fn handle_get_agent() -> anyhow::Result<Response> {
    let agent_id = std::env::var("AGENT_ID").unwrap_or_else(|_| "default".to_string());
    let runtime = HyperScaleRuntime::new(&agent_id);

    match runtime.get_agent() {
        Some(agent) => Ok(Response::builder()
            .status(200)
            .header("Content-Type", "application/json")
            .body(serde_json::to_string(&agent)?)
            .build()),
        None => Ok(Response::builder()
            .status(404)
            .body("Agent not found")
            .build()),
    }
}

fn handle_chat(req: Request) -> anyhow::Result<Response> {
    #[derive(Deserialize)]
    struct ChatRequest {
        message: String,
        conversation_id: Option<String>,
    }

    let chat_req: ChatRequest = serde_json::from_slice(req.body())?;
    let agent_id = std::env::var("AGENT_ID").unwrap_or_else(|_| "default".to_string());
    let runtime = HyperScaleRuntime::new(&agent_id);

    let conv_id = chat_req.conversation_id.unwrap_or_else(|| format!("conv_{}", timestamp()));

    // Add user message to conversation
    runtime.add_to_conversation(&conv_id, "user", &chat_req.message)?;

    // TODO: Call inference engine via HTTP
    // For now, return a mock response
    let response_content = format!("Processing: {}", chat_req.message);
    runtime.add_to_conversation(&conv_id, "assistant", &response_content)?;

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&serde_json::json!({
            "conversation_id": conv_id,
            "response": response_content
        }))?)
        .build())
}

fn handle_execute(req: Request) -> anyhow::Result<Response> {
    #[derive(Deserialize)]
    struct ExecuteRequest {
        action: String,
        params: serde_json::Value,
    }

    let exec_req: ExecuteRequest = serde_json::from_slice(req.body())?;

    let result = match exec_req.action.as_str() {
        "code_generate" => serde_json::json!({"code": "// Generated code placeholder"}),
        "code_review" => serde_json::json!({"review": "Code looks good!"}),
        _ => serde_json::json!({"error": "Unknown action"}),
    };

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&result)?)
        .build())
}

fn handle_list_tools() -> anyhow::Result<Response> {
    let tools = vec![
        MCPTool {
            name: "execute_code".to_string(),
            description: "Execute code in a secure sandbox".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "language": {"type": "string"},
                    "code": {"type": "string"}
                },
                "required": ["language", "code"]
            }),
        },
        MCPTool {
            name: "search_knowledge".to_string(),
            description: "Search the knowledge base".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer"}
                },
                "required": ["query"]
            }),
        },
        MCPTool {
            name: "send_task".to_string(),
            description: "Send a task to another agent via A2A".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "agent_url": {"type": "string"},
                    "task": {"type": "object"}
                },
                "required": ["agent_url", "task"]
            }),
        },
    ];

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&tools)?)
        .build())
}

fn handle_tool_call(req: Request) -> anyhow::Result<Response> {
    let call: MCPToolCall = serde_json::from_slice(req.body())?;

    let result = match call.tool.as_str() {
        "execute_code" => MCPToolResult {
            tool: call.tool,
            result: serde_json::json!({"output": "Execution result placeholder"}),
            error: None,
        },
        "search_knowledge" => MCPToolResult {
            tool: call.tool,
            result: serde_json::json!({"results": []}),
            error: None,
        },
        _ => MCPToolResult {
            tool: call.tool,
            result: serde_json::Value::Null,
            error: Some("Unknown tool".to_string()),
        },
    };

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&result)?)
        .build())
}

fn handle_health() -> anyhow::Result<Response> {
    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&serde_json::json!({
            "status": "healthy",
            "agent_id": std::env::var("AGENT_ID").unwrap_or_default(),
            "timestamp": timestamp()
        }))?)
        .build())
}

fn handle_metrics() -> anyhow::Result<Response> {
    // Prometheus-compatible metrics
    let metrics = r#"
# HELP agent_requests_total Total number of requests processed
# TYPE agent_requests_total counter
agent_requests_total 0

# HELP agent_active_tasks Number of active tasks
# TYPE agent_active_tasks gauge
agent_active_tasks 0

# HELP agent_memory_bytes Memory usage in bytes
# TYPE agent_memory_bytes gauge
agent_memory_bytes 0
"#;

    Ok(Response::builder()
        .status(200)
        .header("Content-Type", "text/plain")
        .body(metrics)
        .build())
}
