use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::aos_heal::{RestconfPayload, ExecutionResult, HealingAction, HealingActionType};

/// ENM (Ericsson Network Manager) client for executing healing actions
#[derive(Debug, Clone)]
pub struct EnmClient {
    pub base_url: String,
    pub auth_token: String,
    pub timeout: u64,
    pub client: reqwest::Client,
}

impl EnmClient {
    pub fn new(base_url: String, auth_token: String) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .expect("Failed to create HTTP client");
        
        Self {
            base_url,
            auth_token,
            timeout: 30,
            client,
        }
    }

    /// Execute RESTCONF payload against ENM
    pub async fn execute_restconf(&self, payload: &RestconfPayload) -> Result<ExecutionResult, Box<dyn std::error::Error>> {
        let start_time = std::time::Instant::now();
        
        let mut request_builder = match payload.method.as_str() {
            "GET" => self.client.get(&payload.endpoint),
            "POST" => self.client.post(&payload.endpoint),
            "PUT" => self.client.put(&payload.endpoint),
            "PATCH" => self.client.patch(&payload.endpoint),
            "DELETE" => self.client.delete(&payload.endpoint),
            _ => return Err("Unsupported HTTP method".into()),
        };
        
        // Add headers
        for (key, value) in &payload.headers {
            request_builder = request_builder.header(key, value);
        }
        
        // Add body if present
        if let Some(body) = &payload.body {
            request_builder = request_builder.body(body.clone());
        }
        
        // Execute request
        let response = request_builder.send().await?;
        let status_code = response.status().as_u16();
        let success = response.status().is_success();
        
        let duration = start_time.elapsed().as_millis() as u64;
        
        Ok(ExecutionResult {
            action: HealingAction {
                action_type: HealingActionType::ProcessRestart, // Default
                target_entity: "unknown".to_string(),
                parameters: HashMap::new(),
                priority: 0.0,
                confidence: 0.0,
                estimated_duration: 0,
                rollback_plan: None,
            },
            success,
            error: if success { None } else { Some(format!("HTTP {}", status_code)) },
            duration,
        })
    }

    /// Get cell information from ENM
    pub async fn get_cell_info(&self, cell_id: &str) -> Result<CellInfo, Box<dyn std::error::Error>> {
        let url = format!("{}/restconf/data/network-topology/cells/{}", self.base_url, cell_id);
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("Accept", "application/json")
            .send()
            .await?;
        
        if response.status().is_success() {
            let cell_info: CellInfo = response.json().await?;
            Ok(cell_info)
        } else {
            Err(format!("Failed to get cell info: {}", response.status()).into())
        }
    }

    /// Get node information from ENM
    pub async fn get_node_info(&self, node_id: &str) -> Result<NodeInfo, Box<dyn std::error::Error>> {
        let url = format!("{}/restconf/data/network-topology/nodes/{}", self.base_url, node_id);
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("Accept", "application/json")
            .send()
            .await?;
        
        if response.status().is_success() {
            let node_info: NodeInfo = response.json().await?;
            Ok(node_info)
        } else {
            Err(format!("Failed to get node info: {}", response.status()).into())
        }
    }

    /// Execute AMOS script on ENM
    pub async fn execute_amos_script(&self, script: &str, target_node: &str) -> Result<AmosExecutionResult, Box<dyn std::error::Error>> {
        let url = format!("{}/restconf/operations/amos-execute", self.base_url);
        
        let payload = AmosExecutionPayload {
            script: script.to_string(),
            target_node: target_node.to_string(),
            timeout: self.timeout,
        };
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;
        
        if response.status().is_success() {
            let result: AmosExecutionResult = response.json().await?;
            Ok(result)
        } else {
            Err(format!("Failed to execute AMOS script: {}", response.status()).into())
        }
    }

    /// Block/unblock a cell
    pub async fn set_cell_state(&self, cell_id: &str, state: CellState) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/restconf/data/network-topology/cells/{}/state", self.base_url, cell_id);
        
        let payload = CellStatePayload {
            cell_state: state,
        };
        
        let response = self.client
            .patch(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(format!("Failed to set cell state: {}", response.status()).into());
        }
        
        Ok(())
    }

    /// Restart a process on a node
    pub async fn restart_process(&self, node_id: &str, process_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!("{}/restconf/operations/restart-process", self.base_url);
        
        let payload = ProcessRestartPayload {
            node_id: node_id.to_string(),
            process_name: process_name.to_string(),
        };
        
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await?;
        
        if !response.status().is_success() {
            return Err(format!("Failed to restart process: {}", response.status()).into());
        }
        
        Ok(())
    }

    /// Get network performance metrics
    pub async fn get_performance_metrics(&self, time_range: TimeRange) -> Result<PerformanceMetrics, Box<dyn std::error::Error>> {
        let url = format!("{}/restconf/data/performance-metrics", self.base_url);
        
        let response = self.client
            .get(&url)
            .header("Authorization", format!("Bearer {}", self.auth_token))
            .header("Accept", "application/json")
            .query(&[
                ("start_time", time_range.start_time.to_string()),
                ("end_time", time_range.end_time.to_string()),
            ])
            .send()
            .await?;
        
        if response.status().is_success() {
            let metrics: PerformanceMetrics = response.json().await?;
            Ok(metrics)
        } else {
            Err(format!("Failed to get performance metrics: {}", response.status()).into())
        }
    }
}

/// Cell information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellInfo {
    pub cell_id: String,
    pub cell_name: String,
    pub cell_state: CellState,
    pub serving_area: String,
    pub frequency_band: String,
    pub max_power: f64,
    pub current_load: f64,
    pub connected_users: u32,
}

/// Node information structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeInfo {
    pub node_id: String,
    pub node_name: String,
    pub node_type: String,
    pub software_version: String,
    pub operational_state: String,
    pub administrative_state: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub processes: Vec<ProcessInfo>,
}

/// Process information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    pub process_name: String,
    pub process_id: u32,
    pub status: String,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub restart_count: u32,
}

/// Cell state enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellState {
    Active,
    Blocked,
    Disabled,
    Maintenance,
}

/// Time range for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start_time: chrono::DateTime<chrono::Utc>,
    pub end_time: chrono::DateTime<chrono::Utc>,
}

/// Performance metrics structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub throughput_mbps: f64,
    pub latency_ms: f64,
    pub packet_loss_rate: f64,
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub active_connections: u32,
    pub error_rate: f64,
}

/// AMOS execution payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmosExecutionPayload {
    pub script: String,
    pub target_node: String,
    pub timeout: u64,
}

/// AMOS execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmosExecutionResult {
    pub success: bool,
    pub output: String,
    pub error_message: Option<String>,
    pub execution_time: u64,
}

/// Cell state change payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellStatePayload {
    pub cell_state: CellState,
}

/// Process restart payload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessRestartPayload {
    pub node_id: String,
    pub process_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enm_client_creation() {
        let client = EnmClient::new(
            "https://test.com".to_string(),
            "test_token".to_string(),
        );
        
        assert_eq!(client.base_url, "https://test.com");
        assert_eq!(client.auth_token, "test_token");
        assert_eq!(client.timeout, 30);
    }

    #[test]
    fn test_cell_info_serialization() {
        let cell_info = CellInfo {
            cell_id: "cell_123".to_string(),
            cell_name: "TestCell".to_string(),
            cell_state: CellState::Active,
            serving_area: "Area1".to_string(),
            frequency_band: "2600MHz".to_string(),
            max_power: 100.0,
            current_load: 0.7,
            connected_users: 150,
        };
        
        let serialized = serde_json::to_string(&cell_info).unwrap();
        let deserialized: CellInfo = serde_json::from_str(&serialized).unwrap();
        
        assert_eq!(cell_info.cell_id, deserialized.cell_id);
        assert_eq!(cell_info.connected_users, deserialized.connected_users);
    }
}