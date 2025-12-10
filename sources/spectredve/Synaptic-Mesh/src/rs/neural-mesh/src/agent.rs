//! Neural agents that form the basis of distributed cognition

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc};
use uuid::Uuid;
use serde::{Deserialize, Serialize};

use crate::{NeuralMeshError, Result, NeuralMesh, MeshNode, ThoughtPattern};

/// Neural agent that performs distributed cognition
#[derive(Debug, Clone)]
pub struct NeuralAgent {
    id: Uuid,
    config: AgentConfig,
    state: Arc<RwLock<AgentState>>,
    network: Arc<RwLock<ruv_fann::Network>>,
    mesh_node: Arc<RwLock<MeshNode>>,
    message_tx: mpsc::UnboundedSender<AgentMessage>,
    metrics: Arc<RwLock<AgentMetrics>>,
}

impl NeuralAgent {
    /// Create a new neural agent
    pub async fn new(config: AgentConfig, mesh: Arc<NeuralMesh>) -> Result<Self> {
        let id = Uuid::new_v4();
        let network = ruv_fann::Network::new(&config.neural_config)?;
        let mesh_node = MeshNode::new(id, config.capabilities.clone());
        
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        
        let agent = Self {
            id,
            config: config.clone(),
            state: Arc::new(RwLock::new(AgentState::Idle)),
            network: Arc::new(RwLock::new(network)),
            mesh_node: Arc::new(RwLock::new(mesh_node)),
            message_tx,
            metrics: Arc::new(RwLock::new(AgentMetrics::new())),
        };

        // Spawn message processing task
        let agent_clone = agent.clone();
        tokio::spawn(async move {
            agent_clone.process_messages(message_rx).await;
        });

        Ok(agent)
    }

    /// Get the agent's unique ID
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Get the agent's capabilities
    pub fn capabilities(&self) -> Vec<String> {
        self.config.capabilities.clone()
    }

    /// Check if the agent is active
    pub fn is_active(&self) -> bool {
        // This would check if the agent is currently processing tasks
        true // Simplified for now
    }

    /// Stop the agent
    pub async fn stop(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        *state = AgentState::Stopped;
        Ok(())
    }

    /// Get the mesh node representation
    pub async fn get_node(&self) -> MeshNode {
        let node = self.mesh_node.read().await;
        node.clone()
    }

    /// Process a thought pattern
    pub async fn think(&self, pattern: ThoughtPattern) -> Result<ThoughtPattern> {
        let start_time = Instant::now();
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Thinking;
        }

        // Process through neural network
        let result = {
            let mut network = self.network.write().await;
            let input = pattern.to_input_vector()?;
            let output = network.run(&input)?;
            ThoughtPattern::from_output_vector(output, pattern.context.clone())?
        };

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.thoughts_processed += 1;
            metrics.total_processing_time += start_time.elapsed();
            metrics.last_activity = Instant::now();
        }

        // Return to idle state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Idle;
        }

        Ok(result)
    }

    /// Learn from a thought pattern
    pub async fn learn(&self, pattern: &ThoughtPattern, target: &ThoughtPattern) -> Result<()> {
        let mut network = self.network.write().await;
        let input = pattern.to_input_vector()?;
        let expected_output = target.to_input_vector()?;
        
        network.train_single(&input, &expected_output)?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.training_iterations += 1;
        }

        Ok(())
    }

    /// Send a message to another agent
    pub async fn send_message(&self, to: Uuid, content: MessageContent) -> Result<()> {
        let message = AgentMessage {
            from: self.id,
            to,
            content,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        self.message_tx.send(message)
            .map_err(|_| NeuralMeshError::Communication("Failed to send message".to_string()))?;

        Ok(())
    }

    /// Get agent metrics
    pub async fn get_metrics(&self) -> AgentMetrics {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    /// Process incoming messages
    async fn process_messages(&self, mut message_rx: mpsc::UnboundedReceiver<AgentMessage>) {
        while let Some(message) = message_rx.recv().await {
            if let Err(e) = self.handle_message(message).await {
                tracing::error!("Error handling message: {}", e);
            }
        }
    }

    /// Handle a single message
    async fn handle_message(&self, message: AgentMessage) -> Result<()> {
        match message.content {
            MessageContent::ThoughtShare(pattern) => {
                self.learn_from_peer(&pattern).await?;
            }
            MessageContent::CollaborationRequest(task) => {
                self.handle_collaboration_request(task).await?;
            }
            MessageContent::SyncRequest => {
                self.handle_sync_request(message.from).await?;
            }
            MessageContent::ModelUpdate(weights) => {
                self.apply_model_update(weights).await?;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.messages_received += 1;
        }

        Ok(())
    }

    /// Learn from a peer's thought pattern
    async fn learn_from_peer(&self, pattern: &ThoughtPattern) -> Result<()> {
        // This would implement peer learning logic
        // For now, just update the network with the pattern
        let mut network = self.network.write().await;
        let input = pattern.to_input_vector()?;
        
        // Use the pattern as both input and target for unsupervised learning
        network.train_single(&input, &input)?;
        
        Ok(())
    }

    /// Handle collaboration request
    async fn handle_collaboration_request(&self, _task: String) -> Result<()> {
        // This would implement collaborative processing
        // For now, just acknowledge the request
        Ok(())
    }

    /// Handle sync request from another agent
    async fn handle_sync_request(&self, from: Uuid) -> Result<()> {
        // Send current model weights to the requesting agent
        let network = self.network.read().await;
        let weights = network.get_weights()?;
        
        self.send_message(from, MessageContent::ModelUpdate(weights)).await?;
        Ok(())
    }

    /// Apply model update from another agent
    async fn apply_model_update(&self, weights: Vec<f32>) -> Result<()> {
        let mut network = self.network.write().await;
        
        // Average with current weights (simple model averaging)
        let current_weights = network.get_weights()?;
        let averaged_weights: Vec<f32> = current_weights
            .iter()
            .zip(weights.iter())
            .map(|(current, new)| (current + new) / 2.0)
            .collect();
        
        network.set_weights(&averaged_weights)?;
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.model_syncs += 1;
        }

        Ok(())
    }
}

/// Configuration for a neural agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub name: String,
    pub neural_config: ruv_fann::NetworkConfig,
    pub capabilities: Vec<String>,
    pub max_connections: usize,
    pub learning_rate: f64,
}

/// Current state of a neural agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Thinking,
    Learning,
    Communicating,
    Syncing,
    Stopped,
}

/// Metrics for monitoring agent performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub thoughts_processed: u64,
    pub training_iterations: u64,
    pub messages_sent: u64,
    pub messages_received: u64,
    pub model_syncs: u64,
    pub total_processing_time: Duration,
    pub last_activity: Instant,
    pub accuracy: f64,
}

impl AgentMetrics {
    fn new() -> Self {
        Self {
            thoughts_processed: 0,
            training_iterations: 0,
            messages_sent: 0,
            messages_received: 0,
            model_syncs: 0,
            total_processing_time: Duration::ZERO,
            last_activity: Instant::now(),
            accuracy: 0.0,
        }
    }

    pub fn average_processing_time(&self) -> Duration {
        if self.thoughts_processed > 0 {
            self.total_processing_time / self.thoughts_processed as u32
        } else {
            Duration::ZERO
        }
    }
}

/// Messages exchanged between agents
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMessage {
    pub from: Uuid,
    pub to: Uuid,
    pub content: MessageContent,
    pub timestamp: u64,
}

/// Content of agent messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageContent {
    ThoughtShare(ThoughtPattern),
    CollaborationRequest(String),
    SyncRequest,
    ModelUpdate(Vec<f32>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_creation() {
        let config = AgentConfig {
            name: "test-agent".to_string(),
            neural_config: ruv_fann::NetworkConfig::default(),
            capabilities: vec!["test".to_string()],
            max_connections: 5,
            learning_rate: 0.01,
        };

        // This test would need a mock NeuralMesh
        // For now, just test the configuration
        assert_eq!(config.name, "test-agent");
        assert_eq!(config.capabilities, vec!["test"]);
    }

    #[tokio::test]
    async fn test_agent_metrics() {
        let metrics = AgentMetrics::new();
        assert_eq!(metrics.thoughts_processed, 0);
        assert_eq!(metrics.training_iterations, 0);
    }

    #[test]
    fn test_agent_message_serialization() {
        let message = AgentMessage {
            from: Uuid::new_v4(),
            to: Uuid::new_v4(),
            content: MessageContent::SyncRequest,
            timestamp: 1234567890,
        };

        let serialized = serde_json::to_string(&message).unwrap();
        let deserialized: AgentMessage = serde_json::from_str(&serialized).unwrap();

        assert_eq!(message.from, deserialized.from);
        assert_eq!(message.to, deserialized.to);
        assert_eq!(message.timestamp, deserialized.timestamp);
    }
}