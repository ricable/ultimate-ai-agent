//! Communication Protocol for Swarm Agents
//! 
//! This module handles inter-agent communication and information sharing.

use crate::models::{RANConfiguration, RANMetrics};
use crate::neural::neural_agent::Experience;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    FitnessReport,
    ConfigurationShare,
    ExperienceShare,
    CoordinationRequest,
    StatusUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMessage {
    pub id: String,
    pub sender_id: String,
    pub receiver_id: Option<String>, // None for broadcast
    pub message_type: MessageType,
    pub payload: MessagePayload,
    pub timestamp: u64,
    pub priority: u8, // 0-255, higher is more important
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessagePayload {
    Fitness(f32),
    Configuration(RANConfiguration),
    Experience(Experience),
    StatusInfo(AgentStatus),
    CoordinationData(CoordinationInfo),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub agent_id: String,
    pub current_fitness: f32,
    pub iterations_without_improvement: u32,
    pub exploration_rate: f32,
    pub specialization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationInfo {
    pub suggested_action: String,
    pub confidence: f32,
    pub reasoning: String,
}

pub struct CommunicationProtocol {
    pub message_queue: VecDeque<SwarmMessage>,
    pub message_history: HashMap<String, Vec<SwarmMessage>>,
    pub blacklist: Vec<String>,
    pub max_queue_size: usize,
    pub max_history_per_agent: usize,
}

impl CommunicationProtocol {
    pub fn new() -> Self {
        Self {
            message_queue: VecDeque::new(),
            message_history: HashMap::new(),
            blacklist: Vec::new(),
            max_queue_size: 1000,
            max_history_per_agent: 100,
        }
    }
    
    pub fn send_message(&mut self, mut message: SwarmMessage) -> Result<(), String> {
        // Validate message
        if self.blacklist.contains(&message.sender_id) {
            return Err("Sender is blacklisted".to_string());
        }
        
        // Set timestamp if not set
        if message.timestamp == 0 {
            message.timestamp = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }
        
        // Add to queue
        if self.message_queue.len() >= self.max_queue_size {
            // Remove oldest low-priority message
            if let Some(pos) = self.message_queue.iter()
                .position(|m| m.priority < 128) {
                self.message_queue.remove(pos);
            } else {
                return Err("Message queue is full".to_string());
            }
        }
        
        // Insert in priority order
        let insert_pos = self.message_queue.iter()
            .position(|m| m.priority < message.priority)
            .unwrap_or(self.message_queue.len());
        
        self.message_queue.insert(insert_pos, message.clone());
        
        // Store in history
        let history = self.message_history
            .entry(message.sender_id.clone())
            .or_insert_with(Vec::new);
        
        history.push(message);
        
        // Trim history if needed
        if history.len() > self.max_history_per_agent {
            history.remove(0);
        }
        
        Ok(())
    }
    
    pub fn receive_messages_for(&mut self, agent_id: &str) -> Vec<SwarmMessage> {
        let mut messages = Vec::new();
        let mut i = 0;
        
        while i < self.message_queue.len() {
            let message = &self.message_queue[i];
            
            // Check if message is for this agent (or broadcast)
            if message.receiver_id.is_none() || 
               message.receiver_id.as_ref() == Some(&agent_id.to_string()) {
                
                // Don't receive own messages
                if message.sender_id != agent_id {
                    messages.push(self.message_queue.remove(i).unwrap());
                } else {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        
        messages
    }
    
    pub fn broadcast_fitness(&mut self, sender_id: String, fitness: f32) -> Result<(), String> {
        let message = SwarmMessage {
            id: format!("fitness_{}_{}", sender_id, self.get_current_timestamp()),
            sender_id,
            receiver_id: None,
            message_type: MessageType::FitnessReport,
            payload: MessagePayload::Fitness(fitness),
            timestamp: self.get_current_timestamp(),
            priority: 100,
        };
        
        self.send_message(message)
    }
    
    pub fn share_configuration(
        &mut self,
        sender_id: String,
        receiver_id: Option<String>,
        config: RANConfiguration,
    ) -> Result<(), String> {
        let message = SwarmMessage {
            id: format!("config_{}_{}", sender_id, self.get_current_timestamp()),
            sender_id,
            receiver_id,
            message_type: MessageType::ConfigurationShare,
            payload: MessagePayload::Configuration(config),
            timestamp: self.get_current_timestamp(),
            priority: 150,
        };
        
        self.send_message(message)
    }
    
    pub fn share_experience(
        &mut self,
        sender_id: String,
        receiver_id: Option<String>,
        experience: Experience,
    ) -> Result<(), String> {
        let message = SwarmMessage {
            id: format!("exp_{}_{}", sender_id, self.get_current_timestamp()),
            sender_id,
            receiver_id,
            message_type: MessageType::ExperienceShare,
            payload: MessagePayload::Experience(experience),
            timestamp: self.get_current_timestamp(),
            priority: 120,
        };
        
        self.send_message(message)
    }
    
    pub fn request_coordination(
        &mut self,
        sender_id: String,
        receiver_id: String,
        info: CoordinationInfo,
    ) -> Result<(), String> {
        let message = SwarmMessage {
            id: format!("coord_{}_{}", sender_id, self.get_current_timestamp()),
            sender_id,
            receiver_id: Some(receiver_id),
            message_type: MessageType::CoordinationRequest,
            payload: MessagePayload::CoordinationData(info),
            timestamp: self.get_current_timestamp(),
            priority: 200,
        };
        
        self.send_message(message)
    }
    
    pub fn get_agent_communications(&self, agent_id: &str) -> Option<&Vec<SwarmMessage>> {
        self.message_history.get(agent_id)
    }
    
    pub fn get_recent_messages(&self, agent_id: &str, count: usize) -> Vec<SwarmMessage> {
        if let Some(history) = self.message_history.get(agent_id) {
            history.iter()
                .rev()
                .take(count)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }
    
    pub fn add_to_blacklist(&mut self, agent_id: String) {
        if !self.blacklist.contains(&agent_id) {
            self.blacklist.push(agent_id);
        }
    }
    
    pub fn remove_from_blacklist(&mut self, agent_id: &str) {
        self.blacklist.retain(|id| id != agent_id);
    }
    
    pub fn get_queue_size(&self) -> usize {
        self.message_queue.len()
    }
    
    pub fn clear_old_messages(&mut self, max_age_seconds: u64) {
        let current_time = self.get_current_timestamp();
        let cutoff_time = current_time.saturating_sub(max_age_seconds);
        
        // Clear from queue
        self.message_queue.retain(|msg| msg.timestamp >= cutoff_time);
        
        // Clear from history
        for history in self.message_history.values_mut() {
            history.retain(|msg| msg.timestamp >= cutoff_time);
        }
    }
    
    fn get_current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }
    
    pub fn get_communication_stats(&self) -> CommunicationStats {
        let total_messages: usize = self.message_history.values()
            .map(|history| history.len())
            .sum();
        
        let agents_count = self.message_history.len();
        let avg_messages_per_agent = if agents_count > 0 {
            total_messages as f32 / agents_count as f32
        } else {
            0.0
        };
        
        let queue_utilization = if self.max_queue_size > 0 {
            (self.message_queue.len() as f32 / self.max_queue_size as f32) * 100.0
        } else {
            0.0
        };
        
        CommunicationStats {
            total_messages,
            active_agents: agents_count,
            queue_size: self.message_queue.len(),
            avg_messages_per_agent,
            queue_utilization,
            blacklisted_agents: self.blacklist.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CommunicationStats {
    pub total_messages: usize,
    pub active_agents: usize,
    pub queue_size: usize,
    pub avg_messages_per_agent: f32,
    pub queue_utilization: f32,
    pub blacklisted_agents: usize,
}

impl Default for CommunicationProtocol {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_sending() {
        let mut protocol = CommunicationProtocol::new();
        
        let message = SwarmMessage {
            id: "test_1".to_string(),
            sender_id: "agent_1".to_string(),
            receiver_id: Some("agent_2".to_string()),
            message_type: MessageType::FitnessReport,
            payload: MessagePayload::Fitness(0.85),
            timestamp: 1234567890,
            priority: 100,
        };
        
        assert!(protocol.send_message(message).is_ok());
        assert_eq!(protocol.get_queue_size(), 1);
    }
    
    #[test]
    fn test_message_receiving() {
        let mut protocol = CommunicationProtocol::new();
        
        let message = SwarmMessage {
            id: "test_1".to_string(),
            sender_id: "agent_1".to_string(),
            receiver_id: Some("agent_2".to_string()),
            message_type: MessageType::FitnessReport,
            payload: MessagePayload::Fitness(0.85),
            timestamp: 1234567890,
            priority: 100,
        };
        
        protocol.send_message(message).unwrap();
        
        let received = protocol.receive_messages_for("agent_2");
        assert_eq!(received.len(), 1);
        assert_eq!(protocol.get_queue_size(), 0);
    }
    
    #[test]
    fn test_broadcast() {
        let mut protocol = CommunicationProtocol::new();
        
        protocol.broadcast_fitness("agent_1".to_string(), 0.95).unwrap();
        
        let received_2 = protocol.receive_messages_for("agent_2");
        let received_3 = protocol.receive_messages_for("agent_3");
        
        assert_eq!(received_2.len(), 1);
        assert_eq!(received_3.len(), 1);
    }
}