//! Synaptic QuDAG Core - DAG networking and consensus for neural mesh networks
//!
//! This crate provides the foundational DAG (Directed Acyclic Graph) networking
//! and consensus mechanisms for the Synaptic Neural Mesh project.

use std::sync::Arc;
use blake3::Hasher;
use ed25519_dalek::{Signature, Signer, VerifyingKey, SigningKey, Verifier};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;
use dashmap::DashMap;
use parking_lot::RwLock;

/// QuDAG specific errors
#[derive(Error, Debug)]
pub enum QuDAGError {
    #[error("Validation error: {0}")]
    ValidationError(String),
    
    #[error("Cryptographic error: {0}")]
    CryptoError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Consensus error: {0}")]
    ConsensusError(String),
}

pub type Result<T> = std::result::Result<T, QuDAGError>;

/// A node in the QuDAG network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuDAGNode {
    pub id: Uuid,
    pub data: Vec<u8>,
    pub hash: Vec<u8>,
    pub parents: Vec<Uuid>,
    pub timestamp: u64,
    pub signature: Option<Vec<u8>>,
}

impl QuDAGNode {
    /// Create a new QuDAG node
    pub fn new(data: &[u8]) -> Self {
        let id = Uuid::new_v4();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let mut hasher = Hasher::new();
        hasher.update(id.as_bytes());
        hasher.update(data);
        hasher.update(&timestamp.to_le_bytes());
        let hash = hasher.finalize();
        
        Self {
            id,
            data: data.to_vec(),
            hash: hash.as_bytes().to_vec(),
            parents: Vec::new(),
            timestamp,
            signature: None,
        }
    }
    
    /// Add parent references
    pub fn add_parents(&mut self, parents: Vec<Uuid>) {
        self.parents = parents;
        self.update_hash();
    }
    
    /// Update the node's hash
    fn update_hash(&mut self) {
        let mut hasher = Hasher::new();
        hasher.update(self.id.as_bytes());
        hasher.update(&self.data);
        hasher.update(&self.timestamp.to_le_bytes());
        for parent in &self.parents {
            hasher.update(parent.as_bytes());
        }
        let hash = hasher.finalize();
        self.hash = hash.as_bytes().to_vec();
    }
    
    /// Get the node's data
    pub fn data(&self) -> &[u8] {
        &self.data
    }
    
    /// Get the node's signature
    pub fn signature(&self) -> Option<&[u8]> {
        self.signature.as_deref()
    }
}

/// QuDAG network manager
pub struct QuDAGNetwork {
    nodes: Arc<DashMap<Uuid, QuDAGNode>>,
    tips: Arc<RwLock<Vec<Uuid>>>,
}

impl QuDAGNetwork {
    /// Create a new QuDAG network
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(DashMap::new()),
            tips: Arc::new(RwLock::new(Vec::new())),
        }
    }
    
    /// Add a node to the network
    pub async fn add_node(&self, mut node: QuDAGNode) -> Result<Uuid> {
        // Select tips as parents if this is not the genesis node
        if !self.nodes.is_empty() {
            let tips = self.tips.read().clone();
            if !tips.is_empty() {
                let parents = tips.iter().take(2).cloned().collect();
                node.add_parents(parents);
            }
        }
        
        let id = node.id;
        self.nodes.insert(id, node);
        
        // Update tips
        self.update_tips(id).await;
        
        Ok(id)
    }
    
    /// Update the tips of the DAG
    async fn update_tips(&self, new_tip: Uuid) {
        let mut tips = self.tips.write();
        
        // Remove any tips that are now parents of the new node
        if let Some(node) = self.nodes.get(&new_tip) {
            tips.retain(|tip| !node.parents.contains(tip));
        }
        
        tips.push(new_tip);
        
        // Keep only the most recent tips
        if tips.len() > 10 {
            let drain_to = tips.len() - 10;
            tips.drain(0..drain_to);
        }
    }
    
    /// Get a node by ID
    pub fn get_node(&self, id: &Uuid) -> Option<QuDAGNode> {
        self.nodes.get(id).map(|node| node.clone())
    }
    
    /// Get current tips
    pub fn get_tips(&self) -> Vec<Uuid> {
        self.tips.read().clone()
    }
    
    /// Get total node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

impl Default for QuDAGNetwork {
    fn default() -> Self {
        Self::new()
    }
}

/// Consensus mechanism for QuDAG
pub struct QuDAGConsensus {
    network: Arc<QuDAGNetwork>,
    confirmation_threshold: usize,
}

impl QuDAGConsensus {
    /// Create a new consensus mechanism
    pub fn new(network: Arc<QuDAGNetwork>) -> Self {
        Self {
            network,
            confirmation_threshold: 2,
        }
    }
    
    /// Check if a node is confirmed
    pub fn is_confirmed(&self, node_id: &Uuid) -> bool {
        let mut confirmations = 0;
        
        // Count how many other nodes reference this one
        for entry in self.network.nodes.iter() {
            if entry.value().parents.contains(node_id) {
                confirmations += 1;
                if confirmations >= self.confirmation_threshold {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Get confirmation depth of a node
    pub fn confirmation_depth(&self, node_id: &Uuid) -> usize {
        let mut depth = 0;
        let mut current_layer = vec![*node_id];
        let mut visited = std::collections::HashSet::new();
        
        while !current_layer.is_empty() {
            let mut next_layer = Vec::new();
            
            for node_id in current_layer {
                if visited.insert(node_id) {
                    // Find all nodes that reference this one
                    for entry in self.network.nodes.iter() {
                        if entry.value().parents.contains(&node_id) {
                            next_layer.push(entry.key().clone());
                        }
                    }
                }
            }
            
            if !next_layer.is_empty() {
                depth += 1;
                current_layer = next_layer;
            } else {
                break;
            }
        }
        
        depth
    }
}

/// Cryptographic utilities
pub mod crypto {
    use super::*;
    use rand::rngs::OsRng;
    use rand::RngCore;
    
    /// Generate a new Ed25519 keypair
    pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
        let mut csprng = OsRng;
        let mut secret_bytes = [0u8; 32];
        csprng.fill_bytes(&mut secret_bytes);
        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let verifying_key = signing_key.verifying_key();
        (signing_key, verifying_key)
    }
    
    /// Sign a QuDAG node
    pub fn sign_node(node: &mut QuDAGNode, signing_key: &SigningKey) -> Result<()> {
        let message = [
            node.id.as_bytes().as_slice(),
            &node.data,
            &node.timestamp.to_le_bytes(),
            &node.hash,
        ].concat();
        
        let signature = signing_key.sign(&message);
        node.signature = Some(signature.to_bytes().to_vec());
        
        Ok(())
    }
    
    /// Verify a node's signature
    pub fn verify_node_signature(node: &QuDAGNode, verifying_key: &VerifyingKey) -> Result<bool> {
        let signature_bytes = node.signature()
            .ok_or_else(|| QuDAGError::ValidationError("No signature found".to_string()))?;
        
        if signature_bytes.len() != 64 {
            return Err(QuDAGError::CryptoError("Invalid signature length".to_string()));
        }
        
        let signature_array: [u8; 64] = signature_bytes.try_into()
            .map_err(|_| QuDAGError::CryptoError("Failed to convert signature".to_string()))?;
        
        let signature = Signature::from_bytes(&signature_array);
        
        let message = [
            node.id.as_bytes().as_slice(),
            &node.data,
            &node.timestamp.to_le_bytes(),
            &node.hash,
        ].concat();
        
        verifying_key.verify(&message, &signature)
            .map_err(|e| QuDAGError::CryptoError(format!("Signature verification failed: {}", e)))?;
        
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_dag_creation() {
        let network = QuDAGNetwork::new();
        
        // Add genesis node
        let genesis = QuDAGNode::new(b"genesis");
        let genesis_id = network.add_node(genesis).await.unwrap();
        
        // Add child nodes
        let child1 = QuDAGNode::new(b"child1");
        let child1_id = network.add_node(child1).await.unwrap();
        
        let child2 = QuDAGNode::new(b"child2");
        let child2_id = network.add_node(child2).await.unwrap();
        
        // Check that children reference genesis
        let child1_node = network.get_node(&child1_id).unwrap();
        assert!(child1_node.parents.contains(&genesis_id));
        
        // Check tips
        let tips = network.get_tips();
        assert!(tips.contains(&child1_id) || tips.contains(&child2_id));
    }
    
    #[test]
    fn test_node_signing() {
        let mut node = QuDAGNode::new(b"test data");
        let (signing_key, verifying_key) = crypto::generate_keypair();
        
        // Sign the node
        crypto::sign_node(&mut node, &signing_key).unwrap();
        assert!(node.signature.is_some());
        
        // Verify the signature
        let is_valid = crypto::verify_node_signature(&node, &verifying_key).unwrap();
        assert!(is_valid);
    }
    
    #[tokio::test]
    async fn test_consensus() {
        let network = Arc::new(QuDAGNetwork::new());
        let consensus = QuDAGConsensus::new(network.clone());
        
        // Add a few nodes to test basic functionality
        let node1 = QuDAGNode::new(b"node1");
        let node1_id = network.add_node(node1).await.unwrap();
        
        let node2 = QuDAGNode::new(b"node2");
        network.add_node(node2).await.unwrap();
        
        // Basic tests
        assert_eq!(network.node_count(), 2);
        assert!(network.get_node(&node1_id).is_some());
        
        // Test that consensus functions don't panic
        let _ = consensus.is_confirmed(&node1_id);
        let _ = consensus.confirmation_depth(&node1_id);
    }
}