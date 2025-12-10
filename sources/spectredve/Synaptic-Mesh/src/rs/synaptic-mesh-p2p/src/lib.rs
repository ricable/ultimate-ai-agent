//! Synaptic Mesh P2P Networking Layer
//! 
//! This module provides WebAssembly bindings for the QuDAG P2P networking
//! components, enabling quantum-resistant mesh networking in JavaScript/Node.js
//! environments.

#![deny(unsafe_code)]

use wasm_bindgen::prelude::*;
use js_sys::{Object, Reflect, Array, Uint8Array, Promise};
use web_sys::console;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod crypto;
mod network;
mod dag;
mod peer;
mod mesh;
mod utils;

pub use crypto::*;
pub use network::*;
pub use dag::*;
pub use peer::*;
pub use mesh::*;
pub use utils::*;

// Enable console_error_panic_hook for better error messages in development
#[cfg(feature = "console_error_panic_hook")]
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

/// Initialize the WASM module
#[wasm_bindgen]
pub fn init() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
    
    log("Synaptic Mesh P2P WASM module initialized");
}

/// Logging utility
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

/// Version information
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// Get module information
#[wasm_bindgen]
pub fn get_info() -> Object {
    let info = Object::new();
    Reflect::set(&info, &"name".into(), &"synaptic-mesh-p2p".into()).unwrap();
    Reflect::set(&info, &"version".into(), &version().into()).unwrap();
    Reflect::set(&info, &"description".into(), &"P2P networking layer for Synaptic Neural Mesh".into()).unwrap();
    info
}

/// Error type for WASM operations
#[derive(Debug, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct P2PError {
    message: String,
    code: String,
}

#[wasm_bindgen]
impl P2PError {
    #[wasm_bindgen(constructor)]
    pub fn new(message: &str, code: &str) -> P2PError {
        P2PError {
            message: message.to_string(),
            code: code.to_string(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn code(&self) -> String {
        self.code.clone()
    }
}

impl From<anyhow::Error> for P2PError {
    fn from(error: anyhow::Error) -> Self {
        P2PError {
            message: error.to_string(),
            code: "INTERNAL_ERROR".to_string(),
        }
    }
}

/// Configuration for P2P networking
#[derive(Debug, Clone, Serialize, Deserialize)]
#[wasm_bindgen]
pub struct P2PConfig {
    port: u16,
    bootstrap_peers: Vec<String>,
    network_id: String,
    enable_dht: bool,
    enable_quantum_crypto: bool,
}

#[wasm_bindgen]
impl P2PConfig {
    #[wasm_bindgen(constructor)]
    pub fn new() -> P2PConfig {
        P2PConfig {
            port: 8080,
            bootstrap_peers: Vec::new(),
            network_id: "synaptic-mesh".to_string(),
            enable_dht: true,
            enable_quantum_crypto: true,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn port(&self) -> u16 {
        self.port
    }

    #[wasm_bindgen(setter)]
    pub fn set_port(&mut self, port: u16) {
        self.port = port;
    }

    #[wasm_bindgen(getter)]
    pub fn network_id(&self) -> String {
        self.network_id.clone()
    }

    #[wasm_bindgen(setter)]
    pub fn set_network_id(&mut self, id: &str) {
        self.network_id = id.to_string();
    }

    #[wasm_bindgen]
    pub fn add_bootstrap_peer(&mut self, peer: &str) {
        self.bootstrap_peers.push(peer.to_string());
    }

    #[wasm_bindgen]
    pub fn get_bootstrap_peers(&self) -> Array {
        let array = Array::new();
        for peer in &self.bootstrap_peers {
            array.push(&JsValue::from_str(peer));
        }
        array
    }
}

impl Default for P2PConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Main P2P networking interface
#[wasm_bindgen]
pub struct SynapticMeshP2P {
    config: P2PConfig,
    peers: HashMap<String, PeerInfo>,
    network: Option<Network>,
    crypto: CryptoEngine,
    dag: DAGNode,
}

#[wasm_bindgen]
impl SynapticMeshP2P {
    #[wasm_bindgen(constructor)]
    pub fn new(config: P2PConfig) -> Result<SynapticMeshP2P, P2PError> {
        Ok(SynapticMeshP2P {
            config,
            peers: HashMap::new(),
            network: None,
            crypto: CryptoEngine::new()?,
            dag: DAGNode::new()?,
        })
    }

    /// Initialize the P2P network
    #[wasm_bindgen]
    pub async fn init(&mut self) -> Result<(), P2PError> {
        log("Initializing Synaptic Mesh P2P network...");
        
        // Initialize crypto engine
        self.crypto.init()?;
        
        // Initialize DAG node
        self.dag.init()?;
        
        // Create network instance
        self.network = Some(Network::new(&self.config)?);
        
        log("P2P network initialized successfully");
        Ok(())
    }

    /// Start the P2P network
    #[wasm_bindgen]
    pub async fn start(&mut self) -> Result<(), P2PError> {
        if let Some(ref mut network) = self.network {
            network.start().await?;
            log("P2P network started");
        } else {
            return Err(P2PError::new("Network not initialized", "NETWORK_NOT_INITIALIZED"));
        }
        Ok(())
    }

    /// Stop the P2P network
    #[wasm_bindgen]
    pub async fn stop(&mut self) -> Result<(), P2PError> {
        if let Some(ref mut network) = self.network {
            network.stop().await?;
            log("P2P network stopped");
        }
        Ok(())
    }

    /// Connect to a peer
    #[wasm_bindgen]
    pub async fn connect_peer(&mut self, address: &str) -> Result<String, P2PError> {
        if let Some(ref mut network) = self.network {
            let peer_id = network.connect_peer(address).await?;
            let info = PeerInfo::new(&peer_id, address);
            self.peers.insert(peer_id.clone(), info);
            log(&format!("Connected to peer: {}", peer_id));
            Ok(peer_id)
        } else {
            Err(P2PError::new("Network not initialized", "NETWORK_NOT_INITIALIZED"))
        }
    }

    /// Disconnect from a peer
    #[wasm_bindgen]
    pub async fn disconnect_peer(&mut self, peer_id: &str) -> Result<(), P2PError> {
        if let Some(ref mut network) = self.network {
            network.disconnect_peer(peer_id).await?;
            self.peers.remove(peer_id);
            log(&format!("Disconnected from peer: {}", peer_id));
        }
        Ok(())
    }

    /// Get connected peers
    #[wasm_bindgen]
    pub fn get_peers(&self) -> Array {
        let array = Array::new();
        for (peer_id, info) in &self.peers {
            let peer_obj = Object::new();
            Reflect::set(&peer_obj, &"id".into(), &peer_id.clone().into()).unwrap();
            Reflect::set(&peer_obj, &"address".into(), &info.address().into()).unwrap();
            Reflect::set(&peer_obj, &"connected_at".into(), &info.connected_at().into()).unwrap();
            array.push(&peer_obj);
        }
        array
    }

    /// Send message to peer
    #[wasm_bindgen]
    pub async fn send_message(&self, peer_id: &str, message: &str) -> Result<(), P2PError> {
        if let Some(ref network) = self.network {
            network.send_message(peer_id, message).await?;
            log(&format!("Message sent to peer: {}", peer_id));
        } else {
            return Err(P2PError::new("Network not initialized", "NETWORK_NOT_INITIALIZED"));
        }
        Ok(())
    }

    /// Broadcast message to all peers
    #[wasm_bindgen]
    pub async fn broadcast_message(&self, message: &str) -> Result<(), P2PError> {
        if let Some(ref network) = self.network {
            network.broadcast_message(message).await?;
            log("Message broadcasted to all peers");
        } else {
            return Err(P2PError::new("Network not initialized", "NETWORK_NOT_INITIALIZED"));
        }
        Ok(())
    }

    /// Get network status
    #[wasm_bindgen]
    pub fn get_status(&self) -> Object {
        let status = Object::new();
        Reflect::set(&status, &"peer_count".into(), &(self.peers.len() as u32).into()).unwrap();
        Reflect::set(&status, &"network_id".into(), &self.config.network_id.clone().into()).unwrap();
        Reflect::set(&status, &"port".into(), &self.config.port.into()).unwrap();
        
        if let Some(ref network) = self.network {
            Reflect::set(&status, &"is_running".into(), &network.is_running().into()).unwrap();
        } else {
            Reflect::set(&status, &"is_running".into(), &false.into()).unwrap();
        }
        
        status
    }

    /// Create DAG message
    #[wasm_bindgen]
    pub async fn create_dag_message(&mut self, content: &str) -> Result<String, P2PError> {
        let message_id = self.dag.create_message(content).await?;
        log(&format!("Created DAG message: {}", message_id));
        Ok(message_id)
    }

    /// Get DAG status
    #[wasm_bindgen]
    pub fn get_dag_status(&self) -> Object {
        self.dag.get_status()
    }

    /// Generate quantum-resistant keys
    #[wasm_bindgen]
    pub fn generate_keys(&mut self) -> Result<Object, P2PError> {
        self.crypto.generate_keys()
    }

    /// Sign data with quantum-resistant signature
    #[wasm_bindgen]
    pub fn sign_data(&self, data: &[u8]) -> Result<Uint8Array, P2PError> {
        self.crypto.sign_data(data)
    }

    /// Verify quantum-resistant signature
    #[wasm_bindgen]
    pub fn verify_signature(&self, data: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, P2PError> {
        self.crypto.verify_signature(data, signature, public_key)
    }

    /// Encrypt data with quantum-resistant encryption
    #[wasm_bindgen]
    pub fn encrypt_data(&self, data: &[u8], public_key: &[u8]) -> Result<Uint8Array, P2PError> {
        self.crypto.encrypt_data(data, public_key)
    }

    /// Decrypt data with quantum-resistant encryption
    #[wasm_bindgen]
    pub fn decrypt_data(&self, encrypted_data: &[u8], private_key: &[u8]) -> Result<Uint8Array, P2PError> {
        self.crypto.decrypt_data(encrypted_data, private_key)
    }
}