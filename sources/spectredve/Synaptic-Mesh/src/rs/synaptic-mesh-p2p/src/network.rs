//! P2P networking module with WebSocket support for WASM
//! 
//! Provides WebSocket-based networking for browser environments

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use js_sys::{Object, Reflect, Array, Promise};
use web_sys::{WebSocket, MessageEvent, CloseEvent, ErrorEvent};
use std::collections::HashMap;
use futures::future::LocalBoxFuture;
use crate::{P2PConfig, P2PError};

/// Network manager for P2P connections
#[wasm_bindgen]
pub struct Network {
    config: P2PConfig,
    connections: HashMap<String, WebSocketConnection>,
    is_running: bool,
    message_handlers: Vec<js_sys::Function>,
}

/// WebSocket connection wrapper
pub struct WebSocketConnection {
    pub socket: WebSocket,
    pub peer_id: String,
    pub address: String,
    pub connected_at: f64,
}

#[wasm_bindgen]
impl Network {
    #[wasm_bindgen(constructor)]
    pub fn new(config: &P2PConfig) -> Result<Network, P2PError> {
        Ok(Network {
            config: config.clone(),
            connections: HashMap::new(),
            is_running: false,
            message_handlers: Vec::new(),
        })
    }

    /// Start the network
    #[wasm_bindgen]
    pub async fn start(&mut self) -> Result<(), P2PError> {
        if self.is_running {
            return Ok(());
        }

        crate::log("Starting P2P network...");
        
        // Connect to bootstrap peers
        for peer_address in &self.config.bootstrap_peers.clone() {
            match self.connect_peer(peer_address).await {
                Ok(peer_id) => {
                    crate::log(&format!("Connected to bootstrap peer: {}", peer_id));
                },
                Err(e) => {
                    crate::log(&format!("Failed to connect to bootstrap peer {}: {}", peer_address, e.message));
                }
            }
        }

        self.is_running = true;
        crate::log("P2P network started successfully");
        Ok(())
    }

    /// Stop the network
    #[wasm_bindgen]
    pub async fn stop(&mut self) -> Result<(), P2PError> {
        if !self.is_running {
            return Ok(());
        }

        crate::log("Stopping P2P network...");

        // Close all connections
        for (_, connection) in self.connections.drain() {
            let _ = connection.socket.close();
        }

        self.is_running = false;
        crate::log("P2P network stopped");
        Ok(())
    }

    /// Check if network is running
    #[wasm_bindgen]
    pub fn is_running(&self) -> bool {
        self.is_running
    }

    /// Connect to a peer via WebSocket
    #[wasm_bindgen]
    pub async fn connect_peer(&mut self, address: &str) -> Result<String, P2PError> {
        // Create WebSocket URL
        let ws_url = if address.starts_with("ws://") || address.starts_with("wss://") {
            address.to_string()
        } else {
            format!("ws://{}", address)
        };

        // Create WebSocket connection
        let socket = WebSocket::new(&ws_url)
            .map_err(|_| P2PError::new("Failed to create WebSocket", "WEBSOCKET_ERROR"))?;

        // Generate peer ID
        let peer_id = format!("peer_{}", uuid::Uuid::new_v4().simple());

        // Set up event handlers
        self.setup_websocket_handlers(&socket, &peer_id)?;

        // Wait for connection to open
        let connection_promise = self.wait_for_connection(&socket).await?;
        
        // Store connection
        let connection = WebSocketConnection {
            socket,
            peer_id: peer_id.clone(),
            address: address.to_string(),
            connected_at: js_sys::Date::now(),
        };
        
        self.connections.insert(peer_id.clone(), connection);

        crate::log(&format!("Connected to peer {} at {}", peer_id, address));
        Ok(peer_id)
    }

    /// Disconnect from a peer
    #[wasm_bindgen]
    pub async fn disconnect_peer(&mut self, peer_id: &str) -> Result<(), P2PError> {
        if let Some(connection) = self.connections.remove(peer_id) {
            let _ = connection.socket.close();
            crate::log(&format!("Disconnected from peer: {}", peer_id));
        }
        Ok(())
    }

    /// Send message to a specific peer
    #[wasm_bindgen]
    pub async fn send_message(&self, peer_id: &str, message: &str) -> Result<(), P2PError> {
        if let Some(connection) = self.connections.get(peer_id) {
            connection.socket.send_with_str(message)
                .map_err(|_| P2PError::new("Failed to send message", "SEND_ERROR"))?;
        } else {
            return Err(P2PError::new("Peer not found", "PEER_NOT_FOUND"));
        }
        Ok(())
    }

    /// Broadcast message to all connected peers
    #[wasm_bindgen]
    pub async fn broadcast_message(&self, message: &str) -> Result<(), P2PError> {
        for (peer_id, connection) in &self.connections {
            match connection.socket.send_with_str(message) {
                Ok(_) => {},
                Err(_) => {
                    crate::log(&format!("Failed to send message to peer: {}", peer_id));
                }
            }
        }
        Ok(())
    }

    /// Get connected peer count
    #[wasm_bindgen]
    pub fn get_peer_count(&self) -> u32 {
        self.connections.len() as u32
    }

    /// Get connected peer IDs
    #[wasm_bindgen]
    pub fn get_peer_ids(&self) -> Array {
        let array = Array::new();
        for peer_id in self.connections.keys() {
            array.push(&JsValue::from_str(peer_id));
        }
        array
    }

    /// Add message handler
    #[wasm_bindgen]
    pub fn add_message_handler(&mut self, handler: js_sys::Function) {
        self.message_handlers.push(handler);
    }

    /// Get network statistics
    #[wasm_bindgen]
    pub fn get_stats(&self) -> Object {
        let stats = Object::new();
        Reflect::set(&stats, &"peer_count".into(), &(self.connections.len() as u32).into()).unwrap();
        Reflect::set(&stats, &"is_running".into(), &self.is_running.into()).unwrap();
        Reflect::set(&stats, &"network_id".into(), &self.config.network_id.clone().into()).unwrap();
        stats
    }

    // Helper methods

    fn setup_websocket_handlers(&self, socket: &WebSocket, peer_id: &str) -> Result<(), P2PError> {
        let peer_id_clone = peer_id.to_string();

        // OnOpen handler
        let onopen_closure = Closure::wrap(Box::new(move |_event| {
            crate::log(&format!("WebSocket connection opened for peer: {}", peer_id_clone));
        }) as Box<dyn FnMut(JsValue)>);
        socket.set_onopen(Some(onopen_closure.as_ref().unchecked_ref()));
        onopen_closure.forget();

        // OnMessage handler
        let peer_id_clone = peer_id.to_string();
        let onmessage_closure = Closure::wrap(Box::new(move |event: MessageEvent| {
            if let Ok(data) = event.data().dyn_into::<js_sys::JsString>() {
                let message = String::from(data);
                crate::log(&format!("Received message from {}: {}", peer_id_clone, message));
                // TODO: Call message handlers
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        socket.set_onmessage(Some(onmessage_closure.as_ref().unchecked_ref()));
        onmessage_closure.forget();

        // OnError handler
        let peer_id_clone = peer_id.to_string();
        let onerror_closure = Closure::wrap(Box::new(move |event: ErrorEvent| {
            crate::log(&format!("WebSocket error for peer {}: {:?}", peer_id_clone, event));
        }) as Box<dyn FnMut(ErrorEvent)>);
        socket.set_onerror(Some(onerror_closure.as_ref().unchecked_ref()));
        onerror_closure.forget();

        // OnClose handler
        let peer_id_clone = peer_id.to_string();
        let onclose_closure = Closure::wrap(Box::new(move |event: CloseEvent| {
            crate::log(&format!("WebSocket closed for peer {}: code={}, reason={}", 
                peer_id_clone, event.code(), event.reason()));
        }) as Box<dyn FnMut(CloseEvent)>);
        socket.set_onclose(Some(onclose_closure.as_ref().unchecked_ref()));
        onclose_closure.forget();

        Ok(())
    }

    async fn wait_for_connection(&self, socket: &WebSocket) -> Result<(), P2PError> {
        // In a real implementation, we would wait for the WebSocket to open
        // For now, we'll just simulate a delay
        use wasm_bindgen_futures::JsFuture;
        
        let promise = Promise::new(&mut |resolve, _reject| {
            let timeout = web_sys::window()
                .unwrap()
                .set_timeout_with_callback_and_timeout_and_arguments_0(
                    &resolve,
                    1000, // 1 second timeout
                ).unwrap();
        });

        JsFuture::from(promise).await
            .map_err(|_| P2PError::new("Connection timeout", "TIMEOUT"))?;

        Ok(())
    }
}

/// Network message types
#[derive(Clone, Debug)]
#[wasm_bindgen]
pub enum MessageType {
    Ping,
    Pong,
    Data,
    DAGMessage,
    PeerDiscovery,
    Handshake,
}

/// Network message structure
#[wasm_bindgen]
pub struct NetworkMessage {
    message_type: MessageType,
    sender_id: String,
    recipient_id: Option<String>,
    payload: Vec<u8>,
    timestamp: f64,
}

#[wasm_bindgen]
impl NetworkMessage {
    #[wasm_bindgen(constructor)]
    pub fn new(message_type: MessageType, sender_id: String, payload: Vec<u8>) -> NetworkMessage {
        NetworkMessage {
            message_type,
            sender_id,
            recipient_id: None,
            payload,
            timestamp: js_sys::Date::now(),
        }
    }

    #[wasm_bindgen(getter)]
    pub fn sender_id(&self) -> String {
        self.sender_id.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn timestamp(&self) -> f64 {
        self.timestamp
    }

    #[wasm_bindgen]
    pub fn set_recipient(&mut self, recipient_id: String) {
        self.recipient_id = Some(recipient_id);
    }

    #[wasm_bindgen]
    pub fn get_payload(&self) -> js_sys::Uint8Array {
        let array = js_sys::Uint8Array::new_with_length(self.payload.len() as u32);
        array.copy_from(&self.payload);
        array
    }
}