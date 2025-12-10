//! WASM bindings for QUIC transport
//!
//! Note: This wraps the WASM stub since browsers don't support UDP/QUIC directly.
//! For production QUIC, use native Node.js builds.

#![cfg(feature = "wasm")]

use crate::{
    QuicClient,
    types::{ConnectionConfig, MessageType, QuicMessage},
};
use bytes::Bytes;
use wasm_bindgen::prelude::*;
use std::net::SocketAddr;

/// WASM wrapper for QuicClient
#[wasm_bindgen]
pub struct WasmQuicClient {
    client: QuicClient,
}

#[wasm_bindgen]
impl WasmQuicClient {
    /// Create new WASM QUIC client
    #[wasm_bindgen(constructor)]
    pub async fn new(config: JsValue) -> Result<WasmQuicClient, JsValue> {
        console_error_panic_hook::set_once();

        let config: ConnectionConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {}", e)))?;

        let client = QuicClient::new(config)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        Ok(Self { client })
    }

    /// Send message to server
    #[wasm_bindgen(js_name = sendMessage)]
    pub async fn send_message(
        &self,
        addr: String,
        message: JsValue,
    ) -> Result<(), JsValue> {
        let socket_addr: SocketAddr = addr.parse()
            .map_err(|e| JsValue::from_str(&format!("Invalid address: {}", e)))?;

        let message: QuicMessage = serde_wasm_bindgen::from_value(message)
            .map_err(|e| JsValue::from_str(&format!("Invalid message: {}", e)))?;

        self.client
            .send_message(socket_addr, message)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Receive message from server
    #[wasm_bindgen(js_name = recvMessage)]
    pub async fn recv_message(&self, addr: String) -> Result<JsValue, JsValue> {
        let socket_addr: SocketAddr = addr.parse()
            .map_err(|e| JsValue::from_str(&format!("Invalid address: {}", e)))?;

        let message = self.client
            .recv_message(socket_addr)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;

        serde_wasm_bindgen::to_value(&message)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Get pool statistics
    #[wasm_bindgen(js_name = poolStats)]
    pub async fn pool_stats(&self) -> Result<JsValue, JsValue> {
        let stats = self.client.pool_stats().await;
        serde_wasm_bindgen::to_value(&stats)
            .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
    }

    /// Close all connections
    #[wasm_bindgen]
    pub async fn close(&self) {
        self.client.close().await;
    }
}

/// Create QUIC message from JavaScript
#[wasm_bindgen(js_name = createQuicMessage)]
pub fn create_quic_message(
    id: String,
    msg_type: String,
    payload: Vec<u8>,
    metadata: JsValue,
) -> Result<JsValue, JsValue> {
    let msg_type = match msg_type.as_str() {
        "task" => MessageType::Task,
        "result" => MessageType::Result,
        "status" => MessageType::Status,
        "coordination" => MessageType::Coordination,
        "heartbeat" => MessageType::Heartbeat,
        custom => MessageType::Custom(custom.to_string()),
    };

    let metadata = if metadata.is_null() || metadata.is_undefined() {
        None
    } else {
        Some(serde_wasm_bindgen::from_value(metadata)
            .map_err(|e| JsValue::from_str(&format!("Invalid metadata: {}", e)))?)
    };

    let message = QuicMessage {
        id,
        msg_type,
        payload: Bytes::from(payload),
        metadata,
        timestamp: js_sys::Date::now() as u64,
    };

    serde_wasm_bindgen::to_value(&message)
        .map_err(|e| JsValue::from_str(&format!("Serialization error: {}", e)))
}

/// Create default connection config
#[wasm_bindgen(js_name = defaultConfig)]
pub fn default_config() -> JsValue {
    let config = ConnectionConfig::default();
    serde_wasm_bindgen::to_value(&config).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    #[wasm_bindgen_test]
    fn test_default_config() {
        let config = default_config();
        assert!(!config.is_null());
    }
}
