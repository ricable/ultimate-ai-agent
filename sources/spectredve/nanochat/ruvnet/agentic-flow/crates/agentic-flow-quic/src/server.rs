//! QUIC server implementation with stream multiplexing

use crate::{
    error::QuicError,
    types::{ConnectionConfig, QuicMessage},
    Result,
};
use quinn::{Connection, Endpoint, RecvStream, SendStream, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::mpsc;

/// QUIC server with stream multiplexing
pub struct QuicServer {
    endpoint: Endpoint,
    config: Arc<ConnectionConfig>,
    message_tx: mpsc::UnboundedSender<(SocketAddr, QuicMessage)>,
}

impl QuicServer {
    /// Create new QUIC server
    pub async fn new(
        addr: SocketAddr,
        config: ConnectionConfig,
    ) -> Result<(Self, mpsc::UnboundedReceiver<(SocketAddr, QuicMessage)>)> {
        let server_config = configure_server()?;

        let endpoint = Endpoint::server(server_config, addr)
            .map_err(|e| QuicError::Connection(e.to_string()))?;

        let (message_tx, message_rx) = mpsc::unbounded_channel();

        Ok((
            Self {
                endpoint,
                config: Arc::new(config),
                message_tx,
            },
            message_rx,
        ))
    }

    /// Start accepting connections
    pub async fn run(&self) -> Result<()> {
        tracing::info!("QUIC server listening on {}", self.endpoint.local_addr()?);

        loop {
            let conn = match self.endpoint.accept().await {
                Some(conn) => conn,
                None => {
                    tracing::warn!("Endpoint closed, stopping server");
                    break;
                }
            };

            let connection = conn.await
                .map_err(|e| QuicError::Connection(e.to_string()))?;

            let remote_addr = connection.remote_address();
            tracing::debug!("Accepted connection from {}", remote_addr);

            let message_tx = self.message_tx.clone();
            let config = self.config.clone();

            tokio::spawn(async move {
                if let Err(e) = handle_connection(connection, remote_addr, message_tx, config).await {
                    tracing::error!("Connection error: {}", e);
                }
            });
        }

        Ok(())
    }

    /// Get local socket address
    pub fn local_addr(&self) -> Result<SocketAddr> {
        self.endpoint.local_addr()
            .map_err(|e| QuicError::Connection(e.to_string()))
    }

    /// Close server
    pub fn close(&self) {
        self.endpoint.close(0u32.into(), b"server shutdown");
    }
}

/// Handle incoming connection with stream multiplexing
async fn handle_connection(
    connection: Connection,
    remote_addr: SocketAddr,
    message_tx: mpsc::UnboundedSender<(SocketAddr, QuicMessage)>,
    config: Arc<ConnectionConfig>,
) -> Result<()> {
    let mut stream_count = 0u32;

    loop {
        let stream = match connection.accept_bi().await {
            Ok(s) => s,
            Err(quinn::ConnectionError::ApplicationClosed(_)) => {
                tracing::debug!("Connection closed by peer");
                break;
            }
            Err(e) => {
                return Err(QuicError::Stream(e.to_string()));
            }
        };

        stream_count += 1;

        if stream_count > config.max_concurrent_streams {
            tracing::warn!("Max concurrent streams reached, rejecting new stream");
            continue;
        }

        let message_tx = message_tx.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_stream(stream, remote_addr, message_tx).await {
                tracing::error!("Stream error: {}", e);
            }
        });
    }

    Ok(())
}

/// Handle bidirectional stream
async fn handle_stream(
    (mut send, mut recv): (SendStream, RecvStream),
    remote_addr: SocketAddr,
    message_tx: mpsc::UnboundedSender<(SocketAddr, QuicMessage)>,
) -> Result<()> {
    let data = recv
        .read_to_end(1024 * 1024) // 1MB max
        .await
        .map_err(|e| QuicError::Stream(e.to_string()))?;

    let message: QuicMessage = serde_json::from_slice(&data)?;

    tracing::debug!(
        "Received message {} of type {:?} from {}",
        message.id,
        message.msg_type,
        remote_addr
    );

    // Forward message to application
    message_tx
        .send((remote_addr, message.clone()))
        .map_err(|_| QuicError::Stream("Failed to forward message".to_string()))?;

    // Send acknowledgment
    let ack = serde_json::json!({
        "status": "received",
        "message_id": message.id,
    });

    let ack_data = serde_json::to_vec(&ack)?;
    send.write_all(&ack_data)
        .await
        .map_err(|e| QuicError::Stream(e.to_string()))?;

    send.finish()
        .map_err(|e| QuicError::Stream(e.to_string()))?;

    Ok(())
}

/// Configure QUIC server with TLS
fn configure_server() -> Result<ServerConfig> {
    let cert = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .map_err(|e| QuicError::Tls(e.to_string()))?;

    let cert_der = CertificateDer::from(cert.cert);
    let key_der = PrivateKeyDer::try_from(cert.key_pair.serialize_der())
        .map_err(|_| QuicError::Tls("Failed to serialize private key".to_string()))?;

    let mut server_crypto = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)
        .map_err(|e| QuicError::Tls(e.to_string()))?;

    server_crypto.max_early_data_size = 1024 * 1024; // 1MB for 0-RTT

    let mut server_config = ServerConfig::with_crypto(Arc::new(
        quinn::crypto::rustls::QuicServerConfig::try_from(server_crypto)
            .map_err(|e| QuicError::Tls(e.to_string()))?
    ));

    let transport_config = Arc::get_mut(&mut server_config.transport)
        .ok_or_else(|| QuicError::ConfigError("Failed to get mutable transport config".to_string()))?;

    transport_config.max_concurrent_bidi_streams(100_u32.into());
    transport_config.max_concurrent_uni_streams(100_u32.into());

    Ok(server_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_server_creation() {
        // Initialize crypto provider for tests
        let _ = rustls::crypto::ring::default_provider().install_default();

        let addr = "127.0.0.1:0".parse().unwrap();
        let config = ConnectionConfig::default();
        let result = QuicServer::new(addr, config).await;
        assert!(result.is_ok());
    }
}
