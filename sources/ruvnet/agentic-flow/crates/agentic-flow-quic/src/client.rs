//! QUIC client implementation with connection pooling

use crate::{
    error::QuicError,
    types::{ConnectionConfig, ConnectionMeta, PoolStats, QuicMessage},
    Result,
};
use quinn::{ClientConfig, Connection, Endpoint};
use rustls::pki_types::{CertificateDer, ServerName};
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::Arc,
};
use tokio::sync::RwLock;

/// QUIC client with connection pooling
#[derive(Clone)]
pub struct QuicClient {
    endpoint: Endpoint,
    config: Arc<ConnectionConfig>,
    pool: Arc<RwLock<ConnectionPool>>,
}

/// Connection pool manager
struct ConnectionPool {
    connections: HashMap<String, PooledConnection>,
    stats: PoolStats,
}

/// Pooled connection with metadata
#[allow(dead_code)]
struct PooledConnection {
    connection: Connection,
    meta: ConnectionMeta,
    last_used: std::time::Instant,
}

impl QuicClient {
    /// Create new QUIC client
    pub async fn new(config: ConnectionConfig) -> Result<Self> {
        let client_config = configure_client()?;

        let mut endpoint = Endpoint::client("0.0.0.0:0".parse().unwrap())
            .map_err(|e| QuicError::Connection(e.to_string()))?;

        endpoint.set_default_client_config(client_config);

        Ok(Self {
            endpoint,
            config: Arc::new(config),
            pool: Arc::new(RwLock::new(ConnectionPool {
                connections: HashMap::new(),
                stats: PoolStats::default(),
            })),
        })
    }

    /// Connect to server with connection pooling
    pub async fn connect(&self, addr: SocketAddr) -> Result<Connection> {
        let key = addr.to_string();

        // Check pool first
        {
            let pool = self.pool.read().await;
            if let Some(pooled) = pool.connections.get(&key) {
                if !pooled.connection.close_reason().is_some() {
                    tracing::debug!("Reusing pooled connection to {}", addr);
                    return Ok(pooled.connection.clone());
                }
            }
        }

        // Create new connection
        tracing::debug!("Creating new connection to {}", addr);
        let _server_name = ServerName::try_from(self.config.server_name.as_str())
            .map_err(|e| QuicError::Tls(e.to_string()))?;

        let connection = self.endpoint
            .connect(addr, &self.config.server_name)
            .map_err(|e| QuicError::Connection(e.to_string()))?
            .await
            .map_err(|e| QuicError::Connection(e.to_string()))?;

        // Add to pool
        let meta = ConnectionMeta {
            remote_addr: addr,
            connection_id: 0, // TODO: Get actual connection ID
            rtt_us: None,
            bytes_sent: 0,
            bytes_received: 0,
        };

        let pooled = PooledConnection {
            connection: connection.clone(),
            meta,
            last_used: std::time::Instant::now(),
        };

        {
            let mut pool = self.pool.write().await;
            pool.connections.insert(key, pooled);
            pool.stats.active += 1;
            pool.stats.total_created += 1;
        }

        Ok(connection)
    }

    /// Send message over QUIC
    pub async fn send_message(
        &self,
        addr: SocketAddr,
        message: QuicMessage,
    ) -> Result<()> {
        let connection = self.connect(addr).await?;
        let (mut send, _recv) = connection
            .open_bi()
            .await
            .map_err(|e| QuicError::Stream(e.to_string()))?;

        let data = serde_json::to_vec(&message)?;
        send.write_all(&data)
            .await
            .map_err(|e| QuicError::Stream(e.to_string()))?;

        send.finish()
            .map_err(|e| QuicError::Stream(e.to_string()))?;

        Ok(())
    }

    /// Receive message from QUIC
    pub async fn recv_message(&self, addr: SocketAddr) -> Result<QuicMessage> {
        let connection = self.connect(addr).await?;
        let (_send, mut recv) = connection
            .open_bi()
            .await
            .map_err(|e| QuicError::Stream(e.to_string()))?;

        let data = recv
            .read_to_end(1024 * 1024) // 1MB max
            .await
            .map_err(|e| QuicError::Stream(e.to_string()))?;

        let message: QuicMessage = serde_json::from_slice(&data)?;
        Ok(message)
    }

    /// Get pool statistics
    pub async fn pool_stats(&self) -> PoolStats {
        let pool = self.pool.read().await;
        pool.stats.clone()
    }

    /// Close all connections
    pub async fn close(&self) {
        let mut pool = self.pool.write().await;
        for (_, pooled) in pool.connections.drain() {
            pooled.connection.close(0u32.into(), b"client shutdown");
        }
        pool.stats.active = 0;
    }
}

/// Configure QUIC client with TLS
fn configure_client() -> Result<ClientConfig> {
    let crypto = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(SkipServerVerification::new())
        .with_no_client_auth();

    Ok(ClientConfig::new(Arc::new(
        quinn::crypto::rustls::QuicClientConfig::try_from(crypto)
            .map_err(|e| QuicError::Tls(e.to_string()))?
    )))
}

/// Skip server certificate verification (for testing)
#[derive(Debug)]
#[allow(dead_code)]
struct SkipServerVerification(Arc<rustls::crypto::CryptoProvider>);

impl SkipServerVerification {
    fn new() -> Arc<Self> {
        Arc::new(Self(Arc::new(rustls::crypto::ring::default_provider())))
    }
}

impl rustls::client::danger::ServerCertVerifier for SkipServerVerification {
    fn verify_server_cert(
        &self,
        _end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ED25519,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        // Initialize crypto provider for tests
        let _ = rustls::crypto::ring::default_provider().install_default();

        let config = ConnectionConfig::default();
        let client = QuicClient::new(config).await;
        assert!(client.is_ok());
    }
}
