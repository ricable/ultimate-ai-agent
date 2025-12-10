//! Integration tests for QUIC client-server communication

#![cfg(not(target_family = "wasm"))]

use agentic_flow_quic::{
    client::QuicClient, server::QuicServer, types::*,
};
use bytes::Bytes;
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{timeout, Duration};

/// Test basic echo: client sends, server echoes, client receives
#[tokio::test]
async fn test_echo_basic() -> Result<(), Box<dyn std::error::Error>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    // Start server on localhost
    let server_addr: SocketAddr = "127.0.0.1:14433".parse()?;
    let config = ConnectionConfig::default();

    let (server, mut rx) = QuicServer::new(server_addr, config.clone()).await?;

    // Spawn server task
    let server_handle = tokio::spawn(async move {
        server.run().await
    });

    // Give server time to start
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Create client
    let client = QuicClient::new(config).await?;

    // Send message
    let message = QuicMessage {
        id: "echo-test-1".to_string(),
        msg_type: MessageType::Task,
        payload: Bytes::from("Hello QUIC!"),
        metadata: None,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64,
    };

    client.send_message(server_addr, message.clone()).await?;

    // Server receives and echoes
    let (_client_addr, received) = timeout(Duration::from_secs(1), rx.recv())
        .await?
        .ok_or("No message received")?;

    assert_eq!(received.id, "echo-test-1");
    assert_eq!(received.payload, Bytes::from("Hello QUIC!"));

    // In real implementation, server would send response via connection
    // For now, just verify message was received correctly

    // Cleanup
    drop(server_handle);
    client.close().await;

    Ok(())
}

/// Test concurrent stream handling
#[tokio::test]
async fn test_concurrent_streams() -> Result<(), Box<dyn std::error::Error>> {
    let server_addr: SocketAddr = "127.0.0.1:14434".parse()?;
    let config = ConnectionConfig::default();

    let (server, mut rx) = QuicServer::new(server_addr, config.clone()).await?;

    let server_handle = tokio::spawn(async move {
        server.run().await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = QuicClient::new(config).await?;

    // Send 10 messages concurrently
    let handles: Vec<_> = (0..10)
        .map(|i| {
            let client_clone = client.clone();
            let addr = server_addr;
            tokio::spawn(async move {
                let message = QuicMessage {
                    id: format!("concurrent-{}", i),
                    msg_type: MessageType::Task,
                    payload: Bytes::from(format!("Message {}", i)),
                    metadata: None,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                client_clone.send_message(addr, message).await
            })
        })
        .collect();

    // Wait for all sends to complete
    for handle in handles {
        handle.await??;
    }

    // Verify messages received
    let mut received_count = 0;
    while let Ok(Some(_)) = timeout(Duration::from_millis(100), rx.recv()).await {
        received_count += 1;
        if received_count >= 10 {
            break;
        }
    }

    assert_eq!(received_count, 10, "Should receive all 10 messages");

    // Cleanup
    drop(server_handle);
    client.close().await;

    Ok(())
}

/// Test connection pool reuse
#[tokio::test]
async fn test_connection_pooling() -> Result<(), Box<dyn std::error::Error>> {
    let server_addr: SocketAddr = "127.0.0.1:14435".parse()?;
    let config = ConnectionConfig::default();

    let (server, _rx) = QuicServer::new(server_addr, config.clone()).await?;

    let server_handle = tokio::spawn(async move {
        server.run().await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = QuicClient::new(config).await?;

    // First message creates connection
    let message1 = QuicMessage {
        id: "pool-test-1".to_string(),
        msg_type: MessageType::Task,
        payload: Bytes::from("First"),
        metadata: None,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64,
    };

    client.send_message(server_addr, message1).await?;

    let stats1 = client.pool_stats().await;
    assert_eq!(stats1.total_created, 1, "Should have 1 connection created");

    // Second message reuses connection
    let message2 = QuicMessage {
        id: "pool-test-2".to_string(),
        msg_type: MessageType::Task,
        payload: Bytes::from("Second"),
        metadata: None,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64,
    };

    client.send_message(server_addr, message2).await?;

    let stats2 = client.pool_stats().await;
    assert_eq!(stats2.total_created, 1, "Should still have 1 connection (reused)");
    assert_eq!(stats2.active, 1, "Connection should be active");

    // Cleanup
    drop(server_handle);
    client.close().await;

    Ok(())
}

/// Test error handling for invalid address
#[tokio::test]
async fn test_invalid_address() -> Result<(), Box<dyn std::error::Error>> {
    let config = ConnectionConfig::default();
    let client = QuicClient::new(config).await?;

    let message = QuicMessage {
        id: "error-test-1".to_string(),
        msg_type: MessageType::Task,
        payload: Bytes::from("Test"),
        metadata: None,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64,
    };

    // Try to send to non-existent server
    let invalid_addr: SocketAddr = "127.0.0.1:19999".parse()?;
    let result = timeout(
        Duration::from_millis(500),
        client.send_message(invalid_addr, message)
    ).await;

    // Should timeout or error
    assert!(result.is_err() || result.unwrap().is_err());

    client.close().await;

    Ok(())
}

/// Test heartbeat message type
#[tokio::test]
async fn test_heartbeat_messages() -> Result<(), Box<dyn std::error::Error>> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let server_addr: SocketAddr = "127.0.0.1:14436".parse()?;
    let config = ConnectionConfig::default();

    let (server, mut rx) = QuicServer::new(server_addr, config.clone()).await?;

    let server_handle = tokio::spawn(async move {
        server.run().await
    });

    tokio::time::sleep(Duration::from_millis(100)).await;

    let client = QuicClient::new(config).await?;

    // Send heartbeat
    let heartbeat = QuicMessage {
        id: "heartbeat-1".to_string(),
        msg_type: MessageType::Heartbeat,
        payload: Bytes::new(),
        metadata: None,
        timestamp: SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis() as u64,
    };

    client.send_message(server_addr, heartbeat).await?;

    // Verify heartbeat received
    let (_addr, received) = timeout(Duration::from_secs(1), rx.recv())
        .await?
        .ok_or("No heartbeat received")?;

    assert_eq!(received.msg_type, MessageType::Heartbeat);

    // Cleanup
    drop(server_handle);
    client.close().await;

    Ok(())
}
