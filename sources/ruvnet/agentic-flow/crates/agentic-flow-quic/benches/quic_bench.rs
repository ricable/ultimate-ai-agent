//! Performance benchmarks for QUIC transport

use agentic_flow_quic::{
    client::QuicClient, server::QuicServer, types::*,
};
use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::runtime::Runtime;

/// Benchmark connection establishment
fn bench_connection_establishment(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let server_addr: SocketAddr = "127.0.0.1:24433".parse().unwrap();

    c.bench_function("connection_establishment", |b| {
        b.to_async(&rt).iter(|| async {
            let config = ConnectionConfig::default();
            let client = QuicClient::new(config).await.unwrap();
            black_box(client);
        });
    });
}

/// Benchmark single message send
fn bench_message_send(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let server_addr: SocketAddr = "127.0.0.1:24434".parse().unwrap();
    let config = ConnectionConfig::default();

    // Setup server
    let (server, _rx) = rt.block_on(async {
        QuicServer::new(server_addr, config.clone()).await.unwrap()
    });

    rt.spawn(async move {
        let _ = server.run().await;
    });

    // Give server time to start
    std::thread::sleep(std::time::Duration::from_millis(100));

    let client = rt.block_on(async {
        QuicClient::new(config).await.unwrap()
    });

    c.bench_function("message_send", |b| {
        b.to_async(&rt).iter(|| async {
            let message = QuicMessage {
                id: "bench-1".to_string(),
                msg_type: MessageType::Task,
                payload: Bytes::from("benchmark"),
                metadata: None,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            };

            black_box(client.send_message(server_addr, message).await.unwrap());
        });
    });
}

/// Benchmark concurrent agent spawning (1, 10, 100 agents)
fn bench_concurrent_agents(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let server_addr: SocketAddr = "127.0.0.1:24435".parse().unwrap();
    let config = ConnectionConfig::default();

    // Setup server
    let (server, _rx) = rt.block_on(async {
        QuicServer::new(server_addr, config.clone()).await.unwrap()
    });

    rt.spawn(async move {
        let _ = server.run().await;
    });

    std::thread::sleep(std::time::Duration::from_millis(100));

    let mut group = c.benchmark_group("concurrent_agents");

    for agent_count in [1, 10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(agent_count),
            agent_count,
            |b, &count| {
                b.to_async(&rt).iter(|| async {
                    let client = QuicClient::new(config.clone()).await.unwrap();

                    let handles: Vec<_> = (0..count)
                        .map(|i| {
                            let client_clone = client.clone();
                            tokio::spawn(async move {
                                let message = QuicMessage {
                                    id: format!("agent-{}", i),
                                    msg_type: MessageType::Task,
                                    payload: Bytes::from(format!("spawn-agent-{}", i)),
                                    metadata: None,
                                    timestamp: SystemTime::now()
                                        .duration_since(UNIX_EPOCH)
                                        .unwrap()
                                        .as_millis() as u64,
                                };

                                client_clone.send_message(server_addr, message).await
                            })
                        })
                        .collect();

                    for handle in handles {
                        let _ = handle.await;
                    }

                    black_box(client);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark message throughput (messages per second)
fn bench_message_throughput(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let server_addr: SocketAddr = "127.0.0.1:24436".parse().unwrap();
    let config = ConnectionConfig::default();

    let (server, _rx) = rt.block_on(async {
        QuicServer::new(server_addr, config.clone()).await.unwrap()
    });

    rt.spawn(async move {
        let _ = server.run().await;
    });

    std::thread::sleep(std::time::Duration::from_millis(100));

    let client = rt.block_on(async {
        QuicClient::new(config).await.unwrap()
    });

    c.bench_function("message_throughput_1000", |b| {
        b.to_async(&rt).iter(|| async {
            for i in 0..1000 {
                let message = QuicMessage {
                    id: format!("throughput-{}", i),
                    msg_type: MessageType::Task,
                    payload: Bytes::from("data"),
                    metadata: None,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };

                let _ = client.send_message(server_addr, message).await;
            }

            black_box(());
        });
    });
}

/// Benchmark connection pool efficiency
fn bench_connection_pool(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let server_addr: SocketAddr = "127.0.0.1:24437".parse().unwrap();
    let config = ConnectionConfig::default();

    let (server, _rx) = rt.block_on(async {
        QuicServer::new(server_addr, config.clone()).await.unwrap()
    });

    rt.spawn(async move {
        let _ = server.run().await;
    });

    std::thread::sleep(std::time::Duration::from_millis(100));

    let client = rt.block_on(async {
        QuicClient::new(config.clone()).await.unwrap()
    });

    // Warmup: create connection
    rt.block_on(async {
        let message = QuicMessage {
            id: "warmup".to_string(),
            msg_type: MessageType::Task,
            payload: Bytes::from("warmup"),
            metadata: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };
        let _ = client.send_message(server_addr, message).await;
    });

    c.bench_function("connection_pool_reuse", |b| {
        b.to_async(&rt).iter(|| async {
            let message = QuicMessage {
                id: "pool-bench".to_string(),
                msg_type: MessageType::Task,
                payload: Bytes::from("reused"),
                metadata: None,
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64,
            };

            black_box(client.send_message(server_addr, message).await.unwrap());
        });
    });
}

criterion_group!(
    benches,
    bench_connection_establishment,
    bench_message_send,
    bench_concurrent_agents,
    bench_message_throughput,
    bench_connection_pool
);
criterion_main!(benches);
