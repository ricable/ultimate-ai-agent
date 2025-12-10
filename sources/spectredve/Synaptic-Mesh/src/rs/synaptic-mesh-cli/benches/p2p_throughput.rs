// P2P network throughput benchmarks

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// Mock P2P message
#[derive(Clone)]
struct P2PMessage {
    id: u64,
    sender: String,
    recipient: String,
    payload: Vec<u8>,
    timestamp: u64,
}

impl P2PMessage {
    fn new(id: u64, sender: String, recipient: String, payload: Vec<u8>) -> Self {
        Self {
            id,
            sender,
            recipient,
            payload,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        }
    }
    
    fn size(&self) -> usize {
        8 + // id
        self.sender.len() +
        self.recipient.len() +
        self.payload.len() +
        8 // timestamp
    }
}

// Mock P2P node
struct P2PNode {
    id: String,
    peers: Arc<Mutex<Vec<String>>>,
    message_buffer: Arc<Mutex<Vec<P2PMessage>>>,
    routing_table: Arc<Mutex<HashMap<String, Vec<String>>>>,
}

impl P2PNode {
    fn new(id: String) -> Self {
        Self {
            id,
            peers: Arc::new(Mutex::new(Vec::new())),
            message_buffer: Arc::new(Mutex::new(Vec::new())),
            routing_table: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    fn connect_peer(&self, peer_id: String) {
        self.peers.lock().unwrap().push(peer_id.clone());
        
        let mut routing = self.routing_table.lock().unwrap();
        routing.entry(peer_id).or_insert_with(Vec::new);
    }
    
    fn send_message(&self, recipient: String, payload: Vec<u8>) -> u64 {
        let message_id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;
        
        let message = P2PMessage::new(
            message_id,
            self.id.clone(),
            recipient,
            payload,
        );
        
        self.message_buffer.lock().unwrap().push(message);
        message_id
    }
    
    fn broadcast_message(&self, payload: Vec<u8>) -> Vec<u64> {
        let peers = self.peers.lock().unwrap().clone();
        let mut message_ids = Vec::new();
        
        for peer in peers {
            let id = self.send_message(peer, payload.clone());
            message_ids.push(id);
        }
        
        message_ids
    }
    
    fn process_messages(&self) -> usize {
        let mut buffer = self.message_buffer.lock().unwrap();
        let count = buffer.len();
        
        // Simulate message processing
        for message in buffer.iter() {
            // Simulate routing decision
            if message.recipient != self.id {
                // Forward message (simplified)
                let _routing = self.routing_table.lock().unwrap();
            }
        }
        
        buffer.clear();
        count
    }
    
    fn get_stats(&self) -> (usize, usize) {
        let peers_count = self.peers.lock().unwrap().len();
        let buffer_size = self.message_buffer.lock().unwrap().len();
        (peers_count, buffer_size)
    }
}

fn bench_message_sending(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2p_message_sending");
    
    let message_sizes = vec![
        (64, "64B"),
        (1024, "1KB"),
        (10240, "10KB"),
        (102400, "100KB"),
        (1048576, "1MB"),
    ];
    
    for (size, description) in message_sizes {
        let node = P2PNode::new("sender".to_string());
        node.connect_peer("receiver".to_string());
        
        let payload = vec![0u8; size];
        
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("send_message", description),
            &payload,
            |b, payload| {
                b.iter(|| {
                    black_box(node.send_message(
                        black_box("receiver".to_string()),
                        black_box(payload.clone()),
                    ));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_network_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2p_network_scaling");
    
    let network_sizes = vec![10, 50, 100, 500, 1000];
    
    for network_size in network_sizes {
        // Create network
        let nodes: Vec<P2PNode> = (0..network_size)
            .map(|i| P2PNode::new(format!("node-{}", i)))
            .collect();
        
        // Connect nodes in mesh topology
        for i in 0..network_size {
            for j in 0..network_size {
                if i != j {
                    nodes[i].connect_peer(format!("node-{}", j));
                }
            }
        }
        
        let payload = vec![42u8; 1024]; // 1KB message
        
        group.bench_function(
            BenchmarkId::new("broadcast", network_size),
            |b| {
                b.iter(|| {
                    black_box(nodes[0].broadcast_message(black_box(payload.clone())));
                });
            },
        );
    }
    
    group.finish();
}

fn bench_message_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2p_message_processing");
    
    let message_counts = vec![100, 1000, 10000, 100000];
    
    for count in message_counts {
        let node = P2PNode::new("processor".to_string());
        
        // Pre-fill message buffer
        for i in 0..count {
            let payload = vec![i as u8; 64];
            node.send_message(format!("dest-{}", i % 10), payload);
        }
        
        group.throughput(Throughput::Elements(count as u64));
        group.bench_function(
            BenchmarkId::new("process_messages", count),
            |b| {
                b.iter(|| {
                    black_box(node.process_messages());
                });
            },
        );
    }
    
    group.finish();
}

fn bench_routing_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2p_routing");
    
    let hop_counts = vec![1, 3, 5, 10, 20];
    
    for hops in hop_counts {
        // Create linear chain of nodes
        let nodes: Vec<P2PNode> = (0..hops + 1)
            .map(|i| P2PNode::new(format!("hop-{}", i)))
            .collect();
        
        // Connect nodes in chain
        for i in 0..hops {
            nodes[i].connect_peer(format!("hop-{}", i + 1));
        }
        
        let payload = vec![0u8; 1024];
        
        group.bench_function(
            BenchmarkId::new("multi_hop_routing", hops),
            |b| {
                b.iter(|| {
                    // Simulate routing through chain
                    for i in 0..hops {
                        black_box(nodes[i].send_message(
                            format!("hop-{}", i + 1),
                            black_box(payload.clone()),
                        ));
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_concurrent_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2p_concurrent");
    
    use std::thread;
    use std::sync::Arc;
    
    let thread_counts = vec![1, 2, 4, 8, 16];
    
    for thread_count in thread_counts {
        let node = Arc::new(P2PNode::new("concurrent-node".to_string()));
        
        // Add peers
        for i in 0..10 {
            node.connect_peer(format!("peer-{}", i));
        }
        
        group.bench_function(
            BenchmarkId::new("concurrent_sends", thread_count),
            |b| {
                b.iter(|| {
                    let handles: Vec<_> = (0..thread_count).map(|i| {
                        let node = Arc::clone(&node);
                        thread::spawn(move || {
                            let payload = vec![i as u8; 1024];
                            for j in 0..10 {
                                node.send_message(format!("peer-{}", j), payload.clone());
                            }
                        })
                    }).collect();
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

fn bench_gossip_propagation(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2p_gossip");
    
    let network_sizes = vec![50, 100, 200];
    let fanout_sizes = vec![3, 5, 10];
    
    for network_size in network_sizes {
        for fanout in &fanout_sizes {
            // Create gossip network
            let nodes: Vec<P2PNode> = (0..network_size)
                .map(|i| P2PNode::new(format!("gossip-{}", i)))
                .collect();
            
            // Connect each node to random subset (gossip topology)
            for i in 0..network_size {
                for j in 0..*fanout {
                    let peer_idx = (i + j + 1) % network_size;
                    nodes[i].connect_peer(format!("gossip-{}", peer_idx));
                }
            }
            
            let payload = vec![0u8; 512]; // 512B gossip message
            
            group.bench_function(
                BenchmarkId::new(
                    "gossip_round",
                    format!("n{}_f{}", network_size, fanout)
                ),
                |b| {
                    b.iter(|| {
                        // Simulate one gossip round
                        for node in &nodes {
                            black_box(node.broadcast_message(black_box(payload.clone())));
                        }
                    });
                },
            );
        }
    }
    
    group.finish();
}

fn bench_bandwidth_utilization(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2p_bandwidth");
    group.measurement_time(Duration::from_secs(10));
    
    let node = P2PNode::new("bandwidth-test".to_string());
    for i in 0..100 {
        node.connect_peer(format!("peer-{}", i));
    }
    
    let message_rates = vec![10, 100, 1000, 10000]; // messages per second
    
    for rate in message_rates {
        let payload = vec![0u8; 1024]; // 1KB per message
        let total_bytes = rate * 1024;
        
        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_function(
            BenchmarkId::new("sustained_throughput", format!("{}mps", rate)),
            |b| {
                b.iter(|| {
                    for _ in 0..rate {
                        black_box(node.send_message(
                            black_box("peer-0".to_string()),
                            black_box(payload.clone()),
                        ));
                    }
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    p2p_benches,
    bench_message_sending,
    bench_network_scaling,
    bench_message_processing,
    bench_routing_efficiency,
    bench_concurrent_operations,
    bench_gossip_propagation,
    bench_bandwidth_utilization
);
criterion_main!(p2p_benches);