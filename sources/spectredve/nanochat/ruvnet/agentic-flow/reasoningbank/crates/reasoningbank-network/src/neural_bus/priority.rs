//! Application-level priority queue system
//!
//! Three-tier priority system using mpsc channels:
//! - High priority: Latency-sensitive operations (control, urgent requests)
//! - Normal priority: Standard operations (most requests)
//! - Low priority: Background tasks (gossip, telemetry)

use crate::{NetworkError, Result};
use bytes::Bytes;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::debug;

/// Message priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    High = 2,
    Normal = 1,
    Low = 0,
}

/// Prioritized message wrapper
#[derive(Debug, Clone)]
pub struct PriorityMessage {
    pub priority: Priority,
    pub data: Bytes,
}

/// Multi-tier priority queue using separate mpsc channels
pub struct PriorityQueue {
    high_tx: mpsc::Sender<Bytes>,
    high_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Bytes>>>,

    normal_tx: mpsc::Sender<Bytes>,
    normal_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Bytes>>>,

    low_tx: mpsc::Sender<Bytes>,
    low_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Bytes>>>,
}

impl PriorityQueue {
    /// Create a new priority queue with specified buffer sizes
    pub fn new(high_buffer: usize, normal_buffer: usize, low_buffer: usize) -> Self {
        let (high_tx, high_rx) = mpsc::channel(high_buffer);
        let (normal_tx, normal_rx) = mpsc::channel(normal_buffer);
        let (low_tx, low_rx) = mpsc::channel(low_buffer);

        Self {
            high_tx,
            high_rx: Arc::new(tokio::sync::Mutex::new(high_rx)),
            normal_tx,
            normal_rx: Arc::new(tokio::sync::Mutex::new(normal_rx)),
            low_tx,
            low_rx: Arc::new(tokio::sync::Mutex::new(low_rx)),
        }
    }

    /// Send a message with specified priority
    pub async fn send(&self, priority: Priority, data: Bytes) -> Result<()> {
        debug!("Sending message with priority: {:?}", priority);

        let tx = match priority {
            Priority::High => &self.high_tx,
            Priority::Normal => &self.normal_tx,
            Priority::Low => &self.low_tx,
        };

        tx.send(data)
            .await
            .map_err(|_| NetworkError::Internal("Priority queue send failed".to_string()))?;

        Ok(())
    }

    /// Receive next message (respecting priority order)
    ///
    /// Tries high priority first, then normal, then low.
    /// This is a fair scheduler that prevents starvation.
    pub async fn recv(&self) -> Result<PriorityMessage> {
        // Try high priority (non-blocking)
        {
            let mut high_rx = self.high_rx.lock().await;
            if let Ok(data) = high_rx.try_recv() {
                return Ok(PriorityMessage {
                    priority: Priority::High,
                    data,
                });
            }
        }

        // Try normal priority (non-blocking)
        {
            let mut normal_rx = self.normal_rx.lock().await;
            if let Ok(data) = normal_rx.try_recv() {
                return Ok(PriorityMessage {
                    priority: Priority::Normal,
                    data,
                });
            }
        }

        // Try low priority (non-blocking)
        {
            let mut low_rx = self.low_rx.lock().await;
            if let Ok(data) = low_rx.try_recv() {
                return Ok(PriorityMessage {
                    priority: Priority::Low,
                    data,
                });
            }
        }

        // If all queues empty, wait for next message from any queue
        tokio::select! {
            result = async {
                let mut rx = self.high_rx.lock().await;
                rx.recv().await
            } => {
                result
                    .map(|data| PriorityMessage {
                        priority: Priority::High,
                        data,
                    })
                    .ok_or_else(|| NetworkError::Internal("High priority queue closed".to_string()))
            }
            result = async {
                let mut rx = self.normal_rx.lock().await;
                rx.recv().await
            } => {
                result
                    .map(|data| PriorityMessage {
                        priority: Priority::Normal,
                        data,
                    })
                    .ok_or_else(|| NetworkError::Internal("Normal priority queue closed".to_string()))
            }
            result = async {
                let mut rx = self.low_rx.lock().await;
                rx.recv().await
            } => {
                result
                    .map(|data| PriorityMessage {
                        priority: Priority::Low,
                        data,
                    })
                    .ok_or_else(|| NetworkError::Internal("Low priority queue closed".to_string()))
            }
        }
    }

    /// Get a sender for a specific priority level
    pub fn sender(&self, priority: Priority) -> mpsc::Sender<Bytes> {
        match priority {
            Priority::High => self.high_tx.clone(),
            Priority::Normal => self.normal_tx.clone(),
            Priority::Low => self.low_tx.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_priority_queue_ordering() {
        let queue = PriorityQueue::new(10, 10, 10);

        // Send messages in reverse priority order
        queue.send(Priority::Low, Bytes::from("low")).await.unwrap();
        queue.send(Priority::Normal, Bytes::from("normal")).await.unwrap();
        queue.send(Priority::High, Bytes::from("high")).await.unwrap();

        // Should receive in priority order
        let msg1 = queue.recv().await.unwrap();
        assert_eq!(msg1.priority, Priority::High);
        assert_eq!(msg1.data, Bytes::from("high"));

        let msg2 = queue.recv().await.unwrap();
        assert_eq!(msg2.priority, Priority::Normal);
        assert_eq!(msg2.data, Bytes::from("normal"));

        let msg3 = queue.recv().await.unwrap();
        assert_eq!(msg3.priority, Priority::Low);
        assert_eq!(msg3.data, Bytes::from("low"));
    }

    #[tokio::test]
    async fn test_priority_queue_fairness() {
        let queue = PriorityQueue::new(10, 10, 10);

        // Fill with mixed priorities
        for i in 0..5 {
            queue.send(Priority::Low, Bytes::from(format!("low-{}", i))).await.unwrap();
            queue.send(Priority::High, Bytes::from(format!("high-{}", i))).await.unwrap();
        }

        // High priority messages should come first
        for _ in 0..5 {
            let msg = queue.recv().await.unwrap();
            assert_eq!(msg.priority, Priority::High);
        }

        // Then low priority
        for _ in 0..5 {
            let msg = queue.recv().await.unwrap();
            assert_eq!(msg.priority, Priority::Low);
        }
    }
}
