//! Immutable Audit Logger
//!
//! Provides cryptographically signed, tamper-proof audit trails for all system operations.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;
use tracing::{debug, info};

#[derive(Error, Debug)]
pub enum AuditError {
    #[error("Failed to create audit entry: {0}")]
    CreationError(String),
    #[error("Failed to sign audit entry: {0}")]
    SigningError(String),
    #[error("Audit verification failed: {0}")]
    VerificationError(String),
    #[error("Storage error: {0}")]
    StorageError(String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEntry {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub event_type: String,
    pub actor: String,
    pub resource: String,
    pub action: String,
    pub outcome: AuditOutcome,
    pub metadata: serde_json::Value,
    pub signature: String,
    pub previous_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AuditOutcome {
    Success,
    Failure,
    Partial,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditChain {
    pub entries: Vec<AuditEntry>,
    pub chain_id: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Clone)]
pub struct AuditLogger {
    chain: Arc<RwLock<AuditChain>>,
    signing_key: Arc<String>,
}

impl AuditLogger {
    pub fn new(signing_key: String) -> Self {
        let chain = AuditChain {
            entries: Vec::new(),
            chain_id: uuid::Uuid::new_v4().to_string(),
            created_at: chrono::Utc::now(),
        };

        Self {
            chain: Arc::new(RwLock::new(chain)),
            signing_key: Arc::new(signing_key),
        }
    }

    /// Log an audit event
    pub async fn log_event(
        &self,
        event_type: &str,
        actor: &str,
        metadata: serde_json::Value,
    ) -> Result<String, AuditError> {
        let entry_id = uuid::Uuid::new_v4().to_string();

        debug!("Creating audit entry: {} for actor: {}", event_type, actor);

        let mut chain = self.chain.write().await;

        // Get previous hash from chain
        let previous_hash = chain
            .entries
            .last()
            .map(|e| e.signature.clone())
            .unwrap_or_else(|| "genesis".to_string());

        let entry = AuditEntry {
            id: entry_id.clone(),
            timestamp: chrono::Utc::now(),
            event_type: event_type.to_string(),
            actor: actor.to_string(),
            resource: metadata
                .get("resource")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            action: metadata
                .get("action")
                .and_then(|v| v.as_str())
                .unwrap_or(event_type)
                .to_string(),
            outcome: AuditOutcome::Success,
            metadata,
            signature: String::new(), // Will be set after signing
            previous_hash: previous_hash.clone(),
        };

        // Sign the entry
        let signature = self.sign_entry(&entry).await?;
        let mut signed_entry = entry;
        signed_entry.signature = signature;

        // Append to chain
        chain.entries.push(signed_entry.clone());

        info!(
            "Audit entry created: {} (type: {}, actor: {})",
            entry_id, event_type, actor
        );

        Ok(entry_id)
    }

    /// Sign an audit entry using cryptographic hash
    async fn sign_entry(&self, entry: &AuditEntry) -> Result<String, AuditError> {
        let mut hasher = Sha256::new();

        // Create deterministic representation
        let data = format!(
            "{}:{}:{}:{}:{}:{}:{}",
            entry.id,
            entry.timestamp.to_rfc3339(),
            entry.event_type,
            entry.actor,
            entry.resource,
            entry.action,
            entry.previous_hash
        );

        hasher.update(data.as_bytes());
        hasher.update(self.signing_key.as_bytes());

        let result = hasher.finalize();
        Ok(hex::encode(result))
    }

    /// Verify the integrity of the audit chain
    pub async fn verify_chain(&self) -> Result<bool, AuditError> {
        let chain = self.chain.read().await;

        if chain.entries.is_empty() {
            return Ok(true);
        }

        debug!("Verifying audit chain with {} entries", chain.entries.len());

        let mut previous_hash = "genesis".to_string();

        for entry in &chain.entries {
            // Verify hash chain
            if entry.previous_hash != previous_hash {
                return Err(AuditError::VerificationError(format!(
                    "Chain broken at entry {}: expected previous hash {}, got {}",
                    entry.id, previous_hash, entry.previous_hash
                )));
            }

            // Verify signature
            let expected_signature = self.sign_entry(entry).await?;
            if entry.signature != expected_signature {
                return Err(AuditError::VerificationError(format!(
                    "Invalid signature for entry {}",
                    entry.id
                )));
            }

            previous_hash = entry.signature.clone();
        }

        info!("Audit chain verified successfully ({} entries)", chain.entries.len());
        Ok(true)
    }

    /// Get audit entries for a specific actor
    pub async fn get_actor_history(&self, actor: &str) -> Vec<AuditEntry> {
        self.chain
            .read()
            .await
            .entries
            .iter()
            .filter(|e| e.actor == actor)
            .cloned()
            .collect()
    }

    /// Get audit entries by event type
    pub async fn get_by_event_type(&self, event_type: &str) -> Vec<AuditEntry> {
        self.chain
            .read()
            .await
            .entries
            .iter()
            .filter(|e| e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Get audit entries within a time range
    pub async fn get_by_time_range(
        &self,
        start: chrono::DateTime<chrono::Utc>,
        end: chrono::DateTime<chrono::Utc>,
    ) -> Vec<AuditEntry> {
        self.chain
            .read()
            .await
            .entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect()
    }

    /// Export audit chain to JSON
    pub async fn export_chain(&self) -> Result<String, AuditError> {
        let chain = self.chain.read().await;
        serde_json::to_string_pretty(&*chain)
            .map_err(|e| AuditError::StorageError(e.to_string()))
    }

    /// Generate audit report
    pub async fn generate_report(&self, actor: Option<&str>) -> AuditReport {
        let chain = self.chain.read().await;

        let entries = if let Some(actor_name) = actor {
            chain
                .entries
                .iter()
                .filter(|e| e.actor == actor_name)
                .cloned()
                .collect()
        } else {
            chain.entries.clone()
        };

        let total_events = entries.len();
        let unique_actors = entries
            .iter()
            .map(|e| e.actor.clone())
            .collect::<std::collections::HashSet<_>>()
            .len();
        let event_types = entries
            .iter()
            .map(|e| e.event_type.clone())
            .collect::<std::collections::HashSet<_>>();

        AuditReport {
            chain_id: chain.chain_id.clone(),
            total_events,
            unique_actors,
            event_types: event_types.into_iter().collect(),
            earliest_event: entries.first().map(|e| e.timestamp),
            latest_event: entries.last().map(|e| e.timestamp),
            generated_at: chrono::Utc::now(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AuditReport {
    pub chain_id: String,
    pub total_events: usize,
    pub unique_actors: usize,
    pub event_types: Vec<String>,
    pub earliest_event: Option<chrono::DateTime<chrono::Utc>>,
    pub latest_event: Option<chrono::DateTime<chrono::Utc>>,
    pub generated_at: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_logging() {
        let logger = AuditLogger::new("test-key".to_string());

        let entry_id = logger
            .log_event(
                "data_access",
                "user1",
                serde_json::json!({
                    "resource": "sample1",
                    "action": "read"
                }),
            )
            .await
            .unwrap();

        assert!(!entry_id.is_empty());
    }

    #[tokio::test]
    async fn test_chain_verification() {
        let logger = AuditLogger::new("test-key".to_string());

        logger
            .log_event("event1", "user1", serde_json::json!({}))
            .await
            .unwrap();
        logger
            .log_event("event2", "user2", serde_json::json!({}))
            .await
            .unwrap();

        let valid = logger.verify_chain().await.unwrap();
        assert!(valid);
    }

    #[tokio::test]
    async fn test_actor_history() {
        let logger = AuditLogger::new("test-key".to_string());

        logger
            .log_event("event1", "user1", serde_json::json!({}))
            .await
            .unwrap();
        logger
            .log_event("event2", "user1", serde_json::json!({}))
            .await
            .unwrap();
        logger
            .log_event("event3", "user2", serde_json::json!({}))
            .await
            .unwrap();

        let history = logger.get_actor_history("user1").await;
        assert_eq!(history.len(), 2);
    }
}
