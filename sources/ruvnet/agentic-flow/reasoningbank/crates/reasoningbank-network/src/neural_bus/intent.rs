//! Intent-capped actions with Ed25519 signatures
//!
//! Provides cryptographic authorization for operations with:
//! - Ed25519 digital signatures
//! - Scope-based permissions
//! - Spend caps for resource limits
//! - Replay protection via nonces

use crate::{NetworkError, Result};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::RwLock;
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};

/// Authorization scope
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scope {
    /// Read-only operations
    Read,
    /// Write operations (requires elevated permissions)
    Write,
    /// Administrative operations
    Admin,
    /// Custom scope with string identifier
    Custom(String),
}

impl Scope {
    pub fn from_str(s: &str) -> Self {
        match s {
            "read" => Self::Read,
            "write" => Self::Write,
            "admin" => Self::Admin,
            _ => Self::Custom(s.to_string()),
        }
    }

    pub fn as_str(&self) -> &str {
        match self {
            Self::Read => "read",
            Self::Write => "write",
            Self::Admin => "admin",
            Self::Custom(s) => s.as_str(),
        }
    }

    /// Check if this scope permits the operation
    pub fn permits(&self, required: &Scope) -> bool {
        match (self, required) {
            (Self::Admin, _) => true,
            (Self::Write, Self::Read) => true,
            (Self::Write, Self::Write) => true,
            (Self::Read, Self::Read) => true,
            (Self::Custom(a), Self::Custom(b)) => a == b,
            _ => false,
        }
    }
}

/// Intent represents an authorized action with resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Key ID for signature verification
    pub kid: String,
    /// Timestamp (Unix milliseconds)
    pub ts: i64,
    /// Nonce for replay protection
    pub nonce: String,
    /// Authorization scope
    pub scope: Scope,
    /// Spend cap (arbitrary units, e.g., compute credits)
    pub cap: u64,
    /// Operation being authorized
    pub op: String,
    /// Ed25519 signature (base64-encoded)
    pub sig: String,
}

impl Intent {
    /// Create a new intent and sign it
    pub fn new(
        kid: String,
        scope: Scope,
        cap: u64,
        op: String,
        signing_key: &SigningKey,
    ) -> Self {
        let ts = chrono::Utc::now().timestamp_millis();
        let nonce = uuid::Uuid::new_v4().to_string();

        // Create message to sign
        let message = format!("{}:{}:{}:{}:{}:{}", kid, ts, nonce, scope.as_str(), cap, op);
        let message_hash = Sha256::digest(message.as_bytes());

        // Sign the hash
        let signature = signing_key.sign(&message_hash);
        let sig = BASE64.encode(signature.to_bytes());

        Self {
            kid,
            ts,
            nonce,
            scope,
            cap,
            op,
            sig,
        }
    }

    /// Verify the intent signature
    pub fn verify(&self, verifying_key: &VerifyingKey) -> Result<()> {
        // Reconstruct message
        let message = format!(
            "{}:{}:{}:{}:{}:{}",
            self.kid, self.ts, self.nonce, self.scope.as_str(), self.cap, self.op
        );
        let message_hash = Sha256::digest(message.as_bytes());

        // Decode signature
        let sig_bytes = BASE64.decode(&self.sig)
            .map_err(|e| NetworkError::Internal(format!("Invalid signature encoding: {}", e)))?;

        let signature = Signature::from_bytes(&sig_bytes.try_into().map_err(|_| {
            NetworkError::Internal("Invalid signature length".to_string())
        })?);

        // Verify signature
        verifying_key
            .verify(&message_hash, &signature)
            .map_err(|e| NetworkError::Internal(format!("Signature verification failed: {}", e)))?;

        Ok(())
    }

    /// Check if intent has expired (5 minute window)
    pub fn is_expired(&self) -> bool {
        let now = chrono::Utc::now().timestamp_millis();
        let age = now - self.ts;
        age > 300_000 // 5 minutes
    }
}

/// Intent verifier with key management and nonce tracking
pub struct IntentVerifier {
    /// Known public keys (kid -> VerifyingKey)
    keys: Arc<RwLock<std::collections::HashMap<String, VerifyingKey>>>,
    /// Used nonces for replay protection (rolling window)
    nonces: Arc<RwLock<HashSet<String>>>,
    /// Maximum nonce cache size
    max_nonces: usize,
}

impl IntentVerifier {
    /// Create a new intent verifier
    pub fn new(max_nonces: usize) -> Self {
        Self {
            keys: Arc::new(RwLock::new(std::collections::HashMap::new())),
            nonces: Arc::new(RwLock::new(HashSet::new())),
            max_nonces,
        }
    }

    /// Register a public key
    pub async fn register_key(&self, kid: String, key: VerifyingKey) {
        let mut keys = self.keys.write().await;
        keys.insert(kid, key);
    }

    /// Verify an intent
    pub async fn verify_intent(&self, intent: &Intent, required_scope: &Scope) -> Result<()> {
        // Check expiration
        if intent.is_expired() {
            return Err(NetworkError::Internal("Intent has expired".to_string()));
        }

        // Check nonce (replay protection)
        let mut nonces = self.nonces.write().await;
        if nonces.contains(&intent.nonce) {
            return Err(NetworkError::Internal("Nonce already used (replay attack?)".to_string()));
        }

        // Check scope
        if !intent.scope.permits(required_scope) {
            return Err(NetworkError::Internal(format!(
                "Insufficient scope: {:?} does not permit {:?}",
                intent.scope, required_scope
            )));
        }

        // Get public key
        let keys = self.keys.read().await;
        let key = keys
            .get(&intent.kid)
            .ok_or_else(|| NetworkError::Internal(format!("Unknown key ID: {}", intent.kid)))?;

        // Verify signature
        intent.verify(key)?;

        // Record nonce
        nonces.insert(intent.nonce.clone());

        // Prune old nonces if needed
        if nonces.len() > self.max_nonces {
            // Keep newest 80% of nonces
            let to_remove = nonces.len() - (self.max_nonces * 4 / 5);
            let old_nonces: Vec<String> = nonces.iter().take(to_remove).cloned().collect();
            for nonce in old_nonces {
                nonces.remove(&nonce);
            }
        }

        Ok(())
    }

    /// Verify intent from frame header
    pub async fn verify_frame_header(
        &self,
        kid: &str,
        ts: i64,
        nonce: &str,
        scope_str: &str,
        cap: u64,
        sig: &str,
        op: &str,
        required_scope: &Scope,
    ) -> Result<()> {
        let intent = Intent {
            kid: kid.to_string(),
            ts,
            nonce: nonce.to_string(),
            scope: Scope::from_str(scope_str),
            cap,
            op: op.to_string(),
            sig: sig.to_string(),
        };

        self.verify_intent(&intent, required_scope).await
    }
}

/// Helper to generate Ed25519 keypairs
pub fn generate_keypair() -> (SigningKey, VerifyingKey) {
    let mut csprng = rand::rngs::OsRng;
    let signing_key = SigningKey::generate(&mut csprng);
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scope_permissions() {
        let admin = Scope::Admin;
        let write = Scope::Write;
        let read = Scope::Read;

        assert!(admin.permits(&write));
        assert!(admin.permits(&read));
        assert!(write.permits(&read));
        assert!(!read.permits(&write));
    }

    #[test]
    fn test_intent_creation_and_verification() {
        let (signing_key, verifying_key) = generate_keypair();
        let intent = Intent::new(
            "test-key".to_string(),
            Scope::Write,
            1000,
            "store_pattern".to_string(),
            &signing_key,
        );

        assert!(intent.verify(&verifying_key).is_ok());
    }

    #[test]
    fn test_intent_expiration() {
        let (signing_key, _) = generate_keypair();
        let mut intent = Intent::new(
            "test-key".to_string(),
            Scope::Read,
            100,
            "test".to_string(),
            &signing_key,
        );

        assert!(!intent.is_expired());

        // Set timestamp to 6 minutes ago
        intent.ts = chrono::Utc::now().timestamp_millis() - 360_000;
        assert!(intent.is_expired());
    }

    #[tokio::test]
    async fn test_verifier_nonce_replay_protection() {
        let (signing_key, verifying_key) = generate_keypair();
        let verifier = IntentVerifier::new(1000);

        verifier.register_key("test-key".to_string(), verifying_key).await;

        let intent = Intent::new(
            "test-key".to_string(),
            Scope::Write,
            1000,
            "test".to_string(),
            &signing_key,
        );

        // First verification should succeed
        assert!(verifier.verify_intent(&intent, &Scope::Read).await.is_ok());

        // Second verification with same nonce should fail
        assert!(verifier.verify_intent(&intent, &Scope::Read).await.is_err());
    }
}
