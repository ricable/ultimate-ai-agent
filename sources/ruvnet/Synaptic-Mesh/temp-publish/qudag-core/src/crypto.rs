//! Quantum-resistant cryptography for QuDAG
//! 
//! Implements post-quantum cryptographic algorithms including ML-DSA for signatures
//! and ML-KEM for key encapsulation, along with quantum fingerprinting.

use std::collections::HashMap;
use std::sync::Arc;
use blake3::{Hash, Hasher};
use ml_dsa::{Keypair as MLDSAKeypair, Signature as MLDSASignature, VerifyingKey, SigningKey};
use ml_kem::{Keypair as MLKEMKeypair, EncapsulationKey, DecapsulationKey, Ciphertext, SharedSecret};
use rand::{RngCore, CryptoRng};
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::{DAGNode, QuDAGError, Result};

/// Main quantum-resistant cryptography handler
#[derive(Debug)]
pub struct QuantumResistantCrypto {
    ml_dsa_keypair: Option<MLDSAKeypair>,
    ml_kem_keypair: Option<MLKEMKeypair>,
    known_keys: Arc<std::sync::RwLock<HashMap<libp2p::PeerId, VerifyingKey>>>,
}

impl QuantumResistantCrypto {
    /// Create a new quantum-resistant crypto instance
    pub fn new() -> Self {
        Self {
            ml_dsa_keypair: None,
            ml_kem_keypair: None,
            known_keys: Arc::new(std::sync::RwLock::new(HashMap::new())),
        }
    }

    /// Generate ML-DSA keypair for signatures
    pub fn generate_ml_dsa_keypair<R: RngCore + CryptoRng>(&mut self, rng: &mut R) -> Result<()> {
        let keypair = MLDSAKeypair::generate(rng);
        self.ml_dsa_keypair = Some(keypair);
        Ok(())
    }

    /// Generate ML-KEM keypair for key encapsulation
    pub fn generate_ml_kem_keypair<R: RngCore + CryptoRng>(&mut self, rng: &mut R) -> Result<()> {
        let keypair = MLKEMKeypair::generate(rng);
        self.ml_kem_keypair = Some(keypair);
        Ok(())
    }

    /// Sign data using ML-DSA
    pub fn sign(&self, data: &[u8]) -> Result<PostQuantumSignature> {
        let keypair = self.ml_dsa_keypair.as_ref()
            .ok_or(QuDAGError::CryptoError("ML-DSA keypair not initialized".to_string()))?;

        let mut hasher = Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();

        let signature = keypair.signing_key().sign(hash.as_bytes());
        
        Ok(PostQuantumSignature {
            algorithm: SignatureAlgorithm::MLDSA,
            signature: signature.to_bytes().to_vec(),
            public_key: keypair.verifying_key().to_bytes().to_vec(),
        })
    }

    /// Verify a signature using ML-DSA
    pub fn verify_signature(&self, node: &DAGNode) -> Result<()> {
        let signature = node.signature()
            .ok_or(QuDAGError::ValidationError("No signature found".to_string()))?;

        if signature.algorithm != SignatureAlgorithm::MLDSA {
            return Err(QuDAGError::CryptoError("Unsupported signature algorithm".to_string()));
        }

        let verifying_key = VerifyingKey::from_bytes(&signature.public_key)
            .map_err(|e| QuDAGError::CryptoError(format!("Invalid public key: {}", e)))?;

        let sig = MLDSASignature::from_bytes(&signature.signature)
            .map_err(|e| QuDAGError::CryptoError(format!("Invalid signature: {}", e)))?;

        let mut hasher = Hasher::new();
        hasher.update(node.data());
        let hash = hasher.finalize();

        verifying_key.verify(hash.as_bytes(), &sig)
            .map_err(|e| QuDAGError::CryptoError(format!("Signature verification failed: {}", e)))?;

        Ok(())
    }

    /// Encapsulate a shared secret using ML-KEM
    pub fn encapsulate<R: RngCore + CryptoRng>(
        &self,
        peer_encaps_key: &[u8],
        rng: &mut R
    ) -> Result<(Ciphertext, SharedSecret)> {
        let encaps_key = EncapsulationKey::from_bytes(peer_encaps_key)
            .map_err(|e| QuDAGError::CryptoError(format!("Invalid encapsulation key: {}", e)))?;

        let (ciphertext, shared_secret) = encaps_key.encapsulate(rng);
        Ok((ciphertext, shared_secret))
    }

    /// Decapsulate a shared secret using ML-KEM
    pub fn decapsulate(&self, ciphertext: &Ciphertext) -> Result<SharedSecret> {
        let keypair = self.ml_kem_keypair.as_ref()
            .ok_or(QuDAGError::CryptoError("ML-KEM keypair not initialized".to_string()))?;

        let shared_secret = keypair.decapsulation_key().decapsulate(ciphertext);
        Ok(shared_secret)
    }

    /// Generate quantum fingerprint for .dark addressing
    pub fn generate_quantum_fingerprint(&self, input: &[u8]) -> QuantumFingerprint {
        let mut hasher = Hasher::new();
        hasher.update(b"qudag-quantum-fingerprint-v1");
        hasher.update(input);
        
        // Add post-quantum key material if available
        if let Some(keypair) = &self.ml_dsa_keypair {
            hasher.update(keypair.verifying_key().as_bytes());
        }
        
        let hash = hasher.finalize();
        
        QuantumFingerprint {
            hash: hash.as_bytes().to_vec(),
            algorithm: "BLAKE3-PQ".to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Verify a quantum fingerprint
    pub fn verify_quantum_fingerprint(
        &self,
        fingerprint: &QuantumFingerprint,
        input: &[u8]
    ) -> Result<bool> {
        let expected = self.generate_quantum_fingerprint(input);
        Ok(fingerprint.hash == expected.hash)
    }

    /// Get ML-DSA public key
    pub fn ml_dsa_public_key(&self) -> Result<Vec<u8>> {
        let keypair = self.ml_dsa_keypair.as_ref()
            .ok_or(QuDAGError::CryptoError("ML-DSA keypair not initialized".to_string()))?;
        Ok(keypair.verifying_key().to_bytes().to_vec())
    }

    /// Get ML-KEM public key
    pub fn ml_kem_public_key(&self) -> Result<Vec<u8>> {
        let keypair = self.ml_kem_keypair.as_ref()
            .ok_or(QuDAGError::CryptoError("ML-KEM keypair not initialized".to_string()))?;
        Ok(keypair.encapsulation_key().as_bytes().to_vec())
    }

    /// Register a peer's verifying key
    pub fn register_peer_key(&self, peer_id: libp2p::PeerId, verifying_key: VerifyingKey) {
        let mut keys = self.known_keys.write().unwrap();
        keys.insert(peer_id, verifying_key);
    }

    /// Get a peer's verifying key
    pub fn get_peer_key(&self, peer_id: &libp2p::PeerId) -> Option<VerifyingKey> {
        let keys = self.known_keys.read().unwrap();
        keys.get(peer_id).cloned()
    }
}

impl Default for QuantumResistantCrypto {
    fn default() -> Self {
        Self::new()
    }
}

/// Post-quantum signature wrapper
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PostQuantumSignature {
    pub algorithm: SignatureAlgorithm,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
}

/// Supported signature algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SignatureAlgorithm {
    MLDSA,
    // Future algorithms can be added here
    Dilithium, // Legacy name for ML-DSA
}

/// Quantum fingerprint for .dark addressing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumFingerprint {
    pub hash: Vec<u8>,
    pub algorithm: String,
    pub timestamp: u64,
}

impl QuantumFingerprint {
    /// Convert to .dark domain format
    pub fn to_dark_domain(&self) -> String {
        let encoded = hex::encode(&self.hash[..16]); // Use first 16 bytes for domain
        format!("{}.dark", encoded)
    }

    /// Parse from .dark domain
    pub fn from_dark_domain(domain: &str) -> Result<Self> {
        if !domain.ends_with(".dark") {
            return Err(QuDAGError::CryptoError("Invalid .dark domain".to_string()));
        }

        let hex_part = domain.strip_suffix(".dark").unwrap();
        let hash_prefix = hex::decode(hex_part)
            .map_err(|e| QuDAGError::CryptoError(format!("Invalid hex in .dark domain: {}", e)))?;

        // For parsing, we only have the prefix, so we'll need additional lookup
        Ok(Self {
            hash: hash_prefix,
            algorithm: "BLAKE3-PQ".to_string(),
            timestamp: 0, // Would need to be looked up
        })
    }
}

/// Zeroizing wrapper for sensitive key material
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct SecretKey {
    inner: Vec<u8>,
}

impl SecretKey {
    pub fn new(key: Vec<u8>) -> Self {
        Self { inner: key }
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.inner
    }
}

/// Re-export ML-DSA and ML-KEM keypairs
pub use ml_dsa::Keypair as MLDSAKeypair;
pub use ml_kem::Keypair as MLKEMKeypair;

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;

    #[test]
    fn test_ml_dsa_keygen() {
        let mut crypto = QuantumResistantCrypto::new();
        let mut rng = thread_rng();
        
        assert!(crypto.generate_ml_dsa_keypair(&mut rng).is_ok());
        assert!(crypto.ml_dsa_public_key().is_ok());
    }

    #[test]
    fn test_ml_kem_keygen() {
        let mut crypto = QuantumResistantCrypto::new();
        let mut rng = thread_rng();
        
        assert!(crypto.generate_ml_kem_keypair(&mut rng).is_ok());
        assert!(crypto.ml_kem_public_key().is_ok());
    }

    #[test]
    fn test_quantum_fingerprint() {
        let crypto = QuantumResistantCrypto::new();
        let data = b"test data";
        
        let fingerprint = crypto.generate_quantum_fingerprint(data);
        assert!(crypto.verify_quantum_fingerprint(&fingerprint, data).unwrap());
        
        let wrong_data = b"wrong data";
        assert!(!crypto.verify_quantum_fingerprint(&fingerprint, wrong_data).unwrap());
    }

    #[test]
    fn test_dark_domain() {
        let crypto = QuantumResistantCrypto::new();
        let data = b"test.example.com";
        
        let fingerprint = crypto.generate_quantum_fingerprint(data);
        let dark_domain = fingerprint.to_dark_domain();
        
        assert!(dark_domain.ends_with(".dark"));
        assert!(dark_domain.len() > 5); // More than just ".dark"
        
        let parsed = QuantumFingerprint::from_dark_domain(&dark_domain);
        assert!(parsed.is_ok());
    }

    #[test]
    fn test_signature_algorithm_serialization() {
        let sig = PostQuantumSignature {
            algorithm: SignatureAlgorithm::MLDSA,
            signature: vec![1, 2, 3],
            public_key: vec![4, 5, 6],
        };
        
        let json = serde_json::to_string(&sig).unwrap();
        let deserialized: PostQuantumSignature = serde_json::from_str(&json).unwrap();
        
        assert_eq!(sig, deserialized);
    }
}