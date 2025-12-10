//! Quantum-resistant cryptography module
//! 
//! Provides WASM bindings for ML-DSA signatures and ML-KEM encryption

use wasm_bindgen::prelude::*;
use js_sys::{Object, Reflect, Uint8Array};
use ml_dsa::{MlDsa65, SigningKey, VerifyingKey};
use ml_kem::{MlKem768, EncapsulationKey, DecapsulationKey};
use rand::rngs::OsRng;
use blake3::Hasher;
use zeroize::Zeroize;
use crate::P2PError;

/// Quantum-resistant cryptographic engine
#[wasm_bindgen]
pub struct CryptoEngine {
    signing_key: Option<SigningKey<MlDsa65>>,
    verifying_key: Option<VerifyingKey<MlDsa65>>,
    encapsulation_key: Option<EncapsulationKey<MlKem768>>,
    decapsulation_key: Option<DecapsulationKey<MlKem768>>,
}

#[wasm_bindgen]
impl CryptoEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<CryptoEngine, P2PError> {
        Ok(CryptoEngine {
            signing_key: None,
            verifying_key: None,
            encapsulation_key: None,
            decapsulation_key: None,
        })
    }

    /// Initialize the crypto engine
    #[wasm_bindgen]
    pub fn init(&mut self) -> Result<(), P2PError> {
        // Generate ML-DSA keys for signatures
        let (signing_key, verifying_key) = MlDsa65::generate(&mut OsRng);
        self.signing_key = Some(signing_key);
        self.verifying_key = Some(verifying_key);

        // Generate ML-KEM keys for encryption
        let (decapsulation_key, encapsulation_key) = MlKem768::generate(&mut OsRng);
        self.encapsulation_key = Some(encapsulation_key);
        self.decapsulation_key = Some(decapsulation_key);

        crate::log("Quantum-resistant crypto engine initialized");
        Ok(())
    }

    /// Generate new key pairs
    #[wasm_bindgen]
    pub fn generate_keys(&mut self) -> Result<Object, P2PError> {
        // Generate ML-DSA keys
        let (signing_key, verifying_key) = MlDsa65::generate(&mut OsRng);
        
        // Generate ML-KEM keys
        let (decapsulation_key, encapsulation_key) = MlKem768::generate(&mut OsRng);

        // Store keys
        self.signing_key = Some(signing_key);
        self.verifying_key = Some(verifying_key.clone());
        self.encapsulation_key = Some(encapsulation_key.clone());
        self.decapsulation_key = Some(decapsulation_key);

        // Create JavaScript object with public keys
        let keys = Object::new();
        
        // Convert verifying key to bytes
        let verifying_key_bytes = verifying_key.as_bytes();
        let verifying_key_array = Uint8Array::new_with_length(verifying_key_bytes.len() as u32);
        verifying_key_array.copy_from(verifying_key_bytes);
        
        // Convert encapsulation key to bytes
        let encapsulation_key_bytes = encapsulation_key.as_bytes();
        let encapsulation_key_array = Uint8Array::new_with_length(encapsulation_key_bytes.len() as u32);
        encapsulation_key_array.copy_from(encapsulation_key_bytes);

        Reflect::set(&keys, &"verifying_key".into(), &verifying_key_array.into())?;
        Reflect::set(&keys, &"encapsulation_key".into(), &encapsulation_key_array.into())?;
        
        Ok(keys)
    }

    /// Sign data with ML-DSA
    #[wasm_bindgen]
    pub fn sign_data(&self, data: &[u8]) -> Result<Uint8Array, P2PError> {
        let signing_key = self.signing_key.as_ref()
            .ok_or_else(|| P2PError::new("Signing key not initialized", "KEY_NOT_INITIALIZED"))?;

        // Hash the data first
        let mut hasher = Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();

        // Sign the hash
        let signature = signing_key.sign(hash.as_bytes());
        let signature_bytes = signature.as_bytes();
        
        let signature_array = Uint8Array::new_with_length(signature_bytes.len() as u32);
        signature_array.copy_from(signature_bytes);
        
        Ok(signature_array)
    }

    /// Verify ML-DSA signature
    #[wasm_bindgen]
    pub fn verify_signature(&self, data: &[u8], signature: &[u8], public_key: &[u8]) -> Result<bool, P2PError> {
        // Parse the verifying key
        let verifying_key = VerifyingKey::<MlDsa65>::from_bytes(public_key.try_into()
            .map_err(|_| P2PError::new("Invalid public key length", "INVALID_KEY"))?);

        // Hash the data
        let mut hasher = Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();

        // Parse the signature
        let signature = ml_dsa::Signature::<MlDsa65>::from_bytes(signature.try_into()
            .map_err(|_| P2PError::new("Invalid signature length", "INVALID_SIGNATURE"))?);

        // Verify the signature
        match verifying_key.verify(hash.as_bytes(), &signature) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    /// Encrypt data with ML-KEM
    #[wasm_bindgen]
    pub fn encrypt_data(&self, data: &[u8], public_key: &[u8]) -> Result<Uint8Array, P2PError> {
        // Parse the encapsulation key
        let encapsulation_key = EncapsulationKey::<MlKem768>::from_bytes(public_key.try_into()
            .map_err(|_| P2PError::new("Invalid encapsulation key length", "INVALID_KEY"))?);

        // Encapsulate to get shared secret
        let (ciphertext, shared_secret) = encapsulation_key.encapsulate(&mut OsRng);

        // Use shared secret to encrypt data (simplified - in practice use AES-GCM)
        let mut encrypted = data.to_vec();
        for (i, byte) in encrypted.iter_mut().enumerate() {
            *byte ^= shared_secret.as_bytes()[i % shared_secret.as_bytes().len()];
        }

        // Combine ciphertext and encrypted data
        let mut result = ciphertext.as_bytes().to_vec();
        result.extend_from_slice(&encrypted);

        let result_array = Uint8Array::new_with_length(result.len() as u32);
        result_array.copy_from(&result);
        
        Ok(result_array)
    }

    /// Decrypt data with ML-KEM
    #[wasm_bindgen]
    pub fn decrypt_data(&self, encrypted_data: &[u8], _private_key: &[u8]) -> Result<Uint8Array, P2PError> {
        let decapsulation_key = self.decapsulation_key.as_ref()
            .ok_or_else(|| P2PError::new("Decapsulation key not initialized", "KEY_NOT_INITIALIZED"))?;

        // Split ciphertext and encrypted data
        const CIPHERTEXT_SIZE: usize = 1088; // ML-KEM-768 ciphertext size
        if encrypted_data.len() < CIPHERTEXT_SIZE {
            return Err(P2PError::new("Invalid encrypted data length", "INVALID_DATA"));
        }

        let ciphertext_bytes = &encrypted_data[..CIPHERTEXT_SIZE];
        let encrypted_payload = &encrypted_data[CIPHERTEXT_SIZE..];

        // Parse ciphertext
        let ciphertext = ml_kem::Ciphertext::<MlKem768>::from_bytes(ciphertext_bytes.try_into()
            .map_err(|_| P2PError::new("Invalid ciphertext", "INVALID_CIPHERTEXT"))?);

        // Decapsulate to get shared secret
        let shared_secret = decapsulation_key.decapsulate(&ciphertext);

        // Decrypt data using shared secret
        let mut decrypted = encrypted_payload.to_vec();
        for (i, byte) in decrypted.iter_mut().enumerate() {
            *byte ^= shared_secret.as_bytes()[i % shared_secret.as_bytes().len()];
        }

        let result_array = Uint8Array::new_with_length(decrypted.len() as u32);
        result_array.copy_from(&decrypted);
        
        Ok(result_array)
    }

    /// Hash data with BLAKE3
    #[wasm_bindgen]
    pub fn hash_data(&self, data: &[u8]) -> Uint8Array {
        let mut hasher = Hasher::new();
        hasher.update(data);
        let hash = hasher.finalize();
        
        let hash_array = Uint8Array::new_with_length(32);
        hash_array.copy_from(hash.as_bytes());
        hash_array
    }

    /// Create quantum fingerprint (hash + signature)
    #[wasm_bindgen]
    pub fn create_fingerprint(&self, data: &[u8]) -> Result<Object, P2PError> {
        // Hash the data
        let hash = self.hash_data(data);
        
        // Sign the hash
        let signature = self.sign_data(data)?;
        
        // Create fingerprint object
        let fingerprint = Object::new();
        Reflect::set(&fingerprint, &"hash".into(), &hash.into())?;
        Reflect::set(&fingerprint, &"signature".into(), &signature.into())?;
        
        Ok(fingerprint)
    }

    /// Verify quantum fingerprint
    #[wasm_bindgen]
    pub fn verify_fingerprint(&self, data: &[u8], fingerprint: &Object, public_key: &[u8]) -> Result<bool, P2PError> {
        // Extract hash and signature from fingerprint
        let hash = Reflect::get(fingerprint, &"hash".into())
            .map_err(|_| P2PError::new("Missing hash in fingerprint", "INVALID_FINGERPRINT"))?;
        let signature = Reflect::get(fingerprint, &"signature".into())
            .map_err(|_| P2PError::new("Missing signature in fingerprint", "INVALID_FINGERPRINT"))?;

        // Convert to Uint8Array
        let signature_array = Uint8Array::new(&signature);
        let signature_bytes = signature_array.to_vec();

        // Verify the signature
        self.verify_signature(data, &signature_bytes, public_key)
    }

    /// Get public keys
    #[wasm_bindgen]
    pub fn get_public_keys(&self) -> Result<Object, P2PError> {
        let verifying_key = self.verifying_key.as_ref()
            .ok_or_else(|| P2PError::new("Keys not initialized", "KEY_NOT_INITIALIZED"))?;
        let encapsulation_key = self.encapsulation_key.as_ref()
            .ok_or_else(|| P2PError::new("Keys not initialized", "KEY_NOT_INITIALIZED"))?;

        let keys = Object::new();
        
        // Verifying key
        let verifying_key_bytes = verifying_key.as_bytes();
        let verifying_key_array = Uint8Array::new_with_length(verifying_key_bytes.len() as u32);
        verifying_key_array.copy_from(verifying_key_bytes);
        
        // Encapsulation key
        let encapsulation_key_bytes = encapsulation_key.as_bytes();
        let encapsulation_key_array = Uint8Array::new_with_length(encapsulation_key_bytes.len() as u32);
        encapsulation_key_array.copy_from(encapsulation_key_bytes);

        Reflect::set(&keys, &"verifying_key".into(), &verifying_key_array.into())?;
        Reflect::set(&keys, &"encapsulation_key".into(), &encapsulation_key_array.into())?;
        
        Ok(keys)
    }
}

impl Default for CryptoEngine {
    fn default() -> Self {
        Self::new().expect("Failed to create CryptoEngine")
    }
}

// Ensure keys are zeroized when dropped
impl Drop for CryptoEngine {
    fn drop(&mut self) {
        if let Some(mut key) = self.signing_key.take() {
            key.zeroize();
        }
        if let Some(mut key) = self.decapsulation_key.take() {
            key.zeroize();
        }
    }
}