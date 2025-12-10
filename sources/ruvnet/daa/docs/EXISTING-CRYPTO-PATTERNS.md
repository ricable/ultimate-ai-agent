# Existing Crypto Implementation Patterns

**Date**: 2025-11-11
**Mission**: Analysis of working crypto implementations in the codebase

## Executive Summary

This document analyzes existing ML-KEM and ML-DSA implementations across the codebase to identify working patterns, import strategies, and code that successfully compiles.

### Key Findings

- ✅ **NAPI-RS ML-KEM**: FULLY WORKING implementation at `/home/user/daa/qudag/qudag-napi/src/crypto.rs`
- ⚠️ **NAPI-RS ML-DSA**: STUBBED (returns zeros)
- ✅ **Core Crypto ML-DSA**: WORKING implementation using `pqcrypto-dilithium`
- ⚠️ **WASM Crypto**: Mock implementations (not real crypto)

---

## 1. Working ML-KEM Implementation (NAPI-RS)

### File: `/home/user/daa/qudag/qudag-napi/src/crypto.rs`

**Status**: ✅ FULLY WORKING - Passes tests, generates real crypto

### Import Pattern

```rust
use ml_kem::{KemCore, MlKem768, MlKem768Params, EncodedSizeUser};
use kem::{Decapsulate, Encapsulate};
use rand::rngs::OsRng;
```

### Key Generation Pattern

```rust
#[napi]
pub fn mlkem768_generate_keypair() -> Result<KeyPair> {
  let mut rng = OsRng;

  // Generate keypair using ML-KEM-768
  let (ek, dk) = MlKem768::generate(&mut rng);

  // Convert to bytes
  let public_key = ek.as_bytes().to_vec();
  let secret_key = dk.as_bytes().to_vec();

  Ok(KeyPair {
    public_key: public_key.into(),
    secret_key: secret_key.into(),
  })
}
```

**Key Points**:
- `MlKem768::generate(&mut rng)` returns `(EncapsulationKey, DecapsulationKey)`
- Keys have `.as_bytes()` method for serialization
- Public key: 1184 bytes, Secret key: 2400 bytes

### Encapsulation Pattern

```rust
#[napi]
pub fn mlkem768_encapsulate(public_key: Buffer) -> Result<EncapsulatedSecret> {
  let mut rng = OsRng;

  // Parse public key from bytes
  let ek_array: [u8; 1184] = public_key.as_ref().try_into()
    .map_err(|_| Error::from_reason("Invalid public key format"))?;
  let ek = ml_kem::kem::EncapsulationKey::<MlKem768Params>::from(&ek_array);

  // Encapsulate to generate shared secret and ciphertext
  let encapsulated = ek.encapsulate(&mut rng);
  let ct = encapsulated.ciphertext();
  let ss = encapsulated.shared_secret();

  Ok(EncapsulatedSecret {
    ciphertext: ct.as_bytes().to_vec().into(),
    shared_secret: ss.as_bytes().to_vec().into(),
  })
}
```

**Key Points**:
- Convert bytes to fixed-size array: `[u8; 1184]`
- Create key: `ml_kem::kem::EncapsulationKey::<MlKem768Params>::from(&array)`
- Encapsulate: `ek.encapsulate(&mut rng)`
- Extract results: `.ciphertext()` and `.shared_secret()`
- Ciphertext: 1088 bytes, Shared secret: 32 bytes

### Decapsulation Pattern

```rust
#[napi]
pub fn mlkem768_decapsulate(ciphertext: Buffer, secret_key: Buffer) -> Result<Buffer> {
  // Parse secret key from bytes
  let dk_array: [u8; 2400] = secret_key.as_ref().try_into()
    .map_err(|_| Error::from_reason("Invalid secret key format"))?;
  let dk = ml_kem::kem::DecapsulationKey::<MlKem768Params>::from(&dk_array);

  // Parse ciphertext
  let ct_array: [u8; 1088] = ciphertext.as_ref().try_into()
    .map_err(|_| Error::from_reason("Invalid ciphertext format"))?;
  let ct = ml_kem::kem::Ciphertext::<MlKem768Params>::from(&ct_array);

  // Decapsulate to recover shared secret
  let ss = dk.decapsulate(&ct);

  Ok(ss.as_bytes().to_vec().into())
}
```

**Key Points**:
- Create decapsulation key: `ml_kem::kem::DecapsulationKey::<MlKem768Params>::from(&array)`
- Create ciphertext: `ml_kem::kem::Ciphertext::<MlKem768Params>::from(&array)`
- Decapsulate: `dk.decapsulate(&ct)`
- Returns shared secret directly

### Type System

```rust
// Key types are parameterized by MlKem768Params
ml_kem::kem::EncapsulationKey::<MlKem768Params>
ml_kem::kem::DecapsulationKey::<MlKem768Params>
ml_kem::kem::Ciphertext::<MlKem768Params>

// Size constants
MlKem768::EK_LEN  // 1184 bytes (encapsulation key)
MlKem768::DK_LEN  // 2400 bytes (decapsulation key)
MlKem768::CT_LEN  // 1088 bytes (ciphertext)
MlKem768::SS_LEN  // 32 bytes (shared secret)
```

---

## 2. Working ML-DSA Implementation (Core Crypto)

### File: `/home/user/daa/qudag/core/crypto/src/ml_dsa/mod.rs`

**Status**: ✅ FULLY WORKING - Uses `pqcrypto-dilithium` crate

### Import Pattern

```rust
use pqcrypto_dilithium::dilithium3::*;
use pqcrypto_traits::sign::{
    PublicKey as PqPublicKeyTrait,
    SecretKey as PqSecretKeyTrait,
    SignedMessage as PqSignedMessageTrait,
};
use rand_core::{CryptoRng, RngCore};
use zeroize::Zeroize;
```

### Constants

```rust
pub const ML_DSA_PUBLIC_KEY_SIZE: usize = 1952;
pub const ML_DSA_SECRET_KEY_SIZE: usize = 4032;
pub const ML_DSA_SIGNATURE_SIZE: usize = 3309;
```

### Key Generation Pattern

```rust
pub fn generate<R: CryptoRng + RngCore>(
    rng: &mut R,
) -> Result<MlDsaKeyPair, MlDsaError> {
    // Generate key pair using pqcrypto
    let (internal_public, internal_secret) = keypair();

    let public_key = <PublicKey as PqPublicKeyTrait>::as_bytes(&internal_public).to_vec();
    let secret_key = <SecretKey as PqSecretKeyTrait>::as_bytes(&internal_secret).to_vec();

    // Validate key sizes
    if public_key.len() != ML_DSA_PUBLIC_KEY_SIZE {
        return Err(MlDsaError::KeyGenerationFailed(format!(
            "Invalid public key size: {}",
            public_key.len()
        )));
    }

    Ok(MlDsaKeyPair {
        public_key,
        secret_key,
        internal_public,
        internal_secret,
    })
}
```

**Key Points**:
- `keypair()` from `pqcrypto_dilithium::dilithium3` generates keys
- No RNG parameter needed (uses internal randomness)
- Keys are stored both as bytes and internal types
- Validates key sizes after generation

### Signing Pattern

```rust
pub fn sign<R: CryptoRng + RngCore>(
    &self,
    message: &[u8],
    rng: &mut R,
) -> Result<Vec<u8>, MlDsaError> {
    // Use pqcrypto-dilithium's signing which includes rejection sampling
    let signed_msg = sign(message, &self.internal_secret);
    let signed_bytes = <SignedMessage as PqSignedMessageTrait>::as_bytes(&signed_msg);

    // Extract signature portion (everything except the message at the end)
    if signed_bytes.len() >= message.len() {
        let sig_len = signed_bytes.len() - message.len();
        Ok(signed_bytes[..sig_len].to_vec())
    } else {
        Err(MlDsaError::SigningFailed(
            "Invalid signed message format".to_string(),
        ))
    }
}
```

**Key Points**:
- `sign(message, &secret_key)` from pqcrypto-dilithium
- Returns `SignedMessage` which includes signature + message
- Extract signature by removing message length from end
- Signature: 3309 bytes for ML-DSA-65

### Verification Pattern

```rust
pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), MlDsaError> {
    // Check signature size
    if signature.len() < 2000 || signature.len() > ML_DSA_SIGNATURE_SIZE {
        return Err(MlDsaError::InvalidSignatureLength {
            expected: ML_DSA_SIGNATURE_SIZE,
            found: signature.len(),
        });
    }

    // Create signed message format expected by pqcrypto
    let mut signed_message_bytes = Vec::with_capacity(signature.len() + message.len());
    signed_message_bytes.extend_from_slice(signature);
    signed_message_bytes.extend_from_slice(message);

    let signed_msg = <SignedMessage as PqSignedMessageTrait>::from_bytes(&signed_message_bytes)
        .map_err(|_| MlDsaError::VerificationFailed)?;

    match open(&signed_msg, &self.internal_key) {
        Ok(verified_msg) => {
            // Use constant-time comparison
            if verified_msg.len() == message.len() && bool::from(verified_msg.ct_eq(message)) {
                Ok(())
            } else {
                Err(MlDsaError::VerificationFailed)
            }
        }
        Err(_) => Err(MlDsaError::VerificationFailed),
    }
}
```

**Key Points**:
- Reconstruct signed message: `signature + message`
- Parse: `SignedMessage::from_bytes(&signed_message_bytes)`
- Verify: `open(&signed_msg, &public_key)`
- Use constant-time comparison for security

### Public Key Usage

```rust
pub fn from_bytes(bytes: &[u8]) -> Result<MlDsaPublicKey, MlDsaError> {
    if bytes.len() != ML_DSA_PUBLIC_KEY_SIZE {
        return Err(MlDsaError::InvalidKeyLength {
            expected: ML_DSA_PUBLIC_KEY_SIZE,
            found: bytes.len(),
        });
    }

    let internal_key = <PublicKey as PqPublicKeyTrait>::from_bytes(bytes)
        .map_err(|_| MlDsaError::InvalidPublicKey("Failed to parse public key".to_string()))?;

    Ok(MlDsaPublicKey {
        key_bytes: bytes.to_vec(),
        internal_key,
    })
}
```

---

## 3. WASM Crypto Implementations (Simplified/Mock)

### Files:
- `/home/user/daa/qudag/qudag-wasm/src/wasm_crypto/ml_kem.rs`
- `/home/user/daa/qudag/qudag-wasm/src/wasm_crypto/ml_dsa.rs`

**Status**: ⚠️ MOCK IMPLEMENTATIONS - Not real cryptography

### ML-KEM WASM (Simplified)

```rust
pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>)> {
    let mut rng = rand::thread_rng();

    // Generate mock public key (1184 bytes for ML-KEM-768)
    let mut public_key = vec![0u8; KYBER_PUBLICKEYBYTES];
    rng.fill_bytes(&mut public_key);

    // Generate mock secret key (2400 bytes for ML-KEM-768)
    let mut secret_key = vec![0u8; KYBER_SECRETKEYBYTES];
    rng.fill_bytes(&mut secret_key);

    Ok((public_key, secret_key))
}
```

**Note**: This is just random bytes, not real ML-KEM

### ML-DSA WASM (Simplified)

```rust
pub fn generate_keypair() -> Result<(Vec<u8>, Vec<u8>)> {
    let mut public_key = vec![0u8; CRYPTO_PUBLICKEYBYTES];
    getrandom(&mut public_key)?;

    let mut secret_key = vec![0u8; CRYPTO_SECRETKEYBYTES];
    getrandom(&mut secret_key)?;

    Ok((public_key, secret_key))
}
```

**Note**: This is just random bytes, not real ML-DSA

### WASM Wrapper Pattern (crypto.rs)

```rust
#[wasm_bindgen]
pub struct WasmMlKemKeyPair {
    public_key: Vec<u8>,
    secret_key: Vec<u8>,
}

#[wasm_bindgen]
impl WasmMlKemKeyPair {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WasmMlKemKeyPair, JsError> {
        let (public_key, secret_key) = ml_kem::generate_keypair()
            .map_err(|e| JsError::new(&format!("Failed to generate ML-KEM keypair: {}", e)))?;

        Ok(Self {
            public_key,
            secret_key,
        })
    }

    #[wasm_bindgen(js_name = "getPublicKey")]
    pub fn get_public_key(&self) -> Vec<u8> {
        self.public_key.clone()
    }
}
```

---

## 4. Error Handling Patterns

### ML-KEM Error Handling

```rust
// NAPI-RS pattern
if public_key.len() != 1184 {
    return Err(Error::from_reason(format!(
      "Invalid public key length: expected 1184 bytes, got {}",
      public_key.len()
    )));
}

// Try-into pattern for fixed arrays
let ek_array: [u8; 1184] = public_key.as_ref().try_into()
    .map_err(|_| Error::from_reason("Invalid public key format"))?;
```

### ML-DSA Error Handling

```rust
// Custom error enum
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MlDsaError {
    #[error("Invalid public key: {0}")]
    InvalidPublicKey(String),

    #[error("Invalid signature length: expected {expected}, found {found}")]
    InvalidSignatureLength { expected: usize, found: usize },

    #[error("Signature verification failed")]
    VerificationFailed,
}

// Usage
pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), MlDsaError> {
    if signature.len() < 2000 || signature.len() > ML_DSA_SIGNATURE_SIZE {
        return Err(MlDsaError::InvalidSignatureLength {
            expected: ML_DSA_SIGNATURE_SIZE,
            found: signature.len(),
        });
    }
    // ...
}
```

---

## 5. Complete Import Checklist

### For ML-KEM (NAPI-RS)

```rust
// Core traits and types
use ml_kem::{KemCore, MlKem768, MlKem768Params, EncodedSizeUser};

// KEM operations
use kem::{Decapsulate, Encapsulate};

// Randomness
use rand::rngs::OsRng;

// For NAPI
use napi::bindgen_prelude::*;
use napi_derive::napi;
```

### For ML-DSA (Core Crypto)

```rust
// PQCrypto Dilithium
use pqcrypto_dilithium::dilithium3::*;

// Traits for key operations
use pqcrypto_traits::sign::{
    PublicKey as PqPublicKeyTrait,
    SecretKey as PqSecretKeyTrait,
    SignedMessage as PqSignedMessageTrait,
};

// Randomness
use rand_core::{CryptoRng, RngCore};

// Security features
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

// Hashing
use sha3::{
    digest::{ExtendableOutput, Update, XofReader},
    Shake128, Shake256,
};

// Error handling
use thiserror::Error;
```

---

## 6. Size Constants Reference

### ML-KEM-768

| Component | Size (bytes) | Constant |
|-----------|-------------|----------|
| Public Key (ek) | 1184 | `MlKem768Params::EK_LEN` |
| Secret Key (dk) | 2400 | `MlKem768Params::DK_LEN` |
| Ciphertext | 1088 | `MlKem768Params::CT_LEN` |
| Shared Secret | 32 | `MlKem768Params::SS_LEN` |

### ML-DSA-65 (Dilithium3)

| Component | Size (bytes) | Constant |
|-----------|-------------|----------|
| Public Key | 1952 | `ML_DSA_PUBLIC_KEY_SIZE` |
| Secret Key | 4032 | `ML_DSA_SECRET_KEY_SIZE` |
| Signature | 3309 | `ML_DSA_SIGNATURE_SIZE` |

---

## 7. Code Snippets That Compile

### Complete ML-KEM Example (NAPI-RS)

```rust
use ml_kem::{KemCore, MlKem768, MlKem768Params, EncodedSizeUser};
use kem::{Decapsulate, Encapsulate};
use rand::rngs::OsRng;
use napi::bindgen_prelude::*;
use napi_derive::napi;

#[napi]
pub fn full_mlkem_example() -> Result<bool> {
  let mut rng = OsRng;

  // 1. Generate keypair
  let (ek, dk) = MlKem768::generate(&mut rng);

  // 2. Encapsulate
  let ek_array: [u8; 1184] = ek.as_bytes().to_vec().try_into()
    .map_err(|_| Error::from_reason("Invalid key size"))?;
  let ek_reconstructed = ml_kem::kem::EncapsulationKey::<MlKem768Params>::from(&ek_array);
  let encapsulated = ek_reconstructed.encapsulate(&mut rng);

  // 3. Decapsulate
  let dk_array: [u8; 2400] = dk.as_bytes().to_vec().try_into()
    .map_err(|_| Error::from_reason("Invalid key size"))?;
  let dk_reconstructed = ml_kem::kem::DecapsulationKey::<MlKem768Params>::from(&dk_array);

  let ct_array: [u8; 1088] = encapsulated.ciphertext().as_bytes().to_vec().try_into()
    .map_err(|_| Error::from_reason("Invalid ciphertext size"))?;
  let ct = ml_kem::kem::Ciphertext::<MlKem768Params>::from(&ct_array);

  let ss_decap = dk_reconstructed.decapsulate(&ct);

  // 4. Verify shared secrets match
  let match_result = encapsulated.shared_secret().as_bytes() == ss_decap.as_bytes();

  Ok(match_result)
}
```

### Complete ML-DSA Example (Core Crypto)

```rust
use pqcrypto_dilithium::dilithium3::*;
use pqcrypto_traits::sign::{
    PublicKey as PqPublicKeyTrait,
    SecretKey as PqSecretKeyTrait,
    SignedMessage as PqSignedMessageTrait,
};
use rand::thread_rng;

pub fn full_mldsa_example() -> Result<bool, Box<dyn std::error::Error>> {
    let mut rng = thread_rng();

    // 1. Generate keypair
    let (pk, sk) = keypair();

    // 2. Sign a message
    let message = b"Hello, quantum-resistant world!";
    let signed_msg = sign(message, &sk);

    // 3. Extract signature
    let signed_bytes = <SignedMessage as PqSignedMessageTrait>::as_bytes(&signed_msg);
    let sig_len = signed_bytes.len() - message.len();
    let signature = &signed_bytes[..sig_len];

    // 4. Verify
    let mut verify_msg = Vec::new();
    verify_msg.extend_from_slice(signature);
    verify_msg.extend_from_slice(message);

    let signed_msg_verify = <SignedMessage as PqSignedMessageTrait>::from_bytes(&verify_msg)?;
    match open(&signed_msg_verify, &pk) {
        Ok(verified_msg) => Ok(verified_msg == message),
        Err(_) => Ok(false),
    }
}
```

---

## 8. Common Pitfalls and Solutions

### Pitfall 1: Type Inference Issues

**Problem**: `MlKem768` type is complex (`Kem<MlKem768Params>`)

**Solution**: Use explicit type annotations
```rust
let ek = ml_kem::kem::EncapsulationKey::<MlKem768Params>::from(&ek_array);
```

### Pitfall 2: Fixed-Size Array Conversions

**Problem**: Converting `Vec<u8>` to `[u8; N]`

**Solution**: Use `try_into()` with proper error handling
```rust
let array: [u8; 1184] = vec.try_into()
    .map_err(|_| Error::from_reason("Invalid size"))?;
```

### Pitfall 3: ML-DSA Signature Format

**Problem**: `sign()` returns signature + message concatenated

**Solution**: Extract signature by removing message length
```rust
let sig_len = signed_bytes.len() - message.len();
let signature = &signed_bytes[..sig_len];
```

### Pitfall 4: ML-DSA Verification Format

**Problem**: `open()` expects signature + message concatenated

**Solution**: Reconstruct signed message format
```rust
let mut signed_message = Vec::new();
signed_message.extend_from_slice(signature);
signed_message.extend_from_slice(message);
let signed_msg = SignedMessage::from_bytes(&signed_message)?;
```

---

## 9. Testing Patterns

### ML-KEM Test Pattern

```rust
#[test]
fn test_mlkem_encapsulate_decapsulate() {
    let keypair = mlkem768_generate_keypair().unwrap();
    let encapsulated = mlkem768_encapsulate(keypair.public_key.clone()).unwrap();

    assert_eq!(encapsulated.ciphertext.len(), 1088);
    assert_eq!(encapsulated.shared_secret.len(), 32);

    let decapsulated_secret = mlkem768_decapsulate(
        encapsulated.ciphertext,
        keypair.secret_key
    ).unwrap();

    assert_eq!(
        encapsulated.shared_secret.as_ref(),
        decapsulated_secret.as_ref(),
        "Shared secrets must match"
    );
}
```

### ML-DSA Test Pattern

```rust
#[test]
fn test_mldsa_sign_verify() {
    let mut rng = thread_rng();
    let keypair = MlDsaKeyPair::generate(&mut rng).unwrap();
    let message = b"test message";

    let signature = keypair.sign(message, &mut rng).unwrap();
    assert_eq!(signature.len(), ML_DSA_SIGNATURE_SIZE);

    let public_key = MlDsaPublicKey::from_bytes(keypair.public_key()).unwrap();
    assert!(public_key.verify(message, &signature).is_ok());
}
```

---

## 10. Recommended Approach for New Implementation

Based on the analysis, here's the recommended approach for implementing crypto in new code:

### For ML-KEM:
1. Use the **NAPI-RS pattern** (`/home/user/daa/qudag/qudag-napi/src/crypto.rs`) as reference
2. Import: `use ml_kem::{KemCore, MlKem768, MlKem768Params, EncodedSizeUser};`
3. Use parameterized types: `EncapsulationKey::<MlKem768Params>`
4. Generate: `MlKem768::generate(&mut OsRng)`
5. Convert to fixed arrays before creating keys from bytes

### For ML-DSA:
1. Use the **Core Crypto pattern** (`/home/user/daa/qudag/core/crypto/src/ml_dsa/mod.rs`) as reference
2. Import: `use pqcrypto_dilithium::dilithium3::*;`
3. Generate: `keypair()` (no RNG parameter)
4. Sign: `sign(message, &secret_key)` returns `SignedMessage`
5. Extract signature by removing message length
6. Verify by reconstructing signed message format

---

## Files Analyzed

### Working Implementations:
- `/home/user/daa/qudag/qudag-napi/src/crypto.rs` (ML-KEM ✅)
- `/home/user/daa/qudag/core/crypto/src/ml_kem/mod.rs` (Custom impl)
- `/home/user/daa/qudag/core/crypto/src/ml_dsa/mod.rs` (ML-DSA ✅)

### Mock/Simplified Implementations:
- `/home/user/daa/qudag/qudag-wasm/src/wasm_crypto/mod.rs`
- `/home/user/daa/qudag/qudag-wasm/src/wasm_crypto/ml_kem.rs`
- `/home/user/daa/qudag/qudag-wasm/src/wasm_crypto/ml_dsa.rs`

### Usage Examples Found:
- 563+ occurrences of `MlKem768` across codebase
- 100+ occurrences of `ml_kem::` imports
- 80+ occurrences of `ml_dsa::` imports

---

## Conclusion

The codebase contains **two distinct working implementations**:

1. **ML-KEM**: Best implementation in NAPI-RS using official `ml_kem` crate
2. **ML-DSA**: Best implementation in Core Crypto using `pqcrypto-dilithium`

The WASM implementations are simplified mocks and should not be used as reference for real cryptography.

For any new crypto implementation, use the patterns documented in sections 1 and 2 of this document.
