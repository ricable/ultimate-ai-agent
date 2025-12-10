# ML-DSA API Implementation Guide

**Research Date:** 2025-11-11
**Target:** ml-dsa crate (RustCrypto implementation)
**Version Notes:** Based on latest stable API (0.1.0-rc.2 API structure)
**Status:** ⚠️  **UNAUDITED IMPLEMENTATION - USE AT YOUR OWN RISK**

## Table of Contents

1. [Overview](#overview)
2. [Crate Comparison](#crate-comparison)
3. [Core API Reference](#core-api-reference)
4. [Working Examples](#working-examples)
5. [Migration Guide](#migration-guide)
6. [Error Handling](#error-handling)
7. [Advanced Features](#advanced-features)
8. [Testing Patterns](#testing-patterns)

---

## Overview

The `ml-dsa` crate provides a pure Rust implementation of the Module-Lattice-Based Digital Signature Standard (ML-DSA) as specified in **FIPS 204 (final)**. ML-DSA was formerly known as CRYSTALS-Dilithium.

### Key Features

- ✅ Pure Rust implementation
- ✅ `no_std` compatible
- ✅ Three security levels (MlDsa44, MlDsa65, MlDsa87)
- ✅ Trait-based design using `signature` crate
- ✅ PKCS#8 support (optional feature)
- ❌ **Not independently audited**

### Security Parameters

| Parameter Set | Security Category | Public Key | Secret Key | Signature |
|--------------|-------------------|------------|------------|-----------|
| MlDsa44      | 2                 | 1,312 B    | 2,560 B    | 2,420 B   |
| MlDsa65      | 3                 | 1,952 B    | 4,032 B    | 3,309 B   |
| MlDsa87      | 5                 | 2,592 B    | 4,896 B    | 4,627 B   |

---

## Crate Comparison

### Current: `pqcrypto-dilithium`

```toml
[dependencies]
pqcrypto-dilithium = "0.5"
pqcrypto-traits = "0.3"
```

**Pros:**
- Stable and tested
- Part of RustPQ project
- Working in production

**Cons:**
- C bindings (not pure Rust)
- Less idiomatic API
- Larger binary size

### Target: `ml-dsa`

```toml
[dependencies]
ml-dsa = "0.1.0"  # or latest version
signature = "2.0"
rand = "0.8"
rand_core = { version = "0.6", features = ["std"] }
```

**Pros:**
- Pure Rust implementation
- Idiomatic trait-based API
- `no_std` compatible
- Smaller binary size

**Cons:**
- Not independently audited
- Newer, less battle-tested
- API still evolving

---

## Core API Reference

### 1. Required Imports

```rust
// Core ml-dsa types
use ml_dsa::{
    MlDsa44,      // Security category 2
    MlDsa65,      // Security category 3 (recommended)
    MlDsa87,      // Security category 5
    KeyGen,       // Key generation trait
};

// Signature traits (re-exported from signature crate)
use ml_dsa::signature::{
    Keypair,              // Keypair trait
    Signer,               // Signing trait
    Verifier,             // Verification trait
    RandomizedSigner,     // Randomized signing trait
    Error as SignatureError,
};

// RNG for key generation
use rand::thread_rng;
use rand_core::{CryptoRng, RngCore};
```

### 2. Type Signatures

#### KeyPair

```rust
pub struct KeyPair<P: MlDsaParams> {
    // Internal fields hidden
}

impl<P: MlDsaParams> KeyPair<P> {
    pub fn signing_key(&self) -> &SigningKey<P>;
    pub fn verifying_key(&self) -> &VerifyingKey<P>;
}

impl<P: MlDsaParams> signature::Keypair for KeyPair<P> {
    type VerifyingKey = VerifyingKey<P>;
}
```

#### SigningKey

```rust
pub struct SigningKey<P: MlDsaParams> {
    // Internal fields hidden
}

impl<P: MlDsaParams> SigningKey<P> {
    // Internal signing (advanced use)
    pub fn sign_internal(&self, Mp: &[&[u8]], rnd: &B32) -> Signature<P>;

    // Randomized signing with context
    pub fn sign_randomized<R: RngCore + CryptoRng>(
        &self,
        M: &[u8],
        ctx: &[u8],
        rng: &mut R,
    ) -> Result<Signature<P>, Error>;

    // Deterministic signing with context
    pub fn sign_deterministic(
        &self,
        M: &[u8],
        ctx: &[u8],
    ) -> Result<Signature<P>, Error>;

    // Encoding
    pub fn encode(&self) -> EncodedSigningKey<P>;
    pub fn decode(enc: &EncodedSigningKey<P>) -> Self;
}

impl<P: MlDsaParams> Signer<Signature<P>> for SigningKey<P> {
    fn try_sign(&self, msg: &[u8]) -> Result<Signature<P>, SignatureError>;
}
```

#### VerifyingKey

```rust
pub struct VerifyingKey<P: MlDsaParams> {
    // Internal fields hidden
}

impl<P: MlDsaParams> VerifyingKey<P> {
    // Internal verification (advanced use)
    pub fn verify_internal(&self, Mp: &[&[u8]], sigma: &Signature<P>) -> bool;

    // Verification with context
    pub fn verify_with_context(
        &self,
        M: &[u8],
        ctx: &[u8],
        sigma: &Signature<P>,
    ) -> bool;

    // Encoding
    pub fn encode(&self) -> EncodedVerifyingKey<P>;
    pub fn decode(enc: &EncodedVerifyingKey<P>) -> Self;
}

impl<P: MlDsaParams> Verifier<Signature<P>> for VerifyingKey<P> {
    fn verify(&self, msg: &[u8], signature: &Signature<P>) -> Result<(), SignatureError>;
}
```

#### Signature

```rust
pub struct Signature<P: MlDsaParams> {
    // Internal fields hidden
}

impl<P: MlDsaParams> Signature<P> {
    pub fn encode(&self) -> EncodedSignature<P>;
    pub fn decode(enc: &EncodedSignature<P>) -> Option<Self>;
}

impl<P: MlDsaParams> TryFrom<&[u8]> for Signature<P> {
    type Error = Error;
    fn try_from(value: &[u8]) -> Result<Self, Self::Error>;
}
```

### 3. KeyGen Trait

```rust
pub trait KeyGen: MlDsaParams {
    type KeyPair: signature::Keypair;

    // Generate key pair from RNG
    fn key_gen<R: RngCore + CryptoRng>(rng: &mut R) -> Self::KeyPair;

    // Generate key pair from seed (internal use)
    fn key_gen_internal(xi: &B32) -> Self::KeyPair;
}

// Implemented for all parameter sets:
impl KeyGen for MlDsa44 { /* ... */ }
impl KeyGen for MlDsa65 { /* ... */ }
impl KeyGen for MlDsa87 { /* ... */ }
```

---

## Working Examples

### Example 1: Basic Key Generation and Signing

```rust
use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::{Keypair, Signer, Verifier};
use rand::thread_rng;

fn basic_example() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Generate key pair
    let mut rng = thread_rng();
    let keypair = MlDsa65::key_gen(&mut rng);

    // 2. Prepare message
    let message = b"Hello, quantum-resistant world!";

    // 3. Sign message (uses trait method)
    let signature = keypair.signing_key().try_sign(message)?;

    // 4. Verify signature
    keypair.verifying_key().verify(message, &signature)?;

    println!("✅ Signature verified successfully!");
    Ok(())
}
```

### Example 2: Key Serialization

```rust
use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::Keypair;
use rand::thread_rng;

fn key_serialization_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let keypair = MlDsa65::key_gen(&mut rng);

    // Encode signing key
    let signing_key_bytes = keypair.signing_key().encode();
    println!("Signing key: {} bytes", signing_key_bytes.len());

    // Encode verifying key
    let verifying_key_bytes = keypair.verifying_key().encode();
    println!("Verifying key: {} bytes", verifying_key_bytes.len());

    // Decode verifying key
    let decoded_vk = <MlDsa65 as ml_dsa::MlDsaParams>::VerifyingKey::decode(
        &verifying_key_bytes
    );

    // Verify decoded key works
    let message = b"Test message";
    let signature = keypair.signing_key().try_sign(message)?;
    decoded_vk.verify(message, &signature)?;

    println!("✅ Key serialization successful!");
    Ok(())
}
```

### Example 3: Different Parameter Sets

```rust
use ml_dsa::{MlDsa44, MlDsa65, MlDsa87, KeyGen};
use ml_dsa::signature::{Signer, Verifier};
use rand::thread_rng;

fn parameter_sets_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let message = b"Test message";

    // Security Category 2 (faster, smaller)
    let kp44 = MlDsa44::key_gen(&mut rng);
    let sig44 = kp44.signing_key().try_sign(message)?;
    kp44.verifying_key().verify(message, &sig44)?;
    println!("MlDsa44 signature: {} bytes", sig44.encode().len());

    // Security Category 3 (recommended balance)
    let kp65 = MlDsa65::key_gen(&mut rng);
    let sig65 = kp65.signing_key().try_sign(message)?;
    kp65.verifying_key().verify(message, &sig65)?;
    println!("MlDsa65 signature: {} bytes", sig65.encode().len());

    // Security Category 5 (maximum security)
    let kp87 = MlDsa87::key_gen(&mut rng);
    let sig87 = kp87.signing_key().try_sign(message)?;
    kp87.verifying_key().verify(message, &sig87)?;
    println!("MlDsa87 signature: {} bytes", sig87.encode().len());

    Ok(())
}
```

### Example 4: Signature Serialization

```rust
use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::{Signer, Verifier};
use rand::thread_rng;

fn signature_serialization_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let keypair = MlDsa65::key_gen(&mut rng);
    let message = b"Important message";

    // Create signature
    let signature = keypair.signing_key().try_sign(message)?;

    // Serialize signature
    let sig_bytes = signature.encode();
    println!("Signature: {} bytes", sig_bytes.len());

    // Deserialize signature
    let decoded_sig = <MlDsa65 as ml_dsa::MlDsaParams>::Signature::decode(&sig_bytes)
        .ok_or("Failed to decode signature")?;

    // Verify deserialized signature
    keypair.verifying_key().verify(message, &decoded_sig)?;

    println!("✅ Signature serialization successful!");
    Ok(())
}
```

### Example 5: Error Handling

```rust
use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::{Signer, Verifier, Error as SignatureError};
use rand::thread_rng;

fn error_handling_example() {
    let mut rng = thread_rng();
    let keypair = MlDsa65::key_gen(&mut rng);
    let message = b"Original message";

    // Sign message
    let signature = keypair.signing_key().try_sign(message)
        .expect("Signing should not fail");

    // Test 1: Tampered message (should fail)
    let tampered_message = b"Tampered message";
    match keypair.verifying_key().verify(tampered_message, &signature) {
        Ok(_) => println!("❌ ERROR: Tampered message verified!"),
        Err(SignatureError::new()) => println!("✅ Correctly rejected tampered message"),
    }

    // Test 2: Wrong key (should fail)
    let other_keypair = MlDsa65::key_gen(&mut rng);
    match other_keypair.verifying_key().verify(message, &signature) {
        Ok(_) => println!("❌ ERROR: Wrong key verified!"),
        Err(_) => println!("✅ Correctly rejected signature with wrong key"),
    }

    // Test 3: Invalid signature bytes
    let invalid_sig_bytes = vec![0u8; 100]; // Too short
    match <MlDsa65 as ml_dsa::MlDsaParams>::Signature::try_from(invalid_sig_bytes.as_slice()) {
        Ok(_) => println!("❌ ERROR: Invalid signature accepted!"),
        Err(_) => println!("✅ Correctly rejected invalid signature format"),
    }
}
```

### Example 6: Batch Verification

```rust
use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::{Keypair, Signer, Verifier};
use rand::thread_rng;

fn batch_verification_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();

    // Create multiple keypairs and messages
    let messages = vec![
        b"Message 1".as_slice(),
        b"Message 2".as_slice(),
        b"Message 3".as_slice(),
    ];

    let mut keypairs = Vec::new();
    let mut signatures = Vec::new();

    for msg in &messages {
        let kp = MlDsa65::key_gen(&mut rng);
        let sig = kp.signing_key().try_sign(msg)?;
        keypairs.push(kp);
        signatures.push(sig);
    }

    // Verify all signatures
    for (i, ((msg, sig), kp)) in messages.iter()
        .zip(signatures.iter())
        .zip(keypairs.iter())
        .enumerate()
    {
        kp.verifying_key().verify(msg, sig)?;
        println!("✅ Signature {} verified", i + 1);
    }

    println!("✅ All {} signatures verified successfully!", messages.len());
    Ok(())
}
```

### Example 7: Context-Based Signing (Advanced)

```rust
use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::Keypair;
use rand::thread_rng;

fn context_signing_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let keypair = MlDsa65::key_gen(&mut rng);

    let message = b"Transaction data";
    let context = b"qudag.transaction.v1"; // Domain separation

    // Randomized signing with context
    let signature = keypair.signing_key()
        .sign_randomized(message, context, &mut rng)?;

    // Verify with context
    let is_valid = keypair.verifying_key()
        .verify_with_context(message, context, &signature);

    if is_valid {
        println!("✅ Context-based signature verified!");
    } else {
        println!("❌ Verification failed!");
    }

    // Deterministic signing with context
    let det_signature = keypair.signing_key()
        .sign_deterministic(message, context)?;

    let is_valid_det = keypair.verifying_key()
        .verify_with_context(message, context, &det_signature);

    if is_valid_det {
        println!("✅ Deterministic context-based signature verified!");
    }

    Ok(())
}
```

---

## Migration Guide

### From `pqcrypto-dilithium` to `ml-dsa`

#### Before (pqcrypto-dilithium)

```rust
use pqcrypto_dilithium::dilithium3::*;
use pqcrypto_traits::sign::{
    PublicKey as PqPublicKeyTrait,
    SecretKey as PqSecretKeyTrait,
    SignedMessage as PqSignedMessageTrait,
};

// Key generation
let (public_key, secret_key) = keypair();

// Signing
let signed_msg = sign(message, &secret_key);

// Verification
let verified_msg = open(&signed_msg, &public_key)?;
```

#### After (ml-dsa)

```rust
use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::{Keypair, Signer, Verifier};
use rand::thread_rng;

// Key generation
let mut rng = thread_rng();
let keypair = MlDsa65::key_gen(&mut rng);

// Signing
let signature = keypair.signing_key().try_sign(message)?;

// Verification
keypair.verifying_key().verify(message, &signature)?;
```

### API Mapping Table

| pqcrypto-dilithium | ml-dsa |
|-------------------|---------|
| `keypair()` | `MlDsa65::key_gen(&mut rng)` |
| `PublicKey` | `VerifyingKey<MlDsa65>` |
| `SecretKey` | `SigningKey<MlDsa65>` |
| `sign(msg, &sk)` | `signing_key.try_sign(msg)?` |
| `open(&signed, &pk)` | `verifying_key.verify(msg, &sig)?` |
| `PublicKey::as_bytes()` | `verifying_key.encode()` |
| `SecretKey::as_bytes()` | `signing_key.encode()` |

### Migration Checklist

- [ ] Update `Cargo.toml` dependencies
- [ ] Replace key generation code
- [ ] Update signing operations
- [ ] Update verification operations
- [ ] Handle new error types (`signature::Error`)
- [ ] Update key serialization/deserialization
- [ ] Test thoroughly with existing data
- [ ] Update documentation
- [ ] Consider security audit implications

---

## Error Handling

### Error Types

```rust
// Primary error type from signature crate
use ml_dsa::signature::Error as SignatureError;

// Error creation
impl SignatureError {
    pub fn new() -> Self;
}

// Common error scenarios
fn handle_errors(
    keypair: &impl Keypair,
    message: &[u8],
    signature: &impl signature::Signature,
) {
    match keypair.verifying_key().verify(message, signature) {
        Ok(()) => println!("Verification successful"),
        Err(e) => {
            eprintln!("Verification failed: {:?}", e);
            // Possible causes:
            // - Message was modified
            // - Signature was corrupted
            // - Wrong public key
            // - Invalid signature format
        }
    }
}
```

### Best Practices

1. **Always handle Result types explicitly**
   ```rust
   // ❌ Bad: Panics on error
   let signature = signing_key.try_sign(message).unwrap();

   // ✅ Good: Proper error handling
   let signature = signing_key.try_sign(message)
       .map_err(|e| format!("Signing failed: {:?}", e))?;
   ```

2. **Use proper error propagation**
   ```rust
   fn sign_message(msg: &[u8]) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
       let mut rng = thread_rng();
       let keypair = MlDsa65::key_gen(&mut rng);
       let signature = keypair.signing_key().try_sign(msg)?;
       Ok(signature.encode().to_vec())
   }
   ```

3. **Implement domain-specific errors**
   ```rust
   #[derive(Debug, thiserror::Error)]
   enum CryptoError {
       #[error("Signature verification failed")]
       VerificationFailed(#[from] signature::Error),

       #[error("Invalid key format")]
       InvalidKeyFormat,

       #[error("Key generation failed")]
       KeyGenerationFailed,
   }
   ```

---

## Advanced Features

### 1. No-Std Support

```rust
#![no_std]

extern crate alloc;
use alloc::vec::Vec;

use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::{Signer, Verifier};

// Works in no_std environments!
fn no_std_example(seed: &[u8; 32]) {
    // Use a custom RNG for no_std
    // (Implementation depends on your platform)
    let mut rng = MyCustomRng::from_seed(seed);
    let keypair = MlDsa65::key_gen(&mut rng);
    // ... rest of the code
}
```

### 2. PKCS#8 Support (Optional Feature)

```toml
[dependencies]
ml-dsa = { version = "0.1.0", features = ["pkcs8"] }
```

```rust
use ml_dsa::{MlDsa65, KeyGen};
use pkcs8::{EncodePrivateKey, EncodePublicKey};

fn pkcs8_example() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let keypair = MlDsa65::key_gen(&mut rng);

    // Encode to PKCS#8 (if feature enabled)
    // let private_key_pem = keypair.signing_key().to_pkcs8_pem()?;
    // let public_key_pem = keypair.verifying_key().to_public_key_pem()?;

    Ok(())
}
```

### 3. Deterministic vs Randomized Signing

```rust
use ml_dsa::{MlDsa65, KeyGen};
use rand::thread_rng;

fn signing_modes() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = thread_rng();
    let keypair = MlDsa65::key_gen(&mut rng);
    let message = b"Test message";
    let context = b"domain.context";

    // Randomized signing (default, recommended)
    let sig1 = keypair.signing_key().sign_randomized(message, context, &mut rng)?;
    let sig2 = keypair.signing_key().sign_randomized(message, context, &mut rng)?;
    // sig1 != sig2 (different random nonces)

    // Deterministic signing (same signature every time)
    let sig3 = keypair.signing_key().sign_deterministic(message, context)?;
    let sig4 = keypair.signing_key().sign_deterministic(message, context)?;
    // sig3 == sig4 (deterministic)

    Ok(())
}
```

---

## Testing Patterns

### Unit Test Template

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use ml_dsa::{MlDsa65, KeyGen};
    use ml_dsa::signature::{Keypair, Signer, Verifier};
    use rand::thread_rng;

    #[test]
    fn test_basic_sign_verify() {
        let mut rng = thread_rng();
        let keypair = MlDsa65::key_gen(&mut rng);
        let message = b"Test message";

        let signature = keypair.signing_key()
            .try_sign(message)
            .expect("Signing should succeed");

        keypair.verifying_key()
            .verify(message, &signature)
            .expect("Verification should succeed");
    }

    #[test]
    fn test_tampered_message_fails() {
        let mut rng = thread_rng();
        let keypair = MlDsa65::key_gen(&mut rng);
        let original = b"Original message";
        let tampered = b"Tampered message";

        let signature = keypair.signing_key()
            .try_sign(original)
            .expect("Signing should succeed");

        assert!(
            keypair.verifying_key()
                .verify(tampered, &signature)
                .is_err(),
            "Tampered message should fail verification"
        );
    }

    #[test]
    fn test_wrong_key_fails() {
        let mut rng = thread_rng();
        let keypair1 = MlDsa65::key_gen(&mut rng);
        let keypair2 = MlDsa65::key_gen(&mut rng);
        let message = b"Test message";

        let signature = keypair1.signing_key()
            .try_sign(message)
            .expect("Signing should succeed");

        assert!(
            keypair2.verifying_key()
                .verify(message, &signature)
                .is_err(),
            "Wrong key should fail verification"
        );
    }

    #[test]
    fn test_key_serialization_roundtrip() {
        let mut rng = thread_rng();
        let keypair = MlDsa65::key_gen(&mut rng);

        // Encode keys
        let vk_bytes = keypair.verifying_key().encode();
        let sk_bytes = keypair.signing_key().encode();

        // Decode keys
        let decoded_vk = <MlDsa65 as ml_dsa::MlDsaParams>::VerifyingKey::decode(&vk_bytes);
        let decoded_sk = <MlDsa65 as ml_dsa::MlDsaParams>::SigningKey::decode(&sk_bytes);

        // Test decoded keys work
        let message = b"Test message";
        let signature = decoded_sk.try_sign(message)
            .expect("Signing with decoded key should work");

        decoded_vk.verify(message, &signature)
            .expect("Verification with decoded key should work");
    }
}
```

### Property-Based Testing with Proptest

```rust
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use ml_dsa::{MlDsa65, KeyGen};
    use ml_dsa::signature::{Signer, Verifier};
    use rand::thread_rng;

    proptest! {
        #[test]
        fn test_sign_verify_any_message(message in prop::collection::vec(any::<u8>(), 0..1000)) {
            let mut rng = thread_rng();
            let keypair = MlDsa65::key_gen(&mut rng);

            let signature = keypair.signing_key()
                .try_sign(&message)
                .expect("Signing should not panic");

            keypair.verifying_key()
                .verify(&message, &signature)
                .expect("Valid signature should verify");
        }

        #[test]
        fn test_different_messages_different_signatures(
            msg1 in prop::collection::vec(any::<u8>(), 1..100),
            msg2 in prop::collection::vec(any::<u8>(), 1..100)
        ) {
            prop_assume!(msg1 != msg2);

            let mut rng = thread_rng();
            let keypair = MlDsa65::key_gen(&mut rng);

            let sig1 = keypair.signing_key().try_sign(&msg1).unwrap();
            let sig2 = keypair.signing_key().try_sign(&msg2).unwrap();

            // Signatures for different messages should be different
            assert_ne!(sig1.encode(), sig2.encode());
        }
    }
}
```

### Benchmarking Template

```rust
#[cfg(test)]
mod benches {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    use ml_dsa::{MlDsa65, KeyGen};
    use ml_dsa::signature::{Signer, Verifier};
    use rand::thread_rng;

    fn bench_key_generation(c: &mut Criterion) {
        c.bench_function("ml_dsa_65_keygen", |b| {
            b.iter(|| {
                let mut rng = thread_rng();
                black_box(MlDsa65::key_gen(&mut rng))
            })
        });
    }

    fn bench_signing(c: &mut Criterion) {
        let mut rng = thread_rng();
        let keypair = MlDsa65::key_gen(&mut rng);
        let message = b"Benchmark message";

        c.bench_function("ml_dsa_65_sign", |b| {
            b.iter(|| {
                black_box(keypair.signing_key().try_sign(black_box(message)))
            })
        });
    }

    fn bench_verification(c: &mut Criterion) {
        let mut rng = thread_rng();
        let keypair = MlDsa65::key_gen(&mut rng);
        let message = b"Benchmark message";
        let signature = keypair.signing_key().try_sign(message).unwrap();

        c.bench_function("ml_dsa_65_verify", |b| {
            b.iter(|| {
                black_box(
                    keypair.verifying_key()
                        .verify(black_box(message), black_box(&signature))
                )
            })
        });
    }

    criterion_group!(benches, bench_key_generation, bench_signing, bench_verification);
    criterion_main!(benches);
}
```

---

## Implementation Notes

### Current QuDAG Implementation Status

**File:** `/home/user/daa/qudag/core/crypto/Cargo.toml`
```toml
pqcrypto-dilithium = "0.5"
```

**File:** `/home/user/daa/qudag/core/crypto/src/ml_dsa/mod.rs`
- Uses `pqcrypto-dilithium` with custom wrapper
- Implements `MlDsaKeyPair`, `MlDsaPublicKey` types
- Provides batch verification
- Includes side-channel resistance features

### Migration Considerations

1. **Binary Compatibility**: The signature format between `pqcrypto-dilithium` and `ml-dsa` should be compatible (both implement FIPS 204), but test thoroughly.

2. **Performance**: Pure Rust implementation may have different performance characteristics. Benchmark before deploying.

3. **Security**: Current implementation uses C bindings which may have been audited. The `ml-dsa` crate is explicitly unaudited.

4. **Testing**: Extensive testing required, especially for:
   - Key serialization compatibility
   - Signature verification across implementations
   - Timing attack resistance
   - Memory safety

### Recommended Approach

1. **Parallel Implementation**: Add `ml-dsa` alongside `pqcrypto-dilithium`
2. **Feature Flags**: Use Cargo features to switch implementations
3. **Cross-Verification**: Verify signatures between implementations
4. **Gradual Rollout**: Test extensively before production deployment

---

## Resources

### Documentation
- **ml-dsa crate docs**: https://docs.rs/ml-dsa/
- **FIPS 204 Standard**: https://nvlpubs.nist.gov/nistpubs/fips/nist.fips.204.pdf
- **RustCrypto signatures**: https://github.com/RustCrypto/signatures

### Alternative Implementations
- **fips204**: https://crates.io/crates/fips204 (pure Rust, constant-time)
- **pqcrypto-dilithium**: https://crates.io/crates/pqcrypto-dilithium (current)
- **libcrux_ml_dsa**: https://docs.rs/libcrux-ml-dsa/

### Related Standards
- **FIPS 203** (ML-KEM): Module-Lattice-Based Key-Encapsulation
- **FIPS 205** (SLH-DSA): Stateless Hash-Based Signatures

---

## Appendix: Complete Example Application

```rust
//! Complete ML-DSA example application
//!
//! This demonstrates a full workflow including:
//! - Key generation and storage
//! - Message signing
//! - Signature verification
//! - Error handling
//! - Serialization

use ml_dsa::{MlDsa65, KeyGen};
use ml_dsa::signature::{Keypair, Signer, Verifier};
use rand::thread_rng;
use std::fs;
use std::path::Path;

#[derive(Debug)]
enum AppError {
    Crypto(String),
    Io(std::io::Error),
}

impl From<std::io::Error> for AppError {
    fn from(e: std::io::Error) -> Self {
        AppError::Io(e)
    }
}

impl From<ml_dsa::signature::Error> for AppError {
    fn from(e: ml_dsa::signature::Error) -> Self {
        AppError::Crypto(format!("{:?}", e))
    }
}

struct SignatureSystem {
    keypair: <MlDsa65 as ml_dsa::KeyGen>::KeyPair,
}

impl SignatureSystem {
    /// Create new signature system with fresh keys
    fn new() -> Self {
        let mut rng = thread_rng();
        let keypair = MlDsa65::key_gen(&mut rng);
        Self { keypair }
    }

    /// Save keys to files
    fn save_keys(&self, dir: &Path) -> Result<(), AppError> {
        fs::create_dir_all(dir)?;

        let vk_bytes = self.keypair.verifying_key().encode();
        let sk_bytes = self.keypair.signing_key().encode();

        fs::write(dir.join("verifying_key.bin"), vk_bytes.as_slice())?;
        fs::write(dir.join("signing_key.bin"), sk_bytes.as_slice())?;

        println!("✅ Keys saved to {}", dir.display());
        Ok(())
    }

    /// Sign a message
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, AppError> {
        let signature = self.keypair.signing_key().try_sign(message)?;
        Ok(signature.encode().to_vec())
    }

    /// Verify a signature
    fn verify(&self, message: &[u8], signature: &[u8]) -> Result<bool, AppError> {
        let sig = <MlDsa65 as ml_dsa::MlDsaParams>::Signature::try_from(signature)
            .map_err(|e| AppError::Crypto(format!("{:?}", e)))?;

        match self.keypair.verifying_key().verify(message, &sig) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false),
        }
    }
}

fn main() -> Result<(), AppError> {
    println!("🔐 ML-DSA Signature System Demo\n");

    // 1. Create signature system
    let system = SignatureSystem::new();
    println!("✅ Generated quantum-resistant key pair");

    // 2. Save keys
    let key_dir = Path::new("./ml_dsa_keys");
    system.save_keys(key_dir)?;

    // 3. Sign a message
    let message = b"Hello, quantum-resistant world!";
    let signature = system.sign(message)?;
    println!("✅ Signed message ({} bytes)", signature.len());

    // 4. Verify signature
    let is_valid = system.verify(message, &signature)?;
    println!("✅ Signature valid: {}", is_valid);

    // 5. Test with tampered message
    let tampered = b"Hello, tampered world!";
    let is_valid_tampered = system.verify(tampered, &signature)?;
    println!("✅ Tampered message valid: {} (should be false)", is_valid_tampered);

    println!("\n🎉 Demo complete!");
    Ok(())
}
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2025-11-11 | 1.0 | Initial research and documentation |

---

## License Note

The `ml-dsa` crate is dual-licensed under Apache 2.0 and MIT licenses. This guide is provided for educational purposes.

**⚠️  SECURITY WARNING**: This implementation has never been independently audited. Use at your own risk in production systems. Consider using audited implementations or obtaining a security audit before deploying in critical systems.
