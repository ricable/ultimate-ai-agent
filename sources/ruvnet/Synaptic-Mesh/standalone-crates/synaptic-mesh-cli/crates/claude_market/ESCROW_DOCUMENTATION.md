# Escrow Management System Documentation

## Overview

The Escrow Management System provides secure, transparent, and auditable transactions for compute contribution rewards in the Synaptic Market. This system ensures that users maintain full control while facilitating trustless exchanges between compute requesters and providers.

**Important**: This escrow system is designed for compute contribution rewards, NOT for Claude access. It ensures compliance with the principle that participants use their own Claude subscriptions.

## Key Features

### 1. Multi-Signature Support
- **Single Signature**: Either party can release funds
- **Both Parties**: Requires agreement from both requester and provider
- **M-of-N Arbitrators**: Configurable threshold of arbitrator signatures
- **Time-Locked**: Automatic release after specified time

### 2. Complete State Machine
```
Created → Funded → Completed → Released/Refunded
                ↓
             Disputed → Resolved
                ↓
             Expired (timeout)
```

### 3. Dispute Resolution
- Either party can raise disputes with evidence
- Arbitrators can review and make decisions
- Support for split settlements
- All decisions are cryptographically signed

### 4. Wallet Integration
- Automatic token locking during escrow
- Atomic transfers on release
- Protection against double-spending
- Full balance tracking

### 5. Comprehensive Audit Trail
- Every state transition is logged
- Actor identification for all operations
- Timestamps and detailed descriptions
- Immutable record of all activities

## API Reference

### Creating an Escrow

```rust
let escrow_agreement = escrow.create_escrow(
    job_id,           // Unique job identifier
    requester,        // PeerId of compute requester
    provider,         // PeerId of compute provider
    amount,           // Reward amount in ruv tokens
    multisig_type,    // Signature requirements
    arbitrators,      // List of arbitrator PeerIds
    timeout_minutes,  // Timeout duration
).await?;
```

### Funding an Escrow

```rust
// Requester locks tokens to fund the escrow
let funded_escrow = escrow.fund_escrow(
    &escrow_id,
    &requester_peer_id,
    &signing_key,
).await?;
```

### Completing a Job

```rust
// Provider marks job as completed
let completed_escrow = escrow.mark_completed(
    &escrow_id,
    &provider_peer_id,
    &signing_key,
).await?;
```

### Releasing Funds

```rust
// Create release authorization
let release_auth = ReleaseAuth {
    escrow_id,
    authorizer: requester_peer_id,
    decision: ReleaseDecision::ApproveRelease,
    signature: sign_message(&signing_key),
    created_at: Utc::now(),
};

// Release funds based on multi-sig requirements
let released_escrow = escrow.release_funds(
    &escrow_id,
    release_auth,
).await?;
```

### Dispute Handling

```rust
// Raise a dispute
let disputed_escrow = escrow.raise_dispute(
    &escrow_id,
    &disputer_peer_id,
    reason,
    evidence_cids,
    &signing_key,
).await?;

// Resolve a dispute (arbitrator only)
let resolved_escrow = escrow.resolve_dispute(
    &escrow_id,
    &arbitrator_peer_id,
    DisputeOutcome::Split { 
        provider_share: 60, 
        requester_share: 40 
    },
    reasoning,
    &signing_key,
).await?;
```

## Multi-Signature Types

### Single Signature
```rust
MultiSigType::Single
```
- Either party can release funds
- Fastest resolution
- Best for trusted relationships

### Both Parties Required
```rust
MultiSigType::BothParties
```
- Both requester and provider must agree
- Ensures mutual satisfaction
- No single point of failure

### Arbitrator Consensus
```rust
MultiSigType::Arbitrators { required: 2, total: 3 }
```
- Requires M-of-N arbitrator signatures
- Decentralized dispute resolution
- Protection against collusion

### Time-Locked Release
```rust
MultiSigType::TimeLocked { 
    release_after: Utc::now() + Duration::days(7) 
}
```
- Automatic release after specified time
- No manual intervention needed
- Useful for recurring providers

## Security Considerations

### Cryptographic Signatures
- All operations require Ed25519 signatures
- Signatures are verified and stored
- Provides non-repudiation

### State Validation
- Strict state machine enforcement
- Invalid transitions are rejected
- Prevents exploitation attempts

### Access Control
- Only parties to escrow can perform actions
- Arbitrators limited to dispute resolution
- Role-based permissions

### Audit Trail
- Complete history of all operations
- Tamper-proof logging
- Useful for dispute investigation

## Database Schema

### Escrows Table
- Stores main escrow agreements
- Tracks state and participants
- JSON fields for complex data

### Signatures Table
- Stores all cryptographic signatures
- Links signatures to signers
- Prevents signature replay

### Release Authorizations
- Tracks release decisions
- Supports multi-signature accumulation
- Maintains decision history

### Arbitrator Decisions
- Records dispute resolutions
- Includes reasoning and outcomes
- Permanent record for accountability

## Best Practices

### For Requesters
1. Choose appropriate multi-sig type based on trust level
2. Set reasonable timeouts (typically 24-72 hours)
3. Provide clear job specifications
4. Document any disputes thoroughly

### For Providers
1. Only accept jobs you can complete
2. Mark jobs completed promptly
3. Keep evidence of work performed
4. Communicate issues early

### For Arbitrators
1. Review all evidence carefully
2. Provide clear reasoning for decisions
3. Consider partial settlements when appropriate
4. Maintain neutrality

## Error Handling

Common errors and solutions:

### Insufficient Balance
```
Error: Insufficient balance: required 100, available 50
```
Solution: Ensure wallet has sufficient funds before creating escrow

### Invalid State Transition
```
Error: Escrow not in funded state
```
Solution: Check escrow state before performing operations

### Unauthorized Access
```
Error: Not a party to this escrow
```
Solution: Verify you're using correct PeerId

### Timeout Expired
```
Error: Escrow has expired
```
Solution: Create new escrow or process timeout

## Integration Example

```rust
use claude_market::escrow::*;
use claude_market::wallet::Wallet;
use std::sync::Arc;

// Initialize services
let wallet = Arc::new(Wallet::new("wallet.db").await?);
let escrow_service = Escrow::new("escrow.db", wallet).await?;

// Create escrow for compute job
let job_id = Uuid::new_v4();
let agreement = escrow_service.create_escrow(
    job_id,
    requester_id,
    provider_id,
    100, // 100 ruv tokens
    MultiSigType::BothParties,
    vec![],
    1440, // 24 hour timeout
).await?;

// Fund escrow
escrow_service.fund_escrow(
    &agreement.id,
    &requester_id,
    &requester_key,
).await?;

// ... job execution ...

// Mark completed
escrow_service.mark_completed(
    &agreement.id,
    &provider_id,
    &provider_key,
).await?;

// Both parties approve release
let req_auth = ReleaseAuth {
    escrow_id: agreement.id,
    authorizer: requester_id,
    decision: ReleaseDecision::ApproveRelease,
    signature: req_signature,
    created_at: Utc::now(),
};

let prov_auth = ReleaseAuth {
    escrow_id: agreement.id,
    authorizer: provider_id,
    decision: ReleaseDecision::ApproveRelease,
    signature: prov_signature,
    created_at: Utc::now(),
};

escrow_service.release_funds(&agreement.id, req_auth).await?;
escrow_service.release_funds(&agreement.id, prov_auth).await?;

// Funds are now transferred to provider
```

## Compliance Notes

1. **No Claude Access Trading**: This escrow system is explicitly for compute contribution rewards, not for trading Claude API access
2. **User Control**: Users maintain full control over their participation
3. **Transparency**: All operations are auditable and transparent
4. **Voluntary Participation**: The system facilitates voluntary compute contribution

## Testing

The escrow system includes comprehensive tests covering:
- Happy path flows
- Edge cases and error conditions
- Multi-signature scenarios
- Dispute resolution
- Timeout processing
- Concurrent operations

Run tests with:
```bash
cargo test -p claude_market
```

## Future Enhancements

Potential improvements for future versions:
1. Reputation-based arbitrator selection
2. Escrow insurance pools
3. Automated job verification
4. Cross-chain token bridges
5. Zero-knowledge proof integration