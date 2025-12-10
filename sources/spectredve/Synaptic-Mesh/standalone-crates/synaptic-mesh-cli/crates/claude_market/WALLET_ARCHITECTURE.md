# Token Wallet Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Synaptic Market Wallet                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │   PeerID A   │    │   PeerID B   │    │   PeerID C   │     │
│  │              │    │              │    │              │     │
│  │ Balance: 500 │    │ Balance: 300 │    │ Balance: 700 │     │
│  │ Locked:  100 │    │ Locked:    0 │    │ Locked:  200 │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                    │                    │              │
│         └────────────────────┴────────────────────┘              │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │   SQLite Storage  │                        │
│                    ├───────────────────┤                        │
│                    │  • balances       │                        │
│                    │  • transfers      │                        │
│                    │  • indexes        │                        │
│                    └───────────────────┘                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Database Schema

### Balances Table
```sql
CREATE TABLE balances (
    peer_id TEXT PRIMARY KEY,      -- PeerID as string
    available INTEGER NOT NULL,     -- Available tokens
    locked INTEGER NOT NULL,        -- Locked in escrow
    updated_at TEXT NOT NULL        -- ISO 8601 timestamp
);
```

### Transfers Table
```sql
CREATE TABLE transfers (
    id TEXT PRIMARY KEY,            -- UUID
    from_peer TEXT NOT NULL,        -- Sender PeerID
    to_peer TEXT NOT NULL,          -- Recipient PeerID
    amount INTEGER NOT NULL,        -- Transfer amount
    memo TEXT,                      -- Optional description
    signature BLOB NOT NULL,        -- Ed25519 signature
    created_at TEXT NOT NULL        -- ISO 8601 timestamp
);
```

## Token Flow Patterns

### 1. Direct Transfer
```
Alice (1000) ─────250 tokens────▶ Bob (250)
      │                               │
      └─ signature: Ed25519 ──────────┘
```

### 2. Escrow Flow
```
Step 1: Lock tokens
Alice (1000) ──lock 500──▶ Alice (500 available, 500 locked)

Step 2: Complete transaction
Alice (500, 500 locked) ──unlock──▶ Alice (1000, 0 locked)
                         ──transfer──▶ Charlie (500)

Step 3: Or refund on failure
Alice (500, 500 locked) ──unlock──▶ Alice (1000, 0 locked)
```

## Security Model

### Cryptographic Signatures
Every transfer is signed using Ed25519:
```
Message = "{id}:{from}:{to}:{amount}:{memo}"
Signature = Ed25519.sign(Message, PrivateKey)
```

### Balance Protection
- Atomic transactions ensure consistency
- Balance checks prevent overdrafts
- Locked tokens can't be spent

## Integration Points

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│    Market    │────▶│    Wallet    │◀────│    Escrow    │
│              │     │              │     │              │
│ Check funds  │     │ Lock/Unlock  │     │ Hold tokens  │
│ for orders   │     │ Transfer     │     │ for trades   │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │  Reputation  │
                    │              │
                    │ Track txns   │
                    └──────────────┘
```

## API Methods

### Core Operations
- `new(db_path)` - Create wallet instance
- `init_schema()` - Initialize database
- `get_balance(peer_id)` - Check balance
- `credit(peer_id, amount)` - Add tokens
- `debit(peer_id, amount)` - Remove tokens

### Escrow Operations
- `lock_tokens(peer_id, amount)` - Lock for escrow
- `unlock_tokens(peer_id, amount)` - Release from escrow

### Transfer Operations
- `transfer(from, to, amount, memo, key)` - P2P transfer
- `get_transfers(peer_id, limit)` - Transaction history

## Compliance Features

1. **No Shared Keys**: Each peer manages their own signing key
2. **Local Control**: All wallet data stored locally in SQLite
3. **Audit Trail**: Complete history of all transactions
4. **Contribution Rewards**: Tokens represent compute contribution

## Performance Characteristics

- **Storage**: O(n) where n = number of peers
- **Transfer**: O(1) atomic operation
- **History Query**: O(log n) with indexes
- **Concurrent Access**: Thread-safe with Mutex

## Example Usage Flow

```rust
// 1. Initialize wallet
let wallet = Wallet::new("wallet.db").await?;
wallet.init_schema().await?;

// 2. Earn tokens by contributing compute
wallet.credit(&my_peer_id, 1000).await?;

// 3. Check balance
let balance = wallet.get_balance(&my_peer_id).await?;
println!("Available: {}, Locked: {}", balance.available, balance.locked);

// 4. Transfer tokens for services
let signing_key = SigningKey::generate(&mut OsRng);
wallet.transfer(
    &my_peer_id,
    &service_provider_id,
    250,
    Some("Neural network training".to_string()),
    &signing_key
).await?;

// 5. Use escrow for large transactions
wallet.lock_tokens(&my_peer_id, 500).await?;
// ... perform work ...
wallet.unlock_tokens(&my_peer_id, 500).await?;
wallet.transfer(&my_peer_id, &worker_id, 500, None, &signing_key).await?;
```