//! Token wallet management for Claude API tokens

use crate::error::{MarketError, Result};
use chrono::{DateTime, Utc};
use ed25519_dalek::{Signer, SigningKey};
use libp2p::PeerId;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::Mutex;
use uuid::Uuid;

/// Token balance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalance {
    /// Owner's peer ID
    pub owner: PeerId,
    /// Available balance
    pub available: u64,
    /// Locked in escrow
    pub locked: u64,
    /// Total balance (available + locked)
    pub total: u64,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
}

/// Token transfer record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTransfer {
    /// Unique transfer ID
    pub id: Uuid,
    /// Sender peer ID
    pub from: PeerId,
    /// Recipient peer ID
    pub to: PeerId,
    /// Transfer amount
    pub amount: u64,
    /// Transfer reason/memo
    pub memo: Option<String>,
    /// Cryptographic signature
    pub signature: Vec<u8>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

/// Wallet for managing Claude API tokens
pub struct Wallet {
    db: Mutex<Connection>,
    signing_keys: Mutex<HashMap<PeerId, SigningKey>>,
}

impl std::fmt::Debug for Wallet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Wallet")
            .field("db", &"<database_connection>")
            .field("signing_keys", &"<signing_keys>")
            .finish()
    }
}

impl Wallet {
    /// Create a new wallet instance
    pub async fn new(db_path: &str) -> Result<Self> {
        let db = Connection::open(db_path)?;
        Ok(Self {
            db: Mutex::new(db),
            signing_keys: Mutex::new(HashMap::new()),
        })
    }

    /// Initialize database schema
    pub async fn init_schema(&self) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS balances (
                peer_id TEXT PRIMARY KEY,
                available INTEGER NOT NULL DEFAULT 0,
                locked INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS transfers (
                id TEXT PRIMARY KEY,
                from_peer TEXT NOT NULL,
                to_peer TEXT NOT NULL,
                amount INTEGER NOT NULL,
                memo TEXT,
                signature BLOB NOT NULL,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_transfers_from ON transfers(from_peer);
            CREATE INDEX IF NOT EXISTS idx_transfers_to ON transfers(to_peer);
            CREATE INDEX IF NOT EXISTS idx_transfers_created ON transfers(created_at);
            "#,
            [],
        )?;

        Ok(())
    }

    /// Get token balance for a peer
    pub async fn get_balance(&self, peer_id: &PeerId) -> Result<TokenBalance> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();

        let balance = db
            .query_row(
                "SELECT available, locked, updated_at FROM balances WHERE peer_id = ?1",
                params![peer_str],
                |row| {
                    let available: i64 = row.get(0)?;
                    let locked: i64 = row.get(1)?;
                    let updated_at: String = row.get(2)?;
                    Ok(TokenBalance {
                        owner: *peer_id,
                        available: available as u64,
                        locked: locked as u64,
                        total: (available + locked) as u64,
                        updated_at: DateTime::parse_from_rfc3339(&updated_at)
                            .unwrap()
                            .with_timezone(&Utc),
                    })
                },
            )
            .optional()?
            .unwrap_or_else(|| TokenBalance {
                owner: *peer_id,
                available: 0,
                locked: 0,
                total: 0,
                updated_at: Utc::now(),
            });

        Ok(balance)
    }

    /// Credit tokens to an account
    pub async fn credit(&self, peer_id: &PeerId, amount: u64) -> Result<TokenBalance> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();
        let now = Utc::now().to_rfc3339();

        db.execute(
            r#"
            INSERT INTO balances (peer_id, available, locked, updated_at)
            VALUES (?1, ?2, 0, ?3)
            ON CONFLICT(peer_id) DO UPDATE SET
                available = available + ?2,
                updated_at = ?3
            "#,
            params![peer_str, amount as i64, now],
        )?;

        drop(db);
        self.get_balance(peer_id).await
    }

    /// Debit tokens from an account
    pub async fn debit(&self, peer_id: &PeerId, amount: u64) -> Result<TokenBalance> {
        let balance = self.get_balance(peer_id).await?;
        
        if balance.available < amount {
            return Err(MarketError::InsufficientBalance {
                required: amount,
                available: balance.available,
            });
        }

        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();
        let now = Utc::now().to_rfc3339();

        db.execute(
            r#"
            UPDATE balances 
            SET available = available - ?2,
                updated_at = ?3
            WHERE peer_id = ?1
            "#,
            params![peer_str, amount as i64, now],
        )?;

        drop(db);
        self.get_balance(peer_id).await
    }

    /// Lock tokens for escrow
    pub async fn lock_tokens(&self, peer_id: &PeerId, amount: u64) -> Result<TokenBalance> {
        let balance = self.get_balance(peer_id).await?;
        
        if balance.available < amount {
            return Err(MarketError::InsufficientBalance {
                required: amount,
                available: balance.available,
            });
        }

        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();
        let now = Utc::now().to_rfc3339();

        db.execute(
            r#"
            UPDATE balances 
            SET available = available - ?2,
                locked = locked + ?2,
                updated_at = ?3
            WHERE peer_id = ?1
            "#,
            params![peer_str, amount as i64, now],
        )?;

        drop(db);
        self.get_balance(peer_id).await
    }

    /// Unlock tokens from escrow
    pub async fn unlock_tokens(&self, peer_id: &PeerId, amount: u64) -> Result<TokenBalance> {
        let balance = self.get_balance(peer_id).await?;
        
        if balance.locked < amount {
            return Err(MarketError::Internal(
                format!("Cannot unlock {} tokens, only {} locked", amount, balance.locked)
            ));
        }

        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();
        let now = Utc::now().to_rfc3339();

        db.execute(
            r#"
            UPDATE balances 
            SET available = available + ?2,
                locked = locked - ?2,
                updated_at = ?3
            WHERE peer_id = ?1
            "#,
            params![peer_str, amount as i64, now],
        )?;

        drop(db);
        self.get_balance(peer_id).await
    }

    /// Transfer tokens between accounts with signature
    pub async fn transfer(
        &self,
        from: &PeerId,
        to: &PeerId,
        amount: u64,
        memo: Option<String>,
        signing_key: &SigningKey,
    ) -> Result<TokenTransfer> {
        // Verify sender has sufficient balance
        let sender_balance = self.get_balance(from).await?;
        if sender_balance.available < amount {
            return Err(MarketError::InsufficientBalance {
                required: amount,
                available: sender_balance.available,
            });
        }

        // Create transfer record
        let transfer = TokenTransfer {
            id: Uuid::new_v4(),
            from: *from,
            to: *to,
            amount,
            memo: memo.clone(),
            signature: Vec::new(),
            created_at: Utc::now(),
        };

        // Sign the transfer
        let message = format!(
            "{}:{}:{}:{}:{}",
            transfer.id,
            transfer.from,
            transfer.to,
            transfer.amount,
            transfer.memo.as_ref().unwrap_or(&String::new())
        );
        let signature = signing_key.sign(message.as_bytes());

        let mut signed_transfer = transfer.clone();
        signed_transfer.signature = signature.to_bytes().to_vec();

        // Execute transfer atomically
        let db = self.db.lock().await;
        let tx = db.unchecked_transaction()?;

        // Debit sender
        tx.execute(
            "UPDATE balances SET available = available - ?2, updated_at = ?3 WHERE peer_id = ?1",
            params![from.to_string(), amount as i64, Utc::now().to_rfc3339()],
        )?;

        // Credit recipient
        tx.execute(
            r#"
            INSERT INTO balances (peer_id, available, locked, updated_at)
            VALUES (?1, ?2, 0, ?3)
            ON CONFLICT(peer_id) DO UPDATE SET
                available = available + ?2,
                updated_at = ?3
            "#,
            params![to.to_string(), amount as i64, Utc::now().to_rfc3339()],
        )?;

        // Record transfer
        tx.execute(
            r#"
            INSERT INTO transfers (id, from_peer, to_peer, amount, memo, signature, created_at)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                signed_transfer.id.to_string(),
                signed_transfer.from.to_string(),
                signed_transfer.to.to_string(),
                signed_transfer.amount as i64,
                signed_transfer.memo,
                signed_transfer.signature,
                signed_transfer.created_at.to_rfc3339(),
            ],
        )?;

        tx.commit()?;

        Ok(signed_transfer)
    }

    /// Get transfer history for a peer
    pub async fn get_transfers(
        &self,
        peer_id: &PeerId,
        limit: u32,
    ) -> Result<Vec<TokenTransfer>> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();

        let mut stmt = db.prepare(
            r#"
            SELECT id, from_peer, to_peer, amount, memo, signature, created_at
            FROM transfers
            WHERE from_peer = ?1 OR to_peer = ?1
            ORDER BY created_at DESC
            LIMIT ?2
            "#,
        )?;

        let transfers = stmt
            .query_map(params![peer_str, limit], |row| {
                Ok(TokenTransfer {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    from: row.get::<_, String>(1)?.parse().unwrap(),
                    to: row.get::<_, String>(2)?.parse().unwrap(),
                    amount: row.get::<_, i64>(3)? as u64,
                    memo: row.get(4)?,
                    signature: row.get(5)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(transfers)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn test_wallet_operations() {
        let wallet = Wallet::new(":memory:").await.unwrap();
        wallet.init_schema().await.unwrap();

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        // Test credit
        let balance = wallet.credit(&peer1, 1000).await.unwrap();
        assert_eq!(balance.available, 1000);
        assert_eq!(balance.total, 1000);

        // Test debit
        let balance = wallet.debit(&peer1, 200).await.unwrap();
        assert_eq!(balance.available, 800);

        // Test lock/unlock
        let balance = wallet.lock_tokens(&peer1, 300).await.unwrap();
        assert_eq!(balance.available, 500);
        assert_eq!(balance.locked, 300);

        let balance = wallet.unlock_tokens(&peer1, 300).await.unwrap();
        assert_eq!(balance.available, 800);
        assert_eq!(balance.locked, 0);

        // Test transfer
        let signing_key = SigningKey::generate(&mut OsRng);
        let transfer = wallet
            .transfer(&peer1, &peer2, 100, Some("test".to_string()), &signing_key)
            .await
            .unwrap();
        
        assert_eq!(transfer.amount, 100);
        assert_eq!(transfer.from, peer1);
        assert_eq!(transfer.to, peer2);

        // Verify balances after transfer
        let balance1 = wallet.get_balance(&peer1).await.unwrap();
        let balance2 = wallet.get_balance(&peer2).await.unwrap();
        assert_eq!(balance1.available, 700);
        assert_eq!(balance2.available, 100);
    }
}