//! Transaction ledger for recording all market activities

use crate::error::Result;
use chrono::{DateTime, Utc};
use libp2p::PeerId;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;
use uuid::Uuid;

/// Transaction type enumeration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TxType {
    /// Token transfer between wallets
    Transfer,
    /// Market order placed
    OrderPlaced,
    /// Market order cancelled
    OrderCancelled,
    /// Trade executed
    TradeExecuted,
    /// Escrow created
    EscrowCreated,
    /// Escrow released
    EscrowReleased,
    /// Escrow refunded
    EscrowRefunded,
    /// Reputation update
    ReputationUpdate,
    /// Fee collection
    FeeCollected,
}

/// Token transaction record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenTx {
    /// Unique transaction ID
    pub id: Uuid,
    /// Transaction type
    pub tx_type: TxType,
    /// Initiator peer ID
    pub from: PeerId,
    /// Target peer ID (if applicable)
    pub to: Option<PeerId>,
    /// Transaction amount
    pub amount: u64,
    /// Related entity ID (order, escrow, etc)
    pub reference_id: Option<Uuid>,
    /// Transaction metadata
    pub metadata: serde_json::Value,
    /// Transaction hash for integrity
    pub hash: String,
    /// Previous transaction hash (for chain)
    pub prev_hash: Option<String>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
}

impl TokenTx {
    /// Calculate transaction hash
    pub fn calculate_hash(&self) -> String {
        let mut hasher = Sha256::new();
        hasher.update(self.id.as_bytes());
        hasher.update(format!("{:?}", self.tx_type).as_bytes());
        hasher.update(self.from.to_bytes());
        if let Some(to) = &self.to {
            hasher.update(to.to_bytes());
        }
        hasher.update(self.amount.to_le_bytes());
        if let Some(ref_id) = &self.reference_id {
            hasher.update(ref_id.as_bytes());
        }
        hasher.update(self.metadata.to_string().as_bytes());
        if let Some(prev) = &self.prev_hash {
            hasher.update(prev.as_bytes());
        }
        hex::encode(hasher.finalize())
    }

    /// Verify transaction hash
    pub fn verify_hash(&self) -> bool {
        self.hash == self.calculate_hash()
    }
}

/// Transaction ledger for immutable record keeping
pub struct Ledger {
    db: Mutex<Connection>,
}

impl Ledger {
    /// Create a new ledger instance
    pub async fn new(db_path: &str) -> Result<Self> {
        let db = Connection::open(db_path)?;
        Ok(Self {
            db: Mutex::new(db),
        })
    }

    /// Initialize database schema
    pub async fn init_schema(&self) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS transactions (
                id TEXT PRIMARY KEY,
                tx_type TEXT NOT NULL,
                from_peer TEXT NOT NULL,
                to_peer TEXT,
                amount INTEGER NOT NULL,
                reference_id TEXT,
                metadata TEXT NOT NULL,
                hash TEXT NOT NULL UNIQUE,
                prev_hash TEXT,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions(from_peer);
            CREATE INDEX IF NOT EXISTS idx_tx_to ON transactions(to_peer);
            CREATE INDEX IF NOT EXISTS idx_tx_type ON transactions(tx_type);
            CREATE INDEX IF NOT EXISTS idx_tx_reference ON transactions(reference_id);
            CREATE INDEX IF NOT EXISTS idx_tx_created ON transactions(created_at);
            CREATE INDEX IF NOT EXISTS idx_tx_hash ON transactions(hash);
            "#,
            [],
        )?;

        Ok(())
    }

    /// Record a new transaction
    pub async fn record_transaction(
        &self,
        tx_type: TxType,
        from: PeerId,
        to: Option<PeerId>,
        amount: u64,
        reference_id: Option<Uuid>,
        metadata: serde_json::Value,
    ) -> Result<TokenTx> {
        let db = self.db.lock().await;
        
        // Get the latest transaction hash for chaining
        let prev_hash: Option<String> = db
            .query_row(
                "SELECT hash FROM transactions ORDER BY created_at DESC LIMIT 1",
                [],
                |row| row.get(0),
            )
            .ok();

        let mut tx = TokenTx {
            id: Uuid::new_v4(),
            tx_type: tx_type.clone(),
            from,
            to,
            amount,
            reference_id,
            metadata,
            hash: String::new(),
            prev_hash,
            created_at: Utc::now(),
        };

        // Calculate hash
        tx.hash = tx.calculate_hash();

        // Insert transaction
        db.execute(
            r#"
            INSERT INTO transactions (
                id, tx_type, from_peer, to_peer, amount, 
                reference_id, metadata, hash, prev_hash, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                tx.id.to_string(),
                format!("{:?}", tx.tx_type),
                tx.from.to_string(),
                tx.to.map(|p| p.to_string()),
                tx.amount as i64,
                tx.reference_id.map(|id| id.to_string()),
                tx.metadata.to_string(),
                tx.hash,
                tx.prev_hash,
                tx.created_at.to_rfc3339(),
            ],
        )?;

        Ok(tx)
    }

    /// Get transaction by ID
    pub async fn get_transaction(&self, id: &Uuid) -> Result<Option<TokenTx>> {
        let db = self.db.lock().await;
        
        let tx = db
            .query_row(
                r#"
                SELECT id, tx_type, from_peer, to_peer, amount,
                       reference_id, metadata, hash, prev_hash, created_at
                FROM transactions
                WHERE id = ?1
                "#,
                params![id.to_string()],
                |row| {
                    Ok(TokenTx {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                        tx_type: match row.get::<_, String>(1)?.as_str() {
                            "Transfer" => TxType::Transfer,
                            "OrderPlaced" => TxType::OrderPlaced,
                            "OrderCancelled" => TxType::OrderCancelled,
                            "TradeExecuted" => TxType::TradeExecuted,
                            "EscrowCreated" => TxType::EscrowCreated,
                            "EscrowReleased" => TxType::EscrowReleased,
                            "EscrowRefunded" => TxType::EscrowRefunded,
                            "ReputationUpdate" => TxType::ReputationUpdate,
                            "FeeCollected" => TxType::FeeCollected,
                            _ => TxType::Transfer,
                        },
                        from: row.get::<_, String>(2)?.parse().unwrap(),
                        to: row.get::<_, Option<String>>(3)?.map(|s| s.parse().unwrap()),
                        amount: row.get::<_, i64>(4)? as u64,
                        reference_id: row.get::<_, Option<String>>(5)?
                            .map(|s| Uuid::parse_str(&s).unwrap()),
                        metadata: serde_json::from_str(&row.get::<_, String>(6)?).unwrap(),
                        hash: row.get(7)?,
                        prev_hash: row.get(8)?,
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                            .unwrap()
                            .with_timezone(&Utc),
                    })
                },
            )
            .ok();

        Ok(tx)
    }

    /// Get transactions for a peer
    pub async fn get_peer_transactions(
        &self,
        peer_id: &PeerId,
        tx_type: Option<TxType>,
        limit: u32,
    ) -> Result<Vec<TokenTx>> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();

        if let Some(tx_type) = tx_type {
            let tx_type_str = format!("{:?}", tx_type);
            let query = r#"
                SELECT id, tx_type, from_peer, to_peer, amount,
                       reference_id, metadata, hash, prev_hash, created_at
                FROM transactions
                WHERE (from_peer = ?1 OR to_peer = ?1) AND tx_type = ?2
                ORDER BY created_at DESC
                LIMIT ?3
                "#;
            let mut stmt = db.prepare(query)?;
            let transactions = stmt.query_map([&peer_str, &tx_type_str, &(limit as i64).to_string()], |row| {
                Ok(TokenTx {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    tx_type: match row.get::<_, String>(1)?.as_str() {
                        "Transfer" => TxType::Transfer,
                        "OrderPlaced" => TxType::OrderPlaced,
                        "OrderCancelled" => TxType::OrderCancelled,
                        "TradeExecuted" => TxType::TradeExecuted,
                        "EscrowCreated" => TxType::EscrowCreated,
                        "EscrowReleased" => TxType::EscrowReleased,
                        "EscrowRefunded" => TxType::EscrowRefunded,
                        "ReputationUpdate" => TxType::ReputationUpdate,
                        "FeeCollected" => TxType::FeeCollected,
                        _ => TxType::Transfer,
                    },
                    from: row.get::<_, String>(2)?.parse().unwrap(),
                    to: row.get::<_, Option<String>>(3)?.map(|s| s.parse().unwrap()),
                    amount: row.get::<_, i64>(4)? as u64,
                    reference_id: row.get::<_, Option<String>>(5)?
                        .map(|s| Uuid::parse_str(&s).unwrap()),
                    metadata: serde_json::from_str(&row.get::<_, String>(6)?).unwrap(),
                    hash: row.get(7)?,
                    prev_hash: row.get(8)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            })?.collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(transactions)
        } else {
            let query = r#"
                SELECT id, tx_type, from_peer, to_peer, amount,
                       reference_id, metadata, hash, prev_hash, created_at
                FROM transactions
                WHERE from_peer = ?1 OR to_peer = ?1
                ORDER BY created_at DESC
                LIMIT ?2
                "#;
            let mut stmt = db.prepare(query)?;
            let transactions = stmt.query_map([&peer_str, &(limit as i64).to_string()], |row| {
                Ok(TokenTx {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    tx_type: match row.get::<_, String>(1)?.as_str() {
                        "Transfer" => TxType::Transfer,
                        "OrderPlaced" => TxType::OrderPlaced,
                        "OrderCancelled" => TxType::OrderCancelled,
                        "TradeExecuted" => TxType::TradeExecuted,
                        "EscrowCreated" => TxType::EscrowCreated,
                        "EscrowReleased" => TxType::EscrowReleased,
                        "EscrowRefunded" => TxType::EscrowRefunded,
                        "ReputationUpdate" => TxType::ReputationUpdate,
                        "FeeCollected" => TxType::FeeCollected,
                        _ => TxType::Transfer,
                    },
                    from: row.get::<_, String>(2)?.parse().unwrap(),
                    to: row.get::<_, Option<String>>(3)?.map(|s| s.parse().unwrap()),
                    amount: row.get::<_, i64>(4)? as u64,
                    reference_id: row.get::<_, Option<String>>(5)?
                        .map(|s| Uuid::parse_str(&s).unwrap()),
                    metadata: serde_json::from_str(&row.get::<_, String>(6)?).unwrap(),
                    hash: row.get(7)?,
                    prev_hash: row.get(8)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            })?.collect::<std::result::Result<Vec<_>, _>>()?;
            Ok(transactions)
        }
    }

    /// Verify ledger integrity by checking hash chain
    pub async fn verify_integrity(&self) -> Result<bool> {
        let db = self.db.lock().await;
        
        let mut stmt = db.prepare(
            r#"
            SELECT id, tx_type, from_peer, to_peer, amount,
                   reference_id, metadata, hash, prev_hash, created_at
            FROM transactions
            ORDER BY created_at ASC
            "#
        )?;

        let transactions = stmt
            .query_map([], |row| {
                Ok(TokenTx {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    tx_type: match row.get::<_, String>(1)?.as_str() {
                        "Transfer" => TxType::Transfer,
                        "OrderPlaced" => TxType::OrderPlaced,
                        "OrderCancelled" => TxType::OrderCancelled,
                        "TradeExecuted" => TxType::TradeExecuted,
                        "EscrowCreated" => TxType::EscrowCreated,
                        "EscrowReleased" => TxType::EscrowReleased,
                        "EscrowRefunded" => TxType::EscrowRefunded,
                        "ReputationUpdate" => TxType::ReputationUpdate,
                        "FeeCollected" => TxType::FeeCollected,
                        _ => TxType::Transfer,
                    },
                    from: row.get::<_, String>(2)?.parse().unwrap(),
                    to: row.get::<_, Option<String>>(3)?.map(|s| s.parse().unwrap()),
                    amount: row.get::<_, i64>(4)? as u64,
                    reference_id: row.get::<_, Option<String>>(5)?
                        .map(|s| Uuid::parse_str(&s).unwrap()),
                    metadata: serde_json::from_str(&row.get::<_, String>(6)?).unwrap(),
                    hash: row.get(7)?,
                    prev_hash: row.get(8)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut prev_hash: Option<String> = None;
        
        for tx in transactions {
            // Verify hash
            if !tx.verify_hash() {
                return Ok(false);
            }
            
            // Verify chain
            if tx.prev_hash != prev_hash {
                return Ok(false);
            }
            
            prev_hash = Some(tx.hash.clone());
        }

        Ok(true)
    }

    /// Get transaction statistics
    pub async fn get_statistics(&self) -> Result<serde_json::Value> {
        let db = self.db.lock().await;
        
        let total_count: i64 = db.query_row(
            "SELECT COUNT(*) FROM transactions",
            [],
            |row| row.get(0),
        )?;

        let total_volume: i64 = db.query_row(
            "SELECT COALESCE(SUM(amount), 0) FROM transactions",
            [],
            |row| row.get(0),
        )?;

        let tx_by_type: Vec<(String, i64)> = db
            .prepare("SELECT tx_type, COUNT(*) FROM transactions GROUP BY tx_type")?
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(serde_json::json!({
            "total_transactions": total_count,
            "total_volume": total_volume,
            "transactions_by_type": tx_by_type.into_iter()
                .map(|(k, v)| (k, v))
                .collect::<std::collections::HashMap<_, _>>(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ledger_operations() {
        let ledger = Ledger::new(":memory:").await.unwrap();
        ledger.init_schema().await.unwrap();

        let peer1 = PeerId::random();
        let peer2 = PeerId::random();

        // Record transactions
        let tx1 = ledger
            .record_transaction(
                TxType::Transfer,
                peer1,
                Some(peer2),
                100,
                None,
                serde_json::json!({"memo": "test transfer"}),
            )
            .await
            .unwrap();

        assert!(tx1.verify_hash());
        assert!(tx1.prev_hash.is_none());

        let tx2 = ledger
            .record_transaction(
                TxType::OrderPlaced,
                peer1,
                None,
                50,
                Some(Uuid::new_v4()),
                serde_json::json!({"order_type": "buy"}),
            )
            .await
            .unwrap();

        assert!(tx2.verify_hash());
        assert_eq!(tx2.prev_hash, Some(tx1.hash.clone()));

        // Verify integrity
        assert!(ledger.verify_integrity().await.unwrap());

        // Get transactions
        let peer_txs = ledger
            .get_peer_transactions(&peer1, None, 10)
            .await
            .unwrap();
        assert_eq!(peer_txs.len(), 2);

        // Get statistics
        let stats = ledger.get_statistics().await.unwrap();
        assert_eq!(stats["total_transactions"], 2);
        assert_eq!(stats["total_volume"], 150);
    }
}