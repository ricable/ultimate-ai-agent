//! Escrow service for secure token trading with multi-signature support
//! 
//! This module provides a complete escrow management system for the Synaptic Market,
//! ensuring secure transactions for compute contribution rewards (not Claude access).
//! All operations are auditable and maintain full user control and transparency.

use crate::error::{MarketError, Result};
use crate::wallet::Wallet;
use chrono::{DateTime, Duration, Utc};
use ed25519_dalek::{Signer, SigningKey, VerifyingKey};
use libp2p::PeerId;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Escrow state machine for compute contribution rewards
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EscrowState {
    /// Escrow created, waiting for funding
    Created,
    /// Escrow is funded and active
    Funded,
    /// Compute task completed, awaiting confirmation
    Completed,
    /// Funds released to provider
    Released,
    /// Funds refunded to requester
    Refunded,
    /// Escrow is under dispute
    Disputed,
    /// Dispute resolved by arbitrator
    Resolved,
    /// Escrow expired without action
    Expired,
}

/// Multi-signature requirement type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MultiSigType {
    /// Single signature required (default)
    Single,
    /// Both parties must sign (2-of-2)
    BothParties,
    /// Majority of arbitrators (M-of-N)
    Arbitrators { required: u8, total: u8 },
    /// Time-locked automatic release
    TimeLocked { release_after: DateTime<Utc> },
}

/// Escrow agreement for compute contribution rewards
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscrowAgreement {
    /// Unique escrow ID
    pub id: Uuid,
    /// Compute job ID this escrow is for
    pub job_id: Uuid,
    /// Requester peer ID (who needs compute)
    pub requester: PeerId,
    /// Provider peer ID (who provides compute)
    pub provider: PeerId,
    /// Reward amount in ruv tokens
    pub amount: u64,
    /// Current state
    pub state: EscrowState,
    /// Multi-signature requirement
    pub multisig_type: MultiSigType,
    /// Arbitrators for dispute resolution
    pub arbitrators: Vec<PeerId>,
    /// Timeout for automatic release/refund
    pub timeout_at: DateTime<Utc>,
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    /// Last update timestamp
    pub updated_at: DateTime<Utc>,
    /// Collected signatures
    pub signatures: HashMap<PeerId, Vec<u8>>,
    /// Dispute details if any
    pub dispute_info: Option<DisputeInfo>,
    /// Audit log of all state transitions
    pub audit_log: Vec<AuditEntry>,
}

/// Dispute information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisputeInfo {
    /// Who raised the dispute
    pub raised_by: PeerId,
    /// Dispute reason
    pub reason: String,
    /// Supporting evidence CIDs
    pub evidence: Vec<String>,
    /// Arbitrator decision
    pub decision: Option<ArbitratorDecision>,
    /// Dispute raised timestamp
    pub raised_at: DateTime<Utc>,
}

/// Arbitrator decision for dispute resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitratorDecision {
    /// Arbitrator who made the decision
    pub arbitrator: PeerId,
    /// Decision: release to provider, refund to requester, or split
    pub outcome: DisputeOutcome,
    /// Reasoning for the decision
    pub reasoning: String,
    /// Decision timestamp
    pub decided_at: DateTime<Utc>,
}

/// Dispute resolution outcome
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DisputeOutcome {
    /// Release full amount to provider
    ReleaseToProvider,
    /// Refund full amount to requester
    RefundToRequester,
    /// Split between parties
    Split { provider_share: u64, requester_share: u64 },
}

/// Audit entry for tracking all escrow operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Timestamp of the operation
    pub timestamp: DateTime<Utc>,
    /// Who performed the operation
    pub actor: PeerId,
    /// What operation was performed
    pub action: String,
    /// Previous state
    pub from_state: EscrowState,
    /// New state
    pub to_state: EscrowState,
    /// Additional details
    pub details: Option<String>,
}

/// Escrow release authorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReleaseAuth {
    /// Escrow ID
    pub escrow_id: Uuid,
    /// Authorizing party
    pub authorizer: PeerId,
    /// Release decision
    pub decision: ReleaseDecision,
    /// Authorization signature
    pub signature: Vec<u8>,
    /// Timestamp
    pub created_at: DateTime<Utc>,
}

/// Release decision types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ReleaseDecision {
    /// Approve release to provider
    ApproveRelease,
    /// Request refund to requester
    RequestRefund,
    /// Raise dispute
    RaiseDispute,
}

/// Escrow service with wallet integration
pub struct Escrow {
    db: Arc<Mutex<Connection>>,
    wallet: Arc<Wallet>,
    verifying_keys: RwLock<HashMap<PeerId, VerifyingKey>>,
}

impl Escrow {
    /// Create a new escrow service with wallet integration
    pub async fn new(db_path: &str, wallet: Arc<Wallet>) -> Result<Self> {
        let db = Connection::open(db_path)?;
        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            wallet,
            verifying_keys: RwLock::new(HashMap::new()),
        })
    }

    /// Initialize database schema
    pub async fn init_schema(&self) -> Result<()> {
        let db = self.db.lock().await;
        
        // Main escrows table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS escrows (
                id TEXT PRIMARY KEY,
                job_id TEXT NOT NULL,
                requester TEXT NOT NULL,
                provider TEXT NOT NULL,
                amount INTEGER NOT NULL,
                state TEXT NOT NULL,
                multisig_type TEXT NOT NULL,
                arbitrators TEXT NOT NULL,
                timeout_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                dispute_info TEXT,
                audit_log TEXT NOT NULL
            )
            "#,
            [],
        )?;

        // Signatures table for multi-sig support
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS escrow_signatures (
                id TEXT PRIMARY KEY,
                escrow_id TEXT NOT NULL,
                signer TEXT NOT NULL,
                signature BLOB NOT NULL,
                signed_at TEXT NOT NULL,
                FOREIGN KEY (escrow_id) REFERENCES escrows(id),
                UNIQUE(escrow_id, signer)
            )
            "#,
            [],
        )?;

        // Release authorizations table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS release_authorizations (
                id TEXT PRIMARY KEY,
                escrow_id TEXT NOT NULL,
                authorizer TEXT NOT NULL,
                decision TEXT NOT NULL,
                signature BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (escrow_id) REFERENCES escrows(id)
            )
            "#,
            [],
        )?;

        // Arbitrator decisions table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS arbitrator_decisions (
                id TEXT PRIMARY KEY,
                escrow_id TEXT NOT NULL,
                arbitrator TEXT NOT NULL,
                outcome TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                decided_at TEXT NOT NULL,
                FOREIGN KEY (escrow_id) REFERENCES escrows(id)
            )
            "#,
            [],
        )?;

        // Create indexes
        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_escrows_job ON escrows(job_id);
            CREATE INDEX IF NOT EXISTS idx_escrows_requester ON escrows(requester);
            CREATE INDEX IF NOT EXISTS idx_escrows_provider ON escrows(provider);
            CREATE INDEX IF NOT EXISTS idx_escrows_state ON escrows(state);
            CREATE INDEX IF NOT EXISTS idx_escrows_timeout ON escrows(timeout_at);
            CREATE INDEX IF NOT EXISTS idx_signatures_escrow ON escrow_signatures(escrow_id);
            CREATE INDEX IF NOT EXISTS idx_release_escrow ON release_authorizations(escrow_id);
            CREATE INDEX IF NOT EXISTS idx_arbitrator_escrow ON arbitrator_decisions(escrow_id);
            "#,
            [],
        )?;

        Ok(())
    }

    /// Create a new escrow agreement for compute contribution rewards
    pub async fn create_escrow(
        &self,
        job_id: Uuid,
        requester: PeerId,
        provider: PeerId,
        amount: u64,
        multisig_type: MultiSigType,
        arbitrators: Vec<PeerId>,
        timeout_minutes: i64,
    ) -> Result<EscrowAgreement> {
        // Validate inputs
        if requester == provider {
            return Err(MarketError::Escrow("Cannot create escrow with same party".to_string()));
        }

        if amount == 0 {
            return Err(MarketError::Escrow("Escrow amount must be non-zero".to_string()));
        }

        // Validate arbitrators for multi-sig
        match multisig_type {
            MultiSigType::Arbitrators { required, total } => {
                if arbitrators.len() != total as usize {
                    return Err(MarketError::Escrow(
                        format!("Expected {} arbitrators, got {}", total, arbitrators.len())
                    ));
                }
                if required > total || required == 0 {
                    return Err(MarketError::Escrow("Invalid arbitrator requirements".to_string()));
                }
            }
            _ => {}
        }

        let now = Utc::now();
        let initial_audit = AuditEntry {
            timestamp: now,
            actor: requester,
            action: "create_escrow".to_string(),
            from_state: EscrowState::Created,
            to_state: EscrowState::Created,
            details: Some(format!("Amount: {} ruv, Provider: {}", amount, provider)),
        };

        let escrow = EscrowAgreement {
            id: Uuid::new_v4(),
            job_id,
            requester,
            provider,
            amount,
            state: EscrowState::Created,
            multisig_type,
            arbitrators: arbitrators.clone(),
            timeout_at: now + Duration::minutes(timeout_minutes),
            created_at: now,
            updated_at: now,
            signatures: HashMap::new(),
            dispute_info: None,
            audit_log: vec![initial_audit],
        };

        // Store in database
        let db = self.db.lock().await;
        db.execute(
            r#"
            INSERT INTO escrows (
                id, job_id, requester, provider, amount, state, multisig_type,
                arbitrators, timeout_at, created_at, updated_at, audit_log
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            params![
                escrow.id.to_string(),
                escrow.job_id.to_string(),
                escrow.requester.to_string(),
                escrow.provider.to_string(),
                escrow.amount as i64,
                format!("{:?}", escrow.state),
                serde_json::to_string(&escrow.multisig_type)?,
                serde_json::to_string(&arbitrators)?,
                escrow.timeout_at.to_rfc3339(),
                escrow.created_at.to_rfc3339(),
                escrow.updated_at.to_rfc3339(),
                serde_json::to_string(&escrow.audit_log)?,
            ],
        )?;

        Ok(escrow)
    }

    /// Fund the escrow (requester locks tokens)
    pub async fn fund_escrow(
        &self,
        escrow_id: &Uuid,
        requester: &PeerId,
        signing_key: &SigningKey,
    ) -> Result<EscrowAgreement> {
        let mut escrow = self.get_escrow(escrow_id).await?
            .ok_or_else(|| MarketError::Escrow("Escrow not found".to_string()))?;

        // Validate state and requester
        if escrow.state != EscrowState::Created {
            return Err(MarketError::Escrow("Escrow already funded or in invalid state".to_string()));
        }

        if escrow.requester != *requester {
            return Err(MarketError::Escrow("Only requester can fund escrow".to_string()));
        }

        // Lock tokens in wallet
        self.wallet.lock_tokens(requester, escrow.amount).await?;

        // Create and store signature
        let message = format!("FUND:{}:{}:{}", escrow.id, requester, escrow.amount);
        let signature = signing_key.sign(message.as_bytes());
        
        // Update state
        escrow.state = EscrowState::Funded;
        escrow.updated_at = Utc::now();
        escrow.signatures.insert(*requester, signature.to_bytes().to_vec());
        
        // Add audit entry
        escrow.audit_log.push(AuditEntry {
            timestamp: escrow.updated_at,
            actor: *requester,
            action: "fund_escrow".to_string(),
            from_state: EscrowState::Created,
            to_state: EscrowState::Funded,
            details: Some(format!("Locked {} ruv tokens", escrow.amount)),
        });

        // Update database
        self.update_escrow_state(&escrow).await?;
        
        // Store signature
        self.store_signature(&escrow.id, requester, &signature.to_bytes().to_vec()).await?;

        Ok(escrow)
    }

    /// Mark job as completed by provider
    pub async fn mark_completed(
        &self,
        escrow_id: &Uuid,
        provider: &PeerId,
        signing_key: &SigningKey,
    ) -> Result<EscrowAgreement> {
        let mut escrow = self.get_escrow(escrow_id).await?
            .ok_or_else(|| MarketError::Escrow("Escrow not found".to_string()))?;

        // Validate state and provider
        if escrow.state != EscrowState::Funded {
            return Err(MarketError::Escrow("Escrow not in funded state".to_string()));
        }

        if escrow.provider != *provider {
            return Err(MarketError::Escrow("Only provider can mark job completed".to_string()));
        }

        // Create and store signature
        let message = format!("COMPLETE:{}:{}", escrow.id, provider);
        let signature = signing_key.sign(message.as_bytes());
        
        // Update state
        escrow.state = EscrowState::Completed;
        escrow.updated_at = Utc::now();
        escrow.signatures.insert(*provider, signature.to_bytes().to_vec());
        
        // Add audit entry
        escrow.audit_log.push(AuditEntry {
            timestamp: escrow.updated_at,
            actor: *provider,
            action: "mark_completed".to_string(),
            from_state: EscrowState::Funded,
            to_state: EscrowState::Completed,
            details: Some("Job execution completed".to_string()),
        });

        // Update database
        self.update_escrow_state(&escrow).await?;
        self.store_signature(&escrow.id, provider, &signature.to_bytes().to_vec()).await?;

        Ok(escrow)
    }

    /// Release funds based on multi-signature requirements
    pub async fn release_funds(
        &self,
        escrow_id: &Uuid,
        release_auth: ReleaseAuth,
    ) -> Result<EscrowAgreement> {
        let mut escrow = self.get_escrow(escrow_id).await?
            .ok_or_else(|| MarketError::Escrow("Escrow not found".to_string()))?;

        // Validate escrow state
        match escrow.state {
            EscrowState::Completed | EscrowState::Disputed => {},
            _ => return Err(MarketError::Escrow("Escrow not ready for release".to_string())),
        }

        // Verify release authorization
        self.verify_release_auth(&escrow, &release_auth)?;

        // Check multi-sig requirements
        let can_release = self.check_multisig_requirements(&escrow, &release_auth).await?;
        
        if !can_release {
            // Store the authorization for later
            self.store_release_auth(&release_auth).await?;
            return Ok(escrow);
        }

        // Process based on decision
        match release_auth.decision {
            ReleaseDecision::ApproveRelease => {
                // Unlock tokens from requester and transfer to provider
                self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                self.wallet.transfer(
                    &escrow.requester,
                    &escrow.provider,
                    escrow.amount,
                    Some(format!("Escrow release for job {}", escrow.job_id)),
                    &SigningKey::from_bytes(&[0; 32]), // System key for internal transfers
                ).await?;
                
                escrow.state = EscrowState::Released;
            }
            ReleaseDecision::RequestRefund => {
                // Unlock tokens back to requester
                self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                escrow.state = EscrowState::Refunded;
            }
            ReleaseDecision::RaiseDispute => {
                self.raise_dispute_internal(&mut escrow, &release_auth).await?;
                return Ok(escrow);
            }
        }

        // Add audit entry
        escrow.audit_log.push(AuditEntry {
            timestamp: Utc::now(),
            actor: release_auth.authorizer,
            action: format!("release_funds_{:?}", release_auth.decision),
            from_state: EscrowState::Completed,
            to_state: escrow.state,
            details: Some(format!("Decision: {:?}", release_auth.decision)),
        });

        escrow.updated_at = Utc::now();
        self.update_escrow_state(&escrow).await?;

        Ok(escrow)
    }

    /// Raise a dispute on the escrow
    pub async fn raise_dispute(
        &self,
        escrow_id: &Uuid,
        raised_by: &PeerId,
        reason: String,
        evidence: Vec<String>,
        signing_key: &SigningKey,
    ) -> Result<EscrowAgreement> {
        let mut escrow = self.get_escrow(escrow_id).await?
            .ok_or_else(|| MarketError::Escrow("Escrow not found".to_string()))?;

        // Validate state
        if escrow.state != EscrowState::Completed && escrow.state != EscrowState::Funded {
            return Err(MarketError::Escrow("Cannot dispute escrow in current state".to_string()));
        }

        // Verify disputer is party to escrow
        if escrow.requester != *raised_by && escrow.provider != *raised_by {
            return Err(MarketError::Escrow("Only parties to escrow can raise disputes".to_string()));
        }

        // Create dispute info
        escrow.dispute_info = Some(DisputeInfo {
            raised_by: *raised_by,
            reason: reason.clone(),
            evidence,
            decision: None,
            raised_at: Utc::now(),
        });

        // Sign the dispute
        let message = format!("DISPUTE:{}:{}:{}", escrow.id, raised_by, reason);
        let signature = signing_key.sign(message.as_bytes());
        escrow.signatures.insert(*raised_by, signature.to_bytes().to_vec());

        // Update state
        let old_state = escrow.state;
        escrow.state = EscrowState::Disputed;
        escrow.updated_at = Utc::now();

        // Add audit entry
        escrow.audit_log.push(AuditEntry {
            timestamp: escrow.updated_at,
            actor: *raised_by,
            action: "raise_dispute".to_string(),
            from_state: old_state,
            to_state: EscrowState::Disputed,
            details: Some(reason),
        });

        self.update_escrow_state(&escrow).await?;
        self.store_signature(&escrow.id, raised_by, &signature.to_bytes().to_vec()).await?;

        Ok(escrow)
    }

    /// Resolve a dispute with arbitrator decision
    pub async fn resolve_dispute(
        &self,
        escrow_id: &Uuid,
        arbitrator: &PeerId,
        outcome: DisputeOutcome,
        reasoning: String,
        signing_key: &SigningKey,
    ) -> Result<EscrowAgreement> {
        let mut escrow = self.get_escrow(escrow_id).await?
            .ok_or_else(|| MarketError::Escrow("Escrow not found".to_string()))?;

        // Validate state
        if escrow.state != EscrowState::Disputed {
            return Err(MarketError::Escrow("Escrow not in disputed state".to_string()));
        }

        // Verify arbitrator is authorized
        if !escrow.arbitrators.contains(arbitrator) {
            return Err(MarketError::Escrow("Not an authorized arbitrator".to_string()));
        }

        // Apply decision
        let decision = ArbitratorDecision {
            arbitrator: *arbitrator,
            outcome,
            reasoning: reasoning.clone(),
            decided_at: Utc::now(),
        };

        if let Some(ref mut dispute) = escrow.dispute_info {
            dispute.decision = Some(decision.clone());
        }

        // Process based on outcome
        match outcome {
            DisputeOutcome::ReleaseToProvider => {
                self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                self.wallet.transfer(
                    &escrow.requester,
                    &escrow.provider,
                    escrow.amount,
                    Some(format!("Dispute resolution for job {}", escrow.job_id)),
                    &SigningKey::from_bytes(&[0; 32]),
                ).await?;
                escrow.state = EscrowState::Released;
            }
            DisputeOutcome::RefundToRequester => {
                self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                escrow.state = EscrowState::Refunded;
            }
            DisputeOutcome::Split { provider_share, requester_share } => {
                if provider_share + requester_share != escrow.amount {
                    return Err(MarketError::Escrow("Split amounts don't match escrow amount".to_string()));
                }
                
                self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                
                if provider_share > 0 {
                    self.wallet.transfer(
                        &escrow.requester,
                        &escrow.provider,
                        provider_share,
                        Some(format!("Dispute split for job {}", escrow.job_id)),
                        &SigningKey::from_bytes(&[0; 32]),
                    ).await?;
                }
                
                escrow.state = EscrowState::Resolved;
            }
        }

        // Sign decision
        let message = format!("RESOLVE:{}:{}:{:?}", escrow.id, arbitrator, outcome);
        let signature = signing_key.sign(message.as_bytes());
        escrow.signatures.insert(*arbitrator, signature.to_bytes().to_vec());

        // Add audit entry
        escrow.audit_log.push(AuditEntry {
            timestamp: Utc::now(),
            actor: *arbitrator,
            action: "resolve_dispute".to_string(),
            from_state: EscrowState::Disputed,
            to_state: escrow.state,
            details: Some(reasoning),
        });

        escrow.updated_at = Utc::now();
        self.update_escrow_state(&escrow).await?;
        
        // Store arbitrator decision
        self.store_arbitrator_decision(&escrow.id, &decision).await?;

        Ok(escrow)
    }

    /// Check for expired escrows and process automatic actions
    pub async fn process_timeouts(&self) -> Result<Vec<Uuid>> {
        let db = self.db.lock().await;
        let now = Utc::now().to_rfc3339();

        // Find expired escrows
        let mut stmt = db.prepare(
            r#"
            SELECT id FROM escrows 
            WHERE state IN ('Created', 'Funded', 'Completed') 
            AND timeout_at < ?1
            "#,
        )?;

        let expired_ids: Vec<String> = stmt
            .query_map(params![now], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        drop(stmt);
        drop(db);

        let mut processed = Vec::new();

        for id_str in expired_ids {
            let id = Uuid::parse_str(&id_str).unwrap();
            if let Ok(escrow) = self.get_escrow(&id).await {
                if let Some(mut escrow) = escrow {
                    match escrow.state {
                        EscrowState::Created => {
                            // Never funded, just expire
                            escrow.state = EscrowState::Expired;
                        }
                        EscrowState::Funded | EscrowState::Completed => {
                            // Check multisig type for automatic action
                            match escrow.multisig_type {
                                MultiSigType::TimeLocked { release_after } => {
                                    if Utc::now() >= release_after {
                                        // Automatic release to provider
                                        self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                                        self.wallet.transfer(
                                            &escrow.requester,
                                            &escrow.provider,
                                            escrow.amount,
                                            Some(format!("Time-locked release for job {}", escrow.job_id)),
                                            &SigningKey::from_bytes(&[0; 32]),
                                        ).await?;
                                        escrow.state = EscrowState::Released;
                                    } else {
                                        // Refund to requester
                                        self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                                        escrow.state = EscrowState::Refunded;
                                    }
                                }
                                _ => {
                                    // Default: refund to requester on timeout
                                    self.wallet.unlock_tokens(&escrow.requester, escrow.amount).await?;
                                    escrow.state = EscrowState::Refunded;
                                }
                            }
                        }
                        _ => continue,
                    }

                    escrow.audit_log.push(AuditEntry {
                        timestamp: Utc::now(),
                        actor: PeerId::random(), // System action
                        action: "timeout_processing".to_string(),
                        from_state: escrow.state,
                        to_state: escrow.state,
                        details: Some("Automatic timeout processing".to_string()),
                    });

                    escrow.updated_at = Utc::now();
                    self.update_escrow_state(&escrow).await?;
                    processed.push(id);
                }
            }
        }

        Ok(processed)
    }

    /// Get escrow by ID with full details
    pub async fn get_escrow(&self, escrow_id: &Uuid) -> Result<Option<EscrowAgreement>> {
        let db = self.db.lock().await;
        
        let escrow_opt = db
            .query_row(
                r#"
                SELECT id, job_id, requester, provider, amount, state, multisig_type,
                       arbitrators, timeout_at, created_at, updated_at, dispute_info, audit_log
                FROM escrows
                WHERE id = ?1
                "#,
                params![escrow_id.to_string()],
                |row| {
                    let escrow = EscrowAgreement {
                        id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                        job_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap(),
                        requester: row.get::<_, String>(2)?.parse().unwrap(),
                        provider: row.get::<_, String>(3)?.parse().unwrap(),
                        amount: row.get::<_, i64>(4)? as u64,
                        state: self.parse_state(&row.get::<_, String>(5)?),
                        multisig_type: serde_json::from_str(&row.get::<_, String>(6)?).unwrap(),
                        arbitrators: serde_json::from_str(&row.get::<_, String>(7)?).unwrap(),
                        timeout_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                            .unwrap()
                            .with_timezone(&Utc),
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(9)?)
                            .unwrap()
                            .with_timezone(&Utc),
                        updated_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(10)?)
                            .unwrap()
                            .with_timezone(&Utc),
                        dispute_info: row.get::<_, Option<String>>(11)?
                            .and_then(|s| serde_json::from_str(&s).ok()),
                        audit_log: serde_json::from_str(&row.get::<_, String>(12)?).unwrap(),
                        signatures: HashMap::new(),
                    };
                    Ok(escrow)
                },
            )
            .optional()?;

        if let Some(mut escrow) = escrow_opt {
            // Load signatures
            let mut stmt = db.prepare(
                "SELECT signer, signature FROM escrow_signatures WHERE escrow_id = ?1"
            )?;
            
            let signatures = stmt.query_map(params![escrow_id.to_string()], |row| {
                let signer: String = row.get(0)?;
                let signature: Vec<u8> = row.get(1)?;
                Ok((signer.parse().unwrap(), signature))
            })?;

            for sig_result in signatures {
                let (signer, signature) = sig_result?;
                escrow.signatures.insert(signer, signature);
            }

            Ok(Some(escrow))
        } else {
            Ok(None)
        }
    }

    /// Get escrows for a peer
    pub async fn get_peer_escrows(
        &self,
        peer_id: &PeerId,
        state: Option<EscrowState>,
    ) -> Result<Vec<EscrowAgreement>> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();

        let mut query = String::from(
            r#"
            SELECT id FROM escrows
            WHERE (requester = ?1 OR provider = ?1)
            "#
        );

        if let Some(state) = state {
            query.push_str(&format!(" AND state = '{:?}'", state));
        }

        query.push_str(" ORDER BY created_at DESC");

        let mut stmt = db.prepare(&query)?;
        let escrow_ids: Vec<String> = stmt
            .query_map(params![peer_str], |row| row.get(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        drop(stmt);
        drop(db);

        let mut escrows = Vec::new();
        for id_str in escrow_ids {
            let id = Uuid::parse_str(&id_str).unwrap();
            if let Some(escrow) = self.get_escrow(&id).await? {
                escrows.push(escrow);
            }
        }

        Ok(escrows)
    }

    /// Get audit log for an escrow
    pub async fn get_audit_log(&self, escrow_id: &Uuid) -> Result<Vec<AuditEntry>> {
        let escrow = self.get_escrow(escrow_id).await?
            .ok_or_else(|| MarketError::Escrow("Escrow not found".to_string()))?;
        
        Ok(escrow.audit_log)
    }

    // Helper methods

    /// Parse state string to enum
    fn parse_state(&self, state_str: &str) -> EscrowState {
        match state_str {
            "Created" => EscrowState::Created,
            "Funded" => EscrowState::Funded,
            "Completed" => EscrowState::Completed,
            "Released" => EscrowState::Released,
            "Refunded" => EscrowState::Refunded,
            "Disputed" => EscrowState::Disputed,
            "Resolved" => EscrowState::Resolved,
            "Expired" => EscrowState::Expired,
            _ => EscrowState::Created,
        }
    }

    /// Update escrow state in database
    async fn update_escrow_state(&self, escrow: &EscrowAgreement) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            UPDATE escrows 
            SET state = ?2, updated_at = ?3, dispute_info = ?4, audit_log = ?5
            WHERE id = ?1
            "#,
            params![
                escrow.id.to_string(),
                format!("{:?}", escrow.state),
                escrow.updated_at.to_rfc3339(),
                escrow.dispute_info.as_ref()
                    .map(|d| serde_json::to_string(d).unwrap()),
                serde_json::to_string(&escrow.audit_log)?,
            ],
        )?;

        Ok(())
    }

    /// Store signature in database
    async fn store_signature(
        &self,
        escrow_id: &Uuid,
        signer: &PeerId,
        signature: &[u8],
    ) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            INSERT INTO escrow_signatures (id, escrow_id, signer, signature, signed_at)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(escrow_id, signer) DO UPDATE SET
                signature = ?4,
                signed_at = ?5
            "#,
            params![
                Uuid::new_v4().to_string(),
                escrow_id.to_string(),
                signer.to_string(),
                signature,
                Utc::now().to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Store release authorization
    async fn store_release_auth(&self, auth: &ReleaseAuth) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            INSERT INTO release_authorizations (
                id, escrow_id, authorizer, decision, signature, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                Uuid::new_v4().to_string(),
                auth.escrow_id.to_string(),
                auth.authorizer.to_string(),
                serde_json::to_string(&auth.decision)?,
                auth.signature,
                auth.created_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Store arbitrator decision
    async fn store_arbitrator_decision(
        &self,
        escrow_id: &Uuid,
        decision: &ArbitratorDecision,
    ) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            INSERT INTO arbitrator_decisions (
                id, escrow_id, arbitrator, outcome, reasoning, decided_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6)
            "#,
            params![
                Uuid::new_v4().to_string(),
                escrow_id.to_string(),
                decision.arbitrator.to_string(),
                serde_json::to_string(&decision.outcome)?,
                decision.reasoning,
                decision.decided_at.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Verify release authorization signature
    fn verify_release_auth(
        &self,
        escrow: &EscrowAgreement,
        auth: &ReleaseAuth,
    ) -> Result<()> {
        if auth.escrow_id != escrow.id {
            return Err(MarketError::Escrow("Release auth escrow ID mismatch".to_string()));
        }

        // Verify authorizer is involved in escrow
        let is_party = auth.authorizer == escrow.requester 
            || auth.authorizer == escrow.provider
            || escrow.arbitrators.contains(&auth.authorizer);
            
        if !is_party {
            return Err(MarketError::Escrow("Authorizer not party to escrow".to_string()));
        }

        // TODO: Verify signature with verifying key
        // let verifying_key = self.verifying_keys.read().await
        //     .get(&auth.authorizer)
        //     .ok_or_else(|| MarketError::Escrow("No verifying key for authorizer".to_string()))?
        //     .clone();
        
        Ok(())
    }

    /// Check if multi-signature requirements are met
    async fn check_multisig_requirements(
        &self,
        escrow: &EscrowAgreement,
        auth: &ReleaseAuth,
    ) -> Result<bool> {
        match escrow.multisig_type {
            MultiSigType::Single => {
                // Single signature from either party
                Ok(auth.authorizer == escrow.requester || auth.authorizer == escrow.provider)
            }
            MultiSigType::BothParties => {
                // Need signatures from both parties
                let db = self.db.lock().await;
                let count: i64 = db.query_row(
                    r#"
                    SELECT COUNT(DISTINCT authorizer) 
                    FROM release_authorizations 
                    WHERE escrow_id = ?1 
                    AND authorizer IN (?2, ?3)
                    AND decision = ?4
                    "#,
                    params![
                        escrow.id.to_string(),
                        escrow.requester.to_string(),
                        escrow.provider.to_string(),
                        serde_json::to_string(&auth.decision)?,
                    ],
                    |row| row.get(0),
                )?;
                
                Ok(count >= 2)
            }
            MultiSigType::Arbitrators { required, total: _ } => {
                // Need M-of-N arbitrator signatures
                let db = self.db.lock().await;
                let arbitrator_sigs: i64 = db.query_row(
                    r#"
                    SELECT COUNT(DISTINCT authorizer) 
                    FROM release_authorizations 
                    WHERE escrow_id = ?1 
                    AND decision = ?2
                    "#,
                    params![
                        escrow.id.to_string(),
                        serde_json::to_string(&auth.decision)?,
                    ],
                    |row| row.get(0),
                )?;
                
                Ok(arbitrator_sigs >= required as i64)
            }
            MultiSigType::TimeLocked { release_after } => {
                // Time-based automatic release
                Ok(Utc::now() >= release_after)
            }
        }
    }

    /// Internal method to raise dispute
    async fn raise_dispute_internal(
        &self,
        escrow: &mut EscrowAgreement,
        auth: &ReleaseAuth,
    ) -> Result<()> {
        escrow.dispute_info = Some(DisputeInfo {
            raised_by: auth.authorizer,
            reason: "Dispute raised through release authorization".to_string(),
            evidence: vec![],
            decision: None,
            raised_at: Utc::now(),
        });

        let old_state = escrow.state;
        escrow.state = EscrowState::Disputed;
        escrow.updated_at = Utc::now();

        escrow.audit_log.push(AuditEntry {
            timestamp: escrow.updated_at,
            actor: auth.authorizer,
            action: "raise_dispute_via_release".to_string(),
            from_state: old_state,
            to_state: EscrowState::Disputed,
            details: Some("Dispute raised through release authorization".to_string()),
        });

        self.update_escrow_state(&escrow).await?;
        self.store_release_auth(auth).await?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    async fn create_test_wallet() -> Arc<Wallet> {
        let wallet = Wallet::new(":memory:").await.unwrap();
        wallet.init_schema().await.unwrap();
        Arc::new(wallet)
    }

    #[tokio::test]
    async fn test_complete_escrow_flow() {
        let wallet = create_test_wallet().await;
        let escrow = Escrow::new(":memory:", wallet.clone()).await.unwrap();
        escrow.init_schema().await.unwrap();

        let requester = PeerId::random();
        let provider = PeerId::random();
        let job_id = Uuid::new_v4();
        let amount = 100;

        // Fund requester wallet
        wallet.credit(&requester, amount + 50).await.unwrap();

        // Create escrow with single signature
        let agreement = escrow
            .create_escrow(
                job_id,
                requester,
                provider,
                amount,
                MultiSigType::Single,
                vec![],
                60,
            )
            .await
            .unwrap();
        assert_eq!(agreement.state, EscrowState::Created);

        // Fund escrow
        let requester_key = SigningKey::generate(&mut OsRng);
        let agreement = escrow
            .fund_escrow(&agreement.id, &requester, &requester_key)
            .await
            .unwrap();
        assert_eq!(agreement.state, EscrowState::Funded);

        // Check wallet balance
        let balance = wallet.get_balance(&requester).await.unwrap();
        assert_eq!(balance.available, 50);
        assert_eq!(balance.locked, amount);

        // Mark job completed
        let provider_key = SigningKey::generate(&mut OsRng);
        let agreement = escrow
            .mark_completed(&agreement.id, &provider, &provider_key)
            .await
            .unwrap();
        assert_eq!(agreement.state, EscrowState::Completed);

        // Release funds
        let release = ReleaseAuth {
            escrow_id: agreement.id,
            authorizer: requester,
            decision: ReleaseDecision::ApproveRelease,
            signature: vec![0; 64], // Mock signature
            created_at: Utc::now(),
        };

        let agreement = escrow.release_funds(&agreement.id, release).await.unwrap();
        assert_eq!(agreement.state, EscrowState::Released);

        // Check final balances
        let req_balance = wallet.get_balance(&requester).await.unwrap();
        let prov_balance = wallet.get_balance(&provider).await.unwrap();
        assert_eq!(req_balance.available, 50);
        assert_eq!(req_balance.locked, 0);
        assert_eq!(prov_balance.available, amount);
    }

    #[tokio::test]
    async fn test_multi_signature_escrow() {
        let wallet = create_test_wallet().await;
        let escrow = Escrow::new(":memory:", wallet.clone()).await.unwrap();
        escrow.init_schema().await.unwrap();

        let requester = PeerId::random();
        let provider = PeerId::random();
        let amount = 200;

        // Fund requester
        wallet.credit(&requester, amount * 2).await.unwrap();

        // Create escrow requiring both parties
        let agreement = escrow
            .create_escrow(
                Uuid::new_v4(),
                requester,
                provider,
                amount,
                MultiSigType::BothParties,
                vec![],
                60,
            )
            .await
            .unwrap();

        // Fund and complete job
        let req_key = SigningKey::generate(&mut OsRng);
        let prov_key = SigningKey::generate(&mut OsRng);
        
        escrow.fund_escrow(&agreement.id, &requester, &req_key).await.unwrap();
        escrow.mark_completed(&agreement.id, &provider, &prov_key).await.unwrap();

        // First release authorization (requester)
        let release1 = ReleaseAuth {
            escrow_id: agreement.id,
            authorizer: requester,
            decision: ReleaseDecision::ApproveRelease,
            signature: vec![0; 64],
            created_at: Utc::now(),
        };

        let agreement = escrow.release_funds(&agreement.id, release1).await.unwrap();
        // Should still be completed, waiting for second signature
        assert_eq!(agreement.state, EscrowState::Completed);

        // Second release authorization (provider)
        let release2 = ReleaseAuth {
            escrow_id: agreement.id,
            authorizer: provider,
            decision: ReleaseDecision::ApproveRelease,
            signature: vec![0; 64],
            created_at: Utc::now(),
        };

        let agreement = escrow.release_funds(&agreement.id, release2).await.unwrap();
        // Now should be released
        assert_eq!(agreement.state, EscrowState::Released);
    }

    #[tokio::test]
    async fn test_dispute_resolution() {
        let wallet = create_test_wallet().await;
        let escrow = Escrow::new(":memory:", wallet.clone()).await.unwrap();
        escrow.init_schema().await.unwrap();

        let requester = PeerId::random();
        let provider = PeerId::random();
        let arbitrator = PeerId::random();
        let amount = 150;

        wallet.credit(&requester, amount * 2).await.unwrap();

        // Create escrow with arbitrators
        let agreement = escrow
            .create_escrow(
                Uuid::new_v4(),
                requester,
                provider,
                amount,
                MultiSigType::Arbitrators { required: 1, total: 1 },
                vec![arbitrator],
                60,
            )
            .await
            .unwrap();

        // Fund and complete
        let req_key = SigningKey::generate(&mut OsRng);
        let prov_key = SigningKey::generate(&mut OsRng);
        
        escrow.fund_escrow(&agreement.id, &requester, &req_key).await.unwrap();
        escrow.mark_completed(&agreement.id, &provider, &prov_key).await.unwrap();

        // Raise dispute
        let agreement = escrow
            .raise_dispute(
                &agreement.id,
                &requester,
                "Work not satisfactory".to_string(),
                vec!["evidence1".to_string()],
                &req_key,
            )
            .await
            .unwrap();
        assert_eq!(agreement.state, EscrowState::Disputed);

        // Arbitrator resolves dispute with split
        let arb_key = SigningKey::generate(&mut OsRng);
        let agreement = escrow
            .resolve_dispute(
                &agreement.id,
                &arbitrator,
                DisputeOutcome::Split {
                    provider_share: amount / 2,
                    requester_share: amount / 2,
                },
                "Both parties partially at fault".to_string(),
                &arb_key,
            )
            .await
            .unwrap();
        assert_eq!(agreement.state, EscrowState::Resolved);

        // Check balances after split
        let req_balance = wallet.get_balance(&requester).await.unwrap();
        let prov_balance = wallet.get_balance(&provider).await.unwrap();
        assert_eq!(req_balance.available, amount + amount / 2); // Initial - locked + refund
        assert_eq!(prov_balance.available, amount / 2);
    }

    #[tokio::test]
    async fn test_timeout_processing() {
        let wallet = create_test_wallet().await;
        let escrow = Escrow::new(":memory:", wallet.clone()).await.unwrap();
        escrow.init_schema().await.unwrap();

        let requester = PeerId::random();
        let provider = PeerId::random();
        let amount = 100;

        wallet.credit(&requester, amount).await.unwrap();

        // Create escrow with very short timeout
        let agreement = escrow
            .create_escrow(
                Uuid::new_v4(),
                requester,
                provider,
                amount,
                MultiSigType::Single,
                vec![],
                -1, // Already expired
            )
            .await
            .unwrap();

        // Fund it
        let key = SigningKey::generate(&mut OsRng);
        escrow.fund_escrow(&agreement.id, &requester, &key).await.unwrap();

        // Process timeouts
        let processed = escrow.process_timeouts().await.unwrap();
        assert_eq!(processed.len(), 1);
        assert_eq!(processed[0], agreement.id);

        // Check escrow was refunded
        let updated = escrow.get_escrow(&agreement.id).await.unwrap().unwrap();
        assert_eq!(updated.state, EscrowState::Refunded);

        // Check balance was restored
        let balance = wallet.get_balance(&requester).await.unwrap();
        assert_eq!(balance.available, amount);
        assert_eq!(balance.locked, 0);
    }

    #[tokio::test]
    async fn test_audit_log() {
        let wallet = create_test_wallet().await;
        let escrow = Escrow::new(":memory:", wallet.clone()).await.unwrap();
        escrow.init_schema().await.unwrap();

        let requester = PeerId::random();
        let provider = PeerId::random();

        let agreement = escrow
            .create_escrow(
                Uuid::new_v4(),
                requester,
                provider,
                50,
                MultiSigType::Single,
                vec![],
                60,
            )
            .await
            .unwrap();

        let audit_log = escrow.get_audit_log(&agreement.id).await.unwrap();
        assert_eq!(audit_log.len(), 1);
        assert_eq!(audit_log[0].action, "create_escrow");
        assert_eq!(audit_log[0].actor, requester);
    }
}