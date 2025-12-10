//! Reputation tracking system for market participants

use crate::error::Result;
use chrono::{DateTime, Utc};
use libp2p::PeerId;
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use uuid::Uuid;

/// Reputation event type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ReputationEvent {
    /// Successful trade completion
    TradeCompleted,
    /// Failed to complete trade
    TradeFailed,
    /// Disputed trade
    TradeDisputed,
    /// Dispute resolved in favor
    DisputeWon,
    /// Dispute resolved against
    DisputeLost,
    /// Fast response time
    FastResponse,
    /// Slow response time
    SlowResponse,
    /// Good feedback from peer
    PositiveFeedback,
    /// Bad feedback from peer
    NegativeFeedback,
    /// New user bonus
    NewUserBonus,
}

impl ReputationEvent {
    /// Get the score impact of this event
    pub fn score_impact(&self) -> f64 {
        match self {
            ReputationEvent::TradeCompleted => 10.0,
            ReputationEvent::TradeFailed => -20.0,
            ReputationEvent::TradeDisputed => -5.0,
            ReputationEvent::DisputeWon => 15.0,
            ReputationEvent::DisputeLost => -25.0,
            ReputationEvent::FastResponse => 2.0,
            ReputationEvent::SlowResponse => -1.0,
            ReputationEvent::PositiveFeedback => 5.0,
            ReputationEvent::NegativeFeedback => -10.0,
            ReputationEvent::NewUserBonus => 50.0,
        }
    }
}

/// Reputation score with history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationScore {
    /// Peer ID
    pub peer_id: PeerId,
    /// Current reputation score
    pub score: f64,
    /// Total number of trades
    pub total_trades: u64,
    /// Successful trades
    pub successful_trades: u64,
    /// Failed trades
    pub failed_trades: u64,
    /// Number of disputes
    pub disputes: u64,
    /// Average response time (seconds)
    pub avg_response_time: Option<f64>,
    /// Last activity timestamp
    pub last_activity: DateTime<Utc>,
    /// Account creation timestamp
    pub created_at: DateTime<Utc>,
}

impl ReputationScore {
    /// Calculate trade success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_trades == 0 {
            0.0
        } else {
            (self.successful_trades as f64 / self.total_trades as f64) * 100.0
        }
    }

    /// Get reputation tier
    pub fn tier(&self) -> &'static str {
        match self.score {
            s if s >= 1000.0 => "Legendary",
            s if s >= 500.0 => "Expert",
            s if s >= 250.0 => "Trusted",
            s if s >= 100.0 => "Reliable",
            s if s >= 50.0 => "Standard",
            s if s >= 0.0 => "New",
            _ => "Risky",
        }
    }

    /// Check if peer is trustworthy
    pub fn is_trustworthy(&self, min_score: f64) -> bool {
        self.score >= min_score && self.success_rate() >= 80.0
    }
}

/// Reputation feedback
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationFeedback {
    /// Unique feedback ID
    pub id: Uuid,
    /// Trade/Escrow ID this feedback is for
    pub reference_id: Uuid,
    /// Who is giving feedback
    pub from_peer: PeerId,
    /// Who is receiving feedback
    pub to_peer: PeerId,
    /// Rating (1-5 stars)
    pub rating: u8,
    /// Text comment
    pub comment: Option<String>,
    /// Feedback timestamp
    pub created_at: DateTime<Utc>,
}

/// Reputation tracking service
pub struct Reputation {
    db: Mutex<Connection>,
}

impl Reputation {
    /// Create a new reputation service
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
            CREATE TABLE IF NOT EXISTS reputation_scores (
                peer_id TEXT PRIMARY KEY,
                score REAL NOT NULL DEFAULT 0.0,
                total_trades INTEGER NOT NULL DEFAULT 0,
                successful_trades INTEGER NOT NULL DEFAULT 0,
                failed_trades INTEGER NOT NULL DEFAULT 0,
                disputes INTEGER NOT NULL DEFAULT 0,
                avg_response_time REAL,
                last_activity TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS reputation_events (
                id TEXT PRIMARY KEY,
                peer_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                score_impact REAL NOT NULL,
                reference_id TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS reputation_feedback (
                id TEXT PRIMARY KEY,
                reference_id TEXT NOT NULL,
                from_peer TEXT NOT NULL,
                to_peer TEXT NOT NULL,
                rating INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_events_peer ON reputation_events(peer_id);
            CREATE INDEX IF NOT EXISTS idx_events_created ON reputation_events(created_at);
            CREATE INDEX IF NOT EXISTS idx_feedback_to ON reputation_feedback(to_peer);
            CREATE INDEX IF NOT EXISTS idx_feedback_from ON reputation_feedback(from_peer);
            CREATE INDEX IF NOT EXISTS idx_feedback_ref ON reputation_feedback(reference_id);
            "#,
            [],
        )?;

        Ok(())
    }

    /// Get or create reputation score for a peer
    pub async fn get_reputation(&self, peer_id: &PeerId) -> Result<ReputationScore> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();

        let score = db
            .query_row(
                r#"
                SELECT score, total_trades, successful_trades, failed_trades,
                       disputes, avg_response_time, last_activity, created_at
                FROM reputation_scores
                WHERE peer_id = ?1
                "#,
                params![peer_str],
                |row| {
                    Ok(ReputationScore {
                        peer_id: *peer_id,
                        score: row.get(0)?,
                        total_trades: row.get::<_, i64>(1)? as u64,
                        successful_trades: row.get::<_, i64>(2)? as u64,
                        failed_trades: row.get::<_, i64>(3)? as u64,
                        disputes: row.get::<_, i64>(4)? as u64,
                        avg_response_time: row.get(5)?,
                        last_activity: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
                            .unwrap()
                            .with_timezone(&Utc),
                        created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                            .unwrap()
                            .with_timezone(&Utc),
                    })
                },
            )
            .ok();

        if let Some(score) = score {
            Ok(score)
        } else {
            // Create new reputation
            let now = Utc::now();
            let new_score = ReputationScore {
                peer_id: *peer_id,
                score: 0.0,
                total_trades: 0,
                successful_trades: 0,
                failed_trades: 0,
                disputes: 0,
                avg_response_time: None,
                last_activity: now,
                created_at: now,
            };

            db.execute(
                r#"
                INSERT INTO reputation_scores (
                    peer_id, score, total_trades, successful_trades, failed_trades,
                    disputes, avg_response_time, last_activity, created_at
                )
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
                "#,
                params![
                    peer_str,
                    new_score.score,
                    new_score.total_trades as i64,
                    new_score.successful_trades as i64,
                    new_score.failed_trades as i64,
                    new_score.disputes as i64,
                    new_score.avg_response_time,
                    new_score.last_activity.to_rfc3339(),
                    new_score.created_at.to_rfc3339(),
                ],
            )?;

            // Give new user bonus - avoid recursion by directly calculating score
            let bonus_impact = ReputationEvent::NewUserBonus.score_impact();
            let mut new_score = new_score;
            new_score.score = (new_score.score as f64 + bonus_impact).max(0.0);
            
            // Record the bonus event in database
            db.execute(
                "INSERT INTO reputation_events (peer_id, event_type, score_impact, created_at) 
                 VALUES (?1, ?2, ?3, ?4)",
                [
                    &peer_str as &dyn rusqlite::ToSql,
                    &("NewUserBonus".to_string()) as &dyn rusqlite::ToSql,
                    &(bonus_impact as f64) as &dyn rusqlite::ToSql,
                    &chrono::Utc::now().to_rfc3339() as &dyn rusqlite::ToSql,
                ],
            )?;

            // Update the score in database
            db.execute(
                "UPDATE reputation_scores SET score = ?1 WHERE peer_id = ?2",
                [&new_score.score as &dyn rusqlite::ToSql, &peer_str as &dyn rusqlite::ToSql],
            )?;

            Ok(new_score)
        }
    }

    /// Record a reputation event
    pub async fn record_event(
        &self,
        peer_id: &PeerId,
        event: ReputationEvent,
        reference_id: Option<Uuid>,
        metadata: Option<serde_json::Value>,
    ) -> Result<ReputationScore> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();
        let impact = event.score_impact();

        // Record the event
        db.execute(
            r#"
            INSERT INTO reputation_events (
                id, peer_id, event_type, score_impact, reference_id, metadata, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                Uuid::new_v4().to_string(),
                peer_str,
                format!("{:?}", event),
                impact,
                reference_id.map(|id| id.to_string()),
                metadata.map(|m| m.to_string()),
                Utc::now().to_rfc3339(),
            ],
        )?;

        // Update reputation score
        let mut update_query = format!(
            "UPDATE reputation_scores SET score = score + {}, last_activity = ?2",
            impact
        );

        match event {
            ReputationEvent::TradeCompleted => {
                update_query.push_str(", total_trades = total_trades + 1, successful_trades = successful_trades + 1");
            }
            ReputationEvent::TradeFailed => {
                update_query.push_str(", total_trades = total_trades + 1, failed_trades = failed_trades + 1");
            }
            ReputationEvent::TradeDisputed => {
                update_query.push_str(", disputes = disputes + 1");
            }
            _ => {}
        }

        update_query.push_str(" WHERE peer_id = ?1");

        db.execute(
            &update_query,
            params![peer_str, Utc::now().to_rfc3339()],
        )?;

        // Instead of recursively calling get_reputation, calculate it directly
        let updated_score = db.query_row(
            "SELECT score, total_trades, successful_trades, failed_trades, 
                    avg_response_time, disputes, last_activity, created_at 
             FROM reputation_scores WHERE peer_id = ?1",
            [&peer_str],
            |row| Ok(ReputationScore {
                peer_id: *peer_id,
                score: row.get::<_, f64>(0)?,
                total_trades: row.get::<_, i64>(1)? as u64,
                successful_trades: row.get::<_, i64>(2)? as u64,
                failed_trades: row.get::<_, i64>(3)? as u64,
                avg_response_time: row.get::<_, Option<f64>>(4)?,
                disputes: row.get::<_, i64>(5)? as u64,
                last_activity: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
                    .map_err(|_| rusqlite::Error::InvalidColumnType(6, "timestamp".to_string(), rusqlite::types::Type::Text))?
                    .with_timezone(&chrono::Utc),
                created_at: chrono::DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                    .map_err(|_| rusqlite::Error::InvalidColumnType(7, "timestamp".to_string(), rusqlite::types::Type::Text))?
                    .with_timezone(&chrono::Utc),
            })
        )?;
        
        Ok(updated_score)
    }

    /// Submit feedback for a trade
    pub async fn submit_feedback(
        &self,
        reference_id: Uuid,
        from_peer: PeerId,
        to_peer: PeerId,
        rating: u8,
        comment: Option<String>,
    ) -> Result<()> {
        if rating < 1 || rating > 5 {
            return Err(crate::error::MarketError::InvalidOrder(
                "Rating must be between 1 and 5".to_string()
            ));
        }

        let feedback = ReputationFeedback {
            id: Uuid::new_v4(),
            reference_id,
            from_peer,
            to_peer,
            rating,
            comment,
            created_at: Utc::now(),
        };

        let db = self.db.lock().await;
        db.execute(
            r#"
            INSERT INTO reputation_feedback (
                id, reference_id, from_peer, to_peer, rating, comment, created_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            params![
                feedback.id.to_string(),
                feedback.reference_id.to_string(),
                feedback.from_peer.to_string(),
                feedback.to_peer.to_string(),
                feedback.rating,
                feedback.comment,
                feedback.created_at.to_rfc3339(),
            ],
        )?;

        drop(db);

        // Record reputation event based on rating
        let event = if rating >= 4 {
            ReputationEvent::PositiveFeedback
        } else if rating <= 2 {
            ReputationEvent::NegativeFeedback
        } else {
            return Ok(()); // Neutral feedback doesn't affect score
        };

        self.record_event(&to_peer, event, Some(reference_id), None).await?;

        Ok(())
    }

    /// Get feedback for a peer
    pub async fn get_feedback(
        &self,
        peer_id: &PeerId,
        limit: u32,
    ) -> Result<Vec<ReputationFeedback>> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();

        let mut stmt = db.prepare(
            r#"
            SELECT id, reference_id, from_peer, to_peer, rating, comment, created_at
            FROM reputation_feedback
            WHERE to_peer = ?1
            ORDER BY created_at DESC
            LIMIT ?2
            "#,
        )?;

        let feedback = stmt
            .query_map(params![peer_str, limit], |row| {
                Ok(ReputationFeedback {
                    id: Uuid::parse_str(&row.get::<_, String>(0)?).unwrap(),
                    reference_id: Uuid::parse_str(&row.get::<_, String>(1)?).unwrap(),
                    from_peer: row.get::<_, String>(2)?.parse().unwrap(),
                    to_peer: row.get::<_, String>(3)?.parse().unwrap(),
                    rating: row.get(4)?,
                    comment: row.get(5)?,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(6)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(feedback)
    }

    /// Get reputation leaderboard
    pub async fn get_leaderboard(&self, limit: u32) -> Result<Vec<ReputationScore>> {
        let db = self.db.lock().await;

        let mut stmt = db.prepare(
            r#"
            SELECT peer_id, score, total_trades, successful_trades, failed_trades,
                   disputes, avg_response_time, last_activity, created_at
            FROM reputation_scores
            ORDER BY score DESC
            LIMIT ?1
            "#,
        )?;

        let scores = stmt
            .query_map(params![limit], |row| {
                Ok(ReputationScore {
                    peer_id: row.get::<_, String>(0)?.parse().unwrap(),
                    score: row.get(1)?,
                    total_trades: row.get::<_, i64>(2)? as u64,
                    successful_trades: row.get::<_, i64>(3)? as u64,
                    failed_trades: row.get::<_, i64>(4)? as u64,
                    disputes: row.get::<_, i64>(5)? as u64,
                    avg_response_time: row.get(6)?,
                    last_activity: DateTime::parse_from_rfc3339(&row.get::<_, String>(7)?)
                        .unwrap()
                        .with_timezone(&Utc),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<_, String>(8)?)
                        .unwrap()
                        .with_timezone(&Utc),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(scores)
    }

    /// Update response time statistics
    pub async fn update_response_time(
        &self,
        peer_id: &PeerId,
        response_time_secs: f64,
    ) -> Result<()> {
        let db = self.db.lock().await;
        let peer_str = peer_id.to_string();

        // Get current average
        let current_avg: Option<f64> = db
            .query_row(
                "SELECT avg_response_time FROM reputation_scores WHERE peer_id = ?1",
                params![peer_str],
                |row| row.get(0),
            )
            .ok()
            .flatten();

        // Calculate new average (simple moving average)
        let new_avg = if let Some(avg) = current_avg {
            (avg * 0.9) + (response_time_secs * 0.1)
        } else {
            response_time_secs
        };

        db.execute(
            "UPDATE reputation_scores SET avg_response_time = ?2 WHERE peer_id = ?1",
            params![peer_str, new_avg],
        )?;

        // Record fast/slow response event
        drop(db);
        let event = if response_time_secs < 60.0 {
            ReputationEvent::FastResponse
        } else if response_time_secs > 300.0 {
            ReputationEvent::SlowResponse
        } else {
            return Ok(());
        };

        self.record_event(peer_id, event, None, None).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reputation_system() {
        let reputation = Reputation::new(":memory:").await.unwrap();
        reputation.init_schema().await.unwrap();

        let peer = PeerId::random();

        // Get initial reputation (should create new)
        let score = reputation.get_reputation(&peer).await.unwrap();
        assert_eq!(score.score, 50.0); // New user bonus
        assert_eq!(score.tier(), "Standard");

        // Record successful trade
        let score = reputation
            .record_event(&peer, ReputationEvent::TradeCompleted, None, None)
            .await
            .unwrap();
        assert_eq!(score.score, 60.0);
        assert_eq!(score.successful_trades, 1);

        // Submit positive feedback
        let trade_id = Uuid::new_v4();
        let reviewer = PeerId::random();
        reputation
            .submit_feedback(trade_id, reviewer, peer, 5, Some("Great trader!".to_string()))
            .await
            .unwrap();

        let score = reputation.get_reputation(&peer).await.unwrap();
        assert_eq!(score.score, 65.0);

        // Get feedback
        let feedback = reputation.get_feedback(&peer, 10).await.unwrap();
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].rating, 5);
    }
}