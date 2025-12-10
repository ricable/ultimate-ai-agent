//! Dynamic pricing mechanisms for compute resources
//!
//! This module implements sophisticated pricing algorithms for neural compute
//! resources based on supply, demand, quality metrics, and network conditions.

use crate::error::Result;
use crate::market::ComputeTaskSpec;
use crate::reputation::ReputationScore;
use chrono::{DateTime, Duration, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::Mutex;

/// Pricing strategy types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PricingStrategy {
    /// Fixed pricing regardless of demand
    Fixed,
    /// Dynamic pricing based on supply/demand
    Dynamic,
    /// Auction-based pricing (first price sealed bid)
    FirstPriceAuction,
    /// Auction-based pricing (second price sealed bid)
    SecondPriceAuction,
    /// Reputation-weighted pricing
    ReputationWeighted,
    /// Time-based surge pricing
    SurgePricing,
}

/// Market demand level
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DemandLevel {
    /// Very low demand (< 20% utilization)
    VeryLow,
    /// Low demand (20-40% utilization)
    Low,
    /// Normal demand (40-60% utilization)
    Normal,
    /// High demand (60-80% utilization)
    High,
    /// Very high demand (> 80% utilization)
    VeryHigh,
}

/// Supply level in the market
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SupplyLevel {
    /// Abundant supply
    Abundant,
    /// Normal supply
    Normal,
    /// Limited supply
    Limited,
    /// Scarce supply
    Scarce,
}

/// Real-time market conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    /// Current demand level
    pub demand_level: DemandLevel,
    /// Current supply level
    pub supply_level: SupplyLevel,
    /// Average compute utilization (0.0 to 1.0)
    pub utilization_rate: f64,
    /// Number of active providers
    pub active_providers: u64,
    /// Number of pending requests
    pub pending_requests: u64,
    /// Average response time in seconds
    pub avg_response_time: f64,
    /// Quality score average across providers
    pub avg_quality_score: f64,
    /// Timestamp of last update
    pub last_updated: DateTime<Utc>,
}

/// Pricing parameters for a specific task type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricingParams {
    /// Base price per compute unit
    pub base_price: u64,
    /// Minimum price floor
    pub min_price: u64,
    /// Maximum price ceiling  
    pub max_price: u64,
    /// Demand multiplier factor
    pub demand_multiplier: f64,
    /// Supply multiplier factor  
    pub supply_multiplier: f64,
    /// Quality bonus percentage
    pub quality_bonus: f64,
    /// Reputation bonus percentage
    pub reputation_bonus: f64,
    /// Urgency multiplier for fast delivery
    pub urgency_multiplier: f64,
}

impl Default for PricingParams {
    fn default() -> Self {
        Self {
            base_price: 10, // 10 ruv tokens per compute unit
            min_price: 1,
            max_price: 1000,
            demand_multiplier: 1.5,
            supply_multiplier: 0.8,
            quality_bonus: 0.2,
            reputation_bonus: 0.3,
            urgency_multiplier: 2.0,
        }
    }
}

/// Dynamic pricing calculation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceQuote {
    /// Calculated price per compute unit
    pub price_per_unit: u64,
    /// Base price before adjustments
    pub base_price: u64,
    /// Applied adjustments
    pub adjustments: PriceAdjustments,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Valid until timestamp
    pub valid_until: DateTime<Utc>,
    /// Reasoning for the price
    pub reasoning: String,
}

/// Price adjustment factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceAdjustments {
    /// Demand adjustment factor
    pub demand_factor: f64,
    /// Supply adjustment factor
    pub supply_factor: f64,
    /// Quality adjustment factor
    pub quality_factor: f64,
    /// Reputation adjustment factor
    pub reputation_factor: f64,
    /// Urgency adjustment factor
    pub urgency_factor: f64,
    /// Network conditions factor
    pub network_factor: f64,
}

/// Historical pricing data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceDataPoint {
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Task type
    pub task_type: String,
    /// Price per unit at this time
    pub price: u64,
    /// Market conditions
    pub conditions: MarketConditions,
    /// Volume of trades
    pub volume: u64,
}

/// Pricing engine for dynamic price calculation
pub struct PricingEngine {
    db: Mutex<Connection>,
    pricing_params: Mutex<HashMap<String, PricingParams>>,
    market_conditions: Mutex<MarketConditions>,
}

impl PricingEngine {
    /// Create a new pricing engine
    pub async fn new(db_path: &str) -> Result<Self> {
        let db = Connection::open(db_path)?;
        
        let default_conditions = MarketConditions {
            demand_level: DemandLevel::Normal,
            supply_level: SupplyLevel::Normal,
            utilization_rate: 0.5,
            active_providers: 0,
            pending_requests: 0,
            avg_response_time: 120.0,
            avg_quality_score: 0.8,
            last_updated: Utc::now(),
        };

        Ok(Self {
            db: Mutex::new(db),
            pricing_params: Mutex::new(HashMap::new()),
            market_conditions: Mutex::new(default_conditions),
        })
    }

    /// Initialize database schema
    pub async fn init_schema(&self) -> Result<()> {
        let db = self.db.lock().await;
        
        // Pricing history table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS pricing_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                task_type TEXT NOT NULL,
                price INTEGER NOT NULL,
                demand_level TEXT NOT NULL,
                supply_level TEXT NOT NULL,
                utilization_rate REAL NOT NULL,
                volume INTEGER NOT NULL,
                conditions TEXT NOT NULL
            )
            "#,
            [],
        )?;

        // Pricing parameters table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS pricing_params (
                task_type TEXT PRIMARY KEY,
                base_price INTEGER NOT NULL,
                min_price INTEGER NOT NULL,
                max_price INTEGER NOT NULL,
                demand_multiplier REAL NOT NULL,
                supply_multiplier REAL NOT NULL,
                quality_bonus REAL NOT NULL,
                reputation_bonus REAL NOT NULL,
                urgency_multiplier REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            "#,
            [],
        )?;

        // Market conditions table
        db.execute(
            r#"
            CREATE TABLE IF NOT EXISTS market_conditions (
                id INTEGER PRIMARY KEY,
                demand_level TEXT NOT NULL,
                supply_level TEXT NOT NULL,
                utilization_rate REAL NOT NULL,
                active_providers INTEGER NOT NULL,
                pending_requests INTEGER NOT NULL,
                avg_response_time REAL NOT NULL,
                avg_quality_score REAL NOT NULL,
                last_updated TEXT NOT NULL
            )
            "#,
            [],
        )?;

        // Create indexes
        db.execute(
            r#"
            CREATE INDEX IF NOT EXISTS idx_pricing_history_timestamp ON pricing_history(timestamp);
            CREATE INDEX IF NOT EXISTS idx_pricing_history_task_type ON pricing_history(task_type);
            CREATE INDEX IF NOT EXISTS idx_pricing_history_price ON pricing_history(price);
            "#,
            [],
        )?;

        Ok(())
    }

    /// Calculate dynamic price for a compute task
    pub async fn calculate_price(
        &self,
        task_spec: &ComputeTaskSpec,
        strategy: PricingStrategy,
        provider_reputation: Option<&ReputationScore>,
        urgency_level: f64, // 0.0 to 1.0
    ) -> Result<PriceQuote> {
        let conditions = self.market_conditions.lock().await.clone();
        let params = self.get_pricing_params(&task_spec.task_type).await?;
        
        let base_price = params.base_price;
        let mut adjustments = PriceAdjustments {
            demand_factor: 1.0,
            supply_factor: 1.0,
            quality_factor: 1.0,
            reputation_factor: 1.0,
            urgency_factor: 1.0,
            network_factor: 1.0,
        };

        let mut reasoning = vec![];

        // Apply strategy-specific calculations
        match strategy {
            PricingStrategy::Fixed => {
                reasoning.push("Fixed pricing strategy applied".to_string());
            }
            PricingStrategy::Dynamic => {
                // Demand adjustment
                adjustments.demand_factor = match conditions.demand_level {
                    DemandLevel::VeryLow => 0.7,
                    DemandLevel::Low => 0.85,
                    DemandLevel::Normal => 1.0,
                    DemandLevel::High => 1.3,
                    DemandLevel::VeryHigh => 1.8,
                };
                reasoning.push(format!("Demand adjustment: {:.2}x", adjustments.demand_factor));

                // Supply adjustment
                adjustments.supply_factor = match conditions.supply_level {
                    SupplyLevel::Abundant => 0.8,
                    SupplyLevel::Normal => 1.0,
                    SupplyLevel::Limited => 1.2,
                    SupplyLevel::Scarce => 1.5,
                };
                reasoning.push(format!("Supply adjustment: {:.2}x", adjustments.supply_factor));

                // Network conditions adjustment
                if conditions.avg_response_time > 300.0 {
                    adjustments.network_factor = 1.1; // Premium for network congestion
                    reasoning.push("Network congestion surcharge applied".to_string());
                }
            }
            PricingStrategy::ReputationWeighted => {
                if let Some(reputation) = provider_reputation {
                    adjustments.reputation_factor = 1.0 + (reputation.score / 1000.0) * params.reputation_bonus;
                    reasoning.push(format!("Reputation bonus: {:.2}x", adjustments.reputation_factor));
                }
            }
            PricingStrategy::SurgePricing => {
                let surge_factor = conditions.utilization_rate.powf(2.0);
                adjustments.demand_factor = 1.0 + surge_factor;
                reasoning.push(format!("Surge pricing: {:.2}x", adjustments.demand_factor));
            }
            _ => {
                // Auction strategies would require bid collection
                reasoning.push("Auction pricing requires bid collection".to_string());
            }
        }

        // Quality adjustment based on task requirements
        if task_spec.privacy_level == crate::market::PrivacyLevel::Confidential {
            adjustments.quality_factor = 1.0 + params.quality_bonus;
            reasoning.push("Privacy premium applied".to_string());
        }

        // Urgency adjustment
        if urgency_level > 0.5 {
            adjustments.urgency_factor = 1.0 + (urgency_level - 0.5) * 2.0 * params.urgency_multiplier;
            reasoning.push(format!("Urgency premium: {:.2}x", adjustments.urgency_factor));
        }

        // Calculate final price
        let adjusted_price = (base_price as f64
            * adjustments.demand_factor
            * adjustments.supply_factor
            * adjustments.quality_factor
            * adjustments.reputation_factor
            * adjustments.urgency_factor
            * adjustments.network_factor) as u64;

        let final_price = adjusted_price.max(params.min_price).min(params.max_price);

        // Calculate confidence based on data availability and market stability
        let confidence = self.calculate_confidence(&task_spec.task_type, &conditions).await?;

        // Price valid for 5 minutes by default
        let valid_until = Utc::now() + Duration::minutes(5);

        Ok(PriceQuote {
            price_per_unit: final_price,
            base_price,
            adjustments,
            confidence,
            valid_until,
            reasoning: reasoning.join(", "),
        })
    }

    /// Update market conditions
    pub async fn update_market_conditions(&self, conditions: MarketConditions) -> Result<()> {
        let db = self.db.lock().await;
        
        // Update in-memory conditions
        *self.market_conditions.lock().await = conditions.clone();

        // Store in database
        db.execute(
            r#"
            INSERT OR REPLACE INTO market_conditions (
                id, demand_level, supply_level, utilization_rate,
                active_providers, pending_requests, avg_response_time,
                avg_quality_score, last_updated
            )
            VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            params![
                format!("{:?}", conditions.demand_level),
                format!("{:?}", conditions.supply_level),
                conditions.utilization_rate,
                conditions.active_providers as i64,
                conditions.pending_requests as i64,
                conditions.avg_response_time,
                conditions.avg_quality_score,
                conditions.last_updated.to_rfc3339(),
            ],
        )?;

        Ok(())
    }

    /// Record a price data point for analysis
    pub async fn record_price_data(&self, data_point: PriceDataPoint) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            INSERT INTO pricing_history (
                timestamp, task_type, price, demand_level, supply_level,
                utilization_rate, volume, conditions
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            params![
                data_point.timestamp.to_rfc3339(),
                data_point.task_type,
                data_point.price as i64,
                format!("{:?}", data_point.conditions.demand_level),
                format!("{:?}", data_point.conditions.supply_level),
                data_point.conditions.utilization_rate,
                data_point.volume as i64,
                serde_json::to_string(&data_point.conditions)?,
            ],
        )?;

        Ok(())
    }

    /// Get pricing parameters for a task type
    async fn get_pricing_params(&self, task_type: &str) -> Result<PricingParams> {
        // Check cache first
        if let Some(params) = self.pricing_params.lock().await.get(task_type) {
            return Ok(params.clone());
        }

        // Load from database
        let db = self.db.lock().await;
        let params = db
            .query_row(
                r#"
                SELECT base_price, min_price, max_price, demand_multiplier,
                       supply_multiplier, quality_bonus, reputation_bonus, urgency_multiplier
                FROM pricing_params
                WHERE task_type = ?1
                "#,
                params![task_type],
                |row| {
                    Ok(PricingParams {
                        base_price: row.get::<_, i64>(0)? as u64,
                        min_price: row.get::<_, i64>(1)? as u64,
                        max_price: row.get::<_, i64>(2)? as u64,
                        demand_multiplier: row.get(3)?,
                        supply_multiplier: row.get(4)?,
                        quality_bonus: row.get(5)?,
                        reputation_bonus: row.get(6)?,
                        urgency_multiplier: row.get(7)?,
                    })
                },
            )
            .unwrap_or_else(|_| PricingParams::default());

        // Cache the result
        self.pricing_params.lock().await.insert(task_type.to_string(), params.clone());

        Ok(params)
    }

    /// Calculate confidence level for pricing
    async fn calculate_confidence(&self, task_type: &str, _conditions: &MarketConditions) -> Result<f64> {
        let db = self.db.lock().await;
        
        // Count recent price data points
        let recent_count: i64 = db.query_row(
            r#"
            SELECT COUNT(*)
            FROM pricing_history
            WHERE task_type = ?1 AND timestamp > datetime('now', '-24 hours')
            "#,
            params![task_type],
            |row| row.get(0),
        ).unwrap_or(0);

        // Calculate confidence based on data availability
        let base_confidence = if recent_count > 10 {
            0.9
        } else if recent_count > 5 {
            0.7
        } else if recent_count > 0 {
            0.5
        } else {
            0.3
        };

        // Adjust for market volatility
        let volatility_factor = self.calculate_volatility(task_type).await?;
        let confidence = base_confidence * (1.0 - volatility_factor);

        Ok(confidence.max(0.1).min(1.0))
    }

    /// Calculate market volatility for a task type
    async fn calculate_volatility(&self, task_type: &str) -> Result<f64> {
        let db = self.db.lock().await;
        
        let prices: Vec<f64> = db.prepare(
            r#"
            SELECT price FROM pricing_history
            WHERE task_type = ?1 AND timestamp > datetime('now', '-24 hours')
            ORDER BY timestamp DESC
            LIMIT 20
            "#,
        )?
        .query_map(params![task_type], |row| {
            Ok(row.get::<_, i64>(0)? as f64)
        })?
        .collect::<std::result::Result<Vec<_>, _>>()?;

        if prices.len() < 2 {
            return Ok(0.0);
        }

        // Calculate coefficient of variation
        let mean = prices.iter().sum::<f64>() / prices.len() as f64;
        let variance = prices.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();
        
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
        
        // Normalize to 0-1 range
        Ok(cv.min(1.0))
    }

    /// Analyze pricing trends for a task type
    pub async fn analyze_trends(&self, task_type: &str, days: u32) -> Result<PriceTrend> {
        let db = self.db.lock().await;
        
        let mut stmt = db.prepare(
            r#"
            SELECT timestamp, price, volume
            FROM pricing_history
            WHERE task_type = ?1 AND timestamp > datetime('now', ?2)
            ORDER BY timestamp ASC
            "#,
        )?;

        let days_str = format!("-{} days", days);
        let data_points: Vec<(DateTime<Utc>, u64, u64)> = stmt
            .query_map(params![task_type, days_str], |row| {
                let timestamp = DateTime::parse_from_rfc3339(&row.get::<_, String>(0)?)
                    .unwrap()
                    .with_timezone(&Utc);
                let price = row.get::<_, i64>(1)? as u64;
                let volume = row.get::<_, i64>(2)? as u64;
                Ok((timestamp, price, volume))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if data_points.is_empty() {
            return Ok(PriceTrend {
                direction: TrendDirection::Stable,
                strength: 0.0,
                avg_price: 0,
                price_change: 0.0,
                volume_trend: TrendDirection::Stable,
            });
        }

        let prices: Vec<u64> = data_points.iter().map(|(_, p, _)| *p).collect();
        let volumes: Vec<u64> = data_points.iter().map(|(_, _, v)| *v).collect();

        let avg_price = prices.iter().sum::<u64>() / prices.len() as u64;
        let first_price = prices[0] as f64;
        let last_price = prices[prices.len() - 1] as f64;
        let price_change = (last_price - first_price) / first_price * 100.0;

        let direction = if price_change > 5.0 {
            TrendDirection::Increasing
        } else if price_change < -5.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        let strength = price_change.abs() / 100.0;

        // Simple volume trend analysis
        let first_half_vol: u64 = volumes.iter().take(volumes.len() / 2).sum();
        let second_half_vol: u64 = volumes.iter().skip(volumes.len() / 2).sum();
        let volume_trend = if second_half_vol > first_half_vol * 110 / 100 {
            TrendDirection::Increasing
        } else if second_half_vol < first_half_vol * 90 / 100 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(PriceTrend {
            direction,
            strength,
            avg_price,
            price_change,
            volume_trend,
        })
    }

    /// Get current market conditions
    pub async fn get_market_conditions(&self) -> MarketConditions {
        self.market_conditions.lock().await.clone()
    }

    /// Update pricing parameters for a task type
    pub async fn update_pricing_params(&self, task_type: String, params: PricingParams) -> Result<()> {
        let db = self.db.lock().await;
        
        db.execute(
            r#"
            INSERT OR REPLACE INTO pricing_params (
                task_type, base_price, min_price, max_price,
                demand_multiplier, supply_multiplier, quality_bonus,
                reputation_bonus, urgency_multiplier, updated_at
            )
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            params![
                task_type,
                params.base_price as i64,
                params.min_price as i64,
                params.max_price as i64,
                params.demand_multiplier,
                params.supply_multiplier,
                params.quality_bonus,
                params.reputation_bonus,
                params.urgency_multiplier,
                Utc::now().to_rfc3339(),
            ],
        )?;

        // Update cache
        self.pricing_params.lock().await.insert(task_type, params);

        Ok(())
    }
}

/// Price trend analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceTrend {
    /// Trend direction
    pub direction: TrendDirection,
    /// Trend strength (0.0 to 1.0)
    pub strength: f64,
    /// Average price over the period
    pub avg_price: u64,
    /// Percentage price change
    pub price_change: f64,
    /// Volume trend direction
    pub volume_trend: TrendDirection,
}

/// Trend direction
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TrendDirection {
    /// Increasing trend
    Increasing,
    /// Decreasing trend
    Decreasing,
    /// Stable/no clear trend
    Stable,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::market::PrivacyLevel;

    #[tokio::test]
    async fn test_pricing_engine_creation() {
        let engine = PricingEngine::new(":memory:").await.unwrap();
        engine.init_schema().await.unwrap();
        
        let conditions = engine.get_market_conditions().await;
        assert_eq!(conditions.demand_level, DemandLevel::Normal);
    }

    #[tokio::test]
    async fn test_dynamic_pricing() {
        let engine = PricingEngine::new(":memory:").await.unwrap();
        engine.init_schema().await.unwrap();

        let task_spec = ComputeTaskSpec {
            task_type: "test_task".to_string(),
            compute_units: 100,
            max_duration_secs: 300,
            required_capabilities: vec!["rust".to_string()],
            min_reputation: None,
            privacy_level: PrivacyLevel::Public,
            encrypted_payload: None,
        };

        let quote = engine
            .calculate_price(&task_spec, PricingStrategy::Dynamic, None, 0.0)
            .await
            .unwrap();

        assert!(quote.price_per_unit > 0);
        assert!(quote.confidence > 0.0);
        assert!(quote.valid_until > Utc::now());
    }

    #[tokio::test]
    async fn test_market_conditions_update() {
        let engine = PricingEngine::new(":memory:").await.unwrap();
        engine.init_schema().await.unwrap();

        let new_conditions = MarketConditions {
            demand_level: DemandLevel::High,
            supply_level: SupplyLevel::Limited,
            utilization_rate: 0.8,
            active_providers: 10,
            pending_requests: 50,
            avg_response_time: 200.0,
            avg_quality_score: 0.85,
            last_updated: Utc::now(),
        };

        engine.update_market_conditions(new_conditions.clone()).await.unwrap();
        
        let retrieved = engine.get_market_conditions().await;
        assert_eq!(retrieved.demand_level, DemandLevel::High);
        assert_eq!(retrieved.supply_level, SupplyLevel::Limited);
    }
}