//! UE metrics processing and feature extraction

use anyhow::Result;
use chrono::{DateTime, Utc, Duration};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::RwLock;
use statrs::statistics::Statistics;

use crate::config::OptMobConfig;
use crate::UeMetrics;

/// Processed UE metrics with normalized values and time-series features
#[derive(Debug, Clone)]
pub struct ProcessedMetrics {
    pub ue_id: String,
    pub cell_id: String,
    pub timestamp: DateTime<Utc>,
    
    // Normalized metrics (0.0 to 1.0)
    pub rsrp_normalized: f64,
    pub sinr_normalized: f64,
    pub speed_normalized: f64,
    pub throughput_normalized: f64,
    pub cqi_normalized: f64,
    pub phr_normalized: f64,
    pub ta_normalized: f64,
    pub load_factor: f64,
    
    // Time-series features
    pub trend_rsrp: f64,
    pub trend_sinr: f64,
    pub variance_rsrp: f64,
    pub variance_sinr: f64,
    pub rate_of_change_rsrp: f64,
    pub rate_of_change_sinr: f64,
    pub stability_score: f64,
    pub mobility_indicator: f64,
}

/// UE metrics processing engine
pub struct UeMetricsProcessor {
    config: Arc<OptMobConfig>,
    metrics_history: Arc<RwLock<HashMap<String, VecDeque<UeMetrics>>>>,
    normalization_stats: Arc<RwLock<NormalizationStats>>,
}

#[derive(Debug, Clone)]
struct NormalizationStats {
    rsrp_min: f64,
    rsrp_max: f64,
    sinr_min: f64,
    sinr_max: f64,
    speed_min: f64,
    speed_max: f64,
    throughput_min: f64,
    throughput_max: f64,
    last_updated: DateTime<Utc>,
}

impl NormalizationStats {
    fn new() -> Self {
        Self {
            rsrp_min: -140.0,
            rsrp_max: -40.0,
            sinr_min: -20.0,
            sinr_max: 30.0,
            speed_min: 0.0,
            speed_max: 200.0,
            throughput_min: 0.0,
            throughput_max: 1000.0,
            last_updated: Utc::now(),
        }
    }
    
    fn update(&mut self, metrics: &UeMetrics) {
        self.rsrp_min = self.rsrp_min.min(metrics.rsrp_dbm);
        self.rsrp_max = self.rsrp_max.max(metrics.rsrp_dbm);
        self.sinr_min = self.sinr_min.min(metrics.sinr_db);
        self.sinr_max = self.sinr_max.max(metrics.sinr_db);
        self.speed_min = self.speed_min.min(metrics.speed_kmh);
        self.speed_max = self.speed_max.max(metrics.speed_kmh);
        self.throughput_min = self.throughput_min.min(metrics.throughput_mbps);
        self.throughput_max = self.throughput_max.max(metrics.throughput_mbps);
        self.last_updated = Utc::now();
    }
}

impl UeMetricsProcessor {
    pub async fn new(config: Arc<OptMobConfig>) -> Result<Self> {
        let metrics_history = Arc::new(RwLock::new(HashMap::new()));
        let normalization_stats = Arc::new(RwLock::new(NormalizationStats::new()));
        
        Ok(Self {
            config,
            metrics_history,
            normalization_stats,
        })
    }
    
    /// Process UE metrics and extract features
    pub async fn process_metrics(&self, metrics: &UeMetrics) -> Result<ProcessedMetrics> {
        // Update normalization statistics
        {
            let mut stats = self.normalization_stats.write().await;
            stats.update(metrics);
        }
        
        // Store metrics in history
        self.store_metrics_history(metrics).await?;
        
        // Get historical data for time-series analysis
        let history = self.get_metrics_history(&metrics.ue_id).await?;
        
        // Normalize current metrics
        let normalized = self.normalize_metrics(metrics).await?;
        
        // Calculate time-series features
        let time_series_features = self.calculate_time_series_features(&history).await?;
        
        let processed = ProcessedMetrics {
            ue_id: metrics.ue_id.clone(),
            cell_id: metrics.cell_id.clone(),
            timestamp: metrics.timestamp,
            
            // Normalized metrics
            rsrp_normalized: normalized.rsrp,
            sinr_normalized: normalized.sinr,
            speed_normalized: normalized.speed,
            throughput_normalized: normalized.throughput,
            cqi_normalized: normalized.cqi,
            phr_normalized: normalized.phr,
            ta_normalized: normalized.ta,
            load_factor: metrics.load_factor,
            
            // Time-series features
            trend_rsrp: time_series_features.trend_rsrp,
            trend_sinr: time_series_features.trend_sinr,
            variance_rsrp: time_series_features.variance_rsrp,
            variance_sinr: time_series_features.variance_sinr,
            rate_of_change_rsrp: time_series_features.rate_of_change_rsrp,
            rate_of_change_sinr: time_series_features.rate_of_change_sinr,
            stability_score: time_series_features.stability_score,
            mobility_indicator: time_series_features.mobility_indicator,
        };
        
        log::debug!("Processed metrics for UE {}: RSRP={:.2}, SINR={:.2}, Speed={:.2}",
                   metrics.ue_id, normalized.rsrp, normalized.sinr, normalized.speed);
        
        Ok(processed)
    }
    
    /// Process multiple UE metrics in batch
    pub async fn process_batch(&self, metrics_batch: &[UeMetrics]) -> Result<Vec<ProcessedMetrics>> {
        let mut processed_batch = Vec::new();
        
        for metrics in metrics_batch {
            let processed = self.process_metrics(metrics).await?;
            processed_batch.push(processed);
        }
        
        Ok(processed_batch)
    }
    
    /// Clean up old metrics history
    pub async fn cleanup_old_metrics(&self, max_age: Duration) -> Result<()> {
        let mut history = self.metrics_history.write().await;
        let cutoff_time = Utc::now() - max_age;
        
        let mut removed_count = 0;
        
        for (ue_id, metrics_queue) in history.iter_mut() {
            let initial_len = metrics_queue.len();
            
            // Remove old metrics
            while let Some(front) = metrics_queue.front() {
                if front.timestamp < cutoff_time {
                    metrics_queue.pop_front();
                } else {
                    break;
                }
            }
            
            removed_count += initial_len - metrics_queue.len();
        }
        
        // Remove empty queues
        history.retain(|_, queue| !queue.is_empty());
        
        log::debug!("Cleaned up {} old metrics entries", removed_count);
        Ok(())
    }
    
    /// Get metrics history for a UE
    pub async fn get_metrics_history(&self, ue_id: &str) -> Result<Vec<UeMetrics>> {
        let history = self.metrics_history.read().await;
        
        if let Some(metrics_queue) = history.get(ue_id) {
            Ok(metrics_queue.iter().cloned().collect())
        } else {
            Ok(Vec::new())
        }
    }
    
    /// Get statistics about metrics processing
    pub async fn get_processing_stats(&self) -> Result<ProcessingStats> {
        let history = self.metrics_history.read().await;
        
        let total_ues = history.len();
        let total_metrics = history.values().map(|q| q.len()).sum::<usize>();
        let avg_metrics_per_ue = if total_ues > 0 {
            total_metrics as f64 / total_ues as f64
        } else {
            0.0
        };
        
        let stats = ProcessingStats {
            total_ues,
            total_metrics,
            avg_metrics_per_ue,
            last_updated: Utc::now(),
        };
        
        Ok(stats)
    }
    
    async fn store_metrics_history(&self, metrics: &UeMetrics) -> Result<()> {
        let mut history = self.metrics_history.write().await;
        
        let ue_queue = history.entry(metrics.ue_id.clone())
            .or_insert_with(VecDeque::new);
        
        ue_queue.push_back(metrics.clone());
        
        // Limit queue size
        const MAX_HISTORY_SIZE: usize = 100;
        while ue_queue.len() > MAX_HISTORY_SIZE {
            ue_queue.pop_front();
        }
        
        Ok(())
    }
    
    async fn normalize_metrics(&self, metrics: &UeMetrics) -> Result<NormalizedMetrics> {
        let stats = self.normalization_stats.read().await;
        
        let normalized = NormalizedMetrics {
            rsrp: self.normalize_value(metrics.rsrp_dbm, stats.rsrp_min, stats.rsrp_max),
            sinr: self.normalize_value(metrics.sinr_db, stats.sinr_min, stats.sinr_max),
            speed: self.normalize_value(metrics.speed_kmh, stats.speed_min, stats.speed_max),
            throughput: self.normalize_value(metrics.throughput_mbps, stats.throughput_min, stats.throughput_max),
            cqi: metrics.cqi as f64 / 15.0, // CQI range is 0-15
            phr: self.normalize_value(metrics.phr_db, -23.0, 40.0), // PHR range
            ta: self.normalize_value(metrics.ta_us, 0.0, 1282.0), // TA range
        };
        
        Ok(normalized)
    }
    
    fn normalize_value(&self, value: f64, min: f64, max: f64) -> f64 {
        if max <= min {
            0.5 // Default to middle value if range is invalid
        } else {
            ((value - min) / (max - min)).clamp(0.0, 1.0)
        }
    }
    
    async fn calculate_time_series_features(&self, history: &[UeMetrics]) -> Result<TimeSeriesFeatures> {
        if history.len() < 2 {
            return Ok(TimeSeriesFeatures::default());
        }
        
        let rsrp_values: Vec<f64> = history.iter().map(|m| m.rsrp_dbm).collect();
        let sinr_values: Vec<f64> = history.iter().map(|m| m.sinr_db).collect();
        let speed_values: Vec<f64> = history.iter().map(|m| m.speed_kmh).collect();
        
        let features = TimeSeriesFeatures {
            trend_rsrp: self.calculate_trend(&rsrp_values),
            trend_sinr: self.calculate_trend(&sinr_values),
            variance_rsrp: rsrp_values.variance(),
            variance_sinr: sinr_values.variance(),
            rate_of_change_rsrp: self.calculate_rate_of_change(&rsrp_values),
            rate_of_change_sinr: self.calculate_rate_of_change(&sinr_values),
            stability_score: self.calculate_stability_score(&rsrp_values, &sinr_values),
            mobility_indicator: self.calculate_mobility_indicator(&speed_values),
        };
        
        Ok(features)
    }
    
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f64;
        let sum_x = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let sum_y = values.iter().sum::<f64>();
        let sum_xy = values.iter().enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let sum_x_squared = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = n * sum_x_squared - sum_x.powi(2);
        
        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }
    
    fn calculate_rate_of_change(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mut total_change = 0.0;
        for i in 1..values.len() {
            total_change += values[i] - values[i-1];
        }
        
        total_change / (values.len() - 1) as f64
    }
    
    fn calculate_stability_score(&self, rsrp_values: &[f64], sinr_values: &[f64]) -> f64 {
        let rsrp_stability = 1.0 / (1.0 + rsrp_values.variance());
        let sinr_stability = 1.0 / (1.0 + sinr_values.variance());
        
        (rsrp_stability + sinr_stability) / 2.0
    }
    
    fn calculate_mobility_indicator(&self, speed_values: &[f64]) -> f64 {
        if speed_values.is_empty() {
            return 0.0;
        }
        
        let avg_speed = speed_values.iter().sum::<f64>() / speed_values.len() as f64;
        let speed_variance = speed_values.variance();
        
        // High mobility = high average speed + high speed variance
        (avg_speed / 100.0 + speed_variance / 1000.0).min(1.0)
    }
}

#[derive(Debug, Clone)]
struct NormalizedMetrics {
    rsrp: f64,
    sinr: f64,
    speed: f64,
    throughput: f64,
    cqi: f64,
    phr: f64,
    ta: f64,
}

#[derive(Debug, Clone)]
struct TimeSeriesFeatures {
    trend_rsrp: f64,
    trend_sinr: f64,
    variance_rsrp: f64,
    variance_sinr: f64,
    rate_of_change_rsrp: f64,
    rate_of_change_sinr: f64,
    stability_score: f64,
    mobility_indicator: f64,
}

impl Default for TimeSeriesFeatures {
    fn default() -> Self {
        Self {
            trend_rsrp: 0.0,
            trend_sinr: 0.0,
            variance_rsrp: 0.0,
            variance_sinr: 0.0,
            rate_of_change_rsrp: 0.0,
            rate_of_change_sinr: 0.0,
            stability_score: 0.5,
            mobility_indicator: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProcessingStats {
    pub total_ues: usize,
    pub total_metrics: usize,
    pub avg_metrics_per_ue: f64,
    pub last_updated: DateTime<Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::OptMobConfig;
    
    #[tokio::test]
    async fn test_processor_creation() {
        let config = Arc::new(OptMobConfig::default());
        let processor = UeMetricsProcessor::new(config).await;
        assert!(processor.is_ok());
    }
    
    #[tokio::test]
    async fn test_metrics_processing() {
        let config = Arc::new(OptMobConfig::default());
        let processor = UeMetricsProcessor::new(config).await.unwrap();
        
        let metrics = UeMetrics::new("UE001".to_string(), "Cell001".to_string())
            .with_rsrp(-85.0)
            .with_sinr(15.0)
            .with_speed(60.0)
            .with_throughput(100.0)
            .with_cqi(12);
        
        let processed = processor.process_metrics(&metrics).await.unwrap();
        
        assert_eq!(processed.ue_id, "UE001");
        assert_eq!(processed.cell_id, "Cell001");
        assert!(processed.rsrp_normalized >= 0.0 && processed.rsrp_normalized <= 1.0);
        assert!(processed.sinr_normalized >= 0.0 && processed.sinr_normalized <= 1.0);
        assert!(processed.speed_normalized >= 0.0 && processed.speed_normalized <= 1.0);
    }
    
    #[test]
    fn test_normalize_value() {
        let config = Arc::new(OptMobConfig::default());
        let processor = UeMetricsProcessor {
            config,
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            normalization_stats: Arc::new(RwLock::new(NormalizationStats::new())),
        };
        
        // Test normalization
        let normalized = processor.normalize_value(-85.0, -140.0, -40.0);
        assert!(normalized >= 0.0 && normalized <= 1.0);
        
        // Test edge cases
        assert_eq!(processor.normalize_value(-140.0, -140.0, -40.0), 0.0);
        assert_eq!(processor.normalize_value(-40.0, -140.0, -40.0), 1.0);
        assert_eq!(processor.normalize_value(-90.0, -140.0, -40.0), 0.5);
    }
    
    #[test]
    fn test_trend_calculation() {
        let config = Arc::new(OptMobConfig::default());
        let processor = UeMetricsProcessor {
            config,
            metrics_history: Arc::new(RwLock::new(HashMap::new())),
            normalization_stats: Arc::new(RwLock::new(NormalizationStats::new())),
        };
        
        // Test increasing trend
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = processor.calculate_trend(&increasing);
        assert!(trend > 0.0);
        
        // Test decreasing trend
        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = processor.calculate_trend(&decreasing);
        assert!(trend < 0.0);
        
        // Test flat trend
        let flat = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let trend = processor.calculate_trend(&flat);
        assert!(trend.abs() < 0.001);
    }
}