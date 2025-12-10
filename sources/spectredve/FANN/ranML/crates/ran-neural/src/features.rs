//! Feature extraction for RAN data

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Timelike};

use crate::{
    NeuralError, NeuralResult, RanData, ModelType,
    FeatureConfig, NormalizationType, MissingValueStrategy,
};
use ran_core::{
    Cell, UE, PerformanceMetrics, KpiType, TimeSeries,
    GeoCoordinate, NetworkTopology,
};

/// Feature extractor for converting RAN data to neural network inputs
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// Configuration
    pub config: FeatureConfig,
    /// Normalization parameters
    pub normalization_params: Option<NormalizationParams>,
    /// Feature statistics for scaling
    pub feature_stats: HashMap<String, FeatureStats>,
    /// Last extraction timestamp
    pub last_extraction: Option<DateTime<Utc>>,
}

impl FeatureExtractor {
    /// Create a new feature extractor
    pub fn new(config: FeatureConfig) -> Self {
        Self {
            config,
            normalization_params: None,
            feature_stats: HashMap::new(),
            last_extraction: None,
        }
    }

    /// Extract features from RAN data
    pub fn extract(&self, data: &RanData) -> NeuralResult<Vec<f64>> {
        let mut features = Vec::new();
        
        for feature_name in &self.config.features {
            let value = self.extract_single_feature(feature_name, data)?;
            features.push(value);
        }

        // Apply normalization if configured
        let normalized_features = if self.config.feature_scaling {
            self.normalize_features(&features)?
        } else {
            features
        };

        Ok(normalized_features)
    }

    /// Extract a single feature from RAN data
    fn extract_single_feature(&self, feature_name: &str, data: &RanData) -> NeuralResult<f64> {
        let value = match feature_name {
            // Cell-related features
            "cell_load" => {
                data.cell.as_ref()
                    .map(|cell| cell.load_percentage())
                    .unwrap_or(0.0)
            }
            "tx_power" => {
                data.cell.as_ref()
                    .map(|cell| cell.config.tx_power)
                    .unwrap_or(0.0)
            }
            "antenna_tilt" => {
                data.cell.as_ref()
                    .map(|cell| cell.config.antenna_tilt)
                    .unwrap_or(0.0)
            }
            "azimuth" => {
                data.cell.as_ref()
                    .map(|cell| cell.config.azimuth)
                    .unwrap_or(0.0)
            }
            "bandwidth" => {
                data.cell.as_ref()
                    .map(|cell| cell.config.bandwidth)
                    .unwrap_or(0.0)
            }
            "active_ues" => {
                data.cell.as_ref()
                    .map(|cell| cell.connected_ues.len() as f64)
                    .unwrap_or(0.0)
            }

            // Performance metrics features
            "throughput" | "dl_throughput" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::AverageCellThroughput))
                    .unwrap_or(0.0)
            }
            "latency" | "user_plane_latency" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::UserPlaneLatency))
                    .unwrap_or(0.0)
            }
            "sinr" | "signal_quality" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::SignalQuality))
                    .unwrap_or(0.0)
            }
            "rsrp" | "signal_strength" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::SignalStrength))
                    .unwrap_or(-100.0)
            }
            "rb_utilization" | "resource_utilization" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::ResourceUtilization))
                    .unwrap_or(0.0)
            }
            "error_rate" | "block_error_rate" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::BlockErrorRate))
                    .unwrap_or(0.0)
            }
            "handover_success_rate" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::HandoverSuccessRate))
                    .unwrap_or(100.0)
            }

            // Temporal features
            "time_of_day" => {
                let hour = data.timestamp.hour() as f64;
                hour / 24.0 // Normalize to [0, 1]
            }
            "day_of_week" => {
                let day = data.timestamp.weekday().num_days_from_monday() as f64;
                day / 7.0 // Normalize to [0, 1]
            }
            "month" => {
                let month = data.timestamp.month() as f64;
                month / 12.0 // Normalize to [0, 1]
            }
            "hour" => {
                data.timestamp.hour() as f64
            }

            // Derived features
            "interference_level" => {
                // Calculate from signal quality and other metrics
                let sinr = data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::SignalQuality))
                    .unwrap_or(10.0);
                (30.0 - sinr).max(0.0) // Higher interference = lower SINR
            }
            "channel_quality" => {
                // Composite of SINR and error rate
                let sinr = data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::SignalQuality))
                    .unwrap_or(0.0);
                let bler = data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::BlockErrorRate))
                    .unwrap_or(0.0);
                (sinr / 30.0) * (1.0 - bler / 100.0) // Normalized quality
            }
            "network_congestion" => {
                // Average load across multiple cells if topology is available
                if let Some(topology) = &data.topology {
                    let stats = topology.get_statistics();
                    stats.average_load / 100.0
                } else {
                    data.cell.as_ref()
                        .map(|cell| cell.load_percentage() / 100.0)
                        .unwrap_or(0.0)
                }
            }

            // UE-related features
            "ue_count" => data.ues.len() as f64,
            "ue_mobility" => {
                // Average UE mobility (simplified)
                if data.ues.is_empty() {
                    0.0
                } else {
                    data.ues.iter()
                        .filter(|ue| ue.state == ran_core::UEState::Connected)
                        .count() as f64 / data.ues.len() as f64
                }
            }

            // Time series derived features
            "throughput_trend" => {
                self.extract_timeseries_trend(data, "throughput")
                    .unwrap_or(0.0)
            }
            "latency_trend" => {
                self.extract_timeseries_trend(data, "latency")
                    .unwrap_or(0.0)
            }
            "load_trend" => {
                self.extract_timeseries_trend(data, "load")
                    .unwrap_or(0.0)
            }

            // Location-based features
            "location_type" => {
                // Simplified location classification
                data.context.get("location_type")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "user_density" => {
                data.context.get("user_density")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }

            // Weather and external factors
            "weather_condition" => {
                data.context.get("weather")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "temperature" => {
                data.context.get("temperature")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(20.0)
            }

            // Service type and QoS features
            "service_type" => {
                data.context.get("service_type")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "qos_class" => {
                data.context.get("qos_class")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "priority_traffic" => {
                data.context.get("priority_traffic")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }

            // Energy and efficiency features
            "energy_efficiency" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::EnergyEfficiency))
                    .unwrap_or(0.0)
            }
            "spectral_efficiency" => {
                data.metrics.as_ref()
                    .and_then(|metrics| metrics.get_kpi_value(KpiType::SpectralEfficiency))
                    .unwrap_or(0.0)
            }

            // Handover specific features
            "source_rsrp" => {
                data.context.get("source_rsrp")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(-100.0)
            }
            "target_rsrp" => {
                data.context.get("target_rsrp")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(-100.0)
            }
            "source_sinr" => {
                data.context.get("source_sinr")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "target_sinr" => {
                data.context.get("target_sinr")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "source_load" => {
                data.context.get("source_load")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "target_load" => {
                data.context.get("target_load")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }
            "hysteresis" => {
                data.context.get("hysteresis")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(3.0)
            }
            "time_to_trigger" => {
                data.context.get("time_to_trigger")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(160.0)
            }

            // Distance and geographic features
            "distance" => {
                data.context.get("distance")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0)
            }

            // Default for unknown features
            _ => {
                tracing::warn!("Unknown feature: {}", feature_name);
                0.0
            }
        };

        // Handle missing values
        let final_value = if value.is_nan() || value.is_infinite() {
            self.handle_missing_value(feature_name, value)?
        } else {
            value
        };

        Ok(final_value)
    }

    /// Extract trend from time series data
    fn extract_timeseries_trend(&self, data: &RanData, series_name: &str) -> Option<f64> {
        if let Some(series) = data.timeseries.get(series_name) {
            if series.points.len() >= 2 {
                let first = series.points.first()?.value;
                let last = series.points.last()?.value;
                if first != 0.0 {
                    return Some((last - first) / first * 100.0); // Percentage change
                }
            }
        }
        None
    }

    /// Handle missing values according to strategy
    fn handle_missing_value(&self, feature_name: &str, _value: f64) -> NeuralResult<f64> {
        match self.config.missing_value_strategy {
            MissingValueStrategy::Zero => Ok(0.0),
            MissingValueStrategy::Mean => {
                // Use feature statistics if available
                if let Some(stats) = self.feature_stats.get(feature_name) {
                    Ok(stats.mean)
                } else {
                    Ok(0.0)
                }
            }
            MissingValueStrategy::Median => {
                if let Some(stats) = self.feature_stats.get(feature_name) {
                    Ok(stats.median)
                } else {
                    Ok(0.0)
                }
            }
            MissingValueStrategy::Mode => {
                if let Some(stats) = self.feature_stats.get(feature_name) {
                    Ok(stats.mode)
                } else {
                    Ok(0.0)
                }
            }
            _ => Ok(0.0), // Default to zero for other strategies
        }
    }

    /// Normalize features according to configuration
    fn normalize_features(&self, features: &[f64]) -> NeuralResult<Vec<f64>> {
        if let Some(ref params) = self.normalization_params {
            match self.config.normalization {
                NormalizationType::StandardScore => {
                    Ok(features.iter()
                        .zip(params.means.iter())
                        .zip(params.stds.iter())
                        .map(|((&value, &mean), &std)| {
                            if std != 0.0 {
                                (value - mean) / std
                            } else {
                                0.0
                            }
                        })
                        .collect())
                }
                NormalizationType::MinMax => {
                    Ok(features.iter()
                        .zip(params.mins.iter())
                        .zip(params.maxs.iter())
                        .map(|((&value, &min), &max)| {
                            if max != min {
                                (value - min) / (max - min)
                            } else {
                                0.0
                            }
                        })
                        .collect())
                }
                NormalizationType::RobustScaling => {
                    Ok(features.iter()
                        .zip(params.medians.iter())
                        .zip(params.iqrs.iter())
                        .map(|((&value, &median), &iqr)| {
                            if iqr != 0.0 {
                                (value - median) / iqr
                            } else {
                                0.0
                            }
                        })
                        .collect())
                }
                NormalizationType::UnitVector => {
                    let norm = features.iter().map(|x| x * x).sum::<f64>().sqrt();
                    if norm != 0.0 {
                        Ok(features.iter().map(|&x| x / norm).collect())
                    } else {
                        Ok(features.to_vec())
                    }
                }
                NormalizationType::None => Ok(features.to_vec()),
            }
        } else {
            Ok(features.to_vec())
        }
    }

    /// Fit normalization parameters from training data
    pub fn fit_normalization(&mut self, training_features: &[Vec<f64>]) -> NeuralResult<()> {
        if training_features.is_empty() {
            return Err(NeuralError::FeatureExtraction("No training data provided".to_string()));
        }

        let num_features = training_features[0].len();
        let mut means = vec![0.0; num_features];
        let mut mins = vec![f64::INFINITY; num_features];
        let mut maxs = vec![f64::NEG_INFINITY; num_features];
        let mut medians = vec![0.0; num_features];

        // Calculate means, mins, maxs
        for features in training_features {
            for (i, &value) in features.iter().enumerate() {
                means[i] += value;
                mins[i] = mins[i].min(value);
                maxs[i] = maxs[i].max(value);
            }
        }

        for mean in &mut means {
            *mean /= training_features.len() as f64;
        }

        // Calculate standard deviations
        let mut stds = vec![0.0; num_features];
        for features in training_features {
            for (i, &value) in features.iter().enumerate() {
                stds[i] += (value - means[i]).powi(2);
            }
        }

        for (i, std) in stds.iter_mut().enumerate() {
            *std = (*std / training_features.len() as f64).sqrt();
        }

        // Calculate medians and IQRs (simplified)
        for i in 0..num_features {
            let mut values: Vec<f64> = training_features.iter()
                .map(|features| features[i])
                .collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let mid = values.len() / 2;
            medians[i] = if values.len() % 2 == 0 {
                (values[mid - 1] + values[mid]) / 2.0
            } else {
                values[mid]
            };
        }

        // Calculate IQRs (simplified - using standard deviation as proxy)
        let iqrs = stds.clone();

        self.normalization_params = Some(NormalizationParams {
            means,
            stds,
            mins,
            maxs,
            medians,
            iqrs,
        });

        Ok(())
    }

    /// Update feature statistics
    pub fn update_feature_stats(&mut self, feature_name: String, stats: FeatureStats) {
        self.feature_stats.insert(feature_name, stats);
    }

    /// Get feature names
    pub fn feature_names(&self) -> &[String] {
        &self.config.features
    }

    /// Get number of features
    pub fn num_features(&self) -> usize {
        self.config.features.len()
    }
}

/// Normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Feature means
    pub means: Vec<f64>,
    /// Feature standard deviations
    pub stds: Vec<f64>,
    /// Feature minimums
    pub mins: Vec<f64>,
    /// Feature maximums
    pub maxs: Vec<f64>,
    /// Feature medians
    pub medians: Vec<f64>,
    /// Feature IQRs
    pub iqrs: Vec<f64>,
}

/// Feature statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureStats {
    /// Mean value
    pub mean: f64,
    /// Median value
    pub median: f64,
    /// Mode value
    pub mode: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Number of samples
    pub count: usize,
}

impl Default for FeatureStats {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            mode: 0.0,
            std_dev: 1.0,
            min: 0.0,
            max: 1.0,
            count: 0,
        }
    }
}

/// Feature vector with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// Feature values
    pub values: Vec<f64>,
    /// Feature names
    pub names: Vec<String>,
    /// Extraction timestamp
    pub timestamp: DateTime<Utc>,
    /// Source data identifier
    pub source_id: Option<String>,
    /// Feature quality score
    pub quality_score: f64,
    /// Missing feature count
    pub missing_count: usize,
}

impl FeatureVector {
    /// Create a new feature vector
    pub fn new(values: Vec<f64>, names: Vec<String>) -> Self {
        let missing_count = values.iter()
            .filter(|&&v| v.is_nan() || v.is_infinite())
            .count();
        
        let quality_score = 1.0 - (missing_count as f64 / values.len() as f64);

        Self {
            values,
            names,
            timestamp: Utc::now(),
            source_id: None,
            quality_score,
            missing_count,
        }
    }

    /// Get feature by name
    pub fn get_feature(&self, name: &str) -> Option<f64> {
        self.names.iter()
            .position(|n| n == name)
            .and_then(|i| self.values.get(i))
            .copied()
    }

    /// Set feature by name
    pub fn set_feature(&mut self, name: &str, value: f64) -> bool {
        if let Some(i) = self.names.iter().position(|n| n == name) {
            self.values[i] = value;
            true
        } else {
            false
        }
    }

    /// Get feature count
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Validate feature vector
    pub fn validate(&self) -> NeuralResult<()> {
        if self.values.len() != self.names.len() {
            return Err(NeuralError::FeatureExtraction(
                "Values and names length mismatch".to_string()
            ));
        }

        for (i, &value) in self.values.iter().enumerate() {
            if value.is_nan() {
                return Err(NeuralError::FeatureExtraction(
                    format!("NaN value in feature '{}'", self.names[i])
                ));
            }
            if value.is_infinite() {
                return Err(NeuralError::FeatureExtraction(
                    format!("Infinite value in feature '{}'", self.names[i])
                ));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ran_core::{Cell, GeoCoordinate};

    #[test]
    fn test_feature_extractor_creation() {
        let config = FeatureConfig::default();
        let extractor = FeatureExtractor::new(config);
        
        assert_eq!(extractor.feature_names().len(), 3);
        assert!(extractor.normalization_params.is_none());
    }

    #[test]
    fn test_basic_feature_extraction() {
        let config = FeatureConfig {
            features: vec!["cell_load".to_string(), "time_of_day".to_string()],
            normalization: NormalizationType::None,
            missing_value_strategy: MissingValueStrategy::Zero,
            feature_scaling: false,
            feature_selection: false,
        };
        
        let extractor = FeatureExtractor::new(config);
        let mut data = RanData::new();
        
        // Add a test cell
        let location = GeoCoordinate::new(52.520008, 13.404954, None);
        let gnodeb_id = uuid::Uuid::new_v4();
        let mut cell = Cell::new(
            "Test Cell".to_string(),
            location,
            gnodeb_id,
            123,
            "12345".to_string(),
        );
        
        // Add some UEs to create load
        cell.add_ue(uuid::Uuid::new_v4()).unwrap();
        cell.add_ue(uuid::Uuid::new_v4()).unwrap();
        
        data.cell = Some(cell);
        
        let features = extractor.extract(&data).unwrap();
        assert_eq!(features.len(), 2);
        assert!(features[0] > 0.0); // cell_load should be > 0
        assert!(features[1] >= 0.0 && features[1] <= 1.0); // time_of_day normalized
    }

    #[test]
    fn test_missing_value_handling() {
        let config = FeatureConfig {
            features: vec!["unknown_feature".to_string()],
            normalization: NormalizationType::None,
            missing_value_strategy: MissingValueStrategy::Zero,
            feature_scaling: false,
            feature_selection: false,
        };
        
        let extractor = FeatureExtractor::new(config);
        let data = RanData::new();
        
        let features = extractor.extract(&data).unwrap();
        assert_eq!(features.len(), 1);
        assert_eq!(features[0], 0.0);
    }

    #[test]
    fn test_feature_vector() {
        let values = vec![1.0, 2.0, 3.0];
        let names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        
        let mut fv = FeatureVector::new(values, names);
        assert_eq!(fv.len(), 3);
        assert_eq!(fv.quality_score, 1.0);
        assert_eq!(fv.missing_count, 0);
        
        assert_eq!(fv.get_feature("f2"), Some(2.0));
        assert!(fv.set_feature("f1", 10.0));
        assert_eq!(fv.get_feature("f1"), Some(10.0));
        
        assert!(fv.validate().is_ok());
    }

    #[test]
    fn test_normalization_params() {
        let params = NormalizationParams {
            means: vec![1.0, 2.0],
            stds: vec![0.5, 1.0],
            mins: vec![0.0, 0.0],
            maxs: vec![2.0, 4.0],
            medians: vec![1.0, 2.0],
            iqrs: vec![0.5, 1.0],
        };
        
        assert_eq!(params.means.len(), 2);
        assert_eq!(params.stds.len(), 2);
    }

    #[test]
    fn test_feature_stats() {
        let stats = FeatureStats {
            mean: 5.0,
            median: 4.0,
            mode: 3.0,
            std_dev: 2.0,
            min: 1.0,
            max: 10.0,
            count: 100,
        };
        
        assert_eq!(stats.mean, 5.0);
        assert_eq!(stats.count, 100);
    }
}