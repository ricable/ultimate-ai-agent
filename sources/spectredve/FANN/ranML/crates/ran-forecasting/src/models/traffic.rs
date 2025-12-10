//! Traffic prediction models for RAN forecasting

use std::collections::HashMap;
use crate::{
    error::{ForecastError, ForecastResult},
    data::{RanTimeSeries, ModelTrainingData, ModelForecastData, ForecastHorizon},
    adapters::{NeuroAdapter, RanModelAdapter},
    RanForecastingModel,
};
use ran_core::KpiType;

/// Traffic predictor for RAN network traffic forecasting
#[derive(Debug)]
pub struct TrafficPredictor {
    /// Underlying forecasting model
    model: Box<dyn RanForecastingModel>,
    /// Predictor configuration
    config: TrafficPredictorConfig,
    /// Training history
    training_metrics: TrafficTrainingMetrics,
}

impl TrafficPredictor {
    /// Create a new traffic predictor
    pub fn new(model: Box<dyn RanForecastingModel>, config: TrafficPredictorConfig) -> Self {
        Self {
            model,
            config,
            training_metrics: TrafficTrainingMetrics::default(),
        }
    }

    /// Create from model type string
    pub fn from_config(model_type: &str, params: HashMap<String, String>) -> ForecastResult<Self> {
        let input_size = params.get("input_size")
            .and_then(|s| s.parse().ok())
            .unwrap_or(168); // Default: 1 week history

        let horizon = params.get("horizon")
            .and_then(|s| s.parse().ok())
            .unwrap_or(24); // Default: 24 hour forecast

        let model = NeuroAdapter::from_name(model_type, input_size, horizon, params)?;

        let config = TrafficPredictorConfig {
            traffic_type: TrafficType::Total,
            prediction_interval: PredictionInterval::Hourly,
            include_seasonality: true,
            include_trends: true,
            include_external_factors: false,
            confidence_level: 0.95,
            min_training_samples: 168, // 1 week minimum
        };

        Ok(Self::new(model, config))
    }

    /// Create a builder for traffic predictor
    pub fn builder() -> TrafficPredictorBuilder {
        TrafficPredictorBuilder::new()
    }

    /// Predict traffic for specific KPI types
    pub fn predict_kpi_traffic(&self, kpi_type: KpiType, data: &ModelTrainingData) -> ForecastResult<ModelForecastData> {
        // Adjust prediction based on KPI characteristics
        let mut forecast = self.model.predict(data)?;

        // Apply KPI-specific post-processing
        match kpi_type {
            KpiType::AverageCellThroughput => {
                // Ensure non-negative throughput
                for value in &mut forecast.values {
                    *value = value.max(0.0);
                }
                
                // Apply typical throughput ranges (in Mbps)
                for value in &mut forecast.values {
                    *value = value.clamp(0.0, 1000.0);
                }
            },
            KpiType::ResourceUtilization => {
                // Clamp to 0-100% range
                for value in &mut forecast.values {
                    *value = value.clamp(0.0, 100.0);
                }
            },
            KpiType::UserPlaneLatency => {
                // Ensure positive latency values
                for value in &mut forecast.values {
                    *value = value.max(0.1); // Minimum 0.1ms
                }
            },
            _ => {
                // Default processing
            }
        }

        Ok(forecast)
    }

    /// Predict traffic anomalies
    pub fn predict_traffic_anomalies(&self, data: &ModelTrainingData) -> ForecastResult<Vec<f64>> {
        let forecast = self.model.predict(data)?;
        let actual_values = &data.values[data.values.len() - forecast.values.len()..];
        
        // Calculate residuals
        let residuals: Vec<f64> = actual_values.iter()
            .zip(forecast.values.iter())
            .map(|(&actual, &predicted)| (actual - predicted).abs())
            .collect();

        // Calculate anomaly scores based on residuals
        let mean_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let residual_std = {
            let variance = residuals.iter()
                .map(|&r| (r - mean_residual).powi(2))
                .sum::<f64>() / residuals.len() as f64;
            variance.sqrt()
        };

        let anomaly_scores = residuals.iter()
            .map(|&residual| {
                if residual_std > 0.0 {
                    (residual - mean_residual) / residual_std
                } else {
                    0.0
                }
            })
            .collect();

        Ok(anomaly_scores)
    }

    /// Get traffic pattern analysis
    pub fn analyze_traffic_patterns(&self, timeseries: &RanTimeSeries) -> ForecastResult<TrafficPatternAnalysis> {
        if timeseries.is_empty() {
            return Err(ForecastError::data_error("Empty time series"));
        }

        let values = timeseries.values();
        let statistics = timeseries.statistics();

        // Detect daily patterns
        let daily_pattern = self.detect_daily_pattern(&values)?;
        
        // Detect weekly patterns
        let weekly_pattern = self.detect_weekly_pattern(&values)?;
        
        // Detect trend
        let trend = self.detect_trend(&values)?;
        
        // Detect seasonality
        let seasonality = self.detect_seasonality(&values)?;

        Ok(TrafficPatternAnalysis {
            daily_pattern,
            weekly_pattern,
            trend,
            seasonality,
            volatility: statistics.std_dev / statistics.mean,
            peak_hours: self.identify_peak_hours(&values),
            low_hours: self.identify_low_hours(&values),
        })
    }

    /// Detect daily traffic patterns
    fn detect_daily_pattern(&self, values: &[f64]) -> ForecastResult<DailyPattern> {
        if values.len() < 24 {
            return Ok(DailyPattern::Insufficient);
        }

        // Group by hour of day (assuming hourly data)
        let mut hourly_averages = vec![0.0; 24];
        let mut hourly_counts = vec![0; 24];

        for (i, &value) in values.iter().enumerate() {
            let hour = i % 24;
            hourly_averages[hour] += value;
            hourly_counts[hour] += 1;
        }

        // Calculate averages
        for (avg, count) in hourly_averages.iter_mut().zip(hourly_counts.iter()) {
            if *count > 0 {
                *avg /= *count as f64;
            }
        }

        // Analyze pattern characteristics
        let max_hour = hourly_averages.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let min_hour = hourly_averages.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let peak_to_valley_ratio = if hourly_averages[min_hour] > 0.0 {
            hourly_averages[max_hour] / hourly_averages[min_hour]
        } else {
            f64::INFINITY
        };

        // Classify pattern based on characteristics
        let pattern_type = match peak_to_valley_ratio {
            r if r > 3.0 => DailyPattern::HighVariation { peak_hour: max_hour, low_hour: min_hour },
            r if r > 1.5 => DailyPattern::ModerateVariation { peak_hour: max_hour, low_hour: min_hour },
            _ => DailyPattern::LowVariation,
        };

        Ok(pattern_type)
    }

    /// Detect weekly traffic patterns
    fn detect_weekly_pattern(&self, values: &[f64]) -> ForecastResult<WeeklyPattern> {
        if values.len() < 168 { // Less than 1 week
            return Ok(WeeklyPattern::Insufficient);
        }

        // Group by day of week (assuming hourly data)
        let mut daily_averages = vec![0.0; 7];
        let mut daily_counts = vec![0; 7];

        for (i, &value) in values.iter().enumerate() {
            let day = (i / 24) % 7;
            daily_averages[day] += value;
            daily_counts[day] += 1;
        }

        // Calculate averages
        for (avg, count) in daily_averages.iter_mut().zip(daily_counts.iter()) {
            if *count > 0 {
                *avg /= *count as f64;
            }
        }

        // Analyze weekday vs weekend patterns
        let weekday_avg = daily_averages[0..5].iter().sum::<f64>() / 5.0;
        let weekend_avg = daily_averages[5..7].iter().sum::<f64>() / 2.0;

        let weekend_ratio = if weekday_avg > 0.0 {
            weekend_avg / weekday_avg
        } else {
            1.0
        };

        let pattern_type = match weekend_ratio {
            r if r < 0.7 => WeeklyPattern::WeekdayDominated,
            r if r > 1.3 => WeeklyPattern::WeekendDominated,
            _ => WeeklyPattern::Balanced,
        };

        Ok(pattern_type)
    }

    /// Detect trend in traffic data
    fn detect_trend(&self, values: &[f64]) -> ForecastResult<TrendType> {
        if values.len() < 10 {
            return Ok(TrendType::Insufficient);
        }

        // Simple linear regression to detect trend
        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum.powi(2));

        let trend_type = match slope {
            s if s > 0.01 => TrendType::Increasing,
            s if s < -0.01 => TrendType::Decreasing,
            _ => TrendType::Stable,
        };

        Ok(trend_type)
    }

    /// Detect seasonality in traffic data
    fn detect_seasonality(&self, values: &[f64]) -> ForecastResult<SeasonalityType> {
        if values.len() < 48 { // Less than 2 days
            return Ok(SeasonalityType::None);
        }

        // Check for daily seasonality (24-hour cycle)
        let daily_score = self.calculate_periodicity_score(values, 24);
        
        // Check for weekly seasonality (168-hour cycle)
        let weekly_score = if values.len() >= 336 { // At least 2 weeks
            self.calculate_periodicity_score(values, 168)
        } else {
            0.0
        };

        match (daily_score > 0.3, weekly_score > 0.3) {
            (true, true) => Ok(SeasonalityType::Both),
            (true, false) => Ok(SeasonalityType::Daily),
            (false, true) => Ok(SeasonalityType::Weekly),
            (false, false) => Ok(SeasonalityType::None),
        }
    }

    /// Calculate periodicity score for a given period
    fn calculate_periodicity_score(&self, values: &[f64], period: usize) -> f64 {
        if values.len() < period * 2 {
            return 0.0;
        }

        let num_cycles = values.len() / period;
        let mut correlations = Vec::new();

        for cycle in 1..num_cycles {
            let mut correlation = 0.0;
            let mut count = 0;

            for i in 0..period {
                if i + cycle * period < values.len() {
                    correlation += values[i] * values[i + cycle * period];
                    count += 1;
                }
            }

            if count > 0 {
                correlations.push(correlation / count as f64);
            }
        }

        if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        }
    }

    /// Identify peak traffic hours
    fn identify_peak_hours(&self, values: &[f64]) -> Vec<usize> {
        if values.len() < 24 {
            return Vec::new();
        }

        let mut hourly_averages = vec![0.0; 24];
        let mut hourly_counts = vec![0; 24];

        for (i, &value) in values.iter().enumerate() {
            let hour = i % 24;
            hourly_averages[hour] += value;
            hourly_counts[hour] += 1;
        }

        for (avg, count) in hourly_averages.iter_mut().zip(hourly_counts.iter()) {
            if *count > 0 {
                *avg /= *count as f64;
            }
        }

        let mean = hourly_averages.iter().sum::<f64>() / 24.0;
        let std_dev = {
            let variance = hourly_averages.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / 24.0;
            variance.sqrt()
        };

        let threshold = mean + std_dev;

        hourly_averages.iter()
            .enumerate()
            .filter(|(_, &avg)| avg > threshold)
            .map(|(hour, _)| hour)
            .collect()
    }

    /// Identify low traffic hours
    fn identify_low_hours(&self, values: &[f64]) -> Vec<usize> {
        if values.len() < 24 {
            return Vec::new();
        }

        let mut hourly_averages = vec![0.0; 24];
        let mut hourly_counts = vec![0; 24];

        for (i, &value) in values.iter().enumerate() {
            let hour = i % 24;
            hourly_averages[hour] += value;
            hourly_counts[hour] += 1;
        }

        for (avg, count) in hourly_averages.iter_mut().zip(hourly_counts.iter()) {
            if *count > 0 {
                *avg /= *count as f64;
            }
        }

        let mean = hourly_averages.iter().sum::<f64>() / 24.0;
        let std_dev = {
            let variance = hourly_averages.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / 24.0;
            variance.sqrt()
        };

        let threshold = mean - std_dev;

        hourly_averages.iter()
            .enumerate()
            .filter(|(_, &avg)| avg < threshold)
            .map(|(hour, _)| hour)
            .collect()
    }

    /// Get training metrics
    pub fn get_training_metrics(&self) -> &TrafficTrainingMetrics {
        &self.training_metrics
    }
}

impl RanForecastingModel for TrafficPredictor {
    fn model_name(&self) -> &str {
        "TrafficPredictor"
    }

    fn fit(&mut self, data: &ModelTrainingData) -> ForecastResult<()> {
        let start_time = std::time::Instant::now();
        
        // Validate data for traffic prediction
        if data.values.len() < self.config.min_training_samples {
            return Err(ForecastError::training_error(format!(
                "Insufficient training data: {} samples, need at least {}",
                data.values.len(),
                self.config.min_training_samples
            )));
        }

        // Check for non-negative traffic values
        if data.values.iter().any(|&v| v < 0.0) {
            return Err(ForecastError::data_error("Traffic values cannot be negative"));
        }

        // Train underlying model
        self.model.fit(data)?;

        // Update training metrics
        self.training_metrics.training_time = start_time.elapsed();
        self.training_metrics.training_samples = data.values.len();
        self.training_metrics.last_training = chrono::Utc::now();

        tracing::info!(
            "Traffic predictor trained on {} samples in {:?}",
            data.values.len(),
            self.training_metrics.training_time
        );

        Ok(())
    }

    fn predict(&self, data: &ModelTrainingData) -> ForecastResult<ModelForecastData> {
        self.model.predict(data)
    }

    fn predict_future(&self, horizon: ForecastHorizon) -> ForecastResult<ModelForecastData> {
        self.model.predict_future(horizon)
    }

    fn update(&mut self, data: &ModelTrainingData) -> ForecastResult<()> {
        self.model.update(data)
    }

    fn reset(&mut self) -> ForecastResult<()> {
        self.model.reset()
    }

    fn get_parameters(&self) -> HashMap<String, String> {
        let mut params = self.model.get_parameters();
        params.insert("predictor_type".to_string(), "TrafficPredictor".to_string());
        params.insert("traffic_type".to_string(), format!("{:?}", self.config.traffic_type));
        params.insert("include_seasonality".to_string(), self.config.include_seasonality.to_string());
        params.insert("include_trends".to_string(), self.config.include_trends.to_string());
        params
    }

    fn supports_online_learning(&self) -> bool {
        self.model.supports_online_learning()
    }

    fn supports_multivariate(&self) -> bool {
        self.model.supports_multivariate()
    }
}

/// Configuration for traffic predictor
#[derive(Debug, Clone)]
pub struct TrafficPredictorConfig {
    /// Type of traffic to predict
    pub traffic_type: TrafficType,
    /// Prediction time interval
    pub prediction_interval: PredictionInterval,
    /// Include seasonal components
    pub include_seasonality: bool,
    /// Include trend components
    pub include_trends: bool,
    /// Include external factors
    pub include_external_factors: bool,
    /// Confidence level for predictions
    pub confidence_level: f64,
    /// Minimum training samples required
    pub min_training_samples: usize,
}

/// Types of traffic to predict
#[derive(Debug, Clone, Copy)]
pub enum TrafficType {
    /// Total traffic volume
    Total,
    /// Uplink traffic
    Uplink,
    /// Downlink traffic
    Downlink,
    /// Data traffic only
    Data,
    /// Voice traffic only
    Voice,
    /// Video traffic only
    Video,
    /// Peak hour traffic
    Peak,
    /// Off-peak traffic
    OffPeak,
}

/// Prediction time intervals
#[derive(Debug, Clone, Copy)]
pub enum PredictionInterval {
    /// Minute-level predictions
    Minutely,
    /// Hour-level predictions
    Hourly,
    /// Day-level predictions
    Daily,
    /// Week-level predictions
    Weekly,
}

/// Training metrics for traffic predictor
#[derive(Debug, Clone, Default)]
pub struct TrafficTrainingMetrics {
    /// Training duration
    pub training_time: std::time::Duration,
    /// Number of training samples
    pub training_samples: usize,
    /// Last training timestamp
    pub last_training: chrono::DateTime<chrono::Utc>,
    /// Training accuracy metrics
    pub accuracy_metrics: HashMap<String, f64>,
}

/// Traffic pattern analysis results
#[derive(Debug, Clone)]
pub struct TrafficPatternAnalysis {
    /// Daily traffic pattern
    pub daily_pattern: DailyPattern,
    /// Weekly traffic pattern
    pub weekly_pattern: WeeklyPattern,
    /// Traffic trend
    pub trend: TrendType,
    /// Seasonality type
    pub seasonality: SeasonalityType,
    /// Traffic volatility (coefficient of variation)
    pub volatility: f64,
    /// Peak traffic hours
    pub peak_hours: Vec<usize>,
    /// Low traffic hours
    pub low_hours: Vec<usize>,
}

/// Daily traffic patterns
#[derive(Debug, Clone)]
pub enum DailyPattern {
    Insufficient,
    HighVariation { peak_hour: usize, low_hour: usize },
    ModerateVariation { peak_hour: usize, low_hour: usize },
    LowVariation,
}

/// Weekly traffic patterns
#[derive(Debug, Clone)]
pub enum WeeklyPattern {
    Insufficient,
    WeekdayDominated,
    WeekendDominated,
    Balanced,
}

/// Traffic trend types
#[derive(Debug, Clone)]
pub enum TrendType {
    Insufficient,
    Increasing,
    Decreasing,
    Stable,
}

/// Seasonality types
#[derive(Debug, Clone)]
pub enum SeasonalityType {
    None,
    Daily,
    Weekly,
    Both,
}

/// Builder for traffic predictor
pub struct TrafficPredictorBuilder {
    model_type: Option<String>,
    model_params: HashMap<String, String>,
    config: TrafficPredictorConfig,
}

impl TrafficPredictorBuilder {
    pub fn new() -> Self {
        Self {
            model_type: None,
            model_params: HashMap::new(),
            config: TrafficPredictorConfig {
                traffic_type: TrafficType::Total,
                prediction_interval: PredictionInterval::Hourly,
                include_seasonality: true,
                include_trends: true,
                include_external_factors: false,
                confidence_level: 0.95,
                min_training_samples: 168,
            },
        }
    }

    pub fn model_type(mut self, model_type: &str) -> Self {
        self.model_type = Some(model_type.to_string());
        self
    }

    pub fn horizon(mut self, horizon: usize) -> Self {
        self.model_params.insert("horizon".to_string(), horizon.to_string());
        self
    }

    pub fn input_window(mut self, window: usize) -> Self {
        self.model_params.insert("input_size".to_string(), window.to_string());
        self
    }

    pub fn traffic_type(mut self, traffic_type: TrafficType) -> Self {
        self.config.traffic_type = traffic_type;
        self
    }

    pub fn prediction_interval(mut self, interval: PredictionInterval) -> Self {
        self.config.prediction_interval = interval;
        self
    }

    pub fn include_seasonality(mut self, include: bool) -> Self {
        self.config.include_seasonality = include;
        self
    }

    pub fn include_trends(mut self, include: bool) -> Self {
        self.config.include_trends = include;
        self
    }

    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level;
        self
    }

    pub fn build(self) -> ForecastResult<TrafficPredictor> {
        let model_type = self.model_type.unwrap_or_else(|| "dlinear".to_string());
        TrafficPredictor::from_config(&model_type, self.model_params)
    }
}

impl Default for TrafficPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_traffic_predictor_config() {
        let config = TrafficPredictorConfig {
            traffic_type: TrafficType::Total,
            prediction_interval: PredictionInterval::Hourly,
            include_seasonality: true,
            include_trends: true,
            include_external_factors: false,
            confidence_level: 0.95,
            min_training_samples: 168,
        };

        assert!(matches!(config.traffic_type, TrafficType::Total));
        assert!(matches!(config.prediction_interval, PredictionInterval::Hourly));
        assert!(config.include_seasonality);
        assert_eq!(config.confidence_level, 0.95);
    }

    #[test]
    fn test_traffic_predictor_builder() {
        let builder = TrafficPredictor::builder()
            .model_type("dlinear")
            .horizon(24)
            .input_window(168)
            .traffic_type(TrafficType::Data)
            .include_seasonality(true);

        // Test builder configuration
        assert_eq!(builder.model_type, Some("dlinear".to_string()));
        assert_eq!(builder.model_params.get("horizon"), Some(&"24".to_string()));
        assert_eq!(builder.model_params.get("input_size"), Some(&"168".to_string()));
        assert!(matches!(builder.config.traffic_type, TrafficType::Data));
    }

    #[test]
    fn test_traffic_types() {
        use std::mem::discriminant;
        
        let types = [
            TrafficType::Total,
            TrafficType::Uplink,
            TrafficType::Downlink,
            TrafficType::Data,
            TrafficType::Voice,
            TrafficType::Video,
            TrafficType::Peak,
            TrafficType::OffPeak,
        ];
        
        // Test that all variants are different
        for i in 0..types.len() {
            for j in (i + 1)..types.len() {
                assert_ne!(discriminant(&types[i]), discriminant(&types[j]));
            }
        }
    }

    #[test]
    fn test_pattern_analysis_enums() {
        // Test daily patterns
        let daily = DailyPattern::HighVariation { peak_hour: 18, low_hour: 3 };
        match daily {
            DailyPattern::HighVariation { peak_hour, low_hour } => {
                assert_eq!(peak_hour, 18);
                assert_eq!(low_hour, 3);
            },
            _ => panic!("Wrong pattern type"),
        }

        // Test weekly patterns
        assert!(matches!(WeeklyPattern::WeekdayDominated, WeeklyPattern::WeekdayDominated));
        assert!(matches!(WeeklyPattern::WeekendDominated, WeeklyPattern::WeekendDominated));
        assert!(matches!(WeeklyPattern::Balanced, WeeklyPattern::Balanced));

        // Test trend types
        assert!(matches!(TrendType::Increasing, TrendType::Increasing));
        assert!(matches!(TrendType::Decreasing, TrendType::Decreasing));
        assert!(matches!(TrendType::Stable, TrendType::Stable));

        // Test seasonality types
        assert!(matches!(SeasonalityType::Daily, SeasonalityType::Daily));
        assert!(matches!(SeasonalityType::Weekly, SeasonalityType::Weekly));
        assert!(matches!(SeasonalityType::Both, SeasonalityType::Both));
        assert!(matches!(SeasonalityType::None, SeasonalityType::None));
    }

    #[test]
    fn test_training_metrics() {
        let metrics = TrafficTrainingMetrics::default();
        assert_eq!(metrics.training_samples, 0);
        assert!(metrics.accuracy_metrics.is_empty());
        assert_eq!(metrics.training_time, std::time::Duration::from_secs(0));
    }
}