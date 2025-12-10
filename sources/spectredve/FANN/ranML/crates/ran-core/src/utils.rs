//! Utility functions and common operations for RAN systems

use std::collections::HashMap;
use std::f64::consts::PI;

use chrono::{DateTime, Utc, Duration};
use ndarray::{Array1, Array2};
use num_traits::Float;
use serde::{Deserialize, Serialize};

use crate::{GeoCoordinate, TimeSeries, TimeSeriesPoint, error::{RanError, RanResult}};

/// RF propagation models for path loss calculation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PropagationModel {
    /// Free space path loss
    FreeSpace,
    /// Okumura-Hata model for urban environments
    OkumuraHata,
    /// COST-231 Hata model
    Cost231Hata,
    /// 3GPP Urban Macro (UMa) model
    ThreeGPPUrbanMacro,
    /// 3GPP Urban Micro (UMi) model
    ThreeGPPUrbanMicro,
    /// 3GPP Rural Macro (RMa) model
    ThreeGPPRuralMacro,
    /// Two-ray ground reflection model
    TwoRayGround,
}

/// RF environment types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnvironmentType {
    /// Dense urban environment
    DenseUrban,
    /// Urban environment
    Urban,
    /// Suburban environment
    Suburban,
    /// Rural environment
    Rural,
    /// Indoor environment
    Indoor,
}

/// RF utilities for radio propagation calculations
pub struct RfUtils;

impl RfUtils {
    /// Calculate path loss using specified propagation model
    pub fn calculate_path_loss(
        model: PropagationModel,
        distance_km: f64,
        frequency_mhz: f64,
        tx_height_m: f64,
        rx_height_m: f64,
        environment: EnvironmentType,
    ) -> RanResult<f64> {
        if distance_km <= 0.0 || frequency_mhz <= 0.0 {
            return Err(RanError::validation(
                "distance_or_frequency",
                "Distance and frequency must be positive",
            ));
        }

        let path_loss = match model {
            PropagationModel::FreeSpace => {
                Self::free_space_path_loss(distance_km, frequency_mhz)
            }
            PropagationModel::OkumuraHata => {
                Self::okumura_hata_path_loss(distance_km, frequency_mhz, tx_height_m, rx_height_m, environment)
            }
            PropagationModel::Cost231Hata => {
                Self::cost231_hata_path_loss(distance_km, frequency_mhz, tx_height_m, rx_height_m, environment)
            }
            PropagationModel::ThreeGPPUrbanMacro => {
                Self::three_gpp_uma_path_loss(distance_km, frequency_mhz, tx_height_m, rx_height_m)
            }
            PropagationModel::ThreeGPPUrbanMicro => {
                Self::three_gpp_umi_path_loss(distance_km, frequency_mhz, tx_height_m, rx_height_m)
            }
            PropagationModel::ThreeGPPRuralMacro => {
                Self::three_gpp_rma_path_loss(distance_km, frequency_mhz, tx_height_m, rx_height_m)
            }
            PropagationModel::TwoRayGround => {
                Self::two_ray_ground_path_loss(distance_km, frequency_mhz, tx_height_m, rx_height_m)
            }
        };

        Ok(path_loss)
    }

    /// Free space path loss in dB
    fn free_space_path_loss(distance_km: f64, frequency_mhz: f64) -> f64 {
        32.45 + 20.0 * distance_km.log10() + 20.0 * frequency_mhz.log10()
    }

    /// Okumura-Hata path loss model
    fn okumura_hata_path_loss(
        distance_km: f64,
        frequency_mhz: f64,
        tx_height_m: f64,
        rx_height_m: f64,
        environment: EnvironmentType,
    ) -> f64 {
        let a_hm = if frequency_mhz >= 400.0 {
            3.2 * (11.75 * rx_height_m).log10().powi(2) - 4.97
        } else {
            (1.1 * frequency_mhz.log10() - 0.7) * rx_height_m 
                - (1.56 * frequency_mhz.log10() - 0.8)
        };

        let path_loss = 69.55 + 26.16 * frequency_mhz.log10() - 13.82 * tx_height_m.log10()
            - a_hm + (44.9 - 6.55 * tx_height_m.log10()) * distance_km.log10();

        // Environment correction factors
        match environment {
            EnvironmentType::DenseUrban => path_loss + 3.0,
            EnvironmentType::Urban => path_loss,
            EnvironmentType::Suburban => path_loss - 2.0 * (frequency_mhz / 28.0).log10().powi(2) - 5.4,
            EnvironmentType::Rural => path_loss - 4.78 * frequency_mhz.log10().powi(2) 
                + 18.33 * frequency_mhz.log10() - 40.94,
            EnvironmentType::Indoor => path_loss + 20.0, // Simplified indoor correction
        }
    }

    /// COST-231 Hata path loss model
    fn cost231_hata_path_loss(
        distance_km: f64,
        frequency_mhz: f64,
        tx_height_m: f64,
        rx_height_m: f64,
        environment: EnvironmentType,
    ) -> f64 {
        let a_hm = 3.2 * (11.75 * rx_height_m).log10().powi(2) - 4.97;
        let cm = match environment {
            EnvironmentType::DenseUrban => 3.0,
            _ => 0.0,
        };

        46.3 + 33.9 * frequency_mhz.log10() - 13.82 * tx_height_m.log10() - a_hm
            + (44.9 - 6.55 * tx_height_m.log10()) * distance_km.log10() + cm
    }

    /// 3GPP Urban Macro (UMa) path loss model
    fn three_gpp_uma_path_loss(distance_km: f64, frequency_mhz: f64, tx_height_m: f64, rx_height_m: f64) -> f64 {
        let distance_m = distance_km * 1000.0;
        let frequency_ghz = frequency_mhz / 1000.0;
        
        // Line of sight probability
        let d_bp = 4.0 * tx_height_m * rx_height_m * frequency_ghz / 0.3; // Breakpoint distance
        
        if distance_m <= d_bp {
            // Before breakpoint
            28.0 + 22.0 * distance_m.log10() + 20.0 * frequency_ghz.log10()
        } else {
            // After breakpoint
            28.0 + 40.0 * distance_m.log10() + 20.0 * frequency_ghz.log10()
                - 9.0 * d_bp.log10()
        }
    }

    /// 3GPP Urban Micro (UMi) path loss model
    fn three_gpp_umi_path_loss(distance_km: f64, frequency_mhz: f64, _tx_height_m: f64, _rx_height_m: f64) -> f64 {
        let distance_m = distance_km * 1000.0;
        let frequency_ghz = frequency_mhz / 1000.0;
        
        32.4 + 21.0 * distance_m.log10() + 20.0 * frequency_ghz.log10()
    }

    /// 3GPP Rural Macro (RMa) path loss model
    fn three_gpp_rma_path_loss(distance_km: f64, frequency_mhz: f64, tx_height_m: f64, rx_height_m: f64) -> f64 {
        let distance_m = distance_km * 1000.0;
        let frequency_ghz = frequency_mhz / 1000.0;
        
        let d_bp = 2.0 * PI * tx_height_m * rx_height_m * frequency_ghz / 0.3;
        
        if distance_m <= d_bp {
            20.0 * distance_m.log10() + 20.0 * frequency_ghz.log10() + 32.4
        } else {
            20.0 * distance_m.log10() + 20.0 * frequency_ghz.log10() + 32.4
                + 25.0 * d_bp.log10()
        }
    }

    /// Two-ray ground reflection path loss model
    fn two_ray_ground_path_loss(distance_km: f64, _frequency_mhz: f64, tx_height_m: f64, rx_height_m: f64) -> f64 {
        let distance_m = distance_km * 1000.0;
        40.0 * distance_m.log10() - 20.0 * tx_height_m.log10() - 20.0 * rx_height_m.log10()
    }

    /// Calculate received signal strength (RSS) in dBm
    pub fn calculate_rss(
        tx_power_dbm: f64,
        path_loss_db: f64,
        tx_gain_dbi: f64,
        rx_gain_dbi: f64,
        additional_loss_db: f64,
    ) -> f64 {
        tx_power_dbm + tx_gain_dbi + rx_gain_dbi - path_loss_db - additional_loss_db
    }

    /// Calculate Signal-to-Interference-plus-Noise Ratio (SINR) in dB
    pub fn calculate_sinr(signal_power_dbm: f64, interference_power_dbm: f64, noise_power_dbm: f64) -> f64 {
        let signal_linear = Self::dbm_to_linear(signal_power_dbm);
        let interference_linear = Self::dbm_to_linear(interference_power_dbm);
        let noise_linear = Self::dbm_to_linear(noise_power_dbm);
        
        let sinr_linear = signal_linear / (interference_linear + noise_linear);
        Self::linear_to_db(sinr_linear)
    }

    /// Convert dBm to linear power (mW)
    pub fn dbm_to_linear(dbm: f64) -> f64 {
        10f64.powf(dbm / 10.0)
    }

    /// Convert linear power (mW) to dBm
    pub fn linear_to_dbm(linear_mw: f64) -> f64 {
        10.0 * linear_mw.log10()
    }

    /// Convert linear ratio to dB
    pub fn linear_to_db(linear: f64) -> f64 {
        10.0 * linear.log10()
    }

    /// Convert dB to linear ratio
    pub fn db_to_linear(db: f64) -> f64 {
        10f64.powf(db / 10.0)
    }

    /// Calculate thermal noise power in dBm
    pub fn thermal_noise_power(temperature_k: f64, bandwidth_hz: f64) -> f64 {
        const BOLTZMANN_CONSTANT: f64 = 1.38064852e-23; // J/K
        let noise_power_watts = BOLTZMANN_CONSTANT * temperature_k * bandwidth_hz;
        let noise_power_mw = noise_power_watts * 1000.0;
        Self::linear_to_dbm(noise_power_mw)
    }
}

/// Time series analysis utilities
pub struct TimeSeriesUtils;

impl TimeSeriesUtils {
    /// Calculate moving average of a time series
    pub fn moving_average<T>(timeseries: &TimeSeries<T>, window_size: usize) -> TimeSeries<f64>
    where
        T: Into<f64> + Clone,
    {
        let mut result = TimeSeries::new(format!("{}_ma_{}", timeseries.id, window_size));
        
        if timeseries.points.len() < window_size {
            return result;
        }

        for i in window_size..=timeseries.points.len() {
            let window_start = i - window_size;
            let window_values: Vec<f64> = timeseries.points[window_start..i]
                .iter()
                .map(|p| p.value.clone().into())
                .collect();
            
            let average = window_values.iter().sum::<f64>() / window_size as f64;
            let timestamp = timeseries.points[i - 1].timestamp;
            
            result.add_point(TimeSeriesPoint::new(timestamp, average));
        }

        result
    }

    /// Calculate exponential weighted moving average (EWMA)
    pub fn ewma<T>(timeseries: &TimeSeries<T>, alpha: f64) -> TimeSeries<f64>
    where
        T: Into<f64> + Clone,
    {
        let mut result = TimeSeries::new(format!("{}_ewma", timeseries.id));
        
        if timeseries.points.is_empty() {
            return result;
        }

        let mut ewma_value = timeseries.points[0].value.clone().into();
        result.add_point(TimeSeriesPoint::new(timeseries.points[0].timestamp, ewma_value));

        for point in timeseries.points.iter().skip(1) {
            let current_value = point.value.clone().into();
            ewma_value = alpha * current_value + (1.0 - alpha) * ewma_value;
            result.add_point(TimeSeriesPoint::new(point.timestamp, ewma_value));
        }

        result
    }

    /// Detect anomalies using statistical methods
    pub fn detect_anomalies<T>(
        timeseries: &TimeSeries<T>,
        threshold_std_dev: f64,
    ) -> Vec<TimeSeriesPoint<bool>>
    where
        T: Into<f64> + Clone,
    {
        let values: Vec<f64> = timeseries.points.iter()
            .map(|p| p.value.clone().into())
            .collect();

        if values.len() < 3 {
            return Vec::new();
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        let std_dev = variance.sqrt();

        let threshold = threshold_std_dev * std_dev;

        timeseries.points.iter()
            .map(|point| {
                let value = point.value.clone().into();
                let is_anomaly = (value - mean).abs() > threshold;
                TimeSeriesPoint::new(point.timestamp, is_anomaly)
            })
            .collect()
    }

    /// Calculate time series correlation coefficient
    pub fn correlation<T, U>(series1: &TimeSeries<T>, series2: &TimeSeries<U>) -> RanResult<f64>
    where
        T: Into<f64> + Clone,
        U: Into<f64> + Clone,
    {
        if series1.points.len() != series2.points.len() {
            return Err(RanError::validation(
                "series_length",
                "Time series must have the same length for correlation",
            ));
        }

        let values1: Vec<f64> = series1.points.iter().map(|p| p.value.clone().into()).collect();
        let values2: Vec<f64> = series2.points.iter().map(|p| p.value.clone().into()).collect();

        let n = values1.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean1 = values1.iter().sum::<f64>() / n;
        let mean2 = values2.iter().sum::<f64>() / n;

        let numerator: f64 = values1.iter()
            .zip(values2.iter())
            .map(|(v1, v2)| (v1 - mean1) * (v2 - mean2))
            .sum();

        let sum_sq1: f64 = values1.iter().map(|v| (v - mean1).powi(2)).sum();
        let sum_sq2: f64 = values2.iter().map(|v| (v - mean2).powi(2)).sum();

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Resample time series to a regular interval
    pub fn resample<T>(
        timeseries: &TimeSeries<T>,
        interval_seconds: i64,
        aggregation: AggregationMethod,
    ) -> TimeSeries<f64>
    where
        T: Into<f64> + Clone,
    {
        let mut result = TimeSeries::new(format!("{}_resampled_{}", timeseries.id, interval_seconds));
        
        if timeseries.points.is_empty() {
            return result;
        }

        let start_time = timeseries.points[0].timestamp;
        let end_time = timeseries.points.last().unwrap().timestamp;
        let interval_duration = Duration::seconds(interval_seconds);

        let mut current_time = start_time;
        
        while current_time <= end_time {
            let window_start = current_time;
            let window_end = current_time + interval_duration;
            
            let window_points: Vec<f64> = timeseries.points.iter()
                .filter(|p| p.timestamp >= window_start && p.timestamp < window_end)
                .map(|p| p.value.clone().into())
                .collect();

            if !window_points.is_empty() {
                let aggregated_value = match aggregation {
                    AggregationMethod::Mean => window_points.iter().sum::<f64>() / window_points.len() as f64,
                    AggregationMethod::Sum => window_points.iter().sum(),
                    AggregationMethod::Min => window_points.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    AggregationMethod::Max => window_points.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    AggregationMethod::Count => window_points.len() as f64,
                };
                
                result.add_point(TimeSeriesPoint::new(current_time, aggregated_value));
            }
            
            current_time = current_time + interval_duration;
        }

        result
    }

    /// Calculate time series trend using linear regression
    pub fn calculate_trend<T>(timeseries: &TimeSeries<T>) -> RanResult<(f64, f64)>
    where
        T: Into<f64> + Clone,
    {
        if timeseries.points.len() < 2 {
            return Err(RanError::validation(
                "timeseries_length",
                "Time series must have at least 2 points for trend calculation",
            ));
        }

        let start_time = timeseries.points[0].timestamp.timestamp() as f64;
        let x_values: Vec<f64> = timeseries.points.iter()
            .map(|p| (p.timestamp.timestamp() as f64 - start_time) / 3600.0) // Hours from start
            .collect();
        let y_values: Vec<f64> = timeseries.points.iter()
            .map(|p| p.value.clone().into())
            .collect();

        let n = x_values.len() as f64;
        let sum_x = x_values.iter().sum::<f64>();
        let sum_y = y_values.iter().sum::<f64>();
        let sum_xy = x_values.iter().zip(y_values.iter()).map(|(x, y)| x * y).sum::<f64>();
        let sum_x2 = x_values.iter().map(|x| x.powi(2)).sum::<f64>();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
        let intercept = (sum_y - slope * sum_x) / n;

        Ok((slope, intercept))
    }
}

/// Aggregation methods for resampling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationMethod {
    /// Calculate mean/average
    Mean,
    /// Calculate sum
    Sum,
    /// Calculate minimum
    Min,
    /// Calculate maximum
    Max,
    /// Count of values
    Count,
}

/// Geographic utilities
pub struct GeoUtils;

impl GeoUtils {
    /// Calculate coverage area for a cell
    pub fn calculate_coverage_area(
        cell_location: &GeoCoordinate,
        tx_power_dbm: f64,
        frequency_mhz: f64,
        antenna_height_m: f64,
        sensitivity_dbm: f64,
        environment: EnvironmentType,
    ) -> f64 {
        // Binary search for maximum distance where signal is above sensitivity
        let mut min_distance = 0.001; // 1 meter
        let mut max_distance = 50.0; // 50 km
        let tolerance = 0.01; // 10 meter tolerance

        while (max_distance - min_distance) > tolerance {
            let test_distance = (min_distance + max_distance) / 2.0;
            
            let path_loss = RfUtils::calculate_path_loss(
                PropagationModel::OkumuraHata,
                test_distance,
                frequency_mhz,
                antenna_height_m,
                1.5, // Typical mobile height
                environment,
            ).unwrap_or(150.0);

            let received_power = tx_power_dbm - path_loss;
            
            if received_power >= sensitivity_dbm {
                min_distance = test_distance;
            } else {
                max_distance = test_distance;
            }
        }

        let coverage_radius_km = min_distance;
        PI * coverage_radius_km.powi(2) // Area in km²
    }

    /// Find optimal cell locations using grid search
    pub fn find_optimal_cell_locations(
        service_area: &[(GeoCoordinate, f64)], // (location, demand)
        num_cells: usize,
        max_cell_range_km: f64,
    ) -> Vec<GeoCoordinate> {
        if service_area.is_empty() || num_cells == 0 {
            return Vec::new();
        }

        // Simple greedy algorithm for demonstration
        // In practice, would use more sophisticated optimization
        let mut cell_locations = Vec::new();
        let mut remaining_demand = service_area.to_vec();

        for _ in 0..num_cells {
            let mut best_location = None;
            let mut best_coverage = 0.0;

            // Try different candidate locations
            for candidate in service_area {
                let coverage = remaining_demand.iter()
                    .filter(|(loc, _)| candidate.0.distance_to(loc) / 1000.0 <= max_cell_range_km)
                    .map(|(_, demand)| demand)
                    .sum::<f64>();

                if coverage > best_coverage {
                    best_coverage = coverage;
                    best_location = Some(candidate.0);
                }
            }

            if let Some(location) = best_location {
                cell_locations.push(location);
                
                // Remove covered demand
                remaining_demand.retain(|(loc, _)| {
                    location.distance_to(loc) / 1000.0 > max_cell_range_km
                });
            }
        }

        cell_locations
    }

    /// Calculate Voronoi tessellation for cell coverage areas
    pub fn calculate_voronoi_cells(cell_locations: &[GeoCoordinate]) -> HashMap<usize, Vec<GeoCoordinate>> {
        // Simplified Voronoi calculation
        // In practice, would use proper computational geometry algorithms
        let mut voronoi_cells = HashMap::new();
        
        // For demonstration, return empty cells
        for (i, _) in cell_locations.iter().enumerate() {
            voronoi_cells.insert(i, Vec::new());
        }
        
        voronoi_cells
    }
}

/// Mathematical utilities
pub struct MathUtils;

impl MathUtils {
    /// Calculate weighted average
    pub fn weighted_average(values: &[f64], weights: &[f64]) -> RanResult<f64> {
        if values.len() != weights.len() {
            return Err(RanError::validation(
                "array_length",
                "Values and weights must have the same length",
            ));
        }

        if values.is_empty() {
            return Ok(0.0);
        }

        let weighted_sum: f64 = values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();
        let weight_sum: f64 = weights.iter().sum();

        if weight_sum == 0.0 {
            Ok(0.0)
        } else {
            Ok(weighted_sum / weight_sum)
        }
    }

    /// Calculate percentile
    pub fn percentile(values: &[f64], percentile: f64) -> RanResult<f64> {
        if values.is_empty() {
            return Err(RanError::validation("values", "Values array cannot be empty"));
        }

        if !(0.0..=100.0).contains(&percentile) {
            return Err(RanError::validation(
                "percentile",
                "Percentile must be between 0 and 100",
            ));
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile / 100.0) * (sorted_values.len() - 1) as f64;
        let lower_index = index.floor() as usize;
        let upper_index = index.ceil() as usize;

        if lower_index == upper_index {
            Ok(sorted_values[lower_index])
        } else {
            let weight = index - lower_index as f64;
            Ok(sorted_values[lower_index] * (1.0 - weight) + sorted_values[upper_index] * weight)
        }
    }

    /// Calculate coefficient of variation
    pub fn coefficient_of_variation(values: &[f64]) -> RanResult<f64> {
        if values.is_empty() {
            return Err(RanError::validation("values", "Values array cannot be empty"));
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        if mean == 0.0 {
            return Ok(0.0);
        }

        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        
        let std_dev = variance.sqrt();
        Ok(std_dev / mean.abs())
    }

    /// Normalize values to [0, 1] range
    pub fn normalize_min_max(values: &[f64]) -> RanResult<Vec<f64>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        if min_val == max_val {
            return Ok(vec![0.5; values.len()]); // All values are the same
        }

        let range = max_val - min_val;
        let normalized = values.iter()
            .map(|v| (v - min_val) / range)
            .collect();

        Ok(normalized)
    }

    /// Calculate z-score normalization
    pub fn normalize_z_score(values: &[f64]) -> RanResult<Vec<f64>> {
        if values.is_empty() {
            return Ok(Vec::new());
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(vec![0.0; values.len()]); // All values are the same
        }

        let normalized = values.iter()
            .map(|v| (v - mean) / std_dev)
            .collect();

        Ok(normalized)
    }
}

/// Configuration validation utilities
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate frequency band
    pub fn validate_frequency_band(frequency_mhz: f64, band: &str) -> RanResult<()> {
        let valid_ranges = match band {
            "n1" => (1920.0, 1980.0), // 2100 MHz band
            "n3" => (1710.0, 1785.0), // 1800 MHz band
            "n7" => (2500.0, 2570.0), // 2600 MHz band
            "n28" => (703.0, 748.0),  // 700 MHz band
            "n78" => (3300.0, 3800.0), // 3.5 GHz band
            _ => return Err(RanError::validation("band", &format!("Unknown band: {}", band))),
        };

        if frequency_mhz < valid_ranges.0 || frequency_mhz > valid_ranges.1 {
            return Err(RanError::validation(
                "frequency",
                &format!("Frequency {} MHz is not valid for band {}", frequency_mhz, band),
            ));
        }

        Ok(())
    }

    /// Validate power levels
    pub fn validate_power_levels(tx_power_dbm: f64, max_power_dbm: f64) -> RanResult<()> {
        if tx_power_dbm < 0.0 {
            return Err(RanError::validation(
                "tx_power",
                "Transmission power cannot be negative",
            ));
        }

        if tx_power_dbm > max_power_dbm {
            return Err(RanError::validation(
                "tx_power",
                &format!("Transmission power {} dBm exceeds maximum {} dBm", tx_power_dbm, max_power_dbm),
            ));
        }

        Ok(())
    }

    /// Validate antenna parameters
    pub fn validate_antenna_parameters(tilt_degrees: f64, azimuth_degrees: f64) -> RanResult<()> {
        if !(-90.0..=90.0).contains(&tilt_degrees) {
            return Err(RanError::validation(
                "antenna_tilt",
                "Antenna tilt must be between -90 and +90 degrees",
            ));
        }

        if !(0.0..360.0).contains(&azimuth_degrees) {
            return Err(RanError::validation(
                "antenna_azimuth",
                "Antenna azimuth must be between 0 and 360 degrees",
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_free_space_path_loss() {
        let path_loss = RfUtils::calculate_path_loss(
            PropagationModel::FreeSpace,
            1.0, // 1 km
            2400.0, // 2.4 GHz
            30.0, // 30m height
            1.5, // 1.5m height
            EnvironmentType::Urban,
        ).unwrap();

        // Free space path loss at 1 km and 2.4 GHz should be around 100 dB
        assert!(path_loss > 90.0 && path_loss < 110.0);
    }

    #[test]
    fn test_rf_calculations() {
        let signal_power = -70.0;
        let interference_power = -90.0;
        let noise_power = -110.0;

        let sinr = RfUtils::calculate_sinr(signal_power, interference_power, noise_power);
        assert!(sinr > 15.0); // Should be good SINR

        let linear_power = RfUtils::dbm_to_linear(30.0); // 30 dBm = 1000 mW
        assert_abs_diff_eq!(linear_power, 1000.0, epsilon = 0.1);

        let dbm_power = RfUtils::linear_to_dbm(1000.0);
        assert_abs_diff_eq!(dbm_power, 30.0, epsilon = 0.1);
    }

    #[test]
    fn test_moving_average() {
        let mut timeseries = TimeSeries::new("test".to_string());
        let now = Utc::now();

        for i in 0..10 {
            timeseries.add_point(TimeSeriesPoint::new(
                now + Duration::seconds(i * 10),
                (i + 1) as f64,
            ));
        }

        let ma = TimeSeriesUtils::moving_average(&timeseries, 3);
        assert_eq!(ma.points.len(), 8); // 10 - 3 + 1
        assert_abs_diff_eq!(ma.points[0].value, 2.0, epsilon = 0.1); // Average of 1, 2, 3
    }

    #[test]
    fn test_ewma() {
        let mut timeseries = TimeSeries::new("test".to_string());
        let now = Utc::now();

        for i in 0..5 {
            timeseries.add_point(TimeSeriesPoint::new(
                now + Duration::seconds(i * 10),
                10.0,
            ));
        }

        let ewma = TimeSeriesUtils::ewma(&timeseries, 0.3);
        assert_eq!(ewma.points.len(), 5);
        // With constant values, EWMA should converge to the value
        assert_abs_diff_eq!(ewma.points.last().unwrap().value, 10.0, epsilon = 0.1);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut timeseries = TimeSeries::new("test".to_string());
        let now = Utc::now();

        // Normal values around 10
        for i in 0..10 {
            let value = 10.0 + (i as f64 - 5.0) * 0.1; // Small variation
            timeseries.add_point(TimeSeriesPoint::new(
                now + Duration::seconds(i * 10),
                value,
            ));
        }

        // Add an outlier
        timeseries.add_point(TimeSeriesPoint::new(
            now + Duration::seconds(100),
            50.0, // Clear outlier
        ));

        let anomalies = TimeSeriesUtils::detect_anomalies(&timeseries, 2.0);
        let anomaly_count = anomalies.iter().filter(|p| p.value).count();
        assert_eq!(anomaly_count, 1); // Should detect the outlier
    }

    #[test]
    fn test_correlation() {
        let mut series1 = TimeSeries::new("series1".to_string());
        let mut series2 = TimeSeries::new("series2".to_string());
        let now = Utc::now();

        // Perfectly correlated series
        for i in 0..10 {
            let value = (i + 1) as f64;
            series1.add_point(TimeSeriesPoint::new(
                now + Duration::seconds(i * 10),
                value,
            ));
            series2.add_point(TimeSeriesPoint::new(
                now + Duration::seconds(i * 10),
                value * 2.0, // Perfectly correlated
            ));
        }

        let correlation = TimeSeriesUtils::correlation(&series1, &series2).unwrap();
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_math_utils() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let weighted_avg = MathUtils::weighted_average(&values, &weights).unwrap();
        assert_abs_diff_eq!(weighted_avg, 3.0, epsilon = 0.01);

        let p50 = MathUtils::percentile(&values, 50.0).unwrap();
        assert_abs_diff_eq!(p50, 3.0, epsilon = 0.01);

        let cv = MathUtils::coefficient_of_variation(&values).unwrap();
        assert!(cv > 0.0 && cv < 1.0);

        let normalized = MathUtils::normalize_min_max(&values).unwrap();
        assert_abs_diff_eq!(normalized[0], 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(normalized[4], 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_config_validation() {
        // Valid frequency for n78 band
        assert!(ConfigValidator::validate_frequency_band(3500.0, "n78").is_ok());
        
        // Invalid frequency for n78 band
        assert!(ConfigValidator::validate_frequency_band(2400.0, "n78").is_err());

        // Valid power levels
        assert!(ConfigValidator::validate_power_levels(30.0, 46.0).is_ok());
        
        // Invalid power levels
        assert!(ConfigValidator::validate_power_levels(50.0, 46.0).is_err());

        // Valid antenna parameters
        assert!(ConfigValidator::validate_antenna_parameters(5.0, 120.0).is_ok());
        
        // Invalid antenna tilt
        assert!(ConfigValidator::validate_antenna_parameters(100.0, 120.0).is_err());
    }

    #[test]
    fn test_coverage_area_calculation() {
        let cell_location = GeoCoordinate::new(52.520008, 13.404954, None);
        let area = GeoUtils::calculate_coverage_area(
            &cell_location,
            46.0, // 46 dBm
            2600.0, // 2.6 GHz
            30.0, // 30m height
            -110.0, // -110 dBm sensitivity
            EnvironmentType::Urban,
        );

        assert!(area > 0.0);
        assert!(area < 1000.0); // Should be reasonable for urban environment
    }
}