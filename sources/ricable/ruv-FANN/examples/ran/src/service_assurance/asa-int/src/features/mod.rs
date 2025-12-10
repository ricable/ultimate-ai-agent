use crate::{NoiseFloorMeasurement, CellParameters, Result, Error};
use ndarray::{Array1, Array2, Axis};
use rustfft::{FftPlanner, num_complex::Complex};
use statrs::statistics::Statistics;
use std::collections::HashMap;

pub struct InterferenceFeatureExtractor {
    fft_planner: FftPlanner<f64>,
    window_size: usize,
    frequency_bins: usize,
    fft_size: usize,
    overlap_factor: f64,
}

impl InterferenceFeatureExtractor {
    pub fn new(window_size: usize, frequency_bins: usize, fft_size: usize, overlap_factor: f64) -> Self {
        Self {
            fft_planner: FftPlanner::new(),
            window_size,
            frequency_bins,
            fft_size,
            overlap_factor,
        }
    }
    
    pub fn extract_features(
        &mut self,
        measurements: &[NoiseFloorMeasurement],
        cell_params: &CellParameters,
    ) -> Result<Array1<f64>> {
        if measurements.is_empty() {
            return Err(Error::InsufficientData("No measurements provided".to_string()));
        }
        
        let mut features = Vec::new();
        
        // Extract statistical features
        features.extend(self.extract_statistical_features(measurements)?);
        
        // Extract spectral features
        features.extend(self.extract_spectral_features(measurements)?);
        
        // Extract temporal features
        features.extend(self.extract_temporal_features(measurements)?);
        
        // Extract cell-specific features
        features.extend(self.extract_cell_features(measurements, cell_params)?);
        
        // Extract noise floor pattern features
        features.extend(self.extract_noise_floor_patterns(measurements)?);
        
        Ok(Array1::from_vec(features))
    }
    
    fn extract_statistical_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Extract noise floor statistics
        let pusch_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pusch)
            .collect();
        let pucch_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pucch)
            .collect();
        let ret_values: Vec<f64> = measurements.iter()
            .map(|m| m.cell_ret)
            .collect();
        let sinr_values: Vec<f64> = measurements.iter()
            .map(|m| m.sinr)
            .collect();
        
        // Statistical moments for each metric
        for values in [&pusch_values, &pucch_values, &ret_values, &sinr_values] {
            if values.is_empty() {
                continue;
            }
            
            features.push(values.mean());
            features.push(values.std_dev());
            features.push(values.min());
            features.push(values.max());
            features.push(values.variance());
            
            // Skewness and kurtosis approximations
            let mean = values.mean();
            let std_dev = values.std_dev();
            if std_dev > 0.0 {
                let skewness = values.iter()
                    .map(|&x| ((x - mean) / std_dev).powi(3))
                    .sum::<f64>() / values.len() as f64;
                let kurtosis = values.iter()
                    .map(|&x| ((x - mean) / std_dev).powi(4))
                    .sum::<f64>() / values.len() as f64;
                features.push(skewness);
                features.push(kurtosis);
            } else {
                features.push(0.0);
                features.push(0.0);
            }
        }
        
        // Cross-correlation features
        features.push(self.pearson_correlation(&pusch_values, &pucch_values));
        features.push(self.pearson_correlation(&pusch_values, &ret_values));
        features.push(self.pearson_correlation(&sinr_values, &ret_values));
        
        Ok(features)
    }
    
    fn extract_spectral_features(&mut self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Extract PUSCH noise floor spectral features
        let pusch_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pusch)
            .collect();
        
        let spectrum = self.compute_power_spectrum(&pusch_values)?;
        
        // Spectral centroid
        let centroid = self.compute_spectral_centroid(&spectrum);
        features.push(centroid);
        
        // Spectral rolloff
        let rolloff = self.compute_spectral_rolloff(&spectrum, 0.85);
        features.push(rolloff);
        
        // Spectral flatness
        let flatness = self.compute_spectral_flatness(&spectrum);
        features.push(flatness);
        
        // Spectral entropy
        let entropy = self.compute_spectral_entropy(&spectrum);
        features.push(entropy);
        
        // Dominant frequency components
        let dominant_freqs = self.find_dominant_frequencies(&spectrum, 5);
        features.extend(dominant_freqs);
        
        // Spectral bands energy
        let band_energies = self.compute_band_energies(&spectrum, 8);
        features.extend(band_energies);
        
        Ok(features)
    }
    
    fn extract_temporal_features(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Extract temporal patterns
        let pusch_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pusch)
            .collect();
        
        // Trend analysis
        let trend = self.compute_linear_trend(&pusch_values);
        features.push(trend);
        
        // Volatility
        let volatility = self.compute_volatility(&pusch_values);
        features.push(volatility);
        
        // Autocorrelation at different lags
        for lag in [1, 2, 5, 10] {
            let autocorr = self.compute_autocorrelation(&pusch_values, lag);
            features.push(autocorr);
        }
        
        // Zero crossing rate
        let zcr = self.compute_zero_crossing_rate(&pusch_values);
        features.push(zcr);
        
        // Rate of change
        let roc = self.compute_rate_of_change(&pusch_values);
        features.push(roc);
        
        Ok(features)
    }
    
    fn extract_cell_features(&self, measurements: &[NoiseFloorMeasurement], cell_params: &CellParameters) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Cell-specific features
        features.push(cell_params.tx_power);
        features.push(cell_params.antenna_count as f64);
        features.push(cell_params.bandwidth_mhz);
        
        // Technology encoding (one-hot)
        let tech_encoding = match cell_params.technology.as_str() {
            "LTE" => [1.0, 0.0, 0.0],
            "NR" => [0.0, 1.0, 0.0],
            _ => [0.0, 0.0, 1.0],
        };
        features.extend(tech_encoding);
        
        // Frequency band impact
        let band_impact = self.compute_frequency_band_impact(&cell_params.frequency_band);
        features.push(band_impact);
        
        // Load-related features
        let avg_active_users = measurements.iter()
            .map(|m| m.active_users as f64)
            .sum::<f64>() / measurements.len() as f64;
        let avg_prb_util = measurements.iter()
            .map(|m| m.prb_utilization)
            .sum::<f64>() / measurements.len() as f64;
        
        features.push(avg_active_users);
        features.push(avg_prb_util);
        
        // Load vs noise floor correlation
        let users: Vec<f64> = measurements.iter().map(|m| m.active_users as f64).collect();
        let noise: Vec<f64> = measurements.iter().map(|m| m.noise_floor_pusch).collect();
        let correlation = self.pearson_correlation(&users, &noise);
        features.push(correlation);
        
        Ok(features)
    }
    
    fn extract_noise_floor_patterns(&self, measurements: &[NoiseFloorMeasurement]) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // PUSCH vs PUCCH noise floor differences
        let pusch_pucch_diff: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pusch - m.noise_floor_pucch)
            .collect();
        
        features.push(pusch_pucch_diff.mean());
        features.push(pusch_pucch_diff.std_dev());
        
        // Noise floor stability
        let pusch_values: Vec<f64> = measurements.iter()
            .map(|m| m.noise_floor_pusch)
            .collect();
        let stability = self.compute_stability_index(&pusch_values);
        features.push(stability);
        
        // Anomaly detection features
        let anomaly_score = self.compute_anomaly_score(&pusch_values);
        features.push(anomaly_score);
        
        // Pattern regularity
        let regularity = self.compute_pattern_regularity(&pusch_values);
        features.push(regularity);
        
        Ok(features)
    }
    
    // Helper methods
    fn compute_power_spectrum(&mut self, signal: &[f64]) -> Result<Vec<f64>> {
        let mut input: Vec<Complex<f64>> = signal.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Pad to FFT size
        input.resize(self.fft_size, Complex::new(0.0, 0.0));
        
        let fft = self.fft_planner.plan_fft_forward(input.len());
        fft.process(&mut input);
        
        let spectrum: Vec<f64> = input.iter()
            .map(|c| c.norm_sqr())
            .collect();
        
        Ok(spectrum)
    }
    
    fn compute_spectral_centroid(&self, spectrum: &[f64]) -> f64 {
        let sum_weighted: f64 = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| i as f64 * mag)
            .sum();
        let sum_magnitude: f64 = spectrum.iter().sum();
        
        if sum_magnitude > 0.0 {
            sum_weighted / sum_magnitude
        } else {
            0.0
        }
    }
    
    fn compute_spectral_rolloff(&self, spectrum: &[f64], threshold: f64) -> f64 {
        let total_energy: f64 = spectrum.iter().sum();
        let target_energy = total_energy * threshold;
        
        let mut cumulative_energy = 0.0;
        for (i, &energy) in spectrum.iter().enumerate() {
            cumulative_energy += energy;
            if cumulative_energy >= target_energy {
                return i as f64 / spectrum.len() as f64;
            }
        }
        
        1.0
    }
    
    fn compute_spectral_flatness(&self, spectrum: &[f64]) -> f64 {
        let geometric_mean = spectrum.iter()
            .map(|&x| if x > 0.0 { x.ln() } else { f64::NEG_INFINITY })
            .sum::<f64>() / spectrum.len() as f64;
        
        let arithmetic_mean = spectrum.iter().sum::<f64>() / spectrum.len() as f64;
        
        if arithmetic_mean > 0.0 && geometric_mean.is_finite() {
            geometric_mean.exp() / arithmetic_mean
        } else {
            0.0
        }
    }
    
    fn compute_spectral_entropy(&self, spectrum: &[f64]) -> f64 {
        let total: f64 = spectrum.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        
        let entropy = spectrum.iter()
            .map(|&x| {
                let p = x / total;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        
        entropy / (spectrum.len() as f64).ln()
    }
    
    fn find_dominant_frequencies(&self, spectrum: &[f64], count: usize) -> Vec<f64> {
        let mut indexed_spectrum: Vec<(usize, f64)> = spectrum.iter()
            .enumerate()
            .map(|(i, &mag)| (i, mag))
            .collect();
        
        indexed_spectrum.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        indexed_spectrum.into_iter()
            .take(count)
            .map(|(i, _)| i as f64 / spectrum.len() as f64)
            .collect()
    }
    
    fn compute_band_energies(&self, spectrum: &[f64], bands: usize) -> Vec<f64> {
        let band_size = spectrum.len() / bands;
        let mut energies = Vec::new();
        
        for i in 0..bands {
            let start = i * band_size;
            let end = if i == bands - 1 { spectrum.len() } else { (i + 1) * band_size };
            
            let energy: f64 = spectrum[start..end].iter().sum();
            energies.push(energy);
        }
        
        energies
    }
    
    fn compute_linear_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let n = values.len() as f64;
        let x_sum = (0..values.len()).map(|i| i as f64).sum::<f64>();
        let y_sum = values.iter().sum::<f64>();
        let xy_sum = values.iter()
            .enumerate()
            .map(|(i, &y)| i as f64 * y)
            .sum::<f64>();
        let x_sq_sum = (0..values.len()).map(|i| (i as f64).powi(2)).sum::<f64>();
        
        let slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum);
        slope
    }
    
    fn compute_volatility(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let returns: Vec<f64> = values.windows(2)
            .map(|w| (w[1] - w[0]) / w[0].abs().max(1e-10))
            .collect();
        
        returns.std_dev()
    }
    
    fn compute_autocorrelation(&self, values: &[f64], lag: usize) -> f64 {
        if values.len() <= lag {
            return 0.0;
        }
        
        let n = values.len() - lag;
        let mean = values.mean();
        
        let numerator: f64 = values.iter()
            .take(n)
            .zip(values.iter().skip(lag))
            .map(|(&x, &y)| (x - mean) * (y - mean))
            .sum();
        
        let denominator: f64 = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
    
    fn compute_zero_crossing_rate(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let crossings = values.windows(2)
            .map(|w| if w[0] * w[1] < 0.0 { 1 } else { 0 })
            .sum::<i32>();
        
        crossings as f64 / (values.len() - 1) as f64
    }
    
    fn compute_rate_of_change(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let changes: Vec<f64> = values.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        changes.mean()
    }
    
    fn compute_frequency_band_impact(&self, band: &str) -> f64 {
        // Assign impact scores based on frequency band characteristics
        match band {
            "B1" | "B3" | "B7" => 0.8,  // High interference bands
            "B20" | "B28" => 0.6,       // Medium interference bands
            "B8" | "B38" => 0.4,        // Lower interference bands
            _ => 0.5,                    // Unknown/default
        }
    }
    
    fn compute_stability_index(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 1.0;
        }
        
        let mean = values.mean();
        let std_dev = values.std_dev();
        
        if std_dev > 0.0 {
            1.0 / (1.0 + std_dev / mean.abs().max(1e-10))
        } else {
            1.0
        }
    }
    
    fn compute_anomaly_score(&self, values: &[f64]) -> f64 {
        if values.len() < 3 {
            return 0.0;
        }
        
        let mean = values.mean();
        let std_dev = values.std_dev();
        
        let anomaly_count = values.iter()
            .map(|&x| if (x - mean).abs() > 2.0 * std_dev { 1 } else { 0 })
            .sum::<i32>();
        
        anomaly_count as f64 / values.len() as f64
    }
    
    fn compute_pattern_regularity(&self, values: &[f64]) -> f64 {
        if values.len() < 4 {
            return 0.0;
        }
        
        // Compute differences between consecutive values
        let diffs: Vec<f64> = values.windows(2)
            .map(|w| w[1] - w[0])
            .collect();
        
        // Measure regularity as inverse of variance in differences
        let diff_variance = diffs.variance();
        1.0 / (1.0 + diff_variance)
    }
    
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        if x.len() != y.len() || x.len() < 2 {
            return 0.0;
        }
        
        let n = x.len() as f64;
        let sum_x = x.iter().sum::<f64>();
        let sum_y = y.iter().sum::<f64>();
        let sum_xy = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum::<f64>();
        let sum_x2 = x.iter().map(|&a| a * a).sum::<f64>();
        let sum_y2 = y.iter().map(|&b| b * b).sum::<f64>();
        
        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
        
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}