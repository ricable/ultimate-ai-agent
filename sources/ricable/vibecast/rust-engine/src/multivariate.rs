// Multivariate analysis for correlated RAN metrics
// Neurodivergent approach: understanding relationships through multiple correlation methods

use crate::MultiVariatePoint;
use anyhow::Result;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

pub struct MultivariateAnalyzer {
    feature_names: Vec<String>,
    correlation_matrix: Option<Array2<f64>>,
}

impl MultivariateAnalyzer {
    pub fn new() -> Self {
        Self {
            feature_names: Vec::new(),
            correlation_matrix: None,
        }
    }

    pub fn fit(&mut self, data: &[MultiVariatePoint]) -> Result<()> {
        if data.is_empty() {
            anyhow::bail!("No data provided");
        }

        // Extract feature names
        self.feature_names = data[0].features.keys().cloned().collect();
        let n_features = self.feature_names.len();

        // Build feature matrix
        let mut feature_matrix = Array2::<f64>::zeros((data.len(), n_features));

        for (i, point) in data.iter().enumerate() {
            for (j, feature_name) in self.feature_names.iter().enumerate() {
                if let Some(&value) = point.features.get(feature_name) {
                    feature_matrix[[i, j]] = value;
                }
            }
        }

        // Compute correlation matrix
        self.correlation_matrix = Some(self.compute_correlation(&feature_matrix)?);

        Ok(())
    }

    fn compute_correlation(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = data.dim();
        let mut corr = Array2::<f64>::zeros((n_features, n_features));

        // Compute means
        let means: Vec<f64> = (0..n_features)
            .map(|j| {
                let col = data.column(j);
                col.sum() / n_samples as f64
            })
            .collect();

        // Compute standard deviations
        let std_devs: Vec<f64> = (0..n_features)
            .map(|j| {
                let col = data.column(j);
                let mean = means[j];
                let variance = col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
                variance.sqrt()
            })
            .collect();

        // Compute correlation coefficients
        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    corr[[i, j]] = 1.0;
                } else {
                    let col_i = data.column(i);
                    let col_j = data.column(j);

                    let covariance: f64 = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&xi, &xj)| (xi - means[i]) * (xj - means[j]))
                        .sum::<f64>() / n_samples as f64;

                    let correlation = if std_devs[i] > 0.0 && std_devs[j] > 0.0 {
                        covariance / (std_devs[i] * std_devs[j])
                    } else {
                        0.0
                    };

                    corr[[i, j]] = correlation;
                }
            }
        }

        Ok(corr)
    }

    pub fn get_correlation_matrix(&self) -> Option<&Array2<f64>> {
        self.correlation_matrix.as_ref()
    }

    pub fn find_correlated_features(&self, threshold: f64) -> Vec<(String, String, f64)> {
        let mut correlations = Vec::new();

        if let Some(corr_matrix) = &self.correlation_matrix {
            let n_features = self.feature_names.len();

            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    let corr_value = corr_matrix[[i, j]];
                    if corr_value.abs() >= threshold {
                        correlations.push((
                            self.feature_names[i].clone(),
                            self.feature_names[j].clone(),
                            corr_value,
                        ));
                    }
                }
            }
        }

        // Sort by absolute correlation value
        correlations.sort_by(|a, b| b.2.abs().partial_cmp(&a.2.abs()).unwrap());
        correlations
    }

    /// Identify principal components for dimensionality reduction
    pub fn find_principal_features(&self, variance_threshold: f64) -> Result<Vec<String>> {
        // Simplified PCA-like approach
        // In production, use proper PCA implementation

        if let Some(corr_matrix) = &self.correlation_matrix {
            let n_features = self.feature_names.len();

            // Calculate sum of absolute correlations for each feature
            let mut feature_importance: Vec<(String, f64)> = (0..n_features)
                .map(|i| {
                    let importance: f64 = (0..n_features)
                        .map(|j| if i != j { corr_matrix[[i, j]].abs() } else { 0.0 })
                        .sum();
                    (self.feature_names[i].clone(), importance)
                })
                .collect();

            // Sort by importance
            feature_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Select top features
            let total_importance: f64 = feature_importance.iter().map(|(_, imp)| imp).sum();
            let mut cumulative = 0.0;
            let mut selected = Vec::new();

            for (feature, importance) in feature_importance {
                cumulative += importance / total_importance;
                selected.push(feature);

                if cumulative >= variance_threshold {
                    break;
                }
            }

            Ok(selected)
        } else {
            anyhow::bail!("Correlation matrix not computed. Call fit() first.")
        }
    }

    /// Detect multivariate anomalies using Mahalanobis distance
    pub fn detect_multivariate_anomalies(
        &self,
        data: &[MultiVariatePoint],
        threshold: f64,
    ) -> Result<Vec<(usize, f64)>> {
        // Simplified multivariate anomaly detection
        // In production, use proper Mahalanobis distance

        let mut anomalies = Vec::new();

        for (idx, point) in data.iter().enumerate() {
            let mut deviation = 0.0;
            let mut count = 0;

            for feature_name in &self.feature_names {
                if let Some(&value) = point.features.get(feature_name) {
                    // Simplified: use absolute value as proxy for distance
                    deviation += value.abs();
                    count += 1;
                }
            }

            if count > 0 {
                let avg_deviation = deviation / count as f64;
                if avg_deviation > threshold {
                    anomalies.push((idx, avg_deviation));
                }
            }
        }

        Ok(anomalies)
    }

    /// Analyze feature interactions for RAN optimization
    pub fn analyze_ran_interactions(&self) -> HashMap<String, Vec<String>> {
        let mut interactions = HashMap::new();

        // Find strong correlations that indicate RAN interactions
        let correlated = self.find_correlated_features(0.7);

        for (feat1, feat2, _corr) in correlated {
            interactions
                .entry(feat1.clone())
                .or_insert_with(Vec::new)
                .push(feat2.clone());

            interactions
                .entry(feat2)
                .or_insert_with(Vec::new)
                .push(feat1);
        }

        interactions
    }
}

impl Default for MultivariateAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
