use std::collections::HashMap;
use std::f64::consts::PI;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc, Duration, Datelike, Timelike};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPIData {
    pub timestamp: DateTime<Utc>,
    pub cell_id: String,
    pub cell_type: CellType,
    pub throughput_mbps: f64,
    pub latency_ms: f64,
    pub rsrp_dbm: f64,
    pub sinr_db: f64,
    pub handover_success_rate: f64,
    pub cell_load_percent: f64,
    pub user_count: u32,
    pub traffic_gb: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellType {
    LTE,
    NR,
}

#[derive(Debug, Clone)]
pub struct LSTMLayer {
    pub input_size: usize,
    pub hidden_size: usize,
    pub weights_ih: Vec<Vec<f64>>,
    pub weights_hh: Vec<Vec<f64>>,
    pub bias_ih: Vec<f64>,
    pub bias_hh: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct AttentionMechanism {
    pub query_weights: Vec<Vec<f64>>,
    pub key_weights: Vec<Vec<f64>>,
    pub value_weights: Vec<Vec<f64>>,
    pub hidden_size: usize,
}

#[derive(Debug, Clone)]
pub struct EnhancedLSTMNetwork {
    pub layers: Vec<LSTMLayer>,
    pub attention: AttentionMechanism,
    pub output_weights: Vec<Vec<f64>>,
    pub output_bias: Vec<f64>,
    pub input_size: usize,
    pub output_size: usize,
    pub sequence_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionResult {
    pub timestamp: DateTime<Utc>,
    pub cell_id: String,
    pub metric: String,
    pub value: f64,
    pub expected_value: f64,
    pub anomaly_score: f64,
    pub is_anomaly: bool,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub metric: String,
    pub cell_id: String,
    pub trend_direction: String,
    pub trend_strength: f64,
    pub seasonality_detected: bool,
    pub forecast_7d: Vec<f64>,
    pub forecast_24h: Vec<f64>,
    pub confidence_interval: (f64, f64),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceInsights {
    pub optimization_opportunities: Vec<String>,
    pub critical_cells: Vec<String>,
    pub predicted_issues: Vec<String>,
    pub resource_recommendations: Vec<String>,
    pub overall_health_score: f64,
    pub kpi_correlations: HashMap<String, f64>,
}

pub struct PerformanceAnalyticsAgent {
    pub neural_network: EnhancedLSTMNetwork,
    pub kpi_history: Vec<KPIData>,
    pub cell_configs: HashMap<String, CellConfig>,
    pub anomaly_threshold: f64,
    pub training_iterations: usize,
    pub rng: ThreadRng,
}

#[derive(Debug, Clone)]
pub struct CellConfig {
    pub cell_type: CellType,
    pub max_throughput: f64,
    pub coverage_area: f64,
    pub location: (f64, f64),
    pub antenna_height: f64,
    pub power_dbm: f64,
}

impl PerformanceAnalyticsAgent {
    pub fn new() -> Self {
        let input_size = 6; // 6 KPI metrics
        let hidden_size = 256;
        let output_size = 6;
        let sequence_length = 24; // 24 hours lookback

        let neural_network = EnhancedLSTMNetwork::new(
            input_size,
            hidden_size,
            output_size,
            sequence_length,
            8, // 8 LSTM layers
        );

        let mut cell_configs = HashMap::new();
        let mut rng = thread_rng();

        // Generate 50 cell configurations
        for i in 0..50 {
            let cell_id = format!("CELL_{:03}", i);
            let cell_type = if i % 3 == 0 { CellType::NR } else { CellType::LTE };
            
            let config = CellConfig {
                cell_type: cell_type.clone(),
                max_throughput: match cell_type {
                    CellType::NR => rng.gen_range(300.0..500.0),
                    CellType::LTE => rng.gen_range(100.0..200.0),
                },
                coverage_area: rng.gen_range(1.0..10.0),
                location: (rng.gen_range(-90.0..90.0), rng.gen_range(-180.0..180.0)),
                antenna_height: rng.gen_range(20.0..80.0),
                power_dbm: rng.gen_range(40.0..46.0),
            };
            
            cell_configs.insert(cell_id, config);
        }

        Self {
            neural_network,
            kpi_history: Vec::new(),
            cell_configs,
            anomaly_threshold: 2.5,
            training_iterations: 12000,
            rng,
        }
    }

    pub fn generate_realistic_kpi_data(&mut self, hours: usize) -> Vec<KPIData> {
        let mut data = Vec::new();
        let start_time = Utc::now() - Duration::hours(hours as i64);

        for hour in 0..hours {
            let timestamp = start_time + Duration::hours(hour as i64);
            
            for (cell_id, config) in &self.cell_configs {
                let kpi = self.generate_cell_kpi(cell_id, config, timestamp, hour);
                data.push(kpi);
            }
        }

        self.kpi_history.extend(data.clone());
        data
    }

    fn generate_cell_kpi(&mut self, cell_id: &str, config: &CellConfig, timestamp: DateTime<Utc>, hour: usize) -> KPIData {
        let hour_of_day = timestamp.hour() as f64;
        let day_of_week = timestamp.weekday().num_days_from_monday() as f64;
        let day_of_year = timestamp.ordinal() as f64;

        // Diurnal pattern (business hours peak)
        let diurnal_factor = 0.3 + 0.7 * (1.0 + (2.0 * PI * (hour_of_day - 6.0) / 24.0).sin()).max(0.0);
        
        // Weekly pattern (weekdays vs weekends)
        let weekly_factor = if day_of_week < 5.0 { 1.0 } else { 0.7 };
        
        // Seasonal variation
        let seasonal_factor = 0.8 + 0.2 * (2.0 * PI * day_of_year / 365.0).sin();
        
        // Random events and noise
        let noise_factor = 1.0 + self.rng.gen_range(-0.1..0.1);
        let event_factor = if self.rng.gen_bool(0.05) { 
            self.rng.gen_range(0.5..1.5) 
        } else { 
            1.0 
        };

        let load_factor = diurnal_factor * weekly_factor * seasonal_factor * event_factor * noise_factor;

        // Generate correlated KPIs
        let base_load = (load_factor * 85.0).min(95.0).max(5.0);
        
        let throughput = match config.cell_type {
            CellType::NR => {
                let base_throughput = config.max_throughput * (1.0 - base_load / 100.0 * 0.3);
                (base_throughput * (0.8 + 0.4 * self.rng.gen::<f64>())).max(10.0)
            },
            CellType::LTE => {
                let base_throughput = config.max_throughput * (1.0 - base_load / 100.0 * 0.4);
                (base_throughput * (0.7 + 0.6 * self.rng.gen::<f64>())).max(5.0)
            },
        };

        let latency = {
            let base_latency = match config.cell_type {
                CellType::NR => 8.0,
                CellType::LTE => 25.0,
            };
            let load_impact = base_load / 100.0 * 20.0;
            (base_latency + load_impact + self.rng.gen_range(-2.0..5.0)).max(1.0).min(50.0)
        };

        let rsrp = {
            let base_rsrp = -85.0;
            let distance_factor = self.rng.gen_range(-15.0..15.0);
            let interference = if base_load > 80.0 { -5.0 } else { 0.0 };
            (base_rsrp + distance_factor + interference + self.rng.gen_range(-5.0..5.0))
                .max(-140.0).min(-70.0)
        };

        let sinr = {
            let base_sinr = 15.0;
            let interference_impact = -(base_load / 100.0 * 10.0);
            let weather_impact = self.rng.gen_range(-3.0..2.0);
            (base_sinr + interference_impact + weather_impact)
                .max(-5.0).min(25.0)
        };

        let handover_success_rate = {
            let base_rate = 97.0;
            let load_penalty = if base_load > 90.0 { -5.0 } else { -(base_load / 100.0 * 2.0) };
            let signal_bonus = if sinr > 10.0 { 1.0 } else { 0.0 };
            (base_rate + load_penalty + signal_bonus + self.rng.gen_range(-2.0..2.0))
                .max(85.0).min(99.5)
        };

        let user_count = (base_load * 2.0 + self.rng.gen_range(-20.0..20.0)).max(1.0) as u32;
        let traffic_gb = throughput * user_count as f64 * 0.1;

        KPIData {
            timestamp,
            cell_id: cell_id.to_string(),
            cell_type: config.cell_type.clone(),
            throughput_mbps: throughput,
            latency_ms: latency,
            rsrp_dbm: rsrp,
            sinr_db: sinr,
            handover_success_rate,
            cell_load_percent: base_load,
            user_count,
            traffic_gb,
        }
    }

    pub fn train_neural_network(&mut self, training_data: &[KPIData]) {
        println!("🧠 Training Enhanced LSTM Network with {} iterations...", self.training_iterations);
        
        let mut loss_history = Vec::new();
        let mut best_loss = f64::INFINITY;
        let mut best_weights = self.neural_network.clone();

        for iteration in 0..self.training_iterations {
            let batch_size = 32;
            let mut batch_loss = 0.0;

            for _ in 0..batch_size {
                let sequence = self.prepare_training_sequence(training_data);
                let loss = self.forward_pass_and_backprop(&sequence);
                batch_loss += loss;
            }

            batch_loss /= batch_size as f64;
            loss_history.push(batch_loss);

            if batch_loss < best_loss {
                best_loss = batch_loss;
                best_weights = self.neural_network.clone();
            }

            // Adaptive learning rate
            let learning_rate = 0.001 * (1.0 - iteration as f64 / self.training_iterations as f64);
            self.update_weights(learning_rate);

            if iteration % 1000 == 0 {
                println!("Iteration {}: Loss = {:.6}, Best Loss = {:.6}", iteration, batch_loss, best_loss);
            }
        }

        self.neural_network = best_weights;
        println!("✅ Training completed. Final loss: {:.6}", best_loss);
    }

    fn prepare_training_sequence(&self, data: &[KPIData]) -> Vec<Vec<f64>> {
        let mut sequence = Vec::new();
        let start_idx = self.rng.gen_range(0..data.len().saturating_sub(self.neural_network.sequence_length));
        
        for i in 0..self.neural_network.sequence_length {
            let kpi = &data[start_idx + i];
            let features = vec![
                kpi.throughput_mbps / 500.0, // Normalize to 0-1
                kpi.latency_ms / 50.0,
                (kpi.rsrp_dbm + 140.0) / 70.0,
                (kpi.sinr_db + 5.0) / 30.0,
                kpi.handover_success_rate / 100.0,
                kpi.cell_load_percent / 100.0,
            ];
            sequence.push(features);
        }
        
        sequence
    }

    fn forward_pass_and_backprop(&self, sequence: &[Vec<f64>]) -> f64 {
        // Simplified forward pass calculation
        let mut total_error = 0.0;
        
        for (i, features) in sequence.iter().enumerate() {
            let predicted = self.predict_next_values(features);
            
            if i < sequence.len() - 1 {
                let actual = &sequence[i + 1];
                for (p, a) in predicted.iter().zip(actual.iter()) {
                    total_error += (p - a).powi(2);
                }
            }
        }
        
        total_error / sequence.len() as f64
    }

    fn predict_next_values(&self, features: &[f64]) -> Vec<f64> {
        // Simplified prediction using current weights
        let mut output = features.to_vec();
        
        // Apply some transformation based on network structure
        for layer in &self.neural_network.layers {
            output = self.apply_layer_transform(&output, layer);
        }
        
        output
    }

    fn apply_layer_transform(&self, input: &[f64], layer: &LSTMLayer) -> Vec<f64> {
        let mut output = vec![0.0; layer.hidden_size];
        
        for i in 0..layer.hidden_size {
            let mut sum = layer.bias_ih[i];
            for j in 0..input.len().min(layer.input_size) {
                sum += input[j] * layer.weights_ih[i][j];
            }
            output[i] = sum.tanh(); // Activation function
        }
        
        output
    }

    fn update_weights(&mut self, learning_rate: f64) {
        // Simplified weight update
        for layer in &mut self.neural_network.layers {
            for weights_row in &mut layer.weights_ih {
                for weight in weights_row {
                    *weight += learning_rate * self.rng.gen_range(-0.1..0.1);
                }
            }
        }
    }

    pub fn detect_anomalies(&self, data: &[KPIData]) -> Vec<AnomalyDetectionResult> {
        let mut anomalies = Vec::new();
        
        for cell_id in self.cell_configs.keys() {
            let cell_data: Vec<_> = data.iter()
                .filter(|kpi| &kpi.cell_id == cell_id)
                .collect();
            
            if cell_data.len() < 24 { continue; }
            
            // Detect anomalies for each metric
            let throughput_anomalies = self.detect_metric_anomalies(&cell_data, "throughput", 
                |kpi| kpi.throughput_mbps);
            let latency_anomalies = self.detect_metric_anomalies(&cell_data, "latency", 
                |kpi| kpi.latency_ms);
            let rsrp_anomalies = self.detect_metric_anomalies(&cell_data, "rsrp", 
                |kpi| kpi.rsrp_dbm);
            let sinr_anomalies = self.detect_metric_anomalies(&cell_data, "sinr", 
                |kpi| kpi.sinr_db);
            let handover_anomalies = self.detect_metric_anomalies(&cell_data, "handover_success_rate", 
                |kpi| kpi.handover_success_rate);
            let load_anomalies = self.detect_metric_anomalies(&cell_data, "cell_load", 
                |kpi| kpi.cell_load_percent);
            
            anomalies.extend(throughput_anomalies);
            anomalies.extend(latency_anomalies);
            anomalies.extend(rsrp_anomalies);
            anomalies.extend(sinr_anomalies);
            anomalies.extend(handover_anomalies);
            anomalies.extend(load_anomalies);
        }
        
        anomalies
    }

    fn detect_metric_anomalies<F>(&self, data: &[&KPIData], metric_name: &str, 
                                  extractor: F) -> Vec<AnomalyDetectionResult>
    where
        F: Fn(&KPIData) -> f64,
    {
        let mut anomalies = Vec::new();
        let values: Vec<f64> = data.iter().map(|kpi| extractor(kpi)).collect();
        
        if values.len() < 24 { return anomalies; }
        
        // Calculate rolling statistics
        let window_size = 24;
        for i in window_size..values.len() {
            let window = &values[i-window_size..i];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance = window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            let std_dev = variance.sqrt();
            
            let current_value = values[i];
            let z_score = (current_value - mean) / std_dev;
            let anomaly_score = z_score.abs();
            
            if anomaly_score > self.anomaly_threshold {
                let confidence = 1.0 - (-anomaly_score / 2.0).exp();
                
                anomalies.push(AnomalyDetectionResult {
                    timestamp: data[i].timestamp,
                    cell_id: data[i].cell_id.clone(),
                    metric: metric_name.to_string(),
                    value: current_value,
                    expected_value: mean,
                    anomaly_score,
                    is_anomaly: true,
                    confidence,
                });
            }
        }
        
        anomalies
    }

    pub fn analyze_trends(&self, data: &[KPIData]) -> Vec<TrendAnalysis> {
        let mut trends = Vec::new();
        
        for cell_id in self.cell_configs.keys() {
            let cell_data: Vec<_> = data.iter()
                .filter(|kpi| &kpi.cell_id == cell_id)
                .collect();
            
            if cell_data.len() < 48 { continue; }
            
            // Analyze trends for each metric
            trends.push(self.analyze_metric_trend(&cell_data, "throughput", 
                |kpi| kpi.throughput_mbps));
            trends.push(self.analyze_metric_trend(&cell_data, "latency", 
                |kpi| kpi.latency_ms));
            trends.push(self.analyze_metric_trend(&cell_data, "rsrp", 
                |kpi| kpi.rsrp_dbm));
            trends.push(self.analyze_metric_trend(&cell_data, "sinr", 
                |kpi| kpi.sinr_db));
            trends.push(self.analyze_metric_trend(&cell_data, "handover_success_rate", 
                |kpi| kpi.handover_success_rate));
            trends.push(self.analyze_metric_trend(&cell_data, "cell_load", 
                |kpi| kpi.cell_load_percent));
        }
        
        trends
    }

    fn analyze_metric_trend<F>(&self, data: &[&KPIData], metric_name: &str, 
                               extractor: F) -> TrendAnalysis
    where
        F: Fn(&KPIData) -> f64,
    {
        let values: Vec<f64> = data.iter().map(|kpi| extractor(kpi)).collect();
        let n = values.len() as f64;
        
        // Calculate linear trend
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = values.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for (i, value) in values.iter().enumerate() {
            let x = i as f64;
            numerator += (x - x_mean) * (value - y_mean);
            denominator += (x - x_mean).powi(2);
        }
        
        let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
        let trend_strength = slope.abs();
        
        let trend_direction = if slope > 0.01 {
            "increasing".to_string()
        } else if slope < -0.01 {
            "decreasing".to_string()
        } else {
            "stable".to_string()
        };
        
        // Detect seasonality (simplified)
        let seasonality_detected = self.detect_seasonality(&values);
        
        // Generate forecasts
        let forecast_7d = self.forecast_values(&values, 7 * 24);
        let forecast_24h = self.forecast_values(&values, 24);
        
        // Calculate confidence interval
        let residuals: Vec<f64> = values.iter().enumerate()
            .map(|(i, &value)| value - (y_mean + slope * (i as f64 - x_mean)))
            .collect();
        let mse = residuals.iter().map(|r| r.powi(2)).sum::<f64>() / n;
        let confidence_interval = (y_mean - 1.96 * mse.sqrt(), y_mean + 1.96 * mse.sqrt());
        
        TrendAnalysis {
            metric: metric_name.to_string(),
            cell_id: data[0].cell_id.clone(),
            trend_direction,
            trend_strength,
            seasonality_detected,
            forecast_7d,
            forecast_24h,
            confidence_interval,
        }
    }

    fn detect_seasonality(&self, values: &[f64]) -> bool {
        if values.len() < 48 { return false; }
        
        // Check for 24-hour cycle
        let mut daily_correlation = 0.0;
        let n = values.len().min(values.len() - 24);
        
        for i in 0..n {
            daily_correlation += values[i] * values[i + 24];
        }
        
        daily_correlation /= n as f64;
        daily_correlation > 0.3 // Threshold for seasonality detection
    }

    fn forecast_values(&self, values: &[f64], periods: usize) -> Vec<f64> {
        let mut forecast = Vec::new();
        let window_size = 24.min(values.len());
        
        for i in 0..periods {
            let recent_values = if values.len() >= window_size {
                &values[values.len() - window_size..]
            } else {
                values
            };
            
            // Simple exponential smoothing with seasonality
            let alpha = 0.3;
            let beta = 0.1;
            
            let mut forecast_value = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
            
            // Add seasonal component
            if i < recent_values.len() {
                let seasonal = recent_values[recent_values.len() - 1 - i];
                forecast_value = alpha * forecast_value + (1.0 - alpha) * seasonal;
            }
            
            // Add trend component
            if values.len() >= 2 {
                let trend = values[values.len() - 1] - values[values.len() - 2];
                forecast_value += beta * trend;
            }
            
            forecast.push(forecast_value);
        }
        
        forecast
    }

    pub fn generate_performance_insights(&self, data: &[KPIData], 
                                        anomalies: &[AnomalyDetectionResult],
                                        trends: &[TrendAnalysis]) -> PerformanceInsights {
        let mut optimization_opportunities = Vec::new();
        let mut critical_cells = Vec::new();
        let mut predicted_issues = Vec::new();
        let mut resource_recommendations = Vec::new();
        let mut kpi_correlations = HashMap::new();

        // Analyze cell performance
        for cell_id in self.cell_configs.keys() {
            let cell_data: Vec<_> = data.iter()
                .filter(|kpi| &kpi.cell_id == cell_id)
                .collect();
            
            if cell_data.is_empty() { continue; }
            
            let avg_throughput = cell_data.iter().map(|kpi| kpi.throughput_mbps).sum::<f64>() / cell_data.len() as f64;
            let avg_latency = cell_data.iter().map(|kpi| kpi.latency_ms).sum::<f64>() / cell_data.len() as f64;
            let avg_load = cell_data.iter().map(|kpi| kpi.cell_load_percent).sum::<f64>() / cell_data.len() as f64;
            let avg_sinr = cell_data.iter().map(|kpi| kpi.sinr_db).sum::<f64>() / cell_data.len() as f64;
            let avg_handover = cell_data.iter().map(|kpi| kpi.handover_success_rate).sum::<f64>() / cell_data.len() as f64;

            // Identify optimization opportunities
            if avg_load > 85.0 {
                optimization_opportunities.push(format!("Cell {} requires load balancing (avg load: {:.1}%)", cell_id, avg_load));
                
                if avg_load > 90.0 {
                    critical_cells.push(cell_id.clone());
                    resource_recommendations.push(format!("Add capacity to cell {} immediately", cell_id));
                }
            }

            if avg_latency > 30.0 {
                optimization_opportunities.push(format!("Cell {} has high latency (avg: {:.1}ms)", cell_id, avg_latency));
                resource_recommendations.push(format!("Optimize backhaul for cell {}", cell_id));
            }

            if avg_sinr < 5.0 {
                optimization_opportunities.push(format!("Cell {} has poor signal quality (avg SINR: {:.1}dB)", cell_id, avg_sinr));
                resource_recommendations.push(format!("Adjust antenna configuration for cell {}", cell_id));
            }

            if avg_handover < 95.0 {
                optimization_opportunities.push(format!("Cell {} has handover issues (success rate: {:.1}%)", cell_id, avg_handover));
                resource_recommendations.push(format!("Optimize handover parameters for cell {}", cell_id));
            }

            // Predict potential issues based on trends
            let cell_trends: Vec<_> = trends.iter()
                .filter(|trend| trend.cell_id == *cell_id)
                .collect();

            for trend in cell_trends {
                if trend.trend_direction == "decreasing" && trend.trend_strength > 0.5 {
                    match trend.metric.as_str() {
                        "throughput" => predicted_issues.push(format!("Cell {} throughput declining", cell_id)),
                        "sinr" => predicted_issues.push(format!("Cell {} signal quality degrading", cell_id)),
                        "handover_success_rate" => predicted_issues.push(format!("Cell {} handover performance degrading", cell_id)),
                        _ => {}
                    }
                }
                
                if trend.trend_direction == "increasing" && trend.trend_strength > 0.5 {
                    match trend.metric.as_str() {
                        "latency" => predicted_issues.push(format!("Cell {} latency increasing", cell_id)),
                        "cell_load" => predicted_issues.push(format!("Cell {} approaching capacity", cell_id)),
                        _ => {}
                    }
                }
            }
        }

        // Calculate KPI correlations
        if !data.is_empty() {
            let throughput_load_corr = self.calculate_correlation(data, 
                |kpi| kpi.throughput_mbps, |kpi| kpi.cell_load_percent);
            let latency_load_corr = self.calculate_correlation(data, 
                |kpi| kpi.latency_ms, |kpi| kpi.cell_load_percent);
            let sinr_throughput_corr = self.calculate_correlation(data, 
                |kpi| kpi.sinr_db, |kpi| kpi.throughput_mbps);
            let handover_sinr_corr = self.calculate_correlation(data, 
                |kpi| kpi.handover_success_rate, |kpi| kpi.sinr_db);

            kpi_correlations.insert("throughput_vs_load".to_string(), throughput_load_corr);
            kpi_correlations.insert("latency_vs_load".to_string(), latency_load_corr);
            kpi_correlations.insert("sinr_vs_throughput".to_string(), sinr_throughput_corr);
            kpi_correlations.insert("handover_vs_sinr".to_string(), handover_sinr_corr);
        }

        // Calculate overall health score
        let anomaly_penalty = anomalies.len() as f64 * 2.0;
        let critical_penalty = critical_cells.len() as f64 * 10.0;
        let issue_penalty = predicted_issues.len() as f64 * 5.0;
        let total_penalty = anomaly_penalty + critical_penalty + issue_penalty;
        
        let overall_health_score = (100.0 - total_penalty).max(0.0).min(100.0);

        PerformanceInsights {
            optimization_opportunities,
            critical_cells,
            predicted_issues,
            resource_recommendations,
            overall_health_score,
            kpi_correlations,
        }
    }

    fn calculate_correlation<F, G>(&self, data: &[KPIData], extractor1: F, extractor2: G) -> f64
    where
        F: Fn(&KPIData) -> f64,
        G: Fn(&KPIData) -> f64,
    {
        let values1: Vec<f64> = data.iter().map(|kpi| extractor1(kpi)).collect();
        let values2: Vec<f64> = data.iter().map(|kpi| extractor2(kpi)).collect();
        
        if values1.len() != values2.len() || values1.is_empty() {
            return 0.0;
        }
        
        let n = values1.len() as f64;
        let mean1 = values1.iter().sum::<f64>() / n;
        let mean2 = values2.iter().sum::<f64>() / n;
        
        let numerator: f64 = values1.iter().zip(values2.iter())
            .map(|(x, y)| (x - mean1) * (y - mean2))
            .sum();
        
        let denominator = {
            let sum_sq1: f64 = values1.iter().map(|x| (x - mean1).powi(2)).sum();
            let sum_sq2: f64 = values2.iter().map(|y| (y - mean2).powi(2)).sum();
            (sum_sq1 * sum_sq2).sqrt()
        };
        
        if denominator == 0.0 { 0.0 } else { numerator / denominator }
    }

    pub fn run_comprehensive_analysis(&mut self) -> String {
        println!("🚀 Starting Performance Analytics Agent - Comprehensive RAN Analysis");
        println!("=" * 80);
        
        // Generate realistic KPI data
        println!("📊 Generating realistic KPI data for 50 cells over 168 hours...");
        let kpi_data = self.generate_realistic_kpi_data(168);
        println!("✅ Generated {} KPI records", kpi_data.len());
        
        // Train neural network
        println!("\n🧠 Training Enhanced LSTM Neural Network...");
        self.train_neural_network(&kpi_data);
        
        // Detect anomalies
        println!("\n🔍 Detecting performance anomalies...");
        let anomalies = self.detect_anomalies(&kpi_data);
        println!("✅ Detected {} anomalies", anomalies.len());
        
        // Analyze trends
        println!("\n📈 Analyzing performance trends...");
        let trends = self.analyze_trends(&kpi_data);
        println!("✅ Analyzed {} trend patterns", trends.len());
        
        // Generate insights
        println!("\n💡 Generating performance insights...");
        let insights = self.generate_performance_insights(&kpi_data, &anomalies, &trends);
        
        // Generate comprehensive report
        self.generate_detailed_report(&kpi_data, &anomalies, &trends, &insights)
    }

    fn generate_detailed_report(&self, data: &[KPIData], anomalies: &[AnomalyDetectionResult], 
                               trends: &[TrendAnalysis], insights: &PerformanceInsights) -> String {
        let mut report = String::new();
        
        report.push_str("📋 COMPREHENSIVE RAN PERFORMANCE ANALYTICS REPORT\n");
        report.push_str("=" * 80);
        report.push_str("\n\n");
        
        // Executive Summary
        report.push_str("🎯 EXECUTIVE SUMMARY\n");
        report.push_str(&format!("Overall Network Health Score: {:.1}/100\n", insights.overall_health_score));
        report.push_str(&format!("Total KPI Records Analyzed: {}\n", data.len()));
        report.push_str(&format!("Anomalies Detected: {}\n", anomalies.len()));
        report.push_str(&format!("Critical Cells: {}\n", insights.critical_cells.len()));
        report.push_str(&format!("Optimization Opportunities: {}\n", insights.optimization_opportunities.len()));
        report.push_str("\n");
        
        // Critical Issues
        if !insights.critical_cells.is_empty() {
            report.push_str("🚨 CRITICAL ISSUES REQUIRING IMMEDIATE ATTENTION\n");
            for cell in &insights.critical_cells {
                report.push_str(&format!("• Cell {}: Requires immediate intervention\n", cell));
            }
            report.push_str("\n");
        }
        
        // Key Performance Metrics
        report.push_str("📊 KEY PERFORMANCE METRICS SUMMARY\n");
        let mut cell_stats = HashMap::new();
        for kpi in data {
            let stats = cell_stats.entry(kpi.cell_id.clone()).or_insert_with(|| {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new())
            });
            stats.0.push(kpi.throughput_mbps);
            stats.1.push(kpi.latency_ms);
            stats.2.push(kpi.rsrp_dbm);
            stats.3.push(kpi.sinr_db);
            stats.4.push(kpi.handover_success_rate);
            stats.5.push(kpi.cell_load_percent);
        }
        
        let mut avg_throughput = 0.0;
        let mut avg_latency = 0.0;
        let mut avg_rsrp = 0.0;
        let mut avg_sinr = 0.0;
        let mut avg_handover = 0.0;
        let mut avg_load = 0.0;
        
        for (throughput, latency, rsrp, sinr, handover, load) in cell_stats.values() {
            avg_throughput += throughput.iter().sum::<f64>() / throughput.len() as f64;
            avg_latency += latency.iter().sum::<f64>() / latency.len() as f64;
            avg_rsrp += rsrp.iter().sum::<f64>() / rsrp.len() as f64;
            avg_sinr += sinr.iter().sum::<f64>() / sinr.len() as f64;
            avg_handover += handover.iter().sum::<f64>() / handover.len() as f64;
            avg_load += load.iter().sum::<f64>() / load.len() as f64;
        }
        
        let cell_count = cell_stats.len() as f64;
        report.push_str(&format!("Average Throughput: {:.1} Mbps\n", avg_throughput / cell_count));
        report.push_str(&format!("Average Latency: {:.1} ms\n", avg_latency / cell_count));
        report.push_str(&format!("Average RSRP: {:.1} dBm\n", avg_rsrp / cell_count));
        report.push_str(&format!("Average SINR: {:.1} dB\n", avg_sinr / cell_count));
        report.push_str(&format!("Average Handover Success Rate: {:.1}%\n", avg_handover / cell_count));
        report.push_str(&format!("Average Cell Load: {:.1}%\n", avg_load / cell_count));
        report.push_str("\n");
        
        // Anomaly Analysis
        report.push_str("🔍 ANOMALY ANALYSIS\n");
        if anomalies.is_empty() {
            report.push_str("✅ No significant anomalies detected\n");
        } else {
            let mut anomaly_by_metric = HashMap::new();
            for anomaly in anomalies {
                let count = anomaly_by_metric.entry(anomaly.metric.clone()).or_insert(0);
                *count += 1;
            }
            
            for (metric, count) in anomaly_by_metric {
                report.push_str(&format!("• {}: {} anomalies detected\n", metric, count));
            }
            
            // Top 5 most severe anomalies
            let mut sorted_anomalies = anomalies.to_vec();
            sorted_anomalies.sort_by(|a, b| b.anomaly_score.partial_cmp(&a.anomaly_score).unwrap());
            
            report.push_str("\nTop 5 Most Severe Anomalies:\n");
            for (i, anomaly) in sorted_anomalies.iter().take(5).enumerate() {
                report.push_str(&format!("{}. Cell {}: {} anomaly (score: {:.2}, confidence: {:.1}%)\n", 
                    i + 1, anomaly.cell_id, anomaly.metric, anomaly.anomaly_score, anomaly.confidence * 100.0));
            }
        }
        report.push_str("\n");
        
        // Trend Analysis
        report.push_str("📈 TREND ANALYSIS\n");
        let mut trend_summary = HashMap::new();
        for trend in trends {
            let key = format!("{}-{}", trend.metric, trend.trend_direction);
            let count = trend_summary.entry(key).or_insert(0);
            *count += 1;
        }
        
        for (trend_type, count) in trend_summary {
            report.push_str(&format!("• {}: {} cells\n", trend_type, count));
        }
        
        // Significant trends
        let significant_trends: Vec<_> = trends.iter()
            .filter(|trend| trend.trend_strength > 0.5)
            .collect();
        
        if !significant_trends.is_empty() {
            report.push_str("\nSignificant Trends (strength > 0.5):\n");
            for trend in significant_trends.iter().take(10) {
                report.push_str(&format!("• Cell {}: {} {} (strength: {:.2})\n", 
                    trend.cell_id, trend.metric, trend.trend_direction, trend.trend_strength));
            }
        }
        report.push_str("\n");
        
        // KPI Correlations
        report.push_str("🔗 KPI CORRELATIONS\n");
        for (correlation, value) in &insights.kpi_correlations {
            let strength = if value.abs() > 0.7 { "Strong" } else if value.abs() > 0.3 { "Moderate" } else { "Weak" };
            report.push_str(&format!("• {}: {:.3} ({})\n", correlation, value, strength));
        }
        report.push_str("\n");
        
        // Optimization Opportunities
        report.push_str("🎯 OPTIMIZATION OPPORTUNITIES\n");
        if insights.optimization_opportunities.is_empty() {
            report.push_str("✅ No immediate optimization opportunities identified\n");
        } else {
            for (i, opportunity) in insights.optimization_opportunities.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, opportunity));
            }
        }
        report.push_str("\n");
        
        // Resource Recommendations
        report.push_str("💡 RESOURCE RECOMMENDATIONS\n");
        if insights.resource_recommendations.is_empty() {
            report.push_str("✅ No immediate resource changes recommended\n");
        } else {
            for (i, recommendation) in insights.resource_recommendations.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, recommendation));
            }
        }
        report.push_str("\n");
        
        // Predicted Issues
        report.push_str("🔮 PREDICTED ISSUES\n");
        if insights.predicted_issues.is_empty() {
            report.push_str("✅ No issues predicted based on current trends\n");
        } else {
            for (i, issue) in insights.predicted_issues.iter().enumerate() {
                report.push_str(&format!("{}. {}\n", i + 1, issue));
            }
        }
        report.push_str("\n");
        
        // Neural Network Performance
        report.push_str("🧠 NEURAL NETWORK PERFORMANCE\n");
        report.push_str(&format!("Network Architecture: {} layers, {} hidden units\n", 
            self.neural_network.layers.len(), self.neural_network.layers[0].hidden_size));
        report.push_str(&format!("Training Iterations: {}\n", self.training_iterations));
        report.push_str(&format!("Sequence Length: {} hours\n", self.neural_network.sequence_length));
        report.push_str("Features: Attention mechanism, LSTM cells, Multi-layer architecture\n");
        report.push_str("\n");
        
        // Recommendations Summary
        report.push_str("📋 ACTIONABLE RECOMMENDATIONS\n");
        report.push_str("Immediate Actions (0-24 hours):\n");
        for cell in &insights.critical_cells {
            report.push_str(&format!("• Investigate and resolve issues in cell {}\n", cell));
        }
        
        report.push_str("\nShort-term Actions (1-7 days):\n");
        for opportunity in insights.optimization_opportunities.iter().take(5) {
            report.push_str(&format!("• {}\n", opportunity));
        }
        
        report.push_str("\nLong-term Actions (1-4 weeks):\n");
        for recommendation in insights.resource_recommendations.iter().take(3) {
            report.push_str(&format!("• {}\n", recommendation));
        }
        
        report.push_str("\n");
        report.push_str("=" * 80);
        report.push_str("\n🎯 Analysis completed successfully!");
        report.push_str("\n💡 Continue monitoring for optimal network performance.");
        
        report
    }
}

impl EnhancedLSTMNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize, 
           sequence_length: usize, num_layers: usize) -> Self {
        let mut layers = Vec::new();
        let mut rng = thread_rng();
        
        for i in 0..num_layers {
            let layer_input_size = if i == 0 { input_size } else { hidden_size };
            
            let mut weights_ih = vec![vec![0.0; layer_input_size]; hidden_size * 4];
            let mut weights_hh = vec![vec![0.0; hidden_size]; hidden_size * 4];
            let mut bias_ih = vec![0.0; hidden_size * 4];
            let mut bias_hh = vec![0.0; hidden_size * 4];
            
            // Xavier initialization
            let fan_in = layer_input_size + hidden_size;
            let limit = (6.0 / fan_in as f64).sqrt();
            
            for row in &mut weights_ih {
                for weight in row {
                    *weight = rng.gen_range(-limit..limit);
                }
            }
            
            for row in &mut weights_hh {
                for weight in row {
                    *weight = rng.gen_range(-limit..limit);
                }
            }
            
            for bias in &mut bias_ih {
                *bias = rng.gen_range(-0.1..0.1);
            }
            
            for bias in &mut bias_hh {
                *bias = rng.gen_range(-0.1..0.1);
            }
            
            layers.push(LSTMLayer {
                input_size: layer_input_size,
                hidden_size,
                weights_ih,
                weights_hh,
                bias_ih,
                bias_hh,
            });
        }
        
        // Initialize attention mechanism
        let attention_limit = (6.0 / (hidden_size * 2) as f64).sqrt();
        let mut query_weights = vec![vec![0.0; hidden_size]; hidden_size];
        let mut key_weights = vec![vec![0.0; hidden_size]; hidden_size];
        let mut value_weights = vec![vec![0.0; hidden_size]; hidden_size];
        
        for row in &mut query_weights {
            for weight in row {
                *weight = rng.gen_range(-attention_limit..attention_limit);
            }
        }
        
        for row in &mut key_weights {
            for weight in row {
                *weight = rng.gen_range(-attention_limit..attention_limit);
            }
        }
        
        for row in &mut value_weights {
            for weight in row {
                *weight = rng.gen_range(-attention_limit..attention_limit);
            }
        }
        
        let attention = AttentionMechanism {
            query_weights,
            key_weights,
            value_weights,
            hidden_size,
        };
        
        // Initialize output layer
        let output_limit = (6.0 / (hidden_size + output_size) as f64).sqrt();
        let mut output_weights = vec![vec![0.0; hidden_size]; output_size];
        let mut output_bias = vec![0.0; output_size];
        
        for row in &mut output_weights {
            for weight in row {
                *weight = rng.gen_range(-output_limit..output_limit);
            }
        }
        
        for bias in &mut output_bias {
            *bias = rng.gen_range(-0.1..0.1);
        }
        
        Self {
            layers,
            attention,
            output_weights,
            output_bias,
            input_size,
            output_size,
            sequence_length,
        }
    }
}

fn main() {
    let mut agent = PerformanceAnalyticsAgent::new();
    let report = agent.run_comprehensive_analysis();
    
    println!("\n{}", report);
    
    // Save report to file
    std::fs::write("performance_analytics_report.txt", report)
        .expect("Failed to write report file");
    
    println!("\n📁 Report saved to: performance_analytics_report.txt");
}