// Neural Performance Monitor for RAN Intelligence Platform
// ML-Coordinator Agent - Real-time Neural Network Performance Tracking

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub timestamp: u64,
    pub agent_id: String,
    pub metric_name: String,
    pub metric_value: f64,
    pub neural_network_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeStats {
    pub accuracy: f64,
    pub response_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cognitive_load: f64,
    pub training_loss: f64,
    pub prediction_confidence: f64,
    pub throughput_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationAlert {
    pub alert_type: String,
    pub severity: String,
    pub agent_id: String,
    pub message: String,
    pub recommended_action: String,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPerformanceReport {
    pub timestamp: u64,
    pub total_agents: usize,
    pub active_neural_networks: usize,
    pub ensemble_accuracy: f64,
    pub avg_response_time_ms: f64,
    pub total_memory_usage_mb: f64,
    pub performance_alerts: Vec<OptimizationAlert>,
    pub top_performers: Vec<String>,
    pub optimization_recommendations: Vec<String>,
}

pub struct NeuralPerformanceMonitor {
    metrics_history: Arc<RwLock<Vec<PerformanceMetric>>>,
    real_time_stats: Arc<RwLock<HashMap<String, RealTimeStats>>>,
    alerts: Arc<RwLock<Vec<OptimizationAlert>>>,
    baseline_metrics: Arc<RwLock<HashMap<String, f64>>>,
    monitoring_interval: Duration,
}

impl NeuralPerformanceMonitor {
    pub fn new(monitoring_interval_seconds: u64) -> Self {
        Self {
            metrics_history: Arc::new(RwLock::new(Vec::new())),
            real_time_stats: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(Vec::new())),
            baseline_metrics: Arc::new(RwLock::new(HashMap::new())),
            monitoring_interval: Duration::from_secs(monitoring_interval_seconds),
        }
    }

    /// Start continuous monitoring of neural network performance
    pub async fn start_monitoring(&self) {
        println!("📊 Starting neural performance monitoring...");
        
        let metrics_history = Arc::clone(&self.metrics_history);
        let real_time_stats = Arc::clone(&self.real_time_stats);
        let alerts = Arc::clone(&self.alerts);
        let baseline_metrics = Arc::clone(&self.baseline_metrics);
        let interval = self.monitoring_interval;
        
        tokio::spawn(async move {
            let mut monitoring_interval = tokio::time::interval(interval);
            
            loop {
                monitoring_interval.tick().await;
                
                // Collect metrics from all agents
                let current_stats = Self::collect_current_metrics().await;
                
                // Update real-time stats
                {
                    let mut stats = real_time_stats.write().await;
                    for (agent_id, agent_stats) in current_stats {
                        stats.insert(agent_id.clone(), agent_stats.clone());
                        
                        // Store historical metrics
                        let mut history = metrics_history.write().await;
                        let timestamp = SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs();
                        
                        history.push(PerformanceMetric {
                            timestamp,
                            agent_id: agent_id.clone(),
                            metric_name: "accuracy".to_string(),
                            metric_value: agent_stats.accuracy,
                            neural_network_id: format!("nn-{}", agent_id),
                        });
                        
                        history.push(PerformanceMetric {
                            timestamp,
                            agent_id: agent_id.clone(),
                            metric_name: "response_time_ms".to_string(),
                            metric_value: agent_stats.response_time_ms,
                            neural_network_id: format!("nn-{}", agent_id),
                        });
                        
                        history.push(PerformanceMetric {
                            timestamp,
                            agent_id: agent_id.clone(),
                            metric_name: "memory_usage_mb".to_string(),
                            metric_value: agent_stats.memory_usage_mb,
                            neural_network_id: format!("nn-{}", agent_id),
                        });
                    }
                }
                
                // Check for performance issues and generate alerts
                Self::check_performance_alerts(&real_time_stats, &alerts, &baseline_metrics).await;
                
                // Cleanup old metrics (keep last 24 hours)
                Self::cleanup_old_metrics(&metrics_history, 24 * 3600).await;
            }
        });
    }

    /// Collect current metrics from all neural networks
    async fn collect_current_metrics() -> HashMap<String, RealTimeStats> {
        let mut current_metrics = HashMap::new();
        
        // In real implementation, this would query actual neural networks
        // For now, simulate metrics for key agents
        let agent_ids = vec![
            "agent-1751707213117", // Optimization-Engineer
            "agent-1751707213212", // Assurance-Specialist
            "agent-1751707213319", // Intelligence-Researcher
            "agent-1751707213434", // ML-Coordinator
            "agent-1751707229892", // Foundation-Architect
        ];
        
        for agent_id in agent_ids {
            let stats = RealTimeStats {
                accuracy: 0.85 + (rand::random::<f64>() * 0.15), // 85-100%
                response_time_ms: 100.0 + (rand::random::<f64>() * 400.0), // 100-500ms
                memory_usage_mb: 10.0 + (rand::random::<f64>() * 20.0), // 10-30MB
                cognitive_load: rand::random::<f64>(), // 0-1
                training_loss: 0.1 + (rand::random::<f64>() * 0.9), // 0.1-1.0
                prediction_confidence: 0.7 + (rand::random::<f64>() * 0.3), // 70-100%
                throughput_per_second: 10.0 + (rand::random::<f64>() * 90.0), // 10-100 req/s
            };
            
            current_metrics.insert(agent_id.to_string(), stats);
        }
        
        current_metrics
    }

    /// Check for performance issues and generate alerts
    async fn check_performance_alerts(
        real_time_stats: &Arc<RwLock<HashMap<String, RealTimeStats>>>,
        alerts: &Arc<RwLock<Vec<OptimizationAlert>>>,
        baseline_metrics: &Arc<RwLock<HashMap<String, f64>>>,
    ) {
        let stats = real_time_stats.read().await;
        let mut alerts_vec = alerts.write().await;
        let baselines = baseline_metrics.read().await;
        
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        for (agent_id, agent_stats) in stats.iter() {
            // Check accuracy degradation
            if agent_stats.accuracy < 0.85 {
                alerts_vec.push(OptimizationAlert {
                    alert_type: "accuracy_degradation".to_string(),
                    severity: "HIGH".to_string(),
                    agent_id: agent_id.clone(),
                    message: format!("Neural network accuracy dropped to {:.2}%", agent_stats.accuracy * 100.0),
                    recommended_action: "Retrain neural network with recent data".to_string(),
                    timestamp,
                });
            }
            
            // Check response time issues
            if agent_stats.response_time_ms > 500.0 {
                alerts_vec.push(OptimizationAlert {
                    alert_type: "performance_slowdown".to_string(),
                    severity: "MEDIUM".to_string(),
                    agent_id: agent_id.clone(),
                    message: format!("Response time increased to {:.1}ms", agent_stats.response_time_ms),
                    recommended_action: "Optimize neural network architecture or increase compute resources".to_string(),
                    timestamp,
                });
            }
            
            // Check memory usage
            if agent_stats.memory_usage_mb > 25.0 {
                alerts_vec.push(OptimizationAlert {
                    alert_type: "memory_usage_high".to_string(),
                    severity: "LOW".to_string(),
                    agent_id: agent_id.clone(),
                    message: format!("Memory usage at {:.1}MB", agent_stats.memory_usage_mb),
                    recommended_action: "Consider model compression or memory optimization".to_string(),
                    timestamp,
                });
            }
            
            // Check training loss
            if agent_stats.training_loss > 0.8 {
                alerts_vec.push(OptimizationAlert {
                    alert_type: "training_convergence_issue".to_string(),
                    severity: "MEDIUM".to_string(),
                    agent_id: agent_id.clone(),
                    message: format!("Training loss high at {:.3}", agent_stats.training_loss),
                    recommended_action: "Adjust learning rate or check training data quality".to_string(),
                    timestamp,
                });
            }
            
            // Check prediction confidence
            if agent_stats.prediction_confidence < 0.7 {
                alerts_vec.push(OptimizationAlert {
                    alert_type: "low_prediction_confidence".to_string(),
                    severity: "LOW".to_string(),
                    agent_id: agent_id.clone(),
                    message: format!("Prediction confidence low at {:.2}%", agent_stats.prediction_confidence * 100.0),
                    recommended_action: "Increase training data diversity or ensemble size".to_string(),
                    timestamp,
                });
            }
        }
        
        // Keep only recent alerts (last 1 hour)
        let one_hour_ago = timestamp - 3600;
        alerts_vec.retain(|alert| alert.timestamp > one_hour_ago);
    }

    /// Cleanup old metrics to prevent memory bloat
    async fn cleanup_old_metrics(metrics_history: &Arc<RwLock<Vec<PerformanceMetric>>>, retention_seconds: u64) {
        let mut history = metrics_history.write().await;
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - retention_seconds;
        
        history.retain(|metric| metric.timestamp > cutoff_time);
    }

    /// Generate comprehensive performance report
    pub async fn generate_performance_report(&self) -> SwarmPerformanceReport {
        let stats = self.real_time_stats.read().await;
        let alerts = self.alerts.read().await;
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let total_agents = stats.len();
        let active_neural_networks = stats.values().filter(|s| s.accuracy > 0.5).count();
        
        // Calculate ensemble accuracy (weighted average)
        let ensemble_accuracy = if !stats.is_empty() {
            stats.values().map(|s| s.accuracy).sum::<f64>() / stats.len() as f64
        } else {
            0.0
        };
        
        // Calculate average response time
        let avg_response_time_ms = if !stats.is_empty() {
            stats.values().map(|s| s.response_time_ms).sum::<f64>() / stats.len() as f64
        } else {
            0.0
        };
        
        // Calculate total memory usage
        let total_memory_usage_mb = stats.values().map(|s| s.memory_usage_mb).sum::<f64>();
        
        // Find top performers
        let mut performance_pairs: Vec<(String, f64)> = stats.iter()
            .map(|(agent_id, stats)| (agent_id.clone(), stats.accuracy))
            .collect();
        performance_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_performers = performance_pairs.into_iter()
            .take(3)
            .map(|(agent_id, accuracy)| format!("{}: {:.2}%", agent_id, accuracy * 100.0))
            .collect();
        
        // Generate optimization recommendations
        let optimization_recommendations = self.generate_optimization_recommendations(&stats).await;
        
        SwarmPerformanceReport {
            timestamp,
            total_agents,
            active_neural_networks,
            ensemble_accuracy,
            avg_response_time_ms,
            total_memory_usage_mb,
            performance_alerts: alerts.clone(),
            top_performers,
            optimization_recommendations,
        }
    }

    /// Generate optimization recommendations based on current performance
    async fn generate_optimization_recommendations(&self, stats: &HashMap<String, RealTimeStats>) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        // Check overall ensemble performance
        let avg_accuracy = stats.values().map(|s| s.accuracy).sum::<f64>() / stats.len() as f64;
        if avg_accuracy < 0.9 {
            recommendations.push("🎯 Consider ensemble learning to boost overall accuracy".to_string());
        }
        
        // Check response time distribution
        let response_times: Vec<f64> = stats.values().map(|s| s.response_time_ms).collect();
        let max_response_time = response_times.iter().fold(0.0, |max, &x| max.max(x));
        let min_response_time = response_times.iter().fold(f64::MAX, |min, &x| min.min(x));
        
        if max_response_time - min_response_time > 200.0 {
            recommendations.push("⚡ Load balancing needed - large response time variance detected".to_string());
        }
        
        // Check memory usage efficiency
        let total_memory = stats.values().map(|s| s.memory_usage_mb).sum::<f64>();
        if total_memory > 100.0 {
            recommendations.push("💾 Consider memory optimization - high total memory usage".to_string());
        }
        
        // Check cognitive load distribution
        let cognitive_loads: Vec<f64> = stats.values().map(|s| s.cognitive_load).collect();
        let max_cognitive_load = cognitive_loads.iter().fold(0.0, |max, &x| max.max(x));
        
        if max_cognitive_load > 0.8 {
            recommendations.push("🧠 High cognitive load detected - consider workload redistribution".to_string());
        }
        
        // Check prediction confidence
        let avg_confidence = stats.values().map(|s| s.prediction_confidence).sum::<f64>() / stats.len() as f64;
        if avg_confidence < 0.8 {
            recommendations.push("🔮 Low prediction confidence - increase training data or model complexity".to_string());
        }
        
        recommendations
    }

    /// Set baseline metrics for comparison
    pub async fn set_baseline_metrics(&self, metrics: HashMap<String, f64>) {
        let mut baselines = self.baseline_metrics.write().await;
        *baselines = metrics;
    }

    /// Get current performance statistics
    pub async fn get_current_stats(&self) -> HashMap<String, RealTimeStats> {
        self.real_time_stats.read().await.clone()
    }

    /// Get recent alerts
    pub async fn get_recent_alerts(&self, hours: u64) -> Vec<OptimizationAlert> {
        let alerts = self.alerts.read().await;
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - (hours * 3600);
        
        alerts.iter()
            .filter(|alert| alert.timestamp > cutoff_time)
            .cloned()
            .collect()
    }

    /// Export performance data for analysis
    pub async fn export_performance_data(&self, hours: u64) -> Vec<PerformanceMetric> {
        let history = self.metrics_history.read().await;
        let cutoff_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs() - (hours * 3600);
        
        history.iter()
            .filter(|metric| metric.timestamp > cutoff_time)
            .cloned()
            .collect()
    }
}

/// Performance visualization utilities
pub mod visualization {
    use super::*;
    
    pub fn format_performance_dashboard(report: &SwarmPerformanceReport) -> String {
        let mut dashboard = String::new();
        
        dashboard.push_str("🐝 Neural Swarm Performance Dashboard\n");
        dashboard.push_str("=====================================\n\n");
        
        dashboard.push_str(&format!("📊 Overview:\n"));
        dashboard.push_str(&format!("├── Total Agents: {}\n", report.total_agents));
        dashboard.push_str(&format!("├── Active Neural Networks: {}\n", report.active_neural_networks));
        dashboard.push_str(&format!("├── Ensemble Accuracy: {:.2}%\n", report.ensemble_accuracy * 100.0));
        dashboard.push_str(&format!("├── Avg Response Time: {:.1}ms\n", report.avg_response_time_ms));
        dashboard.push_str(&format!("└── Total Memory Usage: {:.1}MB\n\n", report.total_memory_usage_mb));
        
        if !report.top_performers.is_empty() {
            dashboard.push_str("🏆 Top Performers:\n");
            for (i, performer) in report.top_performers.iter().enumerate() {
                dashboard.push_str(&format!("{}. {}\n", i + 1, performer));
            }
            dashboard.push_str("\n");
        }
        
        if !report.performance_alerts.is_empty() {
            dashboard.push_str("⚠️ Performance Alerts:\n");
            for alert in &report.performance_alerts {
                let severity_icon = match alert.severity.as_str() {
                    "HIGH" => "🔴",
                    "MEDIUM" => "🟡",
                    "LOW" => "🟢",
                    _ => "⚪",
                };
                dashboard.push_str(&format!("{} {}: {}\n", severity_icon, alert.alert_type, alert.message));
            }
            dashboard.push_str("\n");
        }
        
        if !report.optimization_recommendations.is_empty() {
            dashboard.push_str("💡 Optimization Recommendations:\n");
            for rec in &report.optimization_recommendations {
                dashboard.push_str(&format!("   {}\n", rec));
            }
        }
        
        dashboard
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_monitor() {
        let monitor = NeuralPerformanceMonitor::new(1);
        
        // Set baseline metrics
        let mut baseline = HashMap::new();
        baseline.insert("accuracy".to_string(), 0.9);
        baseline.insert("response_time_ms".to_string(), 200.0);
        monitor.set_baseline_metrics(baseline).await;
        
        // Generate a performance report
        let report = monitor.generate_performance_report().await;
        
        assert!(report.total_agents >= 0);
        assert!(report.ensemble_accuracy >= 0.0 && report.ensemble_accuracy <= 1.0);
    }
}