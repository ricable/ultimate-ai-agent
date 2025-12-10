//! Training Monitoring and Visualization System
//! 
//! This module provides real-time training monitoring, progress tracking,
//! performance visualization, and comprehensive logging for GPU training.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::gpu_training::{GpuTrainingMetrics, GpuTrainingResult, ModelArchitecture, DeviceInfo};

/// Training monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Enable real-time plotting
    pub enable_plotting: bool,
    /// Update frequency in seconds
    pub update_frequency: f64,
    /// History window size (number of epochs to keep)
    pub history_window: usize,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Save training logs to file
    pub save_logs: bool,
    /// Log file path
    pub log_file_path: String,
    /// Enable GPU memory monitoring
    pub monitor_gpu_memory: bool,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            enable_plotting: true,
            update_frequency: 1.0,
            history_window: 1000,
            enable_profiling: true,
            save_logs: true,
            log_file_path: "training_logs.json".to_string(),
            monitor_gpu_memory: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Alert thresholds for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum training loss before alert
    pub max_training_loss: f32,
    /// Maximum GPU memory usage (MB) before alert
    pub max_gpu_memory: f64,
    /// Maximum training time per epoch (seconds) before alert
    pub max_epoch_time: f64,
    /// Minimum throughput (samples/sec) before alert
    pub min_throughput: f32,
    /// Learning rate below which to alert
    pub min_learning_rate: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_training_loss: 10.0,
            max_gpu_memory: 7000.0, // 7GB
            max_epoch_time: 300.0,  // 5 minutes
            min_throughput: 10.0,   // 10 samples/sec
            min_learning_rate: 1e-8,
        }
    }
}

/// Training event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrainingEvent {
    /// Training started
    TrainingStarted {
        model_name: String,
        architecture: ModelArchitecture,
        timestamp: u64,
    },
    /// Epoch completed
    EpochCompleted {
        model_name: String,
        metrics: GpuTrainingMetrics,
        timestamp: u64,
    },
    /// Training completed
    TrainingCompleted {
        model_name: String,
        result: GpuTrainingResult,
        timestamp: u64,
    },
    /// Alert triggered
    Alert {
        level: AlertLevel,
        message: String,
        model_name: String,
        timestamp: u64,
    },
    /// Checkpoint saved
    CheckpointSaved {
        model_name: String,
        epoch: usize,
        path: String,
        timestamp: u64,
    },
    /// Performance benchmark
    PerformanceBenchmark {
        model_name: String,
        benchmark_result: BenchmarkResult,
        timestamp: u64,
    },
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertLevel {
    Info,
    Warning,
    Error,
    Critical,
}

/// Performance benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub gpu_utilization: f64,
    pub memory_bandwidth: f64,
    pub compute_throughput: f64,
    pub energy_efficiency: f64,
    pub thermal_state: String,
    pub timestamp: u64,
}

/// Real-time training statistics
#[derive(Debug, Clone, Serialize)]
pub struct TrainingStats {
    /// Current training progress
    pub progress: TrainingProgress,
    /// Performance metrics
    pub performance: PerformanceMetrics,
    /// GPU utilization
    pub gpu_stats: GpuStats,
    /// Model comparison data
    pub model_comparison: ModelComparisonStats,
    /// Recent alerts
    pub recent_alerts: Vec<TrainingEvent>,
}

/// Training progress information
#[derive(Debug, Clone, Serialize)]
pub struct TrainingProgress {
    pub models_in_training: usize,
    pub models_completed: usize,
    pub total_models: usize,
    pub overall_progress: f64,
    pub estimated_completion: Option<Duration>,
    pub current_model: Option<String>,
    pub current_epoch: Option<usize>,
    pub total_epochs: Option<usize>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize)]
pub struct PerformanceMetrics {
    pub average_throughput: f32,
    pub peak_throughput: f32,
    pub average_gpu_utilization: f64,
    pub peak_gpu_utilization: f64,
    pub total_training_time: Duration,
    pub energy_consumed: f64, // kWh estimate
    pub carbon_footprint: f64, // kg CO2 estimate
}

/// GPU statistics
#[derive(Debug, Clone, Serialize)]
pub struct GpuStats {
    pub current_memory_usage: f64,
    pub peak_memory_usage: f64,
    pub memory_utilization_history: VecDeque<f64>,
    pub temperature: Option<f64>,
    pub power_draw: Option<f64>,
    pub compute_utilization: Option<f64>,
}

/// Model comparison statistics
#[derive(Debug, Clone, Serialize)]
pub struct ModelComparisonStats {
    pub best_model: Option<String>,
    pub best_loss: Option<f32>,
    pub model_rankings: Vec<ModelRanking>,
    pub convergence_rates: HashMap<String, f64>,
    pub efficiency_scores: HashMap<String, f64>,
}

/// Model ranking information
#[derive(Debug, Clone, Serialize)]
pub struct ModelRanking {
    pub model_name: String,
    pub architecture: ModelArchitecture,
    pub rank: usize,
    pub score: f64,
    pub metrics: HashMap<String, f64>,
}

/// Training monitor main struct
pub struct TrainingMonitor {
    config: MonitorConfig,
    event_sender: broadcast::Sender<TrainingEvent>,
    stats: Arc<Mutex<TrainingStats>>,
    model_metrics: Arc<Mutex<HashMap<String, VecDeque<GpuTrainingMetrics>>>>,
    training_start_times: Arc<Mutex<HashMap<String, Instant>>>,
    log_writer: Option<std::fs::File>,
}

impl TrainingMonitor {
    /// Create new training monitor
    pub fn new(config: MonitorConfig) -> Result<Self> {
        let (event_sender, _) = broadcast::channel(1000);
        
        let stats = Arc::new(Mutex::new(TrainingStats {
            progress: TrainingProgress {
                models_in_training: 0,
                models_completed: 0,
                total_models: 0,
                overall_progress: 0.0,
                estimated_completion: None,
                current_model: None,
                current_epoch: None,
                total_epochs: None,
            },
            performance: PerformanceMetrics {
                average_throughput: 0.0,
                peak_throughput: 0.0,
                average_gpu_utilization: 0.0,
                peak_gpu_utilization: 0.0,
                total_training_time: Duration::new(0, 0),
                energy_consumed: 0.0,
                carbon_footprint: 0.0,
            },
            gpu_stats: GpuStats {
                current_memory_usage: 0.0,
                peak_memory_usage: 0.0,
                memory_utilization_history: VecDeque::new(),
                temperature: None,
                power_draw: None,
                compute_utilization: None,
            },
            model_comparison: ModelComparisonStats {
                best_model: None,
                best_loss: None,
                model_rankings: Vec::new(),
                convergence_rates: HashMap::new(),
                efficiency_scores: HashMap::new(),
            },
            recent_alerts: Vec::new(),
        }));
        
        let model_metrics = Arc::new(Mutex::new(HashMap::new()));
        let training_start_times = Arc::new(Mutex::new(HashMap::new()));
        
        let log_writer = if config.save_logs {
            Some(std::fs::File::create(&config.log_file_path)
                .context("Failed to create log file")?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            event_sender,
            stats,
            model_metrics,
            training_start_times,
            log_writer,
        })
    }
    
    /// Start monitoring a model training
    pub fn start_training(&self, model_name: String, architecture: ModelArchitecture, total_epochs: usize) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        // Record start time
        {
            let mut start_times = self.training_start_times.lock().unwrap();
            start_times.insert(model_name.clone(), Instant::now());
        }
        
        // Initialize metrics history
        {
            let mut metrics = self.model_metrics.lock().unwrap();
            metrics.insert(model_name.clone(), VecDeque::new());
        }
        
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.progress.models_in_training += 1;
            stats.progress.total_models += 1;
            stats.progress.current_model = Some(model_name.clone());
            stats.progress.total_epochs = Some(total_epochs);
            stats.progress.current_epoch = Some(0);
        }
        
        // Send event
        let event = TrainingEvent::TrainingStarted {
            model_name,
            architecture,
            timestamp,
        };
        
        self.send_event(event)?;
        
        log::info!("Started monitoring training");
        Ok(())
    }
    
    /// Record epoch completion
    pub fn record_epoch(&self, model_name: String, metrics: GpuTrainingMetrics) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        // Store metrics
        {
            let mut model_metrics = self.model_metrics.lock().unwrap();
            if let Some(history) = model_metrics.get_mut(&model_name) {
                history.push_back(metrics.clone());
                
                // Keep only recent history
                while history.len() > self.config.history_window {
                    history.pop_front();
                }
            }
        }
        
        // Update stats
        self.update_stats(&model_name, &metrics)?;
        
        // Check for alerts
        self.check_alerts(&model_name, &metrics)?;
        
        // Send event
        let event = TrainingEvent::EpochCompleted {
            model_name,
            metrics,
            timestamp,
        };
        
        self.send_event(event)?;
        
        Ok(())
    }
    
    /// Complete training for a model
    pub fn complete_training(&self, model_name: String, result: GpuTrainingResult) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.progress.models_in_training = stats.progress.models_in_training.saturating_sub(1);
            stats.progress.models_completed += 1;
            stats.progress.overall_progress = 
                stats.progress.models_completed as f64 / stats.progress.total_models as f64;
            
            // Update model comparison
            if stats.model_comparison.best_loss.is_none() || 
               result.best_loss < stats.model_comparison.best_loss.unwrap() {
                stats.model_comparison.best_model = Some(model_name.clone());
                stats.model_comparison.best_loss = Some(result.best_loss);
            }
            
            // Calculate efficiency score
            let efficiency = 1.0 / (result.best_loss as f64 * result.training_time.as_secs_f64());
            stats.model_comparison.efficiency_scores.insert(model_name.clone(), efficiency);
        }
        
        // Send event
        let event = TrainingEvent::TrainingCompleted {
            model_name,
            result,
            timestamp,
        };
        
        self.send_event(event)?;
        
        log::info!("Completed training monitoring");
        Ok(())
    }
    
    /// Record checkpoint save
    pub fn record_checkpoint(&self, model_name: String, epoch: usize, path: String) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        let event = TrainingEvent::CheckpointSaved {
            model_name,
            epoch,
            path,
            timestamp,
        };
        
        self.send_event(event)?;
        Ok(())
    }
    
    /// Get current training statistics
    pub fn get_stats(&self) -> TrainingStats {
        self.stats.lock().unwrap().clone()
    }
    
    /// Get metrics history for a model
    pub fn get_model_metrics(&self, model_name: &str) -> Option<VecDeque<GpuTrainingMetrics>> {
        self.model_metrics.lock().unwrap().get(model_name).cloned()
    }
    
    /// Generate training progress report
    pub fn generate_progress_report(&self) -> String {
        let stats = self.get_stats();
        let mut report = String::new();
        
        report.push_str("=== Training Progress Report ===\n\n");
        
        // Overall progress
        report.push_str(&format!(
            "Overall Progress: {:.1}% ({}/{} models completed)\n",
            stats.progress.overall_progress * 100.0,
            stats.progress.models_completed,
            stats.progress.total_models
        ));
        
        if let Some(current_model) = &stats.progress.current_model {
            report.push_str(&format!("Current Model: {}\n", current_model));
            if let (Some(current_epoch), Some(total_epochs)) = 
                (stats.progress.current_epoch, stats.progress.total_epochs) {
                report.push_str(&format!(
                    "Current Epoch: {}/{} ({:.1}%)\n",
                    current_epoch,
                    total_epochs,
                    (current_epoch as f64 / total_epochs as f64) * 100.0
                ));
            }
        }
        
        // Performance metrics
        report.push_str("\n=== Performance Metrics ===\n");
        report.push_str(&format!(
            "Average Throughput: {:.1} samples/sec\n",
            stats.performance.average_throughput
        ));
        report.push_str(&format!(
            "Peak Throughput: {:.1} samples/sec\n",
            stats.performance.peak_throughput
        ));
        report.push_str(&format!(
            "GPU Utilization: {:.1}% (Peak: {:.1}%)\n",
            stats.performance.average_gpu_utilization,
            stats.performance.peak_gpu_utilization
        ));
        report.push_str(&format!(
            "Total Training Time: {:.1} hours\n",
            stats.performance.total_training_time.as_secs_f64() / 3600.0
        ));
        
        // GPU stats
        report.push_str("\n=== GPU Statistics ===\n");
        report.push_str(&format!(
            "Memory Usage: {:.1} MB (Peak: {:.1} MB)\n",
            stats.gpu_stats.current_memory_usage,
            stats.gpu_stats.peak_memory_usage
        ));
        
        if let Some(temp) = stats.gpu_stats.temperature {
            report.push_str(&format!("Temperature: {:.1}°C\n", temp));
        }
        
        if let Some(power) = stats.gpu_stats.power_draw {
            report.push_str(&format!("Power Draw: {:.1}W\n", power));
        }
        
        // Model comparison
        if let Some(best_model) = &stats.model_comparison.best_model {
            report.push_str("\n=== Best Model ===\n");
            report.push_str(&format!(
                "Model: {} (Loss: {:.6})\n",
                best_model,
                stats.model_comparison.best_loss.unwrap_or(0.0)
            ));
        }
        
        // Recent alerts
        if !stats.recent_alerts.is_empty() {
            report.push_str("\n=== Recent Alerts ===\n");
            for alert in stats.recent_alerts.iter().take(5) {
                if let TrainingEvent::Alert { level, message, .. } = alert {
                    report.push_str(&format!("{:?}: {}\n", level, message));
                }
            }
        }
        
        report
    }
    
    /// Generate performance visualization data
    pub fn generate_visualization_data(&self) -> Result<VisualizationData> {
        let stats = self.get_stats();
        let model_metrics = self.model_metrics.lock().unwrap();
        
        let mut training_curves = HashMap::new();
        let mut throughput_curves = HashMap::new();
        let mut memory_curves = HashMap::new();
        
        for (model_name, metrics) in model_metrics.iter() {
            let epochs: Vec<usize> = metrics.iter().map(|m| m.epoch).collect();
            let losses: Vec<f32> = metrics.iter().map(|m| m.training_loss).collect();
            let throughputs: Vec<f32> = metrics.iter().map(|m| m.throughput).collect();
            let memory_usage: Vec<f64> = metrics.iter().map(|m| m.gpu_memory_used).collect();
            
            training_curves.insert(model_name.clone(), (epochs.clone(), losses));
            throughput_curves.insert(model_name.clone(), (epochs.clone(), throughputs));
            memory_curves.insert(model_name.clone(), (epochs, memory_usage));
        }
        
        Ok(VisualizationData {
            training_curves,
            throughput_curves,
            memory_curves,
            gpu_utilization_history: stats.gpu_stats.memory_utilization_history,
            model_comparison: stats.model_comparison,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        })
    }
    
    /// Update internal statistics
    fn update_stats(&self, model_name: &str, metrics: &GpuTrainingMetrics) -> Result<()> {
        let mut stats = self.stats.lock().unwrap();
        
        // Update current epoch
        stats.progress.current_epoch = Some(metrics.epoch);
        
        // Update performance metrics
        if metrics.throughput > stats.performance.peak_throughput {
            stats.performance.peak_throughput = metrics.throughput;
        }
        
        // Update GPU stats
        stats.gpu_stats.current_memory_usage = metrics.gpu_memory_used;
        if metrics.gpu_memory_used > stats.gpu_stats.peak_memory_usage {
            stats.gpu_stats.peak_memory_usage = metrics.gpu_memory_used;
        }
        
        stats.gpu_stats.memory_utilization_history.push_back(metrics.gpu_memory_used);
        while stats.gpu_stats.memory_utilization_history.len() > self.config.history_window {
            stats.gpu_stats.memory_utilization_history.pop_front();
        }
        
        // Calculate running averages
        let model_metrics = self.model_metrics.lock().unwrap();
        if let Some(history) = model_metrics.get(model_name) {
            if !history.is_empty() {
                stats.performance.average_throughput = 
                    history.iter().map(|m| m.throughput).sum::<f32>() / history.len() as f32;
                
                stats.performance.average_gpu_utilization = 
                    history.iter().map(|m| m.gpu_memory_used).sum::<f64>() / history.len() as f64;
            }
        }
        
        Ok(())
    }
    
    /// Check for alert conditions
    fn check_alerts(&self, model_name: &str, metrics: &GpuTrainingMetrics) -> Result<()> {
        let thresholds = &self.config.alert_thresholds;
        
        // Check training loss
        if metrics.training_loss > thresholds.max_training_loss {
            self.send_alert(
                AlertLevel::Warning,
                format!("High training loss: {:.6}", metrics.training_loss),
                model_name.to_string(),
            )?;
        }
        
        // Check GPU memory
        if metrics.gpu_memory_used > thresholds.max_gpu_memory {
            self.send_alert(
                AlertLevel::Warning,
                format!("High GPU memory usage: {:.1} MB", metrics.gpu_memory_used),
                model_name.to_string(),
            )?;
        }
        
        // Check epoch time
        if metrics.epoch_time.as_secs_f64() > thresholds.max_epoch_time {
            self.send_alert(
                AlertLevel::Warning,
                format!("Slow epoch: {:.1} seconds", metrics.epoch_time.as_secs_f64()),
                model_name.to_string(),
            )?;
        }
        
        // Check throughput
        if metrics.throughput < thresholds.min_throughput {
            self.send_alert(
                AlertLevel::Warning,
                format!("Low throughput: {:.1} samples/sec", metrics.throughput),
                model_name.to_string(),
            )?;
        }
        
        // Check learning rate
        if metrics.learning_rate < thresholds.min_learning_rate {
            self.send_alert(
                AlertLevel::Info,
                format!("Very low learning rate: {:.2e}", metrics.learning_rate),
                model_name.to_string(),
            )?;
        }
        
        Ok(())
    }
    
    /// Send alert
    fn send_alert(&self, level: AlertLevel, message: String, model_name: String) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_secs();
        
        let event = TrainingEvent::Alert {
            level,
            message,
            model_name,
            timestamp,
        };
        
        self.send_event(event)?;
        Ok(())
    }
    
    /// Send training event
    fn send_event(&self, event: TrainingEvent) -> Result<()> {
        // Add to recent alerts if it's an alert
        if let TrainingEvent::Alert { .. } = &event {
            let mut stats = self.stats.lock().unwrap();
            stats.recent_alerts.push(event.clone());
            
            // Keep only recent alerts
            while stats.recent_alerts.len() > 20 {
                stats.recent_alerts.remove(0);
            }
        }
        
        // Broadcast event
        let _ = self.event_sender.send(event.clone());
        
        // Log to file if enabled
        if self.config.save_logs {
            let log_entry = serde_json::to_string(&event)?;
            log::debug!("Training event: {}", log_entry);
        }
        
        Ok(())
    }
    
    /// Subscribe to training events
    pub fn subscribe(&self) -> broadcast::Receiver<TrainingEvent> {
        self.event_sender.subscribe()
    }
}

/// Visualization data structure
#[derive(Debug, Clone, Serialize)]
pub struct VisualizationData {
    pub training_curves: HashMap<String, (Vec<usize>, Vec<f32>)>,
    pub throughput_curves: HashMap<String, (Vec<usize>, Vec<f32>)>,
    pub memory_curves: HashMap<String, (Vec<usize>, Vec<f64>)>,
    pub gpu_utilization_history: VecDeque<f64>,
    pub model_comparison: ModelComparisonStats,
    pub timestamp: u64,
}

/// Training monitor builder for easy configuration
pub struct TrainingMonitorBuilder {
    config: MonitorConfig,
}

impl TrainingMonitorBuilder {
    pub fn new() -> Self {
        Self {
            config: MonitorConfig::default(),
        }
    }
    
    pub fn update_frequency(mut self, freq: f64) -> Self {
        self.config.update_frequency = freq;
        self
    }
    
    pub fn history_window(mut self, size: usize) -> Self {
        self.config.history_window = size;
        self
    }
    
    pub fn enable_plotting(mut self, enable: bool) -> Self {
        self.config.enable_plotting = enable;
        self
    }
    
    pub fn log_file_path(mut self, path: String) -> Self {
        self.config.log_file_path = path;
        self
    }
    
    pub fn alert_thresholds(mut self, thresholds: AlertThresholds) -> Self {
        self.config.alert_thresholds = thresholds;
        self
    }
    
    pub fn build(self) -> Result<TrainingMonitor> {
        TrainingMonitor::new(self.config)
    }
}

impl Default for TrainingMonitorBuilder {
    fn default() -> Self {
        Self::new()
    }
}