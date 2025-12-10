//! Metrics and monitoring for the GenAI service

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use prometheus::{
    Counter, Histogram, Gauge, Registry, CounterVec, HistogramVec, GaugeVec,
    register_counter_vec_with_registry, register_histogram_vec_with_registry,
    register_gauge_vec_with_registry, Opts, HistogramOpts,
};

/// Metrics collector for the GenAI service
pub struct Metrics {
    /// Prometheus registry
    registry: Registry,
    
    /// Request counters
    request_counter: CounterVec,
    
    /// Response time histograms
    response_time_histogram: HistogramVec,
    
    /// Token usage histograms
    token_usage_histogram: HistogramVec,
    
    /// Cache hit/miss counters
    cache_counter: CounterVec,
    
    /// Active connections gauge
    active_connections_gauge: GaugeVec,
    
    /// Queue size gauge
    queue_size_gauge: GaugeVec,
    
    /// Internal metrics storage
    internal_metrics: Arc<RwLock<InternalMetrics>>,
}

/// Internal metrics storage for detailed tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InternalMetrics {
    /// Total number of requests per backend
    pub requests_by_backend: HashMap<String, u64>,
    
    /// Total response time per backend
    pub total_response_time_by_backend: HashMap<String, Duration>,
    
    /// Total tokens used per backend
    pub total_tokens_by_backend: HashMap<String, u64>,
    
    /// Cache statistics
    pub cache_hits: u64,
    pub cache_misses: u64,
    
    /// Error counts by type
    pub errors_by_type: HashMap<String, u64>,
    
    /// Batch processing statistics
    pub batch_requests: u64,
    pub batch_total_size: u64,
    
    /// Start time for uptime calculation
    pub start_time: Instant,
}

impl Default for InternalMetrics {
    fn default() -> Self {
        Self {
            requests_by_backend: HashMap::new(),
            total_response_time_by_backend: HashMap::new(),
            total_tokens_by_backend: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            errors_by_type: HashMap::new(),
            batch_requests: 0,
            batch_total_size: 0,
            start_time: Instant::now(),
        }
    }
}

impl Metrics {
    pub fn new() -> Self {
        let registry = Registry::new();
        
        // Initialize Prometheus metrics
        let request_counter = register_counter_vec_with_registry!(
            Opts::new("genai_requests_total", "Total number of requests"),
            &["backend", "status"],
            registry
        ).expect("Failed to register request counter");
        
        let response_time_histogram = register_histogram_vec_with_registry!(
            HistogramOpts::new("genai_response_time_seconds", "Response time in seconds")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]),
            &["backend"],
            registry
        ).expect("Failed to register response time histogram");
        
        let token_usage_histogram = register_histogram_vec_with_registry!(
            HistogramOpts::new("genai_token_usage", "Token usage per request")
                .buckets(vec![10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]),
            &["backend", "type"],
            registry
        ).expect("Failed to register token usage histogram");
        
        let cache_counter = register_counter_vec_with_registry!(
            Opts::new("genai_cache_total", "Cache hits and misses"),
            &["result"],
            registry
        ).expect("Failed to register cache counter");
        
        let active_connections_gauge = register_gauge_vec_with_registry!(
            Opts::new("genai_active_connections", "Number of active connections"),
            &["backend"],
            registry
        ).expect("Failed to register active connections gauge");
        
        let queue_size_gauge = register_gauge_vec_with_registry!(
            Opts::new("genai_queue_size", "Size of request queues"),
            &["backend"],
            registry
        ).expect("Failed to register queue size gauge");
        
        Self {
            registry,
            request_counter,
            response_time_histogram,
            token_usage_histogram,
            cache_counter,
            active_connections_gauge,
            queue_size_gauge,
            internal_metrics: Arc::new(RwLock::new(InternalMetrics::default())),
        }
    }
    
    /// Record a request
    pub fn record_request(&self, backend: &str, latency_ms: u64, tokens: usize) {
        // Update Prometheus metrics
        self.request_counter
            .with_label_values(&[backend, "success"])
            .inc();
        
        self.response_time_histogram
            .with_label_values(&[backend])
            .observe(latency_ms as f64 / 1000.0);
        
        self.token_usage_histogram
            .with_label_values(&[backend, "total"])
            .observe(tokens as f64);
        
        // Update internal metrics
        let internal = self.internal_metrics.clone();
        tokio::spawn(async move {
            let mut metrics = internal.write().await;
            
            *metrics.requests_by_backend.entry(backend.to_string()).or_insert(0) += 1;
            
            let duration = Duration::from_millis(latency_ms);
            *metrics.total_response_time_by_backend.entry(backend.to_string()).or_insert(Duration::ZERO) += duration;
            
            *metrics.total_tokens_by_backend.entry(backend.to_string()).or_insert(0) += tokens as u64;
        });
    }
    
    /// Record a failed request
    pub fn record_error(&self, backend: &str, error_type: &str) {
        self.request_counter
            .with_label_values(&[backend, "error"])
            .inc();
        
        let internal = self.internal_metrics.clone();
        let error_type = error_type.to_string();
        tokio::spawn(async move {
            let mut metrics = internal.write().await;
            *metrics.errors_by_type.entry(error_type).or_insert(0) += 1;
        });
    }
    
    /// Record a cache hit
    pub fn record_cache_hit(&self) {
        self.cache_counter
            .with_label_values(&["hit"])
            .inc();
        
        let internal = self.internal_metrics.clone();
        tokio::spawn(async move {
            let mut metrics = internal.write().await;
            metrics.cache_hits += 1;
        });
    }
    
    /// Record a cache miss
    pub fn record_cache_miss(&self) {
        self.cache_counter
            .with_label_values(&["miss"])
            .inc();
        
        let internal = self.internal_metrics.clone();
        tokio::spawn(async move {
            let mut metrics = internal.write().await;
            metrics.cache_misses += 1;
        });
    }
    
    /// Update active connections gauge
    pub fn set_active_connections(&self, backend: &str, count: i64) {
        self.active_connections_gauge
            .with_label_values(&[backend])
            .set(count as f64);
    }
    
    /// Update queue size gauge
    pub fn set_queue_size(&self, backend: &str, size: usize) {
        self.queue_size_gauge
            .with_label_values(&[backend])
            .set(size as f64);
    }
    
    /// Record batch processing
    pub fn record_batch(&self, backend: &str, batch_size: usize, total_tokens: usize) {
        // Record each request in the batch
        for _ in 0..batch_size {
            self.request_counter
                .with_label_values(&[backend, "batch"])
                .inc();
        }
        
        self.token_usage_histogram
            .with_label_values(&[backend, "batch"])
            .observe(total_tokens as f64);
        
        let internal = self.internal_metrics.clone();
        tokio::spawn(async move {
            let mut metrics = internal.write().await;
            metrics.batch_requests += 1;
            metrics.batch_total_size += batch_size as u64;
        });
    }
    
    /// Get detailed metrics summary
    pub async fn get_summary(&self) -> MetricsSummary {
        let internal = self.internal_metrics.read().await;
        
        // Calculate derived metrics
        let mut backend_summaries = HashMap::new();
        
        for (backend, request_count) in &internal.requests_by_backend {
            let total_response_time = internal.total_response_time_by_backend
                .get(backend)
                .unwrap_or(&Duration::ZERO);
            
            let avg_response_time = if *request_count > 0 {
                total_response_time.as_millis() as f64 / *request_count as f64
            } else {
                0.0
            };
            
            let total_tokens = internal.total_tokens_by_backend
                .get(backend)
                .unwrap_or(&0);
            
            let avg_tokens_per_request = if *request_count > 0 {
                *total_tokens as f64 / *request_count as f64
            } else {
                0.0
            };
            
            backend_summaries.insert(backend.clone(), BackendSummary {
                total_requests: *request_count,
                avg_response_time_ms: avg_response_time,
                total_tokens: *total_tokens,
                avg_tokens_per_request,
            });
        }
        
        let cache_hit_rate = if internal.cache_hits + internal.cache_misses > 0 {
            internal.cache_hits as f64 / (internal.cache_hits + internal.cache_misses) as f64
        } else {
            0.0
        };
        
        let avg_batch_size = if internal.batch_requests > 0 {
            internal.batch_total_size as f64 / internal.batch_requests as f64
        } else {
            0.0
        };
        
        MetricsSummary {
            uptime_seconds: internal.start_time.elapsed().as_secs(),
            backend_summaries,
            cache_hit_rate,
            cache_hits: internal.cache_hits,
            cache_misses: internal.cache_misses,
            total_errors: internal.errors_by_type.values().sum(),
            errors_by_type: internal.errors_by_type.clone(),
            batch_requests: internal.batch_requests,
            avg_batch_size,
        }
    }
    
    /// Get Prometheus metrics as text
    pub fn get_prometheus_metrics(&self) -> String {
        use prometheus::Encoder;
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode_to_string(&metric_families).unwrap_or_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub uptime_seconds: u64,
    pub backend_summaries: HashMap<String, BackendSummary>,
    pub cache_hit_rate: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub total_errors: u64,
    pub errors_by_type: HashMap<String, u64>,
    pub batch_requests: u64,
    pub avg_batch_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendSummary {
    pub total_requests: u64,
    pub avg_response_time_ms: f64,
    pub total_tokens: u64,
    pub avg_tokens_per_request: f64,
}

/// Health check status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: String,
    pub version: String,
    pub uptime_seconds: u64,
    pub active_backends: Vec<String>,
    pub cache_enabled: bool,
    pub batch_enabled: bool,
}

/// Health checker for the GenAI service
pub struct HealthChecker {
    start_time: Instant,
    version: String,
}

impl HealthChecker {
    pub fn new(version: String) -> Self {
        Self {
            start_time: Instant::now(),
            version,
        }
    }
    
    pub fn check_health(&self, active_backends: Vec<String>, cache_enabled: bool, batch_enabled: bool) -> HealthStatus {
        HealthStatus {
            status: "healthy".to_string(),
            version: self.version.clone(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            active_backends,
            cache_enabled,
            batch_enabled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metrics_creation() {
        let metrics = Metrics::new();
        
        // Test that metrics can be recorded
        metrics.record_request("test-backend", 100, 50);
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        metrics.record_error("test-backend", "timeout");
    }
    
    #[tokio::test]
    async fn test_metrics_summary() {
        let metrics = Metrics::new();
        
        // Record some metrics
        metrics.record_request("backend1", 100, 50);
        metrics.record_request("backend1", 200, 75);
        metrics.record_request("backend2", 150, 60);
        metrics.record_cache_hit();
        metrics.record_cache_miss();
        
        // Wait for async updates to complete
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let summary = metrics.get_summary().await;
        
        assert_eq!(summary.cache_hits, 1);
        assert_eq!(summary.cache_misses, 1);
        assert_eq!(summary.cache_hit_rate, 0.5);
        
        assert!(summary.backend_summaries.contains_key("backend1"));
        assert!(summary.backend_summaries.contains_key("backend2"));
        
        let backend1_summary = &summary.backend_summaries["backend1"];
        assert_eq!(backend1_summary.total_requests, 2);
        assert_eq!(backend1_summary.avg_response_time_ms, 150.0);
        assert_eq!(backend1_summary.total_tokens, 125);
        assert_eq!(backend1_summary.avg_tokens_per_request, 62.5);
    }
    
    #[test]
    fn test_health_checker() {
        let checker = HealthChecker::new("1.0.0".to_string());
        
        let health = checker.check_health(
            vec!["openai".to_string(), "local".to_string()],
            true,
            true
        );
        
        assert_eq!(health.status, "healthy");
        assert_eq!(health.version, "1.0.0");
        assert_eq!(health.active_backends.len(), 2);
        assert!(health.cache_enabled);
        assert!(health.batch_enabled);
    }
    
    #[test]
    fn test_prometheus_metrics() {
        let metrics = Metrics::new();
        
        // Record some metrics
        metrics.record_request("test", 100, 50);
        metrics.record_cache_hit();
        
        // Should be able to get Prometheus format
        let prometheus_text = metrics.get_prometheus_metrics();
        assert!(!prometheus_text.is_empty());
    }
}