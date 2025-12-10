//! Utility Functions and Helpers
//! 
//! This module contains various utility functions, data processing helpers,
//! and common functionality used throughout the swarm system.

use std::time::{Duration, Instant};
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

pub mod metrics;
pub mod data_processing;
pub mod validation;
pub mod visualization;

pub use metrics::MetricsCollector;
pub use data_processing::DataProcessor;
pub use validation::Validator;
pub use visualization::ResultVisualizer;

/// Timer utility for performance measurement
#[derive(Debug)]
pub struct Timer {
    start: Instant,
    name: String,
}

impl Timer {
    pub fn new(name: String) -> Self {
        println!("🕐 Starting timer: {}", name);
        Self {
            start: Instant::now(),
            name,
        }
    }
    
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
    
    pub fn lap(&self, checkpoint: &str) {
        let elapsed = self.elapsed();
        println!("⏱️  {} - {}: {:.3}s", self.name, checkpoint, elapsed.as_secs_f64());
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let elapsed = self.elapsed();
        println!("✅ Completed {}: {:.3}s", self.name, elapsed.as_secs_f64());
    }
}

/// Progress tracker for long-running operations
#[derive(Debug)]
pub struct ProgressTracker {
    total: usize,
    current: usize,
    start_time: Instant,
    last_update: Instant,
    update_interval: Duration,
}

impl ProgressTracker {
    pub fn new(total: usize) -> Self {
        Self {
            total,
            current: 0,
            start_time: Instant::now(),
            last_update: Instant::now(),
            update_interval: Duration::from_millis(500),
        }
    }
    
    pub fn update(&mut self, current: usize) {
        self.current = current;
        
        if self.last_update.elapsed() >= self.update_interval {
            self.print_progress();
            self.last_update = Instant::now();
        }
    }
    
    pub fn increment(&mut self) {
        self.update(self.current + 1);
    }
    
    fn print_progress(&self) {
        let percentage = (self.current as f64 / self.total as f64) * 100.0;
        let elapsed = self.start_time.elapsed();
        
        let eta = if self.current > 0 {
            let rate = self.current as f64 / elapsed.as_secs_f64();
            let remaining = (self.total - self.current) as f64 / rate;
            Duration::from_secs_f64(remaining)
        } else {
            Duration::from_secs(0)
        };
        
        println!("📊 Progress: {}/{} ({:.1}%) - Elapsed: {:.1}s - ETA: {:.1}s",
                 self.current, self.total, percentage,
                 elapsed.as_secs_f64(), eta.as_secs_f64());
    }
    
    pub fn finish(&self) {
        let elapsed = self.start_time.elapsed();
        println!("🎯 Completed: {}/{} in {:.2}s", self.total, self.total, elapsed.as_secs_f64());
    }
}

/// File utilities
pub struct FileUtils;

impl FileUtils {
    pub fn ensure_directory<P: AsRef<Path>>(path: P) -> Result<(), std::io::Error> {
        if !path.as_ref().exists() {
            fs::create_dir_all(path)?;
        }
        Ok(())
    }
    
    pub fn get_file_size<P: AsRef<Path>>(path: P) -> Result<u64, std::io::Error> {
        let metadata = fs::metadata(path)?;
        Ok(metadata.len())
    }
    
    pub fn backup_file<P: AsRef<Path>>(path: P) -> Result<(), std::io::Error> {
        let path = path.as_ref();
        if path.exists() {
            let backup_path = format!("{}.backup", path.display());
            fs::copy(path, backup_path)?;
        }
        Ok(())
    }
    
    pub fn cleanup_old_files<P: AsRef<Path>>(
        dir: P,
        pattern: &str,
        max_age_hours: u64,
    ) -> Result<usize, std::io::Error> {
        let dir = dir.as_ref();
        let mut count = 0;
        
        if !dir.is_dir() {
            return Ok(0);
        }
        
        let cutoff = std::time::SystemTime::now() - Duration::from_secs(max_age_hours * 3600);
        
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if let Some(filename) = path.file_name() {
                if filename.to_string_lossy().contains(pattern) {
                    if let Ok(metadata) = entry.metadata() {
                        if let Ok(modified) = metadata.modified() {
                            if modified < cutoff {
                                if fs::remove_file(&path).is_ok() {
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(count)
    }
}

/// Statistics utilities
pub struct StatUtils;

impl StatUtils {
    pub fn mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        values.iter().sum::<f32>() / values.len() as f32
    }
    
    pub fn median(values: &mut [f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = values.len();
        
        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        }
    }
    
    pub fn std_dev(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }
        
        let mean = Self::mean(values);
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;
        
        variance.sqrt()
    }
    
    pub fn percentile(values: &mut [f32], p: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }
        
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (p * (values.len() - 1) as f32).round() as usize;
        values[index.min(values.len() - 1)]
    }
    
    pub fn quartiles(values: &mut [f32]) -> (f32, f32, f32) {
        (
            Self::percentile(values, 0.25),
            Self::percentile(values, 0.5),
            Self::percentile(values, 0.75),
        )
    }
}

/// Mathematical utilities
pub struct MathUtils;

impl MathUtils {
    pub fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    
    pub fn relu(x: f32) -> f32 {
        x.max(0.0)
    }
    
    pub fn tanh(x: f32) -> f32 {
        x.tanh()
    }
    
    pub fn leaky_relu(x: f32, alpha: f32) -> f32 {
        if x > 0.0 { x } else { alpha * x }
    }
    
    pub fn swish(x: f32) -> f32 {
        x * Self::sigmoid(x)
    }
    
    pub fn normalize(values: &mut [f32]) {
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;
        
        if range > 0.0 {
            for value in values {
                *value = (*value - min_val) / range;
            }
        }
    }
    
    pub fn standardize(values: &mut [f32]) {
        let mean = StatUtils::mean(values);
        let std_dev = StatUtils::std_dev(values);
        
        if std_dev > 0.0 {
            for value in values {
                *value = (*value - mean) / std_dev;
            }
        }
    }
    
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }
    
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
        
        if norm_a > 0.0 && norm_b > 0.0 {
            dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Random utilities
pub struct RandomUtils;

impl RandomUtils {
    pub fn random_normal(mean: f32, std_dev: f32) -> f32 {
        use rand::Rng;
        use std::f32::consts::PI;
        
        let mut rng = rand::thread_rng();
        let u1: f32 = rng.gen();
        let u2: f32 = rng.gen();
        
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        mean + std_dev * z0
    }
    
    pub fn random_choice<T: Clone>(items: &[T]) -> Option<T> {
        if items.is_empty() {
            return None;
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let index = rng.gen_range(0..items.len());
        Some(items[index].clone())
    }
    
    pub fn random_sample<T: Clone>(items: &[T], n: usize) -> Vec<T> {
        if n >= items.len() {
            return items.to_vec();
        }
        
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        items.choose_multiple(&mut rng, n).cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timer() {
        let timer = Timer::new("test".to_string());
        std::thread::sleep(Duration::from_millis(10));
        assert!(timer.elapsed() >= Duration::from_millis(10));
    }
    
    #[test]
    fn test_stats() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(StatUtils::mean(&values), 3.0);
        
        let mut values_copy = values.clone();
        assert_eq!(StatUtils::median(&mut values_copy), 3.0);
        
        assert!(StatUtils::std_dev(&values) > 0.0);
    }
    
    #[test]
    fn test_math_utils() {
        assert!(MathUtils::sigmoid(0.0) == 0.5);
        assert!(MathUtils::relu(-1.0) == 0.0);
        assert!(MathUtils::relu(1.0) == 1.0);
        
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert!((MathUtils::cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }
}