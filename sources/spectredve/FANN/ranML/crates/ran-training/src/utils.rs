//! Utility functions and helpers for the training system

use crate::error::{TrainingError, TrainingResult};
use std::path::Path;
use std::time::{Duration, Instant};

/// Performance timer for measuring execution time
pub struct Timer {
    start: Instant,
    label: String,
}

impl Timer {
    /// Start a new timer with a label
    pub fn new(label: &str) -> Self {
        Self {
            start: Instant::now(),
            label: label.to_string(),
        }
    }

    /// Get elapsed time without stopping the timer
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Stop the timer and return elapsed time
    pub fn stop(self) -> Duration {
        let elapsed = self.elapsed();
        log::debug!("Timer '{}' elapsed: {:?}", self.label, elapsed);
        elapsed
    }

    /// Stop the timer and log the result
    pub fn stop_and_log(self) {
        let label = self.label.clone();
        let elapsed = self.stop();
        log::info!("{}: {:?}", label, elapsed);
    }
}

/// Memory usage tracker
pub struct MemoryTracker {
    initial_usage: usize,
    label: String,
}

impl MemoryTracker {
    /// Start tracking memory usage
    pub fn new(label: &str) -> Self {
        Self {
            initial_usage: Self::get_memory_usage(),
            label: label.to_string(),
        }
    }

    /// Get current memory usage in bytes (simplified implementation)
    fn get_memory_usage() -> usize {
        // This is a simplified implementation
        // In a real scenario, you'd use a proper memory tracking library
        0
    }

    /// Get memory usage delta since creation
    pub fn delta(&self) -> isize {
        Self::get_memory_usage() as isize - self.initial_usage as isize
    }

    /// Stop tracking and log memory usage
    pub fn stop_and_log(self) {
        let delta = self.delta();
        if delta > 0 {
            log::info!("{}: +{} bytes", self.label, delta);
        } else {
            log::info!("{}: {} bytes", self.label, delta);
        }
    }
}

/// Progress reporter for long-running operations
pub struct ProgressReporter {
    total: usize,
    current: usize,
    start_time: Instant,
    last_report: Instant,
    report_interval: Duration,
    label: String,
}

impl ProgressReporter {
    /// Create a new progress reporter
    pub fn new(label: &str, total: usize) -> Self {
        Self {
            total,
            current: 0,
            start_time: Instant::now(),
            last_report: Instant::now(),
            report_interval: Duration::from_secs(10),
            label: label.to_string(),
        }
    }

    /// Set reporting interval
    pub fn with_interval(mut self, interval: Duration) -> Self {
        self.report_interval = interval;
        self
    }

    /// Update progress
    pub fn update(&mut self, current: usize) {
        self.current = current;
        
        if self.last_report.elapsed() >= self.report_interval {
            self.report();
            self.last_report = Instant::now();
        }
    }

    /// Increment progress by 1
    pub fn increment(&mut self) {
        self.update(self.current + 1);
    }

    /// Force a progress report
    pub fn report(&self) {
        let percentage = if self.total > 0 {
            (self.current as f64 / self.total as f64) * 100.0
        } else {
            0.0
        };

        let elapsed = self.start_time.elapsed();
        let eta = if self.current > 0 {
            let time_per_item = elapsed.as_secs_f64() / self.current as f64;
            let remaining_items = self.total.saturating_sub(self.current);
            Duration::from_secs_f64(time_per_item * remaining_items as f64)
        } else {
            Duration::from_secs(0)
        };

        log::info!(
            "{}: {}/{} ({:.1}%) - Elapsed: {:?}, ETA: {:?}",
            self.label,
            self.current,
            self.total,
            percentage,
            elapsed,
            eta
        );
    }

    /// Finish and report final statistics
    pub fn finish(self) {
        log::info!(
            "{}: Completed {}/{} items in {:?}",
            self.label,
            self.current,
            self.total,
            self.start_time.elapsed()
        );
    }
}

/// File system utilities
pub struct FileUtils;

impl FileUtils {
    /// Ensure directory exists, create if it doesn't
    pub fn ensure_dir<P: AsRef<Path>>(path: P) -> TrainingResult<()> {
        std::fs::create_dir_all(path)?;
        Ok(())
    }

    /// Check if file exists and is readable
    pub fn is_readable<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists() && path.as_ref().is_file()
    }

    /// Get file size in bytes
    pub fn file_size<P: AsRef<Path>>(path: P) -> TrainingResult<u64> {
        let metadata = std::fs::metadata(path)?;
        Ok(metadata.len())
    }

    /// Copy file with progress reporting
    pub fn copy_with_progress<P: AsRef<Path>>(
        from: P,
        to: P,
        progress_callback: Option<Box<dyn Fn(u64, u64)>>,
    ) -> TrainingResult<()> {
        use std::io::{Read, Write};

        let mut source = std::fs::File::open(&from)?;
        let mut dest = std::fs::File::create(&to)?;

        let total_size = Self::file_size(&from)?;
        let mut copied = 0u64;
        let mut buffer = vec![0u8; 8192]; // 8KB buffer

        loop {
            let bytes_read = source.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            dest.write_all(&buffer[..bytes_read])?;
            copied += bytes_read as u64;

            if let Some(ref callback) = progress_callback {
                callback(copied, total_size);
            }
        }

        Ok(())
    }

    /// Clean up old files in directory
    pub fn cleanup_old_files<P: AsRef<Path>>(
        dir: P,
        max_age: Duration,
        pattern: Option<&str>,
    ) -> TrainingResult<usize> {
        let dir_path = dir.as_ref();
        if !dir_path.exists() {
            return Ok(0);
        }

        let cutoff_time = std::time::SystemTime::now() - max_age;
        let mut removed_count = 0;

        for entry in std::fs::read_dir(dir_path)? {
            let entry = entry?;
            let metadata = entry.metadata()?;

            // Check if file matches pattern (if provided)
            if let Some(pattern) = pattern {
                if let Some(filename) = entry.file_name().to_str() {
                    if !filename.contains(pattern) {
                        continue;
                    }
                }
            }

            // Check if file is older than cutoff
            if let Ok(modified) = metadata.modified() {
                if modified < cutoff_time {
                    if std::fs::remove_file(entry.path()).is_ok() {
                        removed_count += 1;
                    }
                }
            }
        }

        Ok(removed_count)
    }
}

/// Mathematical utilities
pub struct MathUtils;

impl MathUtils {
    /// Calculate moving average
    pub fn moving_average(values: &[f32], window_size: usize) -> Vec<f32> {
        if window_size == 0 || values.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        for i in 0..values.len() {
            let start = if i >= window_size - 1 { i - window_size + 1 } else { 0 };
            let window = &values[start..=i];
            let avg = window.iter().sum::<f32>() / window.len() as f32;
            result.push(avg);
        }
        result
    }

    /// Calculate exponential moving average
    pub fn exponential_moving_average(values: &[f32], alpha: f32) -> Vec<f32> {
        if values.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(values.len());
        result.push(values[0]);

        for i in 1..values.len() {
            let ema = alpha * values[i] + (1.0 - alpha) * result[i - 1];
            result.push(ema);
        }

        result
    }

    /// Calculate percentile
    pub fn percentile(values: &[f32], percentile: f32) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (percentile / 100.0 * (sorted.len() - 1) as f32).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    /// Calculate correlation coefficient
    pub fn correlation(x: &[f32], y: &[f32]) -> f32 {
        if x.len() != y.len() || x.is_empty() {
            return 0.0;
        }

        let n = x.len() as f32;
        let sum_x: f32 = x.iter().sum();
        let sum_y: f32 = y.iter().sum();
        let sum_xy: f32 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f32 = x.iter().map(|a| a * a).sum();
        let sum_y2: f32 = y.iter().map(|b| b * b).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Calculate standard deviation
    pub fn std_dev(values: &[f32]) -> f32 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / (values.len() - 1) as f32;

        variance.sqrt()
    }

    /// Normalize vector to unit length
    pub fn normalize(values: &mut [f32]) {
        let magnitude: f32 = values.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for value in values.iter_mut() {
                *value /= magnitude;
            }
        }
    }

    /// Calculate cosine similarity between two vectors
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            0.0
        } else {
            dot_product / (magnitude_a * magnitude_b)
        }
    }
}

/// String utilities
pub struct StringUtils;

impl StringUtils {
    /// Truncate string to specified length with ellipsis
    pub fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else if max_len <= 3 {
            "...".to_string()
        } else {
            format!("{}...", &s[..max_len - 3])
        }
    }

    /// Convert duration to human-readable string
    pub fn format_duration(duration: Duration) -> String {
        let total_secs = duration.as_secs();
        let hours = total_secs / 3600;
        let minutes = (total_secs % 3600) / 60;
        let seconds = total_secs % 60;

        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}s", seconds)
        }
    }

    /// Format file size in human-readable format
    pub fn format_file_size(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.1} {}", size, UNITS[unit_index])
        }
    }

    /// Generate random string of specified length
    pub fn random_string(length: usize) -> String {
        use rand::Rng;
        const CHARSET: &[u8] = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
        let mut rng = rand::thread_rng();
        
        (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..CHARSET.len());
                CHARSET[idx] as char
            })
            .collect()
    }
}

/// Validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate email address (simplified)
    pub fn is_valid_email(email: &str) -> bool {
        email.contains('@') && email.contains('.') && email.len() > 5
    }

    /// Validate URL (simplified)
    pub fn is_valid_url(url: &str) -> bool {
        url.starts_with("http://") || url.starts_with("https://")
    }

    /// Validate file path exists and is readable
    pub fn is_valid_file_path<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists() && path.as_ref().is_file()
    }

    /// Validate directory path exists
    pub fn is_valid_directory<P: AsRef<Path>>(path: P) -> bool {
        path.as_ref().exists() && path.as_ref().is_dir()
    }

    /// Validate numeric range
    pub fn is_in_range(value: f32, min: f32, max: f32) -> bool {
        value >= min && value <= max
    }

    /// Validate that vector contains no NaN or infinite values
    pub fn is_valid_vector(values: &[f32]) -> bool {
        values.iter().all(|x| x.is_finite())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timer() {
        let timer = Timer::new("test");
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = timer.stop();
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn test_progress_reporter() {
        let mut reporter = ProgressReporter::new("test", 100);
        reporter.update(50);
        assert_eq!(reporter.current, 50);
        reporter.increment();
        assert_eq!(reporter.current, 51);
    }

    #[test]
    fn test_math_utils() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let ma = MathUtils::moving_average(&values, 3);
        assert_eq!(ma.len(), 5);
        assert_eq!(ma[2], 2.0); // (1+2+3)/3
        
        let ema = MathUtils::exponential_moving_average(&values, 0.5);
        assert_eq!(ema.len(), 5);
        assert_eq!(ema[0], 1.0);
        
        let p90 = MathUtils::percentile(&values, 90.0);
        assert_eq!(p90, 5.0);
        
        let corr = MathUtils::correlation(&values, &values);
        assert!((corr - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_string_utils() {
        assert_eq!(StringUtils::truncate("hello world", 5), "he...");
        assert_eq!(StringUtils::truncate("hi", 10), "hi");
        
        let duration = Duration::from_secs(3665); // 1h 1m 5s
        let formatted = StringUtils::format_duration(duration);
        assert_eq!(formatted, "1h 1m 5s");
        
        let size = StringUtils::format_file_size(1536); // 1.5 KB
        assert_eq!(size, "1.5 KB");
        
        let random = StringUtils::random_string(10);
        assert_eq!(random.len(), 10);
    }

    #[test]
    fn test_validation_utils() {
        assert!(ValidationUtils::is_valid_email("test@example.com"));
        assert!(!ValidationUtils::is_valid_email("invalid"));
        
        assert!(ValidationUtils::is_valid_url("https://example.com"));
        assert!(!ValidationUtils::is_valid_url("not-a-url"));
        
        assert!(ValidationUtils::is_in_range(5.0, 0.0, 10.0));
        assert!(!ValidationUtils::is_in_range(15.0, 0.0, 10.0));
        
        assert!(ValidationUtils::is_valid_vector(&[1.0, 2.0, 3.0]));
        assert!(!ValidationUtils::is_valid_vector(&[1.0, f32::NAN, 3.0]));
    }
}