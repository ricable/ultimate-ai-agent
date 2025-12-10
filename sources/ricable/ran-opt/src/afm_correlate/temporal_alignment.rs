use chrono::{DateTime, Utc, Duration};
use std::collections::HashMap;

use super::EvidenceItem;

/// Temporal alignment algorithm for evidence correlation
pub struct TemporalAlignmentAlgorithm {
    temporal_window: Duration,
    alignment_threshold: f32,
    max_time_shift: Duration,
}

impl TemporalAlignmentAlgorithm {
    pub fn new(temporal_window: Duration) -> Self {
        Self {
            temporal_window,
            alignment_threshold: 0.8,
            max_time_shift: Duration::minutes(5),
        }
    }

    /// Align evidence items temporally
    pub fn align(&self, evidence_items: &[EvidenceItem]) -> Vec<EvidenceItem> {
        if evidence_items.is_empty() {
            return Vec::new();
        }

        // Sort by timestamp
        let mut sorted_items: Vec<EvidenceItem> = evidence_items.to_vec();
        sorted_items.sort_by_key(|item| item.timestamp);

        // Detect temporal patterns
        let temporal_patterns = self.detect_temporal_patterns(&sorted_items);
        
        // Align items based on patterns
        let aligned_items = self.align_with_patterns(&sorted_items, &temporal_patterns);
        
        // Apply dynamic time warping for fine-grained alignment
        self.dynamic_time_warping(&aligned_items)
    }

    /// Detect temporal patterns in evidence sequence
    fn detect_temporal_patterns(&self, evidence_items: &[EvidenceItem]) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        // Group by source type
        let source_groups = self.group_by_source(evidence_items);
        
        // Detect patterns within each source
        for (source, items) in source_groups {
            let source_patterns = self.detect_source_patterns(&items);
            patterns.extend(source_patterns);
        }
        
        // Detect cross-source patterns
        let cross_patterns = self.detect_cross_source_patterns(evidence_items);
        patterns.extend(cross_patterns);
        
        patterns
    }

    /// Group evidence items by source
    fn group_by_source(&self, evidence_items: &[EvidenceItem]) -> HashMap<super::EvidenceSource, Vec<EvidenceItem>> {
        let mut groups = HashMap::new();
        
        for item in evidence_items {
            groups.entry(item.source).or_insert_with(Vec::new).push(item.clone());
        }
        
        groups
    }

    /// Detect patterns within a single source
    fn detect_source_patterns(&self, items: &[EvidenceItem]) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        if items.len() < 2 {
            return patterns;
        }
        
        // Detect periodic patterns
        let periodic_patterns = self.detect_periodic_patterns(items);
        patterns.extend(periodic_patterns);
        
        // Detect burst patterns
        let burst_patterns = self.detect_burst_patterns(items);
        patterns.extend(burst_patterns);
        
        // Detect cascade patterns
        let cascade_patterns = self.detect_cascade_patterns(items);
        patterns.extend(cascade_patterns);
        
        patterns
    }

    /// Detect periodic patterns
    fn detect_periodic_patterns(&self, items: &[EvidenceItem]) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        if items.len() < 3 {
            return patterns;
        }
        
        // Calculate intervals between consecutive items
        let intervals: Vec<Duration> = items.windows(2)
            .map(|window| window[1].timestamp - window[0].timestamp)
            .collect();
        
        // Find repeating intervals
        let mut interval_counts = HashMap::new();
        for interval in &intervals {
            *interval_counts.entry(interval.num_seconds()).or_insert(0) += 1;
        }
        
        // Identify patterns with sufficient repetition
        for (interval_secs, count) in interval_counts {
            if count >= 2 {
                let interval = Duration::seconds(interval_secs);
                patterns.push(TemporalPattern {
                    pattern_type: PatternType::Periodic,
                    interval,
                    confidence: count as f32 / intervals.len() as f32,
                    items: self.extract_periodic_items(items, interval),
                });
            }
        }
        
        patterns
    }

    /// Extract items that follow periodic pattern
    fn extract_periodic_items(&self, items: &[EvidenceItem], interval: Duration) -> Vec<String> {
        let mut pattern_items = Vec::new();
        
        if items.is_empty() {
            return pattern_items;
        }
        
        let base_time = items[0].timestamp;
        let tolerance = Duration::seconds(30); // 30 second tolerance
        
        for item in items {
            let time_diff = item.timestamp - base_time;
            let interval_count = time_diff.num_seconds() / interval.num_seconds();
            let expected_time = base_time + interval * interval_count as i32;
            
            if (item.timestamp - expected_time).abs() <= tolerance {
                pattern_items.push(item.id.clone());
            }
        }
        
        pattern_items
    }

    /// Detect burst patterns
    fn detect_burst_patterns(&self, items: &[EvidenceItem]) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        if items.len() < 3 {
            return patterns;
        }
        
        let burst_threshold = Duration::seconds(60); // 1 minute burst window
        let min_burst_items = 3;
        
        let mut current_burst = Vec::new();
        let mut burst_start = items[0].timestamp;
        
        for item in items {
            if item.timestamp - burst_start <= burst_threshold {
                current_burst.push(item.id.clone());
            } else {
                if current_burst.len() >= min_burst_items {
                    patterns.push(TemporalPattern {
                        pattern_type: PatternType::Burst,
                        interval: burst_threshold,
                        confidence: current_burst.len() as f32 / items.len() as f32,
                        items: current_burst.clone(),
                    });
                }
                current_burst.clear();
                current_burst.push(item.id.clone());
                burst_start = item.timestamp;
            }
        }
        
        // Check final burst
        if current_burst.len() >= min_burst_items {
            patterns.push(TemporalPattern {
                pattern_type: PatternType::Burst,
                interval: burst_threshold,
                confidence: current_burst.len() as f32 / items.len() as f32,
                items: current_burst,
            });
        }
        
        patterns
    }

    /// Detect cascade patterns
    fn detect_cascade_patterns(&self, items: &[EvidenceItem]) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        if items.len() < 2 {
            return patterns;
        }
        
        // Look for increasing severity over time
        let mut cascade_items = Vec::new();
        let mut last_severity = 0.0;
        
        for item in items {
            if item.severity > last_severity {
                cascade_items.push(item.id.clone());
                last_severity = item.severity;
            }
        }
        
        if cascade_items.len() >= 2 {
            patterns.push(TemporalPattern {
                pattern_type: PatternType::Cascade,
                interval: Duration::minutes(10),
                confidence: cascade_items.len() as f32 / items.len() as f32,
                items: cascade_items,
            });
        }
        
        patterns
    }

    /// Detect cross-source patterns
    fn detect_cross_source_patterns(&self, evidence_items: &[EvidenceItem]) -> Vec<TemporalPattern> {
        let mut patterns = Vec::new();
        
        // Group by time windows
        let time_windows = self.create_time_windows(evidence_items);
        
        // Look for cross-source correlations within windows
        for (window_start, window_items) in time_windows {
            let cross_pattern = self.analyze_cross_source_window(&window_items);
            if let Some(pattern) = cross_pattern {
                patterns.push(pattern);
            }
        }
        
        patterns
    }

    /// Create time windows for analysis
    fn create_time_windows(&self, evidence_items: &[EvidenceItem]) -> HashMap<DateTime<Utc>, Vec<EvidenceItem>> {
        let mut windows = HashMap::new();
        let window_size = Duration::minutes(5);
        
        for item in evidence_items {
            let window_start = self.get_window_start(item.timestamp, window_size);
            windows.entry(window_start).or_insert_with(Vec::new).push(item.clone());
        }
        
        windows
    }

    /// Get window start time
    fn get_window_start(&self, timestamp: DateTime<Utc>, window_size: Duration) -> DateTime<Utc> {
        let window_millis = window_size.num_milliseconds();
        let timestamp_millis = timestamp.timestamp_millis();
        let window_start_millis = (timestamp_millis / window_millis) * window_millis;
        
        DateTime::from_timestamp_millis(window_start_millis).unwrap()
    }

    /// Analyze cross-source patterns within a time window
    fn analyze_cross_source_window(&self, window_items: &[EvidenceItem]) -> Option<TemporalPattern> {
        if window_items.len() < 2 {
            return None;
        }
        
        // Count unique sources
        let mut source_count = HashMap::new();
        for item in window_items {
            *source_count.entry(item.source).or_insert(0) += 1;
        }
        
        // If we have multiple sources, it's a potential cross-source pattern
        if source_count.len() >= 2 {
            let item_ids: Vec<String> = window_items.iter().map(|item| item.id.clone()).collect();
            let confidence = source_count.len() as f32 / 6.0; // 6 total source types
            
            Some(TemporalPattern {
                pattern_type: PatternType::CrossSource,
                interval: Duration::minutes(5),
                confidence,
                items: item_ids,
            })
        } else {
            None
        }
    }

    /// Align items with detected patterns
    fn align_with_patterns(
        &self,
        evidence_items: &[EvidenceItem],
        patterns: &[TemporalPattern],
    ) -> Vec<EvidenceItem> {
        let mut aligned_items = evidence_items.to_vec();
        
        for pattern in patterns {
            match pattern.pattern_type {
                PatternType::Periodic => {
                    self.align_periodic_items(&mut aligned_items, pattern);
                }
                PatternType::Burst => {
                    self.align_burst_items(&mut aligned_items, pattern);
                }
                PatternType::Cascade => {
                    self.align_cascade_items(&mut aligned_items, pattern);
                }
                PatternType::CrossSource => {
                    self.align_cross_source_items(&mut aligned_items, pattern);
                }
            }
        }
        
        aligned_items
    }

    /// Align periodic items
    fn align_periodic_items(&self, items: &mut [EvidenceItem], pattern: &TemporalPattern) {
        // Find reference time from pattern items
        let pattern_items: Vec<&EvidenceItem> = items.iter()
            .filter(|item| pattern.items.contains(&item.id))
            .collect();
        
        if pattern_items.is_empty() {
            return;
        }
        
        let reference_time = pattern_items[0].timestamp;
        
        // Align items to expected periodic times
        for item in items.iter_mut() {
            if pattern.items.contains(&item.id) {
                let time_diff = item.timestamp - reference_time;
                let interval_count = time_diff.num_seconds() / pattern.interval.num_seconds();
                let expected_time = reference_time + pattern.interval * interval_count as i32;
                
                // Apply small adjustment if within tolerance
                if (item.timestamp - expected_time).abs() <= Duration::seconds(30) {
                    item.timestamp = expected_time;
                }
            }
        }
    }

    /// Align burst items
    fn align_burst_items(&self, items: &mut [EvidenceItem], pattern: &TemporalPattern) {
        // Find burst center
        let burst_items: Vec<&EvidenceItem> = items.iter()
            .filter(|item| pattern.items.contains(&item.id))
            .collect();
        
        if burst_items.is_empty() {
            return;
        }
        
        let burst_center = burst_items.iter()
            .map(|item| item.timestamp.timestamp())
            .sum::<i64>() / burst_items.len() as i64;
        
        let center_time = DateTime::from_timestamp(burst_center, 0).unwrap();
        
        // Align items closer to burst center
        for item in items.iter_mut() {
            if pattern.items.contains(&item.id) {
                let time_diff = item.timestamp - center_time;
                let adjustment = time_diff / 2; // Move halfway to center
                item.timestamp = item.timestamp - adjustment;
            }
        }
    }

    /// Align cascade items
    fn align_cascade_items(&self, items: &mut [EvidenceItem], pattern: &TemporalPattern) {
        // Sort cascade items by severity
        let mut cascade_items: Vec<&mut EvidenceItem> = items.iter_mut()
            .filter(|item| pattern.items.contains(&item.id))
            .collect();
        
        cascade_items.sort_by(|a, b| a.severity.partial_cmp(&b.severity).unwrap());
        
        // Ensure monotonic time progression
        for i in 1..cascade_items.len() {
            if cascade_items[i].timestamp <= cascade_items[i-1].timestamp {
                cascade_items[i].timestamp = cascade_items[i-1].timestamp + Duration::seconds(1);
            }
        }
    }

    /// Align cross-source items
    fn align_cross_source_items(&self, items: &mut [EvidenceItem], pattern: &TemporalPattern) {
        // Find time centroid of cross-source items
        let cross_items: Vec<&EvidenceItem> = items.iter()
            .filter(|item| pattern.items.contains(&item.id))
            .collect();
        
        if cross_items.is_empty() {
            return;
        }
        
        let centroid = cross_items.iter()
            .map(|item| item.timestamp.timestamp())
            .sum::<i64>() / cross_items.len() as i64;
        
        let centroid_time = DateTime::from_timestamp(centroid, 0).unwrap();
        
        // Align items to centroid with small adjustments
        for item in items.iter_mut() {
            if pattern.items.contains(&item.id) {
                let time_diff = item.timestamp - centroid_time;
                let adjustment = time_diff / 4; // Small adjustment toward centroid
                item.timestamp = item.timestamp - adjustment;
            }
        }
    }

    /// Apply dynamic time warping
    fn dynamic_time_warping(&self, aligned_items: &[EvidenceItem]) -> Vec<EvidenceItem> {
        if aligned_items.len() < 2 {
            return aligned_items.to_vec();
        }
        
        let mut warped_items = aligned_items.to_vec();
        
        // Group by source for DTW
        let source_groups = self.group_by_source(&warped_items);
        
        // Apply DTW between source groups
        for (source1, items1) in &source_groups {
            for (source2, items2) in &source_groups {
                if source1 != source2 {
                    let dtw_alignment = self.compute_dtw_alignment(items1, items2);
                    self.apply_dtw_alignment(&mut warped_items, &dtw_alignment);
                }
            }
        }
        
        warped_items
    }

    /// Compute DTW alignment between two sequences
    fn compute_dtw_alignment(&self, seq1: &[EvidenceItem], seq2: &[EvidenceItem]) -> Vec<(usize, usize)> {
        let n = seq1.len();
        let m = seq2.len();
        
        if n == 0 || m == 0 {
            return Vec::new();
        }
        
        // DTW distance matrix
        let mut dtw = vec![vec![f32::INFINITY; m]; n];
        dtw[0][0] = self.compute_item_distance(&seq1[0], &seq2[0]);
        
        // Fill first row and column
        for i in 1..n {
            dtw[i][0] = dtw[i-1][0] + self.compute_item_distance(&seq1[i], &seq2[0]);
        }
        for j in 1..m {
            dtw[0][j] = dtw[0][j-1] + self.compute_item_distance(&seq1[0], &seq2[j]);
        }
        
        // Fill DTW matrix
        for i in 1..n {
            for j in 1..m {
                let cost = self.compute_item_distance(&seq1[i], &seq2[j]);
                dtw[i][j] = cost + dtw[i-1][j].min(dtw[i][j-1]).min(dtw[i-1][j-1]);
            }
        }
        
        // Backtrack to find alignment path
        let mut path = Vec::new();
        let mut i = n - 1;
        let mut j = m - 1;
        
        while i > 0 || j > 0 {
            path.push((i, j));
            
            if i == 0 {
                j -= 1;
            } else if j == 0 {
                i -= 1;
            } else {
                let diag = dtw[i-1][j-1];
                let left = dtw[i][j-1];
                let up = dtw[i-1][j];
                
                if diag <= left && diag <= up {
                    i -= 1;
                    j -= 1;
                } else if left <= up {
                    j -= 1;
                } else {
                    i -= 1;
                }
            }
        }
        
        path.push((0, 0));
        path.reverse();
        path
    }

    /// Compute distance between two evidence items
    fn compute_item_distance(&self, item1: &EvidenceItem, item2: &EvidenceItem) -> f32 {
        let temporal_distance = (item1.timestamp - item2.timestamp).num_seconds().abs() as f32;
        let severity_distance = (item1.severity - item2.severity).abs();
        let confidence_distance = (item1.confidence - item2.confidence).abs();
        
        temporal_distance / 3600.0 + severity_distance + confidence_distance
    }

    /// Apply DTW alignment
    fn apply_dtw_alignment(&self, items: &mut [EvidenceItem], alignment: &[(usize, usize)]) {
        // This is a simplified version - in practice, DTW alignment would be more complex
        for (i, j) in alignment {
            // Apply small time adjustments based on alignment
            // Implementation depends on specific alignment strategy
        }
    }
}

/// Temporal pattern types
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Periodic,
    Burst,
    Cascade,
    CrossSource,
}

/// Temporal pattern structure
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub pattern_type: PatternType,
    pub interval: Duration,
    pub confidence: f32,
    pub items: Vec<String>, // Evidence item IDs
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_temporal_alignment() {
        let algorithm = TemporalAlignmentAlgorithm::new(Duration::minutes(15));
        
        let evidence_items = vec![
            EvidenceItem {
                id: "1".to_string(),
                source: super::super::EvidenceSource::KpiDeviation,
                timestamp: Utc::now(),
                severity: 0.8,
                confidence: 0.9,
                features: vec![0.1, 0.2, 0.3],
                metadata: HashMap::new(),
            },
            EvidenceItem {
                id: "2".to_string(),
                source: super::super::EvidenceSource::AlarmSequence,
                timestamp: Utc::now() + Duration::seconds(30),
                severity: 0.7,
                confidence: 0.85,
                features: vec![0.2, 0.3, 0.4],
                metadata: HashMap::new(),
            },
        ];
        
        let aligned = algorithm.align(&evidence_items);
        assert_eq!(aligned.len(), evidence_items.len());
    }
}