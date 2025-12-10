// User Clustering Algorithms
// Implements clustering algorithms for mobility pattern analysis

use std::collections::HashMap;
use std::f64;
use crate::dtm_mobility::MobilityState;

/// User clustering engine
pub struct UserClusterer {
    /// K-means clustering algorithm
    kmeans: KMeansClusterer,
    
    /// DBSCAN clustering algorithm
    dbscan: DBSCANClusterer,
    
    /// Hierarchical clustering algorithm
    hierarchical: HierarchicalClusterer,
    
    /// Clustering parameters
    params: ClusteringParams,
    
    /// Feature extractors
    feature_extractors: FeatureExtractors,
}

/// K-means clustering algorithm
#[derive(Debug, Clone)]
pub struct KMeansClusterer {
    /// Number of clusters
    pub k: usize,
    
    /// Maximum iterations
    pub max_iterations: usize,
    
    /// Convergence tolerance
    pub tolerance: f64,
    
    /// Cluster centroids
    pub centroids: Vec<Vec<f64>>,
    
    /// Cluster assignments
    pub assignments: Vec<usize>,
}

/// DBSCAN clustering algorithm
#[derive(Debug, Clone)]
pub struct DBSCANClusterer {
    /// Epsilon parameter (neighborhood distance)
    pub epsilon: f64,
    
    /// Minimum points in cluster
    pub min_points: usize,
    
    /// Distance metric
    pub distance_metric: DistanceMetric,
}

/// Hierarchical clustering algorithm
#[derive(Debug, Clone)]
pub struct HierarchicalClusterer {
    /// Linkage method
    pub linkage: LinkageMethod,
    
    /// Distance threshold
    pub distance_threshold: f64,
    
    /// Dendrogram
    pub dendrogram: Vec<ClusterNode>,
}

/// Distance metrics
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Cosine,
    Haversine,
}

/// Linkage methods for hierarchical clustering
#[derive(Debug, Clone, PartialEq)]
pub enum LinkageMethod {
    Single,
    Complete,
    Average,
    Ward,
}

/// Clustering parameters
#[derive(Debug, Clone)]
pub struct ClusteringParams {
    /// Preferred clustering algorithm
    pub algorithm: ClusteringAlgorithm,
    
    /// Auto-select optimal number of clusters
    pub auto_k: bool,
    
    /// Feature normalization
    pub normalize_features: bool,
    
    /// Minimum cluster size
    pub min_cluster_size: usize,
}

/// Clustering algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum ClusteringAlgorithm {
    KMeans,
    DBSCAN,
    Hierarchical,
    Ensemble,
}

/// Feature extractors for different aspects of mobility
#[derive(Debug, Clone)]
pub struct FeatureExtractors {
    /// Spatial features
    pub spatial: SpatialFeatureExtractor,
    
    /// Temporal features
    pub temporal: TemporalFeatureExtractor,
    
    /// Behavioral features
    pub behavioral: BehavioralFeatureExtractor,
}

/// Spatial feature extractor
#[derive(Debug, Clone)]
pub struct SpatialFeatureExtractor {
    /// Geographic bounds for normalization
    pub bounds: GeographicBounds,
    
    /// Spatial resolution
    pub resolution: f64,
}

/// Temporal feature extractor
#[derive(Debug, Clone)]
pub struct TemporalFeatureExtractor {
    /// Time window size
    pub window_size: usize,
    
    /// Temporal patterns
    pub patterns: Vec<TemporalPattern>,
}

/// Behavioral feature extractor
#[derive(Debug, Clone)]
pub struct BehavioralFeatureExtractor {
    /// Handover behavior features
    pub handover_features: bool,
    
    /// Speed distribution features
    pub speed_features: bool,
    
    /// Cell affinity features
    pub cell_affinity_features: bool,
}

/// Geographic bounds
#[derive(Debug, Clone)]
pub struct GeographicBounds {
    pub min_lat: f64,
    pub max_lat: f64,
    pub min_lon: f64,
    pub max_lon: f64,
}

/// Temporal pattern
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Pattern type
    pub pattern_type: TemporalPatternType,
    
    /// Pattern strength
    pub strength: f64,
    
    /// Pattern parameters
    pub parameters: Vec<f64>,
}

/// Temporal pattern types
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalPatternType {
    Hourly,
    Daily,
    Weekly,
    Seasonal,
}

/// Cluster node for hierarchical clustering
#[derive(Debug, Clone)]
pub struct ClusterNode {
    /// Node ID
    pub id: usize,
    
    /// Left child
    pub left: Option<Box<ClusterNode>>,
    
    /// Right child
    pub right: Option<Box<ClusterNode>>,
    
    /// Distance at which clusters were merged
    pub distance: f64,
    
    /// Number of leaves in subtree
    pub leaf_count: usize,
}

/// Clustering result
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    /// Cluster assignments
    pub assignments: HashMap<String, usize>,
    
    /// Cluster centroids
    pub centroids: Vec<Vec<f64>>,
    
    /// Cluster quality metrics
    pub quality_metrics: ClusterQualityMetrics,
    
    /// Cluster descriptions
    pub descriptions: Vec<ClusterDescription>,
}

/// Cluster quality metrics
#[derive(Debug, Clone)]
pub struct ClusterQualityMetrics {
    /// Silhouette score
    pub silhouette_score: f64,
    
    /// Calinski-Harabasz index
    pub calinski_harabasz_index: f64,
    
    /// Davies-Bouldin index
    pub davies_bouldin_index: f64,
    
    /// Inertia (within-cluster sum of squares)
    pub inertia: f64,
}

/// Cluster description
#[derive(Debug, Clone)]
pub struct ClusterDescription {
    /// Cluster ID
    pub cluster_id: usize,
    
    /// Cluster size
    pub size: usize,
    
    /// Dominant mobility state
    pub dominant_mobility_state: MobilityState,
    
    /// Average speed
    pub average_speed: f64,
    
    /// Spatial characteristics
    pub spatial_characteristics: SpatialCharacteristics,
    
    /// Temporal characteristics
    pub temporal_characteristics: TemporalCharacteristics,
}

/// Spatial characteristics of a cluster
#[derive(Debug, Clone)]
pub struct SpatialCharacteristics {
    /// Geographic center
    pub center: (f64, f64),
    
    /// Spatial spread (standard deviation)
    pub spread: f64,
    
    /// Preferred cells
    pub preferred_cells: Vec<String>,
}

/// Temporal characteristics of a cluster
#[derive(Debug, Clone)]
pub struct TemporalCharacteristics {
    /// Active hours
    pub active_hours: Vec<u8>,
    
    /// Peak activity time
    pub peak_activity: u8,
    
    /// Activity pattern
    pub activity_pattern: ActivityPattern,
}

/// Activity pattern
#[derive(Debug, Clone, PartialEq)]
pub enum ActivityPattern {
    Regular,
    Irregular,
    Periodic,
    Sporadic,
}

impl UserClusterer {
    /// Create new user clusterer
    pub fn new() -> Self {
        Self {
            kmeans: KMeansClusterer::new(5),
            dbscan: DBSCANClusterer::new(0.5, 5),
            hierarchical: HierarchicalClusterer::new(),
            params: ClusteringParams::default(),
            feature_extractors: FeatureExtractors::new(),
        }
    }
    
    /// Cluster users based on mobility patterns
    pub fn cluster_users(
        &self,
        user_features: Vec<(String, Vec<f64>)>,
    ) -> Result<HashMap<String, Vec<String>>, String> {
        if user_features.is_empty() {
            return Ok(HashMap::new());
        }
        
        // Extract features and user IDs
        let users: Vec<String> = user_features.iter().map(|(id, _)| id.clone()).collect();
        let mut features: Vec<Vec<f64>> = user_features.into_iter().map(|(_, f)| f).collect();
        
        // Normalize features if requested
        if self.params.normalize_features {
            features = self.normalize_features(features)?;
        }
        
        // Perform clustering based on selected algorithm
        let clustering_result = match self.params.algorithm {
            ClusteringAlgorithm::KMeans => {
                self.kmeans_clustering(&features)?
            }
            ClusteringAlgorithm::DBSCAN => {
                self.dbscan_clustering(&features)?
            }
            ClusteringAlgorithm::Hierarchical => {
                self.hierarchical_clustering(&features)?
            }
            ClusteringAlgorithm::Ensemble => {
                self.ensemble_clustering(&features)?
            }
        };
        
        // Group users by cluster
        let mut clusters: HashMap<String, Vec<String>> = HashMap::new();
        
        for (user_id, cluster_id) in users.iter().zip(clustering_result.assignments.values()) {
            let cluster_key = format!("cluster_{}", cluster_id);
            clusters.entry(cluster_key).or_insert_with(Vec::new).push(user_id.clone());
        }
        
        Ok(clusters)
    }
    
    /// Perform K-means clustering
    fn kmeans_clustering(&self, features: &[Vec<f64>]) -> Result<ClusteringResult, String> {
        let mut kmeans = self.kmeans.clone();
        
        // Determine optimal k if auto_k is enabled
        if self.params.auto_k {
            kmeans.k = self.find_optimal_k(features)?;
        }
        
        // Initialize centroids
        kmeans.initialize_centroids(features)?;
        
        // Perform clustering iterations
        for _ in 0..kmeans.max_iterations {
            let old_centroids = kmeans.centroids.clone();
            
            // Assignment step
            kmeans.assign_points(features)?;
            
            // Update step
            kmeans.update_centroids(features)?;
            
            // Check convergence
            if kmeans.has_converged(&old_centroids) {
                break;
            }
        }
        
        // Create assignments map
        let mut assignments = HashMap::new();
        for (i, &cluster_id) in kmeans.assignments.iter().enumerate() {
            assignments.insert(format!("user_{}", i), cluster_id);
        }
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(features, &kmeans.assignments, &kmeans.centroids)?;
        
        // Generate cluster descriptions
        let descriptions = self.generate_cluster_descriptions(&kmeans.assignments, &kmeans.centroids, features)?;
        
        Ok(ClusteringResult {
            assignments,
            centroids: kmeans.centroids,
            quality_metrics,
            descriptions,
        })
    }
    
    /// Perform DBSCAN clustering
    fn dbscan_clustering(&self, features: &[Vec<f64>]) -> Result<ClusteringResult, String> {
        let dbscan = &self.dbscan;
        let mut assignments = vec![-1i32; features.len()]; // -1 for noise
        let mut visited = vec![false; features.len()];
        let mut cluster_id = 0;
        
        for i in 0..features.len() {
            if visited[i] {
                continue;
            }
            
            visited[i] = true;
            let neighbors = self.find_neighbors(features, i, dbscan.epsilon, &dbscan.distance_metric)?;
            
            if neighbors.len() < dbscan.min_points {
                // Mark as noise
                assignments[i] = -1;
            } else {
                // Start new cluster
                self.expand_cluster(
                    features,
                    &mut assignments,
                    &mut visited,
                    i,
                    neighbors,
                    cluster_id,
                    dbscan,
                )?;
                cluster_id += 1;
            }
        }
        
        // Convert to non-negative assignments
        let assignments: Vec<usize> = assignments.iter()
            .map(|&x| if x >= 0 { x as usize } else { usize::MAX })
            .collect();
        
        // Create assignments map
        let mut assignments_map = HashMap::new();
        for (i, &cluster_id) in assignments.iter().enumerate() {
            assignments_map.insert(format!("user_{}", i), cluster_id);
        }
        
        // Calculate centroids for valid clusters
        let centroids = self.calculate_centroids(features, &assignments)?;
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(features, &assignments, &centroids)?;
        
        // Generate cluster descriptions
        let descriptions = self.generate_cluster_descriptions(&assignments, &centroids, features)?;
        
        Ok(ClusteringResult {
            assignments: assignments_map,
            centroids,
            quality_metrics,
            descriptions,
        })
    }
    
    /// Perform hierarchical clustering
    fn hierarchical_clustering(&self, features: &[Vec<f64>]) -> Result<ClusteringResult, String> {
        let hierarchical = &self.hierarchical;
        
        // Calculate distance matrix
        let distance_matrix = self.calculate_distance_matrix(features)?;
        
        // Build dendrogram
        let dendrogram = self.build_dendrogram(&distance_matrix, &hierarchical.linkage)?;
        
        // Cut dendrogram at specified threshold
        let assignments = self.cut_dendrogram(&dendrogram, hierarchical.distance_threshold)?;
        
        // Create assignments map
        let mut assignments_map = HashMap::new();
        for (i, &cluster_id) in assignments.iter().enumerate() {
            assignments_map.insert(format!("user_{}", i), cluster_id);
        }
        
        // Calculate centroids
        let centroids = self.calculate_centroids(features, &assignments)?;
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(features, &assignments, &centroids)?;
        
        // Generate cluster descriptions
        let descriptions = self.generate_cluster_descriptions(&assignments, &centroids, features)?;
        
        Ok(ClusteringResult {
            assignments: assignments_map,
            centroids,
            quality_metrics,
            descriptions,
        })
    }
    
    /// Perform ensemble clustering
    fn ensemble_clustering(&self, features: &[Vec<f64>]) -> Result<ClusteringResult, String> {
        // Run multiple clustering algorithms
        let kmeans_result = self.kmeans_clustering(features)?;
        let dbscan_result = self.dbscan_clustering(features)?;
        let hierarchical_result = self.hierarchical_clustering(features)?;
        
        // Combine results using consensus clustering
        let consensus_assignments = self.consensus_clustering(vec![
            kmeans_result.assignments,
            dbscan_result.assignments,
            hierarchical_result.assignments,
        ])?;
        
        // Convert to assignment vector
        let mut assignments = vec![0; features.len()];
        for (i, (_, &cluster_id)) in consensus_assignments.iter().enumerate() {
            assignments[i] = cluster_id;
        }
        
        // Calculate centroids
        let centroids = self.calculate_centroids(features, &assignments)?;
        
        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(features, &assignments, &centroids)?;
        
        // Generate cluster descriptions
        let descriptions = self.generate_cluster_descriptions(&assignments, &centroids, features)?;
        
        Ok(ClusteringResult {
            assignments: consensus_assignments,
            centroids,
            quality_metrics,
            descriptions,
        })
    }
    
    /// Normalize features using z-score normalization
    fn normalize_features(&self, features: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>, String> {
        if features.is_empty() {
            return Ok(features);
        }
        
        let num_features = features[0].len();
        let num_samples = features.len();
        
        // Calculate mean and standard deviation for each feature
        let mut means = vec![0.0; num_features];
        let mut stds = vec![0.0; num_features];
        
        // Calculate means
        for sample in &features {
            for (i, &value) in sample.iter().enumerate() {
                means[i] += value;
            }
        }
        for mean in &mut means {
            *mean /= num_samples as f64;
        }
        
        // Calculate standard deviations
        for sample in &features {
            for (i, &value) in sample.iter().enumerate() {
                stds[i] += (value - means[i]).powi(2);
            }
        }
        for std in &mut stds {
            *std = (*std / num_samples as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Avoid division by zero
            }
        }
        
        // Normalize features
        let normalized: Vec<Vec<f64>> = features.into_iter()
            .map(|sample| {
                sample.into_iter()
                    .enumerate()
                    .map(|(i, value)| (value - means[i]) / stds[i])
                    .collect()
            })
            .collect();
        
        Ok(normalized)
    }
    
    /// Find optimal number of clusters using elbow method
    fn find_optimal_k(&self, features: &[Vec<f64>]) -> Result<usize, String> {
        let max_k = (features.len() / 3).min(10).max(2);
        let mut inertias = Vec::new();
        
        for k in 2..=max_k {
            let mut kmeans = KMeansClusterer::new(k);
            kmeans.initialize_centroids(features)?;
            
            // Run clustering
            for _ in 0..kmeans.max_iterations {
                let old_centroids = kmeans.centroids.clone();
                kmeans.assign_points(features)?;
                kmeans.update_centroids(features)?;
                
                if kmeans.has_converged(&old_centroids) {
                    break;
                }
            }
            
            // Calculate inertia
            let inertia = kmeans.calculate_inertia(features)?;
            inertias.push(inertia);
        }
        
        // Find elbow point
        let optimal_k = self.find_elbow_point(&inertias).unwrap_or(3);
        Ok(optimal_k + 2) // Add 2 because we started from k=2
    }
    
    /// Find elbow point in inertia curve
    fn find_elbow_point(&self, inertias: &[f64]) -> Option<usize> {
        if inertias.len() < 3 {
            return None;
        }
        
        let mut max_curvature = 0.0;
        let mut elbow_index = 0;
        
        for i in 1..inertias.len() - 1 {
            let curvature = inertias[i - 1] - 2.0 * inertias[i] + inertias[i + 1];
            if curvature > max_curvature {
                max_curvature = curvature;
                elbow_index = i;
            }
        }
        
        Some(elbow_index)
    }
    
    /// Calculate quality metrics for clustering
    fn calculate_quality_metrics(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
        centroids: &[Vec<f64>],
    ) -> Result<ClusterQualityMetrics, String> {
        let silhouette_score = self.calculate_silhouette_score(features, assignments)?;
        let calinski_harabasz_index = self.calculate_calinski_harabasz_index(features, assignments, centroids)?;
        let davies_bouldin_index = self.calculate_davies_bouldin_index(features, assignments, centroids)?;
        let inertia = self.calculate_inertia(features, assignments, centroids)?;
        
        Ok(ClusterQualityMetrics {
            silhouette_score,
            calinski_harabasz_index,
            davies_bouldin_index,
            inertia,
        })
    }
    
    /// Calculate silhouette score
    fn calculate_silhouette_score(&self, features: &[Vec<f64>], assignments: &[usize]) -> Result<f64, String> {
        let n = features.len();
        let mut silhouette_sum = 0.0;
        
        for i in 0..n {
            let cluster_i = assignments[i];
            
            // Calculate average distance to points in same cluster
            let mut same_cluster_distances = Vec::new();
            for j in 0..n {
                if i != j && assignments[j] == cluster_i {
                    let dist = self.euclidean_distance(&features[i], &features[j]);
                    same_cluster_distances.push(dist);
                }
            }
            
            let a = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };
            
            // Calculate minimum average distance to points in other clusters
            let mut min_other_cluster_distance = f64::INFINITY;
            let unique_clusters: std::collections::HashSet<usize> = assignments.iter().cloned().collect();
            
            for &other_cluster in &unique_clusters {
                if other_cluster != cluster_i {
                    let mut other_cluster_distances = Vec::new();
                    for j in 0..n {
                        if assignments[j] == other_cluster {
                            let dist = self.euclidean_distance(&features[i], &features[j]);
                            other_cluster_distances.push(dist);
                        }
                    }
                    
                    if !other_cluster_distances.is_empty() {
                        let avg_dist = other_cluster_distances.iter().sum::<f64>() / other_cluster_distances.len() as f64;
                        min_other_cluster_distance = min_other_cluster_distance.min(avg_dist);
                    }
                }
            }
            
            let b = min_other_cluster_distance;
            let silhouette = if a.max(b) > 0.0 {
                (b - a) / a.max(b)
            } else {
                0.0
            };
            
            silhouette_sum += silhouette;
        }
        
        Ok(silhouette_sum / n as f64)
    }
    
    /// Calculate Calinski-Harabasz index
    fn calculate_calinski_harabasz_index(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
        centroids: &[Vec<f64>],
    ) -> Result<f64, String> {
        let n = features.len();
        let k = centroids.len();
        
        if k <= 1 {
            return Ok(0.0);
        }
        
        // Calculate overall centroid
        let overall_centroid = self.calculate_overall_centroid(features)?;
        
        // Calculate between-cluster sum of squares
        let mut between_ss = 0.0;
        for (cluster_id, centroid) in centroids.iter().enumerate() {
            let cluster_size = assignments.iter().filter(|&&x| x == cluster_id).count();
            if cluster_size > 0 {
                let dist_sq = self.euclidean_distance_squared(centroid, &overall_centroid);
                between_ss += cluster_size as f64 * dist_sq;
            }
        }
        
        // Calculate within-cluster sum of squares
        let mut within_ss = 0.0;
        for (i, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < centroids.len() {
                let dist_sq = self.euclidean_distance_squared(&features[i], &centroids[cluster_id]);
                within_ss += dist_sq;
            }
        }
        
        if within_ss > 0.0 {
            let ch_index = (between_ss / (k - 1) as f64) / (within_ss / (n - k) as f64);
            Ok(ch_index)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Davies-Bouldin index
    fn calculate_davies_bouldin_index(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
        centroids: &[Vec<f64>],
    ) -> Result<f64, String> {
        let k = centroids.len();
        let mut db_sum = 0.0;
        
        for i in 0..k {
            let mut max_ratio = 0.0;
            
            for j in 0..k {
                if i != j {
                    let s_i = self.calculate_cluster_scatter(features, assignments, i, &centroids[i])?;
                    let s_j = self.calculate_cluster_scatter(features, assignments, j, &centroids[j])?;
                    let d_ij = self.euclidean_distance(&centroids[i], &centroids[j]);
                    
                    if d_ij > 0.0 {
                        let ratio = (s_i + s_j) / d_ij;
                        max_ratio = max_ratio.max(ratio);
                    }
                }
            }
            
            db_sum += max_ratio;
        }
        
        Ok(db_sum / k as f64)
    }
    
    /// Calculate inertia (within-cluster sum of squares)
    fn calculate_inertia(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
        centroids: &[Vec<f64>],
    ) -> Result<f64, String> {
        let mut inertia = 0.0;
        
        for (i, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id < centroids.len() {
                let dist_sq = self.euclidean_distance_squared(&features[i], &centroids[cluster_id]);
                inertia += dist_sq;
            }
        }
        
        Ok(inertia)
    }
    
    /// Calculate overall centroid
    fn calculate_overall_centroid(&self, features: &[Vec<f64>]) -> Result<Vec<f64>, String> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        
        let num_features = features[0].len();
        let mut centroid = vec![0.0; num_features];
        
        for sample in features {
            for (i, &value) in sample.iter().enumerate() {
                centroid[i] += value;
            }
        }
        
        for value in &mut centroid {
            *value /= features.len() as f64;
        }
        
        Ok(centroid)
    }
    
    /// Calculate cluster scatter
    fn calculate_cluster_scatter(
        &self,
        features: &[Vec<f64>],
        assignments: &[usize],
        cluster_id: usize,
        centroid: &[f64],
    ) -> Result<f64, String> {
        let mut scatter = 0.0;
        let mut count = 0;
        
        for (i, &assigned_cluster) in assignments.iter().enumerate() {
            if assigned_cluster == cluster_id {
                let dist = self.euclidean_distance(&features[i], centroid);
                scatter += dist;
                count += 1;
            }
        }
        
        if count > 0 {
            Ok(scatter / count as f64)
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate Euclidean distance
    fn euclidean_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        self.euclidean_distance_squared(a, b).sqrt()
    }
    
    /// Calculate squared Euclidean distance
    fn euclidean_distance_squared(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).powi(2)).sum()
    }
    
    /// Generate cluster descriptions
    fn generate_cluster_descriptions(
        &self,
        assignments: &[usize],
        centroids: &[Vec<f64>],
        features: &[Vec<f64>],
    ) -> Result<Vec<ClusterDescription>, String> {
        let mut descriptions = Vec::new();
        
        for cluster_id in 0..centroids.len() {
            let cluster_members: Vec<usize> = assignments.iter()
                .enumerate()
                .filter(|(_, &c)| c == cluster_id)
                .map(|(i, _)| i)
                .collect();
            
            if cluster_members.is_empty() {
                continue;
            }
            
            // Calculate cluster characteristics
            let size = cluster_members.len();
            let centroid = &centroids[cluster_id];
            
            // Estimate mobility characteristics from features
            let average_speed = centroid.get(0).unwrap_or(&0.0);
            let handover_rate = centroid.get(1).unwrap_or(&0.0);
            
            let dominant_mobility_state = if *average_speed < 0.5 {
                MobilityState::Stationary
            } else if *average_speed < 2.0 {
                MobilityState::Walking
            } else if *average_speed < 30.0 {
                MobilityState::Vehicular
            } else {
                MobilityState::HighSpeed
            };
            
            descriptions.push(ClusterDescription {
                cluster_id,
                size,
                dominant_mobility_state,
                average_speed: *average_speed,
                spatial_characteristics: SpatialCharacteristics {
                    center: (0.0, 0.0), // Would be calculated from actual location data
                    spread: 0.0,
                    preferred_cells: vec![],
                },
                temporal_characteristics: TemporalCharacteristics {
                    active_hours: vec![],
                    peak_activity: 12,
                    activity_pattern: ActivityPattern::Regular,
                },
            });
        }
        
        Ok(descriptions)
    }
    
    /// Calculate centroids for clusters
    fn calculate_centroids(&self, features: &[Vec<f64>], assignments: &[usize]) -> Result<Vec<Vec<f64>>, String> {
        if features.is_empty() {
            return Ok(vec![]);
        }
        
        let num_features = features[0].len();
        let max_cluster = assignments.iter().max().unwrap_or(&0);
        let mut centroids = vec![vec![0.0; num_features]; max_cluster + 1];
        let mut counts = vec![0; max_cluster + 1];
        
        // Sum up features for each cluster
        for (i, &cluster_id) in assignments.iter().enumerate() {
            if cluster_id != usize::MAX {
                for (j, &value) in features[i].iter().enumerate() {
                    centroids[cluster_id][j] += value;
                }
                counts[cluster_id] += 1;
            }
        }
        
        // Calculate averages
        for (cluster_id, count) in counts.iter().enumerate() {
            if *count > 0 {
                for value in &mut centroids[cluster_id] {
                    *value /= *count as f64;
                }
            }
        }
        
        // Remove empty clusters
        let valid_centroids: Vec<Vec<f64>> = centroids.into_iter()
            .zip(counts.iter())
            .filter(|(_, &count)| count > 0)
            .map(|(centroid, _)| centroid)
            .collect();
        
        Ok(valid_centroids)
    }
    
    /// Find neighbors for DBSCAN
    fn find_neighbors(
        &self,
        features: &[Vec<f64>],
        point_idx: usize,
        epsilon: f64,
        distance_metric: &DistanceMetric,
    ) -> Result<Vec<usize>, String> {
        let mut neighbors = Vec::new();
        
        for i in 0..features.len() {
            if i != point_idx {
                let distance = match distance_metric {
                    DistanceMetric::Euclidean => self.euclidean_distance(&features[point_idx], &features[i]),
                    DistanceMetric::Manhattan => self.manhattan_distance(&features[point_idx], &features[i]),
                    DistanceMetric::Cosine => self.cosine_distance(&features[point_idx], &features[i]),
                    DistanceMetric::Haversine => self.haversine_distance(&features[point_idx], &features[i]),
                };
                
                if distance <= epsilon {
                    neighbors.push(i);
                }
            }
        }
        
        Ok(neighbors)
    }
    
    /// Manhattan distance
    fn manhattan_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).sum()
    }
    
    /// Cosine distance
    fn cosine_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        let dot_product: f64 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            1.0
        } else {
            1.0 - dot_product / (norm_a * norm_b)
        }
    }
    
    /// Haversine distance (for geographic coordinates)
    fn haversine_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() < 2 || b.len() < 2 {
            return f64::INFINITY;
        }
        
        const EARTH_RADIUS: f64 = 6371.0; // km
        
        let lat1 = a[0].to_radians();
        let lat2 = b[0].to_radians();
        let delta_lat = (b[0] - a[0]).to_radians();
        let delta_lon = (b[1] - a[1]).to_radians();
        
        let a_val = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a_val.sqrt().asin();
        
        EARTH_RADIUS * c
    }
    
    /// Expand cluster for DBSCAN
    fn expand_cluster(
        &self,
        features: &[Vec<f64>],
        assignments: &mut [i32],
        visited: &mut [bool],
        point_idx: usize,
        mut neighbors: Vec<usize>,
        cluster_id: i32,
        dbscan: &DBSCANClusterer,
    ) -> Result<(), String> {
        assignments[point_idx] = cluster_id;
        
        let mut i = 0;
        while i < neighbors.len() {
            let neighbor_idx = neighbors[i];
            
            if !visited[neighbor_idx] {
                visited[neighbor_idx] = true;
                let neighbor_neighbors = self.find_neighbors(
                    features,
                    neighbor_idx,
                    dbscan.epsilon,
                    &dbscan.distance_metric,
                )?;
                
                if neighbor_neighbors.len() >= dbscan.min_points {
                    for &nn in &neighbor_neighbors {
                        if !neighbors.contains(&nn) {
                            neighbors.push(nn);
                        }
                    }
                }
            }
            
            if assignments[neighbor_idx] == -1 {
                assignments[neighbor_idx] = cluster_id;
            }
            
            i += 1;
        }
        
        Ok(())
    }
    
    /// Calculate distance matrix for hierarchical clustering
    fn calculate_distance_matrix(&self, features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, String> {
        let n = features.len();
        let mut distance_matrix = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in i + 1..n {
                let distance = self.euclidean_distance(&features[i], &features[j]);
                distance_matrix[i][j] = distance;
                distance_matrix[j][i] = distance;
            }
        }
        
        Ok(distance_matrix)
    }
    
    /// Build dendrogram for hierarchical clustering
    fn build_dendrogram(
        &self,
        distance_matrix: &[Vec<f64>],
        linkage: &LinkageMethod,
    ) -> Result<Vec<ClusterNode>, String> {
        // Simplified dendrogram building
        // In practice, this would be more complex
        Ok(vec![])
    }
    
    /// Cut dendrogram at threshold
    fn cut_dendrogram(&self, dendrogram: &[ClusterNode], threshold: f64) -> Result<Vec<usize>, String> {
        // Simplified dendrogram cutting
        // In practice, this would traverse the dendrogram
        Ok(vec![])
    }
    
    /// Consensus clustering
    fn consensus_clustering(
        &self,
        clustering_results: Vec<HashMap<String, usize>>,
    ) -> Result<HashMap<String, usize>, String> {
        // Simplified consensus clustering
        // In practice, this would use more sophisticated methods
        clustering_results.into_iter().next().unwrap_or_else(HashMap::new).into_iter().collect::<Result<HashMap<String, usize>, _>>().map_err(|_| "Consensus clustering failed".to_string())
    }
}

impl KMeansClusterer {
    /// Create new K-means clusterer
    pub fn new(k: usize) -> Self {
        Self {
            k,
            max_iterations: 100,
            tolerance: 1e-4,
            centroids: vec![],
            assignments: vec![],
        }
    }
    
    /// Initialize centroids randomly
    pub fn initialize_centroids(&mut self, features: &[Vec<f64>]) -> Result<(), String> {
        if features.is_empty() {
            return Err("Empty features".to_string());
        }
        
        let num_features = features[0].len();
        let mut rng = fastrand::Rng::new();
        
        self.centroids = (0..self.k)
            .map(|_| {
                let idx = rng.usize(0..features.len());
                features[idx].clone()
            })
            .collect();
        
        self.assignments = vec![0; features.len()];
        
        Ok(())
    }
    
    /// Assign points to closest centroids
    pub fn assign_points(&mut self, features: &[Vec<f64>]) -> Result<(), String> {
        for (i, point) in features.iter().enumerate() {
            let mut min_distance = f64::INFINITY;
            let mut closest_centroid = 0;
            
            for (j, centroid) in self.centroids.iter().enumerate() {
                let distance = point.iter()
                    .zip(centroid.iter())
                    .map(|(&x, &y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt();
                
                if distance < min_distance {
                    min_distance = distance;
                    closest_centroid = j;
                }
            }
            
            self.assignments[i] = closest_centroid;
        }
        
        Ok(())
    }
    
    /// Update centroids
    pub fn update_centroids(&mut self, features: &[Vec<f64>]) -> Result<(), String> {
        if features.is_empty() {
            return Ok(());
        }
        
        let num_features = features[0].len();
        let mut new_centroids = vec![vec![0.0; num_features]; self.k];
        let mut counts = vec![0; self.k];
        
        // Sum up points for each cluster
        for (i, &cluster_id) in self.assignments.iter().enumerate() {
            for (j, &value) in features[i].iter().enumerate() {
                new_centroids[cluster_id][j] += value;
            }
            counts[cluster_id] += 1;
        }
        
        // Calculate averages
        for (cluster_id, count) in counts.iter().enumerate() {
            if *count > 0 {
                for value in &mut new_centroids[cluster_id] {
                    *value /= *count as f64;
                }
            }
        }
        
        self.centroids = new_centroids;
        
        Ok(())
    }
    
    /// Check convergence
    pub fn has_converged(&self, old_centroids: &[Vec<f64>]) -> bool {
        if old_centroids.len() != self.centroids.len() {
            return false;
        }
        
        for (old, new) in old_centroids.iter().zip(self.centroids.iter()) {
            let distance: f64 = old.iter()
                .zip(new.iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum::<f64>()
                .sqrt();
            
            if distance > self.tolerance {
                return false;
            }
        }
        
        true
    }
    
    /// Calculate inertia
    pub fn calculate_inertia(&self, features: &[Vec<f64>]) -> Result<f64, String> {
        let mut inertia = 0.0;
        
        for (i, &cluster_id) in self.assignments.iter().enumerate() {
            let distance_sq: f64 = features[i].iter()
                .zip(self.centroids[cluster_id].iter())
                .map(|(&x, &y)| (x - y).powi(2))
                .sum();
            
            inertia += distance_sq;
        }
        
        Ok(inertia)
    }
}

impl DBSCANClusterer {
    /// Create new DBSCAN clusterer
    pub fn new(epsilon: f64, min_points: usize) -> Self {
        Self {
            epsilon,
            min_points,
            distance_metric: DistanceMetric::Euclidean,
        }
    }
}

impl HierarchicalClusterer {
    /// Create new hierarchical clusterer
    pub fn new() -> Self {
        Self {
            linkage: LinkageMethod::Ward,
            distance_threshold: 0.5,
            dendrogram: vec![],
        }
    }
}

impl Default for ClusteringParams {
    fn default() -> Self {
        Self {
            algorithm: ClusteringAlgorithm::KMeans,
            auto_k: true,
            normalize_features: true,
            min_cluster_size: 5,
        }
    }
}

impl FeatureExtractors {
    /// Create new feature extractors
    pub fn new() -> Self {
        Self {
            spatial: SpatialFeatureExtractor {
                bounds: GeographicBounds {
                    min_lat: 0.0,
                    max_lat: 90.0,
                    min_lon: -180.0,
                    max_lon: 180.0,
                },
                resolution: 0.001,
            },
            temporal: TemporalFeatureExtractor {
                window_size: 24,
                patterns: vec![],
            },
            behavioral: BehavioralFeatureExtractor {
                handover_features: true,
                speed_features: true,
                cell_affinity_features: true,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_user_clusterer_creation() {
        let clusterer = UserClusterer::new();
        assert_eq!(clusterer.kmeans.k, 5);
        assert_eq!(clusterer.dbscan.epsilon, 0.5);
        assert_eq!(clusterer.dbscan.min_points, 5);
    }
    
    #[test]
    fn test_kmeans_clustering() {
        let clusterer = UserClusterer::new();
        let features = vec![
            ("user1".to_string(), vec![1.0, 2.0, 3.0]),
            ("user2".to_string(), vec![1.1, 2.1, 3.1]),
            ("user3".to_string(), vec![10.0, 20.0, 30.0]),
            ("user4".to_string(), vec![10.1, 20.1, 30.1]),
        ];
        
        let result = clusterer.cluster_users(features);
        assert!(result.is_ok());
        
        let clusters = result.unwrap();
        assert!(!clusters.is_empty());
    }
    
    #[test]
    fn test_feature_normalization() {
        let clusterer = UserClusterer::new();
        let features = vec![
            vec![1.0, 10.0, 100.0],
            vec![2.0, 20.0, 200.0],
            vec![3.0, 30.0, 300.0],
        ];
        
        let normalized = clusterer.normalize_features(features);
        assert!(normalized.is_ok());
        
        let normalized = normalized.unwrap();
        assert_eq!(normalized.len(), 3);
        assert_eq!(normalized[0].len(), 3);
        
        // Check that means are approximately 0
        let mean_col_0: f64 = normalized.iter().map(|row| row[0]).sum::<f64>() / 3.0;
        assert!((mean_col_0).abs() < 1e-10);
    }
    
    #[test]
    fn test_distance_calculations() {
        let clusterer = UserClusterer::new();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let euclidean = clusterer.euclidean_distance(&a, &b);
        assert!((euclidean - 5.196152).abs() < 1e-5);
        
        let manhattan = clusterer.manhattan_distance(&a, &b);
        assert!((manhattan - 9.0).abs() < 1e-10);
        
        let cosine = clusterer.cosine_distance(&a, &b);
        assert!(cosine >= 0.0 && cosine <= 1.0);
    }
}