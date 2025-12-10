// DTM Spatial Indexing for Real-time User Location Queries
// Enhanced for standalone swarm demo with real fanndata.csv integration

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// Enhanced spatial indexing system for real user locations from fanndata.csv
pub struct SpatialIndex {
    /// Quad-tree for hierarchical spatial partitioning
    quadtree: QuadTree,
    
    /// Grid-based index for fast rectangular queries
    grid_index: GridIndex,
    
    /// R-tree for efficient range queries
    rtree: RTree,
    
    /// User location cache with history
    user_locations: HashMap<String, UserLocation>,
    
    /// Cell information from fanndata.csv
    cell_locations: HashMap<String, CellLocation>,
    
    /// Spatial index parameters optimized for real network data
    params: SpatialIndexParams,
    
    /// Performance metrics for swarm coordination
    metrics: SpatialMetrics,
}

/// Quad-tree for hierarchical spatial partitioning optimized for cellular networks
#[derive(Debug, Clone)]
pub struct QuadTree {
    /// Root node of the quad-tree
    root: Option<Box<QuadNode>>,
    
    /// Maximum depth of the tree (optimized for cell coverage areas)
    max_depth: usize,
    
    /// Maximum points per node (balanced for query performance)
    max_points: usize,
    
    /// Bounding box covering the network area
    bounds: BoundingBox,
    
    /// Tree statistics for optimization
    stats: QuadTreeStats,
}

/// Enhanced quad-tree node with performance optimizations
#[derive(Debug, Clone)]
pub struct QuadNode {
    /// Node bounding box
    bounds: BoundingBox,
    
    /// Points in this node
    points: Vec<SpatialPoint>,
    
    /// Child nodes (NW, NE, SW, SE)
    children: Option<[Box<QuadNode>; 4]>,
    
    /// Node depth
    depth: usize,
    
    /// Node access frequency for optimization
    access_count: usize,
    
    /// Last update timestamp
    last_update: Instant,
}

/// Grid-based spatial index optimized for cellular coverage
#[derive(Debug, Clone)]
pub struct GridIndex {
    /// Grid cells aligned with cellular coverage areas
    cells: Vec<Vec<GridCell>>,
    
    /// Grid resolution based on average cell radius
    resolution: f64,
    
    /// Grid bounds
    bounds: BoundingBox,
    
    /// Grid dimensions
    width: usize,
    height: usize,
    
    /// Grid performance statistics
    stats: GridStats,
}

/// Enhanced grid cell with cellular network features
#[derive(Debug, Clone)]
pub struct GridCell {
    /// Cell ID
    id: usize,
    
    /// Cell bounds
    bounds: BoundingBox,
    
    /// Spatial points in this cell
    points: Vec<SpatialPoint>,
    
    /// Cell statistics from real network data
    stats: CellStats,
    
    /// Associated cellular base stations
    base_stations: Vec<String>,
    
    /// Coverage quality metrics
    coverage_quality: f64,
}

/// R-tree implementation for range queries
#[derive(Debug, Clone)]
pub struct RTree {
    /// Root node
    root: Option<Box<RNode>>,
    
    /// Maximum entries per node
    max_entries: usize,
    
    /// Minimum entries per node
    min_entries: usize,
    
    /// Tree height
    height: usize,
    
    /// R-tree performance metrics
    query_count: usize,
    avg_query_time: f64,
}

/// R-tree node
#[derive(Debug, Clone)]
pub struct RNode {
    /// Node bounding box
    bounds: BoundingBox,
    
    /// Child nodes (for internal nodes)
    children: Option<Vec<Box<RNode>>>,
    
    /// Data entries (for leaf nodes)
    entries: Option<Vec<SpatialPoint>>,
    
    /// Is this a leaf node?
    is_leaf: bool,
    
    /// Node access statistics
    access_frequency: usize,
}

/// Enhanced spatial point with real network data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialPoint {
    /// User ID from fanndata.csv
    pub user_id: String,
    
    /// Geographic coordinates
    pub location: (f64, f64),
    
    /// Timestamp when location was recorded
    pub timestamp: SystemTime,
    
    /// Cell ID from fanndata.csv
    pub cell_id: String,
    
    /// Signal strength and quality metrics
    pub signal_metrics: SignalMetrics,
    
    /// Mobility characteristics
    pub mobility: MobilityData,
}

/// Signal quality metrics from real network data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalMetrics {
    /// RSRP (Reference Signal Received Power)
    pub rsrp: f64,
    
    /// RSRQ (Reference Signal Received Quality)
    pub rsrq: f64,
    
    /// SINR (Signal to Interference plus Noise Ratio)
    pub sinr: f64,
    
    /// CQI (Channel Quality Indicator)
    pub cqi: u8,
    
    /// Throughput measurements
    pub throughput_ul: f64,
    pub throughput_dl: f64,
}

/// Mobility data extracted from user patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobilityData {
    /// Estimated speed (km/h)
    pub speed: f64,
    
    /// Movement direction (degrees)
    pub direction: f64,
    
    /// Handover count in time window
    pub handover_count: u32,
    
    /// Cell dwell time
    pub dwell_time: f64,
    
    /// Mobility state classification
    pub mobility_state: MobilityState,
}

/// Mobility state classification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MobilityState {
    Stationary,
    Walking,
    Vehicular,
    HighSpeed,
    Unknown,
}

/// Cell location from network topology data
#[derive(Debug, Clone)]
pub struct CellLocation {
    /// Cell ID
    pub cell_id: String,
    
    /// Base station coordinates
    pub location: (f64, f64),
    
    /// Cell coverage radius
    pub coverage_radius: f64,
    
    /// Cell sector information
    pub sector: u8,
    
    /// Cell capacity and load
    pub capacity: CellCapacity,
}

/// Cell capacity and load information
#[derive(Debug, Clone)]
pub struct CellCapacity {
    /// Maximum throughput capacity
    pub max_capacity: f64,
    
    /// Current load percentage
    pub current_load: f64,
    
    /// Number of active users
    pub active_users: usize,
    
    /// Resource utilization metrics
    pub resource_utilization: ResourceUtilization,
}

/// Resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceUtilization {
    /// PRB (Physical Resource Block) utilization
    pub prb_utilization: f64,
    
    /// Processor utilization
    pub cpu_utilization: f64,
    
    /// Memory utilization
    pub memory_utilization: f64,
}

/// User location with enhanced history tracking
#[derive(Debug, Clone)]
pub struct UserLocation {
    /// Current location
    pub current: SpatialPoint,
    
    /// Location history with timestamps
    pub history: Vec<SpatialPoint>,
    
    /// Movement pattern analysis
    pub patterns: MovementPatterns,
    
    /// Prediction for next location
    pub predicted_next: Option<(f64, f64)>,
    
    /// Last update time
    pub last_update: Instant,
}

/// Movement pattern analysis
#[derive(Debug, Clone)]
pub struct MovementPatterns {
    /// Frequent locations
    pub frequent_locations: Vec<(f64, f64)>,
    
    /// Regular routes
    pub regular_routes: Vec<Route>,
    
    /// Time-based patterns
    pub temporal_patterns: Vec<TemporalPattern>,
    
    /// Predictability score
    pub predictability: f64,
}

/// Route information
#[derive(Debug, Clone)]
pub struct Route {
    /// Route points
    pub points: Vec<(f64, f64)>,
    
    /// Usage frequency
    pub frequency: f64,
    
    /// Average travel time
    pub avg_travel_time: f64,
}

/// Temporal pattern
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    /// Hour of day (0-23)
    pub hour: u8,
    
    /// Day of week (0-6)
    pub day_of_week: u8,
    
    /// Location probability distribution
    pub location_distribution: Vec<(f64, f64, f64)>, // (lat, lon, probability)
}

/// Enhanced bounding box with cellular network awareness
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// Minimum coordinates (lat, lon)
    pub min: (f64, f64),
    
    /// Maximum coordinates (lat, lon)
    pub max: (f64, f64),
    
    /// Coverage quality in this area
    pub coverage_quality: f64,
}

/// Cell statistics enhanced with real network data
#[derive(Debug, Clone)]
pub struct CellStats {
    /// Number of users currently in cell
    pub user_count: usize,
    
    /// Average user speed
    pub avg_speed: f64,
    
    /// User density (users per km²)
    pub density: f64,
    
    /// Average signal quality
    pub avg_signal_quality: f64,
    
    /// Handover rate
    pub handover_rate: f64,
    
    /// Cell load percentage
    pub load_percentage: f64,
    
    /// Last update time
    pub last_update: Instant,
}

/// Spatial index parameters optimized for cellular networks
#[derive(Debug, Clone)]
pub struct SpatialIndexParams {
    /// Grid resolution based on average cell radius (meters)
    pub grid_resolution: f64,
    
    /// Maximum quad-tree depth for efficient queries
    pub max_quad_depth: usize,
    
    /// Maximum points per quad-tree node
    pub max_points_per_node: usize,
    
    /// Location history size per user
    pub history_size: usize,
    
    /// Update threshold for significant location changes
    pub update_threshold: f64,
    
    /// Enable predictive location updates
    pub enable_prediction: bool,
    
    /// Cache expiry time for user locations
    pub cache_expiry_seconds: u64,
}

/// Performance metrics for spatial operations
#[derive(Debug, Clone)]
pub struct SpatialMetrics {
    /// Query response times
    pub query_times: QueryTimeMetrics,
    
    /// Index update performance
    pub update_performance: UpdatePerformance,
    
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
    
    /// Cache hit rates
    pub cache_metrics: CacheMetrics,
}

/// Query time metrics
#[derive(Debug, Clone)]
pub struct QueryTimeMetrics {
    /// Average quadtree query time
    pub avg_quadtree_query: f64,
    
    /// Average grid query time
    pub avg_grid_query: f64,
    
    /// Average rtree query time
    pub avg_rtree_query: f64,
    
    /// Query count by type
    pub query_counts: HashMap<String, usize>,
}

/// Update performance metrics
#[derive(Debug, Clone)]
pub struct UpdatePerformance {
    /// Updates per second
    pub updates_per_second: f64,
    
    /// Average update latency
    pub avg_update_latency: f64,
    
    /// Index rebuild frequency
    pub rebuild_frequency: f64,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total memory used by spatial structures
    pub total_memory_bytes: usize,
    
    /// Memory by component
    pub component_memory: HashMap<String, usize>,
    
    /// Peak memory usage
    pub peak_memory_bytes: usize,
}

/// Cache performance metrics
#[derive(Debug, Clone)]
pub struct CacheMetrics {
    /// Cache hit rate
    pub hit_rate: f64,
    
    /// Cache eviction rate
    pub eviction_rate: f64,
    
    /// Average cache lookup time
    pub avg_lookup_time: f64,
}

/// Quad-tree statistics
#[derive(Debug, Clone)]
pub struct QuadTreeStats {
    /// Total nodes in tree
    pub total_nodes: usize,
    
    /// Average depth
    pub avg_depth: f64,
    
    /// Rebalance frequency
    pub rebalance_count: usize,
}

/// Grid statistics
#[derive(Debug, Clone)]
pub struct GridStats {
    /// Active cells
    pub active_cells: usize,
    
    /// Average points per cell
    pub avg_points_per_cell: f64,
    
    /// Load distribution
    pub load_distribution: Vec<f64>,
}

/// Spatial query result with performance metrics
#[derive(Debug, Clone)]
pub struct SpatialQueryResult {
    /// Matching users
    pub users: Vec<UserQueryResult>,
    
    /// Query execution time
    pub execution_time: std::time::Duration,
    
    /// Number of spatial structures accessed
    pub structures_accessed: usize,
    
    /// Result confidence score
    pub confidence: f64,
}

/// Individual user query result
#[derive(Debug, Clone)]
pub struct UserQueryResult {
    /// User ID
    pub user_id: String,
    
    /// Current location
    pub location: (f64, f64),
    
    /// Distance from query point
    pub distance: f64,
    
    /// Signal quality at location
    pub signal_quality: f64,
    
    /// Predicted next location
    pub predicted_location: Option<(f64, f64)>,
}

impl SpatialIndex {
    /// Create new spatial index optimized for cellular network data
    pub fn new() -> Self {
        let params = SpatialIndexParams::default();
        
        // Define network coverage area (can be configured per deployment)
        let coverage_bounds = BoundingBox {
            min: (40.0, -74.5),  // Example: New York area
            max: (41.0, -73.5),
            coverage_quality: 0.9,
        };
        
        Self {
            quadtree: QuadTree::new(coverage_bounds, params.max_quad_depth, params.max_points_per_node),
            grid_index: GridIndex::new(coverage_bounds, params.grid_resolution),
            rtree: RTree::new(),
            user_locations: HashMap::new(),
            cell_locations: HashMap::new(),
            params,
            metrics: SpatialMetrics::new(),
        }
    }
    
    /// Load cell locations from network topology data
    pub fn load_cell_topology(&mut self, cell_data: Vec<CellLocation>) {
        for cell in cell_data {
            self.cell_locations.insert(cell.cell_id.clone(), cell);
        }
    }
    
    /// Update user location from fanndata.csv record
    pub fn update_user_location_from_csv(&mut self, 
        user_id: &str, 
        cell_id: &str,
        timestamp: u64,
        signal_metrics: SignalMetrics,
        throughput_ul: f64,
        throughput_dl: f64,
    ) -> Result<(), String> {
        // Get cell location from topology
        let cell_location = self.cell_locations.get(cell_id)
            .ok_or_else(|| format!("Unknown cell ID: {}", cell_id))?;
        
        // Estimate user location based on cell location and signal strength
        let user_location = self.estimate_user_location(cell_location, &signal_metrics)?;
        
        // Calculate mobility data
        let mobility_data = self.calculate_mobility_data(user_id, user_location, timestamp)?;
        
        let spatial_point = SpatialPoint {
            user_id: user_id.to_string(),
            location: user_location,
            timestamp: UNIX_EPOCH + std::time::Duration::from_secs(timestamp),
            cell_id: cell_id.to_string(),
            signal_metrics,
            mobility: mobility_data,
        };
        
        // Update all spatial structures
        let start_time = Instant::now();
        
        self.quadtree.insert(spatial_point.clone());
        self.grid_index.insert(spatial_point.clone());
        self.rtree.insert(spatial_point.clone());
        
        // Update user location cache with history
        self.update_user_cache(spatial_point);
        
        // Update performance metrics
        let update_time = start_time.elapsed().as_secs_f64();
        self.metrics.update_performance.avg_update_latency = 
            (self.metrics.update_performance.avg_update_latency + update_time) / 2.0;
        
        Ok(())
    }
    
    /// Estimate user location within cell coverage area
    fn estimate_user_location(&self, cell: &CellLocation, signal: &SignalMetrics) -> Result<(f64, f64), String> {
        let base_lat = cell.location.0;
        let base_lon = cell.location.1;
        let radius = cell.coverage_radius;
        
        // Estimate distance from base station based on RSRP
        // Better signal = closer to base station
        let distance_factor = if signal.rsrp > -80.0 {
            0.2  // Very close
        } else if signal.rsrp > -90.0 {
            0.5  // Medium distance
        } else if signal.rsrp > -100.0 {
            0.8  // Far
        } else {
            1.0  // At edge of coverage
        };
        
        let estimated_distance = radius * distance_factor;
        
        // Add some randomization for realistic location distribution
        let angle = fastrand::f64() * 2.0 * std::f64::consts::PI;
        let actual_distance = estimated_distance * (0.5 + fastrand::f64() * 0.5);
        
        let lat_offset = actual_distance * angle.cos() / 111320.0; // Convert meters to degrees
        let lon_offset = actual_distance * angle.sin() / (111320.0 * base_lat.to_radians().cos());
        
        Ok((base_lat + lat_offset, base_lon + lon_offset))
    }
    
    /// Calculate mobility data from location history
    fn calculate_mobility_data(&self, user_id: &str, location: (f64, f64), timestamp: u64) -> Result<MobilityData, String> {
        if let Some(user_loc) = self.user_locations.get(user_id) {
            let last_location = user_loc.current.location;
            let time_diff = timestamp - user_loc.current.timestamp
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            
            if time_diff > 0 {
                let distance = self.haversine_distance(last_location, location);
                let speed = (distance / time_diff as f64) * 3.6; // km/h
                
                let direction = self.calculate_bearing(last_location, location);
                
                let mobility_state = self.classify_mobility_state(speed);
                
                return Ok(MobilityData {
                    speed,
                    direction,
                    handover_count: 0, // Will be updated by handover tracking
                    dwell_time: time_diff as f64,
                    mobility_state,
                });
            }
        }
        
        // Default for new users
        Ok(MobilityData {
            speed: 0.0,
            direction: 0.0,
            handover_count: 0,
            dwell_time: 0.0,
            mobility_state: MobilityState::Unknown,
        })
    }
    
    /// Classify mobility state based on speed
    fn classify_mobility_state(&self, speed: f64) -> MobilityState {
        if speed < 1.0 {
            MobilityState::Stationary
        } else if speed < 10.0 {
            MobilityState::Walking
        } else if speed < 60.0 {
            MobilityState::Vehicular
        } else {
            MobilityState::HighSpeed
        }
    }
    
    /// Calculate bearing between two points
    fn calculate_bearing(&self, from: (f64, f64), to: (f64, f64)) -> f64 {
        let lat1 = from.0.to_radians();
        let lat2 = to.0.to_radians();
        let delta_lon = (to.1 - from.1).to_radians();
        
        let y = delta_lon.sin() * lat2.cos();
        let x = lat1.cos() * lat2.sin() - lat1.sin() * lat2.cos() * delta_lon.cos();
        
        y.atan2(x).to_degrees()
    }
    
    /// Update user location cache with pattern analysis
    fn update_user_cache(&mut self, spatial_point: SpatialPoint) {
        let user_id = spatial_point.user_id.clone();
        
        let user_location = self.user_locations.entry(user_id)
            .or_insert_with(|| UserLocation {
                current: spatial_point.clone(),
                history: Vec::new(),
                patterns: MovementPatterns::new(),
                predicted_next: None,
                last_update: Instant::now(),
            });
        
        // Add current location to history
        user_location.history.push(user_location.current.clone());
        
        // Limit history size
        if user_location.history.len() > self.params.history_size {
            user_location.history.remove(0);
        }
        
        // Update current location
        user_location.current = spatial_point;
        user_location.last_update = Instant::now();
        
        // Update movement patterns
        if user_location.history.len() >= 3 {
            user_location.patterns.update_patterns(&user_location.history);
            
            // Generate location prediction if enabled
            if self.params.enable_prediction {
                user_location.predicted_next = user_location.patterns.predict_next_location();
            }
        }
    }
    
    /// Query users within radius with enhanced filtering
    pub fn query_radius_enhanced(&self, 
        center: (f64, f64), 
        radius_km: f64,
        filter_options: QueryFilter,
    ) -> SpatialQueryResult {
        let start_time = Instant::now();
        let mut users = Vec::new();
        
        // Convert radius to degrees (approximation)
        let radius_deg = radius_km / 111.0;
        
        let query_bounds = BoundingBox {
            min: (center.0 - radius_deg, center.1 - radius_deg),
            max: (center.0 + radius_deg, center.1 + radius_deg),
            coverage_quality: 1.0,
        };
        
        // Use grid index for fast rectangular query
        let candidates = self.grid_index.query_range(query_bounds);
        
        // Filter and enhance results
        for candidate in candidates {
            let distance = self.haversine_distance(center, candidate.location);
            if distance <= radius_km {
                // Apply filters
                if filter_options.apply_filter(&candidate) {
                    let signal_quality = self.calculate_signal_quality(&candidate.signal_metrics);
                    
                    let user_result = UserQueryResult {
                        user_id: candidate.user_id.clone(),
                        location: candidate.location,
                        distance,
                        signal_quality,
                        predicted_location: self.get_predicted_location(&candidate.user_id),
                    };
                    
                    users.push(user_result);
                }
            }
        }
        
        // Sort by distance or signal quality
        users.sort_by(|a, b| {
            match filter_options.sort_by {
                SortBy::Distance => a.distance.partial_cmp(&b.distance).unwrap(),
                SortBy::SignalQuality => b.signal_quality.partial_cmp(&a.signal_quality).unwrap(),
            }
        });
        
        let execution_time = start_time.elapsed();
        let confidence = self.calculate_result_confidence(&users, &filter_options);
        
        SpatialQueryResult {
            users,
            execution_time,
            structures_accessed: 1, // Grid index
            confidence,
        }
    }
    
    /// Calculate overall signal quality score
    fn calculate_signal_quality(&self, metrics: &SignalMetrics) -> f64 {
        // Normalize signal metrics to 0-1 scale
        let rsrp_norm = ((metrics.rsrp + 140.0) / 70.0).max(0.0).min(1.0);
        let rsrq_norm = ((metrics.rsrq + 20.0) / 15.0).max(0.0).min(1.0);
        let sinr_norm = ((metrics.sinr + 10.0) / 40.0).max(0.0).min(1.0);
        
        // Weighted average
        (rsrp_norm * 0.4 + rsrq_norm * 0.3 + sinr_norm * 0.3)
    }
    
    /// Get predicted next location for user
    fn get_predicted_location(&self, user_id: &str) -> Option<(f64, f64)> {
        self.user_locations.get(user_id)
            .and_then(|ul| ul.predicted_next)
    }
    
    /// Calculate confidence score for query results
    fn calculate_result_confidence(&self, users: &[UserQueryResult], filter: &QueryFilter) -> f64 {
        if users.is_empty() {
            return 0.0;
        }
        
        let avg_signal_quality: f64 = users.iter()
            .map(|u| u.signal_quality)
            .sum::<f64>() / users.len() as f64;
        
        let coverage_factor = 0.8; // Based on known coverage quality
        let data_freshness = 0.9;  // Based on recent update frequency
        
        (avg_signal_quality * 0.5 + coverage_factor * 0.3 + data_freshness * 0.2)
    }
    
    /// Find users with similar mobility patterns
    pub fn find_similar_mobility_patterns(&self, user_id: &str, similarity_threshold: f64) -> Vec<String> {
        let target_user = match self.user_locations.get(user_id) {
            Some(user) => user,
            None => return vec![],
        };
        
        let mut similar_users = Vec::new();
        
        for (other_user_id, other_user) in &self.user_locations {
            if other_user_id != user_id {
                let similarity = self.calculate_mobility_similarity(
                    &target_user.patterns,
                    &other_user.patterns,
                );
                
                if similarity >= similarity_threshold {
                    similar_users.push(other_user_id.clone());
                }
            }
        }
        
        similar_users
    }
    
    /// Calculate mobility pattern similarity
    fn calculate_mobility_similarity(&self, pattern1: &MovementPatterns, pattern2: &MovementPatterns) -> f64 {
        // Simplified similarity calculation
        // In practice, this would use more sophisticated pattern matching
        let location_similarity = self.calculate_location_similarity(&pattern1.frequent_locations, &pattern2.frequent_locations);
        let temporal_similarity = self.calculate_temporal_similarity(&pattern1.temporal_patterns, &pattern2.temporal_patterns);
        
        (location_similarity + temporal_similarity) / 2.0
    }
    
    /// Calculate location similarity
    fn calculate_location_similarity(&self, locations1: &[(f64, f64)], locations2: &[(f64, f64)]) -> f64 {
        if locations1.is_empty() || locations2.is_empty() {
            return 0.0;
        }
        
        let mut total_similarity = 0.0;
        let mut comparisons = 0;
        
        for loc1 in locations1 {
            for loc2 in locations2 {
                let distance = self.haversine_distance(*loc1, *loc2);
                let similarity = (-distance / 1000.0).exp(); // Exponential decay with distance
                total_similarity += similarity;
                comparisons += 1;
            }
        }
        
        if comparisons > 0 {
            total_similarity / comparisons as f64
        } else {
            0.0
        }
    }
    
    /// Calculate temporal pattern similarity
    fn calculate_temporal_similarity(&self, patterns1: &[TemporalPattern], patterns2: &[TemporalPattern]) -> f64 {
        // Simplified temporal similarity
        // Compare active hours overlap
        let hours1: std::collections::HashSet<u8> = patterns1.iter().map(|p| p.hour).collect();
        let hours2: std::collections::HashSet<u8> = patterns2.iter().map(|p| p.hour).collect();
        
        let intersection = hours1.intersection(&hours2).count();
        let union = hours1.union(&hours2).count();
        
        if union > 0 {
            intersection as f64 / union as f64
        } else {
            0.0
        }
    }
    
    /// Get spatial index performance metrics
    pub fn get_metrics(&self) -> &SpatialMetrics {
        &self.metrics
    }
    
    /// Optimize spatial structures based on usage patterns
    pub fn optimize_structures(&mut self) {
        // Rebalance quadtree if needed
        if self.quadtree.stats.rebalance_count > 100 {
            self.quadtree.rebalance();
        }
        
        // Optimize grid resolution based on user density
        self.grid_index.optimize_resolution();
        
        // Clean up old user locations
        self.cleanup_expired_locations();
    }
    
    /// Clean up expired user locations
    fn cleanup_expired_locations(&mut self) {
        let expiry_duration = std::time::Duration::from_secs(self.params.cache_expiry_seconds);
        let now = Instant::now();
        
        self.user_locations.retain(|_, user_loc| {
            now.duration_since(user_loc.last_update) < expiry_duration
        });
    }
    
    /// Calculate Haversine distance between two points
    fn haversine_distance(&self, loc1: (f64, f64), loc2: (f64, f64)) -> f64 {
        const EARTH_RADIUS: f64 = 6371.0; // km
        
        let lat1 = loc1.0.to_radians();
        let lat2 = loc2.0.to_radians();
        let delta_lat = (loc2.0 - loc1.0).to_radians();
        let delta_lon = (loc2.1 - loc1.1).to_radians();
        
        let a = (delta_lat / 2.0).sin().powi(2)
            + lat1.cos() * lat2.cos() * (delta_lon / 2.0).sin().powi(2);
        let c = 2.0 * a.sqrt().asin();
        
        EARTH_RADIUS * c
    }
}

/// Query filter options
#[derive(Debug, Clone)]
pub struct QueryFilter {
    /// Minimum signal quality
    pub min_signal_quality: Option<f64>,
    
    /// Maximum speed filter
    pub max_speed: Option<f64>,
    
    /// Mobility state filter
    pub mobility_states: Option<Vec<MobilityState>>,
    
    /// Cell ID filter
    pub cell_ids: Option<Vec<String>>,
    
    /// Sort results by
    pub sort_by: SortBy,
}

/// Sort options for query results
#[derive(Debug, Clone)]
pub enum SortBy {
    Distance,
    SignalQuality,
}

// Implementation for various helper structures and traits
impl QuadTree {
    pub fn new(bounds: BoundingBox, max_depth: usize, max_points: usize) -> Self {
        Self {
            root: None,
            max_depth,
            max_points,
            bounds,
            stats: QuadTreeStats::new(),
        }
    }
    
    pub fn insert(&mut self, point: SpatialPoint) {
        if self.root.is_none() {
            self.root = Some(Box::new(QuadNode {
                bounds: self.bounds,
                points: vec![point],
                children: None,
                depth: 0,
                access_count: 1,
                last_update: Instant::now(),
            }));
        } else {
            self.root.as_mut().unwrap().insert(point, self.max_depth, self.max_points);
        }
        self.stats.total_nodes += 1;
    }
    
    pub fn rebalance(&mut self) {
        // Implement quadtree rebalancing logic
        self.stats.rebalance_count += 1;
    }
}

impl GridIndex {
    pub fn new(bounds: BoundingBox, resolution: f64) -> Self {
        let width = ((bounds.max.1 - bounds.min.1) * 111320.0 / resolution).ceil() as usize;
        let height = ((bounds.max.0 - bounds.min.0) * 111320.0 / resolution).ceil() as usize;
        
        let mut cells = Vec::with_capacity(height);
        for i in 0..height {
            let mut row = Vec::with_capacity(width);
            for j in 0..width {
                row.push(GridCell {
                    id: i * width + j,
                    bounds: BoundingBox {
                        min: (
                            bounds.min.0 + (i as f64 * resolution / 111320.0),
                            bounds.min.1 + (j as f64 * resolution / 111320.0),
                        ),
                        max: (
                            bounds.min.0 + ((i + 1) as f64 * resolution / 111320.0),
                            bounds.min.1 + ((j + 1) as f64 * resolution / 111320.0),
                        ),
                        coverage_quality: 0.8,
                    },
                    points: Vec::new(),
                    stats: CellStats::new(),
                    base_stations: Vec::new(),
                    coverage_quality: 0.8,
                });
            }
            cells.push(row);
        }
        
        Self {
            cells,
            resolution,
            bounds,
            width,
            height,
            stats: GridStats::new(),
        }
    }
    
    pub fn insert(&mut self, point: SpatialPoint) {
        if let Some((row, col)) = self.get_cell_indices(point.location) {
            if row < self.height && col < self.width {
                self.cells[row][col].points.push(point);
                self.cells[row][col].stats.user_count = self.cells[row][col].points.len();
                self.cells[row][col].stats.last_update = Instant::now();
            }
        }
    }
    
    pub fn query_range(&self, bounds: BoundingBox) -> Vec<SpatialPoint> {
        let mut results = Vec::new();
        
        let min_indices = self.get_cell_indices(bounds.min);
        let max_indices = self.get_cell_indices(bounds.max);
        
        if let (Some((min_row, min_col)), Some((max_row, max_col))) = (min_indices, max_indices) {
            for row in min_row..=max_row.min(self.height - 1) {
                for col in min_col..=max_col.min(self.width - 1) {
                    for point in &self.cells[row][col].points {
                        if bounds.contains_point(point.location) {
                            results.push(point.clone());
                        }
                    }
                }
            }
        }
        
        results
    }
    
    fn get_cell_indices(&self, location: (f64, f64)) -> Option<(usize, usize)> {
        if !self.bounds.contains_point(location) {
            return None;
        }
        
        let row = ((location.0 - self.bounds.min.0) * 111320.0 / self.resolution) as usize;
        let col = ((location.1 - self.bounds.min.1) * 111320.0 / self.resolution) as usize;
        
        Some((row, col))
    }
    
    pub fn optimize_resolution(&mut self) {
        // Analyze current usage patterns and adjust grid resolution
        let avg_points = self.stats.avg_points_per_cell;
        
        if avg_points > 50.0 {
            // Grid cells are too dense, increase resolution
            self.resolution *= 0.8;
        } else if avg_points < 5.0 {
            // Grid cells are too sparse, decrease resolution
            self.resolution *= 1.2;
        }
        
        // Note: In practice, this would trigger a grid rebuild
    }
}

impl RTree {
    pub fn new() -> Self {
        Self {
            root: None,
            max_entries: 16,
            min_entries: 4,
            height: 0,
            query_count: 0,
            avg_query_time: 0.0,
        }
    }
    
    pub fn insert(&mut self, point: SpatialPoint) {
        // Simplified R-tree insertion
        if self.root.is_none() {
            self.root = Some(Box::new(RNode {
                bounds: BoundingBox {
                    min: point.location,
                    max: point.location,
                    coverage_quality: 1.0,
                },
                children: None,
                entries: Some(vec![point]),
                is_leaf: true,
                access_frequency: 1,
            }));
            self.height = 1;
        }
        // Full R-tree implementation would be more complex
    }
}

impl BoundingBox {
    pub fn contains_point(&self, point: (f64, f64)) -> bool {
        point.0 >= self.min.0 && point.0 <= self.max.0 &&
        point.1 >= self.min.1 && point.1 <= self.max.1
    }
    
    pub fn intersects(&self, other: BoundingBox) -> bool {
        self.min.0 <= other.max.0 && self.max.0 >= other.min.0 &&
        self.min.1 <= other.max.1 && self.max.1 >= other.min.1
    }
}

impl MovementPatterns {
    pub fn new() -> Self {
        Self {
            frequent_locations: Vec::new(),
            regular_routes: Vec::new(),
            temporal_patterns: Vec::new(),
            predictability: 0.0,
        }
    }
    
    pub fn update_patterns(&mut self, history: &[SpatialPoint]) {
        // Update frequent locations
        self.update_frequent_locations(history);
        
        // Update temporal patterns
        self.update_temporal_patterns(history);
        
        // Calculate predictability
        self.predictability = self.calculate_predictability(history);
    }
    
    fn update_frequent_locations(&mut self, history: &[SpatialPoint]) {
        // Cluster historical locations to find frequent places
        let mut location_counts: HashMap<(i32, i32), usize> = HashMap::new();
        
        for point in history {
            // Discretize location to ~100m grid
            let grid_lat = (point.location.0 * 1000.0) as i32;
            let grid_lon = (point.location.1 * 1000.0) as i32;
            *location_counts.entry((grid_lat, grid_lon)).or_insert(0) += 1;
        }
        
        // Find locations with significant frequency
        let total_points = history.len();
        self.frequent_locations = location_counts.into_iter()
            .filter(|(_, count)| *count > total_points / 10) // At least 10% of time
            .map(|((lat, lon), _)| (lat as f64 / 1000.0, lon as f64 / 1000.0))
            .collect();
    }
    
    fn update_temporal_patterns(&mut self, history: &[SpatialPoint]) {
        // Analyze time-based location patterns
        let mut hourly_locations: HashMap<u8, Vec<(f64, f64)>> = HashMap::new();
        
        for point in history {
            // Simplified time extraction - in practice would use proper datetime parsing
            let hour = ((point.timestamp.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() / 3600) % 24) as u8;
            
            hourly_locations.entry(hour)
                .or_insert_with(Vec::new)
                .push(point.location);
        }
        
        // Create temporal patterns
        self.temporal_patterns = hourly_locations.into_iter()
            .map(|(hour, locations)| {
                let location_distribution = self.create_location_distribution(locations);
                TemporalPattern {
                    hour,
                    day_of_week: 0, // Simplified
                    location_distribution,
                }
            })
            .collect();
    }
    
    fn create_location_distribution(&self, locations: Vec<(f64, f64)>) -> Vec<(f64, f64, f64)> {
        // Create probability distribution over locations
        let mut distribution = Vec::new();
        let total_count = locations.len() as f64;
        
        let mut location_counts: HashMap<(i32, i32), usize> = HashMap::new();
        for location in &locations {
            let grid_key = ((location.0 * 1000.0) as i32, (location.1 * 1000.0) as i32);
            *location_counts.entry(grid_key).or_insert(0) += 1;
        }
        
        for ((lat_grid, lon_grid), count) in location_counts {
            let probability = count as f64 / total_count;
            let lat = lat_grid as f64 / 1000.0;
            let lon = lon_grid as f64 / 1000.0;
            distribution.push((lat, lon, probability));
        }
        
        distribution
    }
    
    fn calculate_predictability(&self, history: &[SpatialPoint]) -> f64 {
        if history.len() < 10 {
            return 0.0;
        }
        
        // Calculate entropy of location sequence
        let mut location_counts: HashMap<(i32, i32), usize> = HashMap::new();
        for point in history {
            let grid_key = ((point.location.0 * 100.0) as i32, (point.location.1 * 100.0) as i32);
            *location_counts.entry(grid_key).or_insert(0) += 1;
        }
        
        let total = history.len() as f64;
        let entropy: f64 = location_counts.values()
            .map(|&count| {
                let p = count as f64 / total;
                -p * p.log2()
            })
            .sum();
        
        // Convert entropy to predictability (0-1 scale)
        let max_entropy = (location_counts.len() as f64).log2();
        if max_entropy > 0.0 {
            1.0 - (entropy / max_entropy)
        } else {
            1.0
        }
    }
    
    pub fn predict_next_location(&self) -> Option<(f64, f64)> {
        // Simple prediction based on current time and patterns
        // Get current hour from system time
        let now = SystemTime::now();
        let current_hour = ((now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() / 3600) % 24) as u8;
        
        for pattern in &self.temporal_patterns {
            if pattern.hour == current_hour {
                // Return most probable location for this hour
                if let Some(best_location) = pattern.location_distribution
                    .iter()
                    .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap()) {
                    return Some((best_location.0, best_location.1));
                }
            }
        }
        
        // Fallback to most frequent location
        self.frequent_locations.first().copied()
    }
}

// Default implementations
impl Default for SpatialIndexParams {
    fn default() -> Self {
        Self {
            grid_resolution: 500.0,  // 500 meters - good for cellular networks
            max_quad_depth: 12,
            max_points_per_node: 32,
            history_size: 200,
            update_threshold: 50.0,  // 50 meters
            enable_prediction: true,
            cache_expiry_seconds: 3600, // 1 hour
        }
    }
}

impl SpatialMetrics {
    pub fn new() -> Self {
        Self {
            query_times: QueryTimeMetrics::new(),
            update_performance: UpdatePerformance::new(),
            memory_usage: MemoryUsage::new(),
            cache_metrics: CacheMetrics::new(),
        }
    }
}

impl QueryTimeMetrics {
    pub fn new() -> Self {
        Self {
            avg_quadtree_query: 0.0,
            avg_grid_query: 0.0,
            avg_rtree_query: 0.0,
            query_counts: HashMap::new(),
        }
    }
}

impl UpdatePerformance {
    pub fn new() -> Self {
        Self {
            updates_per_second: 0.0,
            avg_update_latency: 0.0,
            rebuild_frequency: 0.0,
        }
    }
}

impl MemoryUsage {
    pub fn new() -> Self {
        Self {
            total_memory_bytes: 0,
            component_memory: HashMap::new(),
            peak_memory_bytes: 0,
        }
    }
}

impl CacheMetrics {
    pub fn new() -> Self {
        Self {
            hit_rate: 0.0,
            eviction_rate: 0.0,
            avg_lookup_time: 0.0,
        }
    }
}

impl CellStats {
    pub fn new() -> Self {
        Self {
            user_count: 0,
            avg_speed: 0.0,
            density: 0.0,
            avg_signal_quality: 0.0,
            handover_rate: 0.0,
            load_percentage: 0.0,
            last_update: Instant::now(),
        }
    }
}

impl QuadTreeStats {
    pub fn new() -> Self {
        Self {
            total_nodes: 0,
            avg_depth: 0.0,
            rebalance_count: 0,
        }
    }
}

impl GridStats {
    pub fn new() -> Self {
        Self {
            active_cells: 0,
            avg_points_per_cell: 0.0,
            load_distribution: Vec::new(),
        }
    }
}

impl QueryFilter {
    pub fn apply_filter(&self, point: &SpatialPoint) -> bool {
        // Apply signal quality filter
        if let Some(min_quality) = self.min_signal_quality {
            let quality = self.calculate_signal_quality(&point.signal_metrics);
            if quality < min_quality {
                return false;
            }
        }
        
        // Apply speed filter
        if let Some(max_speed) = self.max_speed {
            if point.mobility.speed > max_speed {
                return false;
            }
        }
        
        // Apply mobility state filter
        if let Some(ref states) = self.mobility_states {
            if !states.contains(&point.mobility.mobility_state) {
                return false;
            }
        }
        
        // Apply cell ID filter
        if let Some(ref cell_ids) = self.cell_ids {
            if !cell_ids.contains(&point.cell_id) {
                return false;
            }
        }
        
        true
    }
    
    fn calculate_signal_quality(&self, metrics: &SignalMetrics) -> f64 {
        // Same calculation as in SpatialIndex
        let rsrp_norm = ((metrics.rsrp + 140.0) / 70.0).max(0.0).min(1.0);
        let rsrq_norm = ((metrics.rsrq + 20.0) / 15.0).max(0.0).min(1.0);
        let sinr_norm = ((metrics.sinr + 10.0) / 40.0).max(0.0).min(1.0);
        
        (rsrp_norm * 0.4 + rsrq_norm * 0.3 + sinr_norm * 0.3)
    }
}

impl Default for QueryFilter {
    fn default() -> Self {
        Self {
            min_signal_quality: None,
            max_speed: None,
            mobility_states: None,
            cell_ids: None,
            sort_by: SortBy::Distance,
        }
    }
}

// Implement required traits for QuadNode
impl QuadNode {
    fn insert(&mut self, point: SpatialPoint, max_depth: usize, max_points: usize) {
        self.access_count += 1;
        self.last_update = Instant::now();
        
        if !self.bounds.contains_point(point.location) {
            return;
        }
        
        if self.children.is_none() {
            self.points.push(point);
            
            if self.points.len() > max_points && self.depth < max_depth {
                self.split(max_depth, max_points);
            }
        } else {
            if let Some(children) = &mut self.children {
                for child in children.iter_mut() {
                    if child.bounds.contains_point(point.location) {
                        child.insert(point.clone(), max_depth, max_points);
                        break;
                    }
                }
            }
        }
    }
    
    fn split(&mut self, max_depth: usize, max_points: usize) {
        let mid_lat = (self.bounds.min.0 + self.bounds.max.0) / 2.0;
        let mid_lon = (self.bounds.min.1 + self.bounds.max.1) / 2.0;
        
        // Create four quadrants
        let nw = QuadNode {
            bounds: BoundingBox {
                min: (mid_lat, self.bounds.min.1),
                max: (self.bounds.max.0, mid_lon),
                coverage_quality: self.bounds.coverage_quality,
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
            access_count: 0,
            last_update: Instant::now(),
        };
        
        let ne = QuadNode {
            bounds: BoundingBox {
                min: (mid_lat, mid_lon),
                max: self.bounds.max,
                coverage_quality: self.bounds.coverage_quality,
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
            access_count: 0,
            last_update: Instant::now(),
        };
        
        let sw = QuadNode {
            bounds: BoundingBox {
                min: self.bounds.min,
                max: (mid_lat, mid_lon),
                coverage_quality: self.bounds.coverage_quality,
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
            access_count: 0,
            last_update: Instant::now(),
        };
        
        let se = QuadNode {
            bounds: BoundingBox {
                min: (self.bounds.min.0, mid_lon),
                max: (mid_lat, self.bounds.max.1),
                coverage_quality: self.bounds.coverage_quality,
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
            access_count: 0,
            last_update: Instant::now(),
        };
        
        self.children = Some([
            Box::new(nw),
            Box::new(ne),
            Box::new(sw),
            Box::new(se),
        ]);
        
        // Redistribute points
        let points = std::mem::take(&mut self.points);
        for point in points {
            self.insert(point, max_depth, max_points);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spatial_index_creation() {
        let index = SpatialIndex::new();
        assert_eq!(index.params.grid_resolution, 500.0);
        assert_eq!(index.params.max_quad_depth, 12);
        assert!(index.params.enable_prediction);
    }
    
    #[test]
    fn test_user_location_estimation() {
        let index = SpatialIndex::new();
        let cell = CellLocation {
            cell_id: "test_cell".to_string(),
            location: (40.7128, -74.0060),
            coverage_radius: 1000.0,
            sector: 1,
            capacity: CellCapacity {
                max_capacity: 100.0,
                current_load: 50.0,
                active_users: 25,
                resource_utilization: ResourceUtilization {
                    prb_utilization: 0.6,
                    cpu_utilization: 0.4,
                    memory_utilization: 0.3,
                },
            },
        };
        
        let signal = SignalMetrics {
            rsrp: -85.0,
            rsrq: -10.0,
            sinr: 15.0,
            cqi: 12,
            throughput_ul: 10.0,
            throughput_dl: 50.0,
        };
        
        let location = index.estimate_user_location(&cell, &signal).unwrap();
        
        // Check that estimated location is within cell coverage
        let distance = index.haversine_distance(cell.location, location);
        assert!(distance <= cell.coverage_radius / 1000.0); // Convert to km
    }
    
    #[test]
    fn test_mobility_classification() {
        let index = SpatialIndex::new();
        
        assert_eq!(index.classify_mobility_state(0.5), MobilityState::Stationary);
        assert_eq!(index.classify_mobility_state(5.0), MobilityState::Walking);
        assert_eq!(index.classify_mobility_state(30.0), MobilityState::Vehicular);
        assert_eq!(index.classify_mobility_state(80.0), MobilityState::HighSpeed);
    }
    
    #[test]
    fn test_signal_quality_calculation() {
        let index = SpatialIndex::new();
        let good_signal = SignalMetrics {
            rsrp: -75.0,
            rsrq: -8.0,
            sinr: 20.0,
            cqi: 15,
            throughput_ul: 20.0,
            throughput_dl: 100.0,
        };
        
        let quality = index.calculate_signal_quality(&good_signal);
        assert!(quality > 0.7); // Should be high quality
        
        let poor_signal = SignalMetrics {
            rsrp: -110.0,
            rsrq: -15.0,
            sinr: 5.0,
            cqi: 3,
            throughput_ul: 1.0,
            throughput_dl: 5.0,
        };
        
        let poor_quality = index.calculate_signal_quality(&poor_signal);
        assert!(poor_quality < 0.3); // Should be poor quality
    }
    
    #[test]
    fn test_movement_pattern_prediction() {
        let mut patterns = MovementPatterns::new();
        
        // Create some historical data
        let history = vec![
            SpatialPoint {
                user_id: "test_user".to_string(),
                location: (40.7128, -74.0060),
                timestamp: SystemTime::now(),
                cell_id: "cell_1".to_string(),
                signal_metrics: SignalMetrics {
                    rsrp: -80.0, rsrq: -10.0, sinr: 15.0, cqi: 12,
                    throughput_ul: 10.0, throughput_dl: 50.0,
                },
                mobility: MobilityData {
                    speed: 5.0, direction: 90.0, handover_count: 0,
                    dwell_time: 300.0, mobility_state: MobilityState::Walking,
                },
            },
            // Add more historical points...
        ];
        
        patterns.update_patterns(&history);
        
        // Test prediction
        let prediction = patterns.predict_next_location();
        assert!(prediction.is_some());
    }
}