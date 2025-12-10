// Spatial Indexing for Real-time Queries
// Implements efficient spatial data structures for mobility tracking

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Spatial indexing system for user locations
pub struct SpatialIndex {
    /// Quad-tree for spatial partitioning
    quadtree: QuadTree,
    
    /// Grid-based index for fast queries
    grid_index: GridIndex,
    
    /// R-tree for range queries
    rtree: RTree,
    
    /// User location cache
    user_locations: HashMap<String, UserLocation>,
    
    /// Spatial index parameters
    params: SpatialIndexParams,
}

/// Quad-tree for hierarchical spatial partitioning
#[derive(Debug, Clone)]
pub struct QuadTree {
    /// Root node of the quad-tree
    root: Option<Box<QuadNode>>,
    
    /// Maximum depth of the tree
    max_depth: usize,
    
    /// Maximum points per node
    max_points: usize,
    
    /// Bounding box of the entire area
    bounds: BoundingBox,
}

/// Quad-tree node
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
}

/// Grid-based spatial index
#[derive(Debug, Clone)]
pub struct GridIndex {
    /// Grid cells
    cells: Vec<Vec<GridCell>>,
    
    /// Grid resolution (meters per cell)
    resolution: f64,
    
    /// Grid bounds
    bounds: BoundingBox,
    
    /// Grid dimensions
    width: usize,
    height: usize,
}

/// Grid cell
#[derive(Debug, Clone)]
pub struct GridCell {
    /// Cell ID
    id: usize,
    
    /// Cell bounds
    bounds: BoundingBox,
    
    /// Points in this cell
    points: Vec<SpatialPoint>,
    
    /// Cell statistics
    stats: CellStats,
}

/// R-tree for range queries
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
}

/// Spatial point with user information
#[derive(Debug, Clone)]
pub struct SpatialPoint {
    /// User ID
    pub user_id: String,
    
    /// Location coordinates
    pub location: (f64, f64),
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Additional metadata
    pub metadata: SpatialMetadata,
}

/// Spatial metadata
#[derive(Debug, Clone)]
pub struct SpatialMetadata {
    /// Cell ID
    pub cell_id: String,
    
    /// Signal strength
    pub signal_strength: f64,
    
    /// Speed estimate
    pub speed: f64,
    
    /// Direction
    pub direction: f64,
}

/// User location with history
#[derive(Debug, Clone)]
pub struct UserLocation {
    /// Current location
    pub current: SpatialPoint,
    
    /// Location history
    pub history: Vec<SpatialPoint>,
    
    /// Last update time
    pub last_update: Instant,
}

/// Bounding box for spatial regions
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    /// Minimum coordinates
    pub min: (f64, f64),
    
    /// Maximum coordinates
    pub max: (f64, f64),
}

/// Cell statistics
#[derive(Debug, Clone)]
pub struct CellStats {
    /// Number of users
    pub user_count: usize,
    
    /// Average speed
    pub avg_speed: f64,
    
    /// User density
    pub density: f64,
    
    /// Last update time
    pub last_update: Instant,
}

/// Spatial index parameters
#[derive(Debug, Clone)]
pub struct SpatialIndexParams {
    /// Grid resolution in meters
    pub grid_resolution: f64,
    
    /// Maximum quad-tree depth
    pub max_quad_depth: usize,
    
    /// Maximum points per quad-tree node
    pub max_points_per_node: usize,
    
    /// Location history size
    pub history_size: usize,
    
    /// Update frequency threshold
    pub update_threshold: f64,
}

/// Query result
#[derive(Debug, Clone)]
pub struct SpatialQueryResult {
    /// Matching users
    pub users: Vec<String>,
    
    /// Query execution time
    pub execution_time: std::time::Duration,
    
    /// Number of nodes visited
    pub nodes_visited: usize,
}

impl SpatialIndex {
    /// Create new spatial index
    pub fn new() -> Self {
        let params = SpatialIndexParams::default();
        
        // Define global bounds (can be adjusted for specific regions)
        let global_bounds = BoundingBox {
            min: (-90.0, -180.0),
            max: (90.0, 180.0),
        };
        
        Self {
            quadtree: QuadTree::new(global_bounds, params.max_quad_depth, params.max_points_per_node),
            grid_index: GridIndex::new(global_bounds, params.grid_resolution),
            rtree: RTree::new(),
            user_locations: HashMap::new(),
            params,
        }
    }
    
    /// Update user location
    pub fn update_user_location(&mut self, user_id: &str, location: (f64, f64)) {
        let spatial_point = SpatialPoint {
            user_id: user_id.to_string(),
            location,
            timestamp: Instant::now(),
            metadata: SpatialMetadata {
                cell_id: String::new(),
                signal_strength: 0.0,
                speed: 0.0,
                direction: 0.0,
            },
        };
        
        // Update quad-tree
        self.quadtree.insert(spatial_point.clone());
        
        // Update grid index
        self.grid_index.insert(spatial_point.clone());
        
        // Update R-tree
        self.rtree.insert(spatial_point.clone());
        
        // Update user location cache
        let user_location = self.user_locations.entry(user_id.to_string())
            .or_insert_with(|| UserLocation {
                current: spatial_point.clone(),
                history: Vec::new(),
                last_update: Instant::now(),
            });
        
        // Add to history
        user_location.history.push(user_location.current.clone());
        
        // Limit history size
        if user_location.history.len() > self.params.history_size {
            user_location.history.remove(0);
        }
        
        // Update current location
        user_location.current = spatial_point;
        user_location.last_update = Instant::now();
    }
    
    /// Query users within radius
    pub fn query_radius(&self, center: (f64, f64), radius_km: f64) -> Vec<String> {
        let start_time = Instant::now();
        
        // Convert radius to degrees (approximation)
        let radius_deg = radius_km / 111.0; // 1 degree ≈ 111 km
        
        let query_bounds = BoundingBox {
            min: (center.0 - radius_deg, center.1 - radius_deg),
            max: (center.0 + radius_deg, center.1 + radius_deg),
        };
        
        // Use grid index for fast rectangular query
        let candidates = self.grid_index.query_range(query_bounds);
        
        // Filter by exact distance
        let mut results = Vec::new();
        for candidate in candidates {
            let distance = self.haversine_distance(center, candidate.location);
            if distance <= radius_km {
                results.push(candidate.user_id);
            }
        }
        
        results
    }
    
    /// Query users within bounding box
    pub fn query_bbox(&self, bounds: BoundingBox) -> Vec<String> {
        // Use quad-tree for efficient range query
        let points = self.quadtree.query_range(bounds);
        points.into_iter().map(|p| p.user_id).collect()
    }
    
    /// Find nearest neighbors
    pub fn find_nearest_neighbors(&self, location: (f64, f64), k: usize) -> Vec<(String, f64)> {
        let mut all_distances = Vec::new();
        
        // Calculate distances to all users
        for (user_id, user_location) in &self.user_locations {
            let distance = self.haversine_distance(location, user_location.current.location);
            all_distances.push((user_id.clone(), distance));
        }
        
        // Sort by distance and take top k
        all_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_distances.truncate(k);
        
        all_distances
    }
    
    /// Get user location history
    pub fn get_user_history(&self, user_id: &str) -> Option<&Vec<SpatialPoint>> {
        self.user_locations.get(user_id).map(|ul| &ul.history)
    }
    
    /// Get cell statistics
    pub fn get_cell_stats(&self, cell_bounds: BoundingBox) -> CellStats {
        let points = self.grid_index.query_range(cell_bounds);
        
        let user_count = points.len();
        let avg_speed = if user_count > 0 {
            points.iter().map(|p| p.metadata.speed).sum::<f64>() / user_count as f64
        } else {
            0.0
        };
        
        let area = (cell_bounds.max.0 - cell_bounds.min.0) * (cell_bounds.max.1 - cell_bounds.min.1);
        let density = if area > 0.0 {
            user_count as f64 / area
        } else {
            0.0
        };
        
        CellStats {
            user_count,
            avg_speed,
            density,
            last_update: Instant::now(),
        }
    }
    
    /// Calculate Haversine distance
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

impl QuadTree {
    /// Create new quad-tree
    pub fn new(bounds: BoundingBox, max_depth: usize, max_points: usize) -> Self {
        Self {
            root: None,
            max_depth,
            max_points,
            bounds,
        }
    }
    
    /// Insert point into quad-tree
    pub fn insert(&mut self, point: SpatialPoint) {
        if self.root.is_none() {
            self.root = Some(Box::new(QuadNode {
                bounds: self.bounds,
                points: vec![point],
                children: None,
                depth: 0,
            }));
        } else {
            self.root.as_mut().unwrap().insert(point, self.max_depth, self.max_points);
        }
    }
    
    /// Query range in quad-tree
    pub fn query_range(&self, bounds: BoundingBox) -> Vec<SpatialPoint> {
        let mut results = Vec::new();
        
        if let Some(root) = &self.root {
            root.query_range(bounds, &mut results);
        }
        
        results
    }
}

impl QuadNode {
    /// Insert point into quad-tree node
    fn insert(&mut self, point: SpatialPoint, max_depth: usize, max_points: usize) {
        if !self.bounds.contains_point(point.location) {
            return;
        }
        
        if self.children.is_none() {
            self.points.push(point);
            
            // Split if necessary
            if self.points.len() > max_points && self.depth < max_depth {
                self.split(max_depth, max_points);
            }
        } else {
            // Insert into appropriate child
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
    
    /// Split quad-tree node
    fn split(&mut self, max_depth: usize, max_points: usize) {
        let mid_x = (self.bounds.min.0 + self.bounds.max.0) / 2.0;
        let mid_y = (self.bounds.min.1 + self.bounds.max.1) / 2.0;
        
        // Create four children
        let nw = QuadNode {
            bounds: BoundingBox {
                min: (mid_x, self.bounds.min.1),
                max: (self.bounds.max.0, mid_y),
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
        };
        
        let ne = QuadNode {
            bounds: BoundingBox {
                min: (mid_x, mid_y),
                max: self.bounds.max,
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
        };
        
        let sw = QuadNode {
            bounds: BoundingBox {
                min: self.bounds.min,
                max: (mid_x, mid_y),
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
        };
        
        let se = QuadNode {
            bounds: BoundingBox {
                min: (self.bounds.min.0, mid_y),
                max: (mid_x, self.bounds.max.1),
            },
            points: Vec::new(),
            children: None,
            depth: self.depth + 1,
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
    
    /// Query range in quad-tree node
    fn query_range(&self, bounds: BoundingBox, results: &mut Vec<SpatialPoint>) {
        if !self.bounds.intersects(bounds) {
            return;
        }
        
        // Check points in this node
        for point in &self.points {
            if bounds.contains_point(point.location) {
                results.push(point.clone());
            }
        }
        
        // Recursively check children
        if let Some(children) = &self.children {
            for child in children.iter() {
                child.query_range(bounds, results);
            }
        }
    }
}

impl GridIndex {
    /// Create new grid index
    pub fn new(bounds: BoundingBox, resolution: f64) -> Self {
        let width = ((bounds.max.0 - bounds.min.0) * 111000.0 / resolution).ceil() as usize;
        let height = ((bounds.max.1 - bounds.min.1) * 111000.0 / resolution).ceil() as usize;
        
        let mut cells = Vec::with_capacity(height);
        for i in 0..height {
            let mut row = Vec::with_capacity(width);
            for j in 0..width {
                let cell_bounds = BoundingBox {
                    min: (
                        bounds.min.0 + (i as f64 * resolution / 111000.0),
                        bounds.min.1 + (j as f64 * resolution / 111000.0),
                    ),
                    max: (
                        bounds.min.0 + ((i + 1) as f64 * resolution / 111000.0),
                        bounds.min.1 + ((j + 1) as f64 * resolution / 111000.0),
                    ),
                };
                
                row.push(GridCell {
                    id: i * width + j,
                    bounds: cell_bounds,
                    points: Vec::new(),
                    stats: CellStats {
                        user_count: 0,
                        avg_speed: 0.0,
                        density: 0.0,
                        last_update: Instant::now(),
                    },
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
        }
    }
    
    /// Insert point into grid
    pub fn insert(&mut self, point: SpatialPoint) {
        if let Some((row, col)) = self.get_cell_indices(point.location) {
            if row < self.height && col < self.width {
                self.cells[row][col].points.push(point);
                self.cells[row][col].stats.user_count = self.cells[row][col].points.len();
                self.cells[row][col].stats.last_update = Instant::now();
            }
        }
    }
    
    /// Query range in grid
    pub fn query_range(&self, bounds: BoundingBox) -> Vec<SpatialPoint> {
        let mut results = Vec::new();
        
        // Find cell range
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
    
    /// Get cell indices for location
    fn get_cell_indices(&self, location: (f64, f64)) -> Option<(usize, usize)> {
        if !self.bounds.contains_point(location) {
            return None;
        }
        
        let row = ((location.0 - self.bounds.min.0) * 111000.0 / self.resolution) as usize;
        let col = ((location.1 - self.bounds.min.1) * 111000.0 / self.resolution) as usize;
        
        Some((row, col))
    }
}

impl RTree {
    /// Create new R-tree
    pub fn new() -> Self {
        Self {
            root: None,
            max_entries: 16,
            min_entries: 4,
            height: 0,
        }
    }
    
    /// Insert point into R-tree
    pub fn insert(&mut self, point: SpatialPoint) {
        if self.root.is_none() {
            self.root = Some(Box::new(RNode {
                bounds: BoundingBox {
                    min: point.location,
                    max: point.location,
                },
                children: None,
                entries: Some(vec![point]),
                is_leaf: true,
            }));
            self.height = 1;
        } else {
            // Complex R-tree insertion would go here
            // For now, just add to root
            if let Some(root) = &mut self.root {
                if let Some(entries) = &mut root.entries {
                    entries.push(point);
                    root.bounds = root.bounds.union_point(point.location);
                }
            }
        }
    }
}

impl BoundingBox {
    /// Check if bounding box contains point
    pub fn contains_point(&self, point: (f64, f64)) -> bool {
        point.0 >= self.min.0 && point.0 <= self.max.0 &&
        point.1 >= self.min.1 && point.1 <= self.max.1
    }
    
    /// Check if bounding boxes intersect
    pub fn intersects(&self, other: BoundingBox) -> bool {
        self.min.0 <= other.max.0 && self.max.0 >= other.min.0 &&
        self.min.1 <= other.max.1 && self.max.1 >= other.min.1
    }
    
    /// Union with a point
    pub fn union_point(&self, point: (f64, f64)) -> BoundingBox {
        BoundingBox {
            min: (self.min.0.min(point.0), self.min.1.min(point.1)),
            max: (self.max.0.max(point.0), self.max.1.max(point.1)),
        }
    }
    
    /// Calculate area
    pub fn area(&self) -> f64 {
        (self.max.0 - self.min.0) * (self.max.1 - self.min.1)
    }
}

impl Default for SpatialIndexParams {
    fn default() -> Self {
        Self {
            grid_resolution: 100.0, // 100 meters
            max_quad_depth: 10,
            max_points_per_node: 16,
            history_size: 100,
            update_threshold: 10.0, // 10 meters
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_spatial_index_creation() {
        let index = SpatialIndex::new();
        assert_eq!(index.params.grid_resolution, 100.0);
        assert_eq!(index.params.max_quad_depth, 10);
    }
    
    #[test]
    fn test_bounding_box_operations() {
        let bbox = BoundingBox {
            min: (0.0, 0.0),
            max: (10.0, 10.0),
        };
        
        assert!(bbox.contains_point((5.0, 5.0)));
        assert!(!bbox.contains_point((15.0, 5.0)));
        
        let other = BoundingBox {
            min: (5.0, 5.0),
            max: (15.0, 15.0),
        };
        
        assert!(bbox.intersects(other));
        assert_eq!(bbox.area(), 100.0);
    }
    
    #[test]
    fn test_quadtree_operations() {
        let bounds = BoundingBox {
            min: (0.0, 0.0),
            max: (100.0, 100.0),
        };
        
        let mut quadtree = QuadTree::new(bounds, 5, 10);
        
        let point = SpatialPoint {
            user_id: "test_user".to_string(),
            location: (50.0, 50.0),
            timestamp: Instant::now(),
            metadata: SpatialMetadata {
                cell_id: "cell_001".to_string(),
                signal_strength: -80.0,
                speed: 5.0,
                direction: 0.0,
            },
        };
        
        quadtree.insert(point);
        
        let query_bounds = BoundingBox {
            min: (40.0, 40.0),
            max: (60.0, 60.0),
        };
        
        let results = quadtree.query_range(query_bounds);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].user_id, "test_user");
    }
    
    #[test]
    fn test_user_location_update() {
        let mut index = SpatialIndex::new();
        
        index.update_user_location("user1", (40.7128, -74.0060)); // New York
        index.update_user_location("user2", (34.0522, -118.2437)); // Los Angeles
        
        let ny_neighbors = index.find_nearest_neighbors((40.7128, -74.0060), 1);
        assert_eq!(ny_neighbors.len(), 1);
        assert_eq!(ny_neighbors[0].0, "user1");
        assert!(ny_neighbors[0].1 < 1.0); // Should be very close
    }
    
    #[test]
    fn test_radius_query() {
        let mut index = SpatialIndex::new();
        
        // Add users in NYC area
        index.update_user_location("user1", (40.7128, -74.0060));
        index.update_user_location("user2", (40.7589, -73.9851));
        index.update_user_location("user3", (40.6892, -74.0445));
        
        // Add user in LA (far away)
        index.update_user_location("user4", (34.0522, -118.2437));
        
        // Query within 10km of NYC
        let nearby_users = index.query_radius((40.7128, -74.0060), 10.0);
        
        // Should find NYC users but not LA user
        assert!(nearby_users.contains(&"user1".to_string()));
        assert!(nearby_users.contains(&"user2".to_string()));
        assert!(nearby_users.contains(&"user3".to_string()));
        assert!(!nearby_users.contains(&"user4".to_string()));
    }
}