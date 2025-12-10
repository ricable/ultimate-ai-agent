//! Event system for network state changes and monitoring

use std::collections::HashMap;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{GeoCoordinate, error::{RanError, RanResult}};

/// Network event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    // Cell events
    /// Cell activation
    CellActivated,
    /// Cell deactivation
    CellDeactivated,
    /// Cell configuration change
    CellConfigChanged,
    /// Cell performance degradation
    CellPerformanceDegraded,
    /// Cell failure detected
    CellFailure,
    /// Cell recovery completed
    CellRecovered,
    
    // UE events
    /// UE connection established
    UEConnected,
    /// UE disconnection
    UEDisconnected,
    /// UE handover initiated
    UEHandoverStarted,
    /// UE handover completed
    UEHandoverCompleted,
    /// UE handover failed
    UEHandoverFailed,
    /// UE location update
    UELocationUpdated,
    
    // Network events
    /// Network congestion detected
    NetworkCongestion,
    /// Network optimization triggered
    OptimizationTriggered,
    /// Network optimization completed
    OptimizationCompleted,
    /// Load balancing executed
    LoadBalancingExecuted,
    /// Interference detected
    InterferenceDetected,
    /// Coverage hole detected
    CoverageHoleDetected,
    
    // Performance events
    /// KPI threshold exceeded
    KPIThresholdExceeded,
    /// SLA violation
    SLAViolation,
    /// QoS degradation
    QoSDegradation,
    /// Throughput anomaly
    ThroughputAnomaly,
    /// Latency spike
    LatencySpike,
    
    // System events
    /// System maintenance started
    MaintenanceStarted,
    /// System maintenance completed
    MaintenanceCompleted,
    /// Software upgrade initiated
    SoftwareUpgrade,
    /// Configuration backup created
    ConfigurationBackup,
    /// Alarm raised
    AlarmRaised,
    /// Alarm cleared
    AlarmCleared,
    
    // Security events
    /// Security threat detected
    SecurityThreat,
    /// Unauthorized access attempt
    UnauthorizedAccess,
    /// Security policy violation
    SecurityPolicyViolation,
    
    // Resource events
    /// Resource allocation changed
    ResourceAllocationChanged,
    /// Resource limit exceeded
    ResourceLimitExceeded,
    /// Resource optimization completed
    ResourceOptimizationCompleted,
}

impl EventType {
    /// Get the category of this event type
    pub fn category(&self) -> EventCategory {
        match self {
            EventType::CellActivated | EventType::CellDeactivated | 
            EventType::CellConfigChanged | EventType::CellPerformanceDegraded |
            EventType::CellFailure | EventType::CellRecovered => EventCategory::Cell,
            
            EventType::UEConnected | EventType::UEDisconnected |
            EventType::UEHandoverStarted | EventType::UEHandoverCompleted |
            EventType::UEHandoverFailed | EventType::UELocationUpdated => EventCategory::UE,
            
            EventType::NetworkCongestion | EventType::OptimizationTriggered |
            EventType::OptimizationCompleted | EventType::LoadBalancingExecuted |
            EventType::InterferenceDetected | EventType::CoverageHoleDetected => EventCategory::Network,
            
            EventType::KPIThresholdExceeded | EventType::SLAViolation |
            EventType::QoSDegradation | EventType::ThroughputAnomaly |
            EventType::LatencySpike => EventCategory::Performance,
            
            EventType::MaintenanceStarted | EventType::MaintenanceCompleted |
            EventType::SoftwareUpgrade | EventType::ConfigurationBackup |
            EventType::AlarmRaised | EventType::AlarmCleared => EventCategory::System,
            
            EventType::SecurityThreat | EventType::UnauthorizedAccess |
            EventType::SecurityPolicyViolation => EventCategory::Security,
            
            EventType::ResourceAllocationChanged | EventType::ResourceLimitExceeded |
            EventType::ResourceOptimizationCompleted => EventCategory::Resource,
        }
    }

    /// Check if this event type is critical
    pub fn is_critical(&self) -> bool {
        match self {
            EventType::CellFailure | EventType::UEHandoverFailed |
            EventType::NetworkCongestion | EventType::SLAViolation |
            EventType::SecurityThreat | EventType::UnauthorizedAccess |
            EventType::CoverageHoleDetected => true,
            _ => false,
        }
    }

    /// Get the default severity for this event type
    pub fn default_severity(&self) -> EventSeverity {
        if self.is_critical() {
            EventSeverity::Critical
        } else {
            match self.category() {
                EventCategory::Security => EventSeverity::High,
                EventCategory::Performance => EventSeverity::Medium,
                EventCategory::Network => EventSeverity::Medium,
                EventCategory::Cell => EventSeverity::Medium,
                EventCategory::UE => EventSeverity::Low,
                EventCategory::System => EventSeverity::Low,
                EventCategory::Resource => EventSeverity::Medium,
            }
        }
    }
}

/// Event categories for grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventCategory {
    /// Cell-related events
    Cell,
    /// UE-related events
    UE,
    /// Network-wide events
    Network,
    /// Performance-related events
    Performance,
    /// System maintenance events
    System,
    /// Security events
    Security,
    /// Resource management events
    Resource,
}

/// Event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EventSeverity {
    /// Informational events
    Info,
    /// Low severity events
    Low,
    /// Medium severity events
    Medium,
    /// High severity events
    High,
    /// Critical events requiring immediate attention
    Critical,
}

impl EventSeverity {
    /// Get numeric value for severity (higher = more severe)
    pub fn numeric_value(&self) -> u8 {
        match self {
            EventSeverity::Info => 0,
            EventSeverity::Low => 1,
            EventSeverity::Medium => 2,
            EventSeverity::High => 3,
            EventSeverity::Critical => 4,
        }
    }

    /// Create from numeric value
    pub fn from_numeric(value: u8) -> Self {
        match value {
            0 => EventSeverity::Info,
            1 => EventSeverity::Low,
            2 => EventSeverity::Medium,
            3 => EventSeverity::High,
            _ => EventSeverity::Critical,
        }
    }
}

/// Network event representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkEvent {
    /// Unique event identifier
    pub id: Uuid,
    /// Event type
    pub event_type: EventType,
    /// Event severity
    pub severity: EventSeverity,
    /// Source element that generated the event
    pub source_id: Option<Uuid>,
    /// Source element type
    pub source_type: Option<String>,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event title/summary
    pub title: String,
    /// Detailed event description
    pub description: String,
    /// Event location (if applicable)
    pub location: Option<GeoCoordinate>,
    /// Event data payload
    pub data: HashMap<String, serde_json::Value>,
    /// Event tags for categorization
    pub tags: Vec<String>,
    /// Related event IDs
    pub related_events: Vec<Uuid>,
    /// Event acknowledged flag
    pub acknowledged: bool,
    /// Acknowledgment timestamp
    pub acknowledged_at: Option<DateTime<Utc>>,
    /// User who acknowledged the event
    pub acknowledged_by: Option<String>,
    /// Event resolution status
    pub resolved: bool,
    /// Resolution timestamp
    pub resolved_at: Option<DateTime<Utc>>,
    /// Resolution description
    pub resolution: Option<String>,
    /// Event priority (1-10, higher is more important)
    pub priority: u8,
    /// Expiration time for the event
    pub expires_at: Option<DateTime<Utc>>,
}

impl NetworkEvent {
    /// Create a new network event
    pub fn new(
        event_type: EventType,
        title: String,
        description: String,
    ) -> Self {
        Self {
            id: Uuid::new_v4(),
            event_type,
            severity: event_type.default_severity(),
            source_id: None,
            source_type: None,
            timestamp: Utc::now(),
            title,
            description,
            location: None,
            data: HashMap::new(),
            tags: Vec::new(),
            related_events: Vec::new(),
            acknowledged: false,
            acknowledged_at: None,
            acknowledged_by: None,
            resolved: false,
            resolved_at: None,
            resolution: None,
            priority: 5, // Medium priority
            expires_at: None,
        }
    }

    /// Create a new event with source information
    pub fn with_source(
        event_type: EventType,
        title: String,
        description: String,
        source_id: Uuid,
        source_type: String,
    ) -> Self {
        let mut event = Self::new(event_type, title, description);
        event.source_id = Some(source_id);
        event.source_type = Some(source_type);
        event
    }

    /// Set event severity
    pub fn with_severity(mut self, severity: EventSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set event location
    pub fn with_location(mut self, location: GeoCoordinate) -> Self {
        self.location = Some(location);
        self
    }

    /// Set event priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10).max(1);
        self
    }

    /// Add event data
    pub fn add_data<K, V>(&mut self, key: K, value: V) -> RanResult<()>
    where
        K: Into<String>,
        V: Serialize,
    {
        let json_value = serde_json::to_value(value)
            .map_err(|e| RanError::Serialization(e))?;
        self.data.insert(key.into(), json_value);
        Ok(())
    }

    /// Get typed data value
    pub fn get_data<T>(&self, key: &str) -> RanResult<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        match self.data.get(key) {
            Some(value) => {
                let typed_value = serde_json::from_value(value.clone())
                    .map_err(|e| RanError::Serialization(e))?;
                Ok(Some(typed_value))
            }
            None => Ok(None),
        }
    }

    /// Add a tag
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Add related event
    pub fn add_related_event(&mut self, event_id: Uuid) {
        if !self.related_events.contains(&event_id) {
            self.related_events.push(event_id);
        }
    }

    /// Acknowledge the event
    pub fn acknowledge(&mut self, acknowledged_by: String) {
        self.acknowledged = true;
        self.acknowledged_at = Some(Utc::now());
        self.acknowledged_by = Some(acknowledged_by);
    }

    /// Resolve the event
    pub fn resolve(&mut self, resolution: String) {
        self.resolved = true;
        self.resolved_at = Some(Utc::now());
        self.resolution = Some(resolution);
    }

    /// Check if event is expired
    pub fn is_expired(&self) -> bool {
        self.expires_at.map_or(false, |expires| Utc::now() > expires)
    }

    /// Check if event is active (not acknowledged and not resolved)
    pub fn is_active(&self) -> bool {
        !self.acknowledged && !self.resolved && !self.is_expired()
    }

    /// Get event age in seconds
    pub fn age_seconds(&self) -> i64 {
        (Utc::now() - self.timestamp).num_seconds()
    }

    /// Check if event needs escalation
    pub fn needs_escalation(&self, escalation_threshold_seconds: i64) -> bool {
        self.is_active() && 
        self.severity >= EventSeverity::High &&
        self.age_seconds() > escalation_threshold_seconds
    }

    /// Get event impact score
    pub fn impact_score(&self) -> f64 {
        let severity_weight = match self.severity {
            EventSeverity::Info => 0.1,
            EventSeverity::Low => 0.3,
            EventSeverity::Medium => 0.5,
            EventSeverity::High => 0.8,
            EventSeverity::Critical => 1.0,
        };
        
        let priority_weight = (self.priority as f64) / 10.0;
        let age_factor = if self.is_active() {
            (self.age_seconds() as f64 / 3600.0).min(24.0) / 24.0 // Max 24 hours
        } else {
            0.0
        };
        
        (severity_weight * 0.5 + priority_weight * 0.3 + age_factor * 0.2) * 100.0
    }
}

/// Event filter for querying events
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventFilter {
    /// Filter by event types
    pub event_types: Option<Vec<EventType>>,
    /// Filter by severity levels
    pub severities: Option<Vec<EventSeverity>>,
    /// Filter by source IDs
    pub source_ids: Option<Vec<Uuid>>,
    /// Filter by source types
    pub source_types: Option<Vec<String>>,
    /// Filter by tags
    pub tags: Option<Vec<String>>,
    /// Filter by time range
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Filter by acknowledged status
    pub acknowledged: Option<bool>,
    /// Filter by resolved status
    pub resolved: Option<bool>,
    /// Filter by minimum severity
    pub min_severity: Option<EventSeverity>,
    /// Filter by location bounds
    pub location_bounds: Option<(GeoCoordinate, f64)>, // (center, radius_km)
    /// Filter by minimum priority
    pub min_priority: Option<u8>,
    /// Include expired events
    pub include_expired: bool,
}

impl EventFilter {
    /// Create a new empty filter
    pub fn new() -> Self {
        Self::default()
    }

    /// Filter by event type
    pub fn with_event_type(mut self, event_type: EventType) -> Self {
        self.event_types = Some(vec![event_type]);
        self
    }

    /// Filter by multiple event types
    pub fn with_event_types(mut self, event_types: Vec<EventType>) -> Self {
        self.event_types = Some(event_types);
        self
    }

    /// Filter by minimum severity
    pub fn with_min_severity(mut self, severity: EventSeverity) -> Self {
        self.min_severity = Some(severity);
        self
    }

    /// Filter by time range
    pub fn with_time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.time_range = Some((start, end));
        self
    }

    /// Filter active events only
    pub fn active_only(mut self) -> Self {
        self.acknowledged = Some(false);
        self.resolved = Some(false);
        self.include_expired = false;
        self
    }

    /// Check if an event matches this filter
    pub fn matches(&self, event: &NetworkEvent) -> bool {
        // Check event types
        if let Some(ref types) = self.event_types {
            if !types.contains(&event.event_type) {
                return false;
            }
        }

        // Check severities
        if let Some(ref severities) = self.severities {
            if !severities.contains(&event.severity) {
                return false;
            }
        }

        // Check minimum severity
        if let Some(min_sev) = self.min_severity {
            if event.severity < min_sev {
                return false;
            }
        }

        // Check source IDs
        if let Some(ref source_ids) = self.source_ids {
            if let Some(source_id) = event.source_id {
                if !source_ids.contains(&source_id) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check source types
        if let Some(ref source_types) = self.source_types {
            if let Some(ref source_type) = event.source_type {
                if !source_types.contains(source_type) {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check tags
        if let Some(ref filter_tags) = self.tags {
            if !filter_tags.iter().any(|tag| event.tags.contains(tag)) {
                return false;
            }
        }

        // Check time range
        if let Some((start, end)) = self.time_range {
            if event.timestamp < start || event.timestamp > end {
                return false;
            }
        }

        // Check acknowledged status
        if let Some(ack) = self.acknowledged {
            if event.acknowledged != ack {
                return false;
            }
        }

        // Check resolved status
        if let Some(res) = self.resolved {
            if event.resolved != res {
                return false;
            }
        }

        // Check minimum priority
        if let Some(min_priority) = self.min_priority {
            if event.priority < min_priority {
                return false;
            }
        }

        // Check location bounds
        if let Some((center, radius_km)) = self.location_bounds {
            if let Some(event_location) = event.location {
                let distance = center.distance_to(&event_location) / 1000.0; // Convert to km
                if distance > radius_km {
                    return false;
                }
            } else {
                return false;
            }
        }

        // Check expired status
        if !self.include_expired && event.is_expired() {
            return false;
        }

        true
    }
}

/// Event statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStatistics {
    /// Total number of events
    pub total_events: usize,
    /// Events by severity
    pub by_severity: HashMap<EventSeverity, usize>,
    /// Events by type
    pub by_type: HashMap<EventType, usize>,
    /// Events by category
    pub by_category: HashMap<EventCategory, usize>,
    /// Active events count
    pub active_events: usize,
    /// Average event age in hours
    pub avg_age_hours: f64,
    /// Events requiring escalation
    pub escalation_required: usize,
    /// Resolution rate percentage
    pub resolution_rate: f64,
}

impl EventStatistics {
    /// Calculate statistics from a list of events
    pub fn from_events(events: &[NetworkEvent]) -> Self {
        let total_events = events.len();
        let mut by_severity = HashMap::new();
        let mut by_type = HashMap::new();
        let mut by_category = HashMap::new();
        let mut active_events = 0;
        let mut total_age_seconds = 0i64;
        let mut escalation_required = 0;
        let mut resolved_count = 0;

        for event in events {
            // Count by severity
            *by_severity.entry(event.severity).or_insert(0) += 1;
            
            // Count by type
            *by_type.entry(event.event_type).or_insert(0) += 1;
            
            // Count by category
            *by_category.entry(event.event_type.category()).or_insert(0) += 1;
            
            // Count active events
            if event.is_active() {
                active_events += 1;
            }
            
            // Sum age
            total_age_seconds += event.age_seconds();
            
            // Count escalation required
            if event.needs_escalation(3600) { // 1 hour threshold
                escalation_required += 1;
            }
            
            // Count resolved events
            if event.resolved {
                resolved_count += 1;
            }
        }

        let avg_age_hours = if total_events > 0 {
            (total_age_seconds as f64) / (total_events as f64) / 3600.0
        } else {
            0.0
        };

        let resolution_rate = if total_events > 0 {
            (resolved_count as f64) / (total_events as f64) * 100.0
        } else {
            0.0
        };

        Self {
            total_events,
            by_severity,
            by_type,
            by_category,
            active_events,
            avg_age_hours,
            escalation_required,
            resolution_rate,
        }
    }
}

/// Event handler trait for processing events
pub trait EventHandler: Send + Sync {
    /// Handle a network event
    fn handle_event(&self, event: &NetworkEvent) -> RanResult<()>;
    
    /// Get handler name
    fn name(&self) -> &str;
    
    /// Check if handler can process this event type
    fn can_handle(&self, event_type: EventType) -> bool;
}

/// Event bus for managing event distribution
pub struct EventBus {
    /// Registered event handlers
    handlers: Vec<Box<dyn EventHandler>>,
    /// Event filters for handlers
    handler_filters: HashMap<String, EventFilter>,
}

impl EventBus {
    /// Create a new event bus
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
            handler_filters: HashMap::new(),
        }
    }

    /// Register an event handler
    pub fn register_handler(&mut self, handler: Box<dyn EventHandler>) {
        self.handlers.push(handler);
    }

    /// Register an event handler with filter
    pub fn register_handler_with_filter(
        &mut self,
        handler: Box<dyn EventHandler>,
        filter: EventFilter,
    ) {
        let handler_name = handler.name().to_string();
        self.handler_filters.insert(handler_name, filter);
        self.handlers.push(handler);
    }

    /// Publish an event to all matching handlers
    pub fn publish(&self, event: &NetworkEvent) -> Vec<RanResult<()>> {
        let mut results = Vec::new();
        
        for handler in &self.handlers {
            // Check if handler can handle this event type
            if !handler.can_handle(event.event_type) {
                continue;
            }
            
            // Check filter if configured
            if let Some(filter) = self.handler_filters.get(handler.name()) {
                if !filter.matches(event) {
                    continue;
                }
            }
            
            // Handle the event
            let result = handler.handle_event(event);
            results.push(result);
        }
        
        results
    }

    /// Get number of registered handlers
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_creation() {
        let event = NetworkEvent::new(
            EventType::CellFailure,
            "Cell Failed".to_string(),
            "Cell XYZ has failed".to_string(),
        );

        assert_eq!(event.event_type, EventType::CellFailure);
        assert_eq!(event.severity, EventSeverity::Critical);
        assert!(event.is_active());
        assert!(!event.is_expired());
    }

    #[test]
    fn test_event_with_source() {
        let source_id = Uuid::new_v4();
        let event = NetworkEvent::with_source(
            EventType::UEConnected,
            "UE Connected".to_string(),
            "UE connected to cell".to_string(),
            source_id,
            "Cell".to_string(),
        );

        assert_eq!(event.source_id, Some(source_id));
        assert_eq!(event.source_type, Some("Cell".to_string()));
    }

    #[test]
    fn test_event_data() {
        let mut event = NetworkEvent::new(
            EventType::KPIThresholdExceeded,
            "Throughput Threshold".to_string(),
            "Throughput exceeded threshold".to_string(),
        );

        event.add_data("throughput", 150.0).unwrap();
        event.add_data("threshold", 100.0).unwrap();

        let throughput: f64 = event.get_data("throughput").unwrap().unwrap();
        assert_eq!(throughput, 150.0);
    }

    #[test]
    fn test_event_acknowledgment() {
        let mut event = NetworkEvent::new(
            EventType::AlarmRaised,
            "Test Alarm".to_string(),
            "Test alarm description".to_string(),
        );

        assert!(event.is_active());
        
        event.acknowledge("admin".to_string());
        assert!(event.acknowledged);
        assert!(!event.is_active());
    }

    #[test]
    fn test_event_filter() {
        let filter = EventFilter::new()
            .with_event_type(EventType::CellFailure)
            .with_min_severity(EventSeverity::High)
            .active_only();

        let event1 = NetworkEvent::new(
            EventType::CellFailure,
            "Cell Failed".to_string(),
            "Description".to_string(),
        );

        let mut event2 = NetworkEvent::new(
            EventType::UEConnected,
            "UE Connected".to_string(),
            "Description".to_string(),
        );
        event2.acknowledge("admin".to_string());

        assert!(filter.matches(&event1));
        assert!(!filter.matches(&event2));
    }

    #[test]
    fn test_event_statistics() {
        let events = vec![
            NetworkEvent::new(
                EventType::CellFailure,
                "Cell Failed".to_string(),
                "Description".to_string(),
            ),
            NetworkEvent::new(
                EventType::UEConnected,
                "UE Connected".to_string(),
                "Description".to_string(),
            ),
        ];

        let stats = EventStatistics::from_events(&events);
        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.active_events, 2);
        assert!(stats.by_severity.contains_key(&EventSeverity::Critical));
        assert!(stats.by_severity.contains_key(&EventSeverity::Low));
    }

    #[test]
    fn test_event_severity_ordering() {
        assert!(EventSeverity::Critical > EventSeverity::High);
        assert!(EventSeverity::High > EventSeverity::Medium);
        assert!(EventSeverity::Medium > EventSeverity::Low);
        assert!(EventSeverity::Low > EventSeverity::Info);
    }

    #[test]
    fn test_event_impact_score() {
        let critical_event = NetworkEvent::new(
            EventType::CellFailure,
            "Critical Failure".to_string(),
            "Description".to_string(),
        ).with_priority(10);

        let info_event = NetworkEvent::new(
            EventType::UEConnected,
            "UE Connected".to_string(),
            "Description".to_string(),
        ).with_priority(1);

        assert!(critical_event.impact_score() > info_event.impact_score());
    }

    #[test]
    fn test_event_categories() {
        assert_eq!(EventType::CellFailure.category(), EventCategory::Cell);
        assert_eq!(EventType::UEConnected.category(), EventCategory::UE);
        assert_eq!(EventType::NetworkCongestion.category(), EventCategory::Network);
        assert_eq!(EventType::SecurityThreat.category(), EventCategory::Security);
    }
}