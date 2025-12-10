//! Audit logger implementation

/// Audit logger stub
pub struct AuditLogger;

impl AuditLogger {
    /// Create new audit logger
    pub fn new() -> Self {
        Self
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}
