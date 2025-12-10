//! gRPC services implementation

/// gRPC service stub
pub struct GrpcService;

impl GrpcService {
    /// Create new gRPC service
    pub fn new() -> Self {
        Self
    }
}

impl Default for GrpcService {
    fn default() -> Self {
        Self::new()
    }
}
