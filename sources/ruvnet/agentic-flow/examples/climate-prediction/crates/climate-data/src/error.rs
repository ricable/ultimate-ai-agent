use thiserror::Error;

/// Result type for climate data operations
pub type Result<T> = std::result::Result<T, ClimateDataError>;

/// Errors that can occur during climate data operations
#[derive(Error, Debug)]
pub enum ClimateDataError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("API error: {status_code} - {message}")]
    ApiError {
        status_code: u16,
        message: String,
    },

    #[error("Data validation failed: {0}")]
    ValidationError(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitError(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Cloud storage error: {0}")]
    CloudStorageError(String),

    #[error("Invalid coordinates: lat={lat}, lon={lon}")]
    InvalidCoordinates { lat: f64, lon: f64 },

    #[error("Invalid date range: start={start}, end={end}")]
    InvalidDateRange { start: String, end: String },

    #[error("Missing API key for provider: {0}")]
    MissingApiKey(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

impl ClimateDataError {
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ClimateDataError::HttpError(_)
                | ClimateDataError::ApiError {
                    status_code: 500..=599,
                    ..
                }
                | ClimateDataError::RateLimitError(_)
        )
    }
}
