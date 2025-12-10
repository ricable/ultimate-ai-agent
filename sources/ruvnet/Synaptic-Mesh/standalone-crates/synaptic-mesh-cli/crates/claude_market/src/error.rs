//! Error types for the Claude Market

use thiserror::Error;

/// Result type alias for Market operations
pub type Result<T> = std::result::Result<T, MarketError>;

/// Market error types
#[derive(Error, Debug)]
pub enum MarketError {
    /// Database operation failed
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
    
    /// Serialization/deserialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Cryptographic operation failed
    #[error("Crypto error: {0}")]
    Crypto(String),
    
    /// Insufficient balance for operation
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance {
        /// Required amount
        required: u64,
        /// Available amount
        available: u64,
    },
    
    /// Invalid order parameters
    #[error("Invalid order: {0}")]
    InvalidOrder(String),
    
    /// Order not found
    #[error("Order not found: {0}")]
    OrderNotFound(String),
    
    /// Escrow operation failed
    #[error("Escrow error: {0}")]
    Escrow(String),
    
    /// Reputation check failed
    #[error("Reputation too low: {current} < {required}")]
    InsufficientReputation {
        /// Current reputation
        current: f64,
        /// Required reputation
        required: f64,
    },
    
    /// Transaction validation failed
    #[error("Invalid transaction: {0}")]
    InvalidTransaction(String),
    
    /// Network communication error
    #[error("Network error: {0}")]
    Network(String),
    
    /// Timeout occurred
    #[error("Operation timed out: {0}")]
    Timeout(String),
    
    /// Generic internal error
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<ed25519_dalek::SignatureError> for MarketError {
    fn from(e: ed25519_dalek::SignatureError) -> Self {
        MarketError::Crypto(e.to_string())
    }
}