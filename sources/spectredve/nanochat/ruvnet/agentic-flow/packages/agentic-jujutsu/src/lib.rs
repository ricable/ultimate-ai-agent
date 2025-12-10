//! # Agentic-Jujutsu
//!
//! N-API native Jujutsu VCS wrapper for AI agent collaboration and learning.
//!
//! ## Features
//!
//! - Zero-dependency installation (jj binary embedded)
//! - Zero-copy jj CLI operations
//! - Operation log parsing and tracking
//! - Conflict detection and resolution
//! - N-API bindings for JavaScript/TypeScript
//! - AgentDB integration
//! - MCP protocol support

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod agent_coordination;
pub mod agentdb_sync;
pub mod config;
pub mod crypto;
pub mod error;
pub mod hooks;
pub mod mcp;
pub mod native;
pub mod operations;
pub mod quantum_signing;
pub mod reasoning_bank;
pub mod types;
pub mod wrapper;

// Re-exports
pub use agent_coordination::{AgentConflict, AgentCoordination, AgentStats, CoordinationStats};
pub use agentdb_sync::{AgentDBEpisode, AgentDBSync, TaskStatistics};
pub use config::JJConfig;
pub use crypto::{generate_signing_keypair, OperationSignature, SigningKeypair as MLDSAKeypair};
pub use error::{JJError, Result};
pub use hooks::{HookContext, HookEventType, JJHookEvent, JJHooksIntegration};
pub use operations::{JJOperation, JJOperationLog, OperationType};
pub use quantum_signing::{CommitSignature, QuantumSigner, SigningKeypair};
pub use reasoning_bank::{DecisionSuggestion, LearningStats, Pattern, ReasoningBank, Trajectory};
pub use types::{JJBranch, JJCommit, JJConflict, JJResult};
pub use wrapper::JJWrapper;

/// Version of the agentic-jujutsu crate
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
