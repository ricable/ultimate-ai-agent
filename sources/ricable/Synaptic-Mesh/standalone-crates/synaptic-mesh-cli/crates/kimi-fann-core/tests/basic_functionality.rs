//! Basic functionality tests using extern crate import

extern crate kimi_fann_core;

use kimi_fann_core::*;

#[test]
fn test_expert_creation() {
    let expert = MicroExpert::new(ExpertDomain::Coding);
    let result = expert.process("test");
    assert!(!result.is_empty());
    // Check for either "Coding" or "coding" (case insensitive)
    assert!(result.to_lowercase().contains("coding") || result.contains("programming"));
}

#[test]
fn test_router_creation() {
    let mut router = ExpertRouter::new();
    router.add_expert(MicroExpert::new(ExpertDomain::Mathematics));
    
    let result = router.route("test query");
    assert!(!result.is_empty());
}

#[test]
fn test_runtime_creation() {
    let config = ProcessingConfig::new();
    let mut runtime = KimiRuntime::new(config);
    
    let result = runtime.process("test query");
    assert!(!result.is_empty());
}

#[test]
fn test_version_available() {
    assert!(!VERSION.is_empty());
    assert!(VERSION.contains('.'));
}