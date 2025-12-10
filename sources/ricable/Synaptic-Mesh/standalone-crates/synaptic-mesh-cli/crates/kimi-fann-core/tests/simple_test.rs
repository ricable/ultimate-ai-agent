//! Simple integration test to verify basic functionality

#[test]
fn test_basic_functionality() {
    use kimi_fann_core::{MicroExpert, ExpertDomain, VERSION};
    
    // Test expert creation
    let expert = MicroExpert::new(ExpertDomain::Coding);
    let result = expert.process("test query");
    assert!(!result.is_empty());
    assert!(result.contains("Coding"));
    
    // Test version
    assert!(!VERSION.is_empty());
}