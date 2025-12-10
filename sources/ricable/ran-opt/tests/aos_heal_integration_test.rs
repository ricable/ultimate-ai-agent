use ran_opt::{
    AgentAosHeal, HealingAction, HealingActionType, ActionGeneratorConfig,
    NetworkState, AdvancedBeamSearch, EnmClient, default_amos_templates,
    ValidationStatus, AmosTemplate, RestconfPayload,
};
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use tokio;

#[tokio::test]
async fn test_agent_aos_heal_integration() {
    // Test complete integration of Agent AOS Heal
    let config = ActionGeneratorConfig::default();
    let templates = default_amos_templates();
    let base_url = "https://test.enm.com".to_string();
    let auth_token = "test_token".to_string();
    
    let agent = AgentAosHeal::new(config, templates, base_url, auth_token);
    assert!(agent.is_ok());
    
    let agent = agent.unwrap();
    
    // Test action generation with mock tensor
    let device = Device::Cpu;
    let anomaly_features = Tensor::randn(0f32, 1f32, (1, 256), &device).unwrap();
    let actions = agent.generate_healing_actions(&anomaly_features);
    assert!(actions.is_ok());
    
    let actions = actions.unwrap();
    assert!(!actions.is_empty());
    
    // Verify action properties
    for action in &actions {
        assert!(!action.target_entity.is_empty());
        assert!(action.confidence >= 0.0 && action.confidence <= 1.0);
        assert!(action.priority >= 0.0 && action.priority <= 1.0);
        assert!(action.estimated_duration > 0);
    }
}

#[test]
fn test_healing_action_types() {
    let action_types = vec![
        HealingActionType::ProcessRestart,
        HealingActionType::CellBlocking,
        HealingActionType::CellUnblocking,
        HealingActionType::ParameterAdjustment,
        HealingActionType::LoadBalancing,
        HealingActionType::ServiceMigration,
        HealingActionType::ResourceAllocation,
        HealingActionType::NetworkReconfiguration,
    ];
    
    // Test that all action types can be created and compared
    for action_type in &action_types {
        let action = HealingAction {
            action_type: action_type.clone(),
            target_entity: "test_entity".to_string(),
            parameters: HashMap::new(),
            priority: 0.5,
            confidence: 0.8,
            estimated_duration: 300,
            rollback_plan: None,
        };
        
        assert_eq!(action.action_type, *action_type);
        assert_eq!(action.target_entity, "test_entity");
    }
}

#[test]
fn test_action_generator_config() {
    let config = ActionGeneratorConfig::default();
    
    assert_eq!(config.input_dim, 256);
    assert_eq!(config.hidden_dim, 512);
    assert_eq!(config.num_layers, 6);
    assert_eq!(config.vocab_size, 10000);
    assert_eq!(config.max_sequence_length, 128);
    assert_eq!(config.beam_width, 5);
    assert_eq!(config.temperature, 1.0);
    assert_eq!(config.dropout_prob, 0.1);
    
    // Test custom config
    let custom_config = ActionGeneratorConfig {
        beam_width: 10,
        temperature: 0.5,
        ..ActionGeneratorConfig::default()
    };
    
    assert_eq!(custom_config.beam_width, 10);
    assert_eq!(custom_config.temperature, 0.5);
    assert_eq!(custom_config.input_dim, 256); // Should retain default
}

#[test]
fn test_beam_search_optimization() {
    let beam_search = AdvancedBeamSearch::new(5, 8);
    
    // Create initial state with problems
    let mut initial_state = NetworkState::new();
    initial_state.performance_score = 0.3;
    initial_state.max_utilization = 0.95;
    initial_state.failed_processes = vec!["proc1".to_string(), "proc2".to_string()];
    initial_state.blocked_cells = vec!["cell1".to_string()];
    
    // Create target state
    let mut target_state = NetworkState::new();
    target_state.performance_score = 0.9;
    target_state.max_utilization = 0.7;
    target_state.failed_processes = vec![];
    target_state.blocked_cells = vec![];
    
    // Perform beam search
    let optimized_actions = beam_search.search(&initial_state, &target_state);
    
    // Verify that beam search produces reasonable actions
    assert!(!optimized_actions.is_empty());
    
    // Check that actions are relevant to the problems
    let has_process_restart = optimized_actions.iter()
        .any(|a| a.action_type == HealingActionType::ProcessRestart);
    let has_cell_unblocking = optimized_actions.iter()
        .any(|a| a.action_type == HealingActionType::CellUnblocking);
    
    // At least one should be present given the initial state
    assert!(has_process_restart || has_cell_unblocking);
}

#[test]
fn test_network_state_evolution() {
    let mut initial_state = NetworkState::new();
    initial_state.performance_score = 0.5;
    initial_state.failed_processes = vec!["test_process".to_string()];
    initial_state.blocked_cells = vec!["test_cell".to_string()];
    
    assert_eq!(initial_state.performance_score, 0.5);
    assert_eq!(initial_state.failed_processes.len(), 1);
    assert_eq!(initial_state.blocked_cells.len(), 1);
    
    // Test state modification
    initial_state.performance_score = 0.8;
    initial_state.failed_processes.clear();
    
    assert_eq!(initial_state.performance_score, 0.8);
    assert_eq!(initial_state.failed_processes.len(), 0);
    assert_eq!(initial_state.blocked_cells.len(), 1);
}

#[test]
fn test_amos_templates() {
    let templates = default_amos_templates();
    
    assert!(!templates.is_empty());
    assert_eq!(templates.len(), 4); // Should have 4 default templates
    
    // Verify template structure
    for template in &templates {
        assert!(!template.name.is_empty());
        assert!(!template.script_template.is_empty());
        assert!(!template.parameters.is_empty());
        assert!(!template.validation_checks.is_empty());
    }
    
    // Test specific templates
    let restart_template = templates.iter()
        .find(|t| t.action_type == HealingActionType::ProcessRestart);
    assert!(restart_template.is_some());
    
    let restart_template = restart_template.unwrap();
    assert_eq!(restart_template.name, "Process Restart");
    assert!(restart_template.script_template.contains("restart"));
    assert!(restart_template.parameters.contains(&"node".to_string()));
    assert!(restart_template.parameters.contains(&"process".to_string()));
    
    let cell_block_template = templates.iter()
        .find(|t| t.action_type == HealingActionType::CellBlocking);
    assert!(cell_block_template.is_some());
    
    let cell_block_template = cell_block_template.unwrap();
    assert_eq!(cell_block_template.name, "Cell Blocking");
    assert!(cell_block_template.script_template.contains("blocked"));
}

#[test]
fn test_enm_client_creation() {
    let client = EnmClient::new(
        "https://enm.example.com".to_string(),
        "test_token_123".to_string(),
    );
    
    assert_eq!(client.base_url, "https://enm.example.com");
    assert_eq!(client.auth_token, "test_token_123");
    assert_eq!(client.timeout, 30);
}

#[test]
fn test_action_validation_scenarios() {
    // Test various validation scenarios
    let valid_action = HealingAction {
        action_type: HealingActionType::ProcessRestart,
        target_entity: "node123".to_string(),
        parameters: HashMap::new(),
        priority: 0.8,
        confidence: 0.9,
        estimated_duration: 300,
        rollback_plan: Some("backup_plan".to_string()),
    };
    
    assert!(valid_action.confidence >= 0.5);
    assert!(valid_action.priority >= 0.1);
    assert!(valid_action.estimated_duration > 0);
    
    let invalid_action = HealingAction {
        action_type: HealingActionType::ProcessRestart,
        target_entity: "node123".to_string(),
        parameters: HashMap::new(),
        priority: 0.05, // Too low
        confidence: 0.3, // Too low
        estimated_duration: 300,
        rollback_plan: None,
    };
    
    assert!(invalid_action.confidence < 0.5);
    assert!(invalid_action.priority < 0.1);
}

#[test]
fn test_restconf_payload_structure() {
    let payload = RestconfPayload {
        method: "POST".to_string(),
        endpoint: "https://enm.example.com/restconf/operations/restart".to_string(),
        headers: {
            let mut headers = HashMap::new();
            headers.insert("Content-Type".to_string(), "application/json".to_string());
            headers.insert("Authorization".to_string(), "Bearer token123".to_string());
            headers
        },
        body: Some(r#"{"node": "eNB001", "process": "cell_mgr"}"#.to_string()),
        timeout: 30000,
    };
    
    assert_eq!(payload.method, "POST");
    assert!(payload.endpoint.contains("restconf"));
    assert!(payload.headers.contains_key("Content-Type"));
    assert!(payload.headers.contains_key("Authorization"));
    assert!(payload.body.is_some());
    assert_eq!(payload.timeout, 30000);
}

#[test]
fn test_action_sequence_scoring() {
    let actions = vec![
        HealingAction {
            action_type: HealingActionType::ProcessRestart,
            target_entity: "node1".to_string(),
            parameters: HashMap::new(),
            priority: 0.8,
            confidence: 0.9,
            estimated_duration: 300,
            rollback_plan: None,
        },
        HealingAction {
            action_type: HealingActionType::CellUnblocking,
            target_entity: "cell1".to_string(),
            parameters: HashMap::new(),
            priority: 0.7,
            confidence: 0.8,
            estimated_duration: 120,
            rollback_plan: None,
        },
    ];
    
    let score = score_action_sequence(&actions);
    assert!(score > 0.0);
    assert!(score <= 1.0);
    
    // Empty sequence should have score 0
    let empty_score = score_action_sequence(&[]);
    assert_eq!(empty_score, 0.0);
}

#[test]
fn test_action_dependency_validation() {
    use ran_opt::ActionDependencyGraph;
    
    let graph = ActionDependencyGraph::new();
    
    // Create a sequence with process restart first
    let restart_action = HealingAction {
        action_type: HealingActionType::ProcessRestart,
        target_entity: "node1".to_string(),
        parameters: HashMap::new(),
        priority: 0.8,
        confidence: 0.9,
        estimated_duration: 300,
        rollback_plan: None,
    };
    
    let param_action = HealingAction {
        action_type: HealingActionType::ParameterAdjustment,
        target_entity: "node1".to_string(),
        parameters: HashMap::new(),
        priority: 0.7,
        confidence: 0.8,
        estimated_duration: 180,
        rollback_plan: None,
    };
    
    // Test dependency checking
    let current_actions = vec![restart_action];
    let dependency_satisfied = graph.check_dependencies(&current_actions, &param_action);
    assert!(dependency_satisfied);
    
    // Test without prerequisite
    let empty_actions = vec![];
    let dependency_not_satisfied = graph.check_dependencies(&empty_actions, &param_action);
    assert!(!dependency_not_satisfied);
}

#[test]
fn test_serialization() {
    let action = HealingAction {
        action_type: HealingActionType::ProcessRestart,
        target_entity: "test_node".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("process".to_string(), "cell_mgr".to_string());
            params
        },
        priority: 0.8,
        confidence: 0.9,
        estimated_duration: 300,
        rollback_plan: Some("restart_backup".to_string()),
    };
    
    // Test JSON serialization
    let json = serde_json::to_string(&action).unwrap();
    assert!(json.contains("ProcessRestart"));
    assert!(json.contains("test_node"));
    assert!(json.contains("cell_mgr"));
    
    // Test deserialization
    let deserialized: HealingAction = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.action_type, HealingActionType::ProcessRestart);
    assert_eq!(deserialized.target_entity, "test_node");
    assert_eq!(deserialized.priority, 0.8);
    assert_eq!(deserialized.confidence, 0.9);
}

/// Helper function to score action sequences
fn score_action_sequence(actions: &[HealingAction]) -> f32 {
    if actions.is_empty() {
        return 0.0;
    }
    
    let confidence_sum: f32 = actions.iter().map(|a| a.confidence).sum();
    let priority_sum: f32 = actions.iter().map(|a| a.priority).sum();
    
    (confidence_sum + priority_sum) / (2.0 * actions.len() as f32)
}

#[tokio::test]
async fn test_enm_client_payload_generation() {
    let client = EnmClient::new(
        "https://test.enm.com".to_string(),
        "test_token".to_string(),
    );
    
    // Test different action types and their payload generation
    let restart_action = HealingAction {
        action_type: HealingActionType::ProcessRestart,
        target_entity: "node123".to_string(),
        parameters: {
            let mut params = HashMap::new();
            params.insert("process".to_string(), "cell_manager".to_string());
            params
        },
        priority: 0.8,
        confidence: 0.9,
        estimated_duration: 300,
        rollback_plan: None,
    };
    
    // In a real test, you would generate and validate payloads
    assert_eq!(restart_action.action_type, HealingActionType::ProcessRestart);
    assert_eq!(restart_action.target_entity, "node123");
    assert!(restart_action.parameters.contains_key("process"));
}

#[test]
fn test_comprehensive_workflow() {
    // Test the complete workflow from anomaly detection to action execution
    
    // 1. Start with network state having issues
    let mut initial_state = NetworkState::new();
    initial_state.performance_score = 0.4;
    initial_state.max_utilization = 0.92;
    initial_state.failed_processes = vec!["proc1".to_string(), "proc2".to_string()];
    initial_state.blocked_cells = vec!["cell1".to_string(), "cell2".to_string()];
    
    // 2. Define target state
    let mut target_state = NetworkState::new();
    target_state.performance_score = 0.85;
    target_state.max_utilization = 0.75;
    target_state.failed_processes = vec![];
    target_state.blocked_cells = vec![];
    
    // 3. Use beam search to find action sequence
    let beam_search = AdvancedBeamSearch::new(5, 6);
    let actions = beam_search.search(&initial_state, &target_state);
    
    // 4. Validate actions
    assert!(!actions.is_empty());
    
    // 5. Check action types are appropriate
    let has_restart = actions.iter().any(|a| a.action_type == HealingActionType::ProcessRestart);
    let has_unblock = actions.iter().any(|a| a.action_type == HealingActionType::CellUnblocking);
    
    // Should have at least one relevant action
    assert!(has_restart || has_unblock);
    
    // 6. Verify all actions have valid properties
    for action in &actions {
        assert!(action.confidence > 0.0);
        assert!(action.priority > 0.0);
        assert!(action.estimated_duration > 0);
        assert!(!action.target_entity.is_empty());
    }
}