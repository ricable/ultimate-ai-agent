//! Property-based tests for operations

use agentic_jujutsu::{JJOperation, JJOperationLog, OperationType};
use proptest::prelude::*;

// Strategy for generating OperationType
prop_compose! {
    fn operation_type_strategy()(op_type in 0u8..8) -> OperationType {
        match op_type {
            0 => OperationType::Commit,
            1 => OperationType::Rebase,
            2 => OperationType::Merge,
            3 => OperationType::Describe,
            4 => OperationType::Branch,
            5 => OperationType::Checkout,
            6 => OperationType::Restore,
            _ => OperationType::Unknown,
        }
    }
}

// Strategy for generating JJOperation
prop_compose! {
    fn operation_strategy()(
        id in "[a-z0-9]{8,16}",
        user in "[a-z0-9]+@[a-z]+\\.[a-z]{2,4}",
        description in ".*{0,100}",
    ) -> JJOperation {
        JJOperation::new(id, user, description)
    }
}

proptest! {
    #[test]
    fn test_operation_id_preserves_format(id in "[a-z0-9]{8,16}") {
        let op = JJOperation::new(id.clone(), "user@test.com".to_string(), "desc".to_string());
        assert_eq!(op.id, id);
    }

    #[test]
    fn test_operation_user_preserves_email(user in "[a-z0-9]+@[a-z]+\\.[a-z]{2,4}") {
        let op = JJOperation::new("id".to_string(), user.clone(), "desc".to_string());
        assert_eq!(op.user, user);
        assert!(op.user.contains('@'));
    }

    #[test]
    fn test_operation_description_preserved(desc in ".*{0,200}") {
        let op = JJOperation::new("id".to_string(), "user@test.com".to_string(), desc.clone());
        assert_eq!(op.description, desc);
    }

    #[test]
    fn test_operation_serialization_roundtrip(op in operation_strategy()) {
        let json = serde_json::to_string(&op).unwrap();
        let deserialized: JJOperation = serde_json::from_str(&json).unwrap();

        assert_eq!(op.id, deserialized.id);
        assert_eq!(op.user, deserialized.user);
        assert_eq!(op.description, deserialized.description);
    }

    #[test]
    fn test_operation_log_bounded_size(
        max_entries in 1usize..1000,
        num_operations in 0usize..2000,
    ) {
        let mut log = JJOperationLog::new(max_entries);

        for i in 0..num_operations {
            let op = JJOperation::new(
                format!("id_{}", i),
                "user@test.com".to_string(),
                "desc".to_string(),
            );
            log.add_operation(op);
        }

        assert!(log.len() <= max_entries);
        if num_operations > 0 {
            assert!(log.len() > 0);
        }
    }

    #[test]
    fn test_operation_log_preserves_order(count in 1usize..100) {
        let mut log = JJOperationLog::new(count * 2);

        for i in 0..count {
            let op = JJOperation::new(
                format!("id_{}", i),
                "user@test.com".to_string(),
                format!("desc_{}", i),
            );
            log.add_operation(op);
        }

        let recent = log.recent(count);
        for (i, op) in recent.iter().enumerate() {
            assert_eq!(op.id, format!("id_{}", i));
        }
    }

    #[test]
    fn test_operation_timestamp_iso_format(op in operation_strategy()) {
        let iso = op.timestamp_iso();

        // ISO 8601 format should contain 'T' and be reasonable length
        assert!(iso.contains('T'));
        assert!(iso.len() >= 20);
        assert!(iso.len() <= 30);
    }

    #[test]
    fn test_operation_type_serialization_roundtrip(op_type in operation_type_strategy()) {
        let json = serde_json::to_string(&op_type).unwrap();
        let deserialized: OperationType = serde_json::from_str(&json).unwrap();

        assert_eq!(op_type, deserialized);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn test_operation_log_get_by_id_consistency(
        num_ops in 1usize..50,
        query_index in 0usize..50,
    ) {
        let num_ops = num_ops.max(1);
        let mut log = JJOperationLog::new(100);

        let ids: Vec<String> = (0..num_ops)
            .map(|i| format!("id_{}", i))
            .collect();

        for id in &ids {
            let op = JJOperation::new(
                id.clone(),
                "user@test.com".to_string(),
                "desc".to_string(),
            );
            log.add_operation(op);
        }

        if query_index < num_ops {
            let query_id = &ids[query_index];
            let result = log.get_by_id(query_id);
            assert!(result.is_some());
            assert_eq!(result.unwrap().id, *query_id);
        }
    }

    #[test]
    fn test_operation_log_filter_correctness(
        commit_count in 0usize..20,
        rebase_count in 0usize..20,
    ) {
        let mut log = JJOperationLog::new(100);

        for i in 0..commit_count {
            let mut op = JJOperation::new(
                format!("commit_{}", i),
                "user@test.com".to_string(),
                "commit".to_string(),
            );
            op.op_type = OperationType::Commit;
            log.add_operation(op);
        }

        for i in 0..rebase_count {
            let mut op = JJOperation::new(
                format!("rebase_{}", i),
                "user@test.com".to_string(),
                "rebase".to_string(),
            );
            op.op_type = OperationType::Rebase;
            log.add_operation(op);
        }

        let commits = log.filter_by_type(OperationType::Commit);
        let rebases = log.filter_by_type(OperationType::Rebase);

        assert_eq!(commits.len(), commit_count);
        assert_eq!(rebases.len(), rebase_count);
    }
}

#[test]
fn test_operation_parse_from_log_fuzz() {
    let test_cases = vec![
        "@  abc123 user@test.com 2024-01-01 some description",
        "◉  def456 other@test.com 2024-01-02 another desc",
        "invalid line",
        "",
        "@",
        "@ id",
        "@ id user",
        "@ id user date",
        "@ id user date time description here",
    ];

    for line in test_cases {
        let result = JJOperation::parse_from_log(line);
        // Should not panic, either Some or None
        match result {
            Some(op) => {
                assert!(!op.id.is_empty());
                assert!(!op.user.is_empty());
            }
            None => {}
        }
    }
}
