//! Property-based tests for types

use agentic_jujutsu::{JJConflict, JJResult, JJCommit, JJBranch};
use proptest::prelude::*;

// Property strategies
prop_compose! {
    fn conflict_strategy()(
        file_path in "[a-z/]+\\.(rs|js|py|go)",
        description in ".*{0,100}",
        sides in 2usize..10,
    ) -> JJConflict {
        JJConflict::new(file_path, description, sides)
    }
}

prop_compose! {
    fn result_strategy()(
        success in any::<bool>(),
        message in ".*{0,200}",
        exit_code in -1i32..255,
    ) -> JJResult {
        JJResult::new(success, message, exit_code)
    }
}

prop_compose! {
    fn commit_strategy()(
        change_id in "[a-z0-9]{8,16}",
        commit_id in "[a-z0-9]{40}",
        author in "[a-z]+@[a-z]+\\.[a-z]{2,4}",
        message in ".*{0,500}",
    ) -> JJCommit {
        JJCommit::new(change_id, commit_id, author, message)
    }
}

prop_compose! {
    fn branch_strategy()(
        name in "[a-z][a-z0-9-]{0,30}",
        target in "[a-z0-9]{40}",
    ) -> JJBranch {
        JJBranch::new(name, target)
    }
}

proptest! {
    #[test]
    fn test_conflict_preserves_file_path(path in "[a-z/]+\\.(rs|js|py)") {
        let conflict = JJConflict::new(path.clone(), "desc".to_string(), 2);
        assert_eq!(conflict.file_path, path);
    }

    #[test]
    fn test_conflict_sides_positive(sides in 2usize..100) {
        let conflict = JJConflict::new("file.rs".to_string(), "desc".to_string(), sides);
        assert_eq!(conflict.sides, sides);
        assert!(conflict.sides >= 2);
    }

    #[test]
    fn test_conflict_serialization_roundtrip(conflict in conflict_strategy()) {
        let json = serde_json::to_string(&conflict).unwrap();
        let deserialized: JJConflict = serde_json::from_str(&json).unwrap();

        assert_eq!(conflict, deserialized);
    }

    #[test]
    fn test_conflict_resolution_toggle(mut conflict in conflict_strategy()) {
        assert!(!conflict.is_resolved());

        conflict.mark_resolved();
        assert!(conflict.is_resolved());
        assert!(conflict.resolved);
    }

    #[test]
    fn test_result_exit_code_range(code in -1i32..256) {
        let result = JJResult::new(true, "msg".to_string(), code);
        assert_eq!(result.exit_code, code);
    }

    #[test]
    fn test_result_serialization_roundtrip(result in result_strategy()) {
        let json = serde_json::to_string(&result).unwrap();
        let deserialized: JJResult = serde_json::from_str(&json).unwrap();

        assert_eq!(result.success, deserialized.success);
        assert_eq!(result.exit_code, deserialized.exit_code);
    }

    #[test]
    fn test_result_ok_always_success() {
        let result = JJResult::ok("message".to_string());
        assert!(result.success);
        assert_eq!(result.exit_code, 0);
    }

    #[test]
    fn test_result_err_never_success(code in 1i32..256) {
        let result = JJResult::err("error".to_string(), code);
        assert!(!result.success);
        assert_ne!(result.exit_code, 0);
    }

    #[test]
    fn test_commit_preserves_fields(commit in commit_strategy()) {
        let json = serde_json::to_string(&commit).unwrap();
        let deserialized: JJCommit = serde_json::from_str(&json).unwrap();

        assert_eq!(commit.change_id, deserialized.change_id);
        assert_eq!(commit.commit_id, deserialized.commit_id);
        assert_eq!(commit.author, deserialized.author);
    }

    #[test]
    fn test_commit_timestamp_iso_format(commit in commit_strategy()) {
        let iso = commit.timestamp_iso();

        assert!(iso.contains('T'));
        assert!(iso.len() >= 20);
    }

    #[test]
    fn test_branch_name_format(name in "[a-z][a-z0-9-]{0,30}") {
        let branch = JJBranch::new(name.clone(), "target".to_string());

        assert_eq!(branch.name, name);
        assert!(!branch.name.is_empty());
    }

    #[test]
    fn test_branch_current_flag(mut branch in branch_strategy()) {
        assert!(!branch.is_current);

        branch.set_current(true);
        assert!(branch.is_current);

        branch.set_current(false);
        assert!(!branch.is_current);
    }

    #[test]
    fn test_branch_remote_management(mut branch in branch_strategy(), remote in "origin/[a-z-]+") {
        assert!(branch.remote.is_none());

        branch.set_remote(remote.clone());
        assert_eq!(branch.remote, Some(remote));
    }

    #[test]
    fn test_branch_serialization_roundtrip(branch in branch_strategy()) {
        let json = serde_json::to_string(&branch).unwrap();
        let deserialized: JJBranch = serde_json::from_str(&json).unwrap();

        assert_eq!(branch, deserialized);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn test_conflict_clone_equality(conflict in conflict_strategy()) {
        let cloned = conflict.clone();
        assert_eq!(conflict, cloned);
    }

    #[test]
    fn test_commit_equality_by_ids(
        change_id in "[a-z0-9]{8}",
        commit_id in "[a-z0-9]{40}",
    ) {
        let commit1 = JJCommit::new(
            change_id.clone(),
            commit_id.clone(),
            "user1@test.com".to_string(),
            "msg1".to_string(),
        );

        let commit2 = JJCommit::new(
            change_id.clone(),
            commit_id.clone(),
            "user2@test.com".to_string(),
            "msg2".to_string(),
        );

        assert_eq!(commit1.change_id, commit2.change_id);
        assert_eq!(commit1.commit_id, commit2.commit_id);
    }

    #[test]
    fn test_result_message_preservation(msg in ".*{0,500}") {
        let result = JJResult::ok(msg.clone());
        assert_eq!(result.message, msg);
    }
}
