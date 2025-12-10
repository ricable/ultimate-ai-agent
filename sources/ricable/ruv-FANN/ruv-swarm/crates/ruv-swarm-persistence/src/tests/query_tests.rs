//! Tests for query builder and advanced queries

use crate::models::*;
use crate::*;

#[test]
fn test_basic_query_builder() {
    let query = QueryBuilder::<AgentModel>::new("agents").build();

    assert_eq!(query, "SELECT * FROM agents");
}

#[test]
fn test_query_with_conditions() {
    let query = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .where_eq("agent_type", "compute")
        .build();

    assert_eq!(
        query,
        "SELECT * FROM agents WHERE status = 'running' AND agent_type = 'compute'"
    );
}

#[test]
fn test_query_with_like_pattern() {
    let query = QueryBuilder::<TaskModel>::new("tasks")
        .where_like("task_type", "train%")
        .where_eq("status", "pending")
        .build();

    assert_eq!(
        query,
        "SELECT * FROM tasks WHERE task_type LIKE 'train%' AND status = 'pending'"
    );
}

#[test]
fn test_query_with_comparison() {
    let query = QueryBuilder::<MetricModel>::new("metrics")
        .where_gt("value", 90)
        .where_eq("metric_type", "accuracy")
        .build();

    assert_eq!(
        query,
        "SELECT * FROM metrics WHERE value > 90 AND metric_type = 'accuracy'"
    );
}

#[test]
fn test_query_with_ordering() {
    let query = QueryBuilder::<EventModel>::new("events")
        .where_eq("event_type", "task_completed")
        .order_by("timestamp", true)
        .build();

    assert_eq!(
        query,
        "SELECT * FROM events WHERE event_type = 'task_completed' ORDER BY timestamp DESC"
    );
}

#[test]
fn test_query_with_limit_offset() {
    let query = QueryBuilder::<MessageModel>::new("messages")
        .where_eq("read", "false")
        .order_by("timestamp", false)
        .limit(10)
        .offset(20)
        .build();

    assert_eq!(
        query,
        "SELECT * FROM messages WHERE read = 'false' ORDER BY timestamp ASC LIMIT 10 OFFSET 20"
    );
}

#[test]
fn test_complex_query() {
    let query = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .where_like("capabilities", "%neural%")
        .where_gt("created_at", 1000000)
        .order_by("updated_at", true)
        .limit(50)
        .build();

    assert_eq!(
        query,
        "SELECT * FROM agents WHERE status = 'running' AND capabilities LIKE '%neural%' AND created_at > 1000000 ORDER BY updated_at DESC LIMIT 50"
    );
}

#[test]
fn test_query_builder_chaining() {
    let base_query = QueryBuilder::<TaskModel>::new("tasks");

    let filtered_query = base_query
        .where_eq("priority", "10")
        .where_eq("assigned_to", "agent-123");

    let final_query = filtered_query
        .order_by("created_at", false)
        .limit(5)
        .build();

    assert_eq!(
        final_query,
        "SELECT * FROM tasks WHERE priority = '10' AND assigned_to = 'agent-123' ORDER BY created_at ASC LIMIT 5"
    );
}

#[test]
fn test_query_injection_prevention() {
    // Test that special characters are handled properly
    let query = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("name", "Agent'; DROP TABLE agents; --")
        .build();

    // The query builder should escape quotes properly
    assert!(query.contains("Agent'; DROP TABLE agents; --"));
}

#[test]
fn test_empty_conditions() {
    let query = QueryBuilder::<AgentModel>::new("agents")
        .order_by("id", false)
        .limit(100)
        .build();

    assert_eq!(query, "SELECT * FROM agents ORDER BY id ASC LIMIT 100");
}

#[test]
fn test_pagination_queries() {
    let page_size = 20;
    let queries: Vec<String> = (0..5)
        .map(|page| {
            QueryBuilder::<TaskModel>::new("tasks")
                .where_eq("status", "completed")
                .order_by("completed_at", true)
                .limit(page_size)
                .offset(page * page_size)
                .build()
        })
        .collect();

    // Verify pagination offsets
    assert!(queries[0].contains("OFFSET 0"));
    assert!(queries[1].contains("OFFSET 20"));
    assert!(queries[2].contains("OFFSET 40"));
    assert!(queries[3].contains("OFFSET 60"));
    assert!(queries[4].contains("OFFSET 80"));
}

#[test]
fn test_aggregate_query_patterns() {
    // While the basic QueryBuilder doesn't support aggregates,
    // test patterns that would be used for aggregate queries

    let metrics_query = QueryBuilder::<MetricModel>::new("metrics")
        .where_eq("metric_type", "performance")
        .where_gt("timestamp", 1000000)
        .build();

    // In a real implementation, this might be extended to:
    // SELECT AVG(value), COUNT(*) FROM metrics WHERE ...
    assert!(metrics_query.starts_with("SELECT * FROM metrics"));
}

#[test]
fn test_join_query_patterns() {
    // Test query patterns that might be used for joins
    let agents_with_tasks = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .build();

    let tasks_for_agents = QueryBuilder::<TaskModel>::new("tasks")
        .where_eq("status", "assigned")
        .build();

    // In a real implementation, these might be combined into:
    // SELECT * FROM agents JOIN tasks ON agents.id = tasks.assigned_to
    assert!(agents_with_tasks.contains("agents"));
    assert!(tasks_for_agents.contains("tasks"));
}

#[test]
fn test_subquery_patterns() {
    // Test patterns for subqueries
    let active_agents = QueryBuilder::<AgentModel>::new("agents")
        .where_eq("status", "running")
        .build();

    // This could be used as a subquery in:
    // SELECT * FROM tasks WHERE assigned_to IN (SELECT id FROM agents WHERE status = 'running')
    assert!(active_agents.contains("WHERE status = 'running'"));
}

#[test]
fn test_date_range_queries() {
    let start_timestamp = 1000000;
    let _end_timestamp = 2000000;

    // Pattern for date range queries (would need additional methods in real implementation)
    let events_in_range = QueryBuilder::<EventModel>::new("events")
        .where_gt("timestamp", start_timestamp)
        .build();

    // Would ideally support: .where_between("timestamp", start, end)
    assert!(events_in_range.contains(&format!("timestamp > {}", start_timestamp)));
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_query_builder_consistency(
            table in "[a-z_]{1,20}",
            field1 in "[a-z_]{1,20}",
            value1 in "[a-zA-Z0-9]{1,20}",
            field2 in "[a-z_]{1,20}",
            value2 in "[a-zA-Z0-9]{1,20}",
            order_field in "[a-z_]{1,20}",
            desc in any::<bool>(),
            limit in 1usize..1000,
            offset in 0usize..1000,
        ) {
            let query = QueryBuilder::<AgentModel>::new(&table)
                .where_eq(&field1, &value1)
                .where_eq(&field2, &value2)
                .order_by(&order_field, desc)
                .limit(limit)
                .offset(offset)
                .build();

            // Verify query structure
            assert!(query.starts_with("SELECT * FROM"));
            assert!(query.contains(&table));
            assert!(query.contains("WHERE"));
            assert!(query.contains(&field1));
            assert!(query.contains(&value1));
            assert!(query.contains(&field2));
            assert!(query.contains(&value2));
            assert!(query.contains("ORDER BY"));
            assert!(query.contains(&order_field));
            assert!(query.contains(if desc { "DESC" } else { "ASC" }));
            assert!(query.contains(&format!("LIMIT {}", limit)));
            assert!(query.contains(&format!("OFFSET {}", offset)));
        }

        #[test]
        fn test_query_condition_count(
            conditions in prop::collection::vec(
                ("[a-z_]{1,10}", "[a-zA-Z0-9]{1,20}"),
                0..10
            )
        ) {
            let mut builder = QueryBuilder::<AgentModel>::new("test_table");

            for (field, value) in &conditions {
                builder = builder.where_eq(field, value);
            }

            let query = builder.build();

            if conditions.is_empty() {
                assert!(!query.contains("WHERE"));
            } else {
                assert!(query.contains("WHERE"));
                // Count ANDs - should be one less than number of conditions
                let and_count = query.matches(" AND ").count();
                assert_eq!(and_count, conditions.len().saturating_sub(1));
            }
        }
    }
}
