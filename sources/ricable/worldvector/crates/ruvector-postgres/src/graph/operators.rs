// PostgreSQL operators for graph operations

use pgrx::prelude::*;
use pgrx::JsonB;
use serde_json::{json, Value as JsonValue};
use std::collections::HashMap;

use super::{get_or_create_graph, get_graph};
use super::cypher::query as cypher_query;
use super::traversal::{bfs, shortest_path_dijkstra};

/// Create a new graph
///
/// # Example
/// ```sql
/// SELECT ruvector_create_graph('my_graph');
/// ```
#[pg_extern]
fn ruvector_create_graph(name: &str) -> bool {
    get_or_create_graph(name);
    true
}

/// Execute a Cypher query on a graph
///
/// # Example
/// ```sql
/// SELECT ruvector_cypher('my_graph', 'CREATE (n:Person {name: ''Alice''}) RETURN n', NULL);
/// SELECT ruvector_cypher('my_graph', 'MATCH (n:Person) WHERE n.name = $name RETURN n', '{"name": "Alice"}');
/// ```
#[pg_extern]
fn ruvector_cypher(
    graph_name: &str,
    query: &str,
    params: Option<JsonB>,
) -> Result<JsonB, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    let params_json = params.map(|p| p.0);

    let result = cypher_query(&graph, query, params_json)?;

    Ok(JsonB(result))
}

/// Find shortest path between two nodes
///
/// # Example
/// ```sql
/// SELECT ruvector_shortest_path('my_graph', 1, 10, 5);
/// ```
#[pg_extern]
fn ruvector_shortest_path(
    graph_name: &str,
    start_id: i64,
    end_id: i64,
    max_hops: i32,
) -> Result<JsonB, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    let start = start_id as u64;
    let end = end_id as u64;
    let max_hops = max_hops as usize;

    let path = bfs(&graph, start, end, None, max_hops)
        .ok_or_else(|| "No path found".to_string())?;

    let result = json!({
        "nodes": path.nodes,
        "edges": path.edges,
        "length": path.len(),
        "cost": path.cost
    });

    Ok(JsonB(result))
}

/// Find weighted shortest path using Dijkstra's algorithm
///
/// # Example
/// ```sql
/// SELECT ruvector_shortest_path_weighted('my_graph', 1, 10, 'distance');
/// ```
#[pg_extern]
fn ruvector_shortest_path_weighted(
    graph_name: &str,
    start_id: i64,
    end_id: i64,
    weight_property: &str,
) -> Result<JsonB, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    let start = start_id as u64;
    let end = end_id as u64;

    let path = shortest_path_dijkstra(&graph, start, end, weight_property)
        .ok_or_else(|| "No path found".to_string())?;

    let result = json!({
        "nodes": path.nodes,
        "edges": path.edges,
        "length": path.len(),
        "cost": path.cost
    });

    Ok(JsonB(result))
}

/// Get graph statistics
///
/// # Example
/// ```sql
/// SELECT ruvector_graph_stats('my_graph');
/// ```
#[pg_extern]
fn ruvector_graph_stats(graph_name: &str) -> Result<JsonB, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    let stats = graph.stats();

    let result = json!({
        "name": graph_name,
        "node_count": stats.node_count,
        "edge_count": stats.edge_count,
        "labels": stats.labels,
        "edge_types": stats.edge_types
    });

    Ok(JsonB(result))
}

/// Add a node to a graph
///
/// # Example
/// ```sql
/// SELECT ruvector_add_node('my_graph', ARRAY['Person'], '{"name": "Alice", "age": 30}');
/// ```
#[pg_extern]
fn ruvector_add_node(
    graph_name: &str,
    labels: Vec<String>,
    properties: JsonB,
) -> Result<i64, String> {
    let graph = get_or_create_graph(graph_name);

    let props = if let JsonValue::Object(map) = properties.0 {
        map.into_iter()
            .map(|(k, v)| (k, v))
            .collect()
    } else {
        HashMap::new()
    };

    let node_id = graph.add_node(labels, props);

    Ok(node_id as i64)
}

/// Add an edge to a graph
///
/// # Example
/// ```sql
/// SELECT ruvector_add_edge('my_graph', 1, 2, 'KNOWS', '{"since": 2020}');
/// ```
#[pg_extern]
fn ruvector_add_edge(
    graph_name: &str,
    source_id: i64,
    target_id: i64,
    edge_type: &str,
    properties: JsonB,
) -> Result<i64, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    let props = if let JsonValue::Object(map) = properties.0 {
        map.into_iter()
            .map(|(k, v)| (k, v))
            .collect()
    } else {
        HashMap::new()
    };

    let edge_id = graph.add_edge(
        source_id as u64,
        target_id as u64,
        edge_type.to_string(),
        props,
    )?;

    Ok(edge_id as i64)
}

/// Get a node by ID
///
/// # Example
/// ```sql
/// SELECT ruvector_get_node('my_graph', 1);
/// ```
#[pg_extern]
fn ruvector_get_node(
    graph_name: &str,
    node_id: i64,
) -> Result<Option<JsonB>, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    if let Some(node) = graph.nodes.get(node_id as u64) {
        let json = serde_json::to_value(&node)
            .map_err(|e| format!("Serialization error: {}", e))?;
        Ok(Some(JsonB(json)))
    } else {
        Ok(None)
    }
}

/// Get an edge by ID
///
/// # Example
/// ```sql
/// SELECT ruvector_get_edge('my_graph', 1);
/// ```
#[pg_extern]
fn ruvector_get_edge(
    graph_name: &str,
    edge_id: i64,
) -> Result<Option<JsonB>, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    if let Some(edge) = graph.edges.get(edge_id as u64) {
        let json = serde_json::to_value(&edge)
            .map_err(|e| format!("Serialization error: {}", e))?;
        Ok(Some(JsonB(json)))
    } else {
        Ok(None)
    }
}

/// Find nodes by label
///
/// # Example
/// ```sql
/// SELECT ruvector_find_nodes_by_label('my_graph', 'Person');
/// ```
#[pg_extern]
fn ruvector_find_nodes_by_label(
    graph_name: &str,
    label: &str,
) -> Result<JsonB, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    let nodes = graph.nodes.find_by_label(label);

    let json = serde_json::to_value(&nodes)
        .map_err(|e| format!("Serialization error: {}", e))?;

    Ok(JsonB(json))
}

/// Get neighbors of a node
///
/// # Example
/// ```sql
/// SELECT ruvector_get_neighbors('my_graph', 1);
/// ```
#[pg_extern]
fn ruvector_get_neighbors(
    graph_name: &str,
    node_id: i64,
) -> Result<Vec<i64>, String> {
    let graph = get_graph(graph_name)
        .ok_or_else(|| format!("Graph '{}' does not exist", graph_name))?;

    let neighbors = graph.edges.get_neighbors(node_id as u64);

    Ok(neighbors.into_iter().map(|id| id as i64).collect())
}

/// Delete a graph
///
/// # Example
/// ```sql
/// SELECT ruvector_delete_graph('my_graph');
/// ```
#[pg_extern]
fn ruvector_delete_graph(graph_name: &str) -> bool {
    super::delete_graph(graph_name)
}

/// List all graphs
///
/// # Example
/// ```sql
/// SELECT ruvector_list_graphs();
/// ```
#[pg_extern]
fn ruvector_list_graphs() -> Vec<String> {
    super::list_graphs()
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::*;
    use pgrx::prelude::*;

    #[pg_test]
    fn test_create_graph() {
        let result = ruvector_create_graph("test_graph");
        assert!(result);

        let graphs = ruvector_list_graphs();
        assert!(graphs.contains(&"test_graph".to_string()));

        ruvector_delete_graph("test_graph");
    }

    #[pg_test]
    fn test_add_node_and_edge() {
        ruvector_create_graph("test_graph");

        let node1 = ruvector_add_node(
            "test_graph",
            vec!["Person".to_string()],
            JsonB(json!({"name": "Alice"})),
        ).unwrap();

        let node2 = ruvector_add_node(
            "test_graph",
            vec!["Person".to_string()],
            JsonB(json!({"name": "Bob"})),
        ).unwrap();

        let edge = ruvector_add_edge(
            "test_graph",
            node1,
            node2,
            "KNOWS",
            JsonB(json!({"since": 2020})),
        ).unwrap();

        assert!(edge > 0);

        let stats = ruvector_graph_stats("test_graph").unwrap();
        let stats_obj = stats.0.as_object().unwrap();
        assert_eq!(stats_obj["node_count"].as_u64().unwrap(), 2);
        assert_eq!(stats_obj["edge_count"].as_u64().unwrap(), 1);

        ruvector_delete_graph("test_graph");
    }

    #[pg_test]
    fn test_cypher_create_and_match() {
        ruvector_create_graph("test_graph");

        // Create a node
        let create_result = ruvector_cypher(
            "test_graph",
            "CREATE (n:Person {name: 'Alice', age: 30}) RETURN n",
            None,
        );
        assert!(create_result.is_ok());

        // Match the node
        let match_result = ruvector_cypher(
            "test_graph",
            "MATCH (n:Person) WHERE n.name = 'Alice' RETURN n",
            None,
        );
        assert!(match_result.is_ok());

        ruvector_delete_graph("test_graph");
    }

    #[pg_test]
    fn test_shortest_path() {
        ruvector_create_graph("test_graph");

        let n1 = ruvector_add_node(
            "test_graph",
            vec![],
            JsonB(json!({})),
        ).unwrap();

        let n2 = ruvector_add_node(
            "test_graph",
            vec![],
            JsonB(json!({})),
        ).unwrap();

        let n3 = ruvector_add_node(
            "test_graph",
            vec![],
            JsonB(json!({})),
        ).unwrap();

        ruvector_add_edge("test_graph", n1, n2, "KNOWS", JsonB(json!({}))).unwrap();
        ruvector_add_edge("test_graph", n2, n3, "KNOWS", JsonB(json!({}))).unwrap();

        let path = ruvector_shortest_path("test_graph", n1, n3, 10).unwrap();
        let path_obj = path.0.as_object().unwrap();
        assert_eq!(path_obj["length"].as_u64().unwrap(), 3);

        ruvector_delete_graph("test_graph");
    }

    #[pg_test]
    fn test_graph_stats() {
        ruvector_create_graph("test_graph");

        ruvector_add_node(
            "test_graph",
            vec!["Person".to_string()],
            JsonB(json!({"name": "Alice"})),
        ).unwrap();

        let stats = ruvector_graph_stats("test_graph").unwrap();
        let stats_obj = stats.0.as_object().unwrap();

        assert_eq!(stats_obj["node_count"].as_u64().unwrap(), 1);
        assert_eq!(stats_obj["edge_count"].as_u64().unwrap(), 0);

        let labels = stats_obj["labels"].as_array().unwrap();
        assert!(labels.iter().any(|l| l.as_str().unwrap() == "Person"));

        ruvector_delete_graph("test_graph");
    }

    #[pg_test]
    fn test_find_nodes_by_label() {
        ruvector_create_graph("test_graph");

        ruvector_add_node(
            "test_graph",
            vec!["Person".to_string()],
            JsonB(json!({"name": "Alice"})),
        ).unwrap();

        ruvector_add_node(
            "test_graph",
            vec!["Person".to_string()],
            JsonB(json!({"name": "Bob"})),
        ).unwrap();

        let nodes = ruvector_find_nodes_by_label("test_graph", "Person").unwrap();
        let nodes_array = nodes.0.as_array().unwrap();
        assert_eq!(nodes_array.len(), 2);

        ruvector_delete_graph("test_graph");
    }

    #[pg_test]
    fn test_get_neighbors() {
        ruvector_create_graph("test_graph");

        let n1 = ruvector_add_node("test_graph", vec![], JsonB(json!({}))).unwrap();
        let n2 = ruvector_add_node("test_graph", vec![], JsonB(json!({}))).unwrap();
        let n3 = ruvector_add_node("test_graph", vec![], JsonB(json!({}))).unwrap();

        ruvector_add_edge("test_graph", n1, n2, "KNOWS", JsonB(json!({}))).unwrap();
        ruvector_add_edge("test_graph", n1, n3, "KNOWS", JsonB(json!({}))).unwrap();

        let neighbors = ruvector_get_neighbors("test_graph", n1).unwrap();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&n2));
        assert!(neighbors.contains(&n3));

        ruvector_delete_graph("test_graph");
    }
}
