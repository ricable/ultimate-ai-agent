//! Unit tests for swarm topologies

use crate::topology::*;
use crate::agent::AgentId;

#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};
#[cfg(not(feature = "std"))]
use alloc::collections::{BTreeMap as HashMap, BTreeSet as HashSet};

#[test]
fn test_topology_creation() {
    let topology = Topology::new(TopologyType::Mesh);
    assert_eq!(topology.topology_type, TopologyType::Mesh);
    assert!(topology.connections.is_empty());
    assert!(topology.groups.is_empty());
    assert!(topology.hierarchy.is_none());
}

#[test]
fn test_mesh_topology() {
    let agents = vec![
        AgentId::new("agent-1"),
        AgentId::new("agent-2"),
        AgentId::new("agent-3"),
    ];
    
    let topology = Topology::mesh(&agents);
    
    // Each agent should be connected to all others
    for agent in &agents {
        let neighbors = topology.get_neighbors(agent).unwrap();
        assert_eq!(neighbors.len(), 2); // Connected to 2 other agents
        
        for other in &agents {
            if agent != other {
                assert!(neighbors.contains(other));
            }
        }
    }
    
    // Total connections: 3 agents * 2 connections each / 2 (undirected) = 3
    assert_eq!(topology.connection_count(), 3);
    assert!(topology.is_fully_connected());
}

#[test]
fn test_star_topology() {
    let center = AgentId::new("center");
    let agents = vec![
        AgentId::new("agent-1"),
        AgentId::new("agent-2"),
        AgentId::new("agent-3"),
        AgentId::new("agent-4"),
    ];
    
    let topology = Topology::star(center.clone(), &agents);
    
    // Center should be connected to all agents
    let center_neighbors = topology.get_neighbors(&center).unwrap();
    assert_eq!(center_neighbors.len(), 4);
    
    // Each agent should only be connected to center
    for agent in &agents {
        let neighbors = topology.get_neighbors(agent).unwrap();
        assert_eq!(neighbors.len(), 1);
        assert!(neighbors.contains(&center));
    }
}

#[test]
fn test_pipeline_topology() {
    let agents = vec![
        AgentId::new("stage-1"),
        AgentId::new("stage-2"),
        AgentId::new("stage-3"),
        AgentId::new("stage-4"),
    ];
    
    let topology = Topology::pipeline(&agents);
    
    // First agent connects only forward
    let first_neighbors = topology.get_neighbors(&agents[0]).unwrap();
    assert_eq!(first_neighbors.len(), 1);
    assert!(first_neighbors.contains(&agents[1]));
    
    // Middle agents connect both ways
    let middle_neighbors = topology.get_neighbors(&agents[1]).unwrap();
    assert_eq!(middle_neighbors.len(), 2);
    assert!(middle_neighbors.contains(&agents[0]));
    assert!(middle_neighbors.contains(&agents[2]));
    
    // Last agent connects only backward
    let last_neighbors = topology.get_neighbors(&agents[3]).unwrap();
    assert_eq!(last_neighbors.len(), 1);
    assert!(last_neighbors.contains(&agents[2]));
}

#[test]
fn test_add_connection() {
    let mut topology = Topology::new(TopologyType::Custom);
    
    let agent1 = AgentId::new("agent-1");
    let agent2 = AgentId::new("agent-2");
    
    topology.add_connection(agent1.clone(), agent2.clone());
    
    assert!(topology.are_connected(&agent1, &agent2));
    assert!(topology.are_connected(&agent2, &agent1)); // Undirected
}

#[test]
fn test_remove_connection() {
    let mut topology = Topology::new(TopologyType::Custom);
    
    let agent1 = AgentId::new("agent-1");
    let agent2 = AgentId::new("agent-2");
    
    topology.add_connection(agent1.clone(), agent2.clone());
    assert!(topology.are_connected(&agent1, &agent2));
    
    topology.remove_connection(&agent1, &agent2);
    assert!(!topology.are_connected(&agent1, &agent2));
    assert!(!topology.are_connected(&agent2, &agent1));
}

#[test]
fn test_groups() {
    let mut topology = Topology::new(TopologyType::Clustered);
    
    let agent1 = AgentId::new("agent-1");
    let agent2 = AgentId::new("agent-2");
    let agent3 = AgentId::new("agent-3");
    
    topology.add_to_group("compute-cluster", agent1.clone());
    topology.add_to_group("compute-cluster", agent2.clone());
    topology.add_to_group("analysis-cluster", agent3.clone());
    
    let compute_group = topology.get_group("compute-cluster").unwrap();
    assert_eq!(compute_group.len(), 2);
    assert!(compute_group.contains(&agent1));
    assert!(compute_group.contains(&agent2));
    
    let analysis_group = topology.get_group("analysis-cluster").unwrap();
    assert_eq!(analysis_group.len(), 1);
    assert!(analysis_group.contains(&agent3));
}

#[test]
fn test_hierarchy_node() {
    let mut root = HierarchyNode::new(AgentId::new("root"), 0);
    
    let child1 = HierarchyNode::new(AgentId::new("child-1"), 1);
    let child2 = HierarchyNode::new(AgentId::new("child-2"), 1);
    
    root.add_child(child1);
    root.add_child(child2);
    
    assert_eq!(root.children.len(), 2);
    assert_eq!(root.level, 0);
    assert_eq!(root.children[0].level, 1);
}

#[test]
fn test_hierarchy_find() {
    let mut root = HierarchyNode::new(AgentId::new("root"), 0);
    
    let mut child1 = HierarchyNode::new(AgentId::new("child-1"), 1);
    let grandchild = HierarchyNode::new(AgentId::new("grandchild"), 2);
    
    child1.add_child(grandchild);
    root.add_child(child1);
    
    let found = root.find(&AgentId::new("grandchild")).unwrap();
    assert_eq!(found.level, 2);
    
    assert!(root.find(&AgentId::new("nonexistent")).is_none());
}

#[test]
fn test_hierarchy_agents_at_level() {
    let mut root = HierarchyNode::new(AgentId::new("root"), 0);
    
    let child1 = HierarchyNode::new(AgentId::new("child-1"), 1);
    let child2 = HierarchyNode::new(AgentId::new("child-2"), 1);
    
    let mut child1_with_grandchildren = child1;
    child1_with_grandchildren.add_child(HierarchyNode::new(AgentId::new("grandchild-1"), 2));
    child1_with_grandchildren.add_child(HierarchyNode::new(AgentId::new("grandchild-2"), 2));
    
    root.add_child(child1_with_grandchildren);
    root.add_child(child2);
    
    let level0_agents = root.agents_at_level(0);
    assert_eq!(level0_agents.len(), 1);
    
    let level1_agents = root.agents_at_level(1);
    assert_eq!(level1_agents.len(), 2);
    
    let level2_agents = root.agents_at_level(2);
    assert_eq!(level2_agents.len(), 2);
}

#[test]
fn test_topology_connectivity_check() {
    // Fully connected topology
    let agents = vec![
        AgentId::new("a"),
        AgentId::new("b"),
        AgentId::new("c"),
    ];
    let mesh = Topology::mesh(&agents);
    assert!(mesh.is_fully_connected());
    
    // Disconnected topology
    let mut disconnected = Topology::new(TopologyType::Custom);
    disconnected.add_connection(AgentId::new("a"), AgentId::new("b"));
    disconnected.add_connection(AgentId::new("c"), AgentId::new("d"));
    assert!(!disconnected.is_fully_connected());
    
    // Empty topology
    let empty = Topology::new(TopologyType::Custom);
    assert!(empty.is_fully_connected()); // Empty is considered fully connected
}

#[test]
fn test_complex_topology_scenario() {
    let mut topology = Topology::new(TopologyType::Custom);
    
    // Create a complex network
    let agents: Vec<_> = (0..10)
        .map(|i| AgentId::new(format!("agent-{}", i)))
        .collect();
    
    // Create clusters
    for i in 0..3 {
        topology.add_to_group(format!("cluster-{}", i), agents[i * 3].clone());
        topology.add_to_group(format!("cluster-{}", i), agents[i * 3 + 1].clone());
        topology.add_to_group(format!("cluster-{}", i), agents[i * 3 + 2].clone());
        
        // Connect agents within cluster
        topology.add_connection(agents[i * 3].clone(), agents[i * 3 + 1].clone());
        topology.add_connection(agents[i * 3 + 1].clone(), agents[i * 3 + 2].clone());
        topology.add_connection(agents[i * 3].clone(), agents[i * 3 + 2].clone());
    }
    
    // Connect cluster leaders
    topology.add_connection(agents[0].clone(), agents[3].clone());
    topology.add_connection(agents[3].clone(), agents[6].clone());
    
    // Add the 10th agent as a global coordinator
    for i in 0..9 {
        topology.add_connection(agents[9].clone(), agents[i].clone());
    }
    
    // Verify structure
    assert_eq!(topology.groups.len(), 3);
    assert!(topology.get_neighbors(&agents[9]).unwrap().len() == 9); // Connected to all
    assert!(topology.is_fully_connected());
}

#[test]
fn test_topology_type_default() {
    assert_eq!(TopologyType::default(), TopologyType::Mesh);
}

#[cfg(feature = "proptest")]
mod property_tests {
    use super::*;
    use proptest::prelude::*;
    
    proptest! {
        #[test]
        fn test_topology_connections_are_symmetric(
            agent_pairs in prop::collection::vec(
                (0u32..100, 0u32..100),
                1..20
            )
        ) {
            let mut topology = Topology::new(TopologyType::Custom);
            
            for (a, b) in agent_pairs {
                let agent_a = AgentId::new(format!("agent-{}", a));
                let agent_b = AgentId::new(format!("agent-{}", b));
                
                topology.add_connection(agent_a.clone(), agent_b.clone());
                
                // Verify symmetry
                assert_eq!(
                    topology.are_connected(&agent_a, &agent_b),
                    topology.are_connected(&agent_b, &agent_a)
                );
            }
        }
        
        #[test]
        fn test_mesh_topology_properties(
            agent_count in 2usize..20
        ) {
            let agents: Vec<_> = (0..agent_count)
                .map(|i| AgentId::new(format!("agent-{}", i)))
                .collect();
            
            let topology = Topology::mesh(&agents);
            
            // Each agent should have exactly (n-1) connections
            for agent in &agents {
                let neighbors = topology.get_neighbors(agent).unwrap();
                assert_eq!(neighbors.len(), agent_count - 1);
            }
            
            // Total connections should be n*(n-1)/2
            assert_eq!(
                topology.connection_count(),
                agent_count * (agent_count - 1) / 2
            );
            
            // Should always be fully connected
            assert!(topology.is_fully_connected());
        }
    }
}