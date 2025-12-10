use ran_opt::pfs_twin::{
    PfsTwin, NetworkElement, NetworkEdge, NetworkElementType, EdgeType,
};
use ndarray::Array2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("PFS Twin Graph Neural Network Example");
    println!("=====================================");

    // Create a new PFS Twin instance
    let mut pfs_twin = PfsTwin::new(64, 128, 32);
    
    // Add network hierarchy: gNB -> DU -> CU
    println!("\n1. Setting up network hierarchy...");
    let du_ids = vec!["DU1".to_string(), "DU2".to_string()];
    pfs_twin.add_network_hierarchy("gNB1", &du_ids, "CU1");
    
    // Add some cells and their neighbor relationships
    println!("2. Adding cell neighbor relationships...");
    let cell1 = NetworkElement {
        id: "Cell1".to_string(),
        element_type: NetworkElementType::Cell,
        features: vec![1.0, 0.5, 0.8, 0.3],
        position: Some((100.0, 200.0, 10.0)),
    };
    
    let cell2 = NetworkElement {
        id: "Cell2".to_string(),
        element_type: NetworkElementType::Cell,
        features: vec![0.8, 0.6, 0.7, 0.4],
        position: Some((150.0, 250.0, 12.0)),
    };
    
    pfs_twin.topology.add_element(cell1);
    pfs_twin.topology.add_element(cell2);
    
    // Add neighbor relationships
    let neighbor_ids = vec!["Cell2".to_string()];
    pfs_twin.add_cell_neighbors("Cell1", &neighbor_ids);
    
    // Create sample features for processing
    println!("3. Processing network topology with GNN...");
    let features = Array2::from_shape_vec((4, 4), vec![
        // gNB1 features
        1.0, 0.0, 0.0, 0.0,
        // DU1 features  
        0.0, 1.0, 0.0, 0.0,
        // DU2 features
        0.0, 1.0, 0.0, 0.0,
        // CU1 features
        0.0, 0.0, 1.0, 0.0,
    ])?;
    
    // Process topology and generate embeddings
    let embeddings = pfs_twin.process_topology(&features);
    
    println!("Generated embeddings shape: {:?}", embeddings.shape());
    println!("Sample embedding for first node: {:?}", 
             embeddings.row(0).iter().take(5).collect::<Vec<_>>());
    
    // Demonstrate incremental topology updates
    println!("\n4. Demonstrating incremental topology updates...");
    
    let updates = vec![
        ran_opt::pfs_twin::TopologyUpdate::UpdateNodeFeatures(
            "Cell1".to_string(), 
            vec![1.1, 0.6, 0.9, 0.4]
        ),
    ];
    
    pfs_twin.topology.incremental_update(updates);
    println!("Updated Cell1 features");
    
    println!("\n5. Network topology analysis complete!");
    println!("   - Created gNB -> DU -> CU hierarchy");
    println!("   - Added cell neighbor relationships");
    println!("   - Generated node embeddings using GNN");
    println!("   - Demonstrated incremental updates");
    
    Ok(())
}