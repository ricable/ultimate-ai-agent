//! Integration tests for parallel query execution

#[cfg(test)]
mod parallel_tests {
    use ruvector_postgres::index::parallel::*;
    use ruvector_postgres::index::hnsw::{HnswIndex, HnswConfig};
    use ruvector_postgres::distance::DistanceMetric;

    #[test]
    fn test_parallel_worker_estimation() {
        // Small index - no parallelism
        let workers = ruhnsw_estimate_parallel_workers(50, 5000, 10, 40);
        assert_eq!(workers, 0, "Small indexes should not use parallelism");

        // Medium index - some workers
        let workers = ruhnsw_estimate_parallel_workers(2000, 100000, 10, 40);
        assert!(workers > 0 && workers <= 4, "Medium indexes should use 1-4 workers");

        // Large index - more workers
        let workers = ruhnsw_estimate_parallel_workers(10000, 1000000, 10, 40);
        assert!(workers >= 2, "Large indexes should use multiple workers");

        // Complex query - more workers
        let workers_simple = ruhnsw_estimate_parallel_workers(5000, 500000, 10, 40);
        let workers_complex = ruhnsw_estimate_parallel_workers(5000, 500000, 200, 200);
        assert!(
            workers_complex >= workers_simple,
            "Complex queries should use more workers"
        );
    }

    #[test]
    fn test_partition_estimation() {
        // Should create more partitions than workers for load balancing
        let partitions = estimate_partitions(4, 100000);
        assert!(partitions >= 4, "Should have at least as many partitions as workers");
        assert!(partitions <= 50, "Should not create too many partitions");

        // Large dataset should create more partitions
        let partitions_large = estimate_partitions(4, 1000000);
        let partitions_small = estimate_partitions(4, 50000);
        assert!(
            partitions_large >= partitions_small,
            "Larger datasets should have more partitions"
        );
    }

    #[test]
    fn test_shared_state_work_stealing() {
        let state = RuHnswSharedState::new(
            4,   // 4 workers
            16,  // 16 partitions
            128, // 128 dimensions
            10,  // k=10
            40,  // ef_search=40
            DistanceMetric::Euclidean,
        );

        // Workers should be able to claim partitions
        let mut claimed = Vec::new();
        for _ in 0..16 {
            if let Some(partition) = state.get_next_partition() {
                claimed.push(partition);
            }
        }

        assert_eq!(claimed.len(), 16, "All partitions should be claimed");

        // Should return None after all partitions claimed
        assert_eq!(state.get_next_partition(), None);

        // Verify no duplicates
        let mut sorted = claimed.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), claimed.len(), "No duplicate partitions");
    }

    #[test]
    fn test_parallel_result_merging() {
        // Create results from 3 workers
        let worker1 = vec![
            (0.1, ItemPointer::new(1, 1)),
            (0.4, ItemPointer::new(1, 4)),
            (0.7, ItemPointer::new(1, 7)),
        ];

        let worker2 = vec![
            (0.2, ItemPointer::new(2, 2)),
            (0.5, ItemPointer::new(2, 5)),
            (0.8, ItemPointer::new(2, 8)),
        ];

        let worker3 = vec![
            (0.3, ItemPointer::new(3, 3)),
            (0.6, ItemPointer::new(3, 6)),
            (0.9, ItemPointer::new(3, 9)),
        ];

        // Merge top 5 results
        let merged = merge_knn_results(&[worker1, worker2, worker3], 5);

        assert_eq!(merged.len(), 5, "Should return exactly k results");

        // Verify sorted order
        for i in 1..merged.len() {
            assert!(
                merged[i - 1].0 <= merged[i].0,
                "Results should be sorted by distance"
            );
        }

        // Verify we got the actual top 5
        assert_eq!(merged[0].0, 0.1);
        assert_eq!(merged[1].0, 0.2);
        assert_eq!(merged[2].0, 0.3);
        assert_eq!(merged[3].0, 0.4);
        assert_eq!(merged[4].0, 0.5);
    }

    #[test]
    fn test_tournament_merge() {
        // Test tournament tree merge with sorted inputs
        let worker1 = vec![
            (0.1, ItemPointer::new(1, 1)),
            (0.5, ItemPointer::new(1, 5)),
            (0.9, ItemPointer::new(1, 9)),
        ];

        let worker2 = vec![
            (0.2, ItemPointer::new(2, 2)),
            (0.6, ItemPointer::new(2, 6)),
        ];

        let worker3 = vec![
            (0.3, ItemPointer::new(3, 3)),
            (0.4, ItemPointer::new(3, 4)),
            (0.7, ItemPointer::new(3, 7)),
        ];

        let merged = merge_knn_results_tournament(&[worker1, worker2, worker3], 6);

        assert_eq!(merged.len(), 6);

        // Verify sorted order
        let distances: Vec<f32> = merged.iter().map(|(d, _)| *d).collect();
        assert_eq!(distances, vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    }

    #[test]
    fn test_parallel_coordinator() {
        // Create a small HNSW index for testing
        let config = HnswConfig {
            m: 8,
            m0: 16,
            ef_construction: 32,
            ef_search: 20,
            max_elements: 1000,
            metric: DistanceMetric::Euclidean,
            seed: 42,
        };

        let index = HnswIndex::new(3, config);

        // Insert some test vectors
        for i in 0..100 {
            let vector = vec![
                (i as f32) * 0.1,
                (i as f32) * 0.2,
                (i as f32) * 0.3,
            ];
            index.insert(vector);
        }

        // Create parallel coordinator
        let mut coordinator = ParallelScanCoordinator::new(
            2,   // 2 workers
            4,   // 4 partitions
            3,   // 3 dimensions
            10,  // k=10
            20,  // ef_search=20
            DistanceMetric::Euclidean,
        );

        // Execute parallel scan
        let query = vec![0.5, 0.5, 0.5];
        let results = coordinator.execute_parallel_scan(&index, query);

        // Verify results
        assert!(results.len() <= 10, "Should return at most k results");

        // Check that results are sorted
        for i in 1..results.len() {
            assert!(
                results[i - 1].0 <= results[i].0,
                "Results should be sorted by distance"
            );
        }

        // Get statistics
        let stats = coordinator.get_stats();
        assert_eq!(stats.num_workers, 2);
        assert_eq!(stats.total_partitions, 4);
        assert_eq!(stats.completed_workers, 2);
    }

    #[test]
    fn test_item_pointer_mapping() {
        // Test node ID to ItemPointer mapping
        let ip1 = create_item_pointer(0);
        assert_eq!(ip1.block_number, 0);
        assert_eq!(ip1.offset_number, 1);

        let ip2 = create_item_pointer(100);
        assert_eq!(ip2.block_number, 0);
        assert_eq!(ip2.offset_number, 101);

        // Test block boundary (8191 tuples per page)
        let ip3 = create_item_pointer(8191);
        assert_eq!(ip3.block_number, 1);
        assert_eq!(ip3.offset_number, 1);

        let ip4 = create_item_pointer(16382);
        assert_eq!(ip4.block_number, 2);
        assert_eq!(ip4.offset_number, 1);
    }

    #[test]
    fn test_empty_worker_results() {
        // Test merging when some workers have no results
        let worker1 = vec![(0.1, ItemPointer::new(1, 1))];
        let worker2 = vec![];
        let worker3 = vec![(0.2, ItemPointer::new(3, 2))];

        let merged = merge_knn_results(&[worker1, worker2, worker3], 5);

        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].0, 0.1);
        assert_eq!(merged[1].0, 0.2);
    }

    #[test]
    fn test_merge_with_duplicates() {
        // Test that merging handles duplicate ItemPointers correctly
        let worker1 = vec![
            (0.1, ItemPointer::new(1, 1)),
            (0.3, ItemPointer::new(1, 3)),
        ];

        let worker2 = vec![
            (0.1, ItemPointer::new(1, 1)),  // Duplicate
            (0.2, ItemPointer::new(2, 2)),
        ];

        let merged = merge_knn_results(&[worker1, worker2], 3);

        // Should include both instances (heap-based merge doesn't deduplicate)
        assert!(merged.len() >= 3);
    }

    #[test]
    fn test_large_k_merge() {
        // Test merging with k larger than available results
        let worker1 = vec![
            (0.1, ItemPointer::new(1, 1)),
            (0.2, ItemPointer::new(1, 2)),
        ];

        let worker2 = vec![
            (0.3, ItemPointer::new(2, 3)),
        ];

        let merged = merge_knn_results(&[worker1, worker2], 100);

        // Should return all available results
        assert_eq!(merged.len(), 3);
    }

    #[test]
    fn test_parallel_scan_descriptor() {
        use std::sync::Arc;
        use parking_lot::RwLock;

        let shared_state = Arc::new(RwLock::new(RuHnswSharedState::new(
            2, 4, 128, 10, 40,
            DistanceMetric::Euclidean,
        )));

        let query = vec![0.5; 128];
        let desc = RuHnswParallelScanDesc::new(shared_state, 0, query.clone());

        assert_eq!(desc.worker_id, 0);
        assert_eq!(desc.query, query);
        assert_eq!(desc.local_results.len(), 0);
    }

    #[test]
    fn test_metrics_in_parallel_state() {
        let state = RuHnswSharedState::new(
            3, 9, 256, 50, 100,
            DistanceMetric::Cosine,
        );

        assert_eq!(state.num_workers, 3);
        assert_eq!(state.total_partitions, 9);
        assert_eq!(state.dimensions, 256);
        assert_eq!(state.k, 50);
        assert_eq!(state.ef_search, 100);
        assert_eq!(state.metric, DistanceMetric::Cosine);

        // Test completion tracking
        assert_eq!(state.completed_workers.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert!(!state.all_completed());

        state.mark_completed();
        state.mark_completed();
        assert!(!state.all_completed());

        state.mark_completed();
        assert!(state.all_completed());
    }
}
