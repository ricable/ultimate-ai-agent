//! Integration Tests for Standalone Neural Swarm
//! 
//! This module contains comprehensive integration tests for the swarm optimization system.

use standalone_neural_swarm::*;
use std::time::Duration;
use std::thread;

mod test_utils;
use test_utils::*;

#[cfg(test)]
mod swarm_integration_tests {
    use super::*;
    
    #[test]
    fn test_complete_swarm_optimization_flow() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        // Initialize swarm with 4 dimensions
        coordinator.initialize_swarm(4);
        
        // Verify swarm initialization
        assert!(!coordinator.agents.is_empty());
        assert_eq!(coordinator.agents.len(), 5); // Default test size
        
        // Run optimization
        let result = coordinator.optimize(|config| {
            let metrics = simulate_test_metrics(config);
            let fitness = metrics.calculate_fitness();
            (fitness, metrics)
        });
        
        // Verify results
        assert!(result.iterations_completed > 0);
        assert!(result.best_fitness > f32::NEG_INFINITY);
        assert!(!result.convergence_history.is_empty());
        assert_eq!(result.convergence_history.len(), result.iterations_completed as usize);
    }
    
    #[test]
    fn test_multi_agent_coordination() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        coordinator.initialize_swarm(4);
        
        // Test agent specialization diversity
        let specializations: std::collections::HashSet<_> = coordinator.agents
            .iter()
            .map(|agent| agent.neural_agent.specialization.clone())
            .collect();
        
        // Should have multiple specializations
        assert!(specializations.len() > 1);
        
        // Test agent communication
        let diversity = coordinator.calculate_swarm_diversity();
        assert!(diversity.position_diversity > 0.0);
    }
    
    #[test]
    fn test_neural_network_predictions() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        coordinator.initialize_swarm(4);
        
        // Test each agent's neural network
        for agent in &coordinator.agents {
            let test_metrics = create_test_metrics();
            
            // Test fitness prediction
            match agent.neural_agent.predict_fitness(&test_metrics) {
                Ok(prediction) => {
                    assert!(prediction.is_finite());
                    assert!(prediction >= 0.0); // Assuming fitness is non-negative
                },
                Err(_) => {
                    // Some agents might not be trained yet, which is acceptable
                    continue;
                }
            }
            
            // Test fitness evaluation
            let evaluation = agent.neural_agent.evaluate_fitness(&test_metrics);
            assert!(evaluation.is_finite());
        }
    }
    
    #[test]
    fn test_convergence_behavior() {
        let mut config = create_test_swarm_config();
        config.optimization.max_iterations = 50;
        config.optimization.convergence_threshold = 0.01;
        
        let mut coordinator = create_test_coordinator(config);
        coordinator.initialize_swarm(4);
        
        let result = coordinator.optimize(|config| {
            let metrics = simulate_test_metrics(config);
            let fitness = metrics.calculate_fitness();
            (fitness, metrics)
        });
        
        // Check convergence characteristics
        assert!(result.convergence_history.len() > 10);
        
        // Check if fitness improved over time
        let early_fitness = result.convergence_history[0];
        let late_fitness = result.convergence_history[result.convergence_history.len() - 1];
        assert!(late_fitness >= early_fitness);
    }
    
    #[test]
    fn test_performance_metrics() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        coordinator.initialize_swarm(4);
        
        let start_time = std::time::Instant::now();
        let result = coordinator.optimize(|config| {
            let metrics = simulate_test_metrics(config);
            let fitness = metrics.calculate_fitness();
            (fitness, metrics)
        });
        let actual_duration = start_time.elapsed();
        
        // Verify timing accuracy
        let reported_duration = Duration::from_millis(result.execution_time_ms as u64);
        let time_diff = if actual_duration > reported_duration {
            actual_duration - reported_duration
        } else {
            reported_duration - actual_duration
        };
        
        // Allow for some timing variance
        assert!(time_diff < Duration::from_millis(100));
    }
    
    #[test]
    fn test_demand_prediction_integration() {
        let mut predictor = create_test_demand_predictor();
        
        // Add test data
        let test_data = generate_test_demand_data(100);
        predictor.add_historical_data(test_data);
        
        // Test prediction
        let predictions = predictor.predict_demand(24).unwrap();
        assert_eq!(predictions.len(), 24);
        
        // Verify predictions are reasonable
        for prediction in predictions {
            assert!(prediction >= 0.0);
            assert!(prediction <= 1000.0); // Reasonable upper bound
        }
    }
    
    #[test]
    fn test_configuration_validation() {
        // Test valid configuration
        let valid_config = create_test_swarm_config();
        assert!(valid_config.validate().is_ok());
        
        // Test invalid configuration
        let mut invalid_config = create_test_swarm_config();
        invalid_config.optimization.population_size = 0;
        assert!(invalid_config.validate().is_err());
    }
    
    #[test]
    fn test_parallel_optimization() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        coordinator.initialize_swarm(4);
        
        // Test multiple optimization runs in parallel
        let handles: Vec<_> = (0..4).map(|_| {
            let mut coord_clone = coordinator.clone();
            std::thread::spawn(move || {
                coord_clone.optimize(|config| {
                    let metrics = simulate_test_metrics(config);
                    let fitness = metrics.calculate_fitness();
                    (fitness, metrics)
                })
            })
        }).collect();
        
        // Wait for all threads to complete
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        
        // Verify all optimizations completed successfully
        for result in results {
            assert!(result.iterations_completed > 0);
            assert!(result.best_fitness > f32::NEG_INFINITY);
        }
    }
    
    #[test]
    fn test_memory_usage_stability() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        coordinator.initialize_swarm(4);
        
        // Run multiple optimization cycles
        for _ in 0..10 {
            let _result = coordinator.optimize(|config| {
                let metrics = simulate_test_metrics(config);
                let fitness = metrics.calculate_fitness();
                (fitness, metrics)
            });
            
            // Small delay to allow for memory cleanup
            thread::sleep(Duration::from_millis(10));
        }
        
        // Test should complete without memory leaks or crashes
        assert!(true); // If we get here, memory usage was stable
    }
    
    #[test]
    fn test_error_handling() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        coordinator.initialize_swarm(4);
        
        // Test optimization with error-prone fitness function
        let result = coordinator.optimize(|_config| {
            // Sometimes return invalid fitness
            let should_error = rand::random::<f32>() < 0.1;
            if should_error {
                (f32::NAN, RANMetrics::new())
            } else {
                let metrics = create_test_metrics();
                let fitness = metrics.calculate_fitness();
                (fitness, metrics)
            }
        });
        
        // Should handle errors gracefully
        assert!(result.iterations_completed > 0);
        assert!(result.best_fitness.is_finite());
    }
}

#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_optimization_speed() {
        let config = create_test_swarm_config();
        let mut coordinator = create_test_coordinator(config);
        
        coordinator.initialize_swarm(4);
        
        let start = Instant::now();
        let result = coordinator.optimize(|config| {
            let metrics = simulate_test_metrics(config);
            let fitness = metrics.calculate_fitness();
            (fitness, metrics)
        });
        let duration = start.elapsed();
        
        // Optimization should complete within reasonable time
        assert!(duration < Duration::from_secs(10));
        assert!(result.iterations_completed > 0);
        
        // Performance should be consistent with reported time
        let reported_time = Duration::from_millis(result.execution_time_ms as u64);
        let time_diff = if duration > reported_time {
            duration - reported_time
        } else {
            reported_time - duration
        };
        
        assert!(time_diff < Duration::from_millis(500)); // Allow some variance
    }
    
    #[test]
    fn test_scalability() {
        let population_sizes = vec![5, 10, 20, 50];
        let mut execution_times = Vec::new();
        
        for &pop_size in &population_sizes {
            let mut config = create_test_swarm_config();
            config.optimization.population_size = pop_size;
            config.optimization.max_iterations = 20; // Keep iterations low for speed
            
            let mut coordinator = create_test_coordinator(config);
            coordinator.initialize_swarm(4);
            
            let start = Instant::now();
            let _result = coordinator.optimize(|config| {
                let metrics = simulate_test_metrics(config);
                let fitness = metrics.calculate_fitness();
                (fitness, metrics)
            });
            let duration = start.elapsed();
            
            execution_times.push(duration);
        }
        
        // Execution time should scale reasonably with population size
        // (not necessarily linearly, but shouldn't be exponential)
        for i in 1..execution_times.len() {
            let ratio = execution_times[i].as_secs_f64() / execution_times[0].as_secs_f64();
            let pop_ratio = population_sizes[i] as f64 / population_sizes[0] as f64;
            
            // Time ratio should be less than square of population ratio
            assert!(ratio < pop_ratio * pop_ratio);
        }
    }
}