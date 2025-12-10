//! Test Utilities for Neural Swarm Testing
//! 
//! Common utilities and helper functions for testing the swarm optimization system.

use standalone_neural_swarm::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Create a test swarm configuration
pub fn create_test_swarm_config() -> SwarmConfig {
    SwarmConfig {
        optimization: OptimizationConfig {
            population_size: 5,
            max_iterations: 20,
            inertia_weight: 0.7,
            cognitive_weight: 1.5,
            social_weight: 1.5,
            convergence_threshold: 0.01,
            elite_size: 2,
        },
        neural: NeuralConfig {
            learning_rate: 0.01,
            hidden_layers: vec![8, 4],
            activation_function: "relu".to_string(),
            regularization: 0.001,
        },
        system: SystemConfig {
            max_threads: 2,
            memory_limit_gb: 1.0,
            cache_size: 100,
            enable_logging: false,
        },
    }
}

/// Create a test swarm coordinator
pub fn create_test_coordinator(config: SwarmConfig) -> SwarmCoordinator {
    let swarm_params = SwarmParameters {
        population_size: config.optimization.population_size,
        max_iterations: config.optimization.max_iterations,
        inertia_weight: config.optimization.inertia_weight,
        cognitive_weight: config.optimization.cognitive_weight,
        social_weight: config.optimization.social_weight,
        convergence_threshold: config.optimization.convergence_threshold,
        elite_size: config.optimization.elite_size,
    };
    
    SwarmCoordinator::new(swarm_params)
}

/// Create test RAN metrics
pub fn create_test_metrics() -> RANMetrics {
    RANMetrics {
        throughput: 75.0,
        latency: 15.0,
        energy_efficiency: 0.8,
        interference_level: 0.2,
    }
}

/// Simulate test metrics based on configuration
pub fn simulate_test_metrics(config: &RANConfiguration) -> RANMetrics {
    let mut rng = StdRng::seed_from_u64(42);
    
    let mut metrics = RANMetrics::new();
    
    // Simulate realistic relationships between config and metrics
    let power_factor = (config.power_level - 5.0) / 35.0;
    let bandwidth_factor = config.bandwidth / 80.0;
    let freq_factor = (config.frequency_band - 2400.0) / 1100.0;
    
    metrics.throughput = (50.0 + power_factor * 30.0 + bandwidth_factor * 20.0) 
        * (1.0 + rng.gen_range(-0.1..0.1));
    
    metrics.latency = (20.0 - power_factor * 5.0 + freq_factor * 10.0)
        * (1.0 + rng.gen_range(-0.2..0.2));
    metrics.latency = metrics.latency.max(1.0);
    
    metrics.energy_efficiency = (0.6 + bandwidth_factor * 0.2 - power_factor * 0.1)
        * (1.0 + rng.gen_range(-0.1..0.1));
    metrics.energy_efficiency = metrics.energy_efficiency.clamp(0.1, 1.0);
    
    metrics.interference_level = (power_factor * 0.3 + freq_factor * 0.2)
        * (1.0 + rng.gen_range(-0.1..0.1));
    metrics.interference_level = metrics.interference_level.clamp(0.0, 1.0);
    
    metrics
}

/// Create a test demand predictor
pub fn create_test_demand_predictor() -> DemandPredictor {
    DemandPredictor::new(24)
}

/// Generate test demand data
pub fn generate_test_demand_data(count: usize) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::new();
    
    for i in 0..count {
        let base_demand = 50.0;
        let pattern = 20.0 * (2.0 * std::f32::consts::PI * (i as f32) / 24.0).sin();
        let noise = rng.gen_range(-5.0..5.0);
        
        let demand = (base_demand + pattern + noise).max(0.0);
        data.push(demand);
    }
    
    data
}

/// Create a test neural network
pub fn create_test_neural_network() -> NeuralNetwork {
    let config = NeuralNetworkConfig {
        input_size: 4,
        hidden_layers: vec![8, 4],
        output_size: 1,
        learning_rate: 0.01,
        activation_function: "relu".to_string(),
    };
    
    NeuralNetwork::new(config)
}

/// Create test training data for neural networks
pub fn create_test_training_data(count: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut data = Vec::new();
    
    for _ in 0..count {
        let inputs = vec![
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        ];
        
        // Simple target function: sum of squares
        let target = inputs.iter().map(|x| x * x).sum::<f32>();
        let outputs = vec![target];
        
        data.push((inputs, outputs));
    }
    
    data
}

/// Assert that two floating point numbers are approximately equal
pub fn assert_approx_eq(a: f32, b: f32, tolerance: f32) {
    let diff = (a - b).abs();
    assert!(diff < tolerance, "Values {} and {} differ by {} (tolerance: {})", a, b, diff, tolerance);
}

/// Assert that a vector contains only finite values
pub fn assert_finite_vector(vec: &[f32]) {
    for (i, &val) in vec.iter().enumerate() {
        assert!(val.is_finite(), "Non-finite value {} at index {}", val, i);
    }
}

/// Create a test RAN configuration
pub fn create_test_ran_configuration() -> RANConfiguration {
    RANConfiguration {
        cell_id: 123,
        power_level: 25.0,
        antenna_tilt: 0.0,
        bandwidth: 40.0,
        frequency_band: 2800.0,
        modulation_scheme: "64QAM".to_string(),
        mimo_config: "4x4".to_string(),
        beamforming_enabled: true,
    }
}

/// Benchmark helper for timing operations
pub struct BenchmarkTimer {
    start: std::time::Instant,
}

impl BenchmarkTimer {
    pub fn new() -> Self {
        Self {
            start: std::time::Instant::now(),
        }
    }
    
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }
    
    pub fn reset(&mut self) {
        self.start = std::time::Instant::now();
    }
}

/// Mock fitness function for testing
pub fn mock_fitness_function(config: &RANConfiguration) -> (f32, RANMetrics) {
    let metrics = simulate_test_metrics(config);
    let fitness = metrics.calculate_fitness();
    (fitness, metrics)
}

/// Validate optimization result
pub fn validate_optimization_result(result: &SwarmOptimizationResult) -> bool {
    // Check basic validity
    if result.iterations_completed == 0 {
        return false;
    }
    
    if !result.best_fitness.is_finite() {
        return false;
    }
    
    if result.convergence_history.len() != result.iterations_completed as usize {
        return false;
    }
    
    // Check convergence history is valid
    for &fitness in &result.convergence_history {
        if !fitness.is_finite() {
            return false;
        }
    }
    
    // Check that best fitness is actually the best in history
    let max_fitness = result.convergence_history.iter()
        .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    
    if (result.best_fitness - max_fitness).abs() > 1e-6 {
        return false;
    }
    
    true
}

/// Generate random population for testing
pub fn generate_test_population(size: usize, dimensions: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(42);
    let mut population = Vec::new();
    
    for _ in 0..size {
        let individual: Vec<f32> = (0..dimensions)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        population.push(individual);
    }
    
    population
}