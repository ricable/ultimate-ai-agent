use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumBand {
    frequency: f64,        // MHz
    bandwidth: f64,        // MHz
    power_limit: f64,      // dBm
    interference_level: f64, // dBm
    utilization: f64,      // 0.0 to 1.0
    priority: u8,          // 1-10
    regulatory_class: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerControlSettings {
    min_power: f64,        // dBm
    max_power: f64,        // dBm
    step_size: f64,        // dB
    target_sinr: f64,      // dB
    power_headroom: f64,   // dB
    thermal_limit: f64,    // Celsius
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPowerState {
    cell_id: u32,
    current_power: f64,    // dBm
    temperature: f64,      // Celsius
    efficiency: f64,       // 0.0 to 1.0
    load_factor: f64,      // 0.0 to 1.0
    interference_map: HashMap<u32, f64>, // neighbor_id -> interference level
    energy_consumption: f64, // Watts
    cost_per_watt: f64,    // USD/Watt/hour
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumAllocation {
    cell_id: u32,
    allocated_bands: Vec<SpectrumBand>,
    total_bandwidth: f64,  // MHz
    interference_mitigation: f64, // dB
    spectral_efficiency: f64, // bits/Hz
}

pub struct DynamicSpectrumManager {
    available_spectrum: Vec<SpectrumBand>,
    cell_allocations: HashMap<u32, SpectrumAllocation>,
    interference_matrix: HashMap<(u32, u32), f64>,
    optimization_history: Vec<SpectrumOptimizationResult>,
    
    // Machine learning components
    spectrum_predictor: SpectrumPredictor,
    interference_predictor: InterferencePredictor,
}

pub struct PowerController {
    power_settings: PowerControlSettings,
    cell_states: HashMap<u32, CellPowerState>,
    optimization_algorithm: PowerOptimizationAlgorithm,
    energy_model: EnergyConsumptionModel,
    
    // Control parameters
    update_interval: Duration,
    last_update: Instant,
    total_energy_saved: f64,
    cost_savings: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectrumOptimizationResult {
    timestamp: u64,
    total_throughput: f64,
    interference_reduction: f64,
    spectrum_efficiency: f64,
    fairness_index: f64,
    energy_efficiency: f64,
}

pub struct SpectrumPredictor {
    // Simple neural network for spectrum demand prediction
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    learning_rate: f64,
    prediction_horizon: Duration,
}

pub struct InterferencePredictor {
    // Interference prediction model
    interference_model: HashMap<String, f64>,
    spatial_correlation: HashMap<(u32, u32), f64>,
    temporal_patterns: Vec<f64>,
}

pub struct PowerOptimizationAlgorithm {
    // Reinforcement learning for power control
    q_table: HashMap<String, HashMap<String, f64>>,
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
    exploration_decay: f64,
}

pub struct EnergyConsumptionModel {
    // Power consumption model
    base_power: f64,       // Watts
    power_amplifier_efficiency: f64,
    cooling_factor: f64,
    processing_overhead: f64,
    
    // Cost model
    electricity_rate: f64,  // USD/kWh
    carbon_factor: f64,     // kg CO2/kWh
    operational_cost: f64,  // USD/hour
}

impl DynamicSpectrumManager {
    pub fn new() -> Self {
        let mut manager = DynamicSpectrumManager {
            available_spectrum: Vec::new(),
            cell_allocations: HashMap::new(),
            interference_matrix: HashMap::new(),
            optimization_history: Vec::new(),
            spectrum_predictor: SpectrumPredictor::new(),
            interference_predictor: InterferencePredictor::new(),
        };
        
        manager.initialize_spectrum_bands();
        manager
    }
    
    fn initialize_spectrum_bands(&mut self) {
        // Initialize various spectrum bands with realistic parameters
        let bands = vec![
            SpectrumBand {
                frequency: 700.0,
                bandwidth: 10.0,
                power_limit: 46.0,
                interference_level: -100.0,
                utilization: 0.3,
                priority: 8,
                regulatory_class: "700MHz".to_string(),
            },
            SpectrumBand {
                frequency: 850.0,
                bandwidth: 5.0,
                power_limit: 43.0,
                interference_level: -95.0,
                utilization: 0.7,
                priority: 7,
                regulatory_class: "850MHz".to_string(),
            },
            SpectrumBand {
                frequency: 1800.0,
                bandwidth: 20.0,
                power_limit: 43.0,
                interference_level: -90.0,
                utilization: 0.5,
                priority: 6,
                regulatory_class: "1800MHz".to_string(),
            },
            SpectrumBand {
                frequency: 2100.0,
                bandwidth: 15.0,
                power_limit: 43.0,
                interference_level: -85.0,
                utilization: 0.6,
                priority: 5,
                regulatory_class: "2100MHz".to_string(),
            },
            SpectrumBand {
                frequency: 2600.0,
                bandwidth: 20.0,
                power_limit: 43.0,
                interference_level: -80.0,
                utilization: 0.4,
                priority: 4,
                regulatory_class: "2600MHz".to_string(),
            },
            SpectrumBand {
                frequency: 3500.0,
                bandwidth: 100.0,
                power_limit: 40.0,
                interference_level: -75.0,
                utilization: 0.2,
                priority: 9,
                regulatory_class: "3.5GHz".to_string(),
            },
            SpectrumBand {
                frequency: 28000.0,
                bandwidth: 400.0,
                power_limit: 35.0,
                interference_level: -70.0,
                utilization: 0.1,
                priority: 10,
                regulatory_class: "28GHz".to_string(),
            },
        ];
        
        self.available_spectrum = bands;
    }
    
    pub fn optimize_spectrum_allocation(&mut self, cell_demands: &HashMap<u32, f64>) -> Vec<SpectrumAllocation> {
        let mut allocations = Vec::new();
        
        // Sort cells by demand priority
        let mut sorted_cells: Vec<(u32, f64)> = cell_demands.iter()
            .map(|(&id, &demand)| (id, demand))
            .collect();
        sorted_cells.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Allocate spectrum using intelligent algorithm
        for (cell_id, demand) in sorted_cells {
            let allocation = self.allocate_spectrum_for_cell(cell_id, demand);
            allocations.push(allocation);
        }
        
        // Optimize for interference mitigation
        self.optimize_interference_mitigation(&mut allocations);
        
        // Update allocation history
        for allocation in &allocations {
            self.cell_allocations.insert(allocation.cell_id, allocation.clone());
        }
        
        allocations
    }
    
    fn allocate_spectrum_for_cell(&mut self, cell_id: u32, demand: f64) -> SpectrumAllocation {
        let mut allocated_bands = Vec::new();
        let mut total_bandwidth = 0.0;
        let mut remaining_demand = demand;
        
        // Predict future spectrum needs
        let predicted_demand = self.spectrum_predictor.predict_demand(cell_id, demand);
        
        // Sort spectrum bands by priority and availability
        let mut available_bands = self.available_spectrum.clone();
        available_bands.sort_by(|a, b| {
            let score_a = (1.0 - a.utilization) * a.priority as f64;
            let score_b = (1.0 - b.utilization) * b.priority as f64;
            score_b.partial_cmp(&score_a).unwrap()
        });
        
        // Allocate spectrum bands
        for band in available_bands {
            if remaining_demand <= 0.0 {
                break;
            }
            
            // Check if this band can be allocated
            if band.utilization < 0.9 {
                let available_capacity = (1.0 - band.utilization) * band.bandwidth;
                let allocation_amount = available_capacity.min(remaining_demand);
                
                if allocation_amount > 0.0 {
                    let mut allocated_band = band.clone();
                    allocated_band.bandwidth = allocation_amount;
                    allocated_band.utilization = 1.0; // Mark as allocated
                    
                    allocated_bands.push(allocated_band);
                    total_bandwidth += allocation_amount;
                    remaining_demand -= allocation_amount;
                }
            }
        }
        
        // Calculate performance metrics
        let interference_mitigation = self.calculate_interference_mitigation(cell_id, &allocated_bands);
        let spectral_efficiency = self.calculate_spectral_efficiency(&allocated_bands);
        
        SpectrumAllocation {
            cell_id,
            allocated_bands,
            total_bandwidth,
            interference_mitigation,
            spectral_efficiency,
        }
    }
    
    fn optimize_interference_mitigation(&mut self, allocations: &mut Vec<SpectrumAllocation>) {
        // Coordinated interference mitigation
        for i in 0..allocations.len() {
            for j in i+1..allocations.len() {
                let cell_i = allocations[i].cell_id;
                let cell_j = allocations[j].cell_id;
                
                // Check if cells are neighbors (would have interference)
                if self.are_neighbors(cell_i, cell_j) {
                    self.mitigate_interference(&mut allocations[i], &mut allocations[j]);
                }
            }
        }
    }
    
    fn are_neighbors(&self, cell_i: u32, cell_j: u32) -> bool {
        // Simplified neighbor detection - in practice would use geographic data
        (cell_i as i32 - cell_j as i32).abs() <= 5
    }
    
    fn mitigate_interference(&self, allocation_i: &mut SpectrumAllocation, allocation_j: &mut SpectrumAllocation) {
        // Frequency coordination to reduce interference
        let mut interference_reduced = false;
        
        for band_i in &mut allocation_i.allocated_bands {
            for band_j in &mut allocation_j.allocated_bands {
                // Check for frequency overlap
                let freq_diff = (band_i.frequency - band_j.frequency).abs();
                if freq_diff < band_i.bandwidth + band_j.bandwidth {
                    // Reduce power or adjust frequency to mitigate interference
                    band_i.power_limit = (band_i.power_limit - 3.0).max(20.0);
                    band_j.power_limit = (band_j.power_limit - 3.0).max(20.0);
                    
                    allocation_i.interference_mitigation += 3.0;
                    allocation_j.interference_mitigation += 3.0;
                    
                    interference_reduced = true;
                }
            }
        }
    }
    
    fn calculate_interference_mitigation(&self, cell_id: u32, bands: &[SpectrumBand]) -> f64 {
        let mut total_mitigation = 0.0;
        
        for band in bands {
            // Calculate interference reduction based on frequency coordination
            let base_interference = -80.0; // dBm
            let mitigation = base_interference - band.interference_level;
            total_mitigation += mitigation;
        }
        
        total_mitigation
    }
    
    fn calculate_spectral_efficiency(&self, bands: &[SpectrumBand]) -> f64 {
        let mut total_efficiency = 0.0;
        let mut total_bandwidth = 0.0;
        
        for band in bands {
            // Shannon capacity approximation
            let snr_linear = 10.0_f64.powf((band.power_limit - band.interference_level) / 10.0);
            let efficiency = (1.0 + snr_linear).log2();
            
            total_efficiency += efficiency * band.bandwidth;
            total_bandwidth += band.bandwidth;
        }
        
        if total_bandwidth > 0.0 {
            total_efficiency / total_bandwidth
        } else {
            0.0
        }
    }
    
    pub fn get_spectrum_utilization_report(&self) -> String {
        let mut report = String::new();
        report.push_str("📡 SPECTRUM UTILIZATION REPORT\n");
        report.push_str("============================\n\n");
        
        for band in &self.available_spectrum {
            report.push_str(&format!(
                "🔸 {} ({:.0} MHz):\n  Utilization: {:.1}%\n  Bandwidth: {:.1} MHz\n  Power Limit: {:.1} dBm\n  Interference: {:.1} dBm\n  Priority: {}\n\n",
                band.regulatory_class,
                band.frequency,
                band.utilization * 100.0,
                band.bandwidth,
                band.power_limit,
                band.interference_level,
                band.priority
            ));
        }
        
        report
    }
}

impl PowerController {
    pub fn new() -> Self {
        let power_settings = PowerControlSettings {
            min_power: 20.0,      // dBm
            max_power: 46.0,      // dBm
            step_size: 1.0,       // dB
            target_sinr: 10.0,    // dB
            power_headroom: 3.0,  // dB
            thermal_limit: 75.0,  // Celsius
        };
        
        PowerController {
            power_settings,
            cell_states: HashMap::new(),
            optimization_algorithm: PowerOptimizationAlgorithm::new(),
            energy_model: EnergyConsumptionModel::new(),
            update_interval: Duration::from_millis(100),
            last_update: Instant::now(),
            total_energy_saved: 0.0,
            cost_savings: 0.0,
        }
    }
    
    pub fn initialize_cells(&mut self, cell_ids: &[u32]) {
        for &cell_id in cell_ids {
            let cell_state = CellPowerState {
                cell_id,
                current_power: 30.0,  // dBm
                temperature: 45.0,    // Celsius
                efficiency: 0.7,      // 70%
                load_factor: 0.5,     // 50%
                interference_map: HashMap::new(),
                energy_consumption: 25.0, // Watts
                cost_per_watt: 0.12,  // USD/Watt/hour
            };
            
            self.cell_states.insert(cell_id, cell_state);
        }
    }
    
    pub fn optimize_power_control(&mut self) -> HashMap<u32, f64> {
        let mut power_adjustments = HashMap::new();
        
        for (cell_id, cell_state) in self.cell_states.iter_mut() {
            // Get current state
            let current_power = cell_state.current_power;
            let load_factor = cell_state.load_factor;
            let temperature = cell_state.temperature;
            
            // Calculate optimal power using reinforcement learning
            let state_key = format!("{:.1}_{:.1}_{:.1}", current_power, load_factor, temperature);
            let optimal_power = self.optimization_algorithm.get_optimal_power(&state_key, &self.power_settings);
            
            // Apply power adjustment
            let new_power = self.apply_power_adjustment(cell_state, optimal_power);
            let power_change = new_power - current_power;
            
            power_adjustments.insert(*cell_id, power_change);
            
            // Update energy consumption
            let old_energy = cell_state.energy_consumption;
            cell_state.energy_consumption = self.energy_model.calculate_consumption(new_power, load_factor);
            cell_state.current_power = new_power;
            
            // Track energy savings
            let energy_saved = old_energy - cell_state.energy_consumption;
            self.total_energy_saved += energy_saved;
            self.cost_savings += energy_saved * cell_state.cost_per_watt;
            
            // Update algorithm with reward
            let reward = self.calculate_reward(energy_saved, cell_state.efficiency);
            self.optimization_algorithm.update_q_value(&state_key, optimal_power, reward);
        }
        
        power_adjustments
    }
    
    fn apply_power_adjustment(&self, cell_state: &mut CellPowerState, target_power: f64) -> f64 {
        // Apply thermal constraints
        let thermal_limit = if cell_state.temperature > self.power_settings.thermal_limit {
            cell_state.current_power - 5.0 // Reduce power to cool down
        } else {
            target_power
        };
        
        // Apply regulatory limits
        let regulated_power = thermal_limit
            .max(self.power_settings.min_power)
            .min(self.power_settings.max_power);
        
        // Apply interference constraints
        let interference_adjusted = self.adjust_for_interference(cell_state, regulated_power);
        
        interference_adjusted
    }
    
    fn adjust_for_interference(&self, cell_state: &CellPowerState, target_power: f64) -> f64 {
        let mut adjusted_power = target_power;
        
        // Reduce power if high interference from neighbors
        for (&neighbor_id, &interference_level) in &cell_state.interference_map {
            if interference_level > -80.0 { // High interference threshold
                adjusted_power = (adjusted_power - 2.0).max(self.power_settings.min_power);
            }
        }
        
        adjusted_power
    }
    
    fn calculate_reward(&self, energy_saved: f64, efficiency: f64) -> f64 {
        let energy_reward = energy_saved * 0.1;
        let efficiency_reward = efficiency * 0.05;
        
        energy_reward + efficiency_reward
    }
    
    pub fn get_energy_report(&self) -> String {
        let total_consumption: f64 = self.cell_states.values()
            .map(|cell| cell.energy_consumption)
            .sum();
        
        let average_efficiency: f64 = self.cell_states.values()
            .map(|cell| cell.efficiency)
            .sum::<f64>() / self.cell_states.len() as f64;
        
        format!(
            "⚡ ENERGY CONSUMPTION REPORT\n\
             ===========================\n\
             \n\
             📊 Current Metrics:\n\
             • Total Consumption: {:.2} kW\n\
             • Average Efficiency: {:.1}%\n\
             • Energy Saved: {:.2} kWh\n\
             • Cost Savings: ${:.2}\n\
             \n\
             🔋 Per-Cell Breakdown:\n",
            total_consumption / 1000.0,
            average_efficiency * 100.0,
            self.total_energy_saved / 1000.0,
            self.cost_savings
        )
    }
}

impl SpectrumPredictor {
    pub fn new() -> Self {
        SpectrumPredictor {
            weights: vec![vec![0.1; 5]; 3], // Simple 3-layer network
            biases: vec![0.0; 3],
            learning_rate: 0.01,
            prediction_horizon: Duration::from_secs(3600), // 1 hour
        }
    }
    
    pub fn predict_demand(&mut self, cell_id: u32, current_demand: f64) -> f64 {
        // Simple prediction based on historical patterns
        let time_factor = (cell_id % 24) as f64 / 24.0; // Hour of day
        let demand_factor = current_demand;
        
        // Predict future demand using simple heuristics
        let predicted_demand = current_demand * (1.0 + 0.1 * time_factor.sin());
        
        predicted_demand.max(0.0).min(1.0)
    }
}

impl InterferencePredictor {
    pub fn new() -> Self {
        InterferencePredictor {
            interference_model: HashMap::new(),
            spatial_correlation: HashMap::new(),
            temporal_patterns: vec![0.0; 24], // 24-hour pattern
        }
    }
}

impl PowerOptimizationAlgorithm {
    pub fn new() -> Self {
        PowerOptimizationAlgorithm {
            q_table: HashMap::new(),
            learning_rate: 0.1,
            discount_factor: 0.9,
            epsilon: 0.1,
            exploration_decay: 0.995,
        }
    }
    
    pub fn get_optimal_power(&mut self, state: &str, settings: &PowerControlSettings) -> f64 {
        // Epsilon-greedy action selection
        let mut rng = rand::thread_rng();
        
        if rng.gen::<f64>() < self.epsilon {
            // Random exploration
            rng.gen_range(settings.min_power..settings.max_power)
        } else {
            // Greedy exploitation
            self.get_best_action(state, settings)
        }
    }
    
    fn get_best_action(&self, state: &str, settings: &PowerControlSettings) -> f64 {
        if let Some(actions) = self.q_table.get(state) {
            let mut best_power = settings.min_power;
            let mut best_value = f64::NEG_INFINITY;
            
            for (&power_str, &value) in actions {
                if let Ok(power) = power_str.parse::<f64>() {
                    if value > best_value {
                        best_value = value;
                        best_power = power;
                    }
                }
            }
            
            best_power
        } else {
            // Default to mid-range power
            (settings.min_power + settings.max_power) / 2.0
        }
    }
    
    pub fn update_q_value(&mut self, state: &str, action: f64, reward: f64) {
        let action_str = format!("{:.1}", action);
        
        // Initialize state if not exists
        if !self.q_table.contains_key(state) {
            self.q_table.insert(state.to_string(), HashMap::new());
        }
        
        // Update Q-value
        let current_value = self.q_table.get(state).unwrap()
            .get(&action_str).unwrap_or(&0.0);
        
        let new_value = current_value + self.learning_rate * (reward - current_value);
        
        self.q_table.get_mut(state).unwrap()
            .insert(action_str, new_value);
        
        // Decay exploration
        self.epsilon *= self.exploration_decay;
        self.epsilon = self.epsilon.max(0.01);
    }
}

impl EnergyConsumptionModel {
    pub fn new() -> Self {
        EnergyConsumptionModel {
            base_power: 10.0,                // 10W base consumption
            power_amplifier_efficiency: 0.35, // 35% efficiency
            cooling_factor: 0.2,             // 20% for cooling
            processing_overhead: 5.0,        // 5W processing
            electricity_rate: 0.12,          // $0.12/kWh
            carbon_factor: 0.4,              // 0.4 kg CO2/kWh
            operational_cost: 2.0,           // $2/hour operational cost
        }
    }
    
    pub fn calculate_consumption(&self, power_dbm: f64, load_factor: f64) -> f64 {
        // Convert dBm to watts
        let power_watts = 10.0_f64.powf((power_dbm - 30.0) / 10.0);
        
        // Calculate total consumption
        let amplifier_power = power_watts / self.power_amplifier_efficiency;
        let cooling_power = amplifier_power * self.cooling_factor;
        let processing_power = self.processing_overhead * load_factor;
        
        self.base_power + amplifier_power + cooling_power + processing_power
    }
}

fn main() {
    println!("🎯 SPECTRUM AND POWER CONTROLLER INITIALIZING...");
    
    // Initialize spectrum manager
    let mut spectrum_manager = DynamicSpectrumManager::new();
    
    // Initialize power controller
    let mut power_controller = PowerController::new();
    let cell_ids: Vec<u32> = (0..50).collect();
    power_controller.initialize_cells(&cell_ids);
    
    // Simulate spectrum demands
    let mut cell_demands = HashMap::new();
    for cell_id in 0..50 {
        cell_demands.insert(cell_id, rand::random::<f64>());
    }
    
    // Optimize spectrum allocation
    println!("\n📡 OPTIMIZING SPECTRUM ALLOCATION...");
    let allocations = spectrum_manager.optimize_spectrum_allocation(&cell_demands);
    
    for allocation in &allocations[..5] { // Show first 5 cells
        println!("  Cell {}: {:.1} MHz allocated, {:.2} bits/Hz efficiency", 
                 allocation.cell_id, allocation.total_bandwidth, allocation.spectral_efficiency);
    }
    
    // Optimize power control
    println!("\n⚡ OPTIMIZING POWER CONTROL...");
    let power_adjustments = power_controller.optimize_power_control();
    
    for (&cell_id, &adjustment) in power_adjustments.iter().take(5) {
        println!("  Cell {}: Power adjusted by {:.1} dBm", cell_id, adjustment);
    }
    
    // Generate reports
    println!("\n{}", spectrum_manager.get_spectrum_utilization_report());
    println!("{}", power_controller.get_energy_report());
    
    println!("✅ SPECTRUM AND POWER OPTIMIZATION COMPLETED");
}