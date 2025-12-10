use crate::aos_heal::{HealingAction, HealingActionType, ActionSequence};
use std::collections::{HashMap, BinaryHeap};
use std::cmp::Ordering;
use serde::{Deserialize, Serialize};

/// Advanced beam search implementation for optimal action sequence generation
#[derive(Debug, Clone)]
pub struct AdvancedBeamSearch {
    pub beam_width: usize,
    pub max_depth: usize,
    pub diversity_penalty: f32,
    pub action_costs: HashMap<HealingActionType, f32>,
    pub dependency_graph: ActionDependencyGraph,
}

impl AdvancedBeamSearch {
    pub fn new(beam_width: usize, max_depth: usize) -> Self {
        let mut action_costs = HashMap::new();
        action_costs.insert(HealingActionType::ProcessRestart, 5.0);
        action_costs.insert(HealingActionType::CellBlocking, 3.0);
        action_costs.insert(HealingActionType::CellUnblocking, 2.0);
        action_costs.insert(HealingActionType::ParameterAdjustment, 1.0);
        action_costs.insert(HealingActionType::LoadBalancing, 4.0);
        action_costs.insert(HealingActionType::ServiceMigration, 6.0);
        action_costs.insert(HealingActionType::ResourceAllocation, 3.5);
        action_costs.insert(HealingActionType::NetworkReconfiguration, 7.0);
        
        Self {
            beam_width,
            max_depth,
            diversity_penalty: 0.1,
            action_costs,
            dependency_graph: ActionDependencyGraph::new(),
        }
    }

    /// Perform beam search to find optimal action sequence
    pub fn search(&self, initial_state: &NetworkState, target_state: &NetworkState) -> Vec<HealingAction> {
        let mut beam = BinaryHeap::new();
        let initial_candidate = SearchCandidate::new(Vec::new(), initial_state.clone(), 0.0);
        beam.push(initial_candidate);
        
        let mut best_sequence = Vec::new();
        let mut best_score = f32::NEG_INFINITY;
        
        for depth in 0..self.max_depth {
            let mut next_beam = BinaryHeap::new();
            let mut current_candidates = Vec::new();
            
            // Extract all candidates from current beam
            while let Some(candidate) = beam.pop() {
                if current_candidates.len() >= self.beam_width {
                    break;
                }
                current_candidates.push(candidate);
            }
            
            // Generate successors for each candidate
            for candidate in current_candidates {
                let successors = self.generate_successors(&candidate, target_state);
                
                for successor in successors {
                    let score = self.evaluate_candidate(&successor, target_state);
                    
                    if score > best_score {
                        best_score = score;
                        best_sequence = successor.actions.clone();
                    }
                    
                    if next_beam.len() < self.beam_width * 2 {
                        next_beam.push(ScoredCandidate {
                            candidate: successor,
                            score,
                        });
                    } else if score > next_beam.peek().unwrap().score {
                        next_beam.pop();
                        next_beam.push(ScoredCandidate {
                            candidate: successor,
                            score,
                        });
                    }
                }
            }
            
            // Prune to beam width with diversity consideration
            beam = self.prune_with_diversity(next_beam);
            
            if beam.is_empty() {
                break;
            }
        }
        
        best_sequence
    }

    /// Generate successor states from a candidate
    fn generate_successors(&self, candidate: &SearchCandidate, target_state: &NetworkState) -> Vec<SearchCandidate> {
        let mut successors = Vec::new();
        
        // Generate all possible next actions
        let possible_actions = self.generate_possible_actions(&candidate.current_state, target_state);
        
        for action in possible_actions {
            // Check if action is valid given current sequence
            if self.is_action_valid(&candidate.actions, &action) {
                let new_state = self.apply_action(&candidate.current_state, &action);
                let new_cost = candidate.cost + self.get_action_cost(&action);
                
                let mut new_actions = candidate.actions.clone();
                new_actions.push(action);
                
                successors.push(SearchCandidate::new(new_actions, new_state, new_cost));
            }
        }
        
        successors
    }

    /// Generate possible actions based on current and target states
    fn generate_possible_actions(&self, current_state: &NetworkState, target_state: &NetworkState) -> Vec<HealingAction> {
        let mut actions = Vec::new();
        
        // Analyze state differences to generate relevant actions
        if current_state.failed_processes.len() > target_state.failed_processes.len() {
            for process in &current_state.failed_processes {
                if !target_state.failed_processes.contains(process) {
                    actions.push(HealingAction {
                        action_type: HealingActionType::ProcessRestart,
                        target_entity: process.clone(),
                        parameters: HashMap::new(),
                        priority: 0.9,
                        confidence: 0.8,
                        estimated_duration: 300,
                        rollback_plan: None,
                    });
                }
            }
        }
        
        // Check for blocked cells that need unblocking
        if current_state.blocked_cells.len() > target_state.blocked_cells.len() {
            for cell in &current_state.blocked_cells {
                if !target_state.blocked_cells.contains(cell) {
                    actions.push(HealingAction {
                        action_type: HealingActionType::CellUnblocking,
                        target_entity: cell.clone(),
                        parameters: HashMap::new(),
                        priority: 0.7,
                        confidence: 0.9,
                        estimated_duration: 120,
                        rollback_plan: None,
                    });
                }
            }
        }
        
        // Check for performance issues requiring parameter adjustment
        if current_state.performance_score < target_state.performance_score {
            actions.push(HealingAction {
                action_type: HealingActionType::ParameterAdjustment,
                target_entity: "network_params".to_string(),
                parameters: self.generate_parameter_adjustments(current_state, target_state),
                priority: 0.6,
                confidence: 0.7,
                estimated_duration: 180,
                rollback_plan: None,
            });
        }
        
        // Load balancing for high utilization
        if current_state.max_utilization > 0.9 && target_state.max_utilization <= 0.8 {
            actions.push(HealingAction {
                action_type: HealingActionType::LoadBalancing,
                target_entity: "load_balancer".to_string(),
                parameters: HashMap::new(),
                priority: 0.8,
                confidence: 0.8,
                estimated_duration: 240,
                rollback_plan: None,
            });
        }
        
        actions
    }

    /// Check if an action is valid given the current action sequence
    fn is_action_valid(&self, current_actions: &[HealingAction], new_action: &HealingAction) -> bool {
        // Check dependencies
        if !self.dependency_graph.check_dependencies(current_actions, new_action) {
            return false;
        }
        
        // Check for conflicts
        if self.has_conflicts(current_actions, new_action) {
            return false;
        }
        
        // Check for redundant actions
        if self.is_redundant(current_actions, new_action) {
            return false;
        }
        
        true
    }

    /// Check if new action conflicts with existing actions
    fn has_conflicts(&self, current_actions: &[HealingAction], new_action: &HealingAction) -> bool {
        for action in current_actions {
            if action.target_entity == new_action.target_entity {
                match (&action.action_type, &new_action.action_type) {
                    (HealingActionType::CellBlocking, HealingActionType::CellUnblocking) => return true,
                    (HealingActionType::CellUnblocking, HealingActionType::CellBlocking) => return true,
                    _ => {}
                }
            }
        }
        false
    }

    /// Check if action is redundant
    fn is_redundant(&self, current_actions: &[HealingAction], new_action: &HealingAction) -> bool {
        current_actions.iter().any(|action| {
            action.action_type == new_action.action_type &&
            action.target_entity == new_action.target_entity
        })
    }

    /// Apply action to current state and return new state
    fn apply_action(&self, current_state: &NetworkState, action: &HealingAction) -> NetworkState {
        let mut new_state = current_state.clone();
        
        match action.action_type {
            HealingActionType::ProcessRestart => {
                new_state.failed_processes.retain(|p| p != &action.target_entity);
                new_state.performance_score += 0.1;
            },
            HealingActionType::CellUnblocking => {
                new_state.blocked_cells.retain(|c| c != &action.target_entity);
                new_state.performance_score += 0.05;
            },
            HealingActionType::CellBlocking => {
                new_state.blocked_cells.push(action.target_entity.clone());
                new_state.performance_score -= 0.02;
            },
            HealingActionType::ParameterAdjustment => {
                new_state.performance_score += 0.15;
            },
            HealingActionType::LoadBalancing => {
                new_state.max_utilization *= 0.8;
                new_state.performance_score += 0.2;
            },
            _ => {}
        }
        
        new_state.performance_score = new_state.performance_score.clamp(0.0, 1.0);
        new_state.max_utilization = new_state.max_utilization.clamp(0.0, 1.0);
        
        new_state
    }

    /// Get cost of an action
    fn get_action_cost(&self, action: &HealingAction) -> f32 {
        self.action_costs.get(&action.action_type).unwrap_or(&1.0) * 
        (1.0 - action.confidence) * action.priority
    }

    /// Evaluate a candidate solution
    fn evaluate_candidate(&self, candidate: &SearchCandidate, target_state: &NetworkState) -> f32 {
        let state_similarity = self.calculate_state_similarity(&candidate.current_state, target_state);
        let action_efficiency = self.calculate_action_efficiency(&candidate.actions);
        let diversity_bonus = self.calculate_diversity_bonus(&candidate.actions);
        
        state_similarity * 0.6 + action_efficiency * 0.3 + diversity_bonus * 0.1 - candidate.cost * 0.01
    }

    /// Calculate similarity between current and target states
    fn calculate_state_similarity(&self, current: &NetworkState, target: &NetworkState) -> f32 {
        let mut similarity = 0.0;
        
        // Performance score similarity
        similarity += 1.0 - (current.performance_score - target.performance_score).abs();
        
        // Utilization similarity
        similarity += 1.0 - (current.max_utilization - target.max_utilization).abs();
        
        // Failed processes similarity
        let failed_overlap = current.failed_processes.iter()
            .filter(|p| target.failed_processes.contains(p))
            .count() as f32;
        let failed_union = (current.failed_processes.len() + target.failed_processes.len()) as f32 - failed_overlap;
        if failed_union > 0.0 {
            similarity += failed_overlap / failed_union;
        } else {
            similarity += 1.0;
        }
        
        // Blocked cells similarity
        let blocked_overlap = current.blocked_cells.iter()
            .filter(|c| target.blocked_cells.contains(c))
            .count() as f32;
        let blocked_union = (current.blocked_cells.len() + target.blocked_cells.len()) as f32 - blocked_overlap;
        if blocked_union > 0.0 {
            similarity += blocked_overlap / blocked_union;
        } else {
            similarity += 1.0;
        }
        
        similarity / 4.0
    }

    /// Calculate action efficiency
    fn calculate_action_efficiency(&self, actions: &[HealingAction]) -> f32 {
        if actions.is_empty() {
            return 1.0;
        }
        
        let total_confidence: f32 = actions.iter().map(|a| a.confidence).sum();
        let total_priority: f32 = actions.iter().map(|a| a.priority).sum();
        let avg_confidence = total_confidence / actions.len() as f32;
        let avg_priority = total_priority / actions.len() as f32;
        
        (avg_confidence + avg_priority) / 2.0
    }

    /// Calculate diversity bonus
    fn calculate_diversity_bonus(&self, actions: &[HealingAction]) -> f32 {
        if actions.is_empty() {
            return 0.0;
        }
        
        let mut action_types = std::collections::HashSet::new();
        for action in actions {
            action_types.insert(&action.action_type);
        }
        
        action_types.len() as f32 / actions.len() as f32
    }

    /// Prune beam with diversity consideration
    fn prune_with_diversity(&self, mut candidates: BinaryHeap<ScoredCandidate>) -> BinaryHeap<ScoredCandidate> {
        let mut pruned = BinaryHeap::new();
        let mut selected_patterns = Vec::new();
        
        while let Some(scored_candidate) = candidates.pop() {
            if pruned.len() >= self.beam_width {
                break;
            }
            
            let pattern = self.extract_action_pattern(&scored_candidate.candidate.actions);
            let diversity_score = self.calculate_pattern_diversity(&pattern, &selected_patterns);
            
            if diversity_score > self.diversity_penalty || pruned.len() < self.beam_width / 2 {
                selected_patterns.push(pattern);
                pruned.push(scored_candidate);
            }
        }
        
        pruned
    }

    /// Extract action pattern for diversity calculation
    fn extract_action_pattern(&self, actions: &[HealingAction]) -> Vec<HealingActionType> {
        actions.iter().map(|a| a.action_type.clone()).collect()
    }

    /// Calculate pattern diversity
    fn calculate_pattern_diversity(&self, pattern: &[HealingActionType], existing_patterns: &[Vec<HealingActionType>]) -> f32 {
        if existing_patterns.is_empty() {
            return 1.0;
        }
        
        let mut min_similarity = f32::INFINITY;
        
        for existing in existing_patterns {
            let similarity = self.calculate_pattern_similarity(pattern, existing);
            min_similarity = min_similarity.min(similarity);
        }
        
        1.0 - min_similarity
    }

    /// Calculate similarity between two action patterns
    fn calculate_pattern_similarity(&self, pattern1: &[HealingActionType], pattern2: &[HealingActionType]) -> f32 {
        let mut common = 0;
        let max_len = pattern1.len().max(pattern2.len());
        
        for i in 0..max_len {
            if i < pattern1.len() && i < pattern2.len() && pattern1[i] == pattern2[i] {
                common += 1;
            }
        }
        
        if max_len == 0 {
            1.0
        } else {
            common as f32 / max_len as f32
        }
    }

    /// Generate parameter adjustments based on state differences
    fn generate_parameter_adjustments(&self, current: &NetworkState, target: &NetworkState) -> HashMap<String, String> {
        let mut adjustments = HashMap::new();
        
        if current.performance_score < target.performance_score {
            adjustments.insert("transmission_power".to_string(), "increase".to_string());
            adjustments.insert("bandwidth_allocation".to_string(), "optimize".to_string());
        }
        
        if current.max_utilization > target.max_utilization {
            adjustments.insert("load_threshold".to_string(), "decrease".to_string());
        }
        
        adjustments
    }
}

/// Network state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkState {
    pub performance_score: f32,
    pub max_utilization: f32,
    pub failed_processes: Vec<String>,
    pub blocked_cells: Vec<String>,
    pub active_alarms: Vec<String>,
    pub resource_usage: HashMap<String, f32>,
}

impl NetworkState {
    pub fn new() -> Self {
        Self {
            performance_score: 0.0,
            max_utilization: 0.0,
            failed_processes: Vec::new(),
            blocked_cells: Vec::new(),
            active_alarms: Vec::new(),
            resource_usage: HashMap::new(),
        }
    }
}

/// Search candidate with action sequence and state
#[derive(Debug, Clone)]
pub struct SearchCandidate {
    pub actions: Vec<HealingAction>,
    pub current_state: NetworkState,
    pub cost: f32,
}

impl SearchCandidate {
    pub fn new(actions: Vec<HealingAction>, current_state: NetworkState, cost: f32) -> Self {
        Self {
            actions,
            current_state,
            cost,
        }
    }
}

/// Scored candidate for priority queue
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub candidate: SearchCandidate,
    pub score: f32,
}

impl Eq for ScoredCandidate {}

impl PartialEq for ScoredCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Ord for ScoredCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.partial_cmp(&other.score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScoredCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Action dependency graph
#[derive(Debug, Clone)]
pub struct ActionDependencyGraph {
    pub dependencies: HashMap<HealingActionType, Vec<HealingActionType>>,
    pub conflicts: HashMap<HealingActionType, Vec<HealingActionType>>,
}

impl ActionDependencyGraph {
    pub fn new() -> Self {
        let mut dependencies = HashMap::new();
        let mut conflicts = HashMap::new();
        
        // Define dependencies
        dependencies.insert(
            HealingActionType::ParameterAdjustment,
            vec![HealingActionType::ProcessRestart],
        );
        dependencies.insert(
            HealingActionType::LoadBalancing,
            vec![HealingActionType::CellUnblocking],
        );
        
        // Define conflicts
        conflicts.insert(
            HealingActionType::CellBlocking,
            vec![HealingActionType::CellUnblocking],
        );
        conflicts.insert(
            HealingActionType::CellUnblocking,
            vec![HealingActionType::CellBlocking],
        );
        
        Self {
            dependencies,
            conflicts,
        }
    }

    /// Check if dependencies are satisfied
    pub fn check_dependencies(&self, current_actions: &[HealingAction], new_action: &HealingAction) -> bool {
        if let Some(deps) = self.dependencies.get(&new_action.action_type) {
            for dep in deps {
                if !current_actions.iter().any(|a| a.action_type == *dep) {
                    return false;
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_beam_search_creation() {
        let beam_search = AdvancedBeamSearch::new(5, 10);
        assert_eq!(beam_search.beam_width, 5);
        assert_eq!(beam_search.max_depth, 10);
    }

    #[test]
    fn test_network_state_creation() {
        let state = NetworkState::new();
        assert_eq!(state.performance_score, 0.0);
        assert_eq!(state.max_utilization, 0.0);
        assert!(state.failed_processes.is_empty());
    }

    #[test]
    fn test_action_dependency_graph() {
        let graph = ActionDependencyGraph::new();
        assert!(!graph.dependencies.is_empty());
        assert!(!graph.conflicts.is_empty());
    }

    #[test]
    fn test_search_candidate_creation() {
        let actions = Vec::new();
        let state = NetworkState::new();
        let candidate = SearchCandidate::new(actions, state, 0.0);
        assert_eq!(candidate.cost, 0.0);
    }
}