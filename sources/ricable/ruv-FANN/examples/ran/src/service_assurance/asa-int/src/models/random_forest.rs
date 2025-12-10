use super::*;
use crate::{Config, Result, Error};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier {
    trees: Vec<DecisionTree>,
    classes: Vec<String>,
    metrics: ModelMetrics,
    n_trees: usize,
    max_depth: usize,
    min_samples_split: usize,
    feature_importance: Option<Array1<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTree {
    root: Option<TreeNode>,
    classes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    feature_idx: Option<usize>,
    threshold: Option<f64>,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    class_counts: HashMap<String, usize>,
    is_leaf: bool,
}

impl RandomForestClassifier {
    pub fn new(config: &Config) -> Result<Self> {
        Ok(Self {
            trees: Vec::new(),
            classes: config.classification.interference_classes.clone(),
            metrics: ModelMetrics {
                accuracy: 0.0,
                precision: 0.0,
                recall: 0.0,
                f1_score: 0.0,
                confusion_matrix: Array2::zeros((config.classification.interference_classes.len(), config.classification.interference_classes.len())),
                class_metrics: HashMap::new(),
                training_time_seconds: 0.0,
                prediction_time_ms: 0.0,
                last_updated: Utc::now(),
            },
            n_trees: 100,
            max_depth: 10,
            min_samples_split: 2,
            feature_importance: None,
        })
    }
    
    fn bootstrap_sample(&self, features: &Array2<f64>, labels: &[String]) -> (Array2<f64>, Vec<String>) {
        let n_samples = features.nrows();
        let mut bootstrap_features = Array2::zeros((n_samples, features.ncols()));
        let mut bootstrap_labels = Vec::new();
        
        for i in 0..n_samples {
            let random_idx = rand::random::<usize>() % n_samples;
            bootstrap_features.row_mut(i).assign(&features.row(random_idx));
            bootstrap_labels.push(labels[random_idx].clone());
        }
        
        (bootstrap_features, bootstrap_labels)
    }
    
    fn calculate_feature_importance(&mut self, n_features: usize) {
        let mut importance = Array1::zeros(n_features);
        
        for tree in &self.trees {
            if let Some(tree_importance) = tree.get_feature_importance(n_features) {
                importance = importance + tree_importance;
            }
        }
        
        // Normalize by number of trees
        importance = importance / self.trees.len() as f64;
        
        // Normalize to sum to 1
        let sum = importance.sum();
        if sum > 0.0 {
            importance = importance / sum;
        }
        
        self.feature_importance = Some(importance);
    }
}

impl InterferenceClassifier for RandomForestClassifier {
    fn train(&mut self, features: &Array2<f64>, labels: &[String]) -> Result<TrainingResult> {
        let start_time = std::time::Instant::now();
        
        if features.nrows() != labels.len() {
            return Err(Error::InvalidInput("Feature matrix and labels length mismatch".to_string()));
        }
        
        // Train multiple decision trees
        self.trees.clear();
        
        for _ in 0..self.n_trees {
            let (bootstrap_features, bootstrap_labels) = self.bootstrap_sample(features, labels);
            
            let mut tree = DecisionTree::new(self.classes.clone());
            tree.train(&bootstrap_features, &bootstrap_labels, self.max_depth, self.min_samples_split)?;
            
            self.trees.push(tree);
        }
        
        // Calculate feature importance
        self.calculate_feature_importance(features.ncols());
        
        let training_time = start_time.elapsed().as_secs_f64();
        
        // Evaluate on training data
        let mut predictions = Vec::new();
        for row in features.rows() {
            let result = self.predict(&row.to_owned())?;
            predictions.push(result.predicted_class);
        }
        
        let metrics = compute_metrics(labels, &predictions, &self.classes, training_time, 0.0)?;
        
        // Validate accuracy requirement
        validate_accuracy_requirement(&metrics)?;
        
        self.metrics = metrics.clone();
        
        Ok(TrainingResult {
            model_id: format!("random_forest_{}", Utc::now().timestamp()),
            metrics: metrics.clone(),
            epochs_trained: 1,
            convergence_achieved: true,
            final_loss: 0.0,
            validation_metrics: metrics,
        })
    }
    
    fn predict(&self, features: &Array1<f64>) -> Result<ClassificationResult> {
        let start_time = std::time::Instant::now();
        
        if self.trees.is_empty() {
            return Err(Error::ModelTraining("Model not trained yet".to_string()));
        }
        
        // Get predictions from all trees
        let mut class_votes = HashMap::new();
        for class in &self.classes {
            class_votes.insert(class.clone(), 0);
        }
        
        for tree in &self.trees {
            let tree_prediction = tree.predict(features)?;
            *class_votes.get_mut(&tree_prediction).unwrap() += 1;
        }
        
        // Find majority vote
        let (predicted_class, vote_count) = class_votes.iter()
            .max_by_key(|(_, &count)| count)
            .map(|(class, &count)| (class.clone(), count))
            .unwrap();
        
        let confidence = vote_count as f64 / self.trees.len() as f64;
        
        // Build class probabilities
        let mut class_probabilities = HashMap::new();
        for (class, &count) in &class_votes {
            let probability = count as f64 / self.trees.len() as f64;
            class_probabilities.insert(class.clone(), probability);
        }
        
        let prediction_time = start_time.elapsed().as_millis() as f64;
        
        Ok(ClassificationResult {
            predicted_class,
            confidence,
            class_probabilities,
            feature_importance: self.feature_importance.clone(),
            timestamp: Utc::now(),
        })
    }
    
    fn predict_proba(&self, features: &Array1<f64>) -> Result<HashMap<String, f64>> {
        let result = self.predict(features)?;
        Ok(result.class_probabilities)
    }
    
    fn get_feature_importance(&self) -> Option<Array1<f64>> {
        self.feature_importance.clone()
    }
    
    fn get_metrics(&self) -> &ModelMetrics {
        &self.metrics
    }
    
    fn validate_performance(&self, config: &Config) -> Result<bool> {
        let metrics = &self.metrics;
        
        if metrics.accuracy < config.performance.min_accuracy {
            return Ok(false);
        }
        
        if metrics.precision < config.performance.min_precision {
            return Ok(false);
        }
        
        if metrics.recall < config.performance.min_recall {
            return Ok(false);
        }
        
        if metrics.f1_score < config.performance.min_f1_score {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    fn save_model(&self, path: &str) -> Result<()> {
        let model_data = serde_json::to_string_pretty(self)?;
        std::fs::write(format!("{}/random_forest.json", path), model_data)?;
        Ok(())
    }
    
    fn load_model(&mut self, path: &str) -> Result<()> {
        let model_data = std::fs::read_to_string(format!("{}/random_forest.json", path))?;
        let model: RandomForestClassifier = serde_json::from_str(&model_data)?;
        *self = model;
        Ok(())
    }
}

impl DecisionTree {
    fn new(classes: Vec<String>) -> Self {
        Self {
            root: None,
            classes,
        }
    }
    
    fn train(&mut self, features: &Array2<f64>, labels: &[String], max_depth: usize, min_samples_split: usize) -> Result<()> {
        self.root = Some(Box::new(self.build_tree(features, labels, 0, max_depth, min_samples_split)?));
        Ok(())
    }
    
    fn build_tree(&self, features: &Array2<f64>, labels: &[String], depth: usize, max_depth: usize, min_samples_split: usize) -> Result<TreeNode> {
        let mut class_counts = HashMap::new();
        for label in labels {
            *class_counts.entry(label.clone()).or_insert(0) += 1;
        }
        
        // Check stopping criteria
        if depth >= max_depth || labels.len() < min_samples_split || class_counts.len() == 1 {
            return Ok(TreeNode {
                feature_idx: None,
                threshold: None,
                left: None,
                right: None,
                class_counts,
                is_leaf: true,
            });
        }
        
        // Find best split (simplified implementation)
        let (best_feature, best_threshold) = self.find_best_split(features, labels)?;
        
        // Split data
        let (left_features, left_labels, right_features, right_labels) = 
            self.split_data(features, labels, best_feature, best_threshold)?;
        
        if left_labels.is_empty() || right_labels.is_empty() {
            return Ok(TreeNode {
                feature_idx: None,
                threshold: None,
                left: None,
                right: None,
                class_counts,
                is_leaf: true,
            });
        }
        
        // Recursively build left and right subtrees
        let left_child = self.build_tree(&left_features, &left_labels, depth + 1, max_depth, min_samples_split)?;
        let right_child = self.build_tree(&right_features, &right_labels, depth + 1, max_depth, min_samples_split)?;
        
        Ok(TreeNode {
            feature_idx: Some(best_feature),
            threshold: Some(best_threshold),
            left: Some(Box::new(left_child)),
            right: Some(Box::new(right_child)),
            class_counts,
            is_leaf: false,
        })
    }
    
    fn find_best_split(&self, features: &Array2<f64>, labels: &[String]) -> Result<(usize, f64)> {
        let mut best_feature = 0;
        let mut best_threshold = 0.0;
        let mut best_gini = f64::INFINITY;
        
        for feature_idx in 0..features.ncols() {
            let feature_values: Vec<f64> = features.column(feature_idx).to_vec();
            let mut thresholds: Vec<f64> = feature_values.clone();
            thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            thresholds.dedup();
            
            for &threshold in &thresholds {
                let gini = self.calculate_gini_impurity(features, labels, feature_idx, threshold);
                if gini < best_gini {
                    best_gini = gini;
                    best_feature = feature_idx;
                    best_threshold = threshold;
                }
            }
        }
        
        Ok((best_feature, best_threshold))
    }
    
    fn calculate_gini_impurity(&self, features: &Array2<f64>, labels: &[String], feature_idx: usize, threshold: f64) -> f64 {
        let mut left_counts = HashMap::new();
        let mut right_counts = HashMap::new();
        let mut left_total = 0;
        let mut right_total = 0;
        
        for (i, label) in labels.iter().enumerate() {
            if features[[i, feature_idx]] <= threshold {
                *left_counts.entry(label.clone()).or_insert(0) += 1;
                left_total += 1;
            } else {
                *right_counts.entry(label.clone()).or_insert(0) += 1;
                right_total += 1;
            }
        }
        
        let left_gini = self.gini_index(&left_counts, left_total);
        let right_gini = self.gini_index(&right_counts, right_total);
        
        let total = left_total + right_total;
        if total == 0 {
            return 0.0;
        }
        
        (left_total as f64 / total as f64) * left_gini + (right_total as f64 / total as f64) * right_gini
    }
    
    fn gini_index(&self, counts: &HashMap<String, usize>, total: usize) -> f64 {
        if total == 0 {
            return 0.0;
        }
        
        let mut gini = 1.0;
        for &count in counts.values() {
            let probability = count as f64 / total as f64;
            gini -= probability * probability;
        }
        
        gini
    }
    
    fn split_data(&self, features: &Array2<f64>, labels: &[String], feature_idx: usize, threshold: f64) -> Result<(Array2<f64>, Vec<String>, Array2<f64>, Vec<String>)> {
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for i in 0..features.nrows() {
            if features[[i, feature_idx]] <= threshold {
                left_indices.push(i);
            } else {
                right_indices.push(i);
            }
        }
        
        let left_features = Array2::from_shape_vec(
            (left_indices.len(), features.ncols()),
            left_indices.iter()
                .flat_map(|&i| features.row(i).to_vec())
                .collect(),
        ).map_err(|e| Error::InvalidInput(format!("Shape error: {}", e)))?;
        
        let right_features = Array2::from_shape_vec(
            (right_indices.len(), features.ncols()),
            right_indices.iter()
                .flat_map(|&i| features.row(i).to_vec())
                .collect(),
        ).map_err(|e| Error::InvalidInput(format!("Shape error: {}", e)))?;
        
        let left_labels: Vec<String> = left_indices.iter().map(|&i| labels[i].clone()).collect();
        let right_labels: Vec<String> = right_indices.iter().map(|&i| labels[i].clone()).collect();
        
        Ok((left_features, left_labels, right_features, right_labels))
    }
    
    fn predict(&self, features: &Array1<f64>) -> Result<String> {
        if let Some(root) = &self.root {
            self.predict_node(root, features)
        } else {
            Err(Error::ModelTraining("Tree not trained".to_string()))
        }
    }
    
    fn predict_node(&self, node: &TreeNode, features: &Array1<f64>) -> Result<String> {
        if node.is_leaf {
            // Return the class with the highest count
            let (predicted_class, _) = node.class_counts.iter()
                .max_by_key(|(_, &count)| count)
                .ok_or_else(|| Error::Classification("No class found in leaf node".to_string()))?;
            
            Ok(predicted_class.clone())
        } else {
            let feature_idx = node.feature_idx.unwrap();
            let threshold = node.threshold.unwrap();
            
            if features[feature_idx] <= threshold {
                if let Some(left) = &node.left {
                    self.predict_node(left, features)
                } else {
                    Err(Error::Classification("Missing left child".to_string()))
                }
            } else {
                if let Some(right) = &node.right {
                    self.predict_node(right, features)
                } else {
                    Err(Error::Classification("Missing right child".to_string()))
                }
            }
        }
    }
    
    fn get_feature_importance(&self, n_features: usize) -> Option<Array1<f64>> {
        let mut importance = Array1::zeros(n_features);
        
        if let Some(root) = &self.root {
            self.calculate_node_importance(root, &mut importance);
        }
        
        // Normalize
        let sum = importance.sum();
        if sum > 0.0 {
            importance = importance / sum;
        }
        
        Some(importance)
    }
    
    fn calculate_node_importance(&self, node: &TreeNode, importance: &mut Array1<f64>) {
        if !node.is_leaf {
            if let Some(feature_idx) = node.feature_idx {
                // Simple importance calculation (could be improved)
                importance[feature_idx] += 1.0;
            }
            
            if let Some(left) = &node.left {
                self.calculate_node_importance(left, importance);
            }
            
            if let Some(right) = &node.right {
                self.calculate_node_importance(right, importance);
            }
        }
    }
}