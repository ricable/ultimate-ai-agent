"""
Fairness Algorithms and Bias Mitigation Techniques
Advanced algorithms for detecting, measuring, and mitigating bias in AI systems.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import cvxpy as cp
from scipy.optimize import minimize
import warnings

logger = logging.getLogger(__name__)


@dataclass
class BiasMetrics:
    """Container for bias measurement results"""
    demographic_parity: float
    equalized_odds: float
    equality_of_opportunity: float
    predictive_parity: float
    calibration: float
    individual_fairness: float
    overall_bias_score: float
    group_specific_metrics: Dict[str, Dict[str, float]]


@dataclass
class MitigationResult:
    """Results from bias mitigation"""
    original_metrics: BiasMetrics
    mitigated_metrics: BiasMetrics
    improvement_ratio: float
    accuracy_tradeoff: float
    mitigation_method: str
    parameters_used: Dict[str, Any]
    processing_time: float


class FairnessConstraints:
    """Defines fairness constraints for optimization"""
    
    def __init__(self):
        self.constraints = {}
        
    def add_demographic_parity_constraint(self, tolerance: float = 0.1):
        """Add demographic parity constraint"""
        self.constraints['demographic_parity'] = {
            'tolerance': tolerance,
            'type': 'equality'
        }
    
    def add_equalized_odds_constraint(self, tolerance: float = 0.1):
        """Add equalized odds constraint"""
        self.constraints['equalized_odds'] = {
            'tolerance': tolerance,
            'type': 'equality'
        }
    
    def add_accuracy_constraint(self, min_accuracy: float = 0.8):
        """Add minimum accuracy constraint"""
        self.constraints['accuracy'] = {
            'min_value': min_accuracy,
            'type': 'inequality'
        }


class PreProcessingMitigation:
    """Pre-processing bias mitigation techniques"""
    
    def __init__(self):
        self.reweighting_weights = {}
        self.synthetic_data_generator = None
        
    async def reweighting(self, X: np.ndarray, y: np.ndarray, 
                         sensitive_attr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Reweight training samples to achieve fairness"""
        unique_groups = np.unique(sensitive_attr)
        unique_labels = np.unique(y)
        
        # Calculate weights for each group-label combination
        weights = np.ones(len(X))
        
        for group in unique_groups:
            for label in unique_labels:
                mask = (sensitive_attr == group) & (y == label)
                count = np.sum(mask)
                
                if count > 0:
                    # Weight inversely proportional to frequency
                    total_group = np.sum(sensitive_attr == group)
                    total_label = np.sum(y == label)
                    expected_count = (total_group * total_label) / len(X)
                    
                    weight = expected_count / count if count > 0 else 1.0
                    weights[mask] = weight
        
        # Normalize weights
        weights = weights / np.mean(weights)
        self.reweighting_weights = weights
        
        return X, weights
    
    async def disparate_impact_remover(self, X: pd.DataFrame, 
                                     sensitive_attr: str,
                                     repair_level: float = 1.0) -> pd.DataFrame:
        """Remove disparate impact using repair technique"""
        X_repaired = X.copy()
        
        # Get numeric columns only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != sensitive_attr]
        
        if len(numeric_cols) == 0:
            return X_repaired
        
        unique_groups = X[sensitive_attr].unique()
        
        for col in numeric_cols:
            # Calculate group means
            group_means = {}
            overall_mean = X[col].mean()
            
            for group in unique_groups:
                group_mask = X[sensitive_attr] == group
                group_means[group] = X[group_mask][col].mean()
            
            # Apply repair
            for group in unique_groups:
                group_mask = X[sensitive_attr] == group
                current_values = X_repaired[group_mask][col]
                
                # Linear interpolation between original and fair values
                fair_values = overall_mean + (current_values - group_means[group])
                repaired_values = (1 - repair_level) * current_values + repair_level * fair_values
                
                X_repaired.loc[group_mask, col] = repaired_values
        
        return X_repaired
    
    async def synthetic_data_generation(self, X: np.ndarray, y: np.ndarray,
                                      sensitive_attr: np.ndarray,
                                      augmentation_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data to balance representation"""
        unique_groups = np.unique(sensitive_attr)
        
        # Find the largest group size
        max_group_size = max(np.sum(sensitive_attr == group) for group in unique_groups)
        
        X_augmented = [X]
        y_augmented = [y]
        sensitive_augmented = [sensitive_attr]
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            group_size = np.sum(group_mask)
            
            if group_size < max_group_size:
                # Calculate how many samples to generate
                target_size = int(max_group_size * augmentation_factor)
                samples_needed = max(0, target_size - group_size)
                
                if samples_needed > 0:
                    # Generate synthetic samples using SMOTE-like approach
                    group_X = X[group_mask]
                    group_y = y[group_mask]
                    
                    synthetic_X = []
                    synthetic_y = []
                    
                    for _ in range(samples_needed):
                        # Randomly select two samples from the group
                        idx1, idx2 = np.random.choice(len(group_X), 2, replace=True)
                        sample1, sample2 = group_X[idx1], group_X[idx2]
                        
                        # Generate synthetic sample between them
                        alpha = np.random.random()
                        synthetic_sample = alpha * sample1 + (1 - alpha) * sample2
                        synthetic_label = group_y[idx1]  # Use label from first sample
                        
                        synthetic_X.append(synthetic_sample)
                        synthetic_y.append(synthetic_label)
                    
                    if synthetic_X:
                        synthetic_X = np.array(synthetic_X)
                        synthetic_y = np.array(synthetic_y)
                        synthetic_sensitive = np.full(len(synthetic_X), group)
                        
                        X_augmented.append(synthetic_X)
                        y_augmented.append(synthetic_y)
                        sensitive_augmented.append(synthetic_sensitive)
        
        # Combine all data
        X_final = np.vstack(X_augmented)
        y_final = np.hstack(y_augmented)
        sensitive_final = np.hstack(sensitive_augmented)
        
        return X_final, y_final, sensitive_final


class InProcessingMitigation:
    """In-processing bias mitigation techniques"""
    
    def __init__(self):
        self.fairness_constraints = FairnessConstraints()
        self.adversarial_networks = {}
        
    async def adversarial_debiasing(self, X_train: np.ndarray, y_train: np.ndarray,
                                  sensitive_train: np.ndarray,
                                  X_test: np.ndarray = None,
                                  epochs: int = 100,
                                  adversarial_weight: float = 1.0) -> Dict[str, Any]:
        """Adversarial debiasing using neural networks"""
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        s_tensor = torch.FloatTensor(sensitive_train).unsqueeze(1)
        
        input_dim = X_train.shape[1]
        
        # Define classifier network
        class Classifier(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Define adversary network
        class Adversary(nn.Module):
            def __init__(self, hidden_dim=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(1, hidden_dim),  # Takes classifier output
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, predictions):
                return self.network(predictions)
        
        # Initialize networks
        classifier = Classifier(input_dim)
        adversary = Adversary()
        
        # Optimizers
        clf_optimizer = optim.Adam(classifier.parameters(), lr=0.001)
        adv_optimizer = optim.Adam(adversary.parameters(), lr=0.001)
        
        # Loss functions
        criterion = nn.BCELoss()
        
        # Training loop
        training_history = []
        
        for epoch in range(epochs):
            # Train classifier
            clf_optimizer.zero_grad()
            
            y_pred = classifier(X_tensor)
            clf_loss = criterion(y_pred, y_tensor)
            
            # Adversarial loss (classifier tries to fool adversary)
            s_pred = adversary(y_pred)
            adv_loss = criterion(s_pred, 1 - s_tensor)  # Flip labels to fool adversary
            
            total_clf_loss = clf_loss + adversarial_weight * adv_loss
            total_clf_loss.backward(retain_graph=True)
            clf_optimizer.step()
            
            # Train adversary
            adv_optimizer.zero_grad()
            
            y_pred_detached = classifier(X_tensor).detach()
            s_pred = adversary(y_pred_detached)
            adversary_loss = criterion(s_pred, s_tensor)
            
            adversary_loss.backward()
            adv_optimizer.step()
            
            # Log progress
            if epoch % 20 == 0:
                training_history.append({
                    'epoch': epoch,
                    'classifier_loss': clf_loss.item(),
                    'adversary_loss': adversary_loss.item(),
                    'total_loss': total_clf_loss.item()
                })
        
        # Generate predictions
        with torch.no_grad():
            train_predictions = classifier(X_tensor).numpy().flatten()
            
            test_predictions = None
            if X_test is not None:
                X_test_tensor = torch.FloatTensor(X_test)
                test_predictions = classifier(X_test_tensor).numpy().flatten()
        
        return {
            'model': classifier,
            'adversary': adversary,
            'train_predictions': train_predictions,
            'test_predictions': test_predictions,
            'training_history': training_history
        }
    
    async def fairness_constrained_optimization(self, X_train: np.ndarray, y_train: np.ndarray,
                                              sensitive_train: np.ndarray,
                                              fairness_constraint: str = 'demographic_parity',
                                              tolerance: float = 0.1) -> Dict[str, Any]:
        """Fairness-constrained optimization using convex optimization"""
        
        n_samples, n_features = X_train.shape
        
        # Define optimization variables
        w = cp.Variable(n_features)  # Model weights
        b = cp.Variable()            # Bias term
        
        # Predictions
        predictions = X_train @ w + b
        
        # Objective: Minimize logistic loss
        logistic_loss = cp.sum(cp.logistic(-cp.multiply(2*y_train - 1, predictions)))
        objective = cp.Minimize(logistic_loss)
        
        constraints = []
        
        # Add fairness constraints
        if fairness_constraint == 'demographic_parity':
            # Ensure equal positive prediction rates across groups
            unique_groups = np.unique(sensitive_train)
            
            for i, group1 in enumerate(unique_groups):
                for group2 in unique_groups[i+1:]:
                    mask1 = sensitive_train == group1
                    mask2 = sensitive_train == group2
                    
                    if np.sum(mask1) > 0 and np.sum(mask2) > 0:
                        # Mean predictions should be similar across groups
                        mean_pred1 = cp.sum(cp.logistic(X_train[mask1] @ w + b)) / np.sum(mask1)
                        mean_pred2 = cp.sum(cp.logistic(X_train[mask2] @ w + b)) / np.sum(mask2)
                        
                        constraints.append(mean_pred1 - mean_pred2 <= tolerance)
                        constraints.append(mean_pred2 - mean_pred1 <= tolerance)
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.ECOS, verbose=False)
            
            if problem.status == cp.OPTIMAL:
                optimal_w = w.value
                optimal_b = b.value
                
                # Generate predictions
                train_predictions = 1 / (1 + np.exp(-(X_train @ optimal_w + optimal_b)))
                
                return {
                    'weights': optimal_w,
                    'bias': optimal_b,
                    'train_predictions': train_predictions,
                    'optimization_status': 'optimal',
                    'objective_value': problem.value
                }
            else:
                logger.warning(f"Optimization failed with status: {problem.status}")
                return {
                    'optimization_status': problem.status,
                    'error': 'Optimization did not converge'
                }
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            return {
                'optimization_status': 'error',
                'error': str(e)
            }
    
    async def gradient_regularization(self, X_train: np.ndarray, y_train: np.ndarray,
                                    sensitive_train: np.ndarray,
                                    regularization_weight: float = 1.0) -> Dict[str, Any]:
        """Fairness through gradient regularization"""
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        s_tensor = torch.FloatTensor(sensitive_train).unsqueeze(1)
        
        input_dim = X_train.shape[1]
        
        # Define model
        class FairModel(nn.Module):
            def __init__(self, input_dim, hidden_dim=64):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x)
        
        model = FairModel(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        # Training with fairness regularization
        epochs = 100
        training_history = []
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_tensor)
            
            # Primary loss
            primary_loss = criterion(predictions, y_tensor)
            
            # Fairness regularization
            # Penalize correlation between predictions and sensitive attributes
            pred_mean = torch.mean(predictions)
            sens_mean = torch.mean(s_tensor)
            
            covariance = torch.mean((predictions - pred_mean) * (s_tensor - sens_mean))
            regularization_loss = regularization_weight * torch.abs(covariance)
            
            total_loss = primary_loss + regularization_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                training_history.append({
                    'epoch': epoch,
                    'primary_loss': primary_loss.item(),
                    'regularization_loss': regularization_loss.item(),
                    'total_loss': total_loss.item()
                })
        
        # Generate final predictions
        with torch.no_grad():
            final_predictions = model(X_tensor).numpy().flatten()
        
        return {
            'model': model,
            'predictions': final_predictions,
            'training_history': training_history
        }


class PostProcessingMitigation:
    """Post-processing bias mitigation techniques"""
    
    def __init__(self):
        self.calibration_functions = {}
        self.threshold_optimizers = {}
    
    async def threshold_optimization(self, y_true: np.ndarray, y_scores: np.ndarray,
                                   sensitive_attr: np.ndarray,
                                   fairness_metric: str = 'equalized_odds') -> Dict[str, Any]:
        """Optimize classification thresholds for fairness"""
        
        unique_groups = np.unique(sensitive_attr)
        
        if fairness_metric == 'equalized_odds':
            return await self._equalized_odds_threshold_optimization(
                y_true, y_scores, sensitive_attr, unique_groups
            )
        elif fairness_metric == 'demographic_parity':
            return await self._demographic_parity_threshold_optimization(
                y_true, y_scores, sensitive_attr, unique_groups
            )
        else:
            # Default: single threshold optimization
            return await self._single_threshold_optimization(y_true, y_scores)
    
    async def _equalized_odds_threshold_optimization(self, y_true: np.ndarray, y_scores: np.ndarray,
                                                   sensitive_attr: np.ndarray, unique_groups: np.ndarray) -> Dict[str, Any]:
        """Optimize thresholds for equalized odds"""
        
        def objective(thresholds):
            # Calculate TPR and FPR for each group
            tprs = []
            fprs = []
            
            for i, group in enumerate(unique_groups):
                group_mask = sensitive_attr == group
                group_y_true = y_true[group_mask]
                group_y_scores = y_scores[group_mask]
                
                if len(group_y_true) == 0:
                    continue
                
                threshold = thresholds[i]
                group_y_pred = (group_y_scores >= threshold).astype(int)
                
                # Calculate TPR and FPR
                tp = np.sum((group_y_true == 1) & (group_y_pred == 1))
                fn = np.sum((group_y_true == 1) & (group_y_pred == 0))
                fp = np.sum((group_y_true == 0) & (group_y_pred == 1))
                tn = np.sum((group_y_true == 0) & (group_y_pred == 0))
                
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                
                tprs.append(tpr)
                fprs.append(fpr)
            
            # Minimize difference in TPRs and FPRs
            tpr_diff = max(tprs) - min(tprs) if len(tprs) > 1 else 0
            fpr_diff = max(fprs) - min(fprs) if len(fprs) > 1 else 0
            
            return tpr_diff + fpr_diff
        
        # Initial thresholds
        initial_thresholds = np.full(len(unique_groups), 0.5)
        
        # Optimize
        result = minimize(objective, initial_thresholds, 
                         bounds=[(0.01, 0.99)] * len(unique_groups),
                         method='L-BFGS-B')
        
        optimal_thresholds = {group: threshold for group, threshold in zip(unique_groups, result.x)}
        
        # Generate predictions with optimal thresholds
        y_pred_optimized = np.zeros_like(y_true)
        for group, threshold in optimal_thresholds.items():
            group_mask = sensitive_attr == group
            y_pred_optimized[group_mask] = (y_scores[group_mask] >= threshold).astype(int)
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'predictions': y_pred_optimized,
            'optimization_result': result,
            'fairness_metric': 'equalized_odds'
        }
    
    async def _demographic_parity_threshold_optimization(self, y_true: np.ndarray, y_scores: np.ndarray,
                                                       sensitive_attr: np.ndarray, unique_groups: np.ndarray) -> Dict[str, Any]:
        """Optimize thresholds for demographic parity"""
        
        def objective(thresholds):
            positive_rates = []
            
            for i, group in enumerate(unique_groups):
                group_mask = sensitive_attr == group
                group_y_scores = y_scores[group_mask]
                
                if len(group_y_scores) == 0:
                    continue
                
                threshold = thresholds[i]
                positive_rate = np.mean(group_y_scores >= threshold)
                positive_rates.append(positive_rate)
            
            # Minimize difference in positive rates
            return max(positive_rates) - min(positive_rates) if len(positive_rates) > 1 else 0
        
        # Initial thresholds
        initial_thresholds = np.full(len(unique_groups), 0.5)
        
        # Optimize
        result = minimize(objective, initial_thresholds,
                         bounds=[(0.01, 0.99)] * len(unique_groups),
                         method='L-BFGS-B')
        
        optimal_thresholds = {group: threshold for group, threshold in zip(unique_groups, result.x)}
        
        # Generate predictions
        y_pred_optimized = np.zeros_like(y_true)
        for group, threshold in optimal_thresholds.items():
            group_mask = sensitive_attr == group
            y_pred_optimized[group_mask] = (y_scores[group_mask] >= threshold).astype(int)
        
        return {
            'optimal_thresholds': optimal_thresholds,
            'predictions': y_pred_optimized,
            'optimization_result': result,
            'fairness_metric': 'demographic_parity'
        }
    
    async def _single_threshold_optimization(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """Single threshold optimization for maximum accuracy"""
        
        def objective(threshold):
            y_pred = (y_scores >= threshold).astype(int)
            return -accuracy_score(y_true, y_pred)  # Negative for minimization
        
        result = minimize(objective, [0.5], bounds=[(0.01, 0.99)], method='L-BFGS-B')
        optimal_threshold = result.x[0]
        
        y_pred_optimized = (y_scores >= optimal_threshold).astype(int)
        
        return {
            'optimal_threshold': optimal_threshold,
            'predictions': y_pred_optimized,
            'optimization_result': result,
            'fairness_metric': 'accuracy'
        }
    
    async def calibration_based_postprocessing(self, y_true: np.ndarray, y_scores: np.ndarray,
                                             sensitive_attr: np.ndarray) -> Dict[str, Any]:
        """Calibrate predictions to ensure fairness across groups"""
        
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.dummy import DummyClassifier
        
        unique_groups = np.unique(sensitive_attr)
        calibrated_scores = np.zeros_like(y_scores)
        calibration_functions = {}
        
        for group in unique_groups:
            group_mask = sensitive_attr == group
            group_y_true = y_true[group_mask]
            group_y_scores = y_scores[group_mask]
            
            if len(group_y_true) > 10:  # Minimum samples for calibration
                # Create a dummy classifier that outputs the scores
                dummy_clf = DummyClassifier(strategy='constant', constant=0)
                dummy_clf.fit(group_y_scores.reshape(-1, 1), group_y_true)
                
                # Calibrate
                calibrated_clf = CalibratedClassifierCV(dummy_clf, method='isotonic', cv=3)
                
                # Fit calibration
                try:
                    calibrated_clf.fit(group_y_scores.reshape(-1, 1), group_y_true)
                    
                    # Apply calibration
                    calibrated_group_scores = calibrated_clf.predict_proba(
                        group_y_scores.reshape(-1, 1)
                    )[:, 1]
                    
                    calibrated_scores[group_mask] = calibrated_group_scores
                    calibration_functions[group] = calibrated_clf
                    
                except Exception as e:
                    logger.warning(f"Calibration failed for group {group}: {e}")
                    calibrated_scores[group_mask] = group_y_scores
            else:
                calibrated_scores[group_mask] = group_y_scores
        
        return {
            'calibrated_scores': calibrated_scores,
            'calibration_functions': calibration_functions,
            'original_scores': y_scores
        }


class FairnessAlgorithms:
    """Main class coordinating fairness algorithms"""
    
    def __init__(self):
        self.preprocessing = PreProcessingMitigation()
        self.inprocessing = InProcessingMitigation()
        self.postprocessing = PostProcessingMitigation()
        
        # Algorithm registry
        self.algorithms = {
            'reweighting': self.preprocessing.reweighting,
            'disparate_impact_remover': self.preprocessing.disparate_impact_remover,
            'adversarial_debiasing': self.inprocessing.adversarial_debiasing,
            'threshold_optimization': self.postprocessing.threshold_optimization,
            'calibration': self.postprocessing.calibration_based_postprocessing
        }
    
    async def apply_mitigation(self, algorithm: str, **kwargs) -> Dict[str, Any]:
        """Apply a specific bias mitigation algorithm"""
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        start_time = datetime.utcnow()
        
        try:
            result = await self.algorithms[algorithm](**kwargs)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            result['processing_time'] = processing_time
            result['algorithm'] = algorithm
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            logger.error(f"Error applying {algorithm}: {e}")
            return {
                'algorithm': algorithm,
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.utcnow() - start_time).total_seconds()
            }
    
    async def compare_algorithms(self, X_train: np.ndarray, y_train: np.ndarray,
                               sensitive_train: np.ndarray,
                               X_test: np.ndarray, y_test: np.ndarray,
                               sensitive_test: np.ndarray,
                               algorithms: List[str] = None) -> Dict[str, Any]:
        """Compare multiple fairness algorithms"""
        
        if algorithms is None:
            algorithms = ['reweighting', 'adversarial_debiasing', 'threshold_optimization']
        
        results = {}
        
        # Baseline (no mitigation)
        baseline_model = LogisticRegression()
        baseline_model.fit(X_train, y_train)
        baseline_pred = baseline_model.predict(X_test)
        baseline_scores = baseline_model.predict_proba(X_test)[:, 1]
        
        baseline_metrics = await self._calculate_fairness_metrics(
            y_test, baseline_pred, sensitive_test
        )
        
        results['baseline'] = {
            'metrics': baseline_metrics,
            'accuracy': accuracy_score(y_test, baseline_pred),
            'algorithm': 'baseline'
        }
        
        # Test each algorithm
        for algorithm in algorithms:
            try:
                if algorithm == 'reweighting':
                    # Reweighting
                    X_weighted, weights = await self.preprocessing.reweighting(
                        X_train, y_train, sensitive_train
                    )
                    
                    model = LogisticRegression()
                    model.fit(X_weighted, y_train, sample_weight=weights)
                    pred = model.predict(X_test)
                    
                elif algorithm == 'adversarial_debiasing':
                    # Adversarial debiasing
                    adv_result = await self.inprocessing.adversarial_debiasing(
                        X_train, y_train, sensitive_train, X_test
                    )
                    pred = (adv_result['test_predictions'] > 0.5).astype(int)
                    
                elif algorithm == 'threshold_optimization':
                    # Threshold optimization (using baseline scores)
                    thresh_result = await self.postprocessing.threshold_optimization(
                        y_test, baseline_scores, sensitive_test
                    )
                    pred = thresh_result['predictions']
                
                # Calculate metrics
                metrics = await self._calculate_fairness_metrics(
                    y_test, pred, sensitive_test
                )
                
                results[algorithm] = {
                    'metrics': metrics,
                    'accuracy': accuracy_score(y_test, pred),
                    'algorithm': algorithm
                }
                
            except Exception as e:
                logger.error(f"Error testing {algorithm}: {e}")
                results[algorithm] = {
                    'error': str(e),
                    'algorithm': algorithm
                }
        
        return results
    
    async def _calculate_fairness_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        sensitive_attr: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive fairness metrics"""
        from ..ethics.ethical_ai_framework import FairnessMetricsCalculator
        
        calculator = FairnessMetricsCalculator()
        
        metrics = await calculator.calculate_fairness_metrics(
            y_true, y_pred, sensitive_attr
        )
        
        return metrics
    
    async def recommend_algorithm(self, X: np.ndarray, y: np.ndarray,
                                sensitive_attr: np.ndarray,
                                priority: str = 'balanced') -> Dict[str, Any]:
        """Recommend the best fairness algorithm for given data"""
        
        # Analyze data characteristics
        data_analysis = await self._analyze_data_characteristics(X, y, sensitive_attr)
        
        recommendations = []
        
        # Rule-based recommendations
        if data_analysis['class_imbalance'] > 0.3:
            recommendations.append({
                'algorithm': 'reweighting',
                'reason': 'High class imbalance detected',
                'confidence': 0.8
            })
        
        if data_analysis['group_imbalance'] > 0.4:
            recommendations.append({
                'algorithm': 'synthetic_data_generation',
                'reason': 'Significant group imbalance',
                'confidence': 0.7
            })
        
        if data_analysis['feature_count'] > 50:
            recommendations.append({
                'algorithm': 'adversarial_debiasing',
                'reason': 'High-dimensional data benefits from neural approaches',
                'confidence': 0.6
            })
        
        if priority == 'accuracy_preserving':
            recommendations.append({
                'algorithm': 'threshold_optimization',
                'reason': 'Post-processing preserves model accuracy',
                'confidence': 0.9
            })
        
        # Default recommendation
        if not recommendations:
            recommendations.append({
                'algorithm': 'reweighting',
                'reason': 'General-purpose algorithm suitable for most cases',
                'confidence': 0.5
            })
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'data_analysis': data_analysis,
            'recommendations': recommendations,
            'top_recommendation': recommendations[0] if recommendations else None
        }
    
    async def _analyze_data_characteristics(self, X: np.ndarray, y: np.ndarray,
                                          sensitive_attr: np.ndarray) -> Dict[str, Any]:
        """Analyze data characteristics to inform algorithm selection"""
        
        # Class imbalance
        unique_labels, label_counts = np.unique(y, return_counts=True)
        class_imbalance = 1 - min(label_counts) / max(label_counts)
        
        # Group imbalance
        unique_groups, group_counts = np.unique(sensitive_attr, return_counts=True)
        group_imbalance = 1 - min(group_counts) / max(group_counts)
        
        # Feature characteristics
        feature_count = X.shape[1]
        sample_count = X.shape[0]
        
        # Correlation between features and sensitive attribute
        if len(X.shape) == 2:
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], sensitive_attr)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            max_correlation = max(correlations) if correlations else 0
        else:
            max_correlation = 0
        
        return {
            'sample_count': sample_count,
            'feature_count': feature_count,
            'class_imbalance': class_imbalance,
            'group_imbalance': group_imbalance,
            'max_feature_correlation': max_correlation,
            'unique_groups': len(unique_groups),
            'unique_labels': len(unique_labels)
        }