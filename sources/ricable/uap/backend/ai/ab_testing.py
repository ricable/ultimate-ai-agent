# backend/ai/ab_testing.py
# Agent 21: A/B Testing Framework for AI Models with Performance Monitoring

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats
import aiofiles

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class TrafficSplitMethod(Enum):
    RANDOM = "random"
    USER_HASH = "user_hash"
    GEOGRAPHIC = "geographic"
    FEATURE_FLAG = "feature_flag"

@dataclass
class ModelVariant:
    """Model variant in A/B test"""
    variant_id: str
    model_id: str
    model_version: str
    traffic_percentage: float
    name: str
    description: str
    config: Dict[str, Any]

@dataclass
class ExperimentMetric:
    """Metric to track in A/B experiment"""
    metric_name: str
    metric_type: str  # 'conversion', 'numeric', 'latency'
    target_direction: str  # 'increase', 'decrease', 'neutral'
    minimum_detectable_effect: float
    baseline_value: Optional[float] = None

@dataclass
class ABExperiment:
    """A/B testing experiment configuration"""
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    variants: List[ModelVariant]
    metrics: List[ExperimentMetric]
    traffic_split_method: TrafficSplitMethod
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    min_sample_size: int
    confidence_level: float
    created_at: datetime
    updated_at: datetime
    metadata: Dict[str, Any]

class ExperimentDataCollector:
    """Collect and store experiment data"""
    
    def __init__(self, storage_path: str = "./experiments"):
        self.storage_path = storage_path
        self.data_cache: Dict[str, List[Dict]] = {}
    
    async def record_interaction(self, 
                               experiment_id: str,
                               variant_id: str,
                               user_id: str,
                               request_data: Dict[str, Any],
                               response_data: Dict[str, Any],
                               metrics: Dict[str, float],
                               timestamp: datetime = None):
        """Record a single interaction with the model"""
        
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        interaction = {
            'experiment_id': experiment_id,
            'variant_id': variant_id,
            'user_id': user_id,
            'timestamp': timestamp.isoformat(),
            'request_data': request_data,
            'response_data': response_data,
            'metrics': metrics
        }
        
        # Cache data
        if experiment_id not in self.data_cache:
            self.data_cache[experiment_id] = []
        self.data_cache[experiment_id].append(interaction)
        
        # Persist to storage
        await self._persist_interaction(interaction)
    
    async def _persist_interaction(self, interaction: Dict):
        """Persist interaction data to storage"""
        experiment_id = interaction['experiment_id']
        date_str = datetime.fromisoformat(interaction['timestamp']).strftime('%Y-%m-%d')
        
        file_path = f"{self.storage_path}/{experiment_id}/{date_str}.jsonl"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Append to file
        async with aiofiles.open(file_path, 'a') as f:
            await f.write(json.dumps(interaction) + '\n')
    
    async def get_experiment_data(self, 
                                experiment_id: str,
                                start_date: datetime = None,
                                end_date: datetime = None) -> List[Dict]:
        """Retrieve experiment data for analysis"""
        
        # Return cached data if available
        if experiment_id in self.data_cache:
            data = self.data_cache[experiment_id]
            
            # Filter by date range if specified
            if start_date or end_date:
                filtered_data = []
                for interaction in data:
                    interaction_date = datetime.fromisoformat(interaction['timestamp'])
                    if start_date and interaction_date < start_date:
                        continue
                    if end_date and interaction_date > end_date:
                        continue
                    filtered_data.append(interaction)
                return filtered_data
            
            return data
        
        # Load from storage
        return await self._load_experiment_data(experiment_id, start_date, end_date)
    
    async def _load_experiment_data(self, 
                                  experiment_id: str,
                                  start_date: datetime = None,
                                  end_date: datetime = None) -> List[Dict]:
        """Load experiment data from storage"""
        data = []
        experiment_dir = f"{self.storage_path}/{experiment_id}"
        
        if not os.path.exists(experiment_dir):
            return data
        
        # Read all data files
        for file_name in os.listdir(experiment_dir):
            if file_name.endswith('.jsonl'):
                file_path = os.path.join(experiment_dir, file_name)
                async with aiofiles.open(file_path, 'r') as f:
                    async for line in f:
                        if line.strip():
                            interaction = json.loads(line.strip())
                            
                            # Filter by date range
                            interaction_date = datetime.fromisoformat(interaction['timestamp'])
                            if start_date and interaction_date < start_date:
                                continue
                            if end_date and interaction_date > end_date:
                                continue
                            
                            data.append(interaction)
        
        return data

class StatisticalAnalyzer:
    """Statistical analysis for A/B test results"""
    
    @staticmethod
    def calculate_confidence_interval(data: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for the data"""
        if not data:
            return (0.0, 0.0)
        
        mean = np.mean(data)
        std_err = stats.sem(data)
        h = std_err * stats.t.ppf((1 + confidence_level) / 2., len(data) - 1)
        
        return (mean - h, mean + h)
    
    @staticmethod
    def t_test_independent(group_a: List[float], group_b: List[float]) -> Dict[str, float]:
        """Perform independent t-test between two groups"""
        if not group_a or not group_b:
            return {'p_value': 1.0, 't_statistic': 0.0, 'effect_size': 0.0}
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(group_a, group_b)
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt(((len(group_a) - 1) * np.var(group_a, ddof=1) + 
                             (len(group_b) - 1) * np.var(group_b, ddof=1)) / 
                            (len(group_a) + len(group_b) - 2))
        
        effect_size = (np.mean(group_a) - np.mean(group_b)) / pooled_std if pooled_std > 0 else 0.0
        
        return {
            'p_value': float(p_value),
            't_statistic': float(t_stat),
            'effect_size': float(effect_size)
        }
    
    @staticmethod
    def calculate_required_sample_size(baseline_rate: float,
                                     minimum_detectable_effect: float,
                                     alpha: float = 0.05,
                                     power: float = 0.8) -> int:
        """Calculate required sample size for experiment"""
        
        # For proportion/conversion rate tests
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        # Effect size
        effect_size = abs(p2 - p1) / np.sqrt(p1 * (1 - p1))
        
        # Required sample size per group
        sample_size = (2 * (stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power))**2) / (effect_size**2)
        
        return max(int(np.ceil(sample_size)), 100)  # Minimum 100 samples

class ABTestManager:
    """Main A/B testing manager"""
    
    def __init__(self, storage_path: str = "./experiments"):
        self.storage_path = storage_path
        self.experiments: Dict[str, ABExperiment] = {}
        self.data_collector = ExperimentDataCollector(storage_path)
        self.analyzer = StatisticalAnalyzer()
    
    async def create_experiment(self,
                              name: str,
                              description: str,
                              variants: List[Dict[str, Any]],
                              metrics: List[Dict[str, Any]],
                              traffic_split_method: TrafficSplitMethod = TrafficSplitMethod.RANDOM,
                              min_sample_size: int = None,
                              confidence_level: float = 0.95) -> ABExperiment:
        """Create a new A/B testing experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        # Create model variants
        model_variants = []
        total_traffic = 0
        
        for variant_data in variants:
            variant = ModelVariant(
                variant_id=str(uuid.uuid4()),
                model_id=variant_data['model_id'],
                model_version=variant_data['model_version'],
                traffic_percentage=variant_data['traffic_percentage'],
                name=variant_data['name'],
                description=variant_data.get('description', ''),
                config=variant_data.get('config', {})
            )
            model_variants.append(variant)
            total_traffic += variant.traffic_percentage
        
        # Validate traffic split
        if abs(total_traffic - 100.0) > 0.01:
            raise ValueError(f"Traffic percentages must sum to 100%, got {total_traffic}%")
        
        # Create experiment metrics
        experiment_metrics = []
        for metric_data in metrics:
            metric = ExperimentMetric(
                metric_name=metric_data['metric_name'],
                metric_type=metric_data['metric_type'],
                target_direction=metric_data['target_direction'],
                minimum_detectable_effect=metric_data['minimum_detectable_effect'],
                baseline_value=metric_data.get('baseline_value')
            )
            experiment_metrics.append(metric)
        
        # Calculate minimum sample size if not provided
        if min_sample_size is None and experiment_metrics:
            primary_metric = experiment_metrics[0]
            if primary_metric.baseline_value:
                min_sample_size = self.analyzer.calculate_required_sample_size(
                    baseline_rate=primary_metric.baseline_value,
                    minimum_detectable_effect=primary_metric.minimum_detectable_effect
                )
            else:
                min_sample_size = 1000  # Default
        
        # Create experiment
        experiment = ABExperiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            status=ExperimentStatus.DRAFT,
            variants=model_variants,
            metrics=experiment_metrics,
            traffic_split_method=traffic_split_method,
            start_date=None,
            end_date=None,
            min_sample_size=min_sample_size or 1000,
            confidence_level=confidence_level,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            metadata={}
        )
        
        self.experiments[experiment_id] = experiment
        await self._save_experiment(experiment)
        
        return experiment
    
    async def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.RUNNING
        experiment.start_date = datetime.utcnow()
        experiment.updated_at = datetime.utcnow()
        
        await self._save_experiment(experiment)
        return True
    
    async def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment"""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.utcnow()
        experiment.updated_at = datetime.utcnow()
        
        await self._save_experiment(experiment)
        return True
    
    async def assign_variant(self, experiment_id: str, user_id: str) -> Optional[ModelVariant]:
        """Assign a user to an experiment variant"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        if experiment.status != ExperimentStatus.RUNNING:
            return None
        
        # Simple random assignment based on user_id hash
        import hashlib
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        random_value = (user_hash % 100) + 1  # 1-100
        
        cumulative_percentage = 0
        for variant in experiment.variants:
            cumulative_percentage += variant.traffic_percentage
            if random_value <= cumulative_percentage:
                return variant
        
        # Fallback to first variant
        return experiment.variants[0] if experiment.variants else None
    
    async def record_interaction(self, 
                               experiment_id: str,
                               variant_id: str,
                               user_id: str,
                               request_data: Dict[str, Any],
                               response_data: Dict[str, Any],
                               metrics: Dict[str, float]):
        """Record an interaction for analysis"""
        await self.data_collector.record_interaction(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            request_data=request_data,
            response_data=response_data,
            metrics=metrics
        )
    
    async def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Perform statistical analysis of experiment results"""
        if experiment_id not in self.experiments:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiments[experiment_id]
        data = await self.data_collector.get_experiment_data(experiment_id)
        
        if not data:
            return {'error': 'No data available for analysis'}
        
        # Group data by variant
        variant_data = {}
        for interaction in data:
            variant_id = interaction['variant_id']
            if variant_id not in variant_data:
                variant_data[variant_id] = []
            variant_data[variant_id].append(interaction)
        
        # Analyze each metric
        results = {
            'experiment_id': experiment_id,
            'experiment_name': experiment.name,
            'status': experiment.status.value,
            'total_interactions': len(data),
            'variant_results': {},
            'statistical_tests': {}
        }
        
        # Calculate results for each variant
        for variant in experiment.variants:
            variant_interactions = variant_data.get(variant.variant_id, [])
            
            results['variant_results'][variant.variant_id] = {
                'variant_name': variant.name,
                'interactions': len(variant_interactions),
                'metrics': {}
            }
            
            # Calculate metrics for this variant
            for metric in experiment.metrics:
                metric_values = [
                    interaction['metrics'].get(metric.metric_name, 0.0)
                    for interaction in variant_interactions
                    if metric.metric_name in interaction['metrics']
                ]
                
                if metric_values:
                    mean_value = np.mean(metric_values)
                    ci_lower, ci_upper = self.analyzer.calculate_confidence_interval(
                        metric_values, experiment.confidence_level
                    )
                    
                    results['variant_results'][variant.variant_id]['metrics'][metric.metric_name] = {
                        'mean': float(mean_value),
                        'count': len(metric_values),
                        'confidence_interval': [float(ci_lower), float(ci_upper)]
                    }
        
        # Perform statistical tests between variants
        if len(experiment.variants) == 2:
            control_variant = experiment.variants[0]
            treatment_variant = experiment.variants[1]
            
            control_data = variant_data.get(control_variant.variant_id, [])
            treatment_data = variant_data.get(treatment_variant.variant_id, [])
            
            for metric in experiment.metrics:
                control_values = [
                    interaction['metrics'].get(metric.metric_name, 0.0)
                    for interaction in control_data
                    if metric.metric_name in interaction['metrics']
                ]
                
                treatment_values = [
                    interaction['metrics'].get(metric.metric_name, 0.0)
                    for interaction in treatment_data
                    if metric.metric_name in interaction['metrics']
                ]
                
                if control_values and treatment_values:
                    test_result = self.analyzer.t_test_independent(control_values, treatment_values)
                    
                    # Determine if result is statistically significant
                    is_significant = test_result['p_value'] < (1 - experiment.confidence_level)
                    
                    # Calculate percentage change
                    control_mean = np.mean(control_values)
                    treatment_mean = np.mean(treatment_values)
                    percent_change = ((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0
                    
                    results['statistical_tests'][metric.metric_name] = {
                        'p_value': test_result['p_value'],
                        't_statistic': test_result['t_statistic'],
                        'effect_size': test_result['effect_size'],
                        'is_significant': is_significant,
                        'percent_change': float(percent_change),
                        'sample_size_control': len(control_values),
                        'sample_size_treatment': len(treatment_values)
                    }
        
        return results
    
    async def _save_experiment(self, experiment: ABExperiment):
        """Save experiment configuration to storage"""
        experiment_file = f"{self.storage_path}/{experiment.experiment_id}/config.json"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(experiment_file), exist_ok=True)
        
        # Convert to dict
        experiment_dict = asdict(experiment)
        experiment_dict['created_at'] = experiment.created_at.isoformat()
        experiment_dict['updated_at'] = experiment.updated_at.isoformat()
        if experiment.start_date:
            experiment_dict['start_date'] = experiment.start_date.isoformat()
        if experiment.end_date:
            experiment_dict['end_date'] = experiment.end_date.isoformat()
        
        async with aiofiles.open(experiment_file, 'w') as f:
            await f.write(json.dumps(experiment_dict, indent=2))

# Global A/B testing manager
ab_test_manager = ABTestManager()