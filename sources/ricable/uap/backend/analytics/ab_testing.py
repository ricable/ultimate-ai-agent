# File: backend/analytics/ab_testing.py
"""
A/B Testing Framework for UAP Platform

Provides comprehensive A/B testing capabilities for feature optimization,
user experience experiments, and data-driven decision making.
"""

import asyncio
import json
import random
import hashlib
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, asdict, field
from enum import Enum
import statistics
import uuid
from threading import Lock

from .usage_analytics import usage_analytics, track_feature_usage

class ExperimentStatus(Enum):
    """Status of A/B test experiments"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"

class ExperimentType(Enum):
    """Types of A/B test experiments"""
    FEATURE_FLAG = "feature_flag"          # Simple on/off feature testing
    UI_VARIANT = "ui_variant"              # UI component variations
    ALGORITHM_TEST = "algorithm_test"      # Algorithm/model comparisons
    PERFORMANCE_TEST = "performance_test"  # Performance optimizations
    CONTENT_TEST = "content_test"          # Content variations
    WORKFLOW_TEST = "workflow_test"        # User workflow variations

class StatisticalSignificance(Enum):
    """Statistical significance levels"""
    NOT_SIGNIFICANT = "not_significant"     # p > 0.05
    SIGNIFICANT = "significant"             # p <= 0.05
    HIGHLY_SIGNIFICANT = "highly_significant" # p <= 0.01
    VERY_SIGNIFICANT = "very_significant"   # p <= 0.001

class MetricType(Enum):
    """Types of metrics to track in experiments"""
    CONVERSION_RATE = "conversion_rate"     # Boolean conversion metrics
    NUMERIC_VALUE = "numeric_value"         # Continuous numeric metrics
    COUNT = "count"                         # Count-based metrics
    DURATION = "duration"                   # Time-based metrics
    CATEGORICAL = "categorical"             # Categorical metrics

@dataclass
class ExperimentVariant:
    """Definition of an experiment variant"""
    variant_id: str
    name: str
    description: str
    allocation_percentage: float  # 0-100
    configuration: Dict[str, Any]
    is_control: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExperimentMetric:
    """Definition of a metric to track in experiments"""
    metric_id: str
    name: str
    description: str
    metric_type: MetricType
    is_primary: bool = False
    goal: str = "increase"  # "increase", "decrease", "no_change"
    significance_threshold: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['metric_type'] = self.metric_type.value
        return data

@dataclass
class ExperimentConfiguration:
    """Configuration for an A/B test experiment"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    variants: List[ExperimentVariant]
    metrics: List[ExperimentMetric]
    target_audience: Dict[str, Any]  # Targeting criteria
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    minimum_sample_size: int = 100
    confidence_level: float = 0.95
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: ExperimentStatus = ExperimentStatus.DRAFT
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['experiment_type'] = self.experiment_type.value
        data['status'] = self.status.value
        data['variants'] = [v.to_dict() for v in self.variants]
        data['metrics'] = [m.to_dict() for m in self.metrics]
        if self.start_date:
            data['start_date'] = self.start_date.isoformat()
        if self.end_date:
            data['end_date'] = self.end_date.isoformat()
        data['created_at'] = self.created_at.isoformat()
        return data

@dataclass
class ExperimentParticipant:
    """A participant in an A/B test experiment"""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['assigned_at'] = self.assigned_at.isoformat()
        return data

@dataclass
class MetricEvent:
    """An event recording a metric value for an experiment"""
    event_id: str
    experiment_id: str
    variant_id: str
    metric_id: str
    user_id: str
    value: Union[float, int, bool, str]
    timestamp: datetime
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class VariantResults:
    """Results for a specific variant"""
    variant_id: str
    participant_count: int
    metric_values: Dict[str, List[Union[float, int, bool]]]
    conversion_rates: Dict[str, float]
    average_values: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'variant_id': self.variant_id,
            'participant_count': self.participant_count,
            'metric_summaries': {
                metric_id: {
                    'count': len(values),
                    'average': statistics.mean(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0
                }
                for metric_id, values in self.metric_values.items()
            },
            'conversion_rates': self.conversion_rates,
            'average_values': self.average_values,
            'confidence_intervals': {
                metric_id: {'lower': ci[0], 'upper': ci[1]}
                for metric_id, ci in self.confidence_intervals.items()
            }
        }

@dataclass
class ExperimentResults:
    """Complete results for an A/B test experiment"""
    experiment_id: str
    status: ExperimentStatus
    start_date: datetime
    end_date: Optional[datetime]
    total_participants: int
    variant_results: Dict[str, VariantResults]
    statistical_significance: Dict[str, StatisticalSignificance]
    winner: Optional[str] = None
    confidence_level: float = 0.95
    generated_at: datetime = field(default_factory=datetime.utcnow)
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['start_date'] = self.start_date.isoformat()
        if self.end_date:
            data['end_date'] = self.end_date.isoformat()
        data['generated_at'] = self.generated_at.isoformat()
        data['variant_results'] = {
            variant_id: results.to_dict() 
            for variant_id, results in self.variant_results.items()
        }
        data['statistical_significance'] = {
            metric_id: sig.value 
            for metric_id, sig in self.statistical_significance.items()
        }
        return data

class ABTestFramework:
    """Main A/B testing framework"""
    
    def __init__(self):
        self.experiments: Dict[str, ExperimentConfiguration] = {}
        self.participants: Dict[str, List[ExperimentParticipant]] = defaultdict(list)
        self.metric_events: deque = deque(maxlen=100000)
        self.experiment_assignments: Dict[str, Dict[str, str]] = defaultdict(dict)  # user_id -> {exp_id: variant_id}
        self.lock = Lock()
        
        # Framework settings
        self.hash_seed = "uap_ab_testing_2025"
        self.max_experiments_per_user = 10
        
    def create_experiment(self, name: str, description: str, 
                         experiment_type: ExperimentType,
                         variants: List[Dict[str, Any]],
                         metrics: List[Dict[str, Any]],
                         target_audience: Dict[str, Any] = None,
                         minimum_sample_size: int = 100,
                         confidence_level: float = 0.95,
                         created_by: str = "system") -> str:
        """Create a new A/B test experiment"""
        experiment_id = str(uuid.uuid4())
        
        # Create variant objects
        variant_objects = []
        total_allocation = 0
        
        for i, variant_data in enumerate(variants):
            variant = ExperimentVariant(
                variant_id=variant_data.get('variant_id', f"variant_{i}"),
                name=variant_data['name'],
                description=variant_data['description'],
                allocation_percentage=variant_data['allocation_percentage'],
                configuration=variant_data.get('configuration', {}),
                is_control=variant_data.get('is_control', i == 0)
            )
            variant_objects.append(variant)
            total_allocation += variant.allocation_percentage
        
        if abs(total_allocation - 100.0) > 0.01:
            raise ValueError(f"Variant allocations must sum to 100%, got {total_allocation}%")
        
        # Create metric objects
        metric_objects = []
        for metric_data in metrics:
            metric = ExperimentMetric(
                metric_id=metric_data.get('metric_id', str(uuid.uuid4())),
                name=metric_data['name'],
                description=metric_data['description'],
                metric_type=MetricType(metric_data['metric_type']),
                is_primary=metric_data.get('is_primary', False),
                goal=metric_data.get('goal', 'increase'),
                significance_threshold=metric_data.get('significance_threshold', 0.05)
            )
            metric_objects.append(metric)
        
        # Create experiment configuration
        config = ExperimentConfiguration(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            variants=variant_objects,
            metrics=metric_objects,
            target_audience=target_audience or {},
            minimum_sample_size=minimum_sample_size,
            confidence_level=confidence_level,
            created_by=created_by
        )
        
        with self.lock:
            self.experiments[experiment_id] = config
        
        return experiment_id
    
    def start_experiment(self, experiment_id: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> bool:
        """Start an A/B test experiment"""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        
        if experiment.status != ExperimentStatus.DRAFT:
            return False
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_date = start_date or datetime.utcnow()
        experiment.end_date = end_date
        
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an A/B test experiment"""
        if experiment_id not in self.experiments:
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.STOPPED
        experiment.end_date = datetime.utcnow()
        
        return True
    
    def assign_user_to_experiment(self, user_id: str, experiment_id: str,
                                 session_id: Optional[str] = None,
                                 force_variant: Optional[str] = None) -> Optional[str]:
        """Assign a user to an experiment variant"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # Check if experiment is active
        if experiment.status != ExperimentStatus.ACTIVE:
            return None
        
        # Check if user already assigned
        if user_id in self.experiment_assignments and experiment_id in self.experiment_assignments[user_id]:
            return self.experiment_assignments[user_id][experiment_id]
        
        # Check target audience criteria
        if not self._matches_target_audience(user_id, experiment.target_audience):
            return None
        
        # Check maximum experiments per user
        user_experiments = self.experiment_assignments.get(user_id, {})
        if len(user_experiments) >= self.max_experiments_per_user:
            return None
        
        # Assign variant
        if force_variant and any(v.variant_id == force_variant for v in experiment.variants):
            assigned_variant = force_variant
        else:
            assigned_variant = self._determine_variant(user_id, experiment)
        
        if not assigned_variant:
            return None
        
        # Record assignment
        participant = ExperimentParticipant(
            user_id=user_id,
            experiment_id=experiment_id,
            variant_id=assigned_variant,
            assigned_at=datetime.utcnow(),
            session_id=session_id
        )
        
        with self.lock:
            self.participants[experiment_id].append(participant)
            self.experiment_assignments[user_id][experiment_id] = assigned_variant
        
        # Track assignment event
        track_feature_usage(
            user_id=user_id,
            session_id=session_id,
            feature=f"ab_test_{experiment_id}",
            metadata={
                'variant_id': assigned_variant,
                'experiment_name': experiment.name
            }
        )
        
        return assigned_variant
    
    def _determine_variant(self, user_id: str, experiment: ExperimentConfiguration) -> Optional[str]:
        """Determine which variant to assign to a user using consistent hashing"""
        # Create hash of user_id + experiment_id for consistent assignment
        hash_input = f"{user_id}_{experiment.experiment_id}_{self.hash_seed}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 10000
        
        # Assign based on allocation percentages
        cumulative_percentage = 0
        hash_percentage = hash_value / 100.0  # Convert to 0-100 range
        
        for variant in experiment.variants:
            cumulative_percentage += variant.allocation_percentage
            if hash_percentage < cumulative_percentage:
                return variant.variant_id
        
        # Fallback to control if something goes wrong
        control_variant = next((v for v in experiment.variants if v.is_control), None)
        return control_variant.variant_id if control_variant else experiment.variants[0].variant_id
    
    def _matches_target_audience(self, user_id: str, target_audience: Dict[str, Any]) -> bool:
        """Check if user matches target audience criteria"""
        if not target_audience:
            return True
        
        # Basic targeting criteria - would be expanded based on user data available
        # For now, accept all users
        return True
    
    def record_metric_event(self, experiment_id: str, user_id: str, 
                           metric_id: str, value: Union[float, int, bool, str],
                           session_id: Optional[str] = None,
                           metadata: Dict[str, Any] = None) -> bool:
        """Record a metric event for an experiment"""
        if experiment_id not in self.experiments:
            return False
        
        # Check if user is assigned to this experiment
        user_assignments = self.experiment_assignments.get(user_id, {})
        if experiment_id not in user_assignments:
            return False
        
        variant_id = user_assignments[experiment_id]
        
        # Create metric event
        event = MetricEvent(
            event_id=str(uuid.uuid4()),
            experiment_id=experiment_id,
            variant_id=variant_id,
            metric_id=metric_id,
            user_id=user_id,
            value=value,
            timestamp=datetime.utcnow(),
            session_id=session_id,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.metric_events.append(event)
        
        return True
    
    def get_experiment_results(self, experiment_id: str) -> Optional[ExperimentResults]:
        """Get current results for an experiment"""
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        participants = self.participants.get(experiment_id, [])
        
        if not participants:
            return None
        
        # Filter metric events for this experiment
        experiment_events = [e for e in self.metric_events if e.experiment_id == experiment_id]
        
        # Group events by variant
        events_by_variant = defaultdict(list)
        for event in experiment_events:
            events_by_variant[event.variant_id].append(event)
        
        # Calculate results for each variant
        variant_results = {}
        
        for variant in experiment.variants:
            variant_id = variant.variant_id
            variant_participants = [p for p in participants if p.variant_id == variant_id]
            variant_events = events_by_variant.get(variant_id, [])
            
            # Group events by metric
            metric_values = defaultdict(list)
            for event in variant_events:
                if isinstance(event.value, (int, float, bool)):
                    metric_values[event.metric_id].append(float(event.value))
            
            # Calculate conversion rates and averages
            conversion_rates = {}
            average_values = {}
            confidence_intervals = {}
            
            for metric in experiment.metrics:
                metric_id = metric.metric_id
                values = metric_values.get(metric_id, [])
                
                if values:
                    if metric.metric_type == MetricType.CONVERSION_RATE:
                        conversion_rates[metric_id] = sum(values) / len(values)
                    else:
                        average_values[metric_id] = statistics.mean(values)
                    
                    # Calculate confidence interval (simplified)
                    if len(values) > 1:
                        std_err = statistics.stdev(values) / (len(values) ** 0.5)
                        margin = 1.96 * std_err  # 95% confidence interval
                        mean_val = statistics.mean(values)
                        confidence_intervals[metric_id] = (mean_val - margin, mean_val + margin)
                    else:
                        confidence_intervals[metric_id] = (values[0], values[0]) if values else (0, 0)
            
            variant_results[variant_id] = VariantResults(
                variant_id=variant_id,
                participant_count=len(variant_participants),
                metric_values=dict(metric_values),
                conversion_rates=conversion_rates,
                average_values=average_values,
                confidence_intervals=confidence_intervals
            )
        
        # Calculate statistical significance (simplified)
        statistical_significance = {}
        control_variant = next((v for v in experiment.variants if v.is_control), None)
        
        if control_variant:
            control_results = variant_results.get(control_variant.variant_id)
            
            for metric in experiment.metrics:
                metric_id = metric.metric_id
                
                # Simple significance test based on confidence intervals
                if control_results and metric_id in control_results.confidence_intervals:
                    control_ci = control_results.confidence_intervals[metric_id]
                    
                    significant = False
                    for variant_id, results in variant_results.items():
                        if variant_id != control_variant.variant_id and metric_id in results.confidence_intervals:
                            test_ci = results.confidence_intervals[metric_id]
                            # Check if confidence intervals don't overlap
                            if test_ci[0] > control_ci[1] or test_ci[1] < control_ci[0]:
                                significant = True
                                break
                    
                    statistical_significance[metric_id] = (
                        StatisticalSignificance.SIGNIFICANT if significant 
                        else StatisticalSignificance.NOT_SIGNIFICANT
                    )
        
        # Determine winner (simplified)
        winner = None
        primary_metric = next((m for m in experiment.metrics if m.is_primary), None)
        
        if primary_metric and primary_metric.metric_id in statistical_significance:
            if statistical_significance[primary_metric.metric_id] == StatisticalSignificance.SIGNIFICANT:
                # Find variant with best performance for primary metric
                best_value = None
                best_variant = None
                
                for variant_id, results in variant_results.items():
                    if primary_metric.metric_id in results.average_values:
                        value = results.average_values[primary_metric.metric_id]
                        
                        if best_value is None:
                            best_value = value
                            best_variant = variant_id
                        elif (
                            (primary_metric.goal == "increase" and value > best_value) or
                            (primary_metric.goal == "decrease" and value < best_value)
                        ):
                            best_value = value
                            best_variant = variant_id
                
                winner = best_variant
        
        # Generate recommendations
        recommendations = self._generate_experiment_recommendations(
            experiment, variant_results, statistical_significance, winner
        )
        
        return ExperimentResults(
            experiment_id=experiment_id,
            status=experiment.status,
            start_date=experiment.start_date,
            end_date=experiment.end_date,
            total_participants=len(participants),
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            winner=winner,
            confidence_level=experiment.confidence_level,
            recommendations=recommendations
        )
    
    def _generate_experiment_recommendations(self, experiment: ExperimentConfiguration,
                                          variant_results: Dict[str, VariantResults],
                                          significance: Dict[str, StatisticalSignificance],
                                          winner: Optional[str]) -> List[str]:
        """Generate recommendations based on experiment results"""
        recommendations = []
        
        total_participants = sum(r.participant_count for r in variant_results.values())
        
        if total_participants < experiment.minimum_sample_size:
            recommendations.append(
                f"Increase sample size - current: {total_participants}, minimum: {experiment.minimum_sample_size}"
            )
        
        # Check if any metrics show statistical significance
        significant_metrics = [m for m, s in significance.items() if s == StatisticalSignificance.SIGNIFICANT]
        
        if not significant_metrics:
            recommendations.append("No statistically significant results yet - consider running longer")
        
        if winner:
            winner_name = next((v.name for v in experiment.variants if v.variant_id == winner), winner)
            recommendations.append(f"Variant '{winner_name}' shows best performance - consider implementing")
        
        # Check for low engagement
        for variant_id, results in variant_results.items():
            if results.participant_count < 10:
                variant_name = next((v.name for v in experiment.variants if v.variant_id == variant_id), variant_id)
                recommendations.append(f"Low participation in variant '{variant_name}' - check targeting")
        
        return recommendations
    
    def get_user_experiments(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all experiments a user is participating in"""
        user_assignments = self.experiment_assignments.get(user_id, {})
        
        experiments = []
        for experiment_id, variant_id in user_assignments.items():
            if experiment_id in self.experiments:
                experiment = self.experiments[experiment_id]
                variant = next((v for v in experiment.variants if v.variant_id == variant_id), None)
                
                experiments.append({
                    'experiment_id': experiment_id,
                    'experiment_name': experiment.name,
                    'variant_id': variant_id,
                    'variant_name': variant.name if variant else 'Unknown',
                    'variant_config': variant.configuration if variant else {},
                    'status': experiment.status.value
                })
        
        return experiments
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get summary of all experiments"""
        total_experiments = len(self.experiments)
        active_experiments = sum(1 for e in self.experiments.values() if e.status == ExperimentStatus.ACTIVE)
        total_participants = sum(len(participants) for participants in self.participants.values())
        
        experiments_by_status = defaultdict(int)
        for experiment in self.experiments.values():
            experiments_by_status[experiment.status.value] += 1
        
        return {
            'total_experiments': total_experiments,
            'active_experiments': active_experiments,
            'total_participants': total_participants,
            'experiments_by_status': dict(experiments_by_status),
            'total_metric_events': len(self.metric_events)
        }

# Global A/B testing framework instance
ab_testing = ABTestFramework()

# Convenience functions
def create_feature_flag_test(name: str, description: str, 
                           feature_config: Dict[str, Any],
                           allocation_split: float = 0.5) -> str:
    """Create a simple feature flag A/B test"""
    variants = [
        {
            'variant_id': 'control',
            'name': 'Control (Feature Off)',
            'description': 'Control group with feature disabled',
            'allocation_percentage': (1 - allocation_split) * 100,
            'configuration': {'feature_enabled': False},
            'is_control': True
        },
        {
            'variant_id': 'treatment',
            'name': 'Treatment (Feature On)',
            'description': 'Treatment group with feature enabled',
            'allocation_percentage': allocation_split * 100,
            'configuration': {'feature_enabled': True, **feature_config}
        }
    ]
    
    metrics = [
        {
            'metric_id': 'conversion_rate',
            'name': 'Conversion Rate',
            'description': 'Primary conversion metric',
            'metric_type': 'conversion_rate',
            'is_primary': True,
            'goal': 'increase'
        },
        {
            'metric_id': 'engagement_time',
            'name': 'Engagement Time',
            'description': 'Time spent engaging with feature',
            'metric_type': 'duration',
            'goal': 'increase'
        }
    ]
    
    return ab_testing.create_experiment(
        name=name,
        description=description,
        experiment_type=ExperimentType.FEATURE_FLAG,
        variants=variants,
        metrics=metrics
    )

def assign_user_to_variant(user_id: str, experiment_id: str, 
                          session_id: Optional[str] = None) -> Optional[str]:
    """Assign user to experiment variant"""
    return ab_testing.assign_user_to_experiment(user_id, experiment_id, session_id)

def record_conversion(experiment_id: str, user_id: str, 
                     converted: bool = True, session_id: Optional[str] = None) -> bool:
    """Record a conversion event"""
    return ab_testing.record_metric_event(
        experiment_id=experiment_id,
        user_id=user_id,
        metric_id='conversion_rate',
        value=converted,
        session_id=session_id
    )

def record_engagement_time(experiment_id: str, user_id: str, 
                          duration_seconds: float, session_id: Optional[str] = None) -> bool:
    """Record engagement time"""
    return ab_testing.record_metric_event(
        experiment_id=experiment_id,
        user_id=user_id,
        metric_id='engagement_time',
        value=duration_seconds,
        session_id=session_id
    )

def get_experiment_results(experiment_id: str) -> Optional[Dict[str, Any]]:
    """Get experiment results"""
    results = ab_testing.get_experiment_results(experiment_id)
    return results.to_dict() if results else None