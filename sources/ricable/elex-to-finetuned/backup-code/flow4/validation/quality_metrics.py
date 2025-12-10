"""
Comprehensive quality metrics and acceptance criteria for Flow4.

This module defines:
- Quality metrics for each pipeline stage
- Acceptance criteria for different components
- Benchmarking and scoring systems
- Quality gates and thresholds
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QualityGrade(Enum):
    """Quality grades for components."""
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 70-89%
    FAIR = "fair"           # 50-69%
    POOR = "poor"           # 30-49%
    FAILING = "failing"     # 0-29%


class MetricType(Enum):
    """Types of quality metrics."""
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    RELIABILITY = "reliability"
    USABILITY = "usability"
    SAFETY = "safety"


@dataclass
class QualityMetric:
    """Definition of a quality metric."""
    name: str
    description: str
    metric_type: MetricType
    unit: str
    min_threshold: float
    target_value: float
    max_value: float = 100.0
    weight: float = 1.0


@dataclass
class QualityMeasurement:
    """A specific quality measurement."""
    metric: QualityMetric
    measured_value: float
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def score(self) -> float:
        """Calculate normalized score (0-1)."""
        if self.measured_value >= self.metric.target_value:
            return 1.0
        elif self.measured_value >= self.metric.min_threshold:
            return (self.measured_value - self.metric.min_threshold) / (self.metric.target_value - self.metric.min_threshold)
        else:
            return 0.0
    
    @property
    def grade(self) -> QualityGrade:
        """Get quality grade based on score."""
        score_percent = self.score * 100
        if score_percent >= 90:
            return QualityGrade.EXCELLENT
        elif score_percent >= 70:
            return QualityGrade.GOOD
        elif score_percent >= 50:
            return QualityGrade.FAIR
        elif score_percent >= 30:
            return QualityGrade.POOR
        else:
            return QualityGrade.FAILING


@dataclass
class ComponentQualityReport:
    """Quality report for a specific component."""
    component_name: str
    measurements: List[QualityMeasurement]
    overall_score: float
    overall_grade: QualityGrade
    passed_quality_gates: bool
    recommendations: List[str]
    timestamp: str


class QualityMetricsDefinition:
    """Definitions of quality metrics for all Flow4 components."""

    @staticmethod
    def get_document_extraction_metrics() -> List[QualityMetric]:
        """Quality metrics for document extraction phase."""
        return [
            QualityMetric(
                name="extraction_success_rate",
                description="Percentage of files successfully extracted",
                metric_type=MetricType.RELIABILITY,
                unit="percentage",
                min_threshold=95.0,
                target_value=100.0,
                weight=2.0
            ),
            QualityMetric(
                name="file_discovery_accuracy",
                description="Accuracy of file type detection and filtering",
                metric_type=MetricType.ACCURACY,
                unit="percentage",
                min_threshold=90.0,
                target_value=95.0,
                weight=1.5
            ),
            QualityMetric(
                name="extraction_speed",
                description="Files extracted per second",
                metric_type=MetricType.EFFICIENCY,
                unit="files/second",
                min_threshold=5.0,
                target_value=20.0,
                max_value=100.0,
                weight=1.0
            )
        ]

    @staticmethod
    def get_document_conversion_metrics() -> List[QualityMetric]:
        """Quality metrics for document conversion phase."""
        return [
            QualityMetric(
                name="conversion_success_rate",
                description="Percentage of documents successfully converted",
                metric_type=MetricType.RELIABILITY,
                unit="percentage",
                min_threshold=85.0,
                target_value=95.0,
                weight=2.5
            ),
            QualityMetric(
                name="content_preservation",
                description="Percentage of original content preserved",
                metric_type=MetricType.ACCURACY,
                unit="percentage",
                min_threshold=80.0,
                target_value=90.0,
                weight=2.0
            ),
            QualityMetric(
                name="format_consistency",
                description="Consistency of output markdown format",
                metric_type=MetricType.CONSISTENCY,
                unit="percentage",
                min_threshold=85.0,
                target_value=95.0,
                weight=1.5
            ),
            QualityMetric(
                name="conversion_speed",
                description="Documents converted per minute",
                metric_type=MetricType.EFFICIENCY,
                unit="docs/minute",
                min_threshold=5.0,
                target_value=15.0,
                max_value=50.0,
                weight=1.0
            ),
            QualityMetric(
                name="output_quality",
                description="Quality of converted markdown (readability, structure)",
                metric_type=MetricType.USABILITY,
                unit="score",
                min_threshold=6.0,
                target_value=8.0,
                max_value=10.0,
                weight=1.5
            )
        ]

    @staticmethod
    def get_document_chunking_metrics() -> List[QualityMetric]:
        """Quality metrics for document chunking phase."""
        return [
            QualityMetric(
                name="chunking_consistency",
                description="Consistency of chunk sizes and overlap",
                metric_type=MetricType.CONSISTENCY,
                unit="percentage",
                min_threshold=85.0,
                target_value=95.0,
                weight=2.0
            ),
            QualityMetric(
                name="semantic_coherence",
                description="Semantic coherence within chunks",
                metric_type=MetricType.ACCURACY,
                unit="score",
                min_threshold=6.0,
                target_value=8.0,
                max_value=10.0,
                weight=2.5
            ),
            QualityMetric(
                name="chunk_completeness",
                description="Percentage of original content included in chunks",
                metric_type=MetricType.COMPLETENESS,
                unit="percentage",
                min_threshold=95.0,
                target_value=98.0,
                weight=2.0
            ),
            QualityMetric(
                name="metadata_accuracy",
                description="Accuracy of chunk metadata and relationships",
                metric_type=MetricType.ACCURACY,
                unit="percentage",
                min_threshold=90.0,
                target_value=95.0,
                weight=1.5
            ),
            QualityMetric(
                name="chunking_speed",
                description="Chunks created per minute",
                metric_type=MetricType.EFFICIENCY,
                unit="chunks/minute",
                min_threshold=50.0,
                target_value=100.0,
                max_value=500.0,
                weight=1.0
            )
        ]

    @staticmethod
    def get_dataset_generation_metrics() -> List[QualityMetric]:
        """Quality metrics for dataset generation phase."""
        return [
            QualityMetric(
                name="dataset_completeness",
                description="Percentage of chunks successfully converted to training data",
                metric_type=MetricType.COMPLETENESS,
                unit="percentage",
                min_threshold=90.0,
                target_value=95.0,
                weight=2.0
            ),
            QualityMetric(
                name="data_quality",
                description="Quality of generated training examples",
                metric_type=MetricType.ACCURACY,
                unit="score",
                min_threshold=6.0,
                target_value=8.0,
                max_value=10.0,
                weight=3.0
            ),
            QualityMetric(
                name="format_validity",
                description="Percentage of examples with valid format",
                metric_type=MetricType.CONSISTENCY,
                unit="percentage",
                min_threshold=95.0,
                target_value=99.0,
                weight=2.5
            ),
            QualityMetric(
                name="diversity_score",
                description="Diversity of generated examples",
                metric_type=MetricType.ACCURACY,
                unit="score",
                min_threshold=6.0,
                target_value=8.0,
                max_value=10.0,
                weight=2.0
            ),
            QualityMetric(
                name="duplication_rate",
                description="Percentage of duplicate examples (lower is better)",
                metric_type=MetricType.CONSISTENCY,
                unit="percentage",
                min_threshold=10.0,  # Inverted: lower is better
                target_value=2.0,
                max_value=0.0,
                weight=1.5
            ),
            QualityMetric(
                name="generation_speed",
                description="Training examples generated per minute",
                metric_type=MetricType.EFFICIENCY,
                unit="examples/minute",
                min_threshold=10.0,
                target_value=30.0,
                max_value=100.0,
                weight=1.0
            )
        ]

    @staticmethod
    def get_model_training_metrics() -> List[QualityMetric]:
        """Quality metrics for model training phase."""
        return [
            QualityMetric(
                name="training_success_rate",
                description="Percentage of training runs that complete successfully",
                metric_type=MetricType.RELIABILITY,
                unit="percentage",
                min_threshold=80.0,
                target_value=95.0,
                weight=2.5
            ),
            QualityMetric(
                name="model_performance",
                description="Model performance on validation set",
                metric_type=MetricType.ACCURACY,
                unit="score",
                min_threshold=6.0,
                target_value=8.0,
                max_value=10.0,
                weight=3.0
            ),
            QualityMetric(
                name="convergence_stability",
                description="Stability of training convergence",
                metric_type=MetricType.RELIABILITY,
                unit="score",
                min_threshold=6.0,
                target_value=8.0,
                max_value=10.0,
                weight=2.0
            ),
            QualityMetric(
                name="training_efficiency",
                description="Training speed in tokens per second",
                metric_type=MetricType.EFFICIENCY,
                unit="tokens/second",
                min_threshold=100.0,
                target_value=500.0,
                max_value=2000.0,
                weight=1.5
            ),
            QualityMetric(
                name="memory_efficiency",
                description="Memory usage efficiency during training",
                metric_type=MetricType.EFFICIENCY,
                unit="percentage",
                min_threshold=60.0,
                target_value=80.0,
                weight=1.0
            )
        ]

    @staticmethod
    def get_chat_interface_metrics() -> List[QualityMetric]:
        """Quality metrics for chat interface."""
        return [
            QualityMetric(
                name="response_accuracy",
                description="Accuracy of responses to user queries",
                metric_type=MetricType.ACCURACY,
                unit="percentage",
                min_threshold=70.0,
                target_value=85.0,
                weight=3.0
            ),
            QualityMetric(
                name="response_relevance",
                description="Relevance of responses to user queries",
                metric_type=MetricType.ACCURACY,
                unit="percentage",
                min_threshold=75.0,
                target_value=90.0,
                weight=2.5
            ),
            QualityMetric(
                name="response_time",
                description="Average response time in seconds",
                metric_type=MetricType.EFFICIENCY,
                unit="seconds",
                min_threshold=10.0,
                target_value=3.0,
                max_value=1.0,
                weight=2.0
            ),
            QualityMetric(
                name="user_satisfaction",
                description="User satisfaction score",
                metric_type=MetricType.USABILITY,
                unit="score",
                min_threshold=6.0,
                target_value=8.0,
                max_value=10.0,
                weight=2.5
            ),
            QualityMetric(
                name="safety_score",
                description="Safety of responses (avoiding harmful content)",
                metric_type=MetricType.SAFETY,
                unit="percentage",
                min_threshold=95.0,
                target_value=99.0,
                weight=3.0
            ),
            QualityMetric(
                name="error_handling",
                description="Quality of error handling and recovery",
                metric_type=MetricType.RELIABILITY,
                unit="percentage",
                min_threshold=85.0,
                target_value=95.0,
                weight=1.5
            )
        ]


class QualityAssessment:
    """Quality assessment system for Flow4 components."""

    def __init__(self):
        """Initialize quality assessment system."""
        self.metrics_definitions = QualityMetricsDefinition()
        self.measurements = {}
        self.component_reports = {}

    def measure_document_extraction_quality(self, 
                                          extraction_results: Dict[str, Any]) -> ComponentQualityReport:
        """Measure quality of document extraction phase."""
        metrics = self.metrics_definitions.get_document_extraction_metrics()
        measurements = []
        
        # Calculate actual measurements
        total_files = extraction_results.get('total_files', 0)
        extracted_files = extraction_results.get('extracted_files', 0)
        extraction_time = extraction_results.get('extraction_time', 1)
        
        # Extraction success rate
        success_rate = (extracted_files / max(total_files, 1)) * 100
        measurements.append(QualityMeasurement(
            metric=metrics[0],
            measured_value=success_rate,
            timestamp=datetime.now().isoformat()
        ))
        
        # File discovery accuracy (placeholder - would need more detailed analysis)
        discovery_accuracy = 95.0  # Default assumption
        measurements.append(QualityMeasurement(
            metric=metrics[1],
            measured_value=discovery_accuracy,
            timestamp=datetime.now().isoformat()
        ))
        
        # Extraction speed
        extraction_speed = extracted_files / max(extraction_time / 60, 1)  # files per minute converted to files per second
        measurements.append(QualityMeasurement(
            metric=metrics[2],
            measured_value=extraction_speed / 60,  # Convert to files per second
            timestamp=datetime.now().isoformat()
        ))
        
        return self._create_component_report("document_extraction", measurements)

    def measure_document_conversion_quality(self, 
                                          conversion_results: Dict[str, Any]) -> ComponentQualityReport:
        """Measure quality of document conversion phase."""
        metrics = self.metrics_definitions.get_document_conversion_metrics()
        measurements = []
        
        # Extract results
        total_docs = conversion_results.get('total_documents', 0)
        converted_docs = conversion_results.get('converted_documents', 0)
        conversion_time = conversion_results.get('conversion_time', 1)
        output_quality_score = conversion_results.get('output_quality_score', 7.0)
        
        # Conversion success rate
        success_rate = (converted_docs / max(total_docs, 1)) * 100
        measurements.append(QualityMeasurement(
            metric=metrics[0],
            measured_value=success_rate,
            timestamp=datetime.now().isoformat()
        ))
        
        # Content preservation (would require detailed analysis)
        content_preservation = 85.0  # Default assumption
        measurements.append(QualityMeasurement(
            metric=metrics[1],
            measured_value=content_preservation,
            timestamp=datetime.now().isoformat()
        ))
        
        # Format consistency (would require format analysis)
        format_consistency = 90.0  # Default assumption
        measurements.append(QualityMeasurement(
            metric=metrics[2],
            measured_value=format_consistency,
            timestamp=datetime.now().isoformat()
        ))
        
        # Conversion speed
        conversion_speed = converted_docs / max(conversion_time / 60, 1)  # docs per minute
        measurements.append(QualityMeasurement(
            metric=metrics[3],
            measured_value=conversion_speed,
            timestamp=datetime.now().isoformat()
        ))
        
        # Output quality
        measurements.append(QualityMeasurement(
            metric=metrics[4],
            measured_value=output_quality_score,
            timestamp=datetime.now().isoformat()
        ))
        
        return self._create_component_report("document_conversion", measurements)

    def measure_document_chunking_quality(self, 
                                        chunking_results: Dict[str, Any]) -> ComponentQualityReport:
        """Measure quality of document chunking phase."""
        metrics = self.metrics_definitions.get_document_chunking_metrics()
        measurements = []
        
        # Extract results
        total_chunks = chunking_results.get('total_chunks', 0)
        chunking_time = chunking_results.get('chunking_time', 1)
        chunk_stats = chunking_results.get('chunk_statistics', {})
        
        # Chunking consistency (based on chunk size variance)
        chunk_sizes = chunk_stats.get('chunk_sizes', [])
        if chunk_sizes:
            size_variance = statistics.variance(chunk_sizes) if len(chunk_sizes) > 1 else 0
            consistency_score = max(0, 100 - (size_variance / 1000))  # Simplified calculation
        else:
            consistency_score = 0
        
        measurements.append(QualityMeasurement(
            metric=metrics[0],
            measured_value=consistency_score,
            timestamp=datetime.now().isoformat()
        ))
        
        # Semantic coherence (would require semantic analysis)
        semantic_coherence = 7.5  # Default assumption
        measurements.append(QualityMeasurement(
            metric=metrics[1],
            measured_value=semantic_coherence,
            timestamp=datetime.now().isoformat()
        ))
        
        # Chunk completeness
        completeness = chunking_results.get('completeness_percentage', 97.0)
        measurements.append(QualityMeasurement(
            metric=metrics[2],
            measured_value=completeness,
            timestamp=datetime.now().isoformat()
        ))
        
        # Metadata accuracy (would require validation)
        metadata_accuracy = 92.0  # Default assumption
        measurements.append(QualityMeasurement(
            metric=metrics[3],
            measured_value=metadata_accuracy,
            timestamp=datetime.now().isoformat()
        ))
        
        # Chunking speed
        chunking_speed = total_chunks / max(chunking_time / 60, 1)  # chunks per minute
        measurements.append(QualityMeasurement(
            metric=metrics[4],
            measured_value=chunking_speed,
            timestamp=datetime.now().isoformat()
        ))
        
        return self._create_component_report("document_chunking", measurements)

    def measure_dataset_generation_quality(self, 
                                         generation_results: Dict[str, Any]) -> ComponentQualityReport:
        """Measure quality of dataset generation phase."""
        metrics = self.metrics_definitions.get_dataset_generation_metrics()
        measurements = []
        
        # Extract results
        total_examples = generation_results.get('total_examples', 0)
        valid_examples = generation_results.get('valid_examples', 0)
        generation_time = generation_results.get('generation_time', 1)
        quality_scores = generation_results.get('quality_scores', {})
        
        # Dataset completeness
        completeness = (valid_examples / max(total_examples, 1)) * 100 if total_examples > 0 else 0
        measurements.append(QualityMeasurement(
            metric=metrics[0],
            measured_value=completeness,
            timestamp=datetime.now().isoformat()
        ))
        
        # Data quality
        data_quality = quality_scores.get('average_quality', 7.0)
        measurements.append(QualityMeasurement(
            metric=metrics[1],
            measured_value=data_quality,
            timestamp=datetime.now().isoformat()
        ))
        
        # Format validity
        format_validity = quality_scores.get('format_validity', 95.0)
        measurements.append(QualityMeasurement(
            metric=metrics[2],
            measured_value=format_validity,
            timestamp=datetime.now().isoformat()
        ))
        
        # Diversity score
        diversity_score = quality_scores.get('diversity_score', 7.0)
        measurements.append(QualityMeasurement(
            metric=metrics[3],
            measured_value=diversity_score,
            timestamp=datetime.now().isoformat()
        ))
        
        # Duplication rate (inverted metric)
        duplication_rate = quality_scores.get('duplication_rate', 5.0)
        measurements.append(QualityMeasurement(
            metric=metrics[4],
            measured_value=duplication_rate,
            timestamp=datetime.now().isoformat()
        ))
        
        # Generation speed
        generation_speed = total_examples / max(generation_time / 60, 1)  # examples per minute
        measurements.append(QualityMeasurement(
            metric=metrics[5],
            measured_value=generation_speed,
            timestamp=datetime.now().isoformat()
        ))
        
        return self._create_component_report("dataset_generation", measurements)

    def measure_model_training_quality(self, 
                                     training_results: Dict[str, Any]) -> ComponentQualityReport:
        """Measure quality of model training phase."""
        metrics = self.metrics_definitions.get_model_training_metrics()
        measurements = []
        
        # Extract results
        training_success = training_results.get('training_completed', False)
        model_performance = training_results.get('validation_score', 6.5)
        training_metrics = training_results.get('training_metrics', {})
        
        # Training success rate
        success_rate = 100.0 if training_success else 0.0
        measurements.append(QualityMeasurement(
            metric=metrics[0],
            measured_value=success_rate,
            timestamp=datetime.now().isoformat()
        ))
        
        # Model performance
        measurements.append(QualityMeasurement(
            metric=metrics[1],
            measured_value=model_performance,
            timestamp=datetime.now().isoformat()
        ))
        
        # Convergence stability
        convergence_stability = training_metrics.get('convergence_stability', 7.0)
        measurements.append(QualityMeasurement(
            metric=metrics[2],
            measured_value=convergence_stability,
            timestamp=datetime.now().isoformat()
        ))
        
        # Training efficiency
        training_speed = training_metrics.get('tokens_per_second', 200.0)
        measurements.append(QualityMeasurement(
            metric=metrics[3],
            measured_value=training_speed,
            timestamp=datetime.now().isoformat()
        ))
        
        # Memory efficiency
        memory_efficiency = training_metrics.get('memory_efficiency', 70.0)
        measurements.append(QualityMeasurement(
            metric=metrics[4],
            measured_value=memory_efficiency,
            timestamp=datetime.now().isoformat()
        ))
        
        return self._create_component_report("model_training", measurements)

    def measure_chat_interface_quality(self, 
                                     interface_results: Dict[str, Any]) -> ComponentQualityReport:
        """Measure quality of chat interface."""
        metrics = self.metrics_definitions.get_chat_interface_metrics()
        measurements = []
        
        # Extract results
        test_results = interface_results.get('test_results', {})
        performance_metrics = interface_results.get('performance_metrics', {})
        
        # Response accuracy
        accuracy = test_results.get('accuracy_score', 75.0)
        measurements.append(QualityMeasurement(
            metric=metrics[0],
            measured_value=accuracy,
            timestamp=datetime.now().isoformat()
        ))
        
        # Response relevance
        relevance = test_results.get('relevance_score', 80.0)
        measurements.append(QualityMeasurement(
            metric=metrics[1],
            measured_value=relevance,
            timestamp=datetime.now().isoformat()
        ))
        
        # Response time
        response_time = performance_metrics.get('avg_response_time', 5.0)
        measurements.append(QualityMeasurement(
            metric=metrics[2],
            measured_value=response_time,
            timestamp=datetime.now().isoformat()
        ))
        
        # User satisfaction
        satisfaction = test_results.get('user_satisfaction', 7.0)
        measurements.append(QualityMeasurement(
            metric=metrics[3],
            measured_value=satisfaction,
            timestamp=datetime.now().isoformat()
        ))
        
        # Safety score
        safety_score = test_results.get('safety_score', 96.0)
        measurements.append(QualityMeasurement(
            metric=metrics[4],
            measured_value=safety_score,
            timestamp=datetime.now().isoformat()
        ))
        
        # Error handling
        error_handling = test_results.get('error_handling_score', 90.0)
        measurements.append(QualityMeasurement(
            metric=metrics[5],
            measured_value=error_handling,
            timestamp=datetime.now().isoformat()
        ))
        
        return self._create_component_report("chat_interface", measurements)

    def _create_component_report(self, 
                               component_name: str, 
                               measurements: List[QualityMeasurement]) -> ComponentQualityReport:
        """Create a quality report for a component."""
        if not measurements:
            return ComponentQualityReport(
                component_name=component_name,
                measurements=[],
                overall_score=0.0,
                overall_grade=QualityGrade.FAILING,
                passed_quality_gates=False,
                recommendations=["No measurements available"],
                timestamp=datetime.now().isoformat()
            )
        
        # Calculate weighted overall score
        total_weight = sum(m.metric.weight for m in measurements)
        weighted_score = sum(m.score * m.metric.weight for m in measurements) / total_weight
        
        # Determine overall grade
        score_percent = weighted_score * 100
        if score_percent >= 90:
            overall_grade = QualityGrade.EXCELLENT
        elif score_percent >= 70:
            overall_grade = QualityGrade.GOOD
        elif score_percent >= 50:
            overall_grade = QualityGrade.FAIR
        elif score_percent >= 30:
            overall_grade = QualityGrade.POOR
        else:
            overall_grade = QualityGrade.FAILING
        
        # Check quality gates (all critical metrics must pass minimum threshold)
        critical_metrics = [m for m in measurements if m.metric.weight >= 2.0]
        passed_quality_gates = all(m.measured_value >= m.metric.min_threshold for m in critical_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(measurements)
        
        report = ComponentQualityReport(
            component_name=component_name,
            measurements=measurements,
            overall_score=weighted_score,
            overall_grade=overall_grade,
            passed_quality_gates=passed_quality_gates,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
        
        # Store report
        self.component_reports[component_name] = report
        
        return report

    def _generate_recommendations(self, measurements: List[QualityMeasurement]) -> List[str]:
        """Generate recommendations based on measurements."""
        recommendations = []
        
        for measurement in measurements:
            if measurement.measured_value < measurement.metric.min_threshold:
                recommendations.append(
                    f"Improve {measurement.metric.name}: "
                    f"current {measurement.measured_value:.1f}, "
                    f"minimum required {measurement.metric.min_threshold:.1f}"
                )
            elif measurement.measured_value < measurement.metric.target_value:
                recommendations.append(
                    f"Optimize {measurement.metric.name}: "
                    f"current {measurement.measured_value:.1f}, "
                    f"target {measurement.metric.target_value:.1f}"
                )
        
        # Add component-specific recommendations
        if not recommendations:
            recommendations.append("All metrics meet minimum requirements. Consider optimization for target values.")
        
        return recommendations

    def generate_comprehensive_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report for all components."""
        if not self.component_reports:
            return {
                'status': 'error',
                'message': 'No component reports available',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate overall system quality
        all_scores = [report.overall_score for report in self.component_reports.values()]
        overall_score = statistics.mean(all_scores) if all_scores else 0.0
        
        # Determine overall grade
        score_percent = overall_score * 100
        if score_percent >= 90:
            overall_grade = QualityGrade.EXCELLENT
        elif score_percent >= 70:
            overall_grade = QualityGrade.GOOD
        elif score_percent >= 50:
            overall_grade = QualityGrade.FAIR
        elif score_percent >= 30:
            overall_grade = QualityGrade.POOR
        else:
            overall_grade = QualityGrade.FAILING
        
        # Check if all critical quality gates pass
        all_gates_passed = all(report.passed_quality_gates for report in self.component_reports.values())
        
        # Collect all recommendations
        all_recommendations = []
        for report in self.component_reports.values():
            all_recommendations.extend(report.recommendations)
        
        # Component summaries
        component_summaries = {}
        for name, report in self.component_reports.items():
            component_summaries[name] = {
                'overall_score': report.overall_score,
                'overall_grade': report.overall_grade.value,
                'passed_quality_gates': report.passed_quality_gates,
                'measurements_count': len(report.measurements),
                'recommendations_count': len(report.recommendations)
            }
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'overall_quality': {
                'score': overall_score,
                'grade': overall_grade.value,
                'percentage': score_percent,
                'all_quality_gates_passed': all_gates_passed
            },
            'component_summaries': component_summaries,
            'total_components': len(self.component_reports),
            'components_passing_gates': sum(1 for r in self.component_reports.values() if r.passed_quality_gates),
            'system_recommendations': list(set(all_recommendations)),  # Remove duplicates
            'quality_trends': self._calculate_quality_trends()
        }

    def _calculate_quality_trends(self) -> Dict[str, Any]:
        """Calculate quality trends (placeholder for trend analysis)."""
        # This would require historical data to implement properly
        return {
            'trend': 'stable',
            'message': 'Quality trend analysis requires historical data'
        }

    def save_quality_report(self, output_file: str):
        """Save comprehensive quality report to file."""
        report = self.generate_comprehensive_quality_report()
        
        # Add detailed component reports
        detailed_reports = {}
        for name, component_report in self.component_reports.items():
            detailed_reports[name] = {
                'component_name': component_report.component_name,
                'overall_score': component_report.overall_score,
                'overall_grade': component_report.overall_grade.value,
                'passed_quality_gates': component_report.passed_quality_gates,
                'timestamp': component_report.timestamp,
                'measurements': [
                    {
                        'metric_name': m.metric.name,
                        'metric_description': m.metric.description,
                        'metric_type': m.metric.metric_type.value,
                        'measured_value': m.measured_value,
                        'min_threshold': m.metric.min_threshold,
                        'target_value': m.metric.target_value,
                        'score': m.score,
                        'grade': m.grade.value,
                        'weight': m.metric.weight
                    }
                    for m in component_report.measurements
                ],
                'recommendations': component_report.recommendations
            }
        
        report['detailed_component_reports'] = detailed_reports
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Quality report saved to {output_file}")


def main():
    """Main function for testing quality metrics."""
    # Example usage
    assessment = QualityAssessment()
    
    # Example measurements
    extraction_results = {
        'total_files': 100,
        'extracted_files': 95,
        'extraction_time': 60  # seconds
    }
    
    conversion_results = {
        'total_documents': 95,
        'converted_documents': 90,
        'conversion_time': 300,  # seconds
        'output_quality_score': 8.0
    }
    
    chunking_results = {
        'total_chunks': 500,
        'chunking_time': 120,  # seconds
        'chunk_statistics': {
            'chunk_sizes': [800, 900, 850, 750, 900, 800, 950]
        },
        'completeness_percentage': 98.0
    }
    
    # Generate reports
    extraction_report = assessment.measure_document_extraction_quality(extraction_results)
    conversion_report = assessment.measure_document_conversion_quality(conversion_results)
    chunking_report = assessment.measure_document_chunking_quality(chunking_results)
    
    # Generate comprehensive report
    comprehensive_report = assessment.generate_comprehensive_quality_report()
    
    print(json.dumps(comprehensive_report, indent=2))
    
    # Save report
    assessment.save_quality_report("quality_report.json")
    print("Quality report saved to quality_report.json")


if __name__ == "__main__":
    main()