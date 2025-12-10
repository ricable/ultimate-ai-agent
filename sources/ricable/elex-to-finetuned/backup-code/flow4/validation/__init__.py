"""
Flow4 Validation Package

Comprehensive validation, testing, and quality assurance for Flow4 pipeline.

This package provides:
- Pipeline validation for each processing stage
- Dataset validation with quality metrics
- Model validation and performance testing
- Chat interface testing
- Error handling and recovery
- Quality metrics and acceptance criteria
"""

from .pipeline_validator import PipelineValidator, ValidationResult, PipelineValidationReport
from .dataset_validator import DatasetValidator, DatasetValidationResult
from .model_validator import ModelValidator, ModelTestResult, ModelValidationReport
from .chat_tester import ChatInterfaceTester, ChatTestResult, ChatValidationReport
from .error_handler import ErrorHandler, ErrorRecord, ErrorSeverity, ErrorCategory, ErrorRecoveryManager
from .quality_metrics import QualityAssessment, QualityMetric, QualityMeasurement, QualityGrade, MetricType

__version__ = "1.0.0"

__all__ = [
    # Pipeline validation
    "PipelineValidator",
    "ValidationResult", 
    "PipelineValidationReport",
    
    # Dataset validation
    "DatasetValidator",
    "DatasetValidationResult",
    
    # Model validation
    "ModelValidator",
    "ModelTestResult",
    "ModelValidationReport",
    
    # Chat testing
    "ChatInterfaceTester",
    "ChatTestResult", 
    "ChatValidationReport",
    
    # Error handling
    "ErrorHandler",
    "ErrorRecord",
    "ErrorSeverity",
    "ErrorCategory", 
    "ErrorRecoveryManager",
    
    # Quality metrics
    "QualityAssessment",
    "QualityMetric",
    "QualityMeasurement",
    "QualityGrade",
    "MetricType"
]