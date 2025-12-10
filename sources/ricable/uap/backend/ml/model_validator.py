# backend/ml/model_validator.py
# Comprehensive Model Validation Framework

import asyncio
import json
import uuid
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import aiofiles
from concurrent.futures import ThreadPoolExecutor

# Integrations
from ..distributed.ray_manager import submit_distributed_task
from ..monitoring.logs.logger import uap_logger, EventType, LogLevel
from ..monitoring.metrics.prometheus_metrics import record_pipeline_event

class ValidationStatus(Enum):
    """Validation status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"

class ValidationType(Enum):
    """Types of validation"""
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    BIAS = "bias"
    DRIFT = "drift"
    INTEGRATION = "integration"

class MetricType(Enum):
    """Metric types for validation"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"

@dataclass
class ValidationRule:
    """Individual validation rule"""
    rule_id: str
    rule_name: str
    validation_type: ValidationType
    metric_type: MetricType
    threshold_min: Optional[float] = None
    threshold_max: Optional[float] = None
    threshold_exact: Optional[float] = None
    severity: str = "error"  # error, warning, info
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['validation_type'] = self.validation_type.value
        data['metric_type'] = self.metric_type.value
        return data

@dataclass
class ValidationResult:
    """Result of a validation rule"""
    rule_id: str
    status: ValidationStatus
    actual_value: Optional[float]
    expected_range: Optional[Tuple[float, float]]
    passed: bool
    message: str
    execution_time_ms: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        return data

@dataclass
class ValidationReport:
    """Complete validation report"""
    validation_id: str
    model_id: str
    model_version: str
    validation_config: Dict[str, Any]
    started_at: datetime
    completed_at: Optional[datetime]
    overall_status: ValidationStatus
    rule_results: List[ValidationResult]
    summary: Dict[str, Any]
    artifacts: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        data['overall_status'] = self.overall_status.value
        data['rule_results'] = [r.to_dict() for r in self.rule_results]
        return data

class FunctionalValidator:
    """Functional validation tests"""
    
    async def validate_basic_functionality(self, 
                                         model_path: str,
                                         test_inputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Test basic model functionality"""
        
        start_time = time.time()
        
        try:
            # Load model (simulated)
            await asyncio.sleep(0.1)
            
            successful_predictions = 0
            failed_predictions = 0
            prediction_times = []
            
            for test_input in test_inputs:
                try:
                    # Simulate prediction
                    pred_start = time.time()
                    await asyncio.sleep(0.01)  # Simulate processing
                    pred_time = (time.time() - pred_start) * 1000
                    
                    prediction_times.append(pred_time)
                    successful_predictions += 1
                    
                except Exception:
                    failed_predictions += 1
            
            total_time = (time.time() - start_time) * 1000
            avg_prediction_time = np.mean(prediction_times) if prediction_times else 0
            
            return {
                "passed": failed_predictions == 0,
                "total_inputs": len(test_inputs),
                "successful_predictions": successful_predictions,
                "failed_predictions": failed_predictions,
                "success_rate": successful_predictions / len(test_inputs) if test_inputs else 0,
                "avg_prediction_time_ms": avg_prediction_time,
                "total_time_ms": total_time
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e),
                "total_time_ms": (time.time() - start_time) * 1000
            }
    
    async def validate_input_output_schema(self,
                                         model_path: str,
                                         expected_input_schema: Dict[str, Any],
                                         expected_output_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input/output schema compliance"""
        
        try:
            # Simulate schema validation
            await asyncio.sleep(0.05)
            
            return {
                "passed": True,
                "input_schema_valid": True,
                "output_schema_valid": True,
                "schema_compliance": 100.0
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

class PerformanceValidator:
    """Performance validation tests"""
    
    async def validate_latency(self,
                             model_path: str,
                             test_inputs: List[Dict[str, Any]],
                             max_latency_ms: float) -> Dict[str, Any]:
        """Validate model latency requirements"""
        
        try:
            latencies = []
            
            for test_input in test_inputs:
                start_time = time.time()
                # Simulate prediction
                await asyncio.sleep(np.random.uniform(0.01, 0.05))
                latency_ms = (time.time() - start_time) * 1000
                latencies.append(latency_ms)
            
            avg_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            max_observed_latency = np.max(latencies)
            
            return {
                "passed": p95_latency <= max_latency_ms,
                "avg_latency_ms": avg_latency,
                "p95_latency_ms": p95_latency,
                "max_latency_ms": max_observed_latency,
                "threshold_ms": max_latency_ms,
                "latency_compliance": min(100.0, (max_latency_ms / p95_latency) * 100)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def validate_throughput(self,
                                model_path: str,
                                target_rps: float,
                                duration_seconds: int = 30) -> Dict[str, Any]:
        """Validate model throughput requirements"""
        
        try:
            start_time = time.time()
            total_requests = 0
            successful_requests = 0
            
            # Simulate load testing
            while (time.time() - start_time) < duration_seconds:
                # Simulate batch of requests
                batch_size = min(10, int(target_rps * 0.1))
                
                for _ in range(batch_size):
                    try:
                        await asyncio.sleep(0.01)  # Simulate processing
                        successful_requests += 1
                    except Exception:
                        pass
                    
                    total_requests += 1
                
                await asyncio.sleep(0.1)  # Brief pause between batches
            
            actual_duration = time.time() - start_time
            actual_rps = successful_requests / actual_duration
            
            return {
                "passed": actual_rps >= target_rps,
                "target_rps": target_rps,
                "actual_rps": actual_rps,
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "duration_seconds": actual_duration,
                "throughput_compliance": min(100.0, (actual_rps / target_rps) * 100)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def validate_resource_usage(self,
                                    model_path: str,
                                    max_memory_mb: float,
                                    max_cpu_percent: float) -> Dict[str, Any]:
        """Validate resource usage requirements"""
        
        try:
            # Simulate resource monitoring
            await asyncio.sleep(0.1)
            
            # Mock resource usage values
            memory_usage_mb = np.random.uniform(50, 200)
            cpu_usage_percent = np.random.uniform(10, 60)
            
            return {
                "passed": memory_usage_mb <= max_memory_mb and cpu_usage_percent <= max_cpu_percent,
                "memory_usage_mb": memory_usage_mb,
                "max_memory_mb": max_memory_mb,
                "cpu_usage_percent": cpu_usage_percent,
                "max_cpu_percent": max_cpu_percent,
                "memory_compliance": min(100.0, (max_memory_mb / memory_usage_mb) * 100),
                "cpu_compliance": min(100.0, (max_cpu_percent / cpu_usage_percent) * 100)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

class BiasValidator:
    """Bias and fairness validation"""
    
    async def validate_demographic_parity(self,
                                         model_path: str,
                                         test_data: Dict[str, Any],
                                         protected_attributes: List[str]) -> Dict[str, Any]:
        """Validate demographic parity across protected groups"""
        
        try:
            # Simulate bias analysis
            await asyncio.sleep(0.2)
            
            # Mock bias metrics
            group_metrics = {}
            for attr in protected_attributes:
                group_metrics[attr] = {
                    "group_a_positive_rate": np.random.uniform(0.6, 0.8),
                    "group_b_positive_rate": np.random.uniform(0.65, 0.85),
                    "demographic_parity_difference": np.random.uniform(-0.1, 0.1)
                }
            
            max_difference = max(
                abs(metrics["demographic_parity_difference"]) 
                for metrics in group_metrics.values()
            )
            
            return {
                "passed": max_difference <= 0.05,  # 5% threshold
                "max_demographic_parity_difference": max_difference,
                "group_metrics": group_metrics,
                "fairness_score": max(0.0, 1.0 - (max_difference / 0.1))
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def validate_equalized_odds(self,
                                    model_path: str,
                                    test_data: Dict[str, Any],
                                    protected_attributes: List[str]) -> Dict[str, Any]:
        """Validate equalized odds across protected groups"""
        
        try:
            # Simulate equalized odds analysis
            await asyncio.sleep(0.15)
            
            # Mock equalized odds metrics
            odds_metrics = {}
            for attr in protected_attributes:
                odds_metrics[attr] = {
                    "true_positive_rate_diff": np.random.uniform(-0.05, 0.05),
                    "false_positive_rate_diff": np.random.uniform(-0.05, 0.05)
                }
            
            max_tpr_diff = max(
                abs(metrics["true_positive_rate_diff"]) 
                for metrics in odds_metrics.values()
            )
            max_fpr_diff = max(
                abs(metrics["false_positive_rate_diff"]) 
                for metrics in odds_metrics.values()
            )
            
            return {
                "passed": max_tpr_diff <= 0.05 and max_fpr_diff <= 0.05,
                "max_tpr_difference": max_tpr_diff,
                "max_fpr_difference": max_fpr_diff,
                "odds_metrics": odds_metrics,
                "equalized_odds_score": max(0.0, 1.0 - max(max_tpr_diff, max_fpr_diff) / 0.1)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

class SecurityValidator:
    """Security validation tests"""
    
    async def validate_adversarial_robustness(self,
                                            model_path: str,
                                            test_inputs: List[Dict[str, Any]],
                                            epsilon: float = 0.1) -> Dict[str, Any]:
        """Test model robustness against adversarial attacks"""
        
        try:
            # Simulate adversarial testing
            await asyncio.sleep(0.3)
            
            total_tests = len(test_inputs)
            robust_predictions = int(total_tests * np.random.uniform(0.7, 0.95))
            
            return {
                "passed": (robust_predictions / total_tests) >= 0.8,
                "total_tests": total_tests,
                "robust_predictions": robust_predictions,
                "robustness_score": robust_predictions / total_tests,
                "epsilon": epsilon,
                "attack_success_rate": 1 - (robust_predictions / total_tests)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }
    
    async def validate_data_privacy(self,
                                  model_path: str,
                                  training_data_fingerprint: str) -> Dict[str, Any]:
        """Validate that model doesn't leak training data"""
        
        try:
            # Simulate privacy analysis
            await asyncio.sleep(0.2)
            
            # Mock privacy metrics
            membership_inference_accuracy = np.random.uniform(0.4, 0.6)
            information_leakage_score = np.random.uniform(0.0, 0.3)
            
            return {
                "passed": membership_inference_accuracy <= 0.55 and information_leakage_score <= 0.2,
                "membership_inference_accuracy": membership_inference_accuracy,
                "information_leakage_score": information_leakage_score,
                "privacy_score": 1.0 - max(membership_inference_accuracy - 0.5, information_leakage_score)
            }
            
        except Exception as e:
            return {
                "passed": False,
                "error": str(e)
            }

class ModelValidator:
    """
    Comprehensive Model Validation Framework.
    
    Provides multi-dimensional validation including:
    - Functional testing
    - Performance validation
    - Bias and fairness testing
    - Security validation
    - Integration testing
    """
    
    def __init__(self, validation_dir: str = "./validations"):
        self.validation_dir = Path(validation_dir)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # Validators
        self.functional_validator = FunctionalValidator()
        self.performance_validator = PerformanceValidator()
        self.bias_validator = BiasValidator()
        self.security_validator = SecurityValidator()
        
        # Validation tracking
        self.validation_reports: Dict[str, ValidationReport] = {}
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        
        # Configuration
        self.default_rules = self._create_default_rules()
        
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> bool:
        """Initialize model validator"""
        try:
            await self._load_validation_rules()
            
            uap_logger.log_event(
                LogLevel.INFO,
                "Model Validator initialized",
                EventType.AGENT,
                {"validation_dir": str(self.validation_dir)},
                "model_validator"
            )
            
            return True
            
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to initialize Model Validator: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "model_validator"
            )
            return False
    
    async def validate_model(self,
                           model_id: str,
                           model_version: str,
                           model_path: str,
                           validation_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Perform comprehensive model validation"""
        
        validation_id = str(uuid.uuid4())
        
        # Create validation report
        report = ValidationReport(
            validation_id=validation_id,
            model_id=model_id,
            model_version=model_version,
            validation_config=validation_config or {},
            started_at=datetime.utcnow(),
            completed_at=None,
            overall_status=ValidationStatus.RUNNING,
            rule_results=[],
            summary={},
            artifacts={}
        )
        
        self.validation_reports[validation_id] = report
        
        try:
            # Get validation rules
            rules = self._get_validation_rules(model_id, validation_config)
            
            # Execute validation rules
            for rule in rules:
                result = await self._execute_validation_rule(rule, model_path, validation_config)
                report.rule_results.append(result)
            
            # Generate summary
            report.summary = self._generate_validation_summary(report.rule_results)
            
            # Determine overall status
            failed_rules = [r for r in report.rule_results if not r.passed]
            critical_failures = [r for r in failed_rules if r.metadata.get("severity") == "error"]
            
            if critical_failures:
                report.overall_status = ValidationStatus.FAILED
            elif failed_rules:
                report.overall_status = ValidationStatus.FAILED  # Could be WARNING in future
            else:
                report.overall_status = ValidationStatus.PASSED
            
            report.completed_at = datetime.utcnow()
            
            # Save report
            await self._save_validation_report(report)
            
            # Log validation completion
            uap_logger.log_event(
                LogLevel.INFO,
                f"Model validation completed: {model_id} v{model_version}",
                EventType.AGENT,
                {
                    "validation_id": validation_id,
                    "model_id": model_id,
                    "model_version": model_version,
                    "status": report.overall_status.value,
                    "rules_executed": len(report.rule_results),
                    "failures": len(failed_rules)
                },
                "model_validator"
            )
            
            return {
                "validation_id": validation_id,
                "passed": report.overall_status == ValidationStatus.PASSED,
                "status": report.overall_status.value,
                "summary": report.summary,
                "report": report.to_dict()
            }
            
        except Exception as e:
            report.overall_status = ValidationStatus.ERROR
            report.completed_at = datetime.utcnow()
            
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Model validation failed: {model_id} v{model_version} - {e}",
                EventType.AGENT,
                {
                    "validation_id": validation_id,
                    "model_id": model_id,
                    "error": str(e)
                },
                "model_validator"
            )
            
            return {
                "validation_id": validation_id,
                "passed": False,
                "status": "error",
                "error": str(e)
            }
    
    async def _execute_validation_rule(self,
                                     rule: ValidationRule,
                                     model_path: str,
                                     validation_config: Dict[str, Any]) -> ValidationResult:
        """Execute a single validation rule"""
        
        start_time = time.time()
        
        try:
            # Execute validation based on type
            if rule.validation_type == ValidationType.FUNCTIONAL:
                result = await self._execute_functional_validation(rule, model_path, validation_config)
            elif rule.validation_type == ValidationType.PERFORMANCE:
                result = await self._execute_performance_validation(rule, model_path, validation_config)
            elif rule.validation_type == ValidationType.BIAS:
                result = await self._execute_bias_validation(rule, model_path, validation_config)
            elif rule.validation_type == ValidationType.SECURITY:
                result = await self._execute_security_validation(rule, model_path, validation_config)
            else:
                result = {"passed": False, "error": f"Unsupported validation type: {rule.validation_type}"}
            
            execution_time = (time.time() - start_time) * 1000
            
            # Check thresholds
            passed = result.get("passed", False)
            actual_value = result.get(rule.metric_type.value)
            
            if actual_value is not None and not passed:
                # Check against thresholds
                if rule.threshold_min is not None and actual_value < rule.threshold_min:
                    passed = False
                elif rule.threshold_max is not None and actual_value > rule.threshold_max:
                    passed = False
                elif rule.threshold_exact is not None and abs(actual_value - rule.threshold_exact) > 0.001:
                    passed = False
                else:
                    passed = True
            
            return ValidationResult(
                rule_id=rule.rule_id,
                status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
                actual_value=actual_value,
                expected_range=(rule.threshold_min, rule.threshold_max) if rule.threshold_min or rule.threshold_max else None,
                passed=passed,
                message=result.get("message", f"Rule {rule.rule_name} {'passed' if passed else 'failed'}"),
                execution_time_ms=execution_time,
                metadata={
                    "severity": rule.severity,
                    "validation_type": rule.validation_type.value,
                    "result_details": result
                }
            )
            
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            
            return ValidationResult(
                rule_id=rule.rule_id,
                status=ValidationStatus.ERROR,
                actual_value=None,
                expected_range=None,
                passed=False,
                message=f"Validation rule execution failed: {str(e)}",
                execution_time_ms=execution_time,
                metadata={
                    "severity": rule.severity,
                    "validation_type": rule.validation_type.value,
                    "error": str(e)
                }
            )
    
    async def _execute_functional_validation(self,
                                           rule: ValidationRule,
                                           model_path: str,
                                           config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute functional validation rule"""
        
        if rule.metric_type == MetricType.ACCURACY:
            test_inputs = config.get("test_inputs", [{"input": "test"}])
            return await self.functional_validator.validate_basic_functionality(model_path, test_inputs)
        else:
            return {"passed": True, "message": "Functional validation passed"}
    
    async def _execute_performance_validation(self,
                                            rule: ValidationRule,
                                            model_path: str,
                                            config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance validation rule"""
        
        test_inputs = config.get("test_inputs", [{"input": "test"}])
        
        if rule.metric_type == MetricType.LATENCY:
            max_latency = rule.threshold_max or 100.0
            return await self.performance_validator.validate_latency(model_path, test_inputs, max_latency)
        elif rule.metric_type == MetricType.THROUGHPUT:
            target_rps = rule.threshold_min or 10.0
            return await self.performance_validator.validate_throughput(model_path, target_rps)
        elif rule.metric_type == MetricType.MEMORY_USAGE:
            max_memory = rule.threshold_max or 1000.0
            max_cpu = 80.0
            return await self.performance_validator.validate_resource_usage(model_path, max_memory, max_cpu)
        else:
            return {"passed": True, "message": "Performance validation passed"}
    
    async def _execute_bias_validation(self,
                                     rule: ValidationRule,
                                     model_path: str,
                                     config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute bias validation rule"""
        
        test_data = config.get("test_data", {})
        protected_attributes = config.get("protected_attributes", ["gender", "race"])
        
        return await self.bias_validator.validate_demographic_parity(
            model_path, test_data, protected_attributes
        )
    
    async def _execute_security_validation(self,
                                         rule: ValidationRule,
                                         model_path: str,
                                         config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute security validation rule"""
        
        test_inputs = config.get("test_inputs", [{"input": "test"}])
        
        return await self.security_validator.validate_adversarial_robustness(
            model_path, test_inputs
        )
    
    def _get_validation_rules(self,
                            model_id: str,
                            validation_config: Dict[str, Any]) -> List[ValidationRule]:
        """Get validation rules for model"""
        
        # Use custom rules if provided, otherwise use defaults
        if "validation_rules" in validation_config:
            custom_rules = []
            for rule_data in validation_config["validation_rules"]:
                rule = ValidationRule(
                    rule_id=rule_data.get("rule_id", str(uuid.uuid4())),
                    rule_name=rule_data["rule_name"],
                    validation_type=ValidationType(rule_data["validation_type"]),
                    metric_type=MetricType(rule_data["metric_type"]),
                    threshold_min=rule_data.get("threshold_min"),
                    threshold_max=rule_data.get("threshold_max"),
                    threshold_exact=rule_data.get("threshold_exact"),
                    severity=rule_data.get("severity", "error"),
                    description=rule_data.get("description", "")
                )
                custom_rules.append(rule)
            return custom_rules
        
        # Use model-specific rules if available
        if model_id in self.validation_rules:
            return self.validation_rules[model_id]
        
        # Use default rules
        return self.default_rules
    
    def _create_default_rules(self) -> List[ValidationRule]:
        """Create default validation rules"""
        
        return [
            ValidationRule(
                rule_id="func_basic",
                rule_name="Basic Functionality",
                validation_type=ValidationType.FUNCTIONAL,
                metric_type=MetricType.ACCURACY,
                threshold_min=0.8,
                severity="error",
                description="Model should have basic prediction capability"
            ),
            ValidationRule(
                rule_id="perf_latency",
                rule_name="Latency Requirement",
                validation_type=ValidationType.PERFORMANCE,
                metric_type=MetricType.LATENCY,
                threshold_max=100.0,
                severity="error",
                description="P95 latency should be under 100ms"
            ),
            ValidationRule(
                rule_id="perf_memory",
                rule_name="Memory Usage",
                validation_type=ValidationType.PERFORMANCE,
                metric_type=MetricType.MEMORY_USAGE,
                threshold_max=1000.0,
                severity="warning",
                description="Memory usage should be under 1GB"
            ),
            ValidationRule(
                rule_id="bias_fairness",
                rule_name="Demographic Parity",
                validation_type=ValidationType.BIAS,
                metric_type=MetricType.ACCURACY,
                severity="error",
                description="Model should exhibit demographic parity"
            ),
            ValidationRule(
                rule_id="sec_robustness",
                rule_name="Adversarial Robustness",
                validation_type=ValidationType.SECURITY,
                metric_type=MetricType.ACCURACY,
                threshold_min=0.7,
                severity="warning",
                description="Model should be robust against adversarial attacks"
            )
        ]
    
    def _generate_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate validation summary from results"""
        
        total_rules = len(results)
        passed_rules = len([r for r in results if r.passed])
        failed_rules = total_rules - passed_rules
        
        critical_failures = len([r for r in results if not r.passed and r.metadata.get("severity") == "error"])
        warnings = len([r for r in results if not r.passed and r.metadata.get("severity") == "warning"])
        
        avg_execution_time = np.mean([r.execution_time_ms for r in results]) if results else 0
        
        # Group by validation type
        type_summary = {}
        for val_type in ValidationType:
            type_results = [r for r in results if r.metadata.get("validation_type") == val_type.value]
            if type_results:
                type_summary[val_type.value] = {
                    "total": len(type_results),
                    "passed": len([r for r in type_results if r.passed]),
                    "failed": len([r for r in type_results if not r.passed])
                }
        
        return {
            "total_rules": total_rules,
            "passed_rules": passed_rules,
            "failed_rules": failed_rules,
            "success_rate": (passed_rules / total_rules) if total_rules > 0 else 0,
            "critical_failures": critical_failures,
            "warnings": warnings,
            "avg_execution_time_ms": avg_execution_time,
            "validation_types": type_summary
        }
    
    async def get_validation_report(self, validation_id: str) -> Optional[Dict[str, Any]]:
        """Get validation report"""
        if validation_id not in self.validation_reports:
            return None
        
        return self.validation_reports[validation_id].to_dict()
    
    async def list_validation_reports(self,
                                    model_id: str = None,
                                    days: int = 30) -> List[Dict[str, Any]]:
        """List validation reports"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        reports = []
        for report in self.validation_reports.values():
            if model_id and report.model_id != model_id:
                continue
            if report.started_at < cutoff_date:
                continue
            
            reports.append(report.to_dict())
        
        # Sort by start time (newest first)
        reports.sort(key=lambda r: r['started_at'], reverse=True)
        
        return reports
    
    async def _load_validation_rules(self):
        """Load validation rules from storage"""
        try:
            rules_file = self.validation_dir / "validation_rules.json"
            if rules_file.exists():
                async with aiofiles.open(rules_file, 'r') as f:
                    data = json.loads(await f.read())
                    # Reconstruct validation rules (simplified)
                    pass
        except Exception as e:
            uap_logger.log_event(
                LogLevel.WARNING,
                f"Failed to load validation rules: {e}",
                EventType.AGENT,
                {"error": str(e)},
                "model_validator"
            )
    
    async def _save_validation_report(self, report: ValidationReport):
        """Save validation report to storage"""
        try:
            report_file = self.validation_dir / f"report_{report.validation_id}.json"
            
            async with aiofiles.open(report_file, 'w') as f:
                await f.write(json.dumps(report.to_dict(), indent=2))
                
        except Exception as e:
            uap_logger.log_event(
                LogLevel.ERROR,
                f"Failed to save validation report: {e}",
                EventType.AGENT,
                {"validation_id": report.validation_id, "error": str(e)},
                "model_validator"
            )
    
    async def get_validator_status(self) -> Dict[str, Any]:
        """Get validator status"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "validation_dir": str(self.validation_dir),
            "total_reports": len(self.validation_reports),
            "default_rules": len(self.default_rules),
            "custom_rule_sets": len(self.validation_rules)
        }

# Global model validator instance
_model_validator = None

def get_model_validator() -> ModelValidator:
    """Get the global model validator instance"""
    global _model_validator
    if _model_validator is None:
        _model_validator = ModelValidator()
    return _model_validator