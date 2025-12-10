"""
SAFLA Integration Test Runner
============================

This script runs the complete SAFLA integration test suite and provides
comprehensive reporting on system integration readiness.

Test Categories:
1. System Integration Tests - Core component integration
2. End-to-End Workflow Tests - Complete SAFLA workflows
3. Performance Integration Tests - System performance under load
4. Safety Integration Tests - Safety constraint enforcement
5. Deployment Readiness Tests - Production deployment validation

Usage:
    python run_integration_tests.py [options]

Options:
    --environment ENV    Target environment (dev, staging, prod, ha)
    --category CATEGORY  Run specific test category
    --verbose           Enable verbose output
    --report-format     Output format (console, json, html)
    --output-file       Output file for reports
    --parallel          Run tests in parallel (where safe)
    --stress-test       Include stress testing
    --quick             Run quick test suite only
"""

import asyncio
import argparse
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
import subprocess
import pytest
from datetime import datetime

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCategory(Enum):
    """Test categories for integration testing."""
    SYSTEM_INTEGRATION = "system_integration"
    END_TO_END_WORKFLOWS = "end_to_end_workflows"
    PERFORMANCE_INTEGRATION = "performance_integration"
    SAFETY_INTEGRATION = "safety_integration"
    DEPLOYMENT_READINESS = "deployment_readiness"
    ALL = "all"


class TestEnvironment(Enum):
    """Test environments."""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"
    HIGH_AVAILABILITY = "ha"


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    category: TestCategory
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSuiteResult:
    """Complete test suite result."""
    environment: TestEnvironment
    start_time: datetime
    end_time: datetime
    total_duration: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    test_results: List[TestResult] = field(default_factory=list)
    overall_status: str = "unknown"
    integration_readiness_score: float = 0.0
    deployment_approved: bool = False
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class IntegrationTestRunner:
    """Main integration test runner."""
    
    def __init__(self, environment: TestEnvironment, verbose: bool = False):
        self.environment = environment
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.test_results: List[TestResult] = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('integration_test_runner')
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def run_test_category(self, category: TestCategory, parallel: bool = False, stress_test: bool = False) -> List[TestResult]:
        """Run tests for a specific category."""
        self.logger.info(f"Running {category.value} tests for {self.environment.value} environment")
        
        category_results = []
        
        if category == TestCategory.SYSTEM_INTEGRATION:
            category_results.extend(await self._run_system_integration_tests(parallel))
        
        elif category == TestCategory.END_TO_END_WORKFLOWS:
            category_results.extend(await self._run_end_to_end_tests(parallel))
        
        elif category == TestCategory.PERFORMANCE_INTEGRATION:
            category_results.extend(await self._run_performance_tests(parallel, stress_test))
        
        elif category == TestCategory.SAFETY_INTEGRATION:
            category_results.extend(await self._run_safety_tests(parallel))
        
        elif category == TestCategory.DEPLOYMENT_READINESS:
            category_results.extend(await self._run_deployment_readiness_tests(parallel))
        
        elif category == TestCategory.ALL:
            # Run all categories sequentially
            for cat in [TestCategory.SYSTEM_INTEGRATION, TestCategory.END_TO_END_WORKFLOWS,
                       TestCategory.PERFORMANCE_INTEGRATION, TestCategory.SAFETY_INTEGRATION,
                       TestCategory.DEPLOYMENT_READINESS]:
                category_results.extend(await self.run_test_category(cat, parallel, stress_test))
        
        self.test_results.extend(category_results)
        return category_results
    
    async def _run_system_integration_tests(self, parallel: bool = False) -> List[TestResult]:
        """Run system integration tests."""
        test_file = "tests/integration/test_system_integration.py"
        
        test_cases = [
            "test_delta_evaluation_memory_integration",
            "test_meta_cognitive_mcp_integration",
            "test_safety_validator_system_halt_integration",
            "test_memory_optimizer_hybrid_memory_integration",
            "test_complete_self_improvement_cycle",
            "test_memory_driven_decision_making",
            "test_mcp_orchestration_with_meta_cognitive_oversight",
            "test_event_propagation_across_components",
            "test_data_flow_consistency",
            "test_concurrent_component_operations",
            "test_memory_pressure_handling",
            "test_adaptive_optimization_under_load",
            "test_safety_constraint_enforcement_under_load",
            "test_emergency_stop_during_operations",
            "test_error_handling_and_recovery",
            "test_development_configuration",
            "test_production_configuration",
            "test_stress_testing_configuration",
            "test_error_injection_resilience"
        ]
        
        return await self._execute_pytest_tests(test_file, test_cases, TestCategory.SYSTEM_INTEGRATION, parallel)
    
    async def _run_end_to_end_tests(self, parallel: bool = False) -> List[TestResult]:
        """Run end-to-end workflow tests."""
        test_file = "tests/integration/test_end_to_end_workflows.py"
        
        test_cases = [
            "test_complete_self_improvement_cycle_workflow",
            "test_memory_driven_adaptive_workflow",
            "test_multi_agent_coordination_workflow",
            "test_adaptive_performance_optimization_workflow",
            "test_safety_constrained_operation_workflow",
            "test_error_recovery_workflow",
            "test_learning_and_adaptation_workflow",
            "test_cross_component_communication_workflow",
            "test_system_state_management_workflow",
            "test_performance_monitoring_workflow"
        ]
        
        return await self._execute_pytest_tests(test_file, test_cases, TestCategory.END_TO_END_WORKFLOWS, parallel)
    
    async def _run_performance_tests(self, parallel: bool = False, stress_test: bool = False) -> List[TestResult]:
        """Run performance integration tests."""
        test_file = "tests/integration/test_performance_integration.py"
        
        test_cases = [
            "test_mixed_concurrent_operations",
            "test_memory_pressure_performance",
            "test_throughput_scaling",
            "test_cpu_utilization_optimization",
            "test_memory_utilization_optimization",
            "test_performance_degradation_detection_and_recovery"
        ]
        
        if stress_test:
            # Add stress test specific cases
            test_cases.extend([
                "test_extreme_load_conditions",
                "test_sustained_high_throughput",
                "test_memory_exhaustion_recovery"
            ])
        
        return await self._execute_pytest_tests(test_file, test_cases, TestCategory.PERFORMANCE_INTEGRATION, parallel)
    
    async def _run_safety_tests(self, parallel: bool = False) -> List[TestResult]:
        """Run safety integration tests."""
        test_file = "tests/integration/test_safety_integration.py"
        
        test_cases = [
            "test_memory_safety_constraints_across_components",
            "test_delta_evaluation_safety_constraints",
            "test_mcp_orchestration_safety_constraints",
            "test_system_wide_emergency_stop",
            "test_component_specific_emergency_stops",
            "test_constraint_propagation_across_components",
            "test_hierarchical_constraint_enforcement"
        ]
        
        return await self._execute_pytest_tests(test_file, test_cases, TestCategory.SAFETY_INTEGRATION, parallel)
    
    async def _run_deployment_readiness_tests(self, parallel: bool = False) -> List[TestResult]:
        """Run deployment readiness tests."""
        test_file = "tests/integration/test_deployment_readiness.py"
        
        test_cases = [
            "test_development_environment_configuration",
            "test_production_environment_configuration",
            "test_high_availability_configuration",
            "test_production_load_stress_test",
            "test_burst_load_handling",
            "test_component_failure_resilience",
            "test_network_failure_resilience",
            "test_comprehensive_deployment_readiness"
        ]
        
        return await self._execute_pytest_tests(test_file, test_cases, TestCategory.DEPLOYMENT_READINESS, parallel)
    
    async def _execute_pytest_tests(self, test_file: str, test_cases: List[str], 
                                   category: TestCategory, parallel: bool = False) -> List[TestResult]:
        """Execute pytest tests and collect results."""
        results = []
        
        for test_case in test_cases:
            start_time = time.time()
            
            try:
                # Set environment variable for tests to access
                os.environ['SAFLA_TEST_ENVIRONMENT'] = self.environment.value
                
                # Construct pytest command
                cmd = [
                    sys.executable, "-m", "pytest",
                    f"{test_file}::{test_case}",
                    "-v", "--tb=short"
                ]
                
                if self.verbose:
                    cmd.append("-s")
                
                # Execute test
                self.logger.debug(f"Executing: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                duration = time.time() - start_time
                
                # Parse result
                if result.returncode == 0:
                    status = "passed"
                    error_message = None
                elif "SKIPPED" in result.stdout:
                    status = "skipped"
                    error_message = None
                else:
                    status = "failed"
                    error_message = result.stderr or result.stdout
                
                # Extract warnings
                warnings = []
                if "warning" in result.stdout.lower():
                    warnings.append("Test produced warnings - check output")
                
                # Extract metrics (if available in output)
                metrics = self._extract_test_metrics(result.stdout)
                
                test_result = TestResult(
                    test_name=test_case,
                    category=category,
                    status=status,
                    duration=duration,
                    error_message=error_message,
                    warnings=warnings,
                    metrics=metrics
                )
                
                results.append(test_result)
                
                self.logger.info(f"Test {test_case}: {status} ({duration:.2f}s)")
                
                if status == "failed":
                    self.logger.error(f"Test failure details: {error_message}")
            
            except subprocess.TimeoutExpired:
                duration = time.time() - start_time
                test_result = TestResult(
                    test_name=test_case,
                    category=category,
                    status="error",
                    duration=duration,
                    error_message="Test timed out after 300 seconds"
                )
                results.append(test_result)
                self.logger.error(f"Test {test_case} timed out")
            
            except Exception as e:
                duration = time.time() - start_time
                test_result = TestResult(
                    test_name=test_case,
                    category=category,
                    status="error",
                    duration=duration,
                    error_message=str(e)
                )
                results.append(test_result)
                self.logger.error(f"Test {test_case} error: {e}")
        
        return results
    
    def _extract_test_metrics(self, test_output: str) -> Dict[str, Any]:
        """Extract metrics from test output."""
        metrics = {}
        
        # Look for common metric patterns in test output
        lines = test_output.split('\n')
        for line in lines:
            if 'ops/s' in line.lower():
                # Extract operations per second
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'ops/s' in part.lower() and i > 0:
                            metrics['operations_per_second'] = float(parts[i-1])
                            break
                except:
                    pass
            
            if 'latency' in line.lower() and 'ms' in line.lower():
                # Extract latency metrics
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 'ms' in part.lower() and i > 0:
                            metrics['latency_ms'] = float(parts[i-1].replace('ms', ''))
                            break
                except:
                    pass
            
            if 'memory usage' in line.lower():
                # Extract memory usage
                try:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if '%' in part and i > 0:
                            metrics['memory_usage_percent'] = float(part.replace('%', ''))
                            break
                except:
                    pass
        
        return metrics
    
    def generate_test_suite_result(self) -> TestSuiteResult:
        """Generate comprehensive test suite result."""
        if not self.test_results:
            return TestSuiteResult(
                environment=self.environment,
                start_time=datetime.now(),
                end_time=datetime.now(),
                total_duration=0.0,
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=0,
                overall_status="no_tests_run"
            )
        
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == "passed")
        failed_tests = sum(1 for r in self.test_results if r.status == "failed")
        skipped_tests = sum(1 for r in self.test_results if r.status == "skipped")
        error_tests = sum(1 for r in self.test_results if r.status == "error")
        
        total_duration = sum(r.duration for r in self.test_results)
        
        # Calculate integration readiness score
        integration_readiness_score = self._calculate_integration_readiness_score()
        
        # Determine overall status
        if failed_tests == 0 and error_tests == 0:
            overall_status = "passed"
        elif failed_tests > 0 or error_tests > 0:
            if (failed_tests + error_tests) / total_tests > 0.2:  # More than 20% failures
                overall_status = "failed"
            else:
                overall_status = "degraded"
        else:
            overall_status = "unknown"
        
        # Determine deployment approval
        deployment_approved = (
            integration_readiness_score >= 0.8 and
            overall_status in ["passed", "degraded"] and
            failed_tests == 0 and
            error_tests == 0
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics()
        
        return TestSuiteResult(
            environment=self.environment,
            start_time=datetime.now(),  # Would be set at start in real implementation
            end_time=datetime.now(),
            total_duration=total_duration,
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            test_results=self.test_results,
            overall_status=overall_status,
            integration_readiness_score=integration_readiness_score,
            deployment_approved=deployment_approved,
            summary_metrics=summary_metrics,
            recommendations=recommendations
        )
    
    def _calculate_integration_readiness_score(self) -> float:
        """Calculate overall integration readiness score."""
        if not self.test_results:
            return 0.0
        
        # Weight different test categories
        category_weights = {
            TestCategory.SYSTEM_INTEGRATION: 0.3,
            TestCategory.END_TO_END_WORKFLOWS: 0.25,
            TestCategory.PERFORMANCE_INTEGRATION: 0.2,
            TestCategory.SAFETY_INTEGRATION: 0.15,
            TestCategory.DEPLOYMENT_READINESS: 0.1
        }
        
        category_scores = {}
        
        for category in category_weights.keys():
            category_tests = [r for r in self.test_results if r.category == category]
            if category_tests:
                passed = sum(1 for r in category_tests if r.status == "passed")
                total = len(category_tests)
                category_scores[category] = passed / total
            else:
                category_scores[category] = 0.0
        
        # Calculate weighted average
        weighted_score = sum(
            category_scores[category] * weight
            for category, weight in category_weights.items()
        )
        
        return weighted_score
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        # Check for failed tests
        failed_tests = [r for r in self.test_results if r.status == "failed"]
        if failed_tests:
            recommendations.append(f"Address {len(failed_tests)} failed tests before deployment")
        
        # Check for error tests
        error_tests = [r for r in self.test_results if r.status == "error"]
        if error_tests:
            recommendations.append(f"Investigate {len(error_tests)} tests that encountered errors")
        
        # Check performance metrics
        performance_tests = [r for r in self.test_results if r.category == TestCategory.PERFORMANCE_INTEGRATION]
        slow_tests = [r for r in performance_tests if r.duration > 30.0]
        if slow_tests:
            recommendations.append(f"Optimize {len(slow_tests)} slow-running performance tests")
        
        # Check safety tests
        safety_tests = [r for r in self.test_results if r.category == TestCategory.SAFETY_INTEGRATION]
        failed_safety = [r for r in safety_tests if r.status in ["failed", "error"]]
        if failed_safety:
            recommendations.append("Critical: Address safety test failures before any deployment")
        
        # Environment-specific recommendations
        if self.environment == TestEnvironment.PRODUCTION:
            if self._calculate_integration_readiness_score() < 0.95:
                recommendations.append("Production environment requires >95% test success rate")
        
        return recommendations
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary metrics across all tests."""
        if not self.test_results:
            return {}
        
        # Performance metrics
        performance_tests = [r for r in self.test_results if r.category == TestCategory.PERFORMANCE_INTEGRATION]
        avg_ops_per_second = 0.0
        avg_latency_ms = 0.0
        
        if performance_tests:
            ops_metrics = [r.metrics.get('operations_per_second', 0) for r in performance_tests if r.metrics.get('operations_per_second')]
            latency_metrics = [r.metrics.get('latency_ms', 0) for r in performance_tests if r.metrics.get('latency_ms')]
            
            if ops_metrics:
                avg_ops_per_second = sum(ops_metrics) / len(ops_metrics)
            if latency_metrics:
                avg_latency_ms = sum(latency_metrics) / len(latency_metrics)
        
        return {
            'average_test_duration': sum(r.duration for r in self.test_results) / len(self.test_results),
            'longest_test_duration': max(r.duration for r in self.test_results),
            'average_operations_per_second': avg_ops_per_second,
            'average_latency_ms': avg_latency_ms,
            'total_warnings': sum(len(r.warnings) for r in self.test_results),
            'success_rate': sum(1 for r in self.test_results if r.status == "passed") / len(self.test_results)
        }


    async def run_tests(self, categories: List[TestCategory] = None, parallel: bool = False, stress_test: bool = False) -> TestSuiteResult:
        """Run integration tests for specified categories."""
        if categories is None:
            categories = [TestCategory.ALL]
        
        self.logger.info(f"Starting integration test run for environment: {self.environment.value}")
        self.test_results = []  # Reset results
        
        start_time = time.time()
        
        try:
            for category in categories:
                await self.run_test_category(category, parallel, stress_test)
            
            # Generate test suite result
            result = self.generate_test_suite_result()
            
            end_time = time.time()
            result.total_duration = end_time - start_time
            result.start_time = datetime.fromtimestamp(start_time)
            result.end_time = datetime.fromtimestamp(end_time)
            
            self.logger.info(f"Integration test run completed in {result.total_duration:.2f} seconds")
            self.logger.info(f"Results: {result.passed_tests} passed, {result.failed_tests} failed, {result.skipped_tests} skipped")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Integration test run failed: {str(e)}")
            raise
    
    def generate_report(self, result: TestSuiteResult, format_type: str = "console") -> str:
        """Generate test report in specified format."""
        if format_type == "console":
            return TestReportGenerator.generate_console_report(result)
        elif format_type == "json":
            return TestReportGenerator.generate_json_report(result)
        elif format_type == "html":
            return TestReportGenerator.generate_html_report(result)
        else:
            raise ValueError(f"Unsupported report format: {format_type}")
    
    def assess_deployment_readiness(self, result: TestSuiteResult) -> Dict[str, Any]:
        """Assess deployment readiness based on test results."""
        assessment = {
            'deployment_approved': result.deployment_approved,
            'integration_readiness_score': result.integration_readiness_score,
            'overall_status': result.overall_status,
            'environment': result.environment.value,
            'test_summary': {
                'total_tests': result.total_tests,
                'passed_tests': result.passed_tests,
                'failed_tests': result.failed_tests,
                'error_tests': result.error_tests,
                'success_rate': result.passed_tests / result.total_tests if result.total_tests > 0 else 0.0
            },
            'critical_issues': [],
            'recommendations': result.recommendations,
            'deployment_blockers': []
        }
        
        # Identify critical issues and deployment blockers
        for test_result in result.test_results:
            if test_result.status in ["failed", "error"]:
                issue = f"{test_result.test_name}: {test_result.error_message or 'Test failed'}"
                
                # Categorize as critical if it's a safety or security test
                if any(keyword in test_result.test_name.lower() for keyword in ['safety', 'security', 'emergency']):
                    assessment['critical_issues'].append(issue)
                    assessment['deployment_blockers'].append(issue)
                
                # Performance issues are warnings, not blockers
                elif 'performance' in test_result.test_name.lower():
                    if test_result.status == "failed":
                        assessment['deployment_blockers'].append(issue)
        
        # Final deployment decision logic
        has_critical_failures = len(assessment['critical_issues']) > 0
        has_deployment_blockers = len(assessment['deployment_blockers']) > 0
        readiness_threshold_met = result.integration_readiness_score >= 0.8
        
        assessment['deployment_approved'] = (
            not has_critical_failures and
            not has_deployment_blockers and
            readiness_threshold_met and
            result.overall_status in ["passed", "degraded"]
        )
        
        # Add deployment readiness summary
        if assessment['deployment_approved']:
            assessment['deployment_status'] = "APPROVED"
            assessment['deployment_message'] = f"System ready for deployment to {result.environment.value} environment"
        else:
            assessment['deployment_status'] = "BLOCKED"
            reasons = []
            if has_critical_failures:
                reasons.append("critical test failures")
            if has_deployment_blockers:
                reasons.append("deployment blocking issues")
            if not readiness_threshold_met:
                reasons.append(f"readiness score below threshold ({result.integration_readiness_score:.1%} < 80%)")
            
            assessment['deployment_message'] = f"Deployment blocked due to: {', '.join(reasons)}"
        
        return assessment



class TestReportGenerator:
    """Generate test reports in various formats."""
    
    @staticmethod
    def generate_console_report(result: TestSuiteResult) -> str:
        """Generate console-friendly test report."""
        report = []
        report.append("=" * 80)
        report.append("SAFLA INTEGRATION TEST RESULTS")
        report.append("=" * 80)
        report.append(f"Environment: {result.environment.value}")
        report.append(f"Total Duration: {result.total_duration:.2f} seconds")
        report.append(f"Overall Status: {result.overall_status.upper()}")
        report.append(f"Integration Readiness Score: {result.integration_readiness_score:.1%}")
        report.append(f"Deployment Approved: {'YES' if result.deployment_approved else 'NO'}")
        report.append("")
        
        # Test summary
        report.append("TEST SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Tests: {result.total_tests}")
        report.append(f"Passed: {result.passed_tests}")
        report.append(f"Failed: {result.failed_tests}")
        report.append(f"Skipped: {result.skipped_tests}")
        report.append(f"Errors: {result.error_tests}")
        report.append("")
        
        # Category breakdown
        report.append("CATEGORY BREAKDOWN")
        report.append("-" * 40)
        categories = {}
        for test_result in result.test_results:
            cat = test_result.category
            if cat not in categories:
                categories[cat] = {'passed': 0, 'failed': 0, 'skipped': 0, 'error': 0}
            categories[cat][test_result.status] += 1
        
        for category, counts in categories.items():
            total = sum(counts.values())
            passed = counts['passed']
            report.append(f"{category.value}: {passed}/{total} passed ({passed/total:.1%})")
        
        report.append("")
        
        # Failed tests
        failed_tests = [r for r in result.test_results if r.status in ['failed', 'error']]
        if failed_tests:
            report.append("FAILED TESTS")
            report.append("-" * 40)
            for test in failed_tests:
                report.append(f"❌ {test.test_name} ({test.category.value})")
                if test.error_message:
                    report.append(f"   Error: {test.error_message[:100]}...")
            report.append("")
        
        # Performance metrics
        if result.summary_metrics:
            report.append("PERFORMANCE METRICS")
            report.append("-" * 40)
            metrics = result.summary_metrics
            if metrics.get('average_operations_per_second'):
                report.append(f"Average Throughput: {metrics['average_operations_per_second']:.1f} ops/s")
            if metrics.get('average_latency_ms'):
                report.append(f"Average Latency: {metrics['average_latency_ms']:.1f} ms")
            report.append(f"Average Test Duration: {metrics['average_test_duration']:.2f} s")
            report.append(f"Success Rate: {metrics['success_rate']:.1%}")
            report.append("")
        
        # Recommendations
        if result.recommendations:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    @staticmethod
    def generate_json_report(result: TestSuiteResult) -> str:
        """Generate JSON test report."""
        # Convert dataclass to dict for JSON serialization
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                result_dict = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, Enum):
                        result_dict[key] = value.value
                    elif isinstance(value, datetime):
                        result_dict[key] = value.isoformat()
                    elif isinstance(value, list):
                        result_dict[key] = [convert_to_dict(item) for item in value]
                    elif hasattr(value, '__dict__'):
                        result_dict[key] = convert_to_dict(value)
                    else:
                        result_dict[key] = value
                return result_dict
            else:
                return obj
        
        return json.dumps(convert_to_dict(result), indent=2)
    
    @staticmethod
    def generate_html_report(result: TestSuiteResult) -> str:
        """Generate HTML test report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SAFLA Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .category {{ margin: 10px 0; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .skipped {{ color: orange; }}
        .error {{ color: darkred; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .recommendations {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>SAFLA Integration Test Report</h1>
        <p><strong>Environment:</strong> {result.environment.value}</p>
        <p><strong>Overall Status:</strong> <span class="{result.overall_status}">{result.overall_status.upper()}</span></p>
        <p><strong>Integration Readiness Score:</strong> {result.integration_readiness_score:.1%}</p>
        <p><strong>Deployment Approved:</strong> {'YES' if result.deployment_approved else 'NO'}</p>
    </div>
    
    <div class="summary">
        <h2>Test Summary</h2>
        <p>Total Tests: {result.total_tests}</p>
        <p>Passed: <span class="passed">{result.passed_tests}</span></p>
        <p>Failed: <span class="failed">{result.failed_tests}</span></p>
        <p>Skipped: <span class="skipped">{result.skipped_tests}</span></p>
        <p>Errors: <span class="error">{result.error_tests}</span></p>
    </div>
    
    <div class="details">
        <h2>Test Details</h2>
        <table>
            <tr>
                <th>Test Name</th>
                <th>Category</th>
                <th>Status</th>
                <th>Duration (s)</th>
            </tr>
"""
        
        for test in result.test_results:
            html += f"""
            <tr>
                <td>{test.test_name}</td>
                <td>{test.category.value}</td>
                <td class="{test.status}">{test.status.upper()}</td>
                <td>{test.duration:.2f}</td>
            </tr>
"""
        
        html += """
        </table>
    </div>
"""
        
        if result.recommendations:
            html += """
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
"""
            for rec in result.recommendations:
                html += f"            <li>{rec}</li>\n"
            
            html += """
        </ul>
    </div>
"""
        
        html += """
</body>
</html>
"""
        return html


async def main():
    """Main entry point for integration test runner."""
    parser = argparse.ArgumentParser(description="SAFLA Integration Test Runner")
    parser.add_argument("--environment", choices=["dev", "staging", "prod", "ha"], 
                       default="dev", help="Target environment")
    parser.add_argument("--category", choices=["system_integration", "end_to_end_workflows", 
                       "performance_integration", "safety_integration", "deployment_readiness", "all"],
                       default="all", help="Test category to run")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--report-format", choices=["console", "json", "html"], 
                       default="console", help="Report format")
    parser.add_argument("--output-file", help="Output file for report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--stress-test", action="store_true", help="Include stress testing")
    parser.add_argument("--quick", action="store_true", help="Run quick test suite only")
    
    args = parser.parse_args()
    
    # Convert arguments to enums
    environment = TestEnvironment(args.environment)
    category = TestCategory(args.category)
    
    # Create test runner
    runner = IntegrationTestRunner(environment, args.verbose)
    
    print(f"Starting SAFLA integration tests for {environment.value} environment...")
    print(f"Test category: {category.value}")
    print(f"Parallel execution: {args.parallel}")
    print(f"Stress testing: {args.stress_test}")
    print("-" * 80)
    
    # Run tests
    start_time = time.time()
    
    try:
        await runner.run_test_category(category, args.parallel, args.stress_test)
        
        # Generate results
        suite_result = runner.generate_test_suite_result()
        
        # Generate report
        if args.report_format == "console":
            report = TestReportGenerator.generate_console_report(suite_result)
        elif args.report_format == "json":
            report = TestReportGenerator.generate_json_report(suite_result)
        elif args.report_format == "html":
            report = TestReportGenerator.generate_html_report(suite_result)
        
        # Output report
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output_file}")
        else:
            print(report)
        
        # Exit with appropriate code
        if suite_result.overall_status == "passed":
            sys.exit(0)
        elif suite_result.overall_status == "degraded":
            sys.exit(1)
        else:
            sys.exit(2)
    
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())