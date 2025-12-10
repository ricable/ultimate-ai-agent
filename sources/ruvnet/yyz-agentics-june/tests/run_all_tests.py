#!/usr/bin/env python3
"""
Main test runner for the neural network test suite.
Runs all unit tests, integration tests, and benchmarks.
"""

import unittest
import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.unit.test_layers import TestLayers, TestLayerNumericalStability
from tests.unit.test_activations import TestActivationFunctions
from tests.unit.test_loss_functions import TestLossFunctions
from tests.unit.test_optimizers import TestOptimizers, TestConvergenceProperties
from tests.unit.test_initializers import TestWeightInitializers, TestBiasInitializers
from tests.integration.test_neural_network import (
    TestNeuralNetworkIntegration, TestTrainingDynamics,
    TestMemoryAndPerformance, TestNumericalStability,
    TestSaveLoadFunctionality, TestGradientValidation, TestEdgeCases
)
from tests.benchmarks.test_performance import (
    TestLayerPerformance, TestOptimizationPerformance,
    TestMemoryUsage, TestScalingBehavior
)


class CustomTestResult(unittest.TestResult):
    """Custom test result class to capture detailed information."""
    
    def __init__(self, stream=None, descriptions=None, verbosity=None):
        super().__init__(stream, descriptions, verbosity)
        self.test_details = []
        
    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()
        
    def stopTest(self, test):
        super().stopTest(test)
        test_time = time.time() - self.test_start_time
        
        test_info = {
            'name': str(test),
            'class': test.__class__.__name__,
            'method': test._testMethodName,
            'time': test_time,
            'status': 'passed'
        }
        
        if self.errors and self.errors[-1][0] == test:
            test_info['status'] = 'error'
            test_info['error'] = self.errors[-1][1]
        elif self.failures and self.failures[-1][0] == test:
            test_info['status'] = 'failed'
            test_info['failure'] = self.failures[-1][1]
        elif self.skipped and self.skipped[-1][0] == test:
            test_info['status'] = 'skipped'
            test_info['reason'] = self.skipped[-1][1]
            
        self.test_details.append(test_info)


def run_test_suite(test_type: str = 'all', verbose: int = 2) -> Dict:
    """
    Run the test suite.
    
    Args:
        test_type: Type of tests to run ('unit', 'integration', 'benchmark', 'all')
        verbose: Verbosity level
        
    Returns:
        Dictionary with test results
    """
    # Define test categories
    unit_tests = [
        TestLayers,
        TestLayerNumericalStability,
        TestActivationFunctions,
        TestLossFunctions,
        TestOptimizers,
        TestConvergenceProperties,
        TestWeightInitializers,
        TestBiasInitializers,
    ]
    
    integration_tests = [
        TestNeuralNetworkIntegration,
        TestTrainingDynamics,
        TestMemoryAndPerformance,
        TestNumericalStability,
        TestSaveLoadFunctionality,
        TestGradientValidation,
        TestEdgeCases,
    ]
    
    benchmark_tests = [
        TestLayerPerformance,
        TestOptimizationPerformance,
        TestMemoryUsage,
        TestScalingBehavior,
    ]
    
    # Select tests to run
    if test_type == 'unit':
        test_classes = unit_tests
    elif test_type == 'integration':
        test_classes = integration_tests
    elif test_type == 'benchmark':
        test_classes = benchmark_tests
    else:  # 'all'
        test_classes = unit_tests + integration_tests + benchmark_tests
        
    # Create test suite
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    # Run tests with custom result
    runner = unittest.TextTestRunner(verbosity=verbose, resultclass=CustomTestResult)
    result = runner.run(suite)
    
    # Generate report
    report = generate_test_report(result, test_type)
    
    return report


def generate_test_report(result: CustomTestResult, test_type: str) -> Dict:
    """Generate comprehensive test report."""
    report = {
        'timestamp': datetime.now().isoformat(),
        'test_type': test_type,
        'summary': {
            'total': result.testsRun,
            'passed': result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            'failed': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success_rate': 0.0,
            'total_time': sum(t['time'] for t in result.test_details),
        },
        'details': {
            'by_class': {},
            'failed_tests': [],
            'slow_tests': [],
        },
        'test_results': result.test_details,
    }
    
    # Calculate success rate
    if report['summary']['total'] > 0:
        report['summary']['success_rate'] = (
            report['summary']['passed'] / report['summary']['total']
        )
        
    # Group by test class
    for test in result.test_details:
        class_name = test['class']
        if class_name not in report['details']['by_class']:
            report['details']['by_class'][class_name] = {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'errors': 0,
                'skipped': 0,
                'total_time': 0.0,
            }
            
        class_stats = report['details']['by_class'][class_name]
        class_stats['total'] += 1
        class_stats['total_time'] += test['time']
        
        if test['status'] == 'passed':
            class_stats['passed'] += 1
        elif test['status'] == 'failed':
            class_stats['failed'] += 1
            report['details']['failed_tests'].append({
                'name': test['name'],
                'failure': test.get('failure', 'Unknown failure')
            })
        elif test['status'] == 'error':
            class_stats['errors'] += 1
        elif test['status'] == 'skipped':
            class_stats['skipped'] += 1
            
        # Track slow tests (> 1 second)
        if test['time'] > 1.0:
            report['details']['slow_tests'].append({
                'name': test['name'],
                'time': test['time']
            })
            
    # Sort slow tests by time
    report['details']['slow_tests'].sort(key=lambda x: x['time'], reverse=True)
    
    return report


def save_test_report(report: Dict, filename: str = None):
    """Save test report to file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{timestamp}.json"
        
    os.makedirs('test_reports', exist_ok=True)
    filepath = os.path.join('test_reports', filename)
    
    with open(filepath, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\nTest report saved to: {filepath}")
    
    return filepath


def print_test_summary(report: Dict):
    """Print test summary to console."""
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    summary = report['summary']
    print(f"Total Tests: {summary['total']}")
    print(f"Passed: {summary['passed']} ({summary['passed']/summary['total']*100:.1f}%)")
    print(f"Failed: {summary['failed']}")
    print(f"Errors: {summary['errors']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Total Time: {summary['total_time']:.2f} seconds")
    print(f"Success Rate: {summary['success_rate']*100:.1f}%")
    
    if report['details']['failed_tests']:
        print("\n" + "-"*70)
        print("FAILED TESTS:")
        for test in report['details']['failed_tests'][:10]:  # Show first 10
            print(f"  - {test['name']}")
            
    if report['details']['slow_tests']:
        print("\n" + "-"*70)
        print("SLOWEST TESTS:")
        for test in report['details']['slow_tests'][:5]:  # Show top 5
            print(f"  - {test['name']}: {test['time']:.2f}s")
            
    print("\n" + "="*70)


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Neural Network Test Suite Runner')
    parser.add_argument('--type', choices=['unit', 'integration', 'benchmark', 'all'],
                      default='all', help='Type of tests to run')
    parser.add_argument('--verbose', type=int, default=2,
                      help='Verbosity level (0-2)')
    parser.add_argument('--save-report', action='store_true',
                      help='Save test report to file')
    parser.add_argument('--no-summary', action='store_true',
                      help='Skip printing test summary')
    
    args = parser.parse_args()
    
    print(f"Running {args.type} tests...")
    
    # Run tests
    report = run_test_suite(test_type=args.type, verbose=args.verbose)
    
    # Print summary
    if not args.no_summary:
        print_test_summary(report)
        
    # Save report
    if args.save_report:
        save_test_report(report)
        
    # Exit with appropriate code
    if report['summary']['failed'] > 0 or report['summary']['errors'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()