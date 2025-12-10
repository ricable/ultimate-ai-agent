#!/usr/bin/env python3
"""
Integration Test Runner for Ericsson RAN Features Processing System

This script provides convenient execution options for the comprehensive integration test suite:
- Quick tests (5 files only)
- Full test suite (all available tests)
- Performance benchmarks
- Individual test categories

Usage:
    python3 run_integration_tests.py [--quick] [--category <category>] [--verbose]
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_integration_test import IntegrationTestSuite


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run integration tests for Ericsson RAN Features processing system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run quick test with 5 files only
    python3 run_integration_tests.py --quick

    # Run full test suite
    python3 run_integration_tests.py

    # Run only performance tests
    python3 run_integration_tests.py --category performance

    # Run only quality tests
    python3 run_integration_tests.py --category quality

    # Run with verbose output
    python3 run_integration_tests.py --verbose
        """
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick tests only (5 files, basic validation)'
    )

    parser.add_argument(
        '--category', '-c',
        choices=['pipeline', 'performance', 'quality', 'edge-cases'],
        help='Run specific test category'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed test information'
    )

    parser.add_argument(
        '--list-tests',
        action='store_true',
        help='List all available tests and exit'
    )

    return parser.parse_args()


def list_available_tests():
    """List all available tests"""
    print("Available Integration Tests:")
    print("=" * 50)

    categories = {
        'Pipeline Tests': [
            'End-to-End Pipeline (5 files)',
            'End-to-End Pipeline (100 files)'
        ],
        'Performance Tests': [
            'Performance Targets',
            'Memory Usage',
            'Cache Performance'
        ],
        'Quality Tests': [
            'SKILL.md Quality',
            'Reference Structure',
            'Search Indices',
            'ZIP Integrity'
        ],
        'Edge Case Tests': [
            'Missing Files Handling',
            'Corrupted Files Handling',
            'Cache Corruption Recovery',
            'Memory Pressure Handling'
        ]
    }

    for category, tests in categories.items():
        print(f"\n{category}:")
        for test in tests:
            print(f"  • {test}")

    print(f"\nTotal: {sum(len(tests) for tests in categories.values())} tests")


def run_specific_category(test_suite: IntegrationTestSuite, category: str):
    """Run tests from a specific category"""
    category_tests = {
        'pipeline': [
            ('End-to-End Pipeline (5 files)', test_suite.test_end_to_end_5_files),
            ('End-to-End Pipeline (100 files)', test_suite.test_end_to_end_100_files)
        ],
        'performance': [
            ('Performance Targets', test_suite.test_performance_targets),
            ('Memory Usage', test_suite.test_memory_usage),
            ('Cache Performance', test_suite.test_cache_performance)
        ],
        'quality': [
            ('SKILL.md Quality', test_suite.test_skill_md_quality),
            ('Reference Structure', test_suite.test_reference_structure),
            ('Search Indices', test_suite.test_search_indices),
            ('ZIP Integrity', test_suite.test_zip_integrity)
        ],
        'edge-cases': [
            ('Missing Files Handling', test_suite.test_missing_files_handling),
            ('Corrupted Files Handling', test_suite.test_corrupted_files_handling),
            ('Cache Corruption Recovery', test_suite.test_cache_corruption_recovery),
            ('Memory Pressure Handling', test_suite.test_memory_pressure_handling)
        ]
    }

    if category not in category_tests:
        print(f"❌ Unknown category: {category}")
        print(f"Available categories: {', '.join(category_tests.keys())}")
        return False

    print(f"🚀 Running {category.upper()} test category...")
    print("=" * 50)

    tests_to_run = category_tests[category]
    for test_name, test_func in tests_to_run:
        test_suite.run_test(test_name, test_func)

    return True


def run_quick_tests(test_suite: IntegrationTestSuite):
    """Run quick validation tests"""
    print("🚀 Running QUICK integration tests...")
    print("=" * 50)
    print("Testing basic functionality with 5 files only")

    quick_tests = [
        ('End-to-End Pipeline (5 files)', test_suite.test_end_to_end_5_files),
        ('Performance Targets', test_suite.test_performance_targets),
        ('SKILL.md Quality', test_suite.test_skill_md_quality),
        ('ZIP Integrity', test_suite.test_zip_integrity)
    ]

    for test_name, test_func in quick_tests:
        test_suite.run_test(test_name, test_func)

    # Generate quick summary
    passed_tests = [r for r in test_suite.test_results if r.passed]
    failed_tests = [r for r in test_suite.test_results if not r.passed]

    print(f"\n📊 Quick Test Summary:")
    print(f"✅ Passed: {len(passed_tests)}/{len(test_suite.test_results)}")

    if failed_tests:
        print(f"❌ Failed: {len(failed_tests)}")
        for test in failed_tests:
            print(f"   • {test.test_name}: {test.error_message}")
    else:
        print("🎉 All quick tests passed!")


def main():
    """Main entry point"""
    args = parse_arguments()

    if args.list_tests:
        list_available_tests()
        return 0

    # Setup test suite
    try:
        test_suite = IntegrationTestSuite()
    except Exception as e:
        print(f"❌ Failed to initialize test suite: {e}")
        return 1

    # Run tests based on arguments
    try:
        if args.category:
            success = run_specific_category(test_suite, args.category)
            if not success:
                return 1
        elif args.quick:
            run_quick_tests(test_suite)
        else:
            # Run full test suite
            summary = test_suite.run_all_tests()

            # Check results
            failed_count = summary['test_execution']['failed_tests']
            if failed_count > 0:
                print(f"\n❌ {failed_count} test(s) failed.")
                return 1
            else:
                print(f"\n✅ All tests passed!")
                return 0

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 2
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())