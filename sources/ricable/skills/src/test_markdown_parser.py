#!/usr/bin/env python3
"""
Test script for Ericsson Markdown Parser

This script tests the markdown parsing functionality with sample files from the
elex_features_only directory to validate FAJ number extraction, feature name parsing,
table parsing, and other extraction capabilities.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from ericsson_markdown_parser import (
    EricssonMarkdownParser,
    ParsedFeature,
    parse_ericsson_markdown,
    MarkdownParseError
)


def test_single_file_parsing(file_path: str) -> Dict:
    """
    Test parsing a single markdown file and return results.

    Args:
        file_path: Path to the markdown file to test

    Returns:
        Dictionary with test results
    """
    results = {
        'file': file_path,
        'success': False,
        'error': None,
        'feature': None,
        'validation': {}
    }

    try:
        # Parse the file
        feature = parse_ericsson_markdown(file_path)
        results['feature'] = feature
        results['success'] = True

        # Validate extracted data
        validation = {
            'has_faj_id': bool(feature.id),
            'has_name': bool(feature.name),
            'has_description': bool(feature.description),
            'has_parameters': len(feature.parameters) > 0,
            'has_counters': len(feature.counters) > 0,
            'has_dependencies': any(len(feature.dependencies[key]) > 0 for key in feature.dependencies),
            'has_activation': bool(feature.activation_step),
            'parameter_count': len(feature.parameters),
            'counter_count': len(feature.counters),
            'dependency_count': sum(len(feature.dependencies[key]) for key in feature.dependencies)
        }

        results['validation'] = validation

        print(f"✅ Successfully parsed: {Path(file_path).name}")
        print(f"   FAJ ID: {feature.id}")
        print(f"   Name: {feature.name[:50]}{'...' if len(feature.name) > 50 else ''}")
        print(f"   Parameters: {len(feature.parameters)}")
        print(f"   Counters: {len(feature.counters)}")
        print(f"   Dependencies: {validation['dependency_count']}")
        print(f"   CXC Code: {feature.cxc_code or 'Not found'}")

    except MarkdownParseError as e:
        results['error'] = str(e)
        print(f"❌ Failed to parse {Path(file_path).name}: {e}")

    except Exception as e:
        results['error'] = f"Unexpected error: {e}"
        print(f"💥 Unexpected error parsing {Path(file_path).name}: {e}")

    return results


def test_faj_extraction_patterns() -> Dict:
    """
    Test FAJ ID extraction with various patterns.

    Returns:
        Dictionary with test results
    """
    print("\n" + "="*60)
    print("🧪 TESTING FAJ ID EXTRACTION PATTERNS")
    print("="*60)

    test_cases = [
        ("FAJ 121 4219", "121 4219"),
        ("FAJ1214219", "121 4219"),
        ("Feature Identity | FAJ 121 3094", "121 3094"),
        ("Value Package Identity: FAJ 801 0427", "801 0427"),
        ("No FAJ here", None),
        ("FAJ 1234567", "123 4567"),  # Should normalize spacing
    ]

    parser = EricssonMarkdownParser()
    results = {'passed': 0, 'failed': 0, 'cases': []}

    for input_text, expected in test_cases:
        # Create a simple HTML with the test text
        html = f"<p>{input_text}</p>"
        soup = parser.__class__.__new__(parser.__class__)
        soup = parser.__class__.__bases__[0].__new__(parser.__class__.__bases__[0])

        # Use the private method for testing
        faj_id = None
        for pattern in parser.faj_patterns:
            import re
            match = re.search(pattern, input_text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    faj_id = f"{match.group(1)} {match.group(2)}"
                else:
                    faj_id = match.group(1)
                    if re.match(r'\d{6}', faj_id):
                        faj_id = f"{faj_id[:3]} {faj_id[3:]}"
                break

        success = faj_id == expected
        if success:
            results['passed'] += 1
            print(f"✅ FAJ extraction: '{input_text}' -> '{faj_id}'")
        else:
            results['failed'] += 1
            print(f"❌ FAJ extraction: '{input_text}' -> '{faj_id}' (expected '{expected}')")

        results['cases'].append({
            'input': input_text,
            'expected': expected,
            'actual': faj_id,
            'success': success
        })

    return results


def test_parameter_extraction() -> Dict:
    """
    Test parameter extraction from various sources.

    Returns:
        Dictionary with test results
    """
    print("\n" + "="*60)
    print("🧪 TESTING PARAMETER EXTRACTION")
    print("="*60)

    # Create test HTML with parameter table
    test_html = """
    <table>
        <tr><th>Parameter</th><th>Type</th><th>Description</th></tr>
        <tr><td>EUtranCellFDD.lbThreshold</td><td>Integer</td><td>Load balancing threshold</td></tr>
        <tr><td>FeatureState.featureState</td><td>Enumeration</td><td>Feature activation state</td></tr>
    </table>
    <p>The feature uses the Mobility.ueEvaluationTimer parameter for configuration.</p>
    """

    soup = BeautifulSoup(test_html, 'html.parser')
    parser = EricssonMarkdownParser()

    # Test table extraction
    parameters = parser._extract_parameters_from_tables(soup)
    print(f"✅ Extracted {len(parameters)} parameters from table")

    # Test text extraction
    parameters_text = parser._extract_parameters_from_text(soup)
    print(f"✅ Extracted {len(parameters_text)} parameters from text")

    # Test MO class extraction
    test_params = ["EUtranCellFDD.lbThreshold", "FeatureState.featureState", "standaloneParam"]
    for param in test_params:
        mo_class = parser._extract_mo_class(param)
        print(f"✅ MO class for '{param}': '{mo_class}'")

    return {
        'table_parameters': len(parameters),
        'text_parameters': len(parameters_text),
        'mo_classes': {param: parser._extract_mo_class(param) for param in test_params}
    }


def test_counter_extraction() -> Dict:
    """
    Test counter extraction and categorization.

    Returns:
        Dictionary with test results
    """
    print("\n" + "="*60)
    print("🧪 TESTING COUNTER EXTRACTION")
    print("="*60)

    test_text = """
    The feature introduces several new counters: pmThroughputHigh, pmMimoThroughput,
    pmHandoverSuccess, pmSleepModeEnergy, and pmCellLoad. These PM counters help
    monitor the feature performance.
    """

    parser = EricssonMarkdownParser()
    counters = []

    # Find all counter mentions
    for pattern in parser.counter_patterns:
        import re
        matches = re.findall(pattern, test_text, re.IGNORECASE)
        for match in matches:
            counter = {
                'name': f'pm{match.upper()}',
                'category': parser._guess_counter_category(match)
            }
            counters.append(counter)

    print(f"✅ Extracted {len(counters)} counters:")
    for counter in counters:
        print(f"   - {counter['name']} (Category: {counter['category']})")

    return {
        'counters_found': len(counters),
        'counters': counters
    }


def run_comprehensive_test():
    """
    Run comprehensive tests on sample markdown files.
    """
    print("🚀 Starting comprehensive markdown parser tests")
    print("=" * 80)

    # Test with sample files
    source_dir = Path("/Users/cedric/dev/skills/elex_features_only")
    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return

    # Find some sample markdown files
    md_files = list(source_dir.rglob("*.md"))[:5]  # Test first 5 files

    if not md_files:
        print("❌ No markdown files found in source directory")
        return

    print(f"📁 Found {len(md_files)} sample files for testing")

    # Test pattern extraction functions
    faj_results = test_faj_extraction_patterns()
    param_results = test_parameter_extraction()
    counter_results = test_counter_extraction()

    # Test actual file parsing
    print("\n" + "="*60)
    print("🧪 TESTING ACTUAL FILE PARSING")
    print("="*60)

    file_results = []
    successful_parses = 0
    total_parameters = 0
    total_counters = 0
    total_dependencies = 0

    for file_path in md_files:
        result = test_single_file_parsing(str(file_path))
        file_results.append(result)

        if result['success']:
            successful_parses += 1
            validation = result['validation']
            total_parameters += validation['parameter_count']
            total_counters += validation['counter_count']
            total_dependencies += validation['dependency_count']

    # Print summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Files tested: {len(md_files)}")
    print(f"Successfully parsed: {successful_parses}")
    print(f"Failed parses: {len(md_files) - successful_parses}")
    print(f"Total parameters extracted: {total_parameters}")
    print(f"Total counters extracted: {total_counters}")
    print(f"Total dependencies extracted: {total_dependencies}")

    print(f"\nFAJ extraction tests: {faj_results['passed']}/{len(faj_results['cases'])} passed")
    print(f"Parameter extraction: {param_results['table_parameters']} from tables, {param_results['text_parameters']} from text")
    print(f"Counter extraction: {counter_results['counters_found']} counters found")

    # Save detailed results
    results_file = Path("/Users/cedric/dev/skills/test_markdown_parser_results.json")
    results_data = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'file_tests': [
            {
                'file': r['file'],
                'success': r['success'],
                'error': r['error'],
                'validation': r['validation'],
                'feature_data': {
                    'id': r['feature'].id,
                    'name': r['feature'].name,
                    'cxc_code': r['feature'].cxc_code,
                    'parameter_count': len(r['feature'].parameters),
                    'counter_count': len(r['feature'].counters)
                } if r['feature'] else None
            } for r in file_results
        ],
        'pattern_tests': {
            'faj_extraction': faj_results,
            'parameter_extraction': param_results,
            'counter_extraction': counter_results
        },
        'summary': {
            'files_tested': len(md_files),
            'successful_parses': successful_parses,
            'total_parameters': total_parameters,
            'total_counters': total_counters,
            'total_dependencies': total_dependencies
        }
    }

    results_file.write_text(json.dumps(results_data, indent=2))
    print(f"\n💾 Detailed results saved to: {results_file}")

    # Test edge cases
    print("\n" + "="*60)
    print("🧪 TESTING EDGE CASES")
    print("="*60)

    # Test with missing FAJ ID
    try:
        missing_faj_html = "<h1>Test Feature</h1><p>No FAJ ID here</p>"
        soup = BeautifulSoup(missing_faj_html, 'html.parser')
        parser = EricssonMarkdownParser()
        feature = parser._extract_feature_identity(soup)
        if feature is None:
            print("✅ Correctly handled missing FAJ ID")
        else:
            print("❌ Should have returned None for missing FAJ ID")
    except Exception as e:
        print(f"❌ Error testing missing FAJ ID: {e}")

    # Test with malformed table
    try:
        malformed_table_html = """
        <table>
            <tr><th>Name</th></tr>
            <tr><td>Only one column</td></tr>
        </table>
        """
        soup = BeautifulSoup(malformed_table_html, 'html.parser')
        parser = EricssonMarkdownParser()
        params = parser._extract_parameters_from_tables(soup)
        print(f"✅ Handled malformed table gracefully (extracted {len(params)} parameters)")
    except Exception as e:
        print(f"❌ Error handling malformed table: {e}")

    print("\n✅ Comprehensive testing completed!")


if __name__ == "__main__":
    from bs4 import BeautifulSoup
    run_comprehensive_test()