#!/usr/bin/env python3
"""
Test script for Ericsson Search Index System
Validates index building, searching, and performance with sample data
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from ericsson_search_index import EricssonSearchIndexBuilder, SearchResult


def create_sample_features(output_dir: Path) -> Path:
    """Create sample feature data for testing"""
    print("📝 Creating sample feature data for testing...")

    features_dir = output_dir / "sample_features"
    features_dir.mkdir(parents=True, exist_ok=True)

    sample_features = [
        {
            "id": "FAJ 121 3094",
            "name": "Cell Load Balancing Feature",
            "cxc_code": "CXC4011808",
            "value_package": "Capacity Management",
            "value_package_id": "VP123",
            "access_type": "Licensed",
            "node_type": "BSC",
            "description": "Advanced cell load balancing for optimizing network capacity",
            "summary": "Distributes traffic load between cells to improve network performance",
            "parameters": [
                {"name": "loadBalancingThreshold", "mo_class": "Cell", "description": "Threshold for load balancing initiation"},
                {"name": "handoverMargin", "mo_class": "Cell", "description": "Margin for load balancing handovers"},
                {"name": "measurementPeriod", "mo_class": "BSC", "description": "Measurement period for load calculation"}
            ],
            "counters": [
                {"name": "lbAttemtps", "category": "Load Balancing", "description": "Number of load balancing attempts"},
                {"name": "lbSuccesses", "category": "Load Balancing", "description": "Number of successful load balancing operations"},
                {"name": "cellLoadFactor", "category": "Performance", "description": "Current cell load factor"}
            ],
            "events": [
                {"name": "loadBalancingStarted", "trigger": "High cell load detected", "parameters": ["cellId", "loadLevel"]},
                {"name": "loadBalancingCompleted", "trigger": "Load balancing operation finished", "parameters": ["success", "ueCount"]}
            ],
            "dependencies": {
                "prerequisites": ["FAJ 121 3095"],
                "related_features": ["FAJ 121 3096", "FAJ 121 3097"],
                "conflicts": []
            },
            "activation_step": "Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011808 MO instance.",
            "deactivation_step": "Set the FeatureState.featureState attribute to DEACTIVATED in the FeatureState=CXC4011808 MO instance.",
            "engineering_guidelines": "Configure appropriate thresholds based on network capacity requirements.",
            "source_file": "test_cell_load_balancing.md",
            "file_hash": "abc123",
            "processed_at": "2024-01-01T12:00:00Z"
        },
        {
            "id": "FAJ 121 3095",
            "name": "Inter-Frequency Handover Optimization",
            "cxc_code": "CXC4011809",
            "value_package": "Mobility Management",
            "value_package_id": "VP124",
            "access_type": "Licensed",
            "node_type": "RNC",
            "description": "Optimizes handover decisions between different frequency layers",
            "summary": "Improves handover performance for inter-frequency scenarios",
            "parameters": [
                {"name": "interFreqHoMargin", "mo_class": "Cell", "description": "Handover margin for inter-frequency handovers"},
                {"name": "hoHysteresis", "mo_class": "Cell", "description": "Hysteresis value for handover decisions"},
                {"name": "triggerTime", "mo_class": "RNC", "description": "Time threshold for handover triggering"}
            ],
            "counters": [
                {"name": "interFreqHoAttempts", "category": "Handover", "description": "Inter-frequency handover attempts"},
                {"name": "interFreqHoSuccesses", "category": "Handover", "description": "Successful inter-frequency handovers"},
                {"name": "hoFailureRate", "category": "Performance", "description": "Handover failure rate"}
            ],
            "events": [
                {"name": "interFreqHoTriggered", "trigger": "Inter-frequency handover conditions met", "parameters": ["sourceCell", "targetCell", "ueId"]},
                {"name": "interFreqHoCompleted", "trigger": "Handover completed", "parameters": ["success", "executionTime"]}
            ],
            "dependencies": {
                "prerequisites": [],
                "related_features": ["FAJ 121 3094", "FAJ 121 3098"],
                "conflicts": []
            },
            "activation_step": "Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011809 MO instance.",
            "deactivation_step": "Set the FeatureState.featureState attribute to DEACTIVATED in the FeatureState=CXC4011809 MO instance.",
            "engineering_guidelines": "Optimize hysteresis and margins based on propagation conditions.",
            "source_file": "test_inter_freq_handover.md",
            "file_hash": "def456",
            "processed_at": "2024-01-01T12:05:00Z"
        },
        {
            "id": "FAJ 121 3096",
            "name": "Energy Saving Mode",
            "cxc_code": "CXC4011810",
            "value_package": "Power Efficiency",
            "value_package_id": "VP125",
            "access_type": "Licensed",
            "node_type": "BSC",
            "description": "Automated energy saving through selective cell shutdown",
            "summary": "Reduces network power consumption during low traffic periods",
            "parameters": [
                {"name": "energySavingThreshold", "mo_class": "Cell", "description": "Traffic threshold for energy saving activation"},
                {"name": "shutdownTime", "mo_class": "BSC", "description": "Time delay before cell shutdown"},
                {"name": "wakeUpTime", "mo_class": "Cell", "description": "Time for cell to become active again"}
            ],
            "counters": [
                {"name": "energySavingActivations", "category": "Energy", "description": "Number of energy saving activations"},
                {"name": "energySaved", "category": "Energy", "description": "Total energy saved in kWh"},
                {"name": "cellUptime", "category": "Performance", "description": "Cell uptime percentage"}
            ],
            "events": [
                {"name": "energySavingActivated", "trigger": "Low traffic detected", "parameters": ["cellId", "expectedSaving"]},
                {"name": "cellWokeUp", "trigger": "Traffic increase detected", "parameters": ["cellId", "wakeUpTime"]}
            ],
            "dependencies": {
                "prerequisites": ["FAJ 121 3094"],
                "related_features": [],
                "conflicts": ["FAJ 121 3099"]
            },
            "activation_step": "Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011810 MO instance.",
            "deactivation_step": "Set the FeatureState.featureState attribute to DEACTIVATED in the FeatureState=CXC4011810 MO instance.",
            "engineering_guidelines": "Configure appropriate thresholds to maintain service quality.",
            "source_file": "test_energy_saving.md",
            "file_hash": "ghi789",
            "processed_at": "2024-01-01T12:10:00Z"
        },
        {
            "id": "FAJ 121 3097",
            "name": "QoS-Based Resource Allocation",
            "cxc_code": "CXC4011811",
            "value_package": "Quality Management",
            "value_package_id": "VP126",
            "access_type": "Licensed",
            "node_type": "RNC",
            "description": "Intelligent resource allocation based on service quality requirements",
            "summary": "Optimizes resource usage for different QoS classes",
            "parameters": [
                {"name": "qosClassPriority", "mo_class": "UE", "description": "Priority levels for different QoS classes"},
                {"name": "resourceAllocationFactor", "mo_class": "Cell", "description": "Resource allocation factor per QoS class"},
                {"name": "qosMonitoringInterval", "mo_class": "RNC", "description": "QoS monitoring and update interval"}
            ],
            "counters": [
                {"name": "qosViolations", "category": "Quality", "description": "Number of QoS violations"},
                {"name": "resourceUtilization", "category": "Performance", "description": "Resource utilization percentage"},
                {"name": "qosSatisfactionRate", "category": "Quality", "description": "QoS satisfaction rate"}
            ],
            "events": [
                {"name": "qosThresholdBreached", "trigger": "QoS parameters below threshold", "parameters": ["ueId", "qosClass", "violationType"]},
                {"name": "resourcesReallocated", "trigger": "Resource reallocation completed", "parameters": ["ueId", "newAllocation", "reason"]}
            ],
            "dependencies": {
                "prerequisites": [],
                "related_features": ["FAJ 121 3094", "FAJ 121 3095"],
                "conflicts": []
            },
            "activation_step": "Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011811 MO instance.",
            "deactivation_step": "Set the FeatureState.featureState attribute to DEACTIVATED in the FeatureState=CXC4011811 MO instance.",
            "engineering_guidelines": "Monitor QoS parameters regularly and adjust priorities as needed.",
            "source_file": "test_qos_allocation.md",
            "file_hash": "jkl012",
            "processed_at": "2024-01-01T12:15:00Z"
        },
        {
            "id": "FAJ 121 3098",
            "name": "Coverage Enhancement System",
            "cxc_code": "CXC4011812",
            "value_package": "Coverage Optimization",
            "value_package_id": "VP127",
            "access_type": "Licensed",
            "node_type": "BSC",
            "description": "Advanced coverage enhancement techniques for edge areas",
            "summary": "Improves network coverage in challenging geographical areas",
            "parameters": [
                {"name": "coverageMargin", "mo_class": "Cell", "description": "Coverage enhancement margin"},
                {"name": "antennaTiltOptimization", "mo_class": "Cell", "description": "Antenna tilt optimization parameters"},
                {"name": "powerBoostFactor", "mo_class": "Cell", "description": "Power boost factor for coverage areas"}
            ],
            "counters": [
                {"name": "coverageImprovement", "category": "Coverage", "description": "Coverage area improvement in square meters"},
                {"name": "edgeUserThroughput", "category": "Performance", "description": "Average throughput for edge users"},
                {"name": "signalQualityImprovement", "category": "Quality", "description": "Signal quality improvement"}
            ],
            "events": [
                {"name": "coverageOptimized", "trigger": "Coverage optimization completed", "parameters": ["cellId", "improvementLevel"]},
                {"name": "edgeUserDetected", "trigger": "Edge user condition detected", "parameters": ["ueId", "signalStrength"]}
            ],
            "dependencies": {
                "prerequisites": ["FAJ 121 3095"],
                "related_features": [],
                "conflicts": []
            },
            "activation_step": "Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=CXC4011812 MO instance.",
            "deactivation_step": "Set the FeatureState.featureState attribute to DEACTIVATED in the FeatureState=CXC4011812 MO instance.",
            "engineering_guidelines": "Balance coverage enhancement with interference management.",
            "source_file": "test_coverage_enhancement.md",
            "file_hash": "mno345",
            "processed_at": "2024-01-01T12:20:00Z"
        }
    ]

    # Write sample features to files
    for feature in sample_features:
        feature_file = features_dir / f"feature_{feature['id'].replace(' ', '_')}.json"
        feature_file.write_text(json.dumps(feature, indent=2))

    print(f"✅ Created {len(sample_features)} sample features in {features_dir}")
    return features_dir


def test_index_building(features_dir: Path, output_dir: Path):
    """Test the index building process"""
    print("\n🔧 Testing index building...")

    # Initialize index builder
    builder = EricssonSearchIndexBuilder(str(features_dir), str(output_dir))

    # Load features
    start_time = time.time()
    builder.load_features()
    load_time = time.time() - start_time

    print(f"   Loaded {len(builder.features)} features in {load_time:.3f}s")

    # Build indices
    start_time = time.time()
    builder.build_all_indices()
    build_time = time.time() - start_time

    print(f"   Built indices in {build_time:.3f}s")

    # Save indices
    builder.save_indices()

    # Export summary
    builder.export_index_summary()

    return builder


def test_parameter_search(builder: EricssonSearchIndexBuilder):
    """Test parameter-based searching"""
    print("\n🔍 Testing parameter search...")

    test_queries = [
        "loadBalancingThreshold",
        "handoverMargin",
        "qos",
        "energy"
    ]

    for query in test_queries:
        results = builder.search_parameters(query)
        print(f"   Query '{query}': {len(results)} results")
        for result in results[:3]:
            print(f"     - {result.feature_name} ({result.match_type}): {result.match_context}")


def test_counter_search(builder: EricssonSearchIndexBuilder):
    """Test counter-based searching"""
    print("\n📊 Testing counter search...")

    test_queries = [
        "lbSuccesses",
        "hoFailureRate",
        "energySaved",
        "qosViolations"
    ]

    for query in test_queries:
        results = builder.search_counters(query)
        print(f"   Query '{query}': {len(results)} results")
        for result in results[:3]:
            print(f"     - {result.feature_name} ({result.match_type}): {result.match_context}")


def test_cxc_search(builder: EricssonSearchIndexBuilder):
    """Test CXC code searching"""
    print("\n🏷️  Testing CXC code search...")

    test_codes = [
        "CXC4011808",
        "cxc4011809",  # Test case insensitivity
        "CXC4011811",
        "INVALID"
    ]

    for code in test_codes:
        result = builder.search_cxc(code)
        if result:
            print(f"   CXC '{code}': {result.feature_name}")
        else:
            print(f"   CXC '{code}': Not found")


def test_name_search(builder: EricssonSearchIndexBuilder):
    """Test name-based searching"""
    print("\n📝 Testing name search...")

    test_queries = [
        "load balancing",
        "energy saving",
        "coverage",
        "optimization",
        "invalid query"
    ]

    for query in test_queries:
        results = builder.search_names(query)
        print(f"   Query '{query}': {len(results)} results")
        for result in results[:3]:
            print(f"     - {result.feature_name} ({result.match_type}): {result.match_context}")


def test_dependency_search(builder: EricssonSearchIndexBuilder):
    """Test dependency relationship searching"""
    print("\n🔗 Testing dependency search...")

    # Test with feature that has dependencies
    test_feature = "FAJ 121 3096"  # Energy Saving Mode
    results = builder.search_dependencies(test_feature)

    print(f"   Dependencies for '{test_feature}': {len(results)} results")
    for result in results:
        print(f"     - {result.feature_name}: {result.match_context}")


def test_fuzzy_search(builder: EricssonSearchIndexBuilder):
    """Test fuzzy searching"""
    print("\n🔍 Testing fuzzy search...")

    test_queries = [
        "balanc",  # Should match "balancing"
        "energi",  # Should match "energy"
        "covrage",  # Should match "coverage"
        "optmization"  # Should match "optimization"
    ]

    for query in test_queries:
        results = builder.fuzzy_search(query)
        print(f"   Fuzzy '{query}': {len(results)} results")
        for result in results[:3]:
            print(f"     - {result.feature_name} ({result.match_type}): {result.match_context}")


def test_universal_search(builder: EricssonSearchIndexBuilder):
    """Test universal searching across all indices"""
    print("\n🌐 Testing universal search...")

    test_queries = [
        "load",
        "energy",
        "handover",
        "coverage",
        "optimization"
    ]

    for query in test_queries:
        results = builder.universal_search(query)
        print(f"   Universal '{query}': {len(results)} results")
        for result in results[:5]:
            print(f"     - {result.feature_name} [{result.match_type}]: {result.match_context}")


def test_performance(builder: EricssonSearchIndexBuilder):
    """Test search performance"""
    print("\n⚡ Testing search performance...")

    test_queries = ["load", "energy", "handover", "coverage", "optimization", "qos", "threshold"]
    num_iterations = 100

    total_time = 0
    for query in test_queries:
        start_time = time.time()
        for _ in range(num_iterations):
            builder.universal_search(query)
        query_time = time.time() - start_time
        total_time += query_time

        avg_time = (query_time / num_iterations) * 1000  # Convert to milliseconds
        print(f"   '{query}': {avg_time:.2f}ms per search (avg over {num_iterations} iterations)")

    overall_avg = (total_time / (len(test_queries) * num_iterations)) * 1000
    print(f"   Overall average: {overall_avg:.2f}ms per search")


def test_index_persistence(builder: EricssonSearchIndexBuilder, output_dir: Path):
    """Test index saving and loading"""
    print("\n💾 Testing index persistence...")

    # Save indices
    index_file = output_dir / "ericsson_data" / "test_search_index.json"
    builder.save_indices(str(index_file))
    print(f"   Indices saved to {index_file}")

    # Create new builder and load indices
    new_builder = EricssonSearchIndexBuilder(str(builder.features_dir), str(output_dir))
    load_success = new_builder.load_indices(str(index_file))

    if load_success:
        print("   ✅ Indices loaded successfully")

        # Test that loaded indices work
        test_results = new_builder.universal_search("load")
        print(f"   Loaded indices search test: {len(test_results)} results")
    else:
        print("   ❌ Failed to load indices")


def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("🚀 Starting comprehensive search index test...")

    # Setup test environment
    test_output = Path("test_output")
    test_output.mkdir(exist_ok=True)

    try:
        # Create sample data
        features_dir = create_sample_features(test_output)

        # Test index building
        builder = test_index_building(features_dir, test_output)

        # Test different search methods
        test_parameter_search(builder)
        test_counter_search(builder)
        test_cxc_search(builder)
        test_name_search(builder)
        test_dependency_search(builder)
        test_fuzzy_search(builder)
        test_universal_search(builder)

        # Test performance
        test_performance(builder)

        # Test persistence
        test_index_persistence(builder, test_output)

        # Print final statistics
        stats = builder.get_index_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Features indexed: {stats['total_features']}")
        print(f"   Build time: {stats['build_time']:.3f}s")
        print(f"   Parameter entries: {stats['indices']['parameter_index']}")
        print(f"   Counter entries: {stats['indices']['counter_index']}")
        print(f"   CXC codes: {stats['indices']['cxc_index']}")
        print(f"   Name tokens: {stats['indices']['name_tokens_index']}")

        print("\n✅ All tests completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)