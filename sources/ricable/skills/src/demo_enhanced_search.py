#!/usr/bin/env python3
"""
Enhanced Search System Demo
Demonstrates the capabilities of the improved Ericsson search index system
"""

import os
import sys
import json
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from ericsson_feature_processor import EricssonFeatureProcessor
from ericsson_search_index import EricssonSearchIndexBuilder


def demo_search_capabilities(search_interface):
    """Demonstrate various search capabilities"""
    print("\n🔍 Enhanced Search System Demo")
    print("=" * 50)

    # Demo queries
    demo_queries = [
        ("load balancing", "Exact feature name search"),
        ("mimo sleep", "Partial name matching"),
        ("loadBalancingThreshold", "Parameter name search"),
        ("lbAttempts", "Counter name search"),
        ("CXC4011808", "CXC code search"),
        ("handover", "Partial match across all fields"),
        ("interfernce", "Fuzzy search with typo"),
        ("optimzation", "Fuzzy search with typo"),
        ("performanc", "Partial token matching"),
        ("energy saving", "Category-based search")
    ]

    for query, description in demo_queries:
        print(f"\n🎯 {description}")
        print(f"   Query: '{query}'")

        start_time = time.time()
        results = search_interface.universal_search(query, max_results=5)
        search_time = (time.time() - start_time) * 1000

        print(f"   Results: {len(results)} found in {search_time:.2f}ms")

        for i, result in enumerate(results, 1):
            print(f"     {i}. {result.feature_name}")
            print(f"        FAJ: {result.feature_id}")
            print(f"        Match: {result.match_type} ({result.relevance_score:.2f})")
            print(f"        Context: {result.match_context}")
            if result.cxc_code:
                print(f"        CXC: {result.cxc_code}")


def demo_advanced_features(search_interface):
    """Demonstrate advanced search features"""
    print("\n🚀 Advanced Search Features Demo")
    print("=" * 50)

    # 1. Dependency search
    print("\n🔗 Dependency Search Demo")
    feature_id = "FAJ 121 3094"
    dependencies = search_interface.search_dependencies(feature_id)
    print(f"   Dependencies for {feature_id}: {len(dependencies)} related features")
    for dep in dependencies:
        print(f"     - {dep.feature_name} ({dep.match_context})")

    # 2. Category-based search
    print("\n📂 Category-based Search Demo")
    if hasattr(search_interface.search_index, 'category_index'):
        categories = list(search_interface.search_index.category_index.keys())
        print(f"   Available categories: {categories}")
        if 'performance' in categories:
            performance_features = search_interface.search_index.category_index['performance']
            print(f"   Performance category: {len(performance_features)} features")

    # 3. Value package search
    print("\n💼 Value Package Search Demo")
    if hasattr(search_interface.search_index, 'value_package_index'):
        packages = list(search_interface.search_index.value_package_index.keys())
        print(f"   Available value packages: {packages}")

    # 4. Fuzzy search with typos
    print("\n✨ Fuzzy Search with Typos Demo")
    typo_examples = [
        ("loadbalancng", "Missing 'i'"),
        ("mimo slep", "Swapped letters"),
        ("interfernce", "Missing 'e'"),
        ("optimzation", "Missing 'i'"),
        ("handoover", "Extra 'o'")
    ]

    for typo, description in typo_examples:
        results = search_interface.fuzzy_search(typo, max_results=3)
        print(f"   '{typo}' ({description}): {len(results)} results")
        for result in results:
            print(f"     -> {result.feature_name} ({result.relevance_score:.2f})")


def demo_performance_analysis(search_interface):
    """Demonstrate performance analysis"""
    print("\n⚡ Performance Analysis Demo")
    print("=" * 50)

    # Test search performance
    test_queries = [
        "load", "mimo", "interference", "handover", "coverage",
        "parameter", "counter", "optimization", "performance", "energy"
    ]

    print("   Search Performance Test:")
    total_time = 0
    for query in test_queries:
        start_time = time.time()
        results = search_interface.universal_search(query, max_results=10)
        search_time = (time.time() - start_time) * 1000
        total_time += search_time
        print(f"     '{query}': {search_time:.2f}ms ({len(results)} results)")

    avg_time = total_time / len(test_queries)
    print(f"   Average search time: {avg_time:.2f}ms")

    # Show index statistics
    stats = search_interface.get_index_statistics()
    print(f"\n   Index Statistics:")
    print(f"     Total features: {stats['total_features']:,}")
    print(f"     Parameter entries: {stats['indices']['parameter_index']:,}")
    print(f"     Counter entries: {stats['indices']['counter_index']:,}")
    print(f"     CXC codes: {stats['indices']['cxc_index']:,}")
    print(f"     Name tokens: {stats['indices']['name_tokens_index']:,}")
    print(f"     Fuzzy entries: {stats['indices']['fuzzy_index']:,}")
    print(f"     Build time: {stats['build_time']:.2f}s")


def run_enhanced_search_demo():
    """Run the complete enhanced search demo"""
    print("🎉 Enhanced Ericsson Search System Demo")
    print("This demo showcases the improved search capabilities")
    print("including fuzzy matching, partial matching, and performance optimizations.")

    # Check for existing processed data
    output_dir = Path("output")
    if not output_dir.exists():
        print("❌ No output directory found. Please run the feature processor first:")
        print("   python3 ericsson_feature_processor.py --source elex_features_only")
        return 1

    # Try to load existing search indices
    features_dir = output_dir / "ericsson_data" / "features"
    if not features_dir.exists():
        print("❌ No processed features found. Please run the feature processor first.")
        return 1

    try:
        # Initialize search interface
        search_builder = EricssonSearchIndexBuilder(str(features_dir), str(output_dir))

        # Load or build indices
        if not search_builder.load_indices():
            print("🔨 No existing indices found, building new ones...")
            search_builder.load_features()
            search_builder.build_all_indices()
            search_builder.save_indices()
        else:
            print("✅ Loaded existing search indices")

        # Run demos
        demo_search_capabilities(search_builder)
        demo_advanced_features(search_builder)
        demo_performance_analysis(search_builder)

        print("\n✅ Enhanced Search System Demo Completed!")
        print("\nKey Features Demonstrated:")
        print("  🎯 Multi-index searching (names, parameters, counters, CXC codes)")
        print("  🔤 Enhanced tokenization for partial matching")
        print("  ✨ Fuzzy search with typo tolerance")
        print("  🔗 Dependency and relationship searching")
        print("  📂 Category and value package filtering")
        print("  ⚡ High-performance search with millisecond response times")
        print("  💾 Optimized storage with compression")
        print("  🔄 Incremental updates for efficiency")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(run_enhanced_search_demo())