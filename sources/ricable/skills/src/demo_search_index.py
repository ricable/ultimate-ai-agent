#!/usr/bin/env python3
"""
Ericsson Search Index Demonstration
Shows how to use the advanced search index system with Ericsson feature data
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from ericsson_index_integration import EricssonFeatureSystem


def demo_search_capabilities():
    """Demonstrate various search capabilities"""
    print("🚀 Ericsson Feature Search Index Demo")
    print("=" * 50)

    # Check if we have processed data
    features_dir = Path("output/ericsson_data/features")
    if not features_dir.exists():
        print("❌ No processed features found.")
        print("Please run the feature processor first:")
        print("  python3 src/ericsson_feature_processor.py --source elex_features_only --limit 20")
        return False

    # Initialize the system
    system = EricssonFeatureSystem("elex_features_only", "output")

    # Build search indices
    print("🔍 Building search indices...")
    if not system.build_search_indices():
        print("❌ Failed to build search indices")
        return False

    print("✅ Search indices ready!")

    # Demo different search types
    demo_queries = [
        ("Parameter Search", "parameters", "threshold"),
        ("Parameter Search", "parameters", "handover"),
        ("Counter Search", "counters", "load"),
        ("Counter Search", "counters", "qos"),
        ("Name Search", "names", "load"),
        ("Name Search", "names", "energy"),
        ("Universal Search", "universal", "capacity"),
        ("Universal Search", "universal", "optimization"),
        ("Universal Search", "universal", "mobility"),
        ("Universal Search", "universal", "coverage"),
        ("Fuzzy Search", "fuzzy", "optimiz"),
        ("Fuzzy Search", "fuzzy", "handovr"),
    ]

    print("\n🔍 Search Demo Results:")
    print("-" * 50)

    for search_name, search_type, query in demo_queries:
        print(f"\n📋 {search_name}: '{query}'")
        results = system.search_features(query, search_type)

        if results:
            for i, result in enumerate(results[:5], 1):
                print(f"  {i}. {result.feature_name} ({result.feature_id})")
                print(f"     [{result.match_type}] {result.match_context}")
                if result.cxc_code:
                    print(f"     CXC: {result.cxc_code}")
        else:
            print("  No results found")

    # Demo CXC code search
    print(f"\n📋 CXC Code Search Demo:")
    cxc_codes = ["CXC4011808", "CXC4011809"]  # Common Ericsson codes

    for cxc_code in cxc_codes:
        print(f"  Searching for CXC: {cxc_code}")
        results = system.search_features(cxc_code, "cxc")
        if results:
            result = results[0]
            print(f"    Found: {result.feature_name} ({result.feature_id})")
        else:
            print("    Not found")

    # Show system statistics
    stats = system.get_system_statistics()
    print(f"\n📊 System Statistics:")
    print(f"   Total features: {stats['features']['total_processed']:,}")
    if 'search_index' in stats:
        si = stats['search_index']
        print(f"   Build time: {si['build_time']:.2f}s")
        print(f"   Parameter entries: {si['indices']['parameter_index']:,}")
        print(f"   Counter entries: {si['indices']['counter_index']:,}")
        print(f"   CXC codes: {si['indices']['cxc_index']:,}")
        print(f"   Name tokens: {si['indices']['name_tokens_index']:,}")

    print(f"\n🎯 Performance Demo:")
    # Test search performance
    test_queries = ["load", "energy", "handover", "coverage", "optimization"]
    import time

    total_time = 0
    for query in test_queries:
        start_time = time.time()
        for _ in range(50):  # 50 iterations
            system.search_features(query, "universal")
        query_time = time.time() - start_time
        total_time += query_time
        avg_time = (query_time / 50) * 1000
        print(f"   '{query}': {avg_time:.2f}ms per search")

    overall_avg = (total_time / (len(test_queries) * 50)) * 1000
    print(f"   Overall average: {overall_avg:.2f}ms per search")

    return True


def demo_integration_with_processor():
    """Demo integration with the feature processor"""
    print("\n🔗 Integration Demo: Processing + Search")
    print("-" * 50)

    # Initialize system
    system = EricssonFeatureSystem("elex_features_only", "output", batch_size=10)

    # Process a small sample of features
    print("📝 Processing sample features (10 files)...")
    success = system.process_all_features(limit=10)

    if not success:
        print("❌ Feature processing failed")
        return False

    print(f"✅ Processed {len(system.processor.features)} features")

    # Build search indices
    print("🔍 Building search indices...")
    if not system.build_search_indices():
        print("❌ Failed to build search indices")
        return False

    # Test search on the processed data
    print("🔍 Testing search on processed data...")
    test_queries = ["parameter", "counter", "threshold", "feature"]

    for query in test_queries:
        results = system.search_features(query, "universal")
        print(f"  '{query}': {len(results)} results")

    # Export search interface
    system.export_search_interface()

    print("✅ Integration demo completed successfully!")
    return True


def interactive_search_demo():
    """Interactive search demonstration"""
    print("\n🎮 Interactive Search Demo")
    print("Type 'quit' to exit, 'help' for commands")
    print("-" * 30)

    # Initialize system
    system = EricssonFeatureSystem("elex_features_only", "output")

    if not system.build_search_indices():
        print("❌ Failed to build search indices")
        return

    while True:
        try:
            query = input("\n🔍 Enter search query: ").strip()

            if query.lower() == 'quit':
                break
            elif query.lower() == 'help':
                print("Available commands:")
                print("  <query>        - Universal search")
                print("  param:<query>  - Parameter search")
                print("  counter:<q>    - Counter search")
                print("  name:<query>   - Name search")
                print("  cxc:<code>     - CXC code search")
                print("  fuzzy:<query>  - Fuzzy search")
                print("  quit           - Exit")
                continue
            elif query.lower().startswith('param:'):
                search_query = query[6:].strip()
                results = system.search_features(search_query, "parameters")
            elif query.lower().startswith('counter:'):
                search_query = query[8:].strip()
                results = system.search_features(search_query, "counters")
            elif query.lower().startswith('name:'):
                search_query = query[5:].strip()
                results = system.search_features(search_query, "names")
            elif query.lower().startswith('cxc:'):
                search_query = query[4:].strip()
                results = system.search_features(search_query, "cxc")
            elif query.lower().startswith('fuzzy:'):
                search_query = query[6:].strip()
                results = system.search_features(search_query, "fuzzy")
            else:
                results = system.search_features(query, "universal")

            # Display results
            if results:
                print(f"\n📋 Found {len(results)} results:")
                for i, result in enumerate(results[:10], 1):
                    print(f"  {i}. {result.feature_name} ({result.feature_id})")
                    print(f"     [{result.match_type}] {result.match_context}")
                    if result.cxc_code:
                        print(f"     CXC: {result.cxc_code}")

                    # Show feature details if requested
                    if len(results) <= 3:  # Only show details for small result sets
                        details = system.get_feature_details(result.feature_id)
                        if details:
                            desc = details.get('description', '')[:100]
                            if desc:
                                print(f"     📝 {desc}...")

                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more results")
            else:
                print("   No results found")

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main demonstration function"""
    import argparse

    parser = argparse.ArgumentParser(description="Ericsson Search Index Demonstration")
    parser.add_argument("--mode", choices=["demo", "integration", "interactive"],
                       default="demo", help="Demo mode to run")
    parser.add_argument("--source", default="elex_features_only", help="Source directory")

    args = parser.parse_args()

    if args.mode == "demo":
        demo_search_capabilities()
    elif args.mode == "integration":
        demo_integration_with_processor()
    elif args.mode == "interactive":
        interactive_search_demo()

    return 0


if __name__ == "__main__":
    sys.exit(main())