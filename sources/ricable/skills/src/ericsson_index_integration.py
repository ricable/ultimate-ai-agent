#!/usr/bin/env python3
"""
Ericsson Feature Processor and Search Index Integration
Integrates the existing feature processor with the advanced search index system
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(str(Path(__file__).parent))

from ericsson_feature_processor import EricssonFeatureProcessor
from ericsson_search_index import EricssonSearchIndexBuilder


class EricssonFeatureSystem:
    """Integrated system combining feature processing and advanced search indexing"""

    def __init__(self, source_dir: str, output_dir: str = "output", batch_size: int = 50):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

        # Initialize components
        self.processor = EricssonFeatureProcessor(str(source_dir), str(output_dir), batch_size)
        self.index_builder: Optional[EricssonSearchIndexBuilder] = None

        # Paths
        self.features_dir = self.output_dir / "ericsson_data" / "features"
        self.search_index_file = self.output_dir / "ericsson_data" / "search_index.json"

    def process_all_features(self, limit: Optional[int] = None, force_reprocess: bool = False) -> bool:
        """Process all Ericsson feature documentation"""
        print("🚀 Processing Ericsson feature documentation...")

        try:
            # Process features using existing processor
            success = self.processor.process_all(limit)

            if not success and len(self.processor.features) == 0:
                print("❌ Feature processing failed")
                return False

            # Initialize search index builder
            self.index_builder = EricssonSearchIndexBuilder(
                str(self.features_dir),
                str(self.output_dir)
            )

            print(f"✅ Successfully processed {len(self.processor.features)} features")
            return True

        except Exception as e:
            print(f"❌ Error during processing: {e}")
            return False

    def build_search_indices(self, force_rebuild: bool = False) -> bool:
        """Build advanced search indices for processed features"""
        if not self.features_dir.exists():
            print("❌ No processed features found. Run process_all_features() first.")
            return False

        print("🔍 Building advanced search indices...")

        try:
            # Initialize search index builder if not already done
            if self.index_builder is None:
                self.index_builder = EricssonSearchIndexBuilder(
                    str(self.features_dir),
                    str(self.output_dir)
                )

            # Try to load existing indices
            if not force_rebuild and self.search_index_file.exists():
                if self.index_builder.load_indices():
                    print("✅ Loaded existing search indices")
                    return True

            # Load features and build indices
            self.index_builder.load_features()
            self.index_builder.build_all_indices()
            self.index_builder.save_indices()
            self.index_builder.export_index_summary()

            print("✅ Search indices built successfully")
            return True

        except Exception as e:
            print(f"❌ Error building search indices: {e}")
            return False

    def search_features(self, query: str, search_type: str = "universal", max_results: int = 20):
        """Search features using the advanced search system"""
        if self.index_builder is None:
            print("❌ Search indices not built. Run build_search_indices() first.")
            return []

        try:
            if search_type == "universal":
                return self.index_builder.universal_search(query, max_results)
            elif search_type == "parameters":
                return self.index_builder.search_parameters(query, max_results)
            elif search_type == "counters":
                return self.index_builder.search_counters(query, max_results)
            elif search_type == "names":
                return self.index_builder.search_names(query, max_results)
            elif search_type == "cxc":
                result = self.index_builder.search_cxc(query)
                return [result] if result else []
            elif search_type == "fuzzy":
                return self.index_builder.fuzzy_search(query, max_results)
            elif search_type == "dependencies":
                return self.index_builder.search_dependencies(query, max_results)
            else:
                print(f"⚠️  Unknown search type: {search_type}")
                return []

        except Exception as e:
            print(f"❌ Error during search: {e}")
            return []

    def get_feature_details(self, feature_id: str) -> Optional[Dict]:
        """Get detailed information about a specific feature"""
        feature = self.processor.features.get(feature_id)
        if feature:
            return {
                'id': feature.id,
                'name': feature.name,
                'cxc_code': feature.cxc_code,
                'description': feature.description,
                'summary': feature.summary,
                'value_package': feature.value_package,
                'node_type': feature.node_type,
                'parameters': feature.parameters,
                'counters': feature.counters,
                'events': feature.events,
                'dependencies': feature.dependencies,
                'activation_step': feature.activation_step,
                'engineering_guidelines': feature.engineering_guidelines
            }
        return None

    def get_system_statistics(self) -> Dict:
        """Get comprehensive statistics about the processed system"""
        stats = {
            'processing': self.processor.stats,
            'features': {
                'total_processed': len(self.processor.features),
                'error_files': len(self.processor.error_files)
            }
        }

        if self.index_builder:
            stats['search_index'] = self.index_builder.get_index_statistics()

        return stats

    def export_search_interface(self, output_file: Optional[str] = None) -> None:
        """Export a simple search interface script"""
        if output_file is None:
            output_file = self.output_dir / "search_interface.py"

        interface_code = '''#!/usr/bin/env python3
"""
Ericsson Features Search Interface
Simple command-line interface for searching processed Ericsson features
"""

import sys
import json
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from ericsson_index_integration import EricssonFeatureSystem


def main():
    """Interactive search interface"""
    # Initialize system
    system = EricssonFeatureSystem(
        source_dir="elex_features_only",
        output_dir="output"
    )

    # Load existing data
    if not system.build_search_indices():
        print("Failed to load search indices")
        return 1

    print("Ericsson Features Search Interface")
    print("Type 'help' for commands, 'quit' to exit")
    print("-" * 50)

    while True:
        try:
            query = input("\\nEnter search query: ").strip()

            if query.lower() == 'quit':
                break
            elif query.lower() == 'help':
                print_help()
                continue
            elif query.lower() == 'stats':
                stats = system.get_system_statistics()
                print(f"\\nSystem Statistics:")
                print(f"   Total features: {stats['features']['total_processed']:,}")
                if 'search_index' in stats:
                    si = stats['search_index']
                    print(f"   Parameter entries: {si['indices']['parameter_index']:,}")
                    print(f"   Counter entries: {si['indices']['counter_index']:,}")
                    print(f"   CXC codes: {si['indices']['cxc_index']:,}")
                continue
            elif query.lower().startswith('type:'):
                # Specific search type
                search_type = query[5:].strip()
                query = input(f"Enter {search_type} search query: ").strip()
                results = system.search_features(query, search_type)
            elif query.lower().startswith('cxc:'):
                # CXC code search
                cxc_code = query[4:].strip()
                results = system.search_features(cxc_code, "cxc")
            elif query.startswith('dep:'):
                # Dependency search
                feature_id = query[4:].strip()
                results = system.search_features(feature_id, "dependencies")
            else:
                # Universal search
                results = system.search_features(query, "universal")

            # Display results
            if results:
                print(f"\\nFound {len(results)} results:")
                for i, result in enumerate(results[:10], 1):
                    print(f"  {i}. {result.feature_name} ({result.feature_id})")
                    print(f"     [{result.match_type}] {result.match_context}")
                    if result.cxc_code:
                        print(f"     CXC: {result.cxc_code}")

                if len(results) > 10:
                    print(f"  ... and {len(results) - 10} more results")
            else:
                print("   No results found")

        except KeyboardInterrupt:
            print("\\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

    return 0


def print_help():
    """Print help information"""
    help_text = """
Search Commands:
  <query>                    - Universal search across all indices
  type:<search_type> <query> - Specific search type
    Available types: universal, parameters, counters, names, fuzzy, dependencies
  cxc:<code>                - Search by CXC code
  dep:<feature_id>          - Find dependencies for a feature
  stats                     - Show system statistics
  help                      - Show this help
  quit                      - Exit the search interface

Examples:
  load balancing           - Search for load balancing features
  type:parameters qos     - Search for QoS-related parameters
  cxc:CXC4011808          - Find feature with CXC code
  dep:FAJ 121 3096        - Find dependencies for Energy Saving feature
"""
    print(help_text)


if __name__ == "__main__":
    sys.exit(main())
'''

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(interface_code)

        print(f"Search interface exported to {output_path}")
        print(f"   Usage: python3 {output_path}")

    def run_complete_pipeline(self, limit: Optional[int] = None, force_reprocess: bool = False,
                            force_rebuild_indices: bool = False) -> bool:
        """Run the complete pipeline from processing to search-ready indices"""
        print("🚀 Running complete Ericsson feature processing pipeline...")

        # Step 1: Process features
        if not self.process_all_features(limit, force_reprocess):
            return False

        # Step 2: Build search indices
        if not self.build_search_indices(force_rebuild_indices):
            return False

        # Step 3: Export search interface
        self.export_search_interface()

        # Step 4: Print final statistics
        stats = self.get_system_statistics()
        print(f"\n✅ Pipeline completed successfully!")
        print(f"   Processed {stats['features']['total_processed']:,} features")
        if 'search_index' in stats:
            si = stats['search_index']
            print(f"   Built {len(si['indices'])} different search indices")
            print(f"   Search index build time: {si['build_time']:.2f}s")

        return True


def main():
    """Main function for the integrated system"""
    import argparse

    parser = argparse.ArgumentParser(description="Ericsson Feature Processing and Search System")
    parser.add_argument("--source", required=True, help="Source directory with Ericsson markdown files")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit number of files to process (for testing)")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocess all files")
    parser.add_argument("--force-rebuild-indices", action="store_true", help="Force rebuild search indices")
    parser.add_argument("--search", help="Test search with query after processing")
    parser.add_argument("--search-type", default="universal",
                       choices=["universal", "parameters", "counters", "names", "cxc", "fuzzy", "dependencies"],
                       help="Search type for testing")
    parser.add_argument("--export-interface", action="store_true", help="Export search interface script")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")

    args = parser.parse_args()

    # Initialize integrated system
    system = EricssonFeatureSystem(args.source, args.output, args.batch_size)

    # Run processing
    if not system.process_all_features(args.limit, args.force_reprocess):
        return 1

    # Build search indices
    if not system.build_search_indices(args.force_rebuild_indices):
        return 1

    # Export search interface if requested
    if args.export_interface:
        system.export_search_interface()

    # Test search if requested
    if args.search:
        print(f"\n🔍 Testing {args.search_type} search with query: '{args.search}'")
        results = system.search_features(args.search, args.search_type)

        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results[:10], 1):
                print(f"  {i}. {result.feature_name} ({result.feature_id})")
                print(f"     [{result.match_type}] {result.match_context}")
                if result.cxc_code:
                    print(f"     CXC: {result.cxc_code}")
        else:
            print("No results found")

    # Show statistics if requested
    if args.stats:
        stats = system.get_system_statistics()
        print(f"\n📊 System Statistics:")
        if 'processing' in stats:
            proc_time = stats['processing'].get('total_time', 0)
            print(f"   Processing time: {proc_time:.2f}s")
        print(f"   Features processed: {stats['features']['total_processed']:,}")
        print(f"   Error files: {stats['features']['error_files']}")

        if 'search_index' in stats:
            si = stats['search_index']
            print(f"   Index build time: {si['build_time']:.2f}s")
            print(f"   Parameter entries: {si['indices']['parameter_index']:,}")
            print(f"   Counter entries: {si['indices']['counter_index']:,}")
            print(f"   CXC codes: {si['indices']['cxc_index']:,}")
            print(f"   Name tokens: {si['indices']['name_tokens_index']:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())