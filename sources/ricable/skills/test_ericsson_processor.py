#!/usr/bin/env python3
"""
Test script for Ericsson Feature Processor
Starts with 5 files to validate functionality
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ericsson_feature_processor import EricssonFeatureProcessor
from ericsson_skill_generator import EricssonSkillGenerator


def test_with_5_files():
    """Test processing with just 5 files"""
    print("🧪 Testing Ericsson Feature Processor with 5 files\n")

    # Create processor
    processor = EricssonFeatureProcessor(
        source_dir="elex_features_only",
        output_dir="test_output",
        batch_size=5
    )

    # Process only 5 files
    print("Phase 1: Processing 5 test files...")
    processor.process_all(limit=5)

    # Check results
    print(f"\nProcessed {len(processor.features)} features:")
    for feature_id, feature in processor.features.items():
        print(f"  - {feature['name']} (FAJ {feature_id})")
        if feature.get('cxc_code'):
            print(f"    CXC: {feature['cxc_code']}")
        print(f"    Parameters: {len(feature.get('parameters', []))}")
        print(f"    Counters: {len(feature.get('counters', []))}")

    # Generate skill from test data
    print("\nPhase 2: Generating test skill...")
    generator = EricssonSkillGenerator(
        data_dir="test_output/ericsson_data",
        output_dir="test_output"
    )
    generator.generate_skill()

    print("\n✅ Test complete!")
    print("Check test_output/ericsson_ran_features_skill_*_features.zip")
    print("\nTo process all files, run:")
    print("python3 ericsson_feature_processor.py --source elex_features_only")
    print("python3 ericsson_skill_generator.py --data-dir output/ericsson_data")


if __name__ == "__main__":
    test_with_5_files()