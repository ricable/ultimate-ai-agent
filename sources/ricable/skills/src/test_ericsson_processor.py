#!/usr/bin/env python3
"""
Test script for Ericsson Feature Processor
Validates the entire processing pipeline with 5 files
Based on final-plan.md lines 203-212
"""

import sys
import json
import time
import shutil
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ericsson_feature_processor import EricssonFeatureProcessor
from ericsson_skill_generator import EricssonSkillGenerator


class TestValidator:
    """Comprehensive test validator for Ericsson processing pipeline"""

    def __init__(self):
        self.test_start_time = time.time()
        self.source_dir = Path("elex_features_only")
        self.test_output_dir = Path("test_output")

    def validate_prerequisites(self):
        """Validate test prerequisites"""
        print("🔍 Validating test prerequisites...")

        # Check source directory
        if not self.source_dir.exists():
            print(f"❌ Source directory not found: {self.source_dir}")
            return False

        # Count available files
        md_files = list(self.source_dir.rglob("*.md"))
        if len(md_files) < 5:
            print(f"❌ Insufficient markdown files: found {len(md_files)}, need at least 5")
            return False

        print(f"✅ Found {len(md_files)} markdown files available")
        return True

    def clean_test_output(self):
        """Clean previous test output"""
        if self.test_output_dir.exists():
            print("🧹 Cleaning previous test output...")
            shutil.rmtree(self.test_output_dir)

    def run_processing_test(self):
        """Run the 5-file processing test"""
        print("\n📊 Phase 1: Processing 5 test files")
        print("=" * 50)

        # Create processor with small batch size
        processor = EricssonFeatureProcessor(
            source_dir=str(self.source_dir),
            output_dir=str(self.test_output_dir),
            batch_size=5
        )

        # Process exactly 5 files
        processor.process_all(limit=5)

        # Collect processing statistics
        processing_stats = {
            'total_files_found': processor.stats.get('total_files', 0),
            'files_processed': processor.stats.get('processed', 0),
            'features_extracted': len(processor.features),
            'errors': processor.stats.get('errors', 0),
            'processing_time': time.time() - processor.stats.get('start_time', time.time())
        }

        return processor, processing_stats

    def validate_extraction_quality(self, processor):
        """Validate the quality of extracted features"""
        print("\n🔬 Phase 2: Validating extraction quality")
        print("=" * 50)

        quality_stats = {
            'features_with_cxc': 0,
            'features_with_parameters': 0,
            'features_with_counters': 0,
            'features_with_events': 0,
            'features_with_guidelines': 0,
            'total_parameters': 0,
            'total_counters': 0,
            'total_events': 0,
            'unique_cxc_codes': set()
        }

        print(f"📋 Analyzing {len(processor.features)} extracted features:")

        for feature_id, feature in processor.features.items():
            # Feature basic info
            name = feature.name or 'Unknown'
            cxc_code = feature.cxc_code
            params = feature.parameters
            counters = feature.counters
            events = feature.events
            guidelines = feature.engineering_guidelines

            print(f"\n  📄 {name} (FAJ {feature_id})")
            if cxc_code:
                print(f"    🔧 CXC Code: {cxc_code}")
                quality_stats['features_with_cxc'] += 1
                quality_stats['unique_cxc_codes'].add(cxc_code)

            print(f"    ⚙️  Parameters: {len(params)}")
            print(f"    📊 Counters: {len(counters)}")
            print(f"    📡 Events: {len(events)}")
            print(f"    📖 Guidelines: {'Yes' if guidelines else 'No'}")

            # Update counters
            if params:
                quality_stats['features_with_parameters'] += 1
                quality_stats['total_parameters'] += len(params)
            if counters:
                quality_stats['features_with_counters'] += 1
                quality_stats['total_counters'] += len(counters)
            if events:
                quality_stats['features_with_events'] += 1
                quality_stats['total_events'] += len(events)
            if guidelines:
                quality_stats['features_with_guidelines'] += 1

        quality_stats['unique_cxc_codes'] = len(quality_stats['unique_cxc_codes'])
        return quality_stats

    def test_skill_generation(self):
        """Test skill generation from processed data"""
        print("\n🚀 Phase 3: Generating Claude skill")
        print("=" * 50)

        try:
            generator = EricssonSkillGenerator(
                data_dir=str(self.test_output_dir / "ericsson_data"),
                output_dir=str(self.test_output_dir)
            )

            generation_start = time.time()
            generator.generate_skill()
            generation_time = time.time() - generation_start

            # Check generated files
            skill_dir = self.test_output_dir / "ericsson"
            skill_files = list(skill_dir.rglob("*")) if skill_dir.exists() else []

            # Find generated zip file
            zip_files = list(self.test_output_dir.glob("ericsson_ran_features_skill_*_features.zip"))

            generation_stats = {
                'success': True,
                'generation_time': generation_time,
                'skill_files_created': len(skill_files),
                'zip_file_created': len(zip_files) > 0,
                'zip_file_name': zip_files[0].name if zip_files else None,
                'zip_size_mb': zip_files[0].stat().st_size / (1024*1024) if zip_files else 0
            }

            return generation_stats

        except Exception as e:
            print(f"❌ Skill generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_time': 0,
                'skill_files_created': 0,
                'zip_file_created': False
            }

    def validate_indices(self):
        """Validate search indices creation"""
        print("\n📚 Phase 4: Validating search indices")
        print("=" * 50)

        indices_dir = self.test_output_dir / "ericsson_data" / "indices"
        if not indices_dir.exists():
            print("❌ Indices directory not found")
            return {'success': False, 'indices_created': 0}

        index_files = list(indices_dir.glob("*_index.json"))
        indices_stats = {'success': True, 'indices_created': len(index_files)}

        print(f"📊 Created {len(index_files)} search indices:")
        for index_file in index_files:
            index_name = index_file.stem.replace('_index', '')
            try:
                index_data = json.loads(index_file.read_text())
                print(f"  📋 {index_name}: {len(index_data)} entries")
                indices_stats[f'{index_name}_entries'] = len(index_data)
            except Exception as e:
                print(f"  ❌ {index_name}: Error reading - {e}")
                indices_stats[f'{index_name}_entries'] = 0

        return indices_stats

    def print_summary(self, processing_stats, quality_stats, generation_stats, indices_stats):
        """Print comprehensive test summary"""
        total_time = time.time() - self.test_start_time

        print("\n" + "="*60)
        print("🎯 COMPREHENSIVE TEST SUMMARY")
        print("="*60)

        # Processing Summary
        print(f"\n📊 PROCESSING RESULTS:")
        print(f"  Files found: {processing_stats['total_files_found']}")
        print(f"  Files processed: {processing_stats['files_processed']}")
        print(f"  Features extracted: {processing_stats['features_extracted']}")
        print(f"  Processing errors: {processing_stats['errors']}")
        print(f"  Processing time: {processing_stats['processing_time']:.2f} seconds")

        # Quality Summary
        print(f"\n🔬 EXTRACTION QUALITY:")
        print(f"  Features with CXC codes: {quality_stats['features_with_cxc']}")
        print(f"  Features with parameters: {quality_stats['features_with_parameters']}")
        print(f"  Features with counters: {quality_stats['features_with_counters']}")
        print(f"  Features with events: {quality_stats['features_with_events']}")
        print(f"  Features with guidelines: {quality_stats['features_with_guidelines']}")
        print(f"  Total parameters extracted: {quality_stats['total_parameters']}")
        print(f"  Total counters extracted: {quality_stats['total_counters']}")
        print(f"  Unique CXC codes: {quality_stats['unique_cxc_codes']}")

        # Generation Summary
        print(f"\n🚀 SKILL GENERATION:")
        if generation_stats['success']:
            print(f"  ✅ Generation successful")
            print(f"  Generation time: {generation_stats['generation_time']:.2f} seconds")
            print(f"  Skill files created: {generation_stats['skill_files_created']}")
            print(f"  Zip file created: {generation_stats['zip_file_name']}")
            print(f"  Package size: {generation_stats['zip_size_mb']:.2f} MB")
        else:
            print(f"  ❌ Generation failed: {generation_stats.get('error', 'Unknown error')}")

        # Indices Summary
        print(f"\n📚 SEARCH INDICES:")
        if indices_stats['success']:
            print(f"  ✅ Indices created: {indices_stats['indices_created']}")
            for key, value in indices_stats.items():
                if key.endswith('_entries'):
                    index_name = key.replace('_entries', '')
                    print(f"  {index_name}: {value} entries")
        else:
            print(f"  ❌ Indices creation failed")

        # Final Status
        print(f"\n⏱️  TOTAL TEST TIME: {total_time:.2f} seconds")

        # Success criteria
        success = (
            processing_stats['features_extracted'] > 0 and
            generation_stats['success'] and
            generation_stats['zip_file_created'] and
            indices_stats['success']
        )

        if success:
            print(f"\n🎉 ✅ TEST SUCCESSFUL!")
            print(f"  All validation criteria passed")
            print(f"  Sample skill ready for upload")
        else:
            print(f"\n❌ TEST FAILED!")
            print(f"  Some validation criteria not met")

        print(f"\n📁 OUTPUT LOCATION:")
        print(f"  Test data: {self.test_output_dir}/ericsson_data/")
        print(f"  Skill files: {self.test_output_dir}/ericsson/")
        if generation_stats['zip_file_name']:
            print(f"  Upload package: {self.test_output_dir}/{generation_stats['zip_file_name']}")

        print(f"\n🚀 NEXT STEPS:")
        if success:
            print(f"  1. Upload the zip file to Claude")
            print(f"  2. Test with sample RAN queries")
            print(f"  3. Process full dataset with:")
            print(f"     python3 src/ericsson_feature_processor.py --source elex_features_only")
            print(f"     python3 src/ericsson_skill_generator.py --data-dir output/ericsson_data")
        else:
            print(f"  1. Check error messages above")
            print(f"  2. Verify source files in elex_features_only/")
            print(f"  3. Ensure dependencies are installed: pip3 install -r src/requirements.txt")

        return success


def test_with_5_files():
    """Main test function implementing comprehensive validation"""
    print("🧪 Ericsson RAN Features Processor - 5 File Validation Test")
    print("Based on final-plan.md requirements")

    validator = TestValidator()

    # Phase 0: Prerequisites
    if not validator.validate_prerequisites():
        print("\n❌ Prerequisites validation failed")
        return False

    # Phase 0.5: Clean previous output
    validator.clean_test_output()

    try:
        # Phase 1: Processing Test
        processor, processing_stats = validator.run_processing_test()

        if processing_stats['features_extracted'] == 0:
            print("\n❌ No features extracted - processing failed")
            return False

        # Phase 2: Quality Validation
        quality_stats = validator.validate_extraction_quality(processor)

        # Phase 3: Skill Generation Test
        generation_stats = validator.test_skill_generation()

        # Phase 4: Indices Validation
        indices_stats = validator.validate_indices()

        # Phase 5: Comprehensive Summary
        success = validator.print_summary(processing_stats, quality_stats, generation_stats, indices_stats)

        return success

    except Exception as e:
        print(f"\n❌ Test execution failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_with_5_files()
    sys.exit(0 if success else 1)