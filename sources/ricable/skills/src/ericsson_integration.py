#!/usr/bin/env python3
"""
Ericsson RAN Features Integration Script
Complete pipeline from markdown files to Claude skill zip file

This script orchestrates the entire process:
1. Processes markdown files into structured data
2. Generates Claude skill from processed data
3. Packages everything into a ready-to-upload zip file
4. Provides comprehensive error handling and progress reporting
"""

import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

# Import our modules
from ericsson_feature_processor import EricssonFeatureProcessor
from ericsson_skill_generator import EricssonSkillGenerator


class EricssonIntegration:
    """Main integration orchestrator for the complete pipeline"""

    def __init__(self, source_dir: str, output_dir: str = "output"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)

        # Configuration
        self.batch_size = 50
        self.test_mode = False
        self.test_limit = 5

        # State tracking
        self.start_time = time.time()
        self.processor = None
        self.generator = None

        # Statistics
        self.stats = {
            'phase': 'initialization',
            'start_time': self.start_time,
            'features_processed': 0,
            'errors': 0,
            'warnings': []
        }

        print("🚀 Ericsson RAN Features Integration Pipeline")
        print("=" * 60)

    def validate_environment(self) -> bool:
        """Validate the processing environment"""
        print("\n🔍 Validating environment...")

        # Check source directory
        if not self.source_dir.exists():
            print(f"❌ Source directory not found: {self.source_dir}")
            return False

        # Check for markdown files
        md_files = list(self.source_dir.rglob("*.md"))
        if not md_files:
            print(f"❌ No markdown files found in {self.source_dir}")
            return False

        print(f"✅ Source directory: {self.source_dir}")
        print(f"✅ Found {len(md_files)} markdown files")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"✅ Output directory: {self.output_dir}")

        return True

    def run_processing_phase(self, limit: Optional[int] = None) -> bool:
        """Run the markdown processing phase"""
        print("\n" + "=" * 60)
        print("📄 PHASE 1: Processing Markdown Files")
        print("=" * 60)

        self.stats['phase'] = 'processing'

        try:
            # Create processor
            self.processor = EricssonFeatureProcessor(
                source_dir=str(self.source_dir),
                output_dir=str(self.output_dir),
                batch_size=self.batch_size
            )

            # Process files
            print(f"Starting processing...")
            if limit:
                print(f"🎯 TEST MODE: Processing only {limit} files")

            self.processor.process_all(limit=limit)

            # Update statistics
            self.stats['features_processed'] = len(self.processor.features)
            self.stats['errors'] = len(self.processor.error_files)

            # Check if we got any features
            if not self.processor.features:
                print("\n❌ No features were extracted from the markdown files")
                print("This could mean:")
                print("- Files don't contain FAJ IDs in the expected format")
                print("- Markdown structure is different from expected")
                print("- Files might be corrupted or empty")
                return False

            print(f"\n✅ Processing complete!")
            print(f"   Features extracted: {len(self.processor.features)}")
            print(f"   Errors encountered: {len(self.processor.error_files)}")

            if self.processor.error_files:
                print("\n⚠️  Processing errors (first 5):")
                for file, error in self.processor.error_files[:5]:
                    print(f"   {Path(file).name}: {error}")

            return True

        except Exception as e:
            print(f"\n❌ Processing phase failed: {e}")
            print(f"Error details: {traceback.format_exc()}")
            return False

    def run_generation_phase(self) -> bool:
        """Run the skill generation phase"""
        print("\n" + "=" * 60)
        print("🎯 PHASE 2: Generating Claude Skill")
        print("=" * 60)

        self.stats['phase'] = 'generation'

        try:
            # Create generator
            data_dir = self.output_dir / "ericsson_data"
            if not data_dir.exists():
                print(f"❌ Processed data directory not found: {data_dir}")
                return False

            self.generator = EricssonSkillGenerator(
                data_dir=str(data_dir),
                output_dir=str(self.output_dir)
            )

            # Generate skill
            print("Generating Claude skill structure...")
            self.generator.generate_skill()

            print(f"\n✅ Skill generation complete!")
            print(f"   Skill directory: {self.output_dir}/ericsson")

            return True

        except Exception as e:
            print(f"\n❌ Generation phase failed: {e}")
            print(f"Error details: {traceback.format_exc()}")
            return False

    def validate_output(self) -> Tuple[bool, Optional[str]]:
        """Validate the final output"""
        print("\n" + "=" * 60)
        print("✔️  PHASE 3: Validating Output")
        print("=" * 60)

        # Check for zip file
        zip_files = list(self.output_dir.glob("ericsson_ran_features_skill_*.zip"))

        if not zip_files:
            print("❌ No skill zip file found")
            return False, None

        zip_file = zip_files[0]  # Get the most recent

        # Validate zip file contents
        try:
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as zf:
                files = zf.namelist()

            # Check for required files
            required_files = ['SKILL.md', 'references/']
            missing_files = []

            for req_file in required_files:
                if not any(f.startswith(req_file.rstrip('/')) for f in files):
                    missing_files.append(req_file)

            if missing_files:
                print(f"❌ Missing required files in zip: {missing_files}")
                return False, str(zip_file)

            # Get file size
            size_mb = zip_file.stat().st_size / (1024 * 1024)

            print(f"✅ Validation complete!")
            print(f"   Skill file: {zip_file.name}")
            print(f"   File size: {size_mb:.2f} MB")
            print(f"   Total files: {len(files)}")

            return True, str(zip_file)

        except Exception as e:
            print(f"❌ Zip file validation failed: {e}")
            return False, str(zip_file)

    def print_final_summary(self, zip_file: Optional[str] = None):
        """Print final processing summary"""
        duration = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("📊 FINAL SUMMARY")
        print("=" * 60)

        print(f"Total processing time: {duration:.2f} seconds")
        print(f"Features processed: {self.stats['features_processed']}")
        print(f"Errors encountered: {self.stats['errors']}")

        if zip_file and Path(zip_file).exists():
            zip_path = Path(zip_file)
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"Generated skill: {zip_path.name}")
            print(f"Skill size: {size_mb:.2f} MB")

            print("\n🎉 SUCCESS! Ready to upload to Claude")
            print("\nNext steps:")
            print(f"1. Upload {zip_path.name} to Claude.ai/skills")
            print("2. Test with sample queries like:")
            print("   - 'Tell me about MIMO Sleep Mode'")
            print("   - 'How do I activate CXC4011808?'")
            print("   - 'What are the energy saving features?'")

        else:
            print("❌ Processing incomplete - no skill file generated")

        if self.stats['warnings']:
            print("\n⚠️  Warnings:")
            for warning in self.stats['warnings']:
                print(f"   {warning}")

    def run_full_pipeline(self, test_mode: bool = False) -> bool:
        """Run the complete pipeline"""
        self.test_mode = test_mode

        # Phase 0: Environment validation
        if not self.validate_environment():
            return False

        # Phase 1: Processing
        limit = self.test_limit if test_mode else None
        if not self.run_processing_phase(limit):
            return False

        # Phase 2: Generation
        if not self.run_generation_phase():
            return False

        # Phase 3: Validation
        success, zip_file = self.validate_output()

        # Final summary
        self.print_final_summary(zip_file if success else None)

        return success

    def resume_from_cache(self) -> bool:
        """Resume processing from cached data"""
        print("\n🔄 Checking for cached data...")

        cache_dir = self.output_dir / "ericsson_data"
        if not cache_dir.exists():
            print("No cached data found")
            return False

        summary_file = cache_dir / "summary.json"
        if summary_file.exists():
            summary = json.loads(summary_file.read_text())
            print(f"Found cached data for {summary.get('total_features', 0)} features")

            response = input("\nUse cached data and skip to generation? (y/n): ")
            return response.lower().startswith('y')

        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Ericsson RAN Features Integration Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with 5 files (recommended first step)
  python ericsson_integration.py --test

  # Process all files
  python ericsson_integration.py --source elex_features_only

  # Custom configuration
  python ericsson_integration.py --source /path/to/docs --output custom_output --batch-size 25
        """
    )

    parser.add_argument(
        '--source',
        default='elex_features_only',
        help='Source directory containing markdown files (default: elex_features_only)'
    )

    parser.add_argument(
        '--output',
        default='output',
        help='Output directory (default: output)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with only 5 files'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Limit processing to N files'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='Batch size for processing (default: 50)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from cached data if available'
    )

    args = parser.parse_args()

    # Create integration instance
    integration = EricssonIntegration(
        source_dir=args.source,
        output_dir=args.output
    )

    # Configure
    integration.batch_size = args.batch_size

    # Handle test mode
    test_mode = args.test
    limit = args.limit

    if test_mode:
        print("🧪 RUNNING IN TEST MODE")
        limit = 5

    if limit:
        print(f"🎯 LIMITING TO {limit} FILES")

    # Check for resume option
    if args.resume and integration.resume_from_cache():
        # Skip directly to generation phase
        print("\n📄 PHASE 1: Skipping (using cached data)")
        success = integration.run_generation_phase()
        if success:
            success, zip_file = integration.validate_output()
            integration.print_final_summary(zip_file if success else None)
        sys.exit(0 if success else 1)

    # Run full pipeline
    success = integration.run_full_pipeline(test_mode=test_mode)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()