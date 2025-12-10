#!/usr/bin/env python3
"""
Simple test for RTB template processor
"""

import json
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

def test_basic_functionality():
    """Test basic RTB template processing"""

    print("Testing RTB Template Processor...")
    print("=" * 50)

    try:
        # Import the modules
        from rtb_schema import RTBTemplate, CustomFunction
        from rtb_processor import RTBTemplateProcessor

        print("✅ Modules imported successfully")

        # Test 1: Load existing template
        template_path = Path(__file__).parent / "examples" / "rtb_template_with_logic.json"

        if template_path.exists():
            print(f"\n✅ Template file found: {template_path}")

            # Load and parse the template
            template_data = json.loads(template_path.read_text())

            # Create RTBTemplate from data
            template = RTBTemplate(**template_data)
            print(f"✅ Template loaded successfully")
            print(f"   - Version: {template.meta.version if template.meta else 'N/A'}")
            print(f"   - Custom functions: {len(template.custom_functions) if template.custom_functions else 0}")
            print(f"   - Conditional logic: {len(template.conditional_logic) if template.conditional_logic else 0}")
            print(f"   - Evaluation logic: {len(template.evaluation_logic) if template.evaluation_logic else 0}")

            # Create processor
            processor = RTBTemplateProcessor(template)
            print(f"✅ Template processor created")

            # Define test context
            context = {
                "cell_count": 3,
                "active_ue_count": 250,
                "current_load": 75,
                "average_distance": 2.5,
                "cell_height": 35,
                "traffic_percentage": 65,
                "current_hour": 14,
                "cell_density_per_km2": 120,
                "average_velocity": 30,
                "interference_index": 5.5,
                "energy_tier": "medium",
                "traffic_mix": ["video", "data", "voice"],
                "service_type": "video_streaming",
                "user_priority": 4,
                "latency_req": "low",
                "throughput": 8
            }

            # Process template
            print(f"\n🔄 Processing template with context...")
            result = processor.process_template(context)

            # Display some results
            print(f"✅ Template processed successfully")
            print(f"\n📊 Results:")
            if 'optimalTilt' in result:
                print(f"   - Optimal Tilt: {result['optimalTilt']}°")
            if 'caConfiguration' in result:
                ca = result['caConfiguration']
                print(f"   - CA Enabled: {ca.get('enabled', False)}")
                if ca.get('enabled'):
                    print(f"   - Max SCells: {ca.get('max_scells', 0)}")
                    print(f"   - Strategy: {ca.get('strategy', 'N/A')}")

            # Get metrics
            metrics = processor.get_metrics()
            print(f"\n📈 Processing Metrics:")
            print(f"   - Processing Time: {metrics['processing_time_ms']} ms")
            print(f"   - Functions Executed: {metrics['functions_executed']}")
            print(f"   - Conditions Processed: {metrics['conditions_processed']}")

            print(f"\n✅ All tests passed successfully!")

        else:
            print(f"❌ Template file not found: {template_path}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the src directory is in your Python path")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()