"""
RTB Template Processing Example
Demonstrates how to use the advanced RTB template processor with Python logic integration
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rtb_schema import RTBTemplate, CustomFunction
from rtb_processor import RTBTemplateProcessor


def main():
    """Main example function"""

    # Example 1: Load and process an existing template
    print("=" * 60)
    print("Example 1: Processing RTB Template with Custom Logic")
    print("=" * 60)

    template_path = Path(__file__).parent / "rtb_template_with_logic.json"

    if template_path.exists():
        # Load template from file
        template = RTBTemplate(**json.loads(template_path.read_text()))
        processor = RTBTemplateProcessor(template)

        # Define context for processing
        context = {
            "cell_count": 3,
            "active_ue_count": 250,
            "current_load": 75,
            "average_distance": 2.5,  # km
            "cell_height": 35,  # meters
            "traffic_percentage": 65,
            "user_density": 150,
            "hour_of_day": 14,  # 2 PM
            "current_hour": datetime.now().hour,
            "cell_density_per_km2": 120,
            "average_velocity": 30,  # km/h
            "interference_index": 5.5,
            "energy_tier": "medium",
            "traffic_mix": ["video", "data", "voice"],
            "service_type": "video_streaming",
            "user_priority": 4,
            "latency_req": "low",
            "throughput": 8
        }

        # Process template
        print("Processing template with context...")
        result = processor.process_template(context)

        # Display results
        print("\nProcessed Configuration:")
        print("-" * 40)

        # Show conditional results
        if '$cond' in template_path.read_text():
            print("\nConditional Logic Results:")
            if 'enableMassiveMIMO' in result:
                print(f"  Massive MIMO: {result.get('enableMassiveMIMO', 'Not evaluated')}")
            if 'enablePowerSaving' in result:
                print(f"  Power Saving: {result.get('enablePowerSaving', 'Not evaluated')}")
            if 'dynamicBandwidth' in result:
                print(f"  Dynamic Bandwidth: {result.get('dynamicBandwidth', 'Not evaluated')}")

        # Show evaluation results
        if '$eval' in template_path.read_text():
            print("\nEvaluation Function Results:")
            if 'optimalTilt' in result:
                print(f"  Optimal Antenna Tilt: {result.get('optimalTilt', 'Not evaluated')}°")
            if 'caConfiguration' in result:
                ca_config = result.get('caConfiguration', {})
                print(f"  CA Enabled: {ca_config.get('enabled', False)}")
                print(f"  Max SCells: {ca_config.get('max_scells', 0)}")
                print(f"  Strategy: {ca_config.get('strategy', 'N/A')}")
            if 'powerOptimization' in result:
                power_config = result.get('powerOptimization', {})
                print(f"  Target Power: {power_config.get('target_power', 'N/A')} dBm")
                print(f"  Energy Efficiency: {power_config.get('energy_efficiency', 'N/A')}")
                print(f"  Estimated Savings: {power_config.get('estimated_savings', 'N/A')}%")
            if 'qciSelection' in result:
                qci = result.get('qciSelection', {})
                print(f"  Selected QCI: {qci.get('qci', 9)}")
                print(f"  Priority: {qci.get('priority', 9)}")
                print(f"  Type: {qci.get('type', 'NON-GBR')}")

        # Show metrics
        metrics = processor.get_metrics()
        print("\nProcessing Metrics:")
        print(f"  Templates Processed: {metrics['templates_processed']}")
        print(f"  Conditions Processed: {metrics['conditions_processed']}")
        print(f"  Functions Executed: {metrics['functions_executed']}")
        print(f"  Processing Time: {metrics['processing_time_ms']} ms")
        print(f"  Cache Hits: {metrics['cache_hits']}")
        print(f"  Cache Misses: {metrics['cache_misses']}")

        # Save processed configuration
        output_file = Path(__file__).parent / "processed_rtb_config.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nProcessed configuration saved to: {output_file}")

    # Example 2: Create template programmatically
    print("\n" + "=" * 60)
    print("Example 2: Creating Template Programmatically")
    print("=" * 60)

    # Define custom functions
    custom_functions = [
        CustomFunction(
            name="calculateCapacity",
            args=["cell_type", "bandwidth_mhz"],
            body=[
                "# Calculate cell capacity based on type and bandwidth",
                "base_capacity = {",
                "    'macro': {100: 1000, 200: 2000, 400: 4000},",
                "    'pico': {10: 100, 20: 200, 40: 400},",
                "    'femto': {5: 50, 10: 100, 20: 200}",
                "}",
                "",
                "cell_capacity = base_capacity.get(cell_type, {}).get(bandwidth_mhz, 0)",
                "",
                "# Apply technology factor",
                "tech_factor = 1.5 if '5G' in cell_type.lower() else 1.0",
                "return int(cell_capacity * tech_factor)"
            ]
        ),
        CustomFunction(
            name="getOptimalBand",
            args=["user_count", "traffic_type"],
            body=[
                "# Determine optimal frequency band",
                "if user_count > 500:",
                "    if 'video' in traffic_type.lower():",
                "        return '2600'  # mmWave for high capacity",
                "    else:",
                "        return '1800'  # Mid-band for balanced",
                "elif user_count > 100:",
                "    return '800' if 'video' in traffic_type.lower() else '700'",
                "else:",
                "    return '700'",
                ""
            ]
        )
    ]

    # Create template
    dynamic_template = RTBTemplate(
        meta={
            "version": "2.1.0",
            "author": ["Dynamic Template Generator"],
            "description": "Template created programmatically with custom logic",
            "tags": ["dynamic", "5G", "AI"],
            "environment": "prod"
        },
        custom_functions=custom_functions,
        conditional_logic={
            "enableHighCapacity": {
                "condition": "user_count > 200",
                "then_value": {"feature": "highCapacityMode", "enabled": True},
                "else_value": {"feature": "standardMode", "enabled": False}
            }
        },
        evaluation_logic={
            "cellCapacity": {
                "eval": "calculateCapacity",
                "parameters": {"args": ["cell_type", "bandwidth"]}
            },
            "optimalBand": {
                "eval": "getOptimalBand",
                "parameters": {"args": ["total_users", "service_types"]}
            }
        }
    )

    # Create processor
    processor2 = RTBTemplateProcessor(dynamic_template)

    # Define different scenarios
    scenarios = [
        {
            "name": "High Traffic Urban",
            "context": {
                "cell_type": "macro_5G",
                "bandwidth": 100,
                "user_count": 800,
                "traffic_type": ["video", "data", "voice"],
                "total_users": 1200,
                "service_types": ["video_streaming", "web_browsing"]
            }
        },
        {
            "name": "Medium Traffic Suburban",
            "context": {
                "cell_type": "pico",
                "bandwidth": 40,
                "user_count": 150,
                "traffic_type": ["data", "voice"],
                "total_users": 200,
                "service_types": ["web_browsing", "social_media"]
            }
        },
        {
            "name": "Low Traffic Rural",
            "context": {
                "cell_type": "macro",
                "bandwidth": 20,
                "user_count": 50,
                "traffic_type": ["voice"],
                "total_users": 100,
                "service_types": ["voice", "messaging"]
            }
        }
    ]

    # Process each scenario
    for scenario in scenarios:
        print(f"\nProcessing: {scenario['name']}")
        print("-" * 40)

        result = processor2.process_template(scenario['context'])

        if 'cellCapacity' in result:
            print(f"  Calculated Capacity: {result['cellCapacity']} users")
        if 'optimalBand' in result:
            print(f"  Optimal Band: {result['optimalBand']} MHz")
        if 'enableHighCapacity' in result:
            print(f"  High Capacity Mode: {result['enableHighCapacity']['enabled']}")

        # Save scenario result
        scenario_file = Path(__file__).parent / f"scenario_{scenario['name'].lower().replace(' ', '_')}.json"
        with open(scenario_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  Saved to: {scenario_file}")

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()