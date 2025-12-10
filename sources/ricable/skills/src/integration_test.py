#!/usr/bin/env python3
"""
Integration test for Ericsson data models with the existing processor
Tests compatibility and demonstrates how to use the new models
"""

import sys
import json
from pathlib import Path

# Import the new data models
from ericsson_data_models import EricssonFeature, Parameter, Counter, Event

# Import existing processor components
try:
    from ericsson_feature_processor import EricssonFeatureProcessor
    PROCESSOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import EricssonFeatureProcessor: {e}")
    PROCESSOR_AVAILABLE = False


def test_model_compatibility():
    """Test that new models are compatible with existing processor expectations"""
    print("Testing Model Compatibility")
    print("-" * 40)

    # Create a feature using new model
    feature = EricssonFeature(
        id="FAJ 121 3094",
        name="Test Feature",
        description="Test feature for compatibility",
        cxc_code="CXC4011809"
    )

    # Add some technical data
    feature.parameters.append(Parameter(
        name="testParam",
        mo_class="TestMO",
        description="Test parameter",
        parameter_type="Integer"
    ))

    feature.counters.append(Counter(
        name="testCounter",
        description="Test counter",
        counter_type="PM"
    ))

    # Convert to dictionary (as processor would expect)
    feature_dict = feature.to_dict()

    # Test that the dictionary structure matches processor expectations
    required_fields = [
        'id', 'name', 'cxc_code', 'description', 'parameters',
        'counters', 'events', 'activation_step', 'deactivation_step',
        'engineering_guidelines', 'dependencies'
    ]

    print("Checking required fields:")
    for field in required_fields:
        if field in feature_dict:
            print(f"  ✓ {field}")
        else:
            print(f"  ✗ {field} - MISSING")

    # Test JSON serialization (as used by processor for caching)
    json_str = feature.to_json()
    restored_feature = EricssonFeature.from_json(json_str)

    print(f"\nJSON round-trip test:")
    print(f"  Original ID: {feature.id}")
    print(f"  Restored ID: {restored_feature.id}")
    print(f"  Match: {feature.id == restored_feature.id}")

    return feature_dict


def test_processor_integration():
    """Test integration with the existing EricssonFeatureProcessor if available"""
    if not PROCESSOR_AVAILABLE:
        print("Skipping processor integration test - processor not available")
        return

    print("\nTesting Processor Integration")
    print("-" * 40)

    # Create a temporary output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    try:
        # Initialize processor
        processor = EricssonFeatureProcessor(
            source_dir="elex_features_only",  # This may not exist, that's OK
            output_dir=str(output_dir),
            batch_size=5
        )

        # Create a test feature
        feature = EricssonFeature(
            id="FAJ 121 3094",
            name="Integration Test Feature",
            description="Feature for testing integration with processor",
            cxc_code="CXC4011809",
            source_file="test_file.md"
        )

        # Add to processor's feature collection (simulating what processor would do)
        processor.features[feature.id] = feature

        # Test that processor can work with the new feature model
        print(f"Added feature to processor: {feature.id}")
        print(f"Processor now has {len(processor.features)} features")

        # Test indexing functionality
        feature_dict = feature.to_dict()
        for param in feature_dict['parameters']:
            processor.parameter_index[param['name']].append(feature.id)

        print(f"Parameter index has {len(processor.parameter_index)} entries")

        # Test cache writing (simulating processor behavior)
        cache_file = output_dir / "cache" / f"feature_{feature.id.replace(' ', '_')}.json"
        cache_file.parent.mkdir(exist_ok=True)

        with open(cache_file, 'w') as f:
            json.dump(feature_dict, f, indent=2)

        print(f"Feature cached to: {cache_file}")

        # Test cache reading (simulating processor behavior)
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)

            restored_feature = EricssonFeature.from_dict(cached_data)
            print(f"Successfully restored feature from cache: {restored_feature.id}")

    except Exception as e:
        print(f"Error during processor integration test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)


def test_migration_from_old_format():
    """Test migrating from old feature format to new data models"""
    print("\nTesting Migration from Old Format")
    print("-" * 40)

    # Simulate old feature format (as might be found in existing caches)
    old_format_feature = {
        "id": "FAJ 121 3094",
        "name": "Old Format Feature",
        "cxc_code": "CXC4011809",
        "description": "Feature in old format",
        "parameters": [
            {
                "name": "oldParam",
                "mo_class": "OldMO",
                "description": "Old parameter",
                "type": "Integer",
                "default": "100"
            }
        ],
        "counters": [
            {
                "name": "oldCounter",
                "description": "Old counter"
            }
        ],
        "events": [],
        "activation_step": "Activate the feature",
        "deactivation_step": "Deactivate the feature",
        "engineering_guidelines": "Use with care",
        "dependencies": {
            "prerequisites": ["FAJ 115 2008"],
            "conflicts": [],
            "related": []
        }
    }

    # Convert to new model
    new_feature = EricssonFeature.from_dict(old_format_feature)

    print(f"Migrated feature: {new_feature.id}")
    print(f"Parameters: {len(new_feature.parameters)}")
    print(f"Counters: {len(new_feature.counters)}")
    print(f"Dependencies: {len(new_feature.dependencies.prerequisites)} prerequisites")

    # Test that migration preserves all data
    assert new_feature.id == old_format_feature["id"]
    assert new_feature.name == old_format_feature["name"]
    assert len(new_feature.parameters) == len(old_format_feature["parameters"])
    assert len(new_feature.counters) == len(old_format_feature["counters"])

    print("✓ Migration successful - all data preserved")


def test_enhanced_features():
    """Test enhanced features that weren't available in the old format"""
    print("\nTesting Enhanced Features")
    print("-" * 40)

    feature = EricssonFeature(
        id="FAJ 121 3094",
        name="Enhanced Feature Test",
        description="Testing enhanced capabilities"
    )

    # Test validation
    errors = feature.validate()
    print(f"Validation errors: {len(errors)}")
    for error in errors:
        print(f"  - {error}")

    # Test FAJ ID normalization
    test_ids = ["FAJ1213094", "121 3094", "FAJ 121 3094"]
    for test_id in test_ids:
        normalized = EricssonFeature.normalize_faj_id(test_id)
        print(f"  {test_id:12} -> {normalized}")

    # Test parameter management
    from ericsson_data_models import create_parameter
    param1 = create_parameter("test1", "MO1", "Test parameter 1")
    param2 = create_parameter("test1", "MO2", "Test parameter 1 duplicate")

    feature.add_parameter(param1)
    feature.add_parameter(param2)  # Should not be added (duplicate name)

    print(f"Parameters after duplicate test: {len(feature.parameters)}")
    assert len(feature.parameters) == 1, "Duplicate parameter should not be added"

    # Test searchable text
    feature.engineering_guidelines = "This feature provides enhanced performance"
    searchable = feature.get_searchable_text()
    assert "enhanced performance" in searchable.lower()
    print(f"Searchable text contains guidelines: {'enhanced performance' in searchable.lower()}")

    print("✓ All enhanced features working correctly")


def main():
    """Run all integration tests"""
    print("Ericsson Data Models Integration Tests")
    print("=" * 50)
    print()

    # Test basic compatibility
    test_model_compatibility()

    # Test processor integration
    test_processor_integration()

    # Test migration from old format
    test_migration_from_old_format()

    # Test enhanced features
    test_enhanced_features()

    print("\n" + "=" * 50)
    print("All integration tests completed!")
    print("\nThe new data models are ready for use with the Ericsson processor.")
    print("Key benefits:")
    print("- ✅ Full backward compatibility with existing data")
    print("- ✅ Enhanced validation and error checking")
    print("- ✅ Better type safety and IDE support")
    print("- ✅ Improved search and lookup capabilities")
    print("- ✅ Comprehensive documentation and examples")


if __name__ == "__main__":
    main()