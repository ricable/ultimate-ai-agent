#!/usr/bin/env python3
"""
Test script for Ericsson data models
Demonstrates usage with realistic Ericsson feature data
"""

from ericsson_data_models import (
    EricssonFeature, Parameter, Counter, Event, Dependency,
    AccessType, NodeType, PerformanceImpact, NetworkImpact,
    extract_faj_ids_from_text, validate_faj_format, validate_cxc_code,
    create_feature_from_faj_id, create_parameter, create_counter, create_event
)


def test_faj_id_extraction():
    """Test FAJ ID extraction from realistic text"""
    print("Testing FAJ ID Extraction")
    print("-" * 30)

    sample_text = """
    Feature Identity: FAJ 121 3094 - Enhanced Inter-Cell Interference Coordination
    This feature references FAJ 115 2008 and FAJ 118 9001.
    Prerequisites: FAJ 121 3094 must be activated before FAJ 122 4005.
    """

    faj_ids = extract_faj_ids_from_text(sample_text)
    print(f"Found FAJ IDs: {faj_ids}")
    print()


def test_realistic_feature():
    """Test creating a realistic Ericsson feature"""
    print("Creating Realistic Feature")
    print("-" * 30)

    # Create feature using factory function
    feature = create_feature_from_faj_id(
        faj_id="FAJ 121 3094",
        name="Enhanced Inter-Cell Interference Coordination",
        description="Advanced ICIC algorithms for improved cell edge performance in dense networks"
    )

    # Set classification
    feature.cxc_code = "CXC4011809"
    feature.value_package = "Advanced RAN Features"
    feature.access_type = AccessType.LICENSED.value
    feature.node_type = NodeType.DU.value

    # Add technical parameters
    feature.add_parameter(create_parameter(
        name="icicAlgorithm",
        mo_class="UtranCell",
        description="Selection of ICIC algorithm type",
        parameter_type="Enumeration",
        default_value="1",
        range_values="0-3"
    ))

    feature.add_parameter(create_parameter(
        name="icicThreshold",
        mo_class="UtranCell",
        description="ICIC interference threshold",
        parameter_type="Integer",
        default_value="-85",
        range_values="-120..-60",
        unit="dBm"
    ))

    # Add PM counters
    feature.add_counter(create_counter(
        name="numIcicAdjustments",
        description="Number of ICIC adjustments performed",
        counter_type="PM",
        unit="count",
        category="Performance"
    ))

    feature.add_counter(create_counter(
        name="avgIcin",
        description="Average inter-cell interference level",
        counter_type="PM",
        unit="dBm",
        category="Quality"
    ))

    # Add events
    feature.add_event(create_event(
        name="icicAlgorithmChange",
        description="ICIC algorithm was changed",
        trigger_conditions="Manual configuration change or adaptive selection",
        severity="Information",
        category="Configuration"
    ))

    # Set operational steps
    feature.activation_step = (
        "1. Set the FeatureState.featureState attribute to ACTIVATED in the "
        "FeatureState=CXC4011809 MO instance.\n"
        "2. Configure icicAlgorithm parameter in UtranCell MO.\n"
        "3. Set appropriate icicThreshold values."
    )

    feature.deactivation_step = (
        "1. Set the FeatureState.featureState attribute to DEACTIVATED in the "
        "FeatureState=CXC4011809 MO instance."
    )

    # Add engineering guidelines
    feature.engineering_guidelines = (
        "This feature should be deployed in dense urban environments with high "
        "inter-cell interference. Careful planning of ICIC thresholds is required "
        "to balance between interference mitigation and capacity. The feature is "
        "most effective when combined with proper frequency planning and power "
        "control optimization."
    )

    # Set performance impact
    feature.performance_impact.cpu_impact = "Low to Medium"
    feature.performance_impact.memory_impact = "Low"
    feature.performance_impact.throughput_impact = "Positive (up to 15% improvement at cell edge)"

    # Set network impact
    feature.network_impact.signaling_impact = "Low"
    feature.network_impact.handover_impact = "Improved handover success rate"
    feature.network_impact.interference_impact = "Reduced inter-cell interference"

    # Add dependencies
    feature.dependencies.prerequisites = ["FAJ 115 2008", "FAJ 118 9001"]
    feature.dependencies.conflicts = []
    feature.dependencies.related = ["FAJ 122 4005", "FAJ 125 6003"]

    # Validate the feature
    print("Feature validation:")
    errors = feature.validate()
    if errors:
        for error in errors:
            print(f"  ERROR: {error}")
    else:
        print("  ✓ Feature is valid")

    print(f"\nFeature summary:")
    print(f"  ID: {feature.id}")
    print(f"  Name: {feature.name}")
    print(f"  CXC Code: {feature.cxc_code}")
    print(f"  Parameters: {len(feature.parameters)}")
    print(f"  Counters: {len(feature.counters)}")
    print(f"  Events: {len(feature.events)}")
    print(f"  Dependencies: {len(feature.dependencies.prerequisites)} prerequisites")

    # Test parameter lookup
    print(f"\nParameter lookup tests:")
    param = feature.get_parameter("icicAlgorithm")
    if param:
        print(f"  Found parameter: {param.name} (Type: {param.parameter_type})")

    # Test counter lookup
    counter = feature.get_counter("avgIcin")
    if counter:
        print(f"  Found counter: {counter.name} (Unit: {counter.unit})")

    # Test searchable text
    searchable = feature.get_searchable_text()
    print(f"\nSearchable text length: {len(searchable)} characters")
    print(f"Contains 'interference': {'interference' in searchable}")
    print(f"Contains 'icicAlgorithm': {'icicalgorithm' in searchable}")

    print()
    return feature


def test_json_serialization(feature):
    """Test JSON serialization and deserialization"""
    print("JSON Serialization Test")
    print("-" * 30)

    # Serialize to JSON
    json_str = feature.to_json(indent=2)
    print(f"JSON length: {len(json_str)} characters")

    # Deserialize from JSON
    restored_feature = EricssonFeature.from_json(json_str)

    # Verify restoration
    print(f"Restored feature ID: {restored_feature.id}")
    print(f"Restored feature name: {restored_feature.name}")
    print(f"Restored parameters: {len(restored_feature.parameters)}")
    print(f"Restored counters: {len(restored_feature.counters)}")
    print(f"Restored events: {len(restored_feature.events)}")

    # Test that complex objects were properly restored
    param = restored_feature.get_parameter("icicAlgorithm")
    if param:
        print(f"Restored parameter type: {param.mo_class}")

    dep = restored_feature.dependencies
    print(f"Restored dependencies: {len(dep.prerequisites)} prerequisites")

    print()


def test_validation_edge_cases():
    """Test validation with edge cases"""
    print("Validation Edge Cases")
    print("-" * 30)

    test_cases = [
        ("Empty FAJ ID", EricssonFeature(name="Test", description="Test")),
        ("Invalid FAJ ID", EricssonFeature(id="INVALID", name="Test", description="Test")),
        ("Invalid CXC Code", EricssonFeature(id="FAJ 121 3094", name="Test", cxc_code="INVALID", description="Test")),
        ("Invalid Access Type", EricssonFeature(id="FAJ 121 3094", name="Test", access_type="Invalid", description="Test")),
        ("Invalid Node Type", EricssonFeature(id="FAJ 121 3094", name="Test", node_type="Invalid", description="Test")),
    ]

    for case_name, feature in test_cases:
        print(f"{case_name}:")
        errors = feature.validate()
        for error in errors:
            print(f"  - {error}")
        if not errors:
            print("  ✓ Valid")
        print()


def main():
    """Run all tests"""
    print("Ericsson Data Models Test Suite")
    print("=" * 50)
    print()

    # Test FAJ ID extraction
    test_faj_id_extraction()

    # Test realistic feature creation
    feature = test_realistic_feature()

    # Test JSON serialization
    test_json_serialization(feature)

    # Test validation edge cases
    test_validation_edge_cases()

    print("All tests completed!")


if __name__ == "__main__":
    main()