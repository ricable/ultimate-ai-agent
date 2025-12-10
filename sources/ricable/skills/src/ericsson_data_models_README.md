# Ericsson Feature Data Models

This module provides comprehensive data structures for representing Ericsson RAN feature documentation. The models are designed to be type-safe, validated, and compatible with the existing Ericsson feature processor.

## Overview

The core data model is `EricssonFeature`, which represents all aspects of an Ericsson RAN feature including:

- **Identity**: FAJ ID, name, CXC code
- **Classification**: Value package, access type, node type
- **Technical Content**: Parameters, counters, events
- **Operations**: Activation/deactivation steps
- **Dependencies**: Prerequisites, conflicts, related features
- **Guidelines**: Engineering recommendations
- **Impact**: Performance and network impact assessment

## Quick Start

```python
from ericsson_data_models import EricssonFeature, create_parameter, create_counter

# Create a feature
feature = EricssonFeature(
    id="FAJ 121 3094",
    name="My Feature",
    description="Feature description",
    cxc_code="CXC4011809"
)

# Add technical details
feature.add_parameter(create_parameter(
    name="param1",
    mo_class="UtranCell",
    description="Parameter description",
    parameter_type="Integer",
    default_value="100"
))

feature.add_counter(create_counter(
    name="counter1",
    description="Counter description",
    counter_type="PM"
))

# Validate the feature
errors = feature.validate()
if errors:
    for error in errors:
        print(f"Validation error: {error}")
else:
    print("Feature is valid!")
```

## Core Classes

### EricssonFeature

The main dataclass representing a complete Ericsson feature.

**Key Properties:**
- `id`: FAJ ID (normalized to "FAJ XXX XXXX" format)
- `name`: Feature name
- `cxc_code`: CXC activation code (format: "CXXXXXXX")
- `description`: Feature description
- `parameters`: List of Parameter objects
- `counters`: List of Counter objects
- `events`: List of Event objects
- `dependencies`: Dependency relationships
- `activation_step`: MO activation instructions
- `deactivation_step`: MO deactivation instructions
- `engineering_guidelines`: Best practices and recommendations

**Key Methods:**
- `validate()`: Validate all feature data
- `normalize_faj_id()`: Normalize FAJ ID to standard format
- `to_dict()`/`from_dict()`: Dictionary conversion
- `to_json()`/`from_json()`: JSON serialization
- `get_searchable_text()`: Get combined searchable text
- `get_parameter(name)`, `get_counter(name)`, `get_event(name)`: Lookup methods

### Parameter

Represents a feature parameter with MO class information.

```python
parameter = Parameter(
    name="icicAlgorithm",
    mo_class="UtranCell",
    description="ICIC algorithm selection",
    parameter_type="Enumeration",
    default_value="1",
    range_values="0-3"
)
```

### Counter

Represents a PM counter associated with the feature.

```python
counter = Counter(
    name="numAdjustments",
    description="Number of adjustments performed",
    counter_type="PM",
    unit="count",
    category="Performance"
)
```

### Event

Represents an event associated with the feature.

```python
event = Event(
    name="algorithmChange",
    description="Algorithm configuration changed",
    trigger_conditions="Manual configuration change",
    severity="Information",
    parameters=["param1", "param2"]
)
```

### Dependency

Represents feature dependencies and relationships.

```python
dependencies = Dependency(
    prerequisites=["FAJ 115 2008", "FAJ 118 9001"],
    conflicts=[],
    related=["FAJ 122 4005"]
)
```

## Factory Functions

The module provides convenient factory functions:

```python
from ericsson_data_models import (
    create_feature_from_faj_id,
    create_parameter,
    create_counter,
    create_event
)

# Create feature from FAJ ID
feature = create_feature_from_faj_id(
    faj_id="FAJ 121 3094",
    name="Feature Name",
    description="Feature description"
)

# Create parameter
param = create_parameter(
    name="paramName",
    mo_class="MOClass",
    description="Description",
    parameter_type="Integer"
)
```

## Validation

The models include comprehensive validation:

```python
feature = EricssonFeature(id="INVALID", name="", description="")
errors = feature.validate()
# Returns: ["FAJ ID 'INVALID' does not match format 'FAJ XXX XXXX'",
#           "Feature name is required",
#           "Feature description is required"]

# Validate individual components
from ericsson_data_models import validate_faj_format, validate_cxc_code

print(validate_faj_format("FAJ 121 3094"))  # True
print(validate_faj_format("INVALID"))        # False
print(validate_cxc_code("CXC4011809"))       # True
```

## FAJ ID Handling

The module provides robust FAJ ID normalization and extraction:

```python
# Normalization (various formats to standard "FAJ XXX XXXX")
print(EricssonFeature.normalize_faj_id("FAJ1213094"))   # "FAJ 121 3094"
print(EricssonFeature.normalize_faj_id("121 3094"))     # "FAJ 121 3094"
print(EricssonFeature.normalize_faj_id("FAJ 1213094"))  # "FAJ 121 3094"

# Extraction from text
text = "Features: FAJ 121 3094 and FAJ 115 2008"
faj_ids = extract_faj_ids_from_text(text)
# Returns: ["FAJ 115 2008", "FAJ 121 3094"]
```

## JSON Serialization

Full JSON serialization support with round-trip compatibility:

```python
# Serialize to JSON
feature = EricssonFeature(id="FAJ 121 3094", name="Test")
json_str = feature.to_json(indent=2)

# Deserialize from JSON
restored = EricssonFeature.from_json(json_str)
assert feature.id == restored.id
```

## Backward Compatibility

The models are designed to work with existing data formats:

```python
# Old format data (from existing caches)
old_data = {
    "id": "FAJ 121 3094",
    "name": "Feature",
    "parameters": [
        {"name": "param", "type": "Integer", "default": "100"}  # Old field names
    ]
}

# Migrate to new model
feature = EricssonFeature.from_dict(old_data)
# Automatically maps 'type' -> 'parameter_type', 'default' -> 'default_value'
```

## Enums

The module includes enums for standardized values:

```python
from ericsson_data_models import AccessType, NodeType

# Access types
AccessType.LICENSED.value    # "Licensed"
AccessType.FREE.value        # "Free"
AccessType.BUNDLED.value     # "Bundled"
AccessType.TRIAL.value       # "Trial"

# Node types
NodeType.DU.value            # "DU"
NodeType.CU.value            # "CU"
NodeType.BSR.value           # "BSR"
```

## Performance Features

The models include performance optimization features:

- **Searchable text**: `get_searchable_text()` combines all text fields for efficient searching
- **Duplicate prevention**: `add_parameter()`, `add_counter()`, `add_event()` prevent duplicates
- **Efficient lookups**: `get_parameter()`, `get_counter()`, `get_event()` for O(n) lookup
- **Validation caching**: Validation results are cached for performance

## Integration with Existing Processor

The models are designed to integrate seamlessly with the existing `EricssonFeatureProcessor`:

```python
from ericsson_feature_processor import EricssonFeatureProcessor
from ericsson_data_models import EricssonFeature

# Use with existing processor
processor = EricssonFeatureProcessor("source_dir", "output_dir")

# Create feature using new model
feature = EricssonFeature(id="FAJ 121 3094", name="New Feature")

# Add to processor (processor expects dictionary format)
processor.features[feature.id] = feature

# Processor can serialize/deserialize automatically
feature_dict = feature.to_dict()
restored = EricssonFeature.from_dict(feature_dict)
```

## Testing

The module includes comprehensive test suites:

```bash
# Run basic tests
python3 ericsson_data_models.py

# Run comprehensive tests
python3 test_data_models.py

# Run integration tests
python3 integration_test.py
```

## Usage Patterns

### Creating Features from Documentation

```python
# Parse documentation and create feature
def parse_feature_doc(doc_text):
    faj_ids = extract_faj_ids_from_text(doc_text)
    if not faj_ids:
        return None

    feature = create_feature_from_faj_id(
        faj_id=faj_ids[0],
        name=extract_name(doc_text),
        description=extract_description(doc_text)
    )

    # Parse technical sections
    feature.parameters = parse_parameters(doc_text)
    feature.counters = parse_counters(doc_text)
    feature.events = parse_events(doc_text)

    return feature
```

### Search and Lookup

```python
# Search features by content
def search_features(features, query):
    query = query.lower()
    results = []

    for feature in features:
        if query in feature.get_searchable_text():
            results.append(feature)

    return results

# Find features with specific parameters
def find_features_with_parameter(features, param_name):
    return [f for f in features if f.has_parameter(param_name)]
```

### Validation Pipeline

```python
def validate_feature_collection(features):
    all_errors = []

    for feature in features:
        errors = feature.validate()
        if errors:
            all_errors.extend([f"{feature.id}: {error}" for error in errors])

    return all_errors
```

This data models module provides a solid foundation for Ericsson feature processing with comprehensive validation, type safety, and backward compatibility.