#!/usr/bin/env python3
"""
Ericsson Feature Data Models
Core data structures for representing Ericsson RAN feature documentation
"""

import re
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime
from enum import Enum


class AccessType(Enum):
    """Feature access types"""
    LICENSED = "Licensed"
    FREE = "Free"
    BUNDLED = "Bundled"
    TRIAL = "Trial"


class NodeType(Enum):
    """Supported node types"""
    DU = "DU"
    CU = "CU"
    BSR = "BSR"
    RBS = "RBS"
    INDOOR_DUS = "Indoor DUS"
    MICRO_RBS = "Micro RBS"


@dataclass
class Parameter:
    """Represents a feature parameter with MO class information"""
    name: str = ""
    mo_class: str = ""
    description: str = ""
    parameter_type: str = ""
    default_value: str = ""
    range_values: str = ""
    unit: str = ""
    category: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert parameter to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Parameter':
        """Create parameter from dictionary"""
        # Handle legacy field names and map to current structure
        field_mapping = {
            'type': 'parameter_type',
            'default': 'default_value',
            'range': 'range_values'
        }

        # Apply field mapping
        processed_data = {}
        for k, v in data.items():
            mapped_key = field_mapping.get(k, k)
            processed_data[mapped_key] = v or ""

        # Only include fields that actually exist in the dataclass
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in processed_data.items() if k in valid_fields}

        return cls(**filtered_data)


@dataclass
class Counter:
    """Represents a PM counter associated with the feature"""
    name: str = ""
    description: str = ""
    counter_type: str = ""
    category: str = ""
    unit: str = ""
    scope: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert counter to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Counter':
        """Create counter from dictionary"""
        return cls(**{k: v or "" for k, v in data.items()})


@dataclass
class Event:
    """Represents an event associated with the feature"""
    name: str = ""
    description: str = ""
    trigger_conditions: str = ""
    parameters: List[str] = field(default_factory=list)
    severity: str = ""
    category: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary"""
        # Handle parameters field specifically
        if 'parameters' in data and isinstance(data['parameters'], str):
            data['parameters'] = [p.strip() for p in data['parameters'].split(',')]
        return cls(**{k: v or [] if k == 'parameters' else v or "" for k, v in data.items()})


@dataclass
class Dependency:
    """Represents feature dependencies and relationships"""
    prerequisites: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    related: List[str] = field(default_factory=list)
    required_features: List[str] = field(default_factory=list)
    optional_features: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert dependency to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Dependency':
        """Create dependency from dictionary"""
        # Ensure all fields are lists
        processed_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                processed_data[key] = [v.strip() for v in value.split(',') if v.strip()]
            elif isinstance(value, list):
                processed_data[key] = value
            else:
                processed_data[key] = []
        return cls(**processed_data)


@dataclass
class PerformanceImpact:
    """Represents performance impact information"""
    cpu_impact: str = ""
    memory_impact: str = ""
    throughput_impact: str = ""
    latency_impact: str = ""
    capacity_impact: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert performance impact to dictionary"""
        return asdict(self)


@dataclass
class NetworkImpact:
    """Represents network impact information"""
    signaling_impact: str = ""
    throughput_impact: str = ""
    handover_impact: str = ""
    interference_impact: str = ""

    def to_dict(self) -> Dict[str, str]:
        """Convert network impact to dictionary"""
        return asdict(self)


@dataclass
class EricssonFeature:
    """
    Complete data model for Ericsson feature documentation

    This class represents all aspects of an Ericsson RAN feature including:
    - Feature identity (FAJ ID, name, CXC code)
    - Classification (value package, access type, node type)
    - Technical content (parameters, counters, events)
    - Operations (activation/deactivation steps)
    - Dependencies and relationships
    - Engineering guidelines
    - Performance and network impact
    """

    # Identity
    id: str = ""          # FAJ XXX XXXX
    name: str = ""
    cxc_code: Optional[str] = None

    # Classification
    value_package: str = ""
    value_package_id: str = ""
    access_type: str = ""
    node_type: str = ""

    # Content
    description: str = ""
    summary: str = ""

    # Technical Details
    parameters: List[Parameter] = field(default_factory=list)
    counters: List[Counter] = field(default_factory=list)
    events: List[Event] = field(default_factory=list)

    # Dependencies
    dependencies: Dependency = field(default_factory=Dependency)

    # Operations
    activation_step: Optional[str] = None
    deactivation_step: Optional[str] = None

    # Guidelines
    engineering_guidelines: str = ""

    # Metadata
    source_file: str = ""
    file_hash: str = ""
    processed_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Performance and Network Impact
    performance_impact: PerformanceImpact = field(default_factory=PerformanceImpact)
    network_impact: NetworkImpact = field(default_factory=NetworkImpact)

    def __post_init__(self):
        """Post-initialization validation and processing"""
        if self.id:
            self.id = self.normalize_faj_id(self.id)

        # Convert raw parameter dictionaries to Parameter objects
        if self.parameters and isinstance(self.parameters[0], dict):
            self.parameters = [Parameter.from_dict(p) for p in self.parameters]

        # Convert raw counter dictionaries to Counter objects
        if self.counters and isinstance(self.counters[0], dict):
            self.counters = [Counter.from_dict(c) for c in self.counters]

        # Convert raw event dictionaries to Event objects
        if self.events and isinstance(self.events[0], dict):
            self.events = [Event.from_dict(e) for e in self.events]

        # Convert dependencies dict to Dependency object if needed
        if isinstance(self.dependencies, dict):
            self.dependencies = Dependency.from_dict(self.dependencies)

    @staticmethod
    def normalize_faj_id(faj_id: str) -> str:
        """
        Normalize FAJ ID to standard format "FAJ XXX XXXX"

        Args:
            faj_id: Raw FAJ ID string

        Returns:
            Normalized FAJ ID string

        Examples:
            >>> normalize_faj_id("FAJ 1213094")
            'FAJ 121 3094'
            >>> normalize_faj_id("121 3094")
            'FAJ 121 3094'
            >>> normalize_faj_id("FAJ1213094")
            'FAJ 121 3094'
        """
        if not faj_id:
            return ""

        # Remove "FAJ" prefix if present and clean whitespace
        clean_id = re.sub(r'FAJ\s*', '', faj_id.strip())

        # Extract numbers using regex
        match = re.search(r'(\d{3})(\d{4})', clean_id.replace(' ', ''))
        if match:
            return f"FAJ {match.group(1)} {match.group(2)}"

        # Try to match "XXX XXXX" format
        match = re.search(r'(\d{3})\s*(\d{4})', clean_id)
        if match:
            return f"FAJ {match.group(1)} {match.group(2)}"

        return faj_id  # Return original if no pattern matches

    def validate_faj_id(self) -> Tuple[bool, str]:
        """
        Validate FAJ ID format

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.id:
            return False, "FAJ ID is empty"

        # Check if it matches the standard format
        pattern = r'^FAJ\s+\d{3}\s+\d{4}$'
        if not re.match(pattern, self.id):
            return False, f"FAJ ID '{self.id}' does not match format 'FAJ XXX XXXX'"

        return True, ""

    def validate(self) -> List[str]:
        """
        Validate the entire feature object

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate FAJ ID
        is_valid, error_msg = self.validate_faj_id()
        if not is_valid:
            errors.append(error_msg)

        # Validate required fields
        if not self.name.strip():
            errors.append("Feature name is required")

        if not self.description.strip():
            errors.append("Feature description is required")

        # Validate CXC code format if present
        if self.cxc_code:
            cxc_pattern = r'^CXC\d{7}$'
            if not re.match(cxc_pattern, self.cxc_code.strip()):
                errors.append(f"CXC code '{self.cxc_code}' does not match format 'CXXXXXXX'")

        # Validate access type
        valid_access_types = [atype.value for atype in AccessType]
        if self.access_type and self.access_type not in valid_access_types:
            errors.append(f"Invalid access type '{self.access_type}'. Valid types: {valid_access_types}")

        # Validate node type
        valid_node_types = [ntype.value for ntype in NodeType]
        if self.node_type and self.node_type not in valid_node_types:
            errors.append(f"Invalid node type '{self.node_type}'. Valid types: {valid_node_types}")

        return errors

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert feature to dictionary representation

        Returns:
            Dictionary representation of the feature
        """
        data = asdict(self)

        # Convert complex objects to dictionaries
        data['parameters'] = [p.to_dict() for p in self.parameters]
        data['counters'] = [c.to_dict() for c in self.counters]
        data['events'] = [e.to_dict() for e in self.events]
        data['dependencies'] = self.dependencies.to_dict()
        data['performance_impact'] = self.performance_impact.to_dict()
        data['network_impact'] = self.network_impact.to_dict()

        return data

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Convert feature to JSON string

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EricssonFeature':
        """
        Create feature from dictionary

        Args:
            data: Dictionary representation

        Returns:
            EricssonFeature instance
        """
        # Handle special conversions
        if 'parameters' in data and data['parameters']:
            data['parameters'] = [Parameter.from_dict(p) if isinstance(p, dict) else p for p in data['parameters']]

        if 'counters' in data and data['counters']:
            data['counters'] = [Counter.from_dict(c) if isinstance(c, dict) else c for c in data['counters']]

        if 'events' in data and data['events']:
            data['events'] = [Event.from_dict(e) if isinstance(e, dict) else e for e in data['events']]

        if 'dependencies' in data and isinstance(data['dependencies'], dict):
            data['dependencies'] = Dependency.from_dict(data['dependencies'])

        if 'performance_impact' in data and isinstance(data['performance_impact'], dict):
            data['performance_impact'] = PerformanceImpact(**data['performance_impact'])

        if 'network_impact' in data and isinstance(data['network_impact'], dict):
            data['network_impact'] = NetworkImpact(**data['network_impact'])

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> 'EricssonFeature':
        """
        Create feature from JSON string

        Args:
            json_str: JSON string representation

        Returns:
            EricssonFeature instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def get_searchable_text(self) -> str:
        """
        Get all searchable text from the feature

        Returns:
            Combined searchable text string
        """
        texts = [
            self.id,
            self.name,
            self.cxc_code or "",
            self.description,
            self.summary,
            self.engineering_guidelines
        ]

        # Add parameter names and descriptions
        for param in self.parameters:
            texts.extend([param.name, param.description])

        # Add counter names and descriptions
        for counter in self.counters:
            texts.extend([counter.name, counter.description])

        # Add event names and descriptions
        for event in self.events:
            texts.extend([event.name, event.description])

        return " ".join(filter(None, texts)).lower()

    def has_parameter(self, parameter_name: str) -> bool:
        """Check if feature has a specific parameter"""
        return any(p.name.lower() == parameter_name.lower() for p in self.parameters)

    def has_counter(self, counter_name: str) -> bool:
        """Check if feature has a specific counter"""
        return any(c.name.lower() == counter_name.lower() for c in self.counters)

    def has_event(self, event_name: str) -> bool:
        """Check if feature has a specific event"""
        return any(e.name.lower() == event_name.lower() for e in self.events)

    def get_parameter(self, parameter_name: str) -> Optional[Parameter]:
        """Get a specific parameter by name"""
        for param in self.parameters:
            if param.name.lower() == parameter_name.lower():
                return param
        return None

    def get_counter(self, counter_name: str) -> Optional[Counter]:
        """Get a specific counter by name"""
        for counter in self.counters:
            if counter.name.lower() == counter_name.lower():
                return counter
        return None

    def get_event(self, event_name: str) -> Optional[Event]:
        """Get a specific event by name"""
        for event in self.events:
            if event.name.lower() == event_name.lower():
                return event
        return None

    def add_parameter(self, parameter: Parameter):
        """Add a parameter to the feature"""
        if not self.has_parameter(parameter.name):
            self.parameters.append(parameter)

    def add_counter(self, counter: Counter):
        """Add a counter to the feature"""
        if not self.has_counter(counter.name):
            self.counters.append(counter)

    def add_event(self, event: Event):
        """Add an event to the feature"""
        if not self.has_event(event.name):
            self.events.append(event)

    def __str__(self) -> str:
        """String representation of the feature"""
        return f"EricssonFeature(id='{self.id}', name='{self.name}', cxc_code='{self.cxc_code}')"

    def __repr__(self) -> str:
        """Detailed string representation"""
        return (f"EricssonFeature(id='{self.id}', name='{self.name}', "
                f"cxc_code='{self.cxc_code}', parameters={len(self.parameters)}, "
                f"counters={len(self.counters)}, events={len(self.events)})")


# Utility functions for working with features
def validate_faj_format(faj_id: str) -> bool:
    """
    Utility function to validate FAJ ID format

    Args:
        faj_id: FAJ ID string to validate

    Returns:
        True if valid format, False otherwise
    """
    if not faj_id:
        return False

    # Normalize first
    normalized = EricssonFeature.normalize_faj_id(faj_id)
    pattern = r'^FAJ\s+\d{3}\s+\d{4}$'
    return bool(re.match(pattern, normalized))


def extract_faj_ids_from_text(text: str) -> List[str]:
    """
    Extract all FAJ IDs from text

    Args:
        text: Text to search for FAJ IDs

    Returns:
        List of normalized FAJ IDs found in text
    """
    patterns = [
        r'FAJ\s*(\d+\s+\d+)',      # FAJ 121 3094
        r'FAJ\s*(\d{3}\s*\d{4})',  # FAJ 1213094 or FAJ 121 3094
        r'(\d{3}\s*\d{4})',        # Just numbers 121 3094
    ]

    faj_ids = set()
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            normalized = EricssonFeature.normalize_faj_id(match)
            if validate_faj_format(normalized):
                faj_ids.add(normalized)

    return sorted(list(faj_ids))


def validate_cxc_code(cxc_code: str) -> bool:
    """
    Utility function to validate CXC code format

    Args:
        cxc_code: CXC code string to validate

    Returns:
        True if valid format, False otherwise
    """
    if not cxc_code:
        return False

    pattern = r'^CXC\d{7}$'
    return bool(re.match(pattern, cxc_code.strip().upper()))


# Factory functions for common operations
def create_feature_from_faj_id(faj_id: str, name: str = "", description: str = "") -> EricssonFeature:
    """
    Create a basic feature from FAJ ID

    Args:
        faj_id: FAJ ID string
        name: Feature name (optional)
        description: Feature description (optional)

    Returns:
        EricssonFeature instance with normalized FAJ ID
    """
    return EricssonFeature(
        id=EricssonFeature.normalize_faj_id(faj_id),
        name=name,
        description=description
    )


def create_parameter(name: str, mo_class: str, description: str = "", **kwargs) -> Parameter:
    """
    Factory function to create a parameter

    Args:
        name: Parameter name
        mo_class: MO class
        description: Parameter description
        **kwargs: Additional parameter attributes

    Returns:
        Parameter instance
    """
    return Parameter(
        name=name,
        mo_class=mo_class,
        description=description,
        **kwargs
    )


def create_counter(name: str, description: str = "", **kwargs) -> Counter:
    """
    Factory function to create a counter

    Args:
        name: Counter name
        description: Counter description
        **kwargs: Additional counter attributes

    Returns:
        Counter instance
    """
    return Counter(
        name=name,
        description=description,
        **kwargs
    )


def create_event(name: str, description: str = "", **kwargs) -> Event:
    """
    Factory function to create an event

    Args:
        name: Event name
        description: Event description
        **kwargs: Additional event attributes

    Returns:
        Event instance
    """
    return Event(
        name=name,
        description=description,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Ericsson Feature Data Models")
    print("=" * 40)

    # Test FAJ ID normalization
    test_faj_ids = ["FAJ 1213094", "121 3094", "FAJ1213094", "FAJ 121 3094"]
    print("FAJ ID Normalization Tests:")
    for faj_id in test_faj_ids:
        normalized = EricssonFeature.normalize_faj_id(faj_id)
        is_valid = validate_faj_format(normalized)
        print(f"  {faj_id:12} -> {normalized:12} (Valid: {is_valid})")

    # Create a sample feature
    feature = EricssonFeature(
        id="FAJ 121 3094",
        name="Sample Feature",
        cxc_code="CXC4011809",
        description="This is a sample Ericsson feature for testing",
        access_type="Licensed",
        node_type="DU"
    )

    # Add some parameters
    feature.add_parameter(create_parameter(
        name="sampleParam",
        mo_class="SampleMO",
        description="Sample parameter for testing",
        parameter_type="Integer",
        default_value="100"
    ))

    # Add some counters
    feature.add_counter(create_counter(
        name="sampleCounter",
        description="Sample counter for testing",
        counter_type="PM",
        unit="count"
    ))

    # Test validation
    print(f"\nFeature Validation:")
    errors = feature.validate()
    if errors:
        print("  Errors found:")
        for error in errors:
            print(f"    - {error}")
    else:
        print("  Feature is valid!")

    # Test JSON serialization
    print(f"\nJSON Serialization:")
    json_str = feature.to_json(indent=2)
    print(json_str[:200] + "..." if len(json_str) > 200 else json_str)

    # Test deserialization
    print(f"\nJSON Deserialization:")
    restored_feature = EricssonFeature.from_json(json_str)
    print(f"  Restored: {restored_feature}")
    print(f"  Parameters: {len(restored_feature.parameters)}")
    print(f"  Counters: {len(restored_feature.counters)}")