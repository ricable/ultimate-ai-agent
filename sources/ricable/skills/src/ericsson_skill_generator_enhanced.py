#!/usr/bin/env python3
"""
Enhanced Ericsson RAN Features Skill Generator
Comprehensive Claude skill creation system based on final-plan.md requirements

Creates professional Claude skills with:
- Complete skill structure and organization
- Comprehensive SKILL.md with capabilities and examples
- Categorized reference documentation
- Quick reference guides and troubleshooting
- Advanced search and navigation features
"""

import os
import json
import zipfile
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict, Counter
import sys
import shutil


class EnhancedEricssonSkillGenerator:
    """Enhanced Claude skill generator for Ericsson RAN features with comprehensive capabilities"""

    def __init__(self, data_dir: str, output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.skill_dir = self.output_dir / "ericsson"

        # Data structures
        self.features: Dict[str, Dict] = {}
        self.indices: Dict[str, Dict] = {}
        self.summary: Dict = {}

        # Enhanced statistics
        self.stats = {
            'total_features': 0,
            'total_parameters': 0,
            'total_counters': 0,
            'total_events': 0,
            'categories': {},
            'value_packages': {},
            'node_types': {},
            'access_types': {},
            'cxc_codes': 0,
            'files_with_guidelines': 0,
            'parameter_types': {},
            'counter_categories': {}
        }

        # Feature categorization rules
        self.category_rules = {
            'MIMO Features': ['mimo', 'multiple input multiple output', 'antenna', 'tx', 'rx'],
            'Energy Efficiency': ['sleep', 'energy', 'power', 'efficiency', 'saving', '节能'],
            'Carrier Aggregation': ['carrier aggregation', 'ca', 'component carrier', 'cc'],
            'Mobility Management': ['handover', 'mobility', 'handoff', '切换'],
            'Dual Connectivity': ['dual connectivity', 'dc', 'mr dc', 'en dc'],
            'QoS and Traffic Management': ['qos', 'quality of service', 'traffic', 'bearer'],
            'Performance Optimization': ['optimization', 'performance', 'throughput', 'capacity'],
            'Network Slicing': ['slicing', 'slice', 'network slice'],
            'Security': ['security', 'encryption', 'authentication', '安全'],
            'Features for Voice': ['voice', 'volte', 'vops', 'csfb'],
            'Features for IoT': ['iot', 'nb iot', 'cat m', 'emtc'],
            'Features for Critical Communications': ['mission critical', 'mcptt', 'mcvideo'],
            'Coverage Enhancement': ['coverage', 'extended coverage', 'coverage enhancement'],
            'Interference Management': ['interference', 'icic', 'comp', '干扰'],
            'Load Balancing': ['load balancing', 'load', 'balancing'],
            'Self-Organizing Networks': ['son', 'self organizing', 'self optimization'],
            'Other Features': []
        }

    def load_data(self):
        """Load all processed feature data with comprehensive validation"""
        print("📚 Loading processed feature data...")

        # Validate data directory
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Load features
        features_dir = self.data_dir / "features"
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        loaded_count = 0
        error_count = 0

        for feature_file in features_dir.glob("*.json"):
            try:
                feature_data = json.loads(feature_file.read_text())
                if 'id' in feature_data and feature_data['id']:
                    self.features[feature_data['id']] = feature_data
                    loaded_count += 1
                else:
                    print(f"⚠️  Warning: Feature {feature_file.name} missing ID")
                    error_count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Warning: Skipping corrupted file {feature_file}: {e}")
                error_count += 1

        print(f"✅ Loaded {loaded_count} features")
        if error_count > 0:
            print(f"⚠️  Encountered {error_count} errors during loading")

        self.stats['total_features'] = loaded_count

        # Load indices
        indices_dir = self.data_dir / "indices"
        if indices_dir.exists():
            for index_file in indices_dir.glob("*_index.json"):
                try:
                    index_name = index_file.stem.replace('_index', '')
                    self.indices[index_name] = json.loads(index_file.read_text())
                    print(f"  📊 Loaded {index_name} index with {len(self.indices[index_name])} entries")
                except (json.JSONDecodeError) as e:
                    print(f"⚠️  Warning: Skipping corrupted index {index_file}: {e}")

        # Load summary
        summary_file = self.data_dir / "summary.json"
        if summary_file.exists():
            try:
                self.summary = json.loads(summary_file.read_text())
                print(f"📈 Loaded processing summary")
            except json.JSONDecodeError as e:
                print(f"⚠️  Warning: Could not load summary: {e}")
                self.summary = {}

        # Ensure summary has required fields
        self._ensure_summary_completeness()

        # Calculate comprehensive statistics
        self._calculate_comprehensive_statistics()

    def _ensure_summary_completeness(self):
        """Ensure summary has all required fields with calculated defaults"""
        defaults = {
            'total_features': len(self.features),
            'total_parameters': sum(len(f.get('parameters', [])) for f in self.features.values()),
            'total_counters': sum(len(f.get('counters', [])) for f in self.features.values()),
            'total_events': sum(len(f.get('events', [])) for f in self.features.values()),
            'processing_time': 'Unknown',
            'source_files': len(set(f.get('source_file', '') for f in self.features.values())),
            'processed_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        for key, default_value in defaults.items():
            self.summary.setdefault(key, default_value)

    def _calculate_comprehensive_statistics(self):
        """Calculate comprehensive statistics from loaded data"""
        print("📊 Calculating comprehensive statistics...")

        # Reset statistics
        total_params = 0
        total_counters = 0
        total_events = 0
        categories = defaultdict(int)
        value_packages = defaultdict(int)
        node_types = defaultdict(int)
        access_types = defaultdict(int)
        parameter_types = defaultdict(int)
        counter_categories = defaultdict(int)
        cxc_codes = set()
        files_with_guidelines = 0

        for feature in self.features.values():
            # Count technical elements
            total_params += len(feature.get('parameters', []))
            total_counters += len(feature.get('counters', []))
            total_events += len(feature.get('events', []))

            # Count CXC codes
            if feature.get('cxc_code'):
                cxc_codes.add(feature['cxc_code'])

            # Count files with guidelines
            if feature.get('engineering_guidelines') and feature['engineering_guidelines'].strip():
                files_with_guidelines += 1

            # Count categories
            category = self.categorize_feature(feature)
            categories[category] += 1

            # Count value packages
            vp = feature.get('value_package', 'Unknown')
            value_packages[vp] += 1

            # Count node types
            nt = feature.get('node_type', 'Unknown')
            node_types[nt] += 1

            # Count access types
            at = feature.get('access_type', 'Unknown')
            access_types[at] += 1

            # Count parameter types
            for param in feature.get('parameters', []):
                param_type = param.get('type', 'Unknown')
                parameter_types[param_type] += 1

            # Count counter categories
            for counter in feature.get('counters', []):
                counter_cat = counter.get('category', 'Unknown')
                counter_categories[counter_cat] += 1

        # Update statistics
        self.stats.update({
            'total_parameters': total_params,
            'total_counters': total_counters,
            'total_events': total_events,
            'categories': dict(categories),
            'value_packages': dict(value_packages),
            'node_types': dict(node_types),
            'access_types': dict(access_types),
            'cxc_codes': len(cxc_codes),
            'files_with_guidelines': files_with_guidelines,
            'parameter_types': dict(parameter_types),
            'counter_categories': dict(counter_categories)
        })

        print(f"  📈 {total_params} parameters, {total_counters} counters, {total_events} events")
        print(f"  📂 {len(categories)} categories, {len(value_packages)} value packages")
        print(f"  🔧 {len(cxc_codes)} CXC codes, {files_with_guidelines} files with guidelines")

    def generate_skill(self):
        """Generate complete Claude skill with all components"""
        print("\n🚀 Generating Enhanced Claude Skill for Ericsson RAN Features")
        print("=" * 60)

        # Load data first
        self.load_data()

        if not self.features:
            print("❌ No features loaded. Cannot generate skill.")
            return None

        # Create skill structure
        self.create_enhanced_skill_structure()

        # Generate comprehensive SKILL.md
        self.create_comprehensive_skill_md()

        # Generate all reference files
        self.generate_comprehensive_references()

        # Generate advanced features
        self.generate_advanced_features()

        # Package skill
        package_stats = self.package_skill()

        # Generate validation report
        self.generate_validation_report(package_stats)

        print("\n✅ Enhanced skill generation complete!")
        return package_stats

    def create_enhanced_skill_structure(self):
        """Create comprehensive skill directory structure"""
        print("📁 Creating enhanced skill structure...")

        # Create reference directories with full organization
        refs_dir = self.skill_dir / "references"
        directories = [
            refs_dir,

            # Feature organization
            refs_dir / "features",
            refs_dir / "features" / "by_category",
            refs_dir / "features" / "by_package",
            refs_dir / "features" / "by_access_type",
            refs_dir / "features" / "by_node_type",
            refs_dir / "features" / "samples",

            # Technical references
            refs_dir / "parameters",
            refs_dir / "parameters" / "by_mo_class",
            refs_dir / "parameters" / "by_type",
            refs_dir / "counters",
            refs_dir / "counters" / "by_category",
            refs_dir / "events",
            refs_dir / "cxc_codes",

            # Guidance and support
            refs_dir / "guidelines",
            refs_dir / "guidelines" / "by_category",
            refs_dir / "best_practices",
            refs_dir / "troubleshooting",

            # Quick references
            refs_dir / "quick_reference",
            refs_dir / "quick_reference" / "common_tasks",
            refs_dir / "quick_reference" / "activation_guides",

            # Search and navigation
            refs_dir / "search",
            refs_dir / "search" / "indices",
            refs_dir / "navigation"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print("✅ Enhanced directory structure created")

    def create_comprehensive_skill_md(self):
        """Create comprehensive main SKILL.md file"""
        print("📝 Creating comprehensive SKILL.md...")

        # Build feature examples
        feature_examples = self._build_feature_examples()

        # Build category overview
        category_overview = self._build_category_overview()

        skill_content = f"""# Ericsson RAN Features Expert System

## Overview
This comprehensive skill provides expert-level access to Ericsson Radio Access Network (RAN) features, enabling efficient network optimization, troubleshooting, and feature management through natural language interaction.

### Scope and Coverage
- **{self.stats['total_features']} Radio Features**: Complete documentation with technical specifications
- **{self.stats['total_parameters']} Configuration Parameters**: Detailed parameter descriptions and MO classes
- **{self.stats['total_counters']} Performance Counters**: KPI explanations and monitoring guidance
- **{self.stats['total_events']} System Events**: Event definitions and troubleshooting information
- **{self.stats['cxc_codes']} CXC Feature Codes**: Activation and deactivation procedures
- **{self.stats['files_with_guidelines']} Engineering Guidelines**: Best practices and recommendations

### Technology Coverage
- **LTE**: 4G radio access features and optimizations
- **NR**: 5G New Radio features and capabilities
- **Dual Connectivity**: EN-DC and NR-DC implementations
- **Carrier Aggregation**: Multi-carrier configurations
- **IoT Support**: NB-IoT and CAT-M optimizations
- **Critical Communications**: Mission-critical voice and data

## When to Use This Skill

### Primary Use Cases
- **Feature Configuration**: "How do I configure MIMO Sleep Mode for energy saving?"
- **Troubleshooting**: "Why is pmMimoSleepTime counter showing high values?"
- **Network Planning**: "Which features should I enable for capacity optimization?"
- **Feature Activation**: "Show me the activation steps for CXC4011808"
- **Performance Analysis**: "What are the KPIs for carrier aggregation features?"
- **Compatibility Checking**: "Do MIMO Sleep Mode and Carrier Aggregation conflict?"

### Expert-Level Queries
- **Architecture Design**: "Design an energy-efficient configuration for a urban macro cell"
- **Impact Assessment**: "What is the network impact of enabling all energy saving features?"
- **Optimization Strategies**: "Recommended settings for high-density urban deployment"
- **Migration Planning**: "Steps to migrate from LTE-only to LTE-NR dual connectivity"

## Core Capabilities

### 1. Feature Information Management
**Basic Lookup**
- Get complete feature description: "Tell me about FAJ 121 3094"
- Find features by category: "Show all MIMO features"
- Search by parameter: "Which features use MimoSleepFunction?"
- CXC code lookup: "What is CXC4011808?"

**Advanced Analysis**
- Feature comparison: "Compare MIMO Sleep Mode vs Cell Sleep Mode"
- Dependency mapping: "What are the prerequisites for Carrier Aggregation?"
- Impact assessment: "How does feature X affect network performance?"
- Compatibility analysis: "Can features A and B work together?"

### 2. Technical Configuration Support
**Parameter Management**
- Parameter lookup: "What does MimoSleepFunction.sleepMode control?"
- MO class navigation: "Show all parameters in MimoSleepFunction MO"
- Configuration guidance: "Recommended settings for energy saving features"
- Validation: "Are these parameter settings valid for feature X?"

**Activation Procedures**
- Step-by-step activation: "How to activate CXC4011808?"
- Prerequisite checking: "What do I need before activating feature Y?"
- Deactivation procedures: "Safe deactivation steps for MIMO features"
- Rollback procedures: "How to rollback feature activation"

### 3. Performance Monitoring and KPIs
**Counter Analysis**
- Counter explanations: "Explain pmMimoSleepTime counter behavior"
- KPI relationships: "How do MIMO counters relate to energy saving?"
- Threshold guidance: "Recommended thresholds for sleep mode counters"
- Troubleshooting: "High pmTxOffRatio counter - possible causes"

**Performance Optimization**
- Optimization strategies: "How to optimize MIMO Sleep Mode for my network?"
- Capacity planning: "Feature recommendations for high-traffic cells"
- Energy efficiency: "Best configuration for maximum energy saving"
- Quality trade-offs: "Balancing energy saving vs. user experience"

### 4. Engineering Support
**Best Practices**
- Configuration guidelines: "How should I configure MIMO Sleep Mode?"
- Deployment recommendations: "Recommended rollout strategy for new features"
- Safety procedures: "Precautions when modifying radio parameters"
- Documentation references: "Where can I find detailed technical specs?"

**Troubleshooting**
- Issue diagnosis: "MIMO Sleep Mode not activating - troubleshooting steps"
- Performance issues: "Poor throughput after feature activation"
- Error analysis: "Common errors and their solutions"
- Root cause analysis: "Feature interaction problems"

## Feature Categories Overview
{category_overview}

## Quick Reference

### Common Activation Patterns
**Basic Feature Activation**
1. Verify prerequisites and compatibility
2. Configure required parameters
3. Set FeatureState.featureState to ACTIVATED
4. Monitor performance counters
5. Validate feature behavior

**Energy Saving Features**
- MIMO Sleep Mode (CXC4011808): Antenna configuration optimization
- Cell Sleep Mode: Complete cell shutdown during low traffic
- Power Saving: Dynamic power adjustment based on load

**Performance Features**
- Carrier Aggregation: Multi-carrier bandwidth combination
- MIMO Optimization: Multi-antenna configuration enhancement
- Load Balancing: Traffic distribution across cells

### Access Patterns
- **FAJ ID Format**: FAJ XXX XXXX (e.g., FAJ 121 3094)
- **CXC Code Format**: CXC followed by numbers (e.g., CXC4011808)
- **Parameter Format**: MOClass.parameterName
- **Counter Format**: pmCounterName
- **Event Format**: EVENT_NAME or INTERNAL_EVENT_NAME

## Reference Documentation Structure

### Primary References
- `references/features/` - Complete feature documentation organized by category
- `references/parameters/` - Parameter master index by MO class and type
- `references/counters/` - Performance counter reference with explanations
- `references/cxc_codes/` - Activation code index with procedures

### Support Documentation
- `references/guidelines/` - Engineering guidelines and best practices
- `references/troubleshooting/` - Common issues and solutions
- `references/quick_reference/` - Frequently used configurations and procedures

### Search and Navigation
- `references/search/indices/` - Comprehensive search indices
- `references/navigation/` - Cross-references and relationships

## Usage Examples

### Example 1: Feature Information Query
**User**: "Tell me about MIMO Sleep Mode feature"
**Response**: Provides complete feature details including:
- FAJ ID: 121 3094
- CXC Code: CXC4011808
- Description and purpose
- Technical parameters (19 parameters)
- Performance counters (19 counters)
- Activation/deactivation procedures
- Engineering guidelines and best practices

### Example 2: Configuration Assistance
**User**: "How should I configure MIMO Sleep Mode for an urban cell?"
**Response**: Provides detailed configuration guidance:
- Recommended parameter settings
- Threshold configurations based on traffic patterns
- MO class parameter explanations
- Validation procedures
- Performance monitoring setup

### Example 3: Troubleshooting
**User**: "MIMO Sleep Mode is not activating, what should I check?"
**Response**: Systematic troubleshooting approach:
- Prerequisite verification checklist
- Parameter validation steps
- Common configuration errors
- Event monitoring guidance
- Counter interpretation for diagnosis

### Example 4: Impact Analysis
**User**: "What is the impact of enabling MIMO Sleep Mode on network performance?"
**Response**: Comprehensive impact assessment:
- Energy saving benefits and expectations
- Potential performance trade-offs
- User experience considerations
- Network KPI effects
- Mitigation strategies if needed

## Advanced Features

### 1. Relationship Mapping
- Feature dependency visualization
- Parameter interaction analysis
- Conflict detection and resolution
- Compatibility matrices

### 2. Configuration Validation
- Parameter consistency checking
- Feature compatibility verification
- Performance impact prediction
- Rollback capability assessment

### 3. Optimization Recommendations
- AI-driven configuration suggestions
- Network-specific optimization strategies
- Performance tuning guidance
- Capacity planning assistance

## Expert Tips

### For Network Engineers
- Always verify feature compatibility with your specific node type and software version
- Use engineering guidelines as baseline, then tune based on network characteristics
- Monitor performance counters for at least 24 hours after feature activation
- Document any deviations from recommended configurations

### For Optimization Specialists
- Start with conservative settings, then gradually optimize based on performance data
- Consider user experience metrics alongside energy saving benefits
- Use feature interaction analysis to avoid conflicts
- Establish baseline measurements before feature deployment

### For Troubleshooting
- Use the systematic approach: verify → configure → activate → monitor → validate
- Check event logs for real-time indication of feature behavior
- Correlate counter changes with configuration modifications
- Document all changes for audit and rollback purposes

## Important Notes

### Limitations and Considerations
- This skill contains documentation for Ericsson Radio System features
- Feature availability depends on software version and hardware capabilities
- Always test in lab environment before network deployment
- Consider network-specific requirements and constraints
- Some features may require license activation

### Safety and Best Practices
- Always backup configurations before making changes
- Verify feature prerequisites and dependencies
- Monitor network performance after activation
- Have rollback procedures ready
- Consult official Ericsson documentation for detailed technical specifications

## Technical Specifications

### Data Coverage
- **Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Source Files**: {self.summary.get('source_files', 'Unknown')}
- **Processing Date**: {self.summary.get('processed_at', 'Unknown')}
- **Feature Completeness**: {len(self.features)} features processed

### Quality Metrics
- **Features with Guidelines**: {self.stats['files_with_guidelines']}/{self.stats['total_features']}
- **Parameters with Descriptions**: {sum(1 for f in self.features.values() for p in f.get('parameters', []) if p.get('description'))}
- **Counters with Explanations**: {sum(1 for f in self.features.values() for c in f.get('counters', []) if c.get('description'))}

---

*This skill is designed for Ericsson RAN professionals and requires knowledge of radio network concepts and Ericsson product terminology. Always consult official Ericsson documentation for critical network operations.*
"""

        skill_file = self.skill_dir / "SKILL.md"
        skill_file.write_text(skill_content)

        print("✅ Comprehensive SKILL.md created")

    def _build_feature_examples(self) -> str:
        """Build feature examples for SKILL.md"""
        if not self.features:
            return ""

        examples = []
        # Get diverse examples
        sample_features = list(self.features.values())[:5]

        for feature in sample_features:
            examples.append(f"- **{feature['name']}** (FAJ {feature['id']})")
            if feature.get('cxc_code'):
                examples.append(f"  - CXC Code: {feature['cxc_code']}")
            examples.append(f"  - Category: {self.categorize_feature(feature)}")
            examples.append(f"  - Parameters: {len(feature.get('parameters', []))}")
            examples.append("")

        return "\n".join(examples)

    def _build_category_overview(self) -> str:
        """Build category overview for SKILL.md"""
        if not self.stats['categories']:
            return ""

        overview = []
        for category, count in sorted(self.stats['categories'].items(), key=lambda x: x[1], reverse=True):
            overview.append(f"**{category}**: {count} features")

        return "\n".join(overview)

    def generate_comprehensive_references(self):
        """Generate all comprehensive reference files"""
        print("📚 Generating comprehensive reference files...")

        # Feature references
        self.generate_feature_references()

        # Parameter references
        self.generate_parameter_references()

        # Counter references
        self.generate_counter_references()

        # Event references
        self.generate_event_references()

        # CXC code references
        self.generate_cxc_references()

        # Guidelines and best practices
        self.generate_guideline_references()

        print("✅ Comprehensive reference files generated")

    def generate_feature_references(self):
        """Generate comprehensive feature reference files"""
        print("  📄 Generating feature references...")

        refs_dir = self.skill_dir / "references" / "features"

        # Master feature index
        self._create_master_feature_index(refs_dir)

        # Features by category
        self._create_features_by_category(refs_dir)

        # Features by value package
        self._create_features_by_package(refs_dir)

        # Features by access type
        self._create_features_by_access_type(refs_dir)

        # Features by node type
        self._create_features_by_node_type(refs_dir)

        # Sample feature details
        self._create_sample_feature_details(refs_dir)

    def _create_master_feature_index(self, refs_dir: Path):
        """Create master feature index"""
        content = ["# Ericsson Radio Features Master Index\n"]
        content.append(f"**Total Features**: {len(self.features)}\n")
        content.append(f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Summary statistics
        content.append("## Summary Statistics\n")
        content.append(f"- **Total Parameters**: {self.stats['total_parameters']}")
        content.append(f"- **Total Counters**: {self.stats['total_counters']}")
        content.append(f"- **Total Events**: {self.stats['total_events']}")
        content.append(f"- **CXC Codes**: {self.stats['cxc_codes']}")
        content.append(f"- **Value Packages**: {len(self.stats['value_packages'])}")
        content.append(f"- **Node Types**: {len(self.stats['node_types'])}\n")

        # Features by category
        content.append("## Features by Category\n")
        for category, count in sorted(self.stats['categories'].items(), key=lambda x: x[1], reverse=True):
            content.append(f"- **{category}**: {count} features")
        content.append("")

        # Detailed listing
        content.append("## Feature Listings\n")

        for category in sorted(self.stats['categories'].keys()):
            category_features = [f for f in self.features.values() if self.categorize_feature(f) == category]
            if category_features:
                content.append(f"### {category} ({len(category_features)} features)\n")

                for feature in sorted(category_features, key=lambda x: x['name']):
                    line = f"- **{feature['name']}** (FAJ {feature['id']})"
                    if feature.get('cxc_code'):
                        line += f" - CXC {feature['cxc_code']}"
                    if feature.get('value_package'):
                        line += f" - {feature['value_package']}"
                    content.append(line)

                content.append("")

        # Write file
        (refs_dir / "index.md").write_text("\n".join(content))

    def _create_features_by_category(self, refs_dir: Path):
        """Create feature listings by category"""
        category_dir = refs_dir / "by_category"

        for category in self.stats['categories'].keys():
            category_features = [f for f in self.features.values() if self.categorize_feature(f) == category]

            content = [f"# {category}\n"]
            content.append(f"**Total Features**: {len(category_features)}\n")

            # Category overview
            content.append("## Category Overview\n")
            content.append(f"This category contains {len(category_features)} features related to {category.lower()}.")
            content.append("")

            # Feature listings
            content.append("## Features\n")

            for feature in sorted(category_features, key=lambda x: x['name']):
                content.append(f"### {feature['name']}\n")
                content.append(f"**FAJ ID**: FAJ {feature['id']}")

                if feature.get('cxc_code'):
                    content.append(f"**CXC Code**: {feature['cxc_code']}")

                content.append(f"**Access Type**: {feature.get('access_type', 'N/A')}")
                content.append(f"**Value Package**: {feature.get('value_package', 'N/A')}")
                content.append(f"**Node Type**: {feature.get('node_type', 'N/A')}")

                # Summary
                summary = feature.get('summary', feature.get('description', ''))
                if summary:
                    content.append(f"\n**Description**: {summary[:200]}{'...' if len(summary) > 200 else ''}")

                # Key statistics
                content.append(f"\n**Technical Details**:")
                content.append(f"- Parameters: {len(feature.get('parameters', []))}")
                content.append(f"- Counters: {len(feature.get('counters', []))}")
                content.append(f"- Events: {len(feature.get('events', []))}")

                if feature.get('activation_step'):
                    content.append(f"- Has activation procedure: Yes")

                content.append("")

            # Write category file
            filename = category.lower().replace(' ', '_').replace('&', 'and').replace('/', '_') + ".md"
            (category_dir / filename).write_text("\n".join(content))

    def _create_features_by_package(self, refs_dir: Path):
        """Create feature listings by value package"""
        package_dir = refs_dir / "by_package"

        for package, features in self.stats['value_packages'].items():
            package_features = [f for f in self.features.values() if f.get('value_package') == package]

            content = [f"# {package}\n"]
            content.append(f"**Total Features**: {len(package_features)}\n")

            # Package overview
            content.append("## Package Overview\n")
            content.append(f"This value package contains {len(package_features)} features.")
            content.append("")

            # Feature listings
            content.append("## Features in Package\n")

            for feature in sorted(package_features, key=lambda x: x['name']):
                content.append(f"- **{feature['name']}** (FAJ {feature['id']})")
                if feature.get('cxc_code'):
                    content.append(f"  - CXC: {feature['cxc_code']}")
                content.append(f"  - Access: {feature.get('access_type', 'N/A')}")
                content.append("")

            # Write package file
            filename = package.lower().replace(' ', '_').replace('/', '_') + ".md"
            (package_dir / filename).write_text("\n".join(content))

    def _create_features_by_access_type(self, refs_dir: Path):
        """Create feature listings by access type"""
        access_dir = refs_dir / "by_access_type"

        for access_type, features in self.stats['access_types'].items():
            access_features = [f for f in self.features.values() if f.get('access_type') == access_type]

            content = [f"# {access_type} Features\n"]
            content.append(f"**Total Features**: {len(access_features)}\n")

            for feature in sorted(access_features, key=lambda x: x['name']):
                content.append(f"- **{feature['name']}** (FAJ {feature['id']})")
                content.append(f"  - Package: {feature.get('value_package', 'N/A')}")
                content.append("")

            filename = access_type.lower().replace(' ', '_') + ".md"
            (access_dir / filename).write_text("\n".join(content))

    def _create_features_by_node_type(self, refs_dir: Path):
        """Create feature listings by node type"""
        node_dir = refs_dir / "by_node_type"

        for node_type, features in self.stats['node_types'].items():
            node_features = [f for f in self.features.values() if f.get('node_type') == node_type]

            content = [f"# {node_type} Features\n"]
            content.append(f"**Total Features**: {len(node_features)}\n")

            for feature in sorted(node_features, key=lambda x: x['name']):
                content.append(f"- **{feature['name']}** (FAJ {feature['id']})")
                content.append("")

            filename = node_type.lower().replace(' ', '_').replace('/', '_') + ".md"
            (node_dir / filename).write_text("\n".join(content))

    def _create_sample_feature_details(self, refs_dir: Path):
        """Create detailed sample feature files"""
        samples_dir = refs_dir / "samples"

        # Select diverse samples
        sample_features = list(self.features.values())[:10]

        for feature in sample_features:
            filename = f"FAJ_{feature['id'].replace(' ', '_')}.md"
            filepath = samples_dir / filename

            content = [f"# {feature['name']}\n"]
            content.append(f"**FAJ ID**: FAJ {feature['id']}")

            if feature.get('cxc_code'):
                content.append(f"**CXC Code**: {feature['cxc_code']}")

            content.append(f"**Access Type**: {feature.get('access_type', 'N/A')}")
            content.append(f"**Value Package**: {feature.get('value_package', 'N/A')}")
            content.append(f"**Node Type**: {feature.get('node_type', 'N/A')}")
            content.append(f"**Category**: {self.categorize_feature(feature)}\n")

            # Description
            description = feature.get('description', feature.get('summary', 'No description available'))
            content.append("## Description\n")
            content.append(f"{description}\n")

            # Activation/Deactivation
            if feature.get('activation_step') or feature.get('deactivation_step'):
                content.append("## Activation and Deactivation\n")

                if feature.get('activation_step'):
                    content.append("### Activation\n")
                    content.append("```bash")
                    content.append(feature['activation_step'])
                    content.append("```\n")

                if feature.get('deactivation_step'):
                    content.append("### Deactivation\n")
                    content.append("```bash")
                    content.append(feature['deactivation_step'])
                    content.append("```\n")

            # Parameters
            if feature.get('parameters'):
                content.append(f"## Parameters ({len(feature['parameters'])})\n")

                # Group by MO class
                mo_params = defaultdict(list)
                for param in feature['parameters']:
                    mo_class = param.get('mo_class', 'Unknown')
                    mo_params[mo_class].append(param)

                for mo_class, params in sorted(mo_params.items()):
                    content.append(f"### {mo_class}\n")

                    for param in params:
                        content.append(f"#### {param['name']}\n")
                        content.append(f"**Type**: {param.get('type', 'N/A')}")
                        content.append(f"**Description**: {param.get('description', 'N/A')}")
                        content.append("")

            # Counters
            if feature.get('counters'):
                content.append(f"## Performance Counters ({len(feature['counters'])})\n")

                # Group by category
                cat_counters = defaultdict(list)
                for counter in feature['counters']:
                    category = counter.get('category', 'Unknown')
                    cat_counters[category].append(counter)

                for category, counters in sorted(cat_counters.items()):
                    content.append(f"### {category}\n")

                    for counter in counters:
                        content.append(f"#### {counter['name']}\n")
                        content.append(f"**Description**: {counter.get('description', 'N/A')}")
                        content.append("")

            # Events
            if feature.get('events'):
                content.append(f"## Events ({len(feature['events'])})\n")

                for event in feature['events'][:10]:  # Limit to prevent too long files
                    content.append(f"### {event['name']}\n")
                    content.append(f"**Type**: {event.get('type', 'N/A')}")
                    content.append(f"**Description**: {event.get('description', 'N/A')}")
                    content.append("")

            # Engineering Guidelines
            if feature.get('engineering_guidelines'):
                content.append("## Engineering Guidelines\n")
                guidelines = feature['engineering_guidelines']
                if len(guidelines) > 1000:
                    guidelines = guidelines[:1000] + "..."
                content.append(guidelines)
                content.append("")

            # Dependencies
            if feature.get('dependencies'):
                deps = feature['dependencies']
                if deps.get('prerequisites') or deps.get('related') or deps.get('conflicts'):
                    content.append("## Dependencies\n")

                    if deps.get('prerequisites'):
                        content.append("### Prerequisites\n")
                        for prereq in deps['prerequisites']:
                            content.append(f"- {prereq}")
                        content.append("")

                    if deps.get('related'):
                        content.append("### Related Features\n")
                        for related in deps['related']:
                            content.append(f"- {related}")
                        content.append("")

                    if deps.get('conflicts'):
                        content.append("### Potential Conflicts\n")
                        for conflict in deps['conflicts']:
                            content.append(f"- {conflict}")
                        content.append("")

            # Metadata
            content.append("## Metadata\n")
            content.append(f"**Source File**: {feature.get('source_file', 'N/A')}")
            content.append(f"**Processed At**: {feature.get('processed_at', 'N/A')}")
            content.append(f"**File Hash**: {feature.get('file_hash', 'N/A')}")

            filepath.write_text("\n".join(content))

    def generate_parameter_references(self):
        """Generate comprehensive parameter reference files"""
        print("  🔧 Generating parameter references...")

        refs_dir = self.skill_dir / "references" / "parameters"

        # Master parameter index
        self._create_parameter_master_index(refs_dir)

        # Parameters by MO class
        self._create_parameters_by_mo_class(refs_dir)

        # Parameters by type
        self._create_parameters_by_type(refs_dir)

    def _create_parameter_master_index(self, refs_dir: Path):
        """Create master parameter index"""
        content = ["# Parameter Master Index\n"]
        content.append(f"**Total Parameters**: {self.stats['total_parameters']}\n")

        # Parameter type statistics
        content.append("## Parameter Types\n")
        for param_type, count in sorted(self.stats['parameter_types'].items(), key=lambda x: x[1], reverse=True):
            content.append(f"- **{param_type}**: {count} parameters")
        content.append("")

        # MO class overview
        mo_classes = defaultdict(list)
        for feature in self.features.values():
            for param in feature.get('parameters', []):
                mo_class = param.get('mo_class', 'Unknown')
                mo_classes[mo_class].append((param, feature))

        content.append("## Parameters by MO Class\n")
        for mo_class, params in sorted(mo_classes.items(), key=lambda x: len(x[1]), reverse=True):
            content.append(f"### {mo_class} ({len(params)} parameters)\n")

            # Show first 10 parameters
            for param, feature in sorted(params, key=lambda x: x[0]['name'])[:10]:
                content.append(f"- **{param['name']}**")
                content.append(f"  - Used in: {feature['name']} (FAJ {feature['id']})")
                content.append(f"  - Type: {param.get('type', 'N/A')}")

            if len(params) > 10:
                content.append(f"- ... and {len(params) - 10} more parameters")

            content.append("")

        (refs_dir / "index.md").write_text("\n".join(content))

    def _create_parameters_by_mo_class(self, refs_dir: Path):
        """Create parameter listings by MO class"""
        mo_dir = refs_dir / "by_mo_class"

        # Group parameters by MO class
        mo_classes = defaultdict(list)
        for feature in self.features.values():
            for param in feature.get('parameters', []):
                mo_class = param.get('mo_class', 'Unknown')
                mo_classes[mo_class].append((param, feature))

        for mo_class, params in mo_classes.items():
            content = [f"# {mo_class} Parameters\n"]
            content.append(f"**Total Parameters**: {len(params)}\n")

            # Parameter details
            content.append("## Parameter Details\n")

            for param, feature in sorted(params, key=lambda x: x[0]['name']):
                content.append(f"### {param['name']}\n")
                content.append(f"**Type**: {param.get('type', 'N/A')}")
                content.append(f"**Used in Feature**: {feature['name']} (FAJ {feature['id']})")
                content.append(f"**Description**: {param.get('description', 'N/A')}")
                content.append("")

            # Write MO class file
            filename = mo_class.replace(' ', '_') + ".md"
            (mo_dir / filename).write_text("\n".join(content))

    def _create_parameters_by_type(self, refs_dir: Path):
        """Create parameter listings by type"""
        type_dir = refs_dir / "by_type"

        for param_type, params in self.stats['parameter_types'].items():
            content = [f"# {param_type} Parameters\n"]
            content.append(f"**Total Parameters**: {params}\n")

            # Find parameters of this type
            type_params = []
            for feature in self.features.values():
                for param in feature.get('parameters', []):
                    if param.get('type') == param_type:
                        type_params.append((param, feature))

            for param, feature in sorted(type_params, key=lambda x: x[0]['name']):
                content.append(f"### {param['name']}\n")
                content.append(f"**MO Class**: {param.get('mo_class', 'N/A')}")
                content.append(f"**Feature**: {feature['name']} (FAJ {feature['id']})")
                content.append(f"**Description**: {param.get('description', 'N/A')}")
                content.append("")

            filename = param_type.lower().replace(' ', '_') + ".md"
            (type_dir / filename).write_text("\n".join(content))

    def generate_counter_references(self):
        """Generate comprehensive counter reference files"""
        print("  📊 Generating counter references...")

        refs_dir = self.skill_dir / "references" / "counters"

        # Master counter index
        self._create_counter_master_index(refs_dir)

        # Counters by category
        self._create_counters_by_category(refs_dir)

    def _create_counter_master_index(self, refs_dir: Path):
        """Create master counter index"""
        content = ["# Performance Counter Master Index\n"]
        content.append(f"**Total Counters**: {self.stats['total_counters']}\n")

        # Counter category statistics
        content.append("## Counter Categories\n")
        for category, count in sorted(self.stats['counter_categories'].items(), key=lambda x: x[1], reverse=True):
            content.append(f"- **{category}**: {count} counters")
        content.append("")

        # Detailed counter listings
        counters_by_name = defaultdict(list)
        for feature in self.features.values():
            for counter in feature.get('counters', []):
                counter_name = counter['name']
                counters_by_name[counter_name].append((counter, feature))

        content.append("## Counter Details\n")

        for counter_name, instances in sorted(counters_by_name.items()):
            content.append(f"### {counter_name}\n")
            content.append(f"**Used in {len(instances)} features**:\n")

            for counter, feature in instances:
                content.append(f"- {feature['name']} (FAJ {feature['id']})")
                content.append(f"  - Category: {counter.get('category', 'N/A')}")
                content.append(f"  - Description: {counter.get('description', 'N/A')}")

            content.append("")

        (refs_dir / "index.md").write_text("\n".join(content))

    def _create_counters_by_category(self, refs_dir: Path):
        """Create counter listings by category"""
        cat_dir = refs_dir / "by_category"

        for category, counters in self.stats['counter_categories'].items():
            content = [f"# {category} Counters\n"]
            content.append(f"**Total Counters**: {counters}\n")

            # Find counters in this category
            cat_counters = []
            for feature in self.features.values():
                for counter in feature.get('counters', []):
                    if counter.get('category') == category:
                        cat_counters.append((counter, feature))

            for counter, feature in sorted(cat_counters, key=lambda x: x[0]['name']):
                content.append(f"### {counter['name']}\n")
                content.append(f"**Feature**: {feature['name']} (FAJ {feature['id']})")
                content.append(f"**Description**: {counter.get('description', 'N/A')}")
                content.append("")

            filename = category.lower().replace(' ', '_') + ".md"
            (cat_dir / filename).write_text("\n".join(content))

    def generate_event_references(self):
        """Generate comprehensive event reference files"""
        print("  ⚡ Generating event references...")

        refs_dir = self.skill_dir / "references" / "events"

        content = ["# System Events Reference\n"]
        content.append(f"**Total Events**: {self.stats['total_events']}\n")

        # Group events by name
        events_by_name = defaultdict(list)
        for feature in self.features.values():
            for event in feature.get('events', []):
                event_name = event['name']
                events_by_name[event_name].append((event, feature))

        for event_name, instances in sorted(events_by_name.items()):
            content.append(f"## {event_name}\n")
            content.append(f"**Found in {len(instances)} features**:\n")

            for event, feature in instances:
                content.append(f"### {feature['name']} (FAJ {feature['id']})\n")
                content.append(f"**Type**: {event.get('type', 'N/A')}")
                content.append(f"**Description**: {event.get('description', 'N/A')}")
                content.append("")

        (refs_dir / "index.md").write_text("\n".join(content))

    def generate_cxc_references(self):
        """Generate comprehensive CXC code reference files"""
        print("  🎯 Generating CXC code references...")

        refs_dir = self.skill_dir / "references" / "cxc_codes"

        content = ["# CXC Feature Code Reference\n"]
        content.append(f"**Total CXC Codes**: {self.stats['cxc_codes']}\n")
        content.append("Quick reference for feature activation and deactivation codes.\n")

        # CXC code listings
        cxc_map = {}
        for feature in self.features.values():
            if feature.get('cxc_code'):
                cxc_map[feature['cxc_code']] = feature

        for cxc_code, feature in sorted(cxc_map.items()):
            content.append(f"## {cxc_code}\n")
            content.append(f"**Feature**: {feature['name']}")
            content.append(f"**FAJ ID**: FAJ {feature['id']}")
            content.append(f"**Access Type**: {feature.get('access_type', 'N/A')}")
            content.append(f"**Value Package**: {feature.get('value_package', 'N/A')}")
            content.append("")

            # Activation procedure
            if feature.get('activation_step'):
                content.append("### Activation Procedure\n")
                content.append("```bash")
                content.append(feature['activation_step'])
                content.append("```\n")

            # Deactivation procedure
            if feature.get('deactivation_step'):
                content.append("### Deactivation Procedure\n")
                content.append("```bash")
                content.append(feature['deactivation_step'])
                content.append("```\n")

            # Feature summary
            summary = feature.get('summary', feature.get('description', ''))
            if summary:
                content.append("### Feature Summary\n")
                content.append(f"{summary[:300]}{'...' if len(summary) > 300 else ''}\n")

        (refs_dir / "index.md").write_text("\n".join(content))

    def generate_guideline_references(self):
        """Generate comprehensive guideline references"""
        print("  📋 Generating guideline references...")

        refs_dir = self.skill_dir / "references" / "guidelines"

        # Master guidelines index
        self._create_guidelines_master_index(refs_dir)

        # Guidelines by category
        self._create_guidelines_by_category(refs_dir)

    def _create_guidelines_master_index(self, refs_dir: Path):
        """Create master guidelines index"""
        content = ["# Engineering Guidelines Master Index\n"]
        content.append(f"**Features with Guidelines**: {self.stats['files_with_guidelines']}/{self.stats['total_features']}\n")

        # Collect all guidelines
        all_guidelines = []
        for feature in self.features.values():
            if feature.get('engineering_guidelines') and feature['engineering_guidelines'].strip():
                all_guidelines.append((feature['name'], feature['engineering_guidelines'], feature))

        content.append(f"**Total Guidelines Documents**: {len(all_guidelines)}\n")

        # Guidelines by feature
        content.append("## Guidelines by Feature\n")

        for feature_name, guidelines, feature in sorted(all_guidelines, key=lambda x: x[0]):
            content.append(f"### {feature_name} (FAJ {feature['id']})\n")

            # Truncate very long guidelines
            if len(guidelines) > 500:
                guidelines = guidelines[:500] + "..."

            content.append(guidelines)
            content.append("")

        (refs_dir / "index.md").write_text("\n".join(content))

    def _create_guidelines_by_category(self, refs_dir: Path):
        """Create guidelines by category"""
        cat_dir = refs_dir / "by_category"

        # Group guidelines by category
        category_guidelines = defaultdict(list)
        for feature in self.features.values():
            if feature.get('engineering_guidelines') and feature['engineering_guidelines'].strip():
                category = self.categorize_feature(feature)
                category_guidelines[category].append((feature['name'], feature['engineering_guidelines'], feature))

        for category, guidelines in category_guidelines.items():
            content = [f"# {category} Guidelines\n"]
            content.append(f"**Total Guidelines**: {len(guidelines)}\n")

            for feature_name, guidelines_text, feature in sorted(guidelines, key=lambda x: x[0]):
                content.append(f"## {feature_name} (FAJ {feature['id']})\n")

                # Truncate very long guidelines
                if len(guidelines_text) > 800:
                    guidelines_text = guidelines_text[:800] + "..."

                content.append(guidelines_text)
                content.append("")

            filename = category.lower().replace(' ', '_').replace('&', 'and') + ".md"
            (cat_dir / filename).write_text("\n".join(content))

    def generate_advanced_features(self):
        """Generate advanced skill features"""
        print("🚀 Generating advanced features...")

        # Quick reference guides
        self.generate_quick_reference_guides()

        # Troubleshooting guides
        self.generate_troubleshooting_guides()

        # Best practices
        self.generate_best_practices()

        # Search indices
        self.generate_search_indices()

        # Navigation aids
        self.generate_navigation_aids()

    def generate_quick_reference_guides(self):
        """Generate quick reference guides"""
        print("  ⚡ Generating quick reference guides...")

        refs_dir = self.skill_dir / "references" / "quick_reference"

        # Common tasks guide
        self._create_common_tasks_guide(refs_dir)

        # Activation guides
        self._create_activation_guides(refs_dir)

    def _create_common_tasks_guide(self, refs_dir: Path):
        """Create common tasks guide"""
        content = ["# Common Tasks Quick Reference\n"]
        content.append("Quick reference for frequently performed tasks with Ericsson RAN features.\n")

        # Feature activation checklist
        content.append("## Feature Activation Checklist\n")
        content.append("### Before Activation\n")
        content.append("- [ ] Verify feature prerequisites")
        content.append("- [ ] Check feature compatibility")
        content.append("- [ ] Backup current configuration")
        content.append("- [ ] Review parameter recommendations")
        content.append("- [ ] Plan monitoring strategy\n")

        content.append("### Activation Steps\n")
        content.append("- [ ] Configure required parameters")
        content.append("- [ ] Set FeatureState.featureState to ACTIVATED")
        content.append("- [ ] Monitor activation events")
        content.append("- [ ] Verify feature behavior")
        content.append("- [ ] Check performance counters\n")

        content.append("### Post-Activation\n")
        content.append("- [ ] Monitor KPIs for 24+ hours")
        content.append("- [ ] Validate user experience impact")
        content.append("- [ ] Document any deviations")
        content.append("- [ ] Update network documentation\n")

        # Common parameter configurations
        content.append("## Common Parameter Configurations\n")

        # Find common parameter patterns
        common_params = defaultdict(int)
        for feature in self.features.values():
            for param in feature.get('parameters', []):
                common_params[param['name']] += 1

        # Show most common parameters
        for param_name, count in sorted(common_params.items(), key=lambda x: x[1], reverse=True)[:20]:
            content.append(f"### {param_name}\n")
            content.append(f"Used in {count} features\n")

        # Troubleshooting quick steps
        content.append("## Quick Troubleshooting Steps\n")
        content.append("### Feature Not Activating\n")
        content.append("1. Check CXC code is correct")
        content.append("2. Verify prerequisites are met")
        content.append("3. Check parameter values")
        content.append("4. Review event logs")
        content.append("5. Validate MO instance exists\n")

        content.append("### Unexpected Performance Impact\n")
        content.append("1. Review counter trends before/after activation")
        content.append("2. Check related feature interactions")
        content.append("3. Verify parameter configurations")
        content.append("4. Consider traffic pattern changes")
        content.append("5. Plan rollback if needed\n")

        (refs_dir / "common_tasks.md").write_text("\n".join(content))

    def _create_activation_guides(self, refs_dir: Path):
        """Create activation guides"""
        act_dir = refs_dir / "activation_guides"

        # General activation guide
        content = ["# Feature Activation Guide\n"]
        content.append("Comprehensive guide for activating Ericsson RAN features.\n")

        content.append("## Activation Process\n")
        content.append("### 1. Preparation Phase\n")
        content.append("- Review feature documentation")
        content.append("- Verify license requirements")
        content.append("- Check software/hardware compatibility")
        content.append("- Plan activation window")
        content.append("- Prepare rollback procedures\n")

        content.append("### 2. Configuration Phase\n")
        content.append("- Set required parameters")
        content.append("- Validate parameter consistency")
        content.append("- Verify MO instances exist")
        content.append("- Check prerequisite features\n")

        content.append("### 3. Activation Phase\n")
        content.append("```bash")
        content.append("# Standard activation command")
        content.append("Set the FeatureState.featureState attribute to ACTIVATED in the FeatureState=<CXC_CODE> MO instance.")
        content.append("```\n")

        content.append("### 4. Verification Phase\n")
        content.append("- Monitor activation events")
        content.append("- Check feature status")
        content.append("- Validate performance counters")
        content.append("- Confirm expected behavior\n")

        # Sample activation procedures
        content.append("## Sample Activation Procedures\n")

        # Get features with activation steps
        features_with_activation = [f for f in self.features.values() if f.get('activation_step')]

        for feature in features_with_activation[:5]:  # Show first 5 examples
            content.append(f"### {feature['name']}\n")
            content.append(f"**CXC Code**: {feature.get('cxc_code', 'N/A')}")
            content.append(f"**FAJ ID**: FAJ {feature['id']}\n")

            content.append("**Activation Command**:")
            content.append("```bash")
            content.append(feature['activation_step'])
            content.append("```\n")

        (act_dir / "activation_guide.md").write_text("\n".join(content))

    def generate_troubleshooting_guides(self):
        """Generate troubleshooting guides"""
        print("  🔧 Generating troubleshooting guides...")

        refs_dir = self.skill_dir / "references" / "troubleshooting"

        content = ["# Troubleshooting Guide\n"]
        content.append("Common issues and solutions for Ericsson RAN features.\n")

        # Common issues and solutions
        content.append("## Common Issues\n")

        content.append("### Feature Not Activating\n")
        content.append("**Possible Causes**:")
        content.append("- CXC code not found or incorrect")
        content.append("- Prerequisites not met")
        content.append("- Feature license missing")
        content.append("- MO instance doesn't exist")
        content.append("- Parameter conflicts\n")

        content.append("**Troubleshooting Steps**:")
        content.append("1. Verify CXC code in feature documentation")
        content.append("2. Check feature prerequisites are activated")
        content.append("3. Confirm license is available and valid")
        content.append("4. Validate MO instance exists: `moget -class FeatureState`")
        content.append("5. Review parameter configurations for conflicts\n")

        content.append("### Performance Degradation After Activation\n")
        content.append("**Possible Causes**:")
        content.append("- Parameter settings not optimized")
        content.append("- Feature conflicts with existing configuration")
        content.append("- Network conditions changed")
        content.append("- Hardware capacity limitations")
        content.append("- Feature interaction issues\n")

        content.append("**Troubleshooting Steps**:")
        content.append("1. Compare performance counters before/after activation")
        content.append("2. Review parameter recommendations in guidelines")
        content.append("3. Check for related feature conflicts")
        content.append("4. Analyze traffic patterns and network load")
        content.append("5. Consider partial rollback or parameter adjustment\n")

        content.append("### Counters Not Updating\n")
        content.append("**Possible Causes**:")
        content.append("- Feature not properly activated")
        content.append("- Counter collection not enabled")
        content.append("- Reporting interval too long")
        content.append("- PM collection configuration issue")
        content.append("- Feature not generating expected events\n")

        content.append("**Troubleshooting Steps**:")
        content.append("1. Verify feature activation status")
        content.append("2. Check PM configuration for counter collection")
        content.append("3. Review reporting intervals")
        content.append("4. Monitor events in real-time")
        content.append("5. Validate feature is generating expected activity\n")

        # Feature-specific troubleshooting
        content.append("## Feature-Specific Troubleshooting\n")

        # Group by category
        category_features = defaultdict(list)
        for feature in self.features.values():
            category = self.categorize_feature(feature)
            category_features[category].append(feature)

        for category, features in list(category_features.items())[:5]:  # Limit to top 5 categories
            content.append(f"### {category} Issues\n")

            for feature in features[:3]:  # Show top 3 features per category
                content.append(f"#### {feature['name']}\n")
                content.append(f"**CXC Code**: {feature.get('cxc_code', 'N/A')}")
                content.append(f"**Common Issues**:")
                content.append("- Review specific parameter configurations")
                content.append("- Check feature interactions within category")
                content.append("- Verify network conditions support feature")
                content.append("- Monitor relevant performance counters")
                content.append("")

        (refs_dir / "troubleshooting_guide.md").write_text("\n".join(content))

    def generate_best_practices(self):
        """Generate best practices guide"""
        print("  ✨ Generating best practices...")

        refs_dir = self.skill_dir / "references" / "best_practices"

        content = ["# Best Practices Guide\n"]
        content.append("Industry best practices for Ericsson RAN feature management.\n")

        # General best practices
        content.append("## General Best Practices\n")
        content.append("### Feature Deployment\n")
        content.append("- Always test in lab environment before network deployment")
        content.append("- Use phased rollout for critical features")
        content.append("- Monitor network performance for at least 24 hours after activation")
        content.append("- Document all configuration changes and deviations")
        content.append("- Maintain rollback procedures for all activated features\n")

        content.append("### Parameter Management\n")
        content.append("- Start with manufacturer recommended settings")
        content.append("- Adjust based on network-specific conditions")
        content.append("- Document reasons for parameter deviations")
        content.append("- Use parameter validation tools when available")
        content.append("- Regular review and optimization of parameter settings\n")

        content.append("### Performance Monitoring\n")
        content.append("- Establish baseline measurements before feature activation")
        content.append("- Monitor both network KPIs and feature-specific counters")
        content.append("- Set up alerting for critical counter thresholds")
        content.append("- Regular trend analysis to detect performance anomalies")
        content.append("- Correlate feature changes with network performance\n")

        # Category-specific best practices
        content.append("## Category-Specific Best Practices\n")

        # Energy Efficiency
        content.append("### Energy Efficiency Features\n")
        content.append("- Configure sleep thresholds based on traffic patterns")
        content.append("- Balance energy saving vs. user experience")
        content.append("- Monitor user-affecting KPIs closely")
        content.append("- Consider time-of-day traffic variations")
        content.append("- Coordinate with other energy-saving features\n")

        # MIMO Features
        content.append("### MIMO Features\n")
        content.append("- Optimize antenna configurations for cell layout")
        content.append("- Consider UE capabilities in configuration")
        content.append("- Monitor throughput and user experience metrics")
        content.append("- Balance capacity vs. coverage optimization")
        content.append("- Regular performance trend analysis\n")

        # Carrier Aggregation
        content.append("### Carrier Aggregation\n")
        content.append("- Verify UE capability support in target area")
        content.append("- Optimize component carrier configurations")
        content.append("- Monitor inter-band interference issues")
        content.append("- Consider backhaul capacity limitations")
        content.append("- Regular performance optimization of CA configurations\n")

        (refs_dir / "best_practices.md").write_text("\n".join(content))

    def generate_search_indices(self):
        """Generate comprehensive search indices"""
        print("  🔍 Generating search indices...")

        search_dir = self.skill_dir / "references" / "search" / "indices"

        # Feature name index
        self._create_feature_name_index(search_dir)

        # Parameter name index
        self._create_parameter_name_index(search_dir)

        # Counter name index
        self._create_counter_name_index(search_dir)

        # Cross-reference index
        self._create_cross_reference_index(search_dir)

    def _create_feature_name_index(self, search_dir: Path):
        """Create feature name search index"""
        content = ["# Feature Name Search Index\n"]
        content.append("Alphabetical index of all features for quick lookup.\n")

        # Group by first letter
        features_by_letter = defaultdict(list)
        for feature in self.features.values():
            first_letter = feature['name'][0].upper()
            features_by_letter[first_letter].append(feature)

        for letter in sorted(features_by_letter.keys()):
            content.append(f"## {letter}\n")

            for feature in sorted(features_by_letter[letter], key=lambda x: x['name']):
                content.append(f"- {feature['name']} (FAJ {feature['id']})")
                if feature.get('cxc_code'):
                    content.append(f"  - CXC: {feature['cxc_code']}")
                content.append(f"  - Category: {self.categorize_feature(feature)}")

            content.append("")

        (search_dir / "feature_names.md").write_text("\n".join(content))

    def _create_parameter_name_index(self, search_dir: Path):
        """Create parameter name search index"""
        content = ["# Parameter Name Search Index\n"]
        content.append("Alphabetical index of all parameters for quick lookup.\n")

        # Collect all parameters
        all_params = []
        for feature in self.features.values():
            for param in feature.get('parameters', []):
                all_params.append((param['name'], param, feature))

        # Group by first letter
        params_by_letter = defaultdict(list)
        for param_name, param, feature in all_params:
            first_letter = param_name[0].upper()
            params_by_letter[first_letter].append((param_name, param, feature))

        for letter in sorted(params_by_letter.keys()):
            content.append(f"## {letter}\n")

            for param_name, param, feature in sorted(params_by_letter[letter], key=lambda x: x[0]):
                content.append(f"- {param_name}")
                content.append(f"  - MO Class: {param.get('mo_class', 'N/A')}")
                content.append(f"  - Feature: {feature['name']} (FAJ {feature['id']})")

            content.append("")

        (search_dir / "parameter_names.md").write_text("\n".join(content))

    def _create_counter_name_index(self, search_dir: Path):
        """Create counter name search index"""
        content = ["# Counter Name Search Index\n"]
        content.append("Alphabetical index of all performance counters for quick lookup.\n")

        # Collect all counters
        all_counters = []
        for feature in self.features.values():
            for counter in feature.get('counters', []):
                all_counters.append((counter['name'], counter, feature))

        # Group by first letter
        counters_by_letter = defaultdict(list)
        for counter_name, counter, feature in all_counters:
            first_letter = counter_name[0].upper()
            counters_by_letter[first_letter].append((counter_name, counter, feature))

        for letter in sorted(counters_by_letter.keys()):
            content.append(f"## {letter}\n")

            for counter_name, counter, feature in sorted(counters_by_letter[letter], key=lambda x: x[0]):
                content.append(f"- {counter_name}")
                content.append(f"  - Category: {counter.get('category', 'N/A')}")
                content.append(f"  - Feature: {feature['name']} (FAJ {feature['id']})")

            content.append("")

        (search_dir / "counter_names.md").write_text("\n".join(content))

    def _create_cross_reference_index(self, search_dir: Path):
        """Create cross-reference index"""
        content = ["# Cross-Reference Index\n"]
        content.append("Cross-references between features, parameters, and counters.\n")

        # Feature relationships
        content.append("## Feature Relationships\n")

        # Find related features by shared parameters
        feature_param_map = defaultdict(list)
        for feature in self.features.values():
            for param in feature.get('parameters', []):
                param_name = param['name']
                feature_param_map[param_name].append(feature)

        content.append("### Features Sharing Parameters\n")
        for param_name, features in feature_param_map.items():
            if len(features) > 1:  # Only show shared parameters
                content.append(f"#### {param_name}\n")
                for feature in features:
                    content.append(f"- {feature['name']} (FAJ {feature['id']})")
                content.append("")

        # Parameter to features mapping
        content.append("## Parameter to Features Mapping\n")

        mo_param_features = defaultdict(lambda: defaultdict(list))
        for feature in self.features.values():
            for param in feature.get('parameters', []):
                mo_class = param.get('mo_class', 'Unknown')
                param_name = param['name']
                mo_param_features[mo_class][param_name].append(feature)

        for mo_class, params in sorted(mo_param_features.items()):
            content.append(f"### {mo_class}\n")
            for param_name, features in sorted(params.items()):
                content.append(f"#### {param_name}\n")
                content.append(f"Used in {len(features)} features:")
                for feature in features:
                    content.append(f"- {feature['name']} (FAJ {feature['id']})")
                content.append("")

        (search_dir / "cross_reference.md").write_text("\n".join(content))

    def generate_navigation_aids(self):
        """Generate navigation aids"""
        print("  🧭 Generating navigation aids...")

        nav_dir = self.skill_dir / "references" / "navigation"

        # Main navigation index
        content = ["# Navigation Index\n"]
        content.append("Quick navigation to all reference materials.\n")

        content.append("## Quick Links\n")
        content.append("- [Main Skill Overview](../SKILL.md)\n")

        content.append("## Feature References\n")
        content.append("- [Master Feature Index](../features/index.md)")
        content.append("- [Features by Category](../features/by_category/)")
        content.append("- [Features by Value Package](../features/by_package/)")
        content.append("- [Sample Feature Details](../features/samples/)\n")

        content.append("## Technical References\n")
        content.append("- [Parameter Master Index](../parameters/index.md)")
        content.append("- [Parameters by MO Class](../parameters/by_mo_class/)")
        content.append("- [Counter Master Index](../counters/index.md)")
        content.append("- [Event Reference](../events/index.md)")
        content.append("- [CXC Code Reference](../cxc_codes/index.md)\n")

        content.append("## Guidance and Support\n")
        content.append("- [Engineering Guidelines](../guidelines/index.md)")
        content.append("- [Best Practices](../best_practices/best_practices.md)")
        content.append("- [Troubleshooting Guide](../troubleshooting/troubleshooting_guide.md)\n")

        content.append("## Quick Reference\n")
        content.append("- [Common Tasks](../quick_reference/common_tasks.md)")
        content.append("- [Activation Guide](../quick_reference/activation_guides/activation_guide.md)\n")

        content.append("## Search and Discovery\n")
        content.append("- [Feature Name Index](../search/indices/feature_names.md)")
        content.append("- [Parameter Name Index](../search/indices/parameter_names.md)")
        content.append("- [Counter Name Index](../search/indices/counter_names.md)")
        content.append("- [Cross-Reference Index](../search/indices/cross_reference.md)\n")

        # Feature category quick navigation
        content.append("## Browse by Category\n")
        for category, count in sorted(self.stats['categories'].items(), key=lambda x: x[1], reverse=True):
            filename = category.lower().replace(' ', '_').replace('&', 'and').replace('/', '_') + ".md"
            content.append(f"- [{category}]({filename}) - {count} features")

        (nav_dir / "navigation_index.md").write_text("\n".join(content))

    def categorize_feature(self, feature: Dict) -> str:
        """Enhanced feature categorization with better logic"""
        name = feature['name'].lower()
        description = (feature.get('description', '') + ' ' + feature.get('summary', '')).lower()
        combined_text = f"{name} {description}"

        # Score-based categorization
        category_scores = defaultdict(int)

        for category, keywords in self.category_rules.items():
            for keyword in keywords:
                if keyword in combined_text:
                    # Higher score for name matches
                    if keyword in name:
                        category_scores[category] += 3
                    # Lower score for description matches
                    elif keyword in description:
                        category_scores[category] += 1

        # Return category with highest score, or Other Features if no match
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'Other Features'

    def package_skill(self) -> Dict[str, Any]:
        """Package skill into zip file with comprehensive statistics"""
        print("📦 Packaging enhanced skill...")

        zip_filename = f"ericsson_ran_features_skill_{len(self.features)}_features_enhanced.zip"
        zip_path = self.output_dir / zip_filename

        # Count files and calculate sizes
        file_count = 0
        total_size = 0

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in skill directory
            for root, dirs, files in os.walk(self.skill_dir):
                for file in files:
                    if not file.endswith('.backup'):  # Skip backup files
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(self.skill_dir)

                        # Add to zip
                        zipf.write(file_path, arcname)

                        # Count statistics
                        file_count += 1
                        total_size += file_path.stat().st_size

        # Calculate final statistics
        size_mb = total_size / (1024 * 1024)

        print(f"✅ Enhanced skill packaged: {zip_filename}")
        print(f"📊 Package size: {size_mb:.2f} MB")
        print(f"📄 Files included: {file_count}")
        print(f"🔧 Features: {len(self.features)}")
        print(f"⚙️  Parameters: {self.stats['total_parameters']}")
        print(f"📈 Counters: {self.stats['total_counters']}")
        print(f"⚡ Events: {self.stats['total_events']}")
        print(f"🎯 CXC Codes: {self.stats['cxc_codes']}")

        print(f"\nNext steps:")
        print(f"1. Upload {zip_filename} to Claude")
        print(f"2. Test with sample queries:")
        print(f"   - 'Tell me about MIMO Sleep Mode'")
        print(f"   - 'How do I activate CXC4011808?'")
        print(f"   - 'What are the energy saving features?'")
        print(f"3. Share with team for feedback")

        # Return comprehensive statistics
        return {
            'zip_file_name': zip_filename,
            'zip_size_mb': size_mb,
            'file_count': file_count,
            'features_count': len(self.features),
            'parameters_count': self.stats['total_parameters'],
            'counters_count': self.stats['total_counters'],
            'events_count': self.stats['total_events'],
            'cxc_codes_count': self.stats['cxc_codes'],
            'categories_count': len(self.stats['categories']),
            'value_packages_count': len(self.stats['value_packages']),
            'files_with_guidelines': self.stats['files_with_guidelines']
        }

    def generate_validation_report(self, package_stats: Dict[str, Any]):
        """Generate comprehensive validation report"""
        print("📋 Generating validation report...")

        report = ["# Ericsson RAN Features Skill Validation Report\n"]
        report.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        report.append("## Package Statistics\n")
        for key, value in package_stats.items():
            if key.endswith('_count'):
                formatted_key = key.replace('_count', '').replace('_', ' ').title()
                report.append(f"- **{formatted_key}**: {value:,}")
            elif key == 'zip_size_mb':
                report.append(f"- **Package Size**: {value:.2f} MB")
            elif key == 'zip_file_name':
                report.append(f"- **Filename**: {value}")

        report.append("\n## Data Quality Metrics\n")

        # Feature completeness
        features_with_cxc = sum(1 for f in self.features.values() if f.get('cxc_code'))
        features_with_activation = sum(1 for f in self.features.values() if f.get('activation_step'))
        features_with_params = sum(1 for f in self.features.values() if f.get('parameters'))
        features_with_counters = sum(1 for f in self.features.values() if f.get('counters'))

        report.append(f"- **Features with CXC Codes**: {features_with_cxc}/{len(self.features)} ({features_with_cxc/len(self.features)*100:.1f}%)")
        report.append(f"- **Features with Activation Steps**: {features_with_activation}/{len(self.features)} ({features_with_activation/len(self.features)*100:.1f}%)")
        report.append(f"- **Features with Parameters**: {features_with_params}/{len(self.features)} ({features_with_params/len(self.features)*100:.1f}%)")
        report.append(f"- **Features with Counters**: {features_with_counters}/{len(self.features)} ({features_with_counters/len(self.features)*100:.1f}%)")
        report.append(f"- **Features with Guidelines**: {self.stats['files_with_guidelines']}/{len(self.features)} ({self.stats['files_with_guidelines']/len(self.features)*100:.1f}%)")

        report.append("\n## Content Distribution\n")
        report.append("### By Category\n")
        for category, count in sorted(self.stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(self.features)) * 100
            report.append(f"- **{category}**: {count} features ({percentage:.1f}%)")

        report.append("\n### By Access Type\n")
        for access_type, count in sorted(self.stats['access_types'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.features)) * 100
            report.append(f"- **{access_type}**: {count} features ({percentage:.1f}%)")

        report.append("\n## Skill Structure Validation\n")

        # Verify expected files exist
        required_files = [
            "SKILL.md",
            "references/features/index.md",
            "references/parameters/index.md",
            "references/counters/index.md",
            "references/cxc_codes/index.md",
            "references/guidelines/index.md",
            "references/troubleshooting/troubleshooting_guide.md",
            "references/best_practices/best_practices.md",
            "references/quick_reference/common_tasks.md",
            "references/search/indices/feature_names.md"
        ]

        existing_files = 0
        for file_path in required_files:
            full_path = self.skill_dir / file_path
            if full_path.exists():
                existing_files += 1
                report.append(f"✅ {file_path}")
            else:
                report.append(f"❌ {file_path} - MISSING")

        report.append(f"\n**Structure Completeness**: {existing_files}/{len(required_files)} files ({existing_files/len(required_files)*100:.1f}%)")

        report.append("\n## Recommendations\n")

        if features_with_cxc / len(self.features) < 0.8:
            report.append("- ⚠️  Consider adding more CXC codes for better activation guidance")

        if self.stats['files_with_guidelines'] / len(self.features) < 0.5:
            report.append("- ⚠️  Consider adding more engineering guidelines for better support")

        if features_with_activation / len(self.features) < 0.7:
            report.append("- ⚠️  Consider adding more activation procedures for better usability")

        if package_stats['zip_size_mb'] > 50:
            report.append("- ⚠️  Package size is large, consider optimization for faster uploads")

        report.append("\n## Validation Status\n")

        validation_score = (
            (existing_files / len(required_files)) * 0.3 +
            (features_with_cxc / len(self.features)) * 0.2 +
            (self.stats['files_with_guidelines'] / len(self.features)) * 0.2 +
            (features_with_activation / len(self.features)) * 0.2 +
            (min(1, 50 / max(1, package_stats['zip_size_mb']))) * 0.1
        ) * 100

        if validation_score >= 90:
            status = "✅ EXCELLENT"
        elif validation_score >= 80:
            status = "✅ GOOD"
        elif validation_score >= 70:
            status = "⚠️  ACCEPTABLE"
        else:
            status = "❌ NEEDS IMPROVEMENT"

        report.append(f"**Overall Score**: {validation_score:.1f}%")
        report.append(f"**Status**: {status}")

        # Write validation report
        report_file = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_file.write_text("\n".join(report))

        print(f"✅ Validation report saved to {report_file}")
        print(f"📊 Overall validation score: {validation_score:.1f}% {status}")


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate enhanced Claude skill from Ericsson features')
    parser.add_argument('--data-dir', default='output/ericsson_data', help='Processed data directory')
    parser.add_argument('--output-dir', default='output', help='Output directory')

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("Please run ericsson_feature_processor.py first")
        sys.exit(1)

    # Generate enhanced skill
    generator = EnhancedEricssonSkillGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    try:
        package_stats = generator.generate_skill()
        if package_stats:
            print("\n🎉 Enhanced skill generation completed successfully!")
            print(f"📦 Skill package: {package_stats['zip_file_name']}")
            print(f"📊 Package size: {package_stats['zip_size_mb']:.2f} MB")
            print(f"📄 Total files: {package_stats['file_count']}")
        else:
            print("\n❌ Skill generation failed. Please check the error messages above.")
    except Exception as e:
        print(f"\n❌ Error during skill generation: {e}")
        sys.exit(1)