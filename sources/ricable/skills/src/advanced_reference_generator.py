#!/usr/bin/env python3
"""
Ericsson RAN Features - Advanced Reference Generation System
Comprehensive reference file generation and categorization for Claude AI skills

This module creates professional reference documentation with:
- Multi-dimensional categorization (technology, functionality, node type, access)
- Enhanced feature references with activation commands and parameters
- Parameter and counter references with MO classes
- Engineering guidelines and troubleshooting guides
- Navigation structure with cross-references
- Code examples and configuration snippets
- Performance optimization guides

Author: Claude Code Enhanced
Version: 2.0
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from datetime import datetime
from collections import defaultdict, Counter
import sys


class AdvancedReferenceGenerator:
    """Advanced reference generation system for Ericsson RAN Features"""

    def __init__(self, data_dir: str, output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.skill_dir = self.output_dir / "ericsson"
        self.refs_dir = self.skill_dir / "references"

        # Data structures
        self.features: Dict[str, Dict] = {}
        self.indices: Dict[str, Dict] = {}
        self.summary: Dict = {}

        # Categorization matrices
        self.technology_categories = {
            '5G_NR': ['nr', '5g', 'new radio', 'nr-cell', 'nrcell'],
            '4G_LTE': ['lte', '4g', 'eutran', 'eutrancell', 'lte-advanced'],
            'LTE_Advanced_Pro': ['lte-advanced pro', 'lte-a pro', '4.5g'],
            '3G_WCDMA': ['wcdma', '3g', 'utran', 'rnc', 'rbs'],
            '2G_GSM': ['gsm', '2g', 'bts', 'msc'],
            'Multi_Standard': ['dual connectivity', 'inter-rat', 'multi-standard']
        }

        self.functionality_categories = {
            'MIMO': ['mimo', 'multiple input', 'multiple output', 'beamforming', 'massive mimo'],
            'Energy_Efficiency': ['energy', 'sleep', 'power', 'efficiency', 'saving', 'green'],
            'Mobility_Management': ['handover', 'mobility', 'handoff', 'cell reselection', 'tracking'],
            'Capacity_Enhancement': ['capacity', 'throughput', 'carrier aggregation', 'qam', 'spectral'],
            'Coverage_Optimization': ['coverage', 'range', 'extension', 'boosting', 'tilt'],
            'Quality_of_Service': ['qos', 'quality', 'priority', 'latency', 'reliability'],
            'Network_Slicing': ['slicing', 'slice', 'network sharing', 'virtualization'],
            'Self_Optimization': ['son', 'self-optimizing', 'self-organization', 'anr'],
            'Security': ['security', 'encryption', 'authentication', 'key', 'protection']
        }

        self.node_type_mapping = {
            'DU': ['distributed unit', 'du', 'baseband'],
            'CU': ['centralized unit', 'cu', 'central unit'],
            'RBS': ['radio base station', 'rbs', 'base station'],
            'BSR': ['base station router', 'bsr'],
            'RNC': ['radio network controller', 'rnc'],
            'BSC': ['base station controller', 'bsc'],
            'Multi_Node': ['du', 'cu', 'distributed', 'centralized']
        }

        self.access_type_mapping = {
            'Licensed_Spectrum': ['licensed', 'paired', 'fdd'],
            'Unlicensed_Spectrum': ['unlicensed', 'license-exempt', 'wifi'],
            'Shared_Spectrum': ['shared', 'cbrs', 'sasa'],
            'Bundled_Access': ['bundled', 'multi-band', 'carrier aggregation']
        }

    def load_data(self):
        """Load all processed feature data"""
        print("📚 Loading processed feature data for reference generation...")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Load features
        features_dir = self.data_dir / "features"
        if not features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {features_dir}")

        loaded_count = 0
        for feature_file in features_dir.glob("*.json"):
            try:
                feature_data = json.loads(feature_file.read_text())
                if 'id' in feature_data:
                    self.features[feature_data['id']] = feature_data
                    loaded_count += 1
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️  Warning: Skipping corrupted file {feature_file}: {e}")

        print(f"✅ Loaded {loaded_count} features")

        # Load indices
        indices_dir = self.data_dir / "indices"
        if indices_dir.exists():
            for index_file in indices_dir.glob("*_index.json"):
                try:
                    index_name = index_file.stem.replace('_index', '')
                    self.indices[index_name] = json.loads(index_file.read_text())
                    print(f"  📊 Loaded {index_name} index")
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

    def categorize_feature_comprehensive(self, feature: Dict) -> Dict[str, str]:
        """Comprehensive multi-dimensional categorization"""
        name = feature.get('name', '').lower()
        description = feature.get('description', '').lower()
        summary = feature.get('summary', '').lower()
        access_type = feature.get('access_type', '').lower()
        node_type = feature.get('node_type', '').lower()

        combined_text = f"{name} {description} {summary} {access_type} {node_type}"

        categories = {}

        # Technology categorization
        categories['technology'] = self._categorize_by_keywords(
            combined_text, self.technology_categories, 'Multi_Standard'
        )

        # Functionality categorization (can be multiple)
        categories['functionality'] = self._categorize_by_keywords(
            combined_text, self.functionality_categories, 'Other'
        )

        # Node type categorization
        categories['node_type'] = self._categorize_by_keywords(
            combined_text, self.node_type_mapping, 'General_Node'
        )

        # Access type categorization
        categories['access_type'] = self._categorize_by_keywords(
            combined_text, self.access_type_mapping, 'Mixed_Access'
        )

        return categories

    def _categorize_by_keywords(self, text: str, categories: Dict[str, List[str]], default: str) -> str:
        """Categorize text by keyword matching"""
        scores = {}
        text_lower = text.lower()

        for category, keywords in categories.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    # Higher score for exact word matches
                    if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
                        score += 3
                    else:
                        score += 1
            scores[category] = score

        # Return category with highest score, or default if no matches
        best_category = max(scores, key=scores.get)
        return best_category if scores[best_category] > 0 else default

    def generate_comprehensive_references(self):
        """Generate all comprehensive reference files"""
        print("🚀 Generating comprehensive reference system...")

        # Create directory structure
        self._create_advanced_directory_structure()

        # Generate categorized references
        self._generate_technology_references()
        self._generate_functionality_references()
        self._generate_node_type_references()
        self._generate_access_type_references()

        # Generate enhanced feature references
        self._generate_enhanced_feature_references()

        # Generate specialized references
        self._generate_parameter_references()
        self._generate_counter_references()
        self._generate_cxc_references()

        # Generate guidance documentation
        self._generate_engineering_guidelines()
        self._generate_troubleshooting_guides()
        self._generate_best_practices()
        self._generate_performance_guides()

        # Generate navigation and quick reference
        self._generate_navigation_structure()
        self._generate_quick_references()

        print("✅ Comprehensive reference system generated")

    def _create_advanced_directory_structure(self):
        """Create advanced directory structure for references"""
        print("📁 Creating advanced directory structure...")

        directories = [
            self.refs_dir,
            # Categorized feature directories
            self.refs_dir / "features" / "technology",
            self.refs_dir / "features" / "functionality",
            self.refs_dir / "features" / "node_type",
            self.refs_dir / "features" / "access_type",
            self.refs_dir / "features" / "all_features",
            # Specialized references
            self.refs_dir / "parameters",
            self.refs_dir / "counters",
            self.refs_dir / "cxc_codes",
            self.refs_dir / "events",
            # Guidance and documentation
            self.refs_dir / "guidelines" / "engineering",
            self.refs_dir / "guidelines" / "troubleshooting",
            self.refs_dir / "guidelines" / "best_practices",
            self.refs_dir / "guidelines" / "performance",
            # Quick reference
            self.refs_dir / "quick_reference",
            self.refs_dir / "cheat_sheets",
            # Cross-reference indices
            self.refs_dir / "indices"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print("✅ Advanced directory structure created")

    def _generate_technology_references(self):
        """Generate technology-based feature references"""
        print("📱 Generating technology-based references...")

        tech_features = defaultdict(list)

        # Categorize all features by technology
        for feature in self.features.values():
            categories = self.categorize_feature_comprehensive(feature)
            tech = categories['technology']
            tech_features[tech].append(feature)

        # Generate technology index
        tech_index_content = "# Technology Categories\n\n"
        tech_index_content += f"Features categorized by radio technology standard.\n\n"

        for tech, features in sorted(tech_features.items()):
            tech_index_content += f"## {tech.replace('_', ' ')} ({len(features)} features)\n\n"

            # Create technology-specific file
            tech_content = f"# {tech.replace('_', ' ')} Features\n\n"
            tech_content += f"**Total Features**: {len(features)}\n\n"
            tech_content += "## Feature List\n\n"

            for feature in sorted(features, key=lambda x: x.get('name', '')):
                feature_id = feature['id']
                feature_name = feature.get('name', 'Unknown')
                cxc_code = feature.get('cxc_code', '')
                access_type = feature.get('access_type', 'N/A')

                tech_content += f"### {feature_name}\n"
                tech_content += f"- **FAJ ID**: FAJ {feature_id}\n"
                if cxc_code:
                    tech_content += f"- **CXC Code**: {cxc_code}\n"
                tech_content += f"- **Access Type**: {access_type}\n"
                tech_content += f"- **Description**: {feature.get('summary', 'No description')[:200]}...\n\n"

                tech_index_content += f"- [{feature_name}](technology/{tech.lower().replace(' ', '_')}.md#{feature_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}) (FAJ {feature_id})\n"

            # Write technology file
            tech_file = self.refs_dir / "features" / "technology" / f"{tech.lower().replace(' ', '_')}.md"
            tech_file.write_text(tech_content)

        # Write technology index
        index_file = self.refs_dir / "features" / "technology" / "index.md"
        index_file.write_text(tech_index_content)

        print(f"✅ Generated references for {len(tech_features)} technology categories")

    def _generate_functionality_references(self):
        """Generate functionality-based feature references"""
        print("⚙️ Generating functionality-based references...")

        func_features = defaultdict(list)

        # Categorize all features by functionality
        for feature in self.features.values():
            categories = self.categorize_feature_comprehensive(feature)
            func = categories['functionality']
            func_features[func].append(feature)

        # Generate functionality index
        func_index_content = "# Functionality Categories\n\n"
        func_index_content += "Features categorized by primary function and capability.\n\n"

        for func, features in sorted(func_features.items()):
            func_index_content += f"## {func.replace('_', ' ')} ({len(features)} features)\n\n"

            # Create functionality-specific file
            func_content = f"# {func.replace('_', ' ')} Features\n\n"
            func_content += f"**Total Features**: {len(features)}\n\n"

            # Add common configuration patterns for this functionality
            func_content += "## Common Configuration Patterns\n\n"

            # Extract common parameters from features in this category
            common_params = defaultdict(int)
            for feature in features:
                for param in feature.get('parameters', []):
                    param_name = param.get('name', '')
                    if '.' in param_name:  # Only count MO parameters
                        common_params[param_name] += 1

            # Show most common parameters
            top_params = sorted(common_params.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_params:
                func_content += "### Frequently Used Parameters\n\n"
                for param_name, count in top_params:
                    func_content += f"- **{param_name}** (used in {count} features)\n"
                func_content += "\n"

            func_content += "## Feature List\n\n"

            for feature in sorted(features, key=lambda x: x.get('name', '')):
                feature_id = feature['id']
                feature_name = feature.get('name', 'Unknown')
                cxc_code = feature.get('cxc_code', '')
                value_package = feature.get('value_package', 'N/A')

                func_content += f"### {feature_name}\n"
                func_content += f"- **FAJ ID**: FAJ {feature_id}\n"
                if cxc_code:
                    func_content += f"- **CXC Code**: {cxc_code}\n"
                func_content += f"- **Value Package**: {value_package}\n"
                func_content += f"- **Parameters**: {len(feature.get('parameters', []))}\n"
                func_content += f"- **Counters**: {len(feature.get('counters', []))}\n"

                # Add key benefit
                summary = feature.get('summary', '')
                if summary:
                    func_content += f"- **Key Benefit**: {summary[:150]}...\n"
                func_content += "\n"

                func_index_content += f"- [{feature_name}](functionality/{func.lower().replace(' ', '_')}.md#{feature_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}) (FAJ {feature_id})\n"

            # Write functionality file
            func_file = self.refs_dir / "features" / "functionality" / f"{func.lower().replace(' ', '_')}.md"
            func_file.write_text(func_content)

        # Write functionality index
        index_file = self.refs_dir / "features" / "functionality" / "index.md"
        index_file.write_text(func_index_content)

        print(f"✅ Generated references for {len(func_features)} functionality categories")

    def _generate_node_type_references(self):
        """Generate node type-based feature references"""
        print("🏗️ Generating node type-based references...")

        node_features = defaultdict(list)

        # Categorize all features by node type
        for feature in self.features.values():
            categories = self.categorize_feature_comprehensive(feature)
            node = categories['node_type']
            node_features[node].append(feature)

        # Generate node type index
        node_index_content = "# Node Type Categories\n\n"
        node_index_content += "Features categorized by target network node type.\n\n"

        for node, features in sorted(node_features.items()):
            node_index_content += f"## {node.replace('_', ' ')} ({len(features)} features)\n\n"

            # Create node type-specific file
            node_content = f"# {node.replace('_', ' ')} Features\n\n"
            node_content += f"**Total Features**: {len(features)}\n\n"

            # Add node-specific information
            node_content += "## Node Characteristics\n\n"

            if node == 'DU':
                node_content += "- Distributed Unit in 5G split architecture\n"
                node_content += "- Handles real-time radio functions\n"
                node_content += "- Focus on MIMO, beamforming, and radio processing\n\n"
            elif node == 'CU':
                node_content += "- Centralized Unit in 5G split architecture\n"
                node_content += "- Handles non-real-time functions\n"
                node_content += "- Focus on mobility, QoS, and packet processing\n\n"
            elif node == 'RBS':
                node_content += "- Radio Base Station for traditional deployments\n"
                node_content += "- Integrated baseband and radio functions\n"
                node_content += "- Supports multiple radio standards\n\n"

            node_content += "## Compatible Features\n\n"

            for feature in sorted(features, key=lambda x: x.get('name', '')):
                feature_id = feature['id']
                feature_name = feature.get('name', 'Unknown')
                cxc_code = feature.get('cxc_code', '')
                access_type = feature.get('access_type', 'N/A')

                node_content += f"### {feature_name}\n"
                node_content += f"- **FAJ ID**: FAJ {feature_id}\n"
                if cxc_code:
                    node_content += f"- **CXC Code**: {cxc_code}\n"
                node_content += f"- **Access Type**: {access_type}\n"
                node_content += f"- **Value Package**: {feature.get('value_package', 'N/A')}\n"

                # Add activation note if available
                if feature.get('activation_step'):
                    node_content += f"- **Activation**: Available via CXC code\n"
                node_content += "\n"

                node_index_content += f"- [{feature_name}](node_type/{node.lower().replace(' ', '_')}.md#{feature_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}) (FAJ {feature_id})\n"

            # Write node type file
            node_file = self.refs_dir / "features" / "node_type" / f"{node.lower().replace(' ', '_')}.md"
            node_file.write_text(node_content)

        # Write node type index
        index_file = self.refs_dir / "features" / "node_type" / "index.md"
        index_file.write_text(node_index_content)

        print(f"✅ Generated references for {len(node_features)} node type categories")

    def _generate_access_type_references(self):
        """Generate access type-based feature references"""
        print("📡 Generating access type-based references...")

        access_features = defaultdict(list)

        # Categorize all features by access type
        for feature in self.features.values():
            categories = self.categorize_feature_comprehensive(feature)
            access = categories['access_type']
            access_features[access].append(feature)

        # Generate access type index
        access_index_content = "# Access Type Categories\n\n"
        access_index_content += "Features categorized by spectrum access type.\n\n"

        for access, features in sorted(access_features.items()):
            access_index_content += f"## {access.replace('_', ' ')} ({len(features)} features)\n\n"

            # Create access type-specific file
            access_content = f"# {access.replace('_', ' ')} Features\n\n"
            access_content += f"**Total Features**: {len(features)}\n\n"

            # Add access type information
            access_content += "## Access Type Characteristics\n\n"

            if access == 'Licensed_Spectrum':
                access_content += "- Uses licensed frequency bands\n"
                access_content += "- Guaranteed quality of service\n"
                access_content += "- Primary carrier for mobile operators\n\n"
            elif access == 'Unlicensed_Spectrum':
                access_content += "- Uses unlicensed frequency bands\n"
                access_content += "- Shared access with other users\n"
                access_content += "- Typically for capacity augmentation\n\n"
            elif access == 'Shared_Spectrum':
                access_content += "- Uses shared spectrum frameworks\n"
                access_content += "- Coordinated access with priority tiers\n"
                access_content += "- Emerging access models\n\n"

            access_content += "## Compatible Features\n\n"

            for feature in sorted(features, key=lambda x: x.get('name', '')):
                feature_id = feature['id']
                feature_name = feature.get('name', 'Unknown')
                cxc_code = feature.get('cxc_code', '')
                technology = self.categorize_feature_comprehensive(feature)['technology']

                access_content += f"### {feature_name}\n"
                access_content += f"- **FAJ ID**: FAJ {feature_id}\n"
                if cxc_code:
                    access_content += f"- **CXC Code**: {cxc_code}\n"
                access_content += f"- **Technology**: {technology.replace('_', ' ')}\n"
                access_content += f"- **Node Type**: {feature.get('node_type', 'N/A')}\n"
                access_content += "\n"

                access_index_content += f"- [{feature_name}](access_type/{access.lower().replace(' ', '_')}.md#{feature_name.lower().replace(' ', '-').replace('(', '').replace(')', '')}) (FAJ {feature_id})\n"

            # Write access type file
            access_file = self.refs_dir / "features" / "access_type" / f"{access.lower().replace(' ', '_')}.md"
            access_file.write_text(access_content)

        # Write access type index
        index_file = self.refs_dir / "features" / "access_type" / "index.md"
        index_file.write_text(access_index_content)

        print(f"✅ Generated references for {len(access_features)} access type categories")

    def _generate_enhanced_feature_references(self):
        """Generate enhanced individual feature references"""
        print("📄 Generating enhanced feature references...")

        features_dir = self.refs_dir / "features" / "all_features"

        for feature_id, feature in self.features.items():
            categories = self.categorize_feature_comprehensive(feature)

            filename = f"FAJ_{feature_id.replace(' ', '_')}.md"
            filepath = features_dir / filename

            content = f"""# {feature.get('name', 'Unknown Feature')}

## Overview
**FAJ ID**: FAJ {feature_id}
**CXC Code**: {feature.get('cxc_code', 'N/A')}
**Value Package**: {feature.get('value_package', 'N/A')}
**Access Type**: {feature.get('access_type', 'N/A')}
**Node Type**: {feature.get('node_type', 'N/A')}

### Categories
- **Technology**: {categories['technology'].replace('_', ' ')}
- **Functionality**: {categories['functionality'].replace('_', ' ')}
- **Node Type**: {categories['node_type'].replace('_', ' ')}
- **Access Type**: {categories['access_type'].replace('_', ' ')}

## Description
{feature.get('summary', feature.get('description', 'No description available'))}

## Activation and Deactivation

### Activation
"""

            if feature.get('activation_step'):
                content += f"```bash\n{feature['activation_step']}\n```\n\n"
                content += "**Note**: Verify prerequisites before activation.\n\n"
            else:
                content += "Activation steps not documented in source.\n\n"

            content += "### Deactivation\n"
            if feature.get('deactivation_step'):
                content += f"```bash\n{feature['deactivation_step']}\n```\n\n"
            else:
                content += "Deactivation steps not documented in source.\n\n"

            # Parameters section
            parameters = feature.get('parameters', [])
            if parameters:
                content += f"## Parameters ({len(parameters)})\n\n"

                # Group parameters by MO class
                mo_params = defaultdict(list)
                for param in parameters:
                    mo_class = param.get('mo_class', 'Unknown')
                    mo_params[mo_class].append(param)

                for mo_class, params in sorted(mo_params.items()):
                    if mo_class != 'Unknown':
                        content += f"### {mo_class}\n\n"
                        for param in params:
                            param_name = param.get('name', '')
                            param_type = param.get('type', 'N/A')
                            param_desc = param.get('description', 'No description')

                            content += f"#### {param_name}\n"
                            content += f"- **Type**: {param_type}\n"
                            content += f"- **Description**: {param_desc}\n\n"

                # Show unknown parameters separately
                unknown_params = mo_params.get('Unknown', [])
                if unknown_params:
                    content += "### Other Parameters\n\n"
                    for param in unknown_params:
                        content += f"- **{param.get('name', 'Unknown')}**: {param.get('description', 'No description')}\n"
                    content += "\n"

            # Counters section
            counters = feature.get('counters', [])
            if counters:
                content += f"## Performance Counters ({len(counters)})\n\n"

                # Group counters by category
                counter_categories = defaultdict(list)
                for counter in counters:
                    category = counter.get('category', 'General')
                    counter_categories[category].append(counter)

                for category, category_counters in sorted(counter_categories.items()):
                    content += f"### {category}\n\n"
                    for counter in category_counters:
                        counter_name = counter.get('name', '')
                        counter_desc = counter.get('description', 'No description')
                        content += f"- **{counter_name}**: {counter_desc}\n"
                    content += "\n"

            # Events section
            events = feature.get('events', [])
            if events:
                content += f"## Events ({len(events)})\n\n"

                # Group events by name
                event_groups = defaultdict(list)
                for event in events:
                    event_name = event.get('name', '')
                    event_groups[event_name].append(event)

                for event_name, event_list in event_groups.items():
                    content += f"### {event_name}\n\n"
                    for event in event_list:
                        event_type = event.get('type', '')
                        event_desc = event.get('description', 'No description')
                        content += f"- **{event_type}**: {event_desc}\n"
                    content += "\n"

            # Dependencies section
            dependencies = feature.get('dependencies', {})
            if any(dependencies.values()):
                content += "## Dependencies\n\n"

                if dependencies.get('prerequisites'):
                    content += "### Prerequisites\n"
                    for prereq in dependencies['prerequisites']:
                        content += f"- {prereq}\n"
                    content += "\n"

                if dependencies.get('related'):
                    content += "### Related Features\n"
                    for related in dependencies['related']:
                        content += f"- {related}\n"
                    content += "\n"

                if dependencies.get('conflicts'):
                    content += "### Conflicts\n"
                    for conflict in dependencies['conflicts']:
                        content += f"- {conflict}\n"
                    content += "\n"

            # Engineering guidelines
            guidelines = feature.get('engineering_guidelines', '')
            if guidelines:
                content += "## Engineering Guidelines\n\n"
                # Limit length for readability
                if len(guidelines) > 1000:
                    content += guidelines[:1000] + "...\n\n"
                    content += "*Full guidelines available in source documentation.*\n\n"
                else:
                    content += guidelines + "\n\n"

            # Cross-references
            content += "## Cross-References\n\n"
            content += f"- **Technology View**: See [{categories['technology'].replace('_', ' ')} Features](../technology/{categories['technology'].lower().replace(' ', '_')}.md)\n"
            content += f"- **Functionality View**: See [{categories['functionality'].replace('_', ' ')} Features](../functionality/{categories['functionality'].lower().replace(' ', '_')}.md)\n"
            content += f"- **Node Type View**: See [{categories['node_type'].replace('_', ' ')} Features](../node_type/{categories['node_type'].lower().replace(' ', '_')}.md)\n"

            if feature.get('cxc_code'):
                content += f"- **CXC Reference**: See [CXC {feature['cxc_code']}] ../../cxc_codes/index.md#{feature['cxc_code'].lower()}\n"

            filepath.write_text(content)

        # Create master feature index
        index_content = "# All Features\n\n"
        index_content += f"**Total Features**: {len(self.features)}\n\n"
        index_content += "Complete list of all Ericsson RAN features.\n\n"

        for feature_id, feature in sorted(self.features.items()):
            feature_name = feature.get('name', 'Unknown')
            filename = f"FAJ_{feature_id.replace(' ', '_')}.md"
            index_content += f"- [{feature_name}]({filename}) (FAJ {feature_id})\n"

        index_file = features_dir / "index.md"
        index_file.write_text(index_content)

        print(f"✅ Generated {len(self.features)} enhanced feature references")

    def _generate_parameter_references(self):
        """Generate comprehensive parameter references"""
        print("⚙️ Generating parameter references...")

        params_dir = self.refs_dir / "parameters"

        # Group all parameters by MO class
        mo_parameters = defaultdict(list)

        for feature in self.features.values():
            feature_id = feature['id']
            feature_name = feature.get('name', 'Unknown')

            for param in feature.get('parameters', []):
                mo_class = param.get('mo_class', 'Unknown')
                if mo_class != 'Unknown':
                    mo_parameters[mo_class].append({
                        'name': param.get('name', ''),
                        'type': param.get('type', ''),
                        'description': param.get('description', ''),
                        'feature_id': feature_id,
                        'feature_name': feature_name
                    })

        # Generate MO class index
        index_content = "# Parameter Reference\n\n"
        index_content += "Ericsson RAN parameters organized by Managed Object (MO) class.\n\n"

        for mo_class, params in sorted(mo_parameters.items()):
            index_content += f"## {mo_class} ({len(params)} parameters)\n\n"

            # Create MO class-specific file
            mo_content = f"# {mo_class} Parameters\n\n"
            mo_content += f"**Total Parameters**: {len(params)}\n\n"
            mo_content += "## Parameter List\n\n"

            # Group parameters by feature for better organization
            feature_params = defaultdict(list)
            for param in params:
                feature_name = param['feature_name']
                feature_params[feature_name].append(param)

            for feature_name, feature_param_list in feature_params.items():
                mo_content += f"### Used in {feature_name}\n\n"

                for param in feature_param_list:
                    param_name = param['name']
                    param_type = param['type']
                    param_desc = param['description']
                    feature_id = param['feature_id']

                    mo_content += f"#### {param_name}\n"
                    mo_content += f"- **Type**: {param_type}\n"
                    mo_content += f"- **Feature**: FAJ {feature_id}\n"
                    mo_content += f"- **Description**: {param_desc}\n\n"

                mo_content += "---\n\n"

            # Write MO class file
            mo_file = params_dir / f"{mo_class}.md"
            mo_file.write_text(mo_content)

            index_content += f"- [{mo_class}]({mo_class}.md) - {len(params)} parameters\n"

        # Generate parameter type summary
        type_summary = defaultdict(int)
        for feature in self.features.values():
            for param in feature.get('parameters', []):
                param_type = param.get('type', 'Unknown')
                type_summary[param_type] += 1

        if type_summary:
            type_content = "# Parameter Types Summary\n\n"
            type_content += "Distribution of parameter types across all features.\n\n"

            for param_type, count in sorted(type_summary.items()):
                type_content += f"- **{param_type}**: {count} parameters\n"

            type_file = params_dir / "parameter_types.md"
            type_file.write_text(type_content)

        # Write parameter index
        index_file = params_dir / "index.md"
        index_file.write_text(index_content)

        print(f"✅ Generated parameter references for {len(mo_parameters)} MO classes")

    def _generate_counter_references(self):
        """Generate comprehensive counter references"""
        print("📊 Generating counter references...")

        counters_dir = self.refs_dir / "counters"

        # Group all counters by category
        category_counters = defaultdict(list)

        for feature in self.features.values():
            feature_id = feature['id']
            feature_name = feature.get('name', 'Unknown')

            for counter in feature.get('counters', []):
                category = counter.get('category', 'General')
                category_counters[category].append({
                    'name': counter.get('name', ''),
                    'description': counter.get('description', ''),
                    'feature_id': feature_id,
                    'feature_name': feature_name
                })

        # Generate counter category index
        index_content = "# Performance Counter Reference\n\n"
        index_content += "Ericsson RAN performance counters organized by category.\n\n"

        for category, counters in sorted(category_counters.items()):
            index_content += f"## {category} ({len(counters)} counters)\n\n"

            # Create category-specific file
            cat_content = f"# {category} Performance Counters\n\n"
            cat_content += f"**Total Counters**: {len(counters)}\n\n"
            cat_content += "## Counter List\n\n"

            # Group counters by feature
            feature_counters = defaultdict(list)
            for counter in counters:
                feature_name = counter['feature_name']
                feature_counters[feature_name].append(counter)

            for feature_name, feature_counter_list in feature_counters.items():
                cat_content += f"### Used in {feature_name}\n\n"

                for counter in feature_counter_list:
                    counter_name = counter['name']
                    counter_desc = counter['description']
                    feature_id = counter['feature_id']

                    cat_content += f"#### {counter_name}\n"
                    cat_content += f"- **Feature**: FAJ {feature_id}\n"
                    cat_content += f"- **Description**: {counter_desc}\n\n"

                cat_content += "---\n\n"

            # Write category file
            cat_file = counters_dir / f"{category.lower().replace(' ', '_')}.md"
            cat_file.write_text(cat_content)

            index_content += f"- [{category}]({category.lower().replace(' ', '_')}.md) - {len(counters)} counters\n"

        # Generate counter usage summary
        usage_content = "# Counter Usage Summary\n\n"
        usage_content += "Most frequently used performance counters.\n\n"

        # Count counter usage across features
        counter_usage = defaultdict(int)
        for feature in self.features.values():
            for counter in feature.get('counters', []):
                counter_name = counter.get('name', '')
                if counter_name:
                    counter_usage[counter_name] += 1

        # Show top 20 most used counters
        top_counters = sorted(counter_usage.items(), key=lambda x: x[1], reverse=True)[:20]
        if top_counters:
            usage_content += "### Most Used Counters\n\n"
            for counter_name, usage_count in top_counters:
                usage_content += f"- **{counter_name}**: Used in {usage_count} features\n"

        usage_file = counters_dir / "usage_summary.md"
        usage_file.write_text(usage_content)

        # Write counter index
        index_file = counters_dir / "index.md"
        index_file.write_text(index_content)

        print(f"✅ Generated counter references for {len(category_counters)} categories")

    def _generate_cxc_references(self):
        """Generate CXC code references"""
        print("🎯 Generating CXC code references...")

        cxc_dir = self.refs_dir / "cxc_codes"

        # Create CXC index
        cxc_content = "# CXC Feature Code Reference\n\n"
        cxc_content += "Quick reference for Ericsson feature activation codes.\n\n"
        cxc_content += "## CXC Code List\n\n"

        cxc_features = {}
        for feature in self.features.values():
            cxc_code = feature.get('cxc_code')
            if cxc_code:
                cxc_features[cxc_code] = feature

        for cxc_code, feature in sorted(cxc_features.items()):
            feature_id = feature['id']
            feature_name = feature.get('name', 'Unknown')
            access_type = feature.get('access_type', 'N/A')
            value_package = feature.get('value_package', 'N/A')

            cxc_content += f"### {cxc_code}\n\n"
            cxc_content += f"**Feature**: {feature_name}\n"
            cxc_content += f"**FAJ ID**: FAJ {feature_id}\n"
            cxc_content += f"**Access Type**: {access_type}\n"
            cxc_content += f"**Value Package**: {value_package}\n\n"

            if feature.get('activation_step'):
                cxc_content += "**Activation Command**:\n```bash\n"
                cxc_content += feature['activation_step']
                cxc_content += "\n```\n\n"

            if feature.get('deactivation_step'):
                cxc_content += "**Deactivation Command**:\n```bash\n"
                cxc_content += feature['deactivation_step']
                cxc_content += "\n```\n\n"

            cxc_content += "---\n\n"

        # Generate activation guide
        activation_guide = "# Feature Activation Guide\n\n"
        activation_guide += "## General Activation Process\n\n"
        activation_guide += "1. **Verify Prerequisites**: Check that all prerequisite features are activated\n"
        activation_guide += "2. **Check Compatibility**: Ensure compatibility with current software version\n"
        activation_guide += "3. **Review Configuration**: Understand required parameter changes\n"
        activation_guide += "4. **Schedule Maintenance**: Plan activation during maintenance window\n"
        activation_guide += "5. **Execute Activation**: Use the CXC activation command\n"
        activation_guide += "6. **Verify Operation**: Monitor performance counters and events\n"
        activation_guide += "7. **Optimize Settings**: Adjust parameters based on network conditions\n\n"

        activation_guide += "## Activation Commands by Category\n\n"

        # Group CXC codes by functionality
        func_cxc = defaultdict(list)
        for feature in self.features.values():
            cxc_code = feature.get('cxc_code')
            if cxc_code:
                categories = self.categorize_feature_comprehensive(feature)
                func = categories['functionality']
                func_cxc[func].append({
                    'cxc_code': cxc_code,
                    'feature_name': feature.get('name', 'Unknown'),
                    'feature_id': feature['id']
                })

        for func, cxc_list in sorted(func_cxc.items()):
            if len(cxc_list) > 0:
                activation_guide += f"### {func.replace('_', ' ')}\n\n"
                for cxc_info in cxc_list:
                    activation_guide += f"- **{cxc_info['cxc_code']}**: {cxc_info['feature_name']} (FAJ {cxc_info['feature_id']})\n"
                activation_guide += "\n"

        # Write files
        cxc_file = cxc_dir / "index.md"
        cxc_file.write_text(cxc_content)

        activation_file = cxc_dir / "activation_guide.md"
        activation_file.write_text(activation_guide)

        print(f"✅ Generated CXC references for {len(cxc_features)} codes")

    def _generate_engineering_guidelines(self):
        """Generate engineering guidelines compilation"""
        print("📋 Generating engineering guidelines...")

        guidelines_dir = self.refs_dir / "guidelines" / "engineering"

        # Collect all engineering guidelines
        all_guidelines = []

        for feature in self.features.values():
            guidelines = feature.get('engineering_guidelines', '')
            if guidelines and len(guidelines.strip()) > 50:  # Filter out very short guidelines
                categories = self.categorize_feature_comprehensive(feature)
                all_guidelines.append({
                    'feature_name': feature.get('name', 'Unknown'),
                    'feature_id': feature['id'],
                    'cxc_code': feature.get('cxc_code', ''),
                    'functionality': categories['functionality'],
                    'technology': categories['technology'],
                    'guidelines': guidelines
                })

        # Generate main guidelines index
        guidelines_content = "# Engineering Guidelines\n\n"
        guidelines_content += f"Collection of engineering guidelines from {len(all_guidelines)} features.\n\n"

        # Group guidelines by functionality
        func_guidelines = defaultdict(list)
        for guideline in all_guidelines:
            func = guideline['functionality']
            func_guidelines[func].append(guideline)

        for func, guidelines in sorted(func_guidelines.items()):
            if len(guidelines) > 0:
                guidelines_content += f"## {func.replace('_', ' ')} Guidelines\n\n"

                # Create functionality-specific guidelines file
                func_content = f"# {func.replace('_', ' ')} Engineering Guidelines\n\n"
                func_content += f"**Features with Guidelines**: {len(guidelines)}\n\n"

                for guideline in guidelines:
                    feature_name = guideline['feature_name']
                    feature_id = guideline['feature_id']
                    cxc_code = guideline['cxc_code']
                    guideline_text = guideline['guidelines']

                    func_content += f"## {feature_name}\n\n"
                    func_content += f"**FAJ ID**: FAJ {feature_id}\n"
                    if cxc_code:
                        func_content += f"**CXC Code**: {cxc_code}\n"
                    func_content += "\n### Guidelines\n\n"

                    # Clean up and format guidelines
                    cleaned_guidelines = guideline_text.strip()
                    if len(cleaned_guidelines) > 2000:
                        cleaned_guidelines = cleaned_guidelines[:2000] + "..."

                    func_content += cleaned_guidelines + "\n\n"
                    func_content += "---\n\n"

                # Write functionality guidelines file
                func_file = guidelines_dir / f"{func.lower().replace(' ', '_')}.md"
                func_file.write_text(func_content)

                guidelines_content += f"- [{func.replace('_', ' ')} Guidelines]({func.lower().replace(' ', '_')}.md) ({len(guidelines)} features)\n"

        # Generate configuration best practices
        config_content = "# Configuration Best Practices\n\n"
        config_content += "General best practices for Ericsson RAN feature configuration.\n\n"

        config_content += "## Pre-Configuration Checklist\n\n"
        config_content += "1. **Feature Assessment**\n"
        config_content += "   - Understand feature purpose and benefits\n"
        config_content += "   - Identify target use cases\n"
        config_content += "   - Evaluate network readiness\n\n"

        config_content += "2. **Prerequisite Verification**\n"
        config_content += "   - Check required software version\n"
        config_content += "   - Verify hardware compatibility\n"
        config_content += "   - Confirm dependent features\n\n"

        config_content += "3. **Parameter Planning**\n"
        config_content += "   - Review all feature parameters\n"
        config_content += "   - Determine optimal values\n"
        config_content += "   - Plan monitoring requirements\n\n"

        config_content += "## Configuration Guidelines by Technology\n\n"

        # Technology-specific guidelines
        tech_guidelines = {
            'MIMO': [
                "Start with conservative MIMO configurations",
                "Monitor UE capabilities and MIMO performance",
                "Adjust antenna configurations based on field measurements",
                "Consider load balancing when enabling advanced MIMO"
            ],
            'Energy_Efficiency': [
                "Implement gradual sleep mode activation",
                "Monitor QoS impact during energy saving",
                "Configure appropriate wake-up thresholds",
                "Balance energy savings with performance requirements"
            ],
            'Mobility_Management': [
                "Optimize handover parameters for network topology",
                "Monitor handover success rates",
                "Configure appropriate hysteresis values",
                "Consider load and interference in mobility decisions"
            ]
        }

        for tech, guidelines in tech_guidelines.items():
            config_content += f"### {tech.replace('_', ' ')}\n\n"
            for guideline in guidelines:
                config_content += f"- {guideline}\n"
            config_content += "\n"

        # Write files
        guidelines_file = guidelines_dir / "index.md"
        guidelines_file.write_text(guidelines_content)

        config_file = guidelines_dir / "configuration_best_practices.md"
        config_file.write_text(config_content)

        print(f"✅ Generated engineering guidelines for {len(all_guidelines)} features")

    def _generate_troubleshooting_guides(self):
        """Generate troubleshooting guides"""
        print("🔧 Generating troubleshooting guides...")

        troubleshooting_dir = self.refs_dir / "guidelines" / "troubleshooting"

        # Generate main troubleshooting guide
        ts_content = "# Troubleshooting Guides\n\n"
        ts_content += "Common issues and solutions for Ericsson RAN features.\n\n"

        ts_content += "## General Troubleshooting Approach\n\n"
        ts_content += "1. **Problem Identification**\n"
        ts_content += "   - Define symptoms and affected areas\n"
        ts_content += "   - Determine when the issue started\n"
        ts_content += "   - Identify any recent changes\n\n"

        ts_content += "2. **Information Gathering**\n"
        ts_content += "   - Check feature activation status\n"
        ts_content += "   - Review performance counters\n"
        ts_content += "   - Analyze event logs\n"
        ts_content += "   - Verify parameter settings\n\n"

        ts_content += "3. **Analysis Phase**\n"
        ts_content += "   - Compare against baseline performance\n"
        ts_content += "   - Check for feature conflicts\n"
        ts_content += "   - Verify prerequisites\n\n"

        ts_content += "4. **Resolution Actions**\n"
        ts_content += "   - Implement corrective changes\n"
        ts_content += "   - Monitor impact\n"
        ts_content += "   - Document solution\n\n"

        # Feature-specific troubleshooting
        ts_content += "## Feature-Specific Troubleshooting\n\n"

        # Create troubleshooting guides by functionality
        func_troubleshooting = {
            'MIMO': {
                'common_issues': [
                    "MIMO not achieving expected throughput gains",
                    "UE not supporting configured MIMO mode",
                    "MIMO performance degradation"
                ],
                'checks': [
                    "Verify UE MIMO capabilities",
                    "Check antenna configuration",
                    "Review MIMO parameter settings",
                    "Monitor interference levels"
                ],
                'counters': [
                    "pmMimoSleepTime",
                    "pmRadioTxRankDistr",
                    "pmMimoSleepOppTime"
                ]
            },
            'Energy_Efficiency': {
                'common_issues': [
                    "Sleep mode not activating",
                    "Excessive energy consumption",
                    "QoS degradation during energy saving"
                ],
                'checks': [
                    "Verify sleep mode thresholds",
                    "Check traffic patterns",
                    "Review timer configurations",
                    "Monitor QoS metrics"
                ],
                'counters': [
                    "pmTxOffTime",
                    "pmTxOffRatio",
                    "pmPdcpPktDiscDlAqm"
                ]
            },
            'Mobility_Management': {
                'common_issues': [
                    "High handover failure rate",
                    "Ping-pong handovers",
                    "RRC connection drops"
                ],
                'checks': [
                    "Verify handover parameters",
                    "Check neighbor cell relations",
                    "Review mobility timers",
                    "Analyze signal quality measurements"
                ],
                'counters': [
                    "pmHoSuccessRate",
                    "pmRrcConnEstabSuccessRate",
                    "pmUeThpTimeDl"
                ]
            }
        }

        for func, ts_info in func_troubleshooting.items():
            ts_content += f"### {func.replace('_', ' ')}\n\n"

            # Create detailed troubleshooting file
            func_ts_content = f"# {func.replace('_', ' ')} Troubleshooting\n\n"

            func_ts_content += "## Common Issues\n\n"
            for issue in ts_info['common_issues']:
                func_ts_content += f"### {issue}\n\n"
                func_ts_content += "**Symptoms**: [Describe observed symptoms]\n\n"
                func_ts_content += "**Potential Causes**:\n"
                func_ts_content += "- [List potential causes]\n\n"
                func_ts_content += "**Troubleshooting Steps**:\n"
                func_ts_content += "1. [First diagnostic step]\n"
                func_ts_content += "2. [Second diagnostic step]\n"
                func_ts_content += "3. [Resolution action]\n\n"

            func_ts_content += "## Key Performance Counters\n\n"
            for counter in ts_info['counters']:
                func_ts_content += f"- **{counter}**: [Purpose and interpretation]\n"
            func_ts_content += "\n"

            func_ts_content += "## Recommended Checks\n\n"
            for check in ts_info['checks']:
                func_ts_content += f"- {check}\n"
            func_ts_content += "\n"

            # Write functionality troubleshooting file
            func_ts_file = troubleshooting_dir / f"{func.lower().replace(' ', '_')}.md"
            func_ts_file.write_text(func_ts_content)

            # Add summary to main guide
            ts_content += f"Common issues include:\n"
            for issue in ts_info['common_issues']:
                ts_content += f"- {issue}\n"
            ts_content += f"\n**Detailed Guide**: [{func.replace('_', ' ')} Troubleshooting]({func.lower().replace(' ', '_')}.md)\n\n"

        # Generate performance counter troubleshooting
        pc_content = "# Performance Counter Analysis\n\n"
        pc_content += "Guide to analyzing performance counters for troubleshooting.\n\n"

        pc_content += "## Counter Analysis Approach\n\n"
        pc_content += "1. **Baseline Establishment**\n"
        pc_content += "   - Establish normal performance baselines\n"
        pc_content += "   - Identify key performance indicators\n"
        pc_content += "   - Set monitoring thresholds\n\n"

        pc_content += "2. **Trend Analysis**\n"
        pc_content += "   - Monitor counter trends over time\n"
        pc_content += "   - Identify gradual degradation\n"
        pc_content += "   - Correlate with network changes\n\n"

        pc_content += "3. **Threshold Monitoring**\n"
        pc_content += "   - Set appropriate alert thresholds\n"
        pc_content += "   - Monitor for counter spikes\n"
        pc_content += "   - Investigate threshold breaches\n\n"

        # Add counter-specific guidance
        pc_content += "## Key Counter Categories\n\n"
        pc_content += "### MIMO Performance\n"
        pc_content += "- **pmMimoSleepTime**: Time spent in MIMO sleep mode\n"
        pc_content += "- **pmRadioTxRankDistr**: Distribution of MIMO ranks\n"
        pc_content += "- **pmMimoSleepOppTime**: Opportunities for MIMO sleep\n\n"

        pc_content += "### Energy Efficiency\n"
        pc_content += "- **pmTxOffTime**: Time with TX off\n"
        pc_content += "- **pmTxOffRatio**: Percentage of TX off time\n"
        pc_content += "- **pmPdcpPktDiscDlAqm**: PDCP packet discard due to AQM\n\n"

        pc_content += "### Mobility Performance\n"
        pc_content += "- **pmHoSuccessRate**: Handover success rate\n"
        pc_content += "- **pmRrcConnEstabSuccessRate**: RRC connection establishment success\n"
        pc_content += "- **pmUeThpTimeDl**: UE throughput time\n\n"

        # Write files
        main_ts_file = troubleshooting_dir / "index.md"
        main_ts_file.write_text(ts_content)

        pc_ts_file = troubleshooting_dir / "performance_counter_analysis.md"
        pc_ts_file.write_text(pc_content)

        print("✅ Generated troubleshooting guides")

    def _generate_best_practices(self):
        """Generate best practices guides"""
        print("💡 Generating best practices guides...")

        best_practices_dir = self.refs_dir / "guidelines" / "best_practices"

        # Generate main best practices guide
        bp_content = "# Best Practices Guide\n\n"
        bp_content += "Collection of best practices for Ericsson RAN feature deployment and operation.\n\n"

        bp_content += "## Deployment Best Practices\n\n"
        bp_content += "### 1. Planning Phase\n\n"
        bp_content += "- **Network Assessment**: Evaluate current network capabilities and identify improvement opportunities\n"
        bp_content += "- **Feature Selection**: Choose features that address specific network needs\n"
        bp_content += "- **Compatibility Check**: Verify hardware and software compatibility\n"
        bp_content += "- **Capacity Planning**: Assess impact on network capacity and performance\n\n"

        bp_content += "### 2. Implementation Phase\n\n"
        bp_content += "- **Staged Deployment**: Implement features in phases to minimize risk\n"
        bp_content += "- **Pilot Testing**: Test in limited scope before full deployment\n"
        bp_content += "- **Configuration Management**: Maintain proper configuration documentation\n"
        bp_content += "- **Backup Procedures**: Ensure proper backup and rollback procedures\n\n"

        bp_content += "### 3. Optimization Phase\n\n"
        bp_content += "- **Performance Monitoring**: Continuously monitor feature performance\n"
        bp_content += "- **Parameter Tuning**: Optimize parameters based on network conditions\n"
        bp_content += "- **Load Balancing**: Balance feature activation across the network\n"
        bp_content += "- **Feedback Loop**: Collect and act on operational feedback\n\n"

        # Technology-specific best practices
        tech_best_practices = {
            'MIMO': {
                'deployment': [
                    "Start with 2x2 MIMO before advancing to 4x4",
                    "Ensure proper antenna calibration",
                    "Verify UE MIMO capability penetration",
                    "Monitor interference impact on MIMO performance"
                ],
                'optimization': [
                    "Adjust MIMO thresholds based on traffic patterns",
                    "Optimize antenna port configurations",
                    "Balance between complexity and performance gains",
                    "Regular performance assessment and tuning"
                ]
            },
            'Energy_Efficiency': {
                'deployment': [
                    "Implement conservative sleep mode settings initially",
                    "Ensure QoS requirements are met during energy saving",
                    "Test during low traffic periods first",
                    "Monitor user experience impact"
                ],
                'optimization': [
                    "Fine-tune sleep mode thresholds",
                    "Optimize timer configurations",
                    "Balance energy savings with performance",
                    "Seasonal adjustments for traffic patterns"
                ]
            },
            'Mobility_Management': {
                'deployment': [
                    "Verify neighbor cell lists are complete",
                    "Test handover parameters in controlled environment",
                    "Ensure proper network synchronization",
                    "Validate emergency call handling"
                ],
                'optimization': [
                    "Optimize handover margins and hysteresis",
                    "Adjust mobility state timers",
                    "Fine-tune load balancing parameters",
                    "Regular performance analysis"
                ]
            }
        }

        for tech, practices in tech_best_practices.items():
            # Create technology-specific best practices file
            tech_bp_content = f"# {tech.replace('_', ' ')} Best Practices\n\n"

            tech_bp_content += "## Deployment Best Practices\n\n"
            for practice in practices['deployment']:
                tech_bp_content += f"- {practice}\n"
            tech_bp_content += "\n"

            tech_bp_content += "## Optimization Best Practices\n\n"
            for practice in practices['optimization']:
                tech_bp_content += f"- {practice}\n"
            tech_bp_content += "\n"

            # Add monitoring recommendations
            tech_bp_content += "## Monitoring Recommendations\n\n"
            tech_bp_content += "### Key Performance Indicators\n\n"
            if tech == 'MIMO':
                tech_bp_content += "- MIMO throughput gains\n"
                tech_bp_content += "- MIMO rank distribution\n"
                tech_bp_content += "- UE MIMO capability utilization\n"
            elif tech == 'Energy_Efficiency':
                tech_bp_content += "- Energy consumption metrics\n"
                tech_bp_content += "- Sleep mode activation rate\n"
                tech_bp_content += "- QoS impact during energy saving\n"
            elif tech == 'Mobility_Management':
                tech_bp_content += "- Handover success rate\n"
                tech_bp_content += "- Call drop rate\n"
                tech_bp_content += "- Ping-pong handover frequency\n"

            tech_bp_content += "\n### Alert Thresholds\n\n"
            tech_bp_content += "- [Set appropriate alert thresholds]\n"
            tech_bp_content += "- [Define escalation procedures]\n"
            tech_bp_content += "- [Establish monitoring frequency]\n\n"

            # Write technology best practices file
            tech_bp_file = best_practices_dir / f"{tech.lower().replace(' ', '_')}.md"
            tech_bp_file.write_text(tech_bp_content)

            # Add summary to main guide
            bp_content += f"## {tech.replace('_', ' ')} Best Practices\n\n"
            bp_content += f"See [{tech.replace('_', ' ')} Best Practices]({tech.lower().replace(' ', '_')}.md) for detailed guidance.\n\n"

        # Generate operational best practices
        ops_content = "# Operational Best Practices\n\n"
        ops_content += "Best practices for day-to-day operations of Ericsson RAN features.\n\n"

        ops_content += "## Daily Operations\n\n"
        ops_content += "1. **Performance Monitoring**\n"
        ops_content += "   - Review key performance indicators\n"
        ops_content += "   - Check for feature-related alarms\n"
        ops_content += "   - Analyze performance trends\n"
        ops_content += "   - Document any anomalies\n\n"

        ops_content += "2. **Maintenance Activities**\n"
        ops_content += "   - Schedule regular feature health checks\n"
        ops_content += "   - Review parameter configurations\n"
        ops_content += "   - Update feature documentation\n"
        ops_content += "   - Plan software upgrades\n\n"

        ops_content += "## Weekly Operations\n\n"
        ops_content += "1. **Performance Review**\n"
        ops_content += "   - Analyze weekly performance trends\n"
        ops_content += "   - Compare against baselines\n"
        ops_content += "   - Identify optimization opportunities\n"
        ops_content += "   - Plan configuration changes\n\n"

        ops_content += "2. **Feature Assessment**\n"
        ops_content += "   - Review feature utilization rates\n"
        ops_content += "   - Assess feature effectiveness\n"
        ops_content += "   - Identify underperforming features\n"
        ops_content += "   - Plan feature adjustments\n\n"

        # Write files
        main_bp_file = best_practices_dir / "index.md"
        main_bp_file.write_text(bp_content)

        ops_bp_file = best_practices_dir / "operational_best_practices.md"
        ops_bp_file.write_text(ops_content)

        print("✅ Generated best practices guides")

    def _generate_performance_guides(self):
        """Generate performance optimization guides"""
        print("📈 Generating performance optimization guides...")

        performance_dir = self.refs_dir / "guidelines" / "performance"

        # Generate main performance guide
        perf_content = "# Performance Optimization Guide\n\n"
        perf_content += "Comprehensive guide to optimizing Ericsson RAN feature performance.\n\n"

        perf_content += "## Performance Optimization Framework\n\n"
        perf_content += "### 1. Performance Assessment\n\n"
        perf_content += "- **Baseline Establishment**: Create performance baselines before optimization\n"
        perf_content += "- **KPI Definition**: Define relevant key performance indicators\n"
        perf_content += "- **Measurement Planning**: Plan comprehensive performance measurements\n"
        perf_content += "- **Analysis Framework**: Establish systematic analysis approach\n\n"

        perf_content += "### 2. Optimization Strategy\n\n"
        perf_content += "- **Goal Setting**: Define clear optimization objectives\n"
        perf_content += "- **Priority Matrix**: Prioritize optimization initiatives\n"
        perf_content += "- **Resource Planning**: Allocate necessary resources\n"
        perf_content += "- **Timeline Development**: Create realistic optimization timeline\n\n"

        perf_content += "### 3. Implementation Process\n\n"
        perf_content += "- **Change Management**: Follow proper change management procedures\n"
        perf_content += "- **Testing Protocol**: Implement thorough testing protocols\n"
        perf_content += "- **Monitoring Setup**: Establish comprehensive monitoring\n"
        perf_content += "- **Validation Process**: Validate optimization results\n\n"

        # Feature-specific optimization guides
        feature_optimization = {
            'MIMO': {
                'objectives': [
                    "Maximize throughput gains",
                    "Optimize antenna utilization",
                    "Improve spectral efficiency",
                    "Enhance user experience"
                ],
                'parameters': [
                    "MIMO configuration parameters",
                    "Antenna port settings",
                    "Rank adaptation thresholds",
                    "Beamforming parameters"
                ],
                'counters': [
                    "pmMimoSleepTime",
                    "pmRadioTxRankDistr",
                    "pmMimoSleepOppTime"
                ],
                'optimization_steps': [
                    "Analyze current MIMO performance baseline",
                    "Identify underutilized MIMO capabilities",
                    "Optimize antenna configurations",
                    "Adjust rank adaptation parameters",
                    "Monitor performance improvements",
                    "Fine-tune based on results"
                ]
            },
            'Energy_Efficiency': {
                'objectives': [
                    "Minimize energy consumption",
                    "Maintain QoS standards",
                    "Optimize sleep mode utilization",
                    "Reduce operational costs"
                ],
                'parameters': [
                    "Sleep mode thresholds",
                    "Activation/deactivation timers",
                    "Power control parameters",
                    "Load balancing settings"
                ],
                'counters': [
                    "pmTxOffTime",
                    "pmTxOffRatio",
                    "pmPdcpPktDiscDlAqm",
                    "pmPdcpPktDiscDlPelr"
                ],
                'optimization_steps': [
                    "Establish energy consumption baseline",
                    "Configure conservative sleep settings",
                    "Monitor QoS impact during energy saving",
                    "Gradually optimize sleep thresholds",
                    "Analyze cost-benefit improvements",
                    "Adjust based on traffic patterns"
                ]
            },
            'Mobility_Management': {
                'objectives': [
                    "Maximize handover success rate",
                    "Minimize call drops",
                    "Optimize load balancing",
                    "Improve user mobility experience"
                ],
                'parameters': [
                    "Handover margins",
                    "Hysteresis values",
                    "Mobility state timers",
                    "Load balancing parameters"
                ],
                'counters': [
                    "pmHoSuccessRate",
                    "pmRrcConnEstabSuccessRate",
                    "pmUeThpTimeDl",
                    "pmRrcConnEstabAttSum"
                ],
                'optimization_steps': [
                    "Analyze current mobility performance",
                    "Identify mobility problem areas",
                    "Optimize handover parameters",
                    "Adjust load balancing settings",
                    "Monitor mobility KPI improvements",
                    "Fine-tune based on network conditions"
                ]
            }
        }

        for feature, opt_info in feature_optimization.items():
            # Create feature-specific optimization file
            feature_opt_content = f"# {feature.replace('_', ' ')} Performance Optimization\n\n"

            feature_opt_content += "## Optimization Objectives\n\n"
            for objective in opt_info['objectives']:
                feature_opt_content += f"- {objective}\n"
            feature_opt_content += "\n"

            feature_opt_content += "## Key Parameters\n\n"
            for param in opt_info['parameters']:
                feature_opt_content += f"- **{param}**: [Parameter purpose and optimization guidance]\n"
            feature_opt_content += "\n"

            feature_opt_content += "## Key Performance Counters\n\n"
            for counter in opt_info['counters']:
                feature_opt_content += f"- **{counter}**: [Counter interpretation and target values]\n"
            feature_opt_content += "\n"

            feature_opt_content += "## Optimization Process\n\n"
            for i, step in enumerate(opt_info['optimization_steps'], 1):
                feature_opt_content += f"{i}. {step}\n"
            feature_opt_content += "\n"

            feature_opt_content += "## Performance Targets\n\n"
            feature_opt_content += "### Baseline Targets\n\n"
            if feature == 'MIMO':
                feature_opt_content += "- MIMO throughput gain: > 20%\n"
                feature_opt_content += "- High rank utilization: > 60%\n"
                feature_opt_content += "- MIMO availability: > 95%\n"
            elif feature == 'Energy_Efficiency':
                feature_opt_content += "- Energy savings: 10-30%\n"
                feature_opt_content += "- QoS impact: < 5%\n"
                feature_opt_content += "- Sleep mode utilization: > 40%\n"
            elif feature == 'Mobility_Management':
                feature_opt_content += "- Handover success rate: > 95%\n"
                feature_opt_content += "- Call drop rate: < 1%\n"
                feature_opt_content += "- Ping-pong rate: < 10%\n"

            feature_opt_content += "\n### Stretch Targets\n\n"
            feature_opt_content += "- [Define ambitious but achievable targets]\n"
            feature_opt_content += "- [Set milestones for continuous improvement]\n\n"

            # Write feature optimization file
            feature_opt_file = performance_dir / f"{feature.lower().replace(' ', '_')}_optimization.md"
            feature_opt_file.write_text(feature_opt_content)

            # Add summary to main guide
            perf_content += f"## {feature.replace('_', ' ')} Optimization\n\n"
            perf_content += f"See [{feature.replace('_', ' ')} Performance Optimization]({feature.lower().replace(' ', '_')}_optimization.md) for detailed guidance.\n\n"

        # Generate performance monitoring guide
        monitor_content = "# Performance Monitoring Guide\n\n"
        monitor_content += "Comprehensive approach to monitoring Ericsson RAN feature performance.\n\n"

        monitor_content += "## Monitoring Strategy\n\n"
        monitor_content += "### 1. Real-time Monitoring\n\n"
        monitor_content += "- **Live Dashboard**: Real-time performance visualization\n"
        monitor_content += "- **Alert Management**: Proactive alert configuration\n"
        monitor_content += "- **Incident Response**: Rapid incident identification and response\n"
        monitor_content += "- **Health Checks**: Continuous feature health assessment\n\n"

        monitor_content += "### 2. Periodic Analysis\n\n"
        monitor_content += "- **Daily Reviews**: Daily performance trend analysis\n"
        monitor_content += "- **Weekly Reports**: Weekly performance summaries\n"
        monitor_content += "- **Monthly Analysis**: Monthly performance deep-dives\n"
        monitor_content += "- **Quarterly Reviews**: Quarterly strategic assessments\n\n"

        monitor_content += "### 3. Performance Metrics\n\n"
        monitor_content += "#### Capacity Metrics\n\n"
        monitor_content += "- Cell throughput and capacity utilization\n"
        monitor_content += "- User throughput distribution\n"
        monitor_content += "- Resource block utilization\n"
        monitor_content += "- Carrier aggregation efficiency\n\n"

        monitor_content += "#### Quality Metrics\n\n"
        monitor_content += "- Signal quality indicators (RSRP, RSRQ, SINR)\n"
        monitor_content += "- Quality of Experience (QoE) metrics\n"
        monitor_content += "- Latency and jitter measurements\n"
        monitor_content += "- Error rates and retransmissions\n\n"

        monitor_content += "#### Availability Metrics\n\n"
        monitor_content += "- Feature availability and uptime\n"
        monitor_content += "- Service accessibility rates\n"
        monitor_content += "- Network element availability\n"
        monitor_content += "- Maintenance window impact\n\n"

        # Write files
        main_perf_file = performance_dir / "index.md"
        main_perf_file.write_text(perf_content)

        monitor_file = performance_dir / "performance_monitoring.md"
        monitor_file.write_text(monitor_content)

        print("✅ Generated performance optimization guides")

    def _generate_navigation_structure(self):
        """Generate navigation structure with cross-references"""
        print("🧭 Generating navigation structure...")

        # Create main navigation index
        nav_content = "# Ericsson RAN Features - Navigation Index\n\n"
        nav_content += "Complete navigation guide for Ericsson RAN Features documentation.\n\n"

        nav_content += "## Quick Access\n\n"
        nav_content += "- [All Features](features/all_features/index.md) - Complete feature list\n"
        nav_content += "- [Technology Categories](features/technology/index.md) - Features by technology\n"
        nav_content += "- [Functionality Categories](features/functionality/index.md) - Features by function\n"
        nav_content += "- [Node Type Categories](features/node_type/index.md) - Features by node type\n"
        nav_content += "- [Access Type Categories](features/access_type/index.md) - Features by access type\n\n"

        nav_content += "## Specialized References\n\n"
        nav_content += "- [Parameters](parameters/index.md) - Parameter reference by MO class\n"
        nav_content += "- [Performance Counters](counters/index.md) - Counter reference by category\n"
        nav_content += "- [CXC Codes](cxc_codes/index.md) - Feature activation codes\n"
        nav_content += "- [Events](events/index.md) - Event reference\n\n"

        nav_content += "## Guidance and Documentation\n\n"
        nav_content += "- [Engineering Guidelines](guidelines/engineering/index.md) - Technical guidelines\n"
        nav_content += "- [Troubleshooting Guides](guidelines/troubleshooting/index.md) - Troubleshooting help\n"
        nav_content += "- [Best Practices](guidelines/best_practices/index.md) - Operational best practices\n"
        nav_content += "- [Performance Optimization](guidelines/performance/index.md) - Performance guides\n\n"

        nav_content += "## Quick Reference\n\n"
        nav_content += "- [Quick Reference](quick_reference/index.md) - Quick lookups\n"
        nav_content += "- [Cheat Sheets](cheat_sheets/index.md) - Quick reference cards\n"
        nav_content += "- [Common Patterns](quick_reference/common_patterns.md) - Common configuration patterns\n"
        nav_content += "- [Activation Commands](quick_reference/activation_commands.md) - Quick activation guide\n\n"

        # Generate feature cross-reference index
        cross_ref_content = "# Feature Cross-Reference Index\n\n"
        cross_ref_content += "Cross-reference index linking features across all categorization dimensions.\n\n"

        cross_ref_content += "## Cross-Reference Matrix\n\n"
        cross_ref_content += "| Feature | Technology | Functionality | Node Type | Access Type | CXC Code |\n"
        cross_ref_content += "|---------|------------|--------------|-----------|-------------|----------|\n"

        for feature_id, feature in sorted(self.features.items()):
            categories = self.categorize_feature_comprehensive(feature)
            feature_name = feature.get('name', 'Unknown')
            cxc_code = feature.get('cxc_code', 'N/A')

            # Create markdown links
            tech_link = f"[{categories['technology'].replace('_', ' ')}](../features/technology/{categories['technology'].lower().replace(' ', '_')}.md)"
            func_link = f"[{categories['functionality'].replace('_', ' ')}](../features/functionality/{categories['functionality'].lower().replace(' ', '_')}.md)"
            node_link = f"[{categories['node_type'].replace('_', ' ')}](../features/node_type/{categories['node_type'].lower().replace(' ', '_')}.md)"
            access_link = f"[{categories['access_type'].replace('_', ' ')}](../features/access_type/{categories['access_type'].lower().replace(' ', '_')}.md)"
            feature_link = f"[{feature_name}](../features/all_features/FAJ_{feature_id.replace(' ', '_')}.md)"

            cross_ref_content += f"| {feature_link} | {tech_link} | {func_link} | {node_link} | {access_link} | {cxc_code} |\n"

        # Generate search index
        search_content = "# Search Index\n\n"
        search_content += "Keyword search index for Ericsson RAN Features.\n\n"

        # Build keyword index
        keyword_index = defaultdict(list)

        for feature_id, feature in self.features.items():
            feature_name = feature.get('name', '').lower()
            description = feature.get('summary', '').lower()

            # Extract keywords from feature name
            words = re.findall(r'\b\w+\b', feature_name)
            for word in words:
                if len(word) > 2:  # Skip short words
                    keyword_index[word].append({
                        'feature_id': feature_id,
                        'feature_name': feature.get('name', 'Unknown'),
                        'context': 'Feature Name'
                    })

            # Extract keywords from description
            desc_words = re.findall(r'\b\w+\b', description)
            for word in desc_words[:20]:  # Limit to first 20 words
                if len(word) > 3:  # Skip very short words
                    keyword_index[word].append({
                        'feature_id': feature_id,
                        'feature_name': feature.get('name', 'Unknown'),
                        'context': 'Description'
                    })

        # Sort keywords and build search index
        for keyword in sorted(keyword_index.keys()):
            if len(keyword_index[keyword]) <= 10:  # Limit to avoid very common words
                search_content += f"## {keyword.title()}\n\n"
                for item in keyword_index[keyword]:
                    feature_link = f"[{item['feature_name']}](../features/all_features/FAJ_{item['feature_id'].replace(' ', '_')}.md)"
                    search_content += f"- {feature_link} ({item['context']})\n"
                search_content += "\n"

        # Write navigation files
        nav_file = self.refs_dir / "navigation.md"
        nav_file.write_text(nav_content)

        cross_ref_file = self.refs_dir / "indices" / "cross_reference.md"
        cross_ref_file.write_text(cross_ref_content)

        search_file = self.refs_dir / "indices" / "search_index.md"
        search_file.write_text(search_content)

        # Update main references index
        main_index_content = "# Ericsson RAN Features Reference Documentation\n\n"
        main_index_content += f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        main_index_content += f"**Total Features**: {len(self.features)}\n"
        main_index_content += f"**Total Parameters**: {sum(len(f.get('parameters', [])) for f in self.features.values())}\n"
        main_index_content += f"**Total Counters**: {sum(len(f.get('counters', [])) for f in self.features.values())}\n\n"

        main_index_content += "## 📚 Documentation Structure\n\n"
        main_index_content += nav_content

        main_index_file = self.refs_dir / "index.md"
        main_index_file.write_text(main_index_content)

        print("✅ Generated navigation structure")

    def _generate_quick_references(self):
        """Generate quick reference materials"""
        print("⚡ Generating quick references...")

        quick_ref_dir = self.refs_dir / "quick_reference"
        cheat_sheet_dir = self.refs_dir / "cheat_sheets"

        # Generate quick reference guide
        qr_content = "# Quick Reference Guide\n\n"
        qr_content += "Essential information for Ericsson RAN Features at a glance.\n\n"

        # Feature activation quick reference
        qr_content += "## Feature Activation Quick Reference\n\n"
        qr_content += "### Common Activation Pattern\n\n"
        qr_content += "```bash\n# Standard activation command\nset FeatureState.featureState=ACTIVATED FeatureState=<CXC_CODE>\n\n# Standard deactivation command\nset FeatureState.featureState=DEACTIVATED FeatureState=<CXC_CODE>\n```\n\n"

        # Top features by category
        feature_categories = defaultdict(list)
        for feature in self.features.values():
            categories = self.categorize_feature_comprehensive(feature)
            func = categories['functionality']
            feature_categories[func].append(feature)

        qr_content += "### Top Features by Category\n\n"

        for func, features in sorted(feature_categories.items()):
            if len(features) > 0:
                qr_content += f"#### {func.replace('_', ' ')}\n\n"
                # Show up to 5 features per category
                for feature in sorted(features[:5], key=lambda x: x.get('name', '')):
                    feature_name = feature.get('name', 'Unknown')
                    feature_id = feature['id']
                    cxc_code = feature.get('cxc_code', '')
                    qr_content += f"- **{feature_name}** (FAJ {feature_id})"
                    if cxc_code:
                        qr_content += f" - CXC {cxc_code}"
                    qr_content += "\n"
                qr_content += "\n"

        # Generate parameter quick reference
        param_qr_content = "# Parameter Quick Reference\n\n"
        param_qr_content += "Frequently used parameters by MO class.\n\n"

        # Collect parameter usage statistics
        param_usage = defaultdict(int)
        param_details = {}

        for feature in self.features.values():
            for param in feature.get('parameters', []):
                param_name = param.get('name', '')
                mo_class = param.get('mo_class', 'Unknown')
                if param_name and mo_class != 'Unknown':
                    param_usage[param_name] += 1
                    if param_name not in param_details:
                        param_details[param_name] = {
                            'mo_class': mo_class,
                            'type': param.get('type', ''),
                            'description': param.get('description', ''),
                            'features': []
                        }
                    param_details[param_name]['features'].append(feature.get('name', 'Unknown'))

        # Show top 20 most used parameters
        top_params = sorted(param_usage.items(), key=lambda x: x[1], reverse=True)[:20]

        param_qr_content += "### Most Used Parameters\n\n"
        for param_name, usage_count in top_params:
            details = param_details[param_name]
            param_qr_content += f"#### {param_name}\n"
            param_qr_content += f"- **MO Class**: {details['mo_class']}\n"
            param_qr_content += f"- **Type**: {details['type']}\n"
            param_qr_content += f"- **Used in**: {usage_count} features\n"
            if details['description']:
                param_qr_content += f"- **Description**: {details['description'][:100]}...\n"
            param_qr_content += "\n"

        # Generate counter quick reference
        counter_qr_content = "# Performance Counter Quick Reference\n\n"
        counter_qr_content += "Key performance counters for monitoring.\n\n"

        # Collect counter usage statistics
        counter_usage = defaultdict(int)
        counter_details = {}

        for feature in self.features.values():
            for counter in feature.get('counters', []):
                counter_name = counter.get('name', '')
                category = counter.get('category', 'General')
                if counter_name:
                    counter_usage[counter_name] += 1
                    if counter_name not in counter_details:
                        counter_details[counter_name] = {
                            'category': category,
                            'description': counter.get('description', ''),
                            'features': []
                        }
                    counter_details[counter_name]['features'].append(feature.get('name', 'Unknown'))

        # Show top 15 most used counters
        top_counters = sorted(counter_usage.items(), key=lambda x: x[1], reverse=True)[:15]

        counter_qr_content += "### Key Performance Counters\n\n"
        for counter_name, usage_count in top_counters:
            details = counter_details[counter_name]
            counter_qr_content += f"#### {counter_name}\n"
            counter_qr_content += f"- **Category**: {details['category']}\n"
            counter_qr_content += f"- **Used in**: {usage_count} features\n"
            if details['description']:
                counter_qr_content += f"- **Description**: {details['description'][:100]}...\n"
            counter_qr_content += "\n"

        # Generate activation command cheat sheet
        activation_cheat_content = "# Activation Commands Cheat Sheet\n\n"
        activation_cheat_content += "Quick reference for feature activation commands.\n\n"

        activation_cheat_content += "## Command Templates\n\n"
        activation_cheat_content += "### Basic Activation\n"
        activation_cheat_content += "```bash\n# Activate a feature\nset FeatureState.featureState=ACTIVATED FeatureState=<CXC_CODE>\n\n# Deactivate a feature\nset FeatureState.featureState=DEACTIVATED FeatureState=<CXC_CODE>\n```\n\n"

        activation_cheat_content += "### Parameter Configuration\n"
        activation_cheat_content += "```bash\n# Set a parameter value\nset <MO_Class>.<ParameterName>=<Value> <MO_Instance>\n\n# Example: Set MIMO sleep mode\nset MimoSleepFunction.sleepMode=1 MimoSleepFunction=1\n```\n\n"

        activation_cheat_content += "### Verification Commands\n"
        activation_cheat_content += "```bash\n# Check feature status\nget FeatureState.featureState FeatureState=<CXC_CODE>\n\n# Check parameter value\nget <MO_Class>.<ParameterName> <MO_Instance>\n\n# List all features\nlst FeatureState\n```\n\n"

        # Add specific CXC codes
        activation_cheat_content += "## Common CXC Codes\n\n"
        cxc_list = [(f.get('cxc_code', ''), f.get('name', 'Unknown')) for f in self.features.values() if f.get('cxc_code')]
        cxc_list.sort()

        for cxc_code, feature_name in cxc_list[:20]:  # Show first 20
            activation_cheat_content += f"- **{cxc_code}**: {feature_name}\n"

        # Generate troubleshooting cheat sheet
        ts_cheat_content = "# Troubleshooting Cheat Sheet\n\n"
        ts_cheat_content += "Quick troubleshooting steps for common issues.\n\n"

        ts_cheat_content += "## Quick Diagnostic Commands\n\n"
        ts_cheat_content += "```bash\n# Check feature status\nget FeatureState.featureState FeatureState=<CXC_CODE>\n\n# Check alarm status\nlst Alarm\n\n# Check performance counters\nget <CounterName> <PM_Object>\n\n# Check cell status\nget UtranCellFDD.operationalState UtranCellFDD=<CellId>\n```\n\n"

        ts_cheat_content += "## Common Issues and Solutions\n\n"

        ts_cheat_content += "### Feature Not Working\n\n"
        ts_cheat_content += "**Checks**:\n"
        ts_cheat_content += "1. Verify feature activation: `get FeatureState.featureState FeatureState=<CXC_CODE>`\n"
        ts_cheat_content += "2. Check prerequisites are activated\n"
        ts_cheat_content += "3. Verify software version compatibility\n"
        ts_cheat_content += "4. Check for related alarms\n\n"

        ts_cheat_content += "### Performance Degradation\n\n"
        ts_cheat_content += "**Checks**:\n"
        ts_cheat_content += "1. Review performance counters\n"
        ts_cheat_content += "2. Check recent configuration changes\n"
        ts_cheat_content += "3. Analyze interference levels\n"
        ts_cheat_content += "4. Verify capacity utilization\n\n"

        ts_cheat_content += "### MIMO Issues\n\n"
        ts_cheat_content += "**Checks**:\n"
        ts_cheat_content += "1. Check UE MIMO capabilities\n"
        ts_cheat_content += "2. Verify antenna configuration\n"
        ts_cheat_content += "3. Review MIMO parameter settings\n"
        ts_cheat_content += "4. Monitor MIMO performance counters\n\n"

        ts_cheat_content += "### Energy Saving Issues\n\n"
        ts_cheat_content += "**Checks**:\n"
        ts_cheat_content += "1. Verify sleep mode thresholds\n"
        ts_cheat_content += "2. Check traffic patterns\n"
        ts_cheat_content += "3. Review timer configurations\n"
        ts_cheat_content += "4. Monitor QoS impact\n\n"

        # Write quick reference files
        qr_file = quick_ref_dir / "index.md"
        qr_file.write_text(qr_content)

        param_qr_file = quick_ref_dir / "parameters.md"
        param_qr_file.write_text(param_qr_content)

        counter_qr_file = quick_ref_dir / "counters.md"
        counter_qr_file.write_text(counter_qr_content)

        # Write cheat sheet files
        activation_cheat_file = cheat_sheet_dir / "activation_commands.md"
        activation_cheat_file.write_text(activation_cheat_content)

        ts_cheat_file = cheat_sheet_dir / "troubleshooting.md"
        ts_cheat_file.write_text(ts_cheat_content)

        # Generate configuration patterns
        patterns_content = "# Common Configuration Patterns\n\n"
        patterns_content += "Frequently used configuration patterns for Ericsson RAN features.\n\n"

        # Add common patterns based on functionality categories
        for func, features in feature_categories.items():
            if len(features) > 0:
                patterns_content += f"## {func.replace('_', ' ')} Configuration Patterns\n\n"

                # Extract common parameter patterns
                common_params = defaultdict(list)
                for feature in features:
                    for param in feature.get('parameters', []):
                        param_name = param.get('name', '')
                        if '.' in param_name and param.get('mo_class') != 'Unknown':
                            common_params[param_name].append(feature.get('name', 'Unknown'))

                # Show most common parameters for this functionality
                top_func_params = sorted(common_params.items(), key=lambda x: len(x[1]), reverse=True)[:5]

                if top_func_params:
                    patterns_content += "### Key Parameters\n\n"
                    for param_name, feature_list in top_func_params:
                        patterns_content += f"**{param_name}**\n"
                        patterns_content += f"- Used in: {', '.join(feature_list[:3])}\n"
                        patterns_content += f"- Total features: {len(feature_list)}\n\n"

                patterns_content += "### Configuration Example\n\n"
                patterns_content += "```bash\n# Example configuration template\n# [Add specific configuration commands]\n```\n\n"

        patterns_file = quick_ref_dir / "common_patterns.md"
        patterns_file.write_text(patterns_content)

        # Create cheat sheet index
        cheat_index_content = "# Cheat Sheets\n\n"
        cheat_index_content += "Quick reference cheat sheets for common operations.\n\n"
        cheat_index_content += "- [Activation Commands](activation_commands.md) - Feature activation quick reference\n"
        cheat_index_content += "- [Troubleshooting](troubleshooting.md) - Quick troubleshooting steps\n\n"

        cheat_index_file = cheat_sheet_dir / "index.md"
        cheat_index_file.write_text(cheat_index_content)

        print("✅ Generated quick references and cheat sheets")

    def generate_all_references(self):
        """Main method to generate all reference files"""
        print("🚀 Starting comprehensive reference generation...")

        # Load data
        self.load_data()

        # Generate all references
        self.generate_comprehensive_references()

        print("\n✅ All reference files generated successfully!")
        print(f"📁 Reference location: {self.refs_dir}")
        print(f"📊 Features processed: {len(self.features)}")

        # Generate summary
        total_params = sum(len(f.get('parameters', [])) for f in self.features.values())
        total_counters = sum(len(f.get('counters', [])) for f in self.features.values())

        print(f"⚙️  Total parameters: {total_params}")
        print(f"📈 Total counters: {total_counters}")

        # Count generated files
        file_count = len(list(self.refs_dir.rglob("*.md")))
        print(f"📄 Reference files created: {file_count}")


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate comprehensive reference documentation')
    parser.add_argument('--data-dir', default='output/ericsson_data', help='Processed data directory')
    parser.add_argument('--output-dir', default='output', help='Output directory')

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("Please run ericsson_feature_processor.py first")
        sys.exit(1)

    # Generate references
    generator = AdvancedReferenceGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    generator.generate_all_references()