#!/usr/bin/env python3
"""
Ericsson RAN Features Skill Generator
Creates Claude skill from processed feature data
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, List
from datetime import datetime


class EricssonSkillGenerator:
    """Generates Claude skill from processed Ericsson features"""

    def __init__(self, data_dir: str, output_dir: str = "output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.skill_dir = self.output_dir / "ericsson"

        # Load processed data
        self.load_data()

    def load_data(self):
        """Load all processed feature data"""
        print("📚 Loading processed feature data...")

        # Load features
        self.features = {}
        features_dir = self.data_dir / "features"

        for feature_file in features_dir.glob("*.json"):
            feature_data = json.loads(feature_file.read_text())
            self.features[feature_data['id']] = feature_data

        print(f"✅ Loaded {len(self.features)} features")

        # Load indices
        self.indices = {}
        indices_dir = self.data_dir / "indices"

        for index_file in indices_dir.glob("*_index.json"):
            index_name = index_file.stem.replace('_index', '')
            self.indices[index_name] = json.loads(index_file.read_text())

        # Load summary
        summary_file = self.data_dir / "summary.json"
        if summary_file.exists():
            self.summary = json.loads(summary_file.read_text())

    def generate_skill(self):
        """Generate complete Claude skill"""
        print("\n🚀 Generating Claude Skill for Ericsson RAN Features")

        # Create skill structure
        self.create_skill_structure()

        # Generate SKILL.md
        self.create_skill_md()

        # Generate reference files
        self.generate_references()

        # Package skill
        self.package_skill()

        print("\n✅ Skill generation complete!")

    def create_skill_structure(self):
        """Create skill directory structure"""
        print("📁 Creating skill structure...")

        # Create reference directories
        refs_dir = self.skill_dir / "references"
        directories = [
            refs_dir,
            refs_dir / "features",
            refs_dir / "features" / "by_category",
            refs_dir / "features" / "by_package",
            refs_dir / "parameters",
            refs_dir / "counters",
            refs_dir / "cxc_codes",
            refs_dir / "guidelines",
            refs_dir / "quick_reference"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        print("✅ Directory structure created")

    def create_skill_md(self):
        """Create main SKILL.md file"""
        print("📝 Creating SKILL.md...")

        skill_content = f"""# Ericsson RAN Features Expert

## Overview
This skill provides comprehensive access to Ericsson LTE/NR radio features, including:
- {self.summary['total_features']} feature descriptions with technical details
- {self.summary['total_parameters']} parameters with types and descriptions
- {self.summary['total_counters']} performance counters and KPI explanations
- Feature dependencies and relationships
- Engineering guidelines and best practices
- CXC feature codes for activation/deactivation

## When to Use This Skill

Use this skill when you need to:
- Understand Ericsson radio feature capabilities
- Configure feature parameters
- Activate or deactivate features
- Troubleshoot feature-related issues
- Plan feature deployments
- Understand feature interactions

## Capabilities

### Feature Information
- Get complete feature description: "Tell me about FAJ 121 3094"
- List features by category: "Show all MIMO features"
- Find features by parameter: "Which features use MimoSleepFunction?"
- Find feature by CXC code: "What is CXC4011808?"

### Technical Details
- Parameter lookup: "What does MimoSleepFunction.mimoSleepMode do?"
- Counter explanations: "Explain pmMimoSleepTime counter"
- Feature impact: "What is the network impact of MIMO Sleep Mode?"

### Activation and Configuration
- Get activation commands: "How do I activate CXC4011808?"
- Get deactivation commands: "How to deactivate MIMO Sleep Mode?"
- Prerequisites checking: "What do I need before activating this feature?"

### Engineering Support
- Configuration guidelines: "How should I configure MIMO Sleep Mode?"
- Best practices: "What are recommended settings for energy saving?"
- Troubleshooting: "Why is my feature not working?"

## Quick Reference

### Common Feature Categories
"""

        # Add categories
        if 'feature_categories' in self.summary:
            for category, count in sorted(self.summary['feature_categories'].items()):
                skill_content += f"- **{category}**: {count} features\n"

        skill_content += """
### Access Patterns
- FAJ ID format: FAJ XXX XXXX (e.g., FAJ 121 3094)
- CXC Code format: CXC followed by numbers (e.g., CXC4011808)
- Parameter format: MOClass.parameterName
- Counter format: pmCounterName

## Reference Files
- `references/features/` - Complete feature documentation
- `references/parameters/` - Parameter master index
- `references/counters/` - Performance counter reference
- `references/cxc_codes/` - Activation code index
- `references/guidelines/` - Engineering guidelines

## Usage Examples

### Example 1: Feature Lookup
**User**: "Tell me about MIMO Sleep Mode feature"
**Response**: Provides complete feature details including FAJ ID, CXC code, parameters, activation steps, and engineering guidelines.

### Example 2: Activation Help
**User**: "How to activate CXC4011808?"
**Response**: Provides exact activation command, prerequisites, and any related features.

### Example 3: Configuration
**User**: "What are the recommended settings for MIMO Sleep Mode?"
**Response**: Provides configuration guidelines and best practices from engineering documentation.

## Notes
- This skill contains documentation for Ericsson Radio System features
- Always check prerequisites before activating features
- Verify compatibility with your specific node type and software version
- Consult engineering guidelines for optimal configuration

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Features: {self.summary['total_features']}
"""

        skill_file = self.skill_dir / "SKILL.md"
        skill_file.write_text(skill_content)

        print("✅ SKILL.md created")

    def generate_references(self):
        """Generate all reference files"""
        print("📚 Generating reference files...")

        # Feature indexes
        self.generate_feature_indexes()

        # Feature details (sample for demo)
        self.generate_feature_samples()

        # Parameter index
        self.generate_parameter_index()

        # Counter index
        self.generate_counter_index()

        # CXC code index
        self.generate_cxc_index()

        # Engineering guidelines
        self.generate_guidelines()

        # Quick reference
        self.generate_quick_reference()

        print("✅ Reference files generated")

    def generate_feature_indexes(self):
        """Generate feature index files"""
        refs_dir = self.skill_dir / "references" / "features"

        # Master index
        with open(refs_dir / "index.md", 'w') as f:
            f.write("# Ericsson Radio Features Index\n\n")
            f.write(f"**Total Features**: {len(self.features)}\n\n")

            # Group by category
            categories = {}
            for feature in self.features.values():
                category = self.categorize_feature(feature)
                if category not in categories:
                    categories[category] = []
                categories[category].append(feature)

            for category, features in sorted(categories.items()):
                f.write(f"## {category} ({len(features)})\n\n")
                for feature in sorted(features[:20], key=lambda x: x['name']):
                    f.write(f"- {feature['name']} (FAJ {feature['id']}")
                    if feature.get('cxc_code'):
                        f.write(f", CXC {feature['cxc_code']}")
                    f.write(")\n")
                if len(features) > 20:
                    f.write(f"- ... and {len(features) - 20} more features\n")
                f.write("\n")

        # By value package
        with open(refs_dir / "by_package" / "index.md", 'w') as f:
            f.write("# Features by Value Package\n\n")

            packages = {}
            for feature in self.features.values():
                package = feature.get('value_package', 'Unknown')
                if package not in packages:
                    packages[package] = []
                packages[package].append(feature)

            for package, features in sorted(packages.items()):
                f.write(f"## {package}\n\n")
                for feature in features[:10]:
                    f.write(f"- {feature['name']} (FAJ {feature['id']})\n")
                f.write("\n")

    def categorize_feature(self, feature: Dict) -> str:
        """Categorize a feature"""
        name = feature['name'].lower()

        if 'mimo' in name:
            return 'MIMO Features'
        elif 'sleep' in name or 'energy' in name:
            return 'Energy Efficiency'
        elif 'carrier' in name or 'aggregation' in name:
            return 'Carrier Aggregation'
        elif 'handover' in name or 'mobility' in name:
            return 'Mobility Management'
        elif 'dual' in name:
            return 'Dual Connectivity'
        else:
            return 'Other Features'

    def generate_feature_samples(self):
        """Generate sample feature files (first 10 for demo)"""
        refs_dir = self.skill_dir / "references" / "features"
        sample_features = list(self.features.values())[:10]

        for feature in sample_features:
            filename = f"FAJ_{feature['id'].replace(' ', '_')}.md"
            filepath = refs_dir / filename

            content = f"""# {feature['name']}

**FAJ ID**: FAJ {feature['id']}
"""
            if feature.get('cxc_code'):
                content += f"**CXC Code**: {feature['cxc_code']}  \n"
            content += f"""**Access Type**: {feature.get('access_type', 'N/A')}
**Value Package**: {feature.get('value_package', 'N/A')}
**Node Type**: {feature.get('node_type', 'N/A')}

## Description
{feature.get('description', feature.get('summary', 'No description available'))}

## Activation
"""
            if feature.get('activation_step'):
                content += f"```bash\n{feature['activation_step']}\n```\n\n"
            else:
                content += "Activation steps not documented\n\n"

            if feature.get('deactivation_step'):
                content += "## Deactivation\n"
                content += f"```bash\n{feature['deactivation_step']}\n```\n\n"

            if feature.get('parameters'):
                content += f"## Parameters ({len(feature['parameters'])})\n\n"
                for param in feature['parameters'][:10]:
                    content += f"### {param['name']}\n"
                    content += f"- **Type**: {param.get('type', 'N/A')}\n"
                    content += f"- **MO Class**: {param.get('mo_class', 'N/A')}\n"
                    content += f"- **Description**: {param.get('description', 'N/A')}\n\n"

            if feature.get('counters'):
                content += f"## Performance Counters ({len(feature['counters'])})\n\n"
                for counter in feature['counters'][:5]:
                    content += f"- **{counter['name']}**: {counter.get('description', 'N/A')}\n"

            if feature.get('engineering_guidelines'):
                content += "\n## Engineering Guidelines\n\n"
                content += feature['engineering_guidelines'][:500]
                if len(feature['engineering_guidelines']) > 500:
                    content += "..."
                content += "\n"

            filepath.write_text(content)

    def generate_parameter_index(self):
        """Generate parameter master index"""
        refs_dir = self.skill_dir / "references" / "parameters"

        with open(refs_dir / "index.md", 'w') as f:
            f.write("# Parameter Master Index\n\n")

            # Group by MO class
            mo_params = {}
            if 'parameters' in self.indices:
                for param_name, feature_ids in self.indices['parameters'].items():
                    for fid in feature_ids[:3]:  # Limit to 3 features per param
                        if fid in self.features:
                            feature = self.features[fid]
                            for param in feature.get('parameters', []):
                                if param_name in param['name'].lower():
                                    mo_class = param.get('mo_class', 'Unknown')
                                    if mo_class not in mo_params:
                                        mo_params[mo_class] = []
                                    mo_params[mo_class].append((param, feature))
                                    break

            for mo_class, params in sorted(mo_params.items()):
                f.write(f"## {mo_class}\n\n")
                for param, feature in params[:10]:  # Limit to 10 per MO
                    f.write(f"- **{param['name']}** - Used in {feature['name']} (FAJ {feature['id']})\n")
                f.write("\n")

    def generate_counter_index(self):
        """Generate counter index"""
        refs_dir = self.skill_dir / "references" / "counters"

        with open(refs_dir / "index.md", 'w') as f:
            f.write("# Performance Counter Index\n\n")

            if 'counters' in self.indices:
                for counter_name, feature_ids in sorted(self.indices['counters'].items())[:50]:
                    f.write(f"## {counter_name}\n\n")
                    f.write(f"**Used in {len(feature_ids)} features**:\n\n")

                    for fid in feature_ids[:5]:
                        if fid in self.features:
                            feature = self.features[fid]
                            f.write(f"- {feature['name']} (FAJ {feature['id']})\n")
                    f.write("\n")

    def generate_cxc_index(self):
        """Generate CXC code index"""
        refs_dir = self.skill_dir / "references" / "cxc_codes"

        with open(refs_dir / "index.md", 'w') as f:
            f.write("# CXC Feature Code Index\n\n")
            f.write("Quick reference for feature activation codes:\n\n")

            for cxc_code, feature_id in sorted(self.indices.get('cxc_codes', {}).items()):
                if feature_id in self.features:
                    feature = self.features[feature_id]
                    f.write(f"## {cxc_code}\n\n")
                    f.write(f"**Feature**: {feature['name']}\n")
                    f.write(f"**FAJ ID**: FAJ {feature['id']}\n")
                    f.write(f"**Access Type**: {feature.get('access_type', 'N/A')}\n\n")

                    if feature.get('activation_step'):
                        f.write("**Activation**:\n```bash\n")
                        f.write(feature['activation_step'])
                        f.write("\n```\n\n")

                    if feature.get('deactivation_step'):
                        f.write("**Deactivation**:\n```bash\n")
                        f.write(feature['deactivation_step'])
                        f.write("\n```\n\n")

    def generate_guidelines(self):
        """Generate engineering guidelines compilation"""
        refs_dir = self.skill_dir / "references" / "guidelines"

        with open(refs_dir / "index.md", 'w') as f:
            f.write("# Engineering Guidelines\n\n")
            f.write("Collection of engineering guidelines from features:\n\n")

            # Group by category
            guidelines_by_category = {
                'MIMO': [],
                'Energy Efficiency': [],
                'Configuration': [],
                'Troubleshooting': [],
                'Best Practices': []
            }

            for feature in list(self.features.values())[:20]:  # Sample for demo
                guidelines = feature.get('engineering_guidelines', '')
                if guidelines:
                    category = 'Best Practices'
                    name = feature['name'].lower()
                    if 'mimo' in name:
                        category = 'MIMO'
                    elif 'sleep' in name or 'energy' in name:
                        category = 'Energy Efficiency'
                    elif 'configur' in name:
                        category = 'Configuration'

                    guidelines_by_category[category].append((feature['name'], guidelines[:500]))

            for category, guidelines in guidelines_by_category.items():
                if guidelines:
                    f.write(f"## {category}\n\n")
                    for feature_name, guideline in guidelines[:5]:
                        f.write(f"### {feature_name}\n\n")
                        f.write(f"{guideline}...\n\n")

    def generate_quick_reference(self):
        """Generate quick reference guide"""
        refs_dir = self.skill_dir / "references" / "quick_reference"

        with open(refs_dir / "common_patterns.md", 'w') as f:
            f.write("# Common Feature Patterns\n\n")
            f.write("## Frequently Used Features\n\n")

            # Find features with most parameters
            most_params = sorted(
                self.features.values(),
                key=lambda x: len(x.get('parameters', [])),
                reverse=True
            )[:10]

            f.write("### Features with Most Parameters\n\n")
            for feature in most_params:
                f.write(f"- {feature['name']} (FAJ {feature['id']}) - {len(feature.get('parameters', []))} parameters\n")

            f.write("\n### Energy Saving Features\n\n")
            for feature in self.features.values():
                if 'sleep' in feature['name'].lower() or 'energy' in feature['name'].lower():
                    f.write(f"- {feature['name']} (FAJ {feature['id']})")
                    if feature.get('cxc_code'):
                        f.write(f" - CXC {feature['cxc_code']}")
                    f.write("\n")

    def package_skill(self):
        """Package skill into zip file"""
        print("📦 Packaging skill...")

        zip_filename = f"ericsson_ran_features_skill_{len(self.features)}_features.zip"
        zip_path = self.output_dir / zip_filename

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in skill directory
            for root, dirs, files in os.walk(self.skill_dir):
                for file in files:
                    if not file.endswith('.backup'):  # Skip backup files
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.skill_dir)
                        zipf.write(file_path, arcname)

        # Get file size
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)

        print(f"✅ Skill packaged: {zip_filename}")
        print(f"📊 Package size: {size_mb:.2f} MB")
        print(f"\nNext steps:")
        print(f"1. Upload {zip_filename} to Claude")
        print(f"2. Test with sample queries")
        print(f"3. Share with team for feedback")


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate Claude skill from Ericsson features')
    parser.add_argument('--data-dir', default='output/ericsson_data', help='Processed data directory')
    parser.add_argument('--output-dir', default='output', help='Output directory')

    args = parser.parse_args()

    # Check if data exists
    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("Please run ericsson_feature_processor.py first")
        sys.exit(1)

    # Generate skill
    generator = EricssonSkillGenerator(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    generator.generate_skill()