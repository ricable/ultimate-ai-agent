# Ericsson Features Documentation Processing Plan
## Integrating Skill_Seekers for Local Technical Documentation Processing

### Executive Summary

This plan outlines the integration of the existing Skill_Seekers documentation processing framework with Ericsson's technical feature documentation to create a comprehensive, operational Claude skill. The goal is to process all feature descriptions in the `elex_features_only/` directory into a fully functional skill that can query, analyze, and provide insights about Ericsson features, their parameters, counters, events, and engineering guidelines.

### Project Overview

- **Source Code**: `Skill_Seekers/` directory - Existing web-based documentation scraper
- **Target Data**: `elex_features_only/` directory - Local Ericsson feature documentation (6 batches, ~2000+ features)
- **Output**: Single Claude skill zip file containing comprehensive feature knowledge
- **Focus Areas**: Parameters, Counters, Events, Feature Relationships, Engineering Guidelines

### Phase 1: Analysis and Adaptation (Days 1-2)

#### 1.1 Codebase Analysis
- ✅ Analyzed Skill_Seekers architecture (`doc_scraper.py` - 940 lines)
- ✅ Understanding Ericsson documentation structure from 15 random samples
- ✅ Identified key differences between web scraping and local file processing

#### 1.2 Architecture Adaptation
**Current Skill_Seekers Flow:**
```
Web URL → BFS Crawling → Content Extraction → AI Enhancement → Skill Packaging
```

**Required Local Processing Flow:**
```
Local Files → Markdown Parsing → Feature Extraction → Relationship Mapping → Skill Generation
```

### Phase 2: Core Processing Engine Development (Days 3-5)

#### 2.1 Create `ericsson_processor.py`
Based on `doc_scraper.py` structure, modified for local file processing:

```python
#!/usr/bin/env python3
"""
Ericsson Feature Documentation Processor
Adapted from Skill_Seekers doc_scraper.py for local markdown processing
"""

import os
import sys
import json
import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque
import markdown
from bs4 import BeautifulSoup


@dataclass
class EricssonFeature:
    """Data model for Ericsson feature documentation"""
    id: str  # FAJ XXX XXXX
    name: str
    value_package: str
    access_type: str
    node_type: str
    cxc_code: Optional[str] = None
    parameters: List[Dict] = field(default_factory=list)
    counters: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)
    dependencies: Dict = field(default_factory=dict)
    engineering_guidelines: str = ""
    network_impact: Dict = field(default_factory=dict)
    activation_procedure: str = ""
    activation_step: Optional[str] = None
    deactivation_step: Optional[str] = None
    content: str = ""
    url: str = ""


class EricssonFeatureProcessor:
    """Process Ericsson feature documentation from local markdown files"""

    def __init__(self):
        self.features = {}
        self.parameter_index = defaultdict(list)
        self.counter_index = defaultdict(list)
        self.event_index = defaultdict(list)
        self.cxc_index = {}
        self.relationship_graph = {}
        self.processed_files = 0
        self.errors = []

        # Output directories
        self.data_dir = "output/ericsson_data"
        self.skill_dir = "output/ericsson"

        # Create directories
        os.makedirs(f"{self.data_dir}/features", exist_ok=True)
        os.makedirs(f"{self.skill_dir}/references", exist_ok=True)
        os.makedirs(f"{self.skill_dir}/scripts", exist_ok=True)
        os.makedirs(f"{self.skill_dir}/assets", exist_ok=True)

    def process_directory(self, directory_path):
        """Process all markdown files in directory structure"""
        print(f"🔍 Processing Ericsson features from: {directory_path}")

        # Find all markdown files recursively
        md_files = list(Path(directory_path).rglob("*.md"))
        print(f"📊 Found {len(md_files)} markdown files")

        for file_path in md_files:
            try:
                feature = self.parse_feature_file(file_path)
                if feature:
                    self.features[feature.id] = feature
                    self.update_indices(feature)
                    self.processed_files += 1

                if self.processed_files % 100 == 0:
                    print(f"  Processed {self.processed_files} features...")

            except Exception as e:
                self.errors.append(f"Error processing {file_path}: {str(e)}")
                print(f"  ✗ Error: {file_path} - {e}")

        print(f"\n✅ Processed {self.processed_files} features")
        if self.errors:
            print(f"⚠ Encountered {len(self.errors)} errors")

        # Save processing summary
        self.save_summary()

    def parse_feature_file(self, file_path):
        """Parse a single Ericsson feature markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Convert markdown to HTML for easier parsing
        html = markdown.markdown(content, extensions=['tables', 'fenced_code'])
        soup = BeautifulSoup(html, 'html.parser')

        # Extract feature identity
        feature = self.extract_feature_identity(content, soup)
        if not feature:
            return None

        feature.url = str(file_path)
        feature.content = content

        # Extract components
        feature.parameters = self.extract_parameters(soup)
        feature.counters = self.extract_counters(soup)
        feature.events = self.extract_events(soup)
        feature.dependencies = self.extract_dependencies(soup)
        feature.cxc_code = self.extract_cxc_code(content)
        feature.activation_step = self.extract_activation_step(content)
        feature.deactivation_step = self.extract_deactivation_step(content)
        feature.engineering_guidelines = self.extract_engineering_guidelines(soup)

        # Save feature data
        self.save_feature(feature)

        return feature

    def extract_feature_identity(self, content, soup):
        """Extract FAJ number, name, and other identity information"""
        # Look for feature identity table
        identity_pattern = r'Feature Identity\s*\|\s*FAJ\s+(\d+\s+\d+)'
        cxc_pattern = r'FeatureState=(CXC\d+)\s+MO instance'

        # Extract FAJ number
        faj_match = re.search(identity_pattern, content)
        if not faj_match:
            return None

        faj_id = faj_match.group(1)

        # Extract feature name (usually in first h1 or h2)
        name_elem = soup.find(['h1', 'h2'])
        name = name_elem.get_text().strip() if name_elem else f"Feature {faj_id}"

        # Extract other metadata
        value_package = self.extract_table_field(content, "Value Package")
        access_type = self.extract_table_field(content, "Access Type")
        node_type = self.extract_table_field(content, "Node Type")

        return EricssonFeature(
            id=faj_id,
            name=name,
            value_package=value_package or "Unknown",
            access_type=access_type or "Unknown",
            node_type=node_type or "Unknown"
        )

    def extract_table_field(self, content, field_name):
        """Extract value from markdown table"""
        pattern = rf'{field_name}\s*\|\s*([^\n|]+)'
        match = re.search(pattern, content, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def extract_parameters(self, soup):
        """Extract parameters from feature documentation"""
        parameters = []

        # Look for parameter tables
        tables = soup.find_all('table')
        for table in tables:
            headers = table.find_all('th')
            if headers and any('parameter' in th.get_text().lower() for th in headers):
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        param = {
                            'name': cells[0].get_text().strip(),
                            'type': cells[1].get_text().strip(),
                            'description': cells[2].get_text().strip(),
                            'mo_class': self.extract_mo_class(cells[0].get_text())
                        }
                        parameters.append(param)

        return parameters

    def extract_mo_class(self, param_name):
        """Extract MO class from parameter name"""
        # Common patterns
        if '.' in param_name:
            return param_name.split('.')[0]
        return "Unknown"

    def extract_counters(self, soup):
        """Extract PM counters from feature documentation"""
        counters = []

        # Look for counter mentions in text
        text = soup.get_text()

        # Pattern for pm counters
        counter_pattern = r'pm(\w+)'
        matches = re.findall(counter_pattern, text)

        # Also look in code blocks
        code_blocks = soup.find_all('code')
        for code in code_blocks:
            matches.extend(re.findall(counter_pattern, code.get_text()))

        # Remove duplicates
        unique_counters = list(set(matches))

        for counter in unique_counters:
            counters.append({
                'name': f'pm{counter}',
                'description': self.extract_counter_description(soup, counter)
            })

        return counters

    def extract_counter_description(self, soup, counter_name):
        """Extract description for a specific counter"""
        # Look for counter in tables or lists
        text = soup.get_text()

        # Look for counter near its mention
        pattern = rf'pm{counter_name}[^.]*\.'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return match.group(0).strip()

        return f"Performance counter {counter_name}"

    def extract_events(self, soup):
        """Extract events from feature documentation"""
        events = []

        # Look for event tables
        tables = soup.find_all('table')
        for table in tables:
            headers = table.find_all('th')
            if headers and any('event' in th.get_text().lower() for th in headers):
                rows = table.find_all('tr')[1:]  # Skip header
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 3:
                        event = {
                            'name': cells[0].get_text().strip(),
                            'parameter': cells[1].get_text().strip(),
                            'description': cells[2].get_text().strip()
                        }
                        events.append(event)

        return events

    def extract_dependencies(self, soup):
        """Extract feature dependencies and relationships"""
        dependencies = {
            'prerequisites': [],
            'related': [],
            'conflicts': []
        }

        # Look for dependency tables
        tables = soup.find_all('table')
        for table in tables:
            text = table.get_text().lower()

            if 'prerequisite' in text:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        dependencies['prerequisites'].append(cells[0].get_text().strip())

            elif 'related' in text:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        dependencies['related'].append(cells[0].get_text().strip())

            elif 'conflict' in text:
                rows = table.find_all('tr')[1:]
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        dependencies['conflicts'].append(cells[0].get_text().strip())

        return dependencies

    def extract_cxc_code(self, content):
        """Extract CXC feature code from activation section"""
        activation_pattern = r'FeatureState=(CXC\d+)\s+MO instance'
        match = re.search(activation_pattern, content)
        return match.group(1) if match else None

    def extract_activation_step(self, content):
        """Extract the exact activation step"""
        step_pattern = r'1\. Set the FeatureState\.featureState attribute to ACTIVATED in the FeatureState=(CXC\d+) MO instance\.'
        match = re.search(step_pattern, content)
        return match.group(0) if match else None

    def extract_deactivation_step(self, content):
        """Extract the exact deactivation step"""
        step_pattern = r'1\. Set the FeatureState\.featureState attribute to DEACTIVATED in the FeatureState=(CXC\d+) MO instance\.'
        match = re.search(step_pattern, content)
        return match.group(0) if match else None

    def extract_engineering_guidelines(self, soup):
        """Extract engineering guidelines from feature documentation"""
        guidelines = []

        # Look for sections with engineering guidelines
        headings = soup.find_all(['h2', 'h3', 'h4'])
        for heading in headings:
            text = heading.get_text().lower()
            if any(word in text for word in ['guideline', 'engineering', 'recommend', 'best practice']):
                # Get content until next heading
                content = []
                next_elem = heading.next_sibling
                while next_elem and next_elem.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if hasattr(next_elem, 'get_text'):
                        content.append(next_elem.get_text().strip())
                    next_elem = next_elem.next_sibling

                if content:
                    guidelines.append(' '.join(content))

        return '\n\n'.join(guidelines)

    def update_indices(self, feature):
        """Update search indices for fast lookup"""
        # Parameter index
        for param in feature.parameters:
            self.parameter_index[param['name'].lower()].append(feature.id)

        # Counter index
        for counter in feature.counters:
            self.counter_index[counter['name'].lower()].append(feature.id)

        # Event index
        for event in feature.events:
            self.event_index[event['name'].lower()].append(feature.id)

        # CXC index
        if feature.cxc_code:
            self.cxc_index[feature.cxc_code] = feature.id

    def save_feature(self, feature):
        """Save feature data to JSON file"""
        filename = f"feature_{feature.id.replace(' ', '_')}.json"
        filepath = os.path.join(self.data_dir, "features", filename)

        feature_data = {
            'id': feature.id,
            'name': feature.name,
            'value_package': feature.value_package,
            'access_type': feature.access_type,
            'node_type': feature.node_type,
            'cxc_code': feature.cxc_code,
            'parameters': feature.parameters,
            'counters': feature.counters,
            'events': feature.events,
            'dependencies': feature.dependencies,
            'engineering_guidelines': feature.engineering_guidelines,
            'activation_step': feature.activation_step,
            'deactivation_step': feature.deactivation_step,
            'url': feature.url
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(feature_data, f, indent=2, ensure_ascii=False)

    def save_summary(self):
        """Save processing summary"""
        summary = {
            'total_features': len(self.features),
            'processed_files': self.processed_files,
            'parameters_count': sum(len(f.parameters) for f in self.features.values()),
            'counters_count': sum(len(f.counters) for f in self.features.values()),
            'events_count': sum(len(f.events) for f in self.features.values()),
            'errors': self.errors[:10]  # Save first 10 errors
        }

        with open(f"{self.data_dir}/summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def build_feature_relationships(self):
        """Build feature relationship graph"""
        print("🔗 Building feature relationships...")

        for feature_id, feature in self.features.items():
            relationships = {
                'prerequisites': [],
                'related': [],
                'conflicts': []
            }

            # Process dependencies
            for dep_type, dep_list in feature.dependencies.items():
                for dep_name in dep_list:
                    # Try to find dependency feature by name
                    for other_id, other_feature in self.features.items():
                        if dep_name.lower() in other_feature.name.lower():
                            relationships[dep_type].append(other_id)
                            break

            self.relationship_graph[feature_id] = relationships

        print(f"✅ Built relationships for {len(self.relationship_graph)} features")

#### 2.2 Feature Extraction Components

**A. Markdown Parser Enhancement**
- Extend existing `extract_content()` function for local markdown files
- Handle YAML frontmatter extraction
- Parse Ericsson-specific table formats
- Extract code blocks and technical specifications

**B. Feature Identity Extraction**
```python
def extract_feature_identity(self, content):
    """Extract FAJ number, name, and CXC code from feature"""
    identity_pattern = r'Feature Identity\s*\|\s*FAJ\s+(\d+\s+\d+)'
    cxc_pattern = r'FeatureState=(CXC\d+)\s+MO instance'

    return {
        'faj_number': match.group(1),
        'cxc_code': cxc_match.group(1) if cxc_match else None,
        'name': self.extract_feature_name(content)
    }
```

**C. Parameter Extraction Engine**
```python
def extract_parameters(self, content):
    """Extract parameters from feature documentation"""
    param_pattern = r'\|\s*([^|]+)\s*\|\s*(Introduced|Affected|Affecting)\s*\|\s*([^|]+)\s*\|'
    return {
        'name': param_name,
        'type': param_type,
        'description': description,
        'mo_class': extract_mo_class(param_name)
    }
```

**C. Counter Extraction Engine**
```python
def extract_counters(self, content):
    """Extract PM counters from feature documentation"""
    counter_patterns = [
        r'- (EUtranCellFDD|EUtranCellTDD|EUtranFreqRelation)\.pm(\w+)',
        r'\|\s*pm(\w+)\s*\|\s*([^|]+)\s*\|'
    ]
```

**D. Event Extraction Engine**
```python
def extract_events(self, content):
    """Extract events and their parameters"""
    event_pattern = r'\|\s*(\w+)\s*\|\s*(EVENT_PARAM_\w+)\s*\|\s*([^|]+)\s*\|'
```

#### 2.3 Feature Relationship Mapping
```python
def build_feature_relationships(self):
    """Map feature dependencies and relationships"""
    for feature in self.features:
        relationships = self.extract_dependency_table(feature.content)
        self.relationship_graph[feature.id] = {
            'prerequisites': relationships.get('Prerequisite', []),
            'related': relationships.get('Related', []),
            'conflicts': relationships.get('Conflicting', [])
        }
```

**F. CXC Code Extraction Engine**
```python
def extract_cxc_code(self, content):
    """Extract CXC feature code from activation section"""
    # Look in activation/deactivation sections
    activation_pattern = r'FeatureState=(CXC\d+)\s+MO instance'

    # Handle both activation and deactivation sections
    matches = re.findall(activation_pattern, content)

    return {
        'cxc_code': matches[0] if matches else None,
        'activation_step': self.extract_activation_step(content),
        'deactivation_step': self.extract_deactivation_step(content)
    }

def extract_activation_step(self, content):
    """Extract the exact activation step"""
    step_pattern = r'1\. Set the FeatureState\.featureState attribute to ACTIVATED in the FeatureState=(CXC\d+) MO instance\.'
    match = re.search(step_pattern, content)
    return match.group(0) if match else None

def extract_deactivation_step(self, content):
    """Extract the exact deactivation step"""
    step_pattern = r'1\. Set the FeatureState\.featureState attribute to DEACTIVATED in the FeatureState=(CXC\d+) MO instance\.'
    match = re.search(step_pattern, content)
    return match.group(0) if match else None
```

### Phase 3: Data Structuring and Indexing (Days 6-7)

#### 3.1 Create Comprehensive Data Model
```python
@dataclass
class EricssonFeature:
    id: str  # FAJ XXX XXXX
    name: str
    value_package: str
    access_type: str
    node_type: str
    cxc_code: str  # CXC code for feature activation
    parameters: List[Parameter]
    counters: List[Counter]
    events: List[Event]
    dependencies: Dict[str, List[str]]
    engineering_guidelines: str
    network_impact: Dict
    activation_procedure: str
    activation_step: str  # Exact activation command
    deactivation_step: str  # Exact deactivation command
```

#### 3.2 Build Search Indices
```python
def build_search_indices(self):
    """Build fast lookup indices for queries"""
    # Parameter index - Find features by parameter
    self.parameter_index = {}
    for param in self.all_parameters:
        self.parameter_index[param.name.lower()] = param.features

    # Counter index - Find features by counter
    self.counter_index = {}
    for counter in self.all_counters:
        self.counter_index[counter.name.lower()] = counter.feature

    # CXC code index - Find features by CXC code
    self.cxc_index = {}
    for feature in self.features:
        if feature.cxc_code:
            self.cxc_index[feature.cxc_code] = feature

    # Feature relationship graph for dependency queries
    self.dependency_graph = self.build_graph()
```

### Phase 4: Skill Generation (Days 8-9)

#### 4.1 Enhanced SKILL.md Generation
Create specialized skill description for Ericsson features:

```markdown
# Ericsson Radio Features Expert

## Overview
This skill provides comprehensive access to Ericsson LTE/NR radio features, including:
- 2000+ feature descriptions with technical details
- Complete parameter documentation with types and descriptions
- PM counters and KPI explanations
- Event definitions and triggers
- Feature dependencies and relationships
- Engineering guidelines and best practices
- CXC feature codes for activation/deactivation

## Capabilities

### Feature Information
- Get complete feature description: "Tell me about feature FAJ 121 5201"
- List features by category: "Show all MIMO-related features"
- Find features by parameter: "Which features use initPreschedulingEnable?"
- Find feature by CXC code: "What is feature CXC4011253?"

### Technical Details
- Parameter lookup: "What does ENodeBFunction.initPreschedulingEnable do?"
- Counter explanations: "Explain pmMimoSleepTime counter"
- Event details: "When is INTERNAL_PROC_UE_CTXT_RELEASE triggered?"

### Activation and Configuration
- Get activation commands: "How do I activate feature CXC4011072?"
- Get deactivation commands: "How do I deactivate TTI Bundling?"
- Exact MO configuration steps: "Show the exact command to activate FAJ 121 2051"

### Dependencies and Relationships
- Prerequisite checking: "What do I need for NR DL Carrier Aggregation?"
- Conflict detection: "Does feature X conflict with feature Y?"
- Feature grouping: "Show all features in the same value package"

### Engineering Support
- Configuration guidelines: "How do I configure MIMO Sleep Mode?"
- Troubleshooting: "What are common issues with CA configuration?"
- Best practices: "What are the recommended settings for feature X?"
```

#### 4.2 Reference File Organization
```
references/
├── features/
│   ├── index.md  # Master feature list
│   ├── by_category.md  # Features grouped by category
│   ├── by_access_type.md  # LTE vs NR features
│   ├── by_value_package.md  # Features by package
│   └── by_cxc_code.md  # Features by CXC code
├── parameters/
│   ├── index.md  # Parameter master index
│   ├── introduced.md  # New parameters by feature
│   ├── modified.md  # Modified parameters
│   └── by_mo_class.md  # Parameters by managed object
├── counters/
│   ├── index.md  # Counter master index
│   ├── by_mo_class.md  # Counters by object
│   └── kpi_impact.md  # KPI relationships
├── events/
│   ├── index.md  # Event master index
│   └── by_trigger.md  # Events by trigger type
├── cxc_codes/
│   ├── index.md  # CXC code master index
│   ├── activation_commands.md  # Exact activation steps
│   └── deactivation_commands.md  # Exact deactivation steps
├── relationships/
│   ├── dependencies.md  # Feature dependency graph
│   ├── conflicts.md  # Feature conflicts
│   └── prerequisites.md  # Prerequisite chains
└── engineering/
    ├── guidelines.md  # Engineering guidelines index
    ├── configuration.md  # Configuration examples
    └── troubleshooting.md  # Common issues
```

### Phase 5: Query Processing and Response Generation (Days 10-11)

#### 5.1 Create `query_engine.py`
```python
#!/usr/bin/env python3
"""
Ericsson Feature Query Engine
Handles natural language queries about Ericsson features
"""

import re
from typing import Dict, List, Optional, Tuple


class EricssonQueryEngine:
    """Advanced query engine for Ericsson feature documentation"""

    def __init__(self, processor):
        self.processor = processor
        self.features = processor.features
        self.parameter_index = processor.parameter_index
        self.counter_index = processor.counter_index
        self.event_index = processor.event_index
        self.cxc_index = processor.cxc_index
        self.relationship_graph = processor.relationship_graph

    def query(self, user_query: str) -> str:
        """Main query entry point"""
        query_lower = user_query.lower()

        # Route query based on patterns
        if "faj" in query_lower:
            return self.get_feature_by_faj(user_query)
        elif "cxc" in query_lower:
            return self.get_feature_by_cxc(user_query)
        elif "parameter" in query_lower or "param" in query_lower:
            return self.query_parameter(user_query)
        elif "counter" in query_lower or "pm" in query_lower:
            return self.query_counter(user_query)
        elif "event" in query_lower:
            return self.query_event(user_query)
        elif "activate" in query_lower or "deactivate" in query_lower:
            return self.get_activation_commands(user_query)
        elif "conflict" in query_lower or "prerequisite" in query_lower:
            return self.query_relationships(user_query)
        elif "list" in query_lower and ("mimo" in query_lower or "carrier" in query_lower):
            return self.list_features_by_category(user_query)
        else:
            return self.general_search(user_query)

    def get_feature_by_faj(self, query: str) -> str:
        """Find feature by FAJ number"""
        # Extract FAJ number
        faj_pattern = r'FAJ\s*(\d+\s+\d+)'
        match = re.search(faj_pattern, query, re.IGNORECASE)

        if not match:
            return "Please provide a valid FAJ number (e.g., FAJ 121 5201)"

        faj_id = match.group(1)
        feature = self.features.get(faj_id)

        if not feature:
            return f"No feature found with FAJ {faj_id}"

        return self.format_feature_response(feature)

    def get_feature_by_cxc(self, query: str) -> str:
        """Find feature by CXC code"""
        cxc_pattern = r'CXC\d+'
        cxc_match = re.search(cxc_pattern, query)

        if not cxc_match:
            return "Please provide a valid CXC code (e.g., CXC4011253)"

        cxc_code = cxc_match.group()
        feature_id = self.cxc_index.get(cxc_code)

        if not feature_id:
            return f"No feature found with CXC code {cxc_code}"

        feature = self.features[feature_id]
        return self.format_feature_response(feature)

    def format_feature_response(self, feature) -> str:
        """Format comprehensive feature response"""
        response = f"## {feature.name}\n\n"
        response += f"**FAJ ID**: FAJ {feature.id}\n"
        response += f"**CXC Code**: {feature.cxc_code or 'N/A'}\n"
        response += f"**Access Type**: {feature.access_type}\n"
        response += f"**Value Package**: {feature.value_package}\n"
        response += f"**Node Type**: {feature.node_type}\n\n"

        # Activation/Deactivation
        if feature.activation_step:
            response += "### Activation\n"
            response += f"```bash\n{feature.activation_step}\n```\n\n"

        if feature.deactivation_step:
            response += "### Deactivation\n"
            response += f"```bash\n{feature.deactivation_step}\n```\n\n"

        # Parameters
        if feature.parameters:
            response += f"### Parameters ({len(feature.parameters)})\n"
            for param in feature.parameters[:10]:  # Limit to first 10
                response += f"- **{param['name']}** ({param['mo_class']})\n"
                response += f"  - {param['description']}\n"
            if len(feature.parameters) > 10:
                response += f"- ... and {len(feature.parameters) - 10} more parameters\n"
            response += "\n"

        # Counters
        if feature.counters:
            response += f"### Performance Counters ({len(feature.counters)})\n"
            for counter in feature.counters[:10]:  # Limit to first 10
                response += f"- **{counter['name']}**: {counter['description']}\n"
            if len(feature.counters) > 10:
                response += f"- ... and {len(feature.counters) - 10} more counters\n"
            response += "\n"

        # Events
        if feature.events:
            response += f"### Events ({len(feature.events)})\n"
            for event in feature.events[:5]:  # Limit to first 5
                response += f"- **{event['name']}**: {event['description']}\n"
            response += "\n"

        # Dependencies
        deps = feature.dependencies
        if deps.get('prerequisites') or deps.get('conflicts'):
            response += "### Dependencies\n"
            if deps.get('prerequisites'):
                response += "**Prerequisites**:\n"
                for prereq in deps['prerequisites'][:5]:
                    response += f"- {prereq}\n"
            if deps.get('conflicts'):
                response += "**Conflicts**:\n"
                for conflict in deps['conflicts'][:5]:
                    response += f"- {conflict}\n"
            response += "\n"

        # Engineering Guidelines
        if feature.engineering_guidelines:
            response += "### Engineering Guidelines\n"
            # Truncate if too long
            guidelines = feature.engineering_guidelines[:500]
            response += f"{guidelines}...\n\n"

        return response

    def query_parameter(self, query: str) -> str:
        """Find features using a parameter"""
        # Extract parameter name from query
        param_patterns = [
            r'parameter\s+(\w+)',
            r'param\s+(\w+)',
            r'(\w+\.\w+)',  # MO.param format
        ]

        for pattern in param_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                param_name = match.group(1).lower()

                # Search in parameter index
                if param_name in self.parameter_index:
                    feature_ids = self.parameter_index[param_name]
                    response = f"## Parameter: {param_name}\n\n"
                    response += f"Used in {len(feature_ids)} features:\n\n"

                    for fid in feature_ids[:10]:  # Limit to 10 features
                        feature = self.features[fid]
                        response += f"- **{feature.name}** (FAJ {feature.id}"
                        if feature.cxc_code:
                            response += f", CXC {feature.cxc_code}"
                        response += ")\n"

                    if len(feature_ids) > 10:
                        response += f"\n... and {len(feature_ids) - 10} more features\n"

                    # Get parameter details from first feature
                    if feature_ids:
                        feature = self.features[feature_ids[0]]
                        for param in feature.parameters:
                            if param_name in param['name'].lower():
                                response += f"\n### Parameter Details\n"
                                response += f"**Type**: {param['type']}\n"
                                response += f"**MO Class**: {param['mo_class']}\n"
                                response += f"**Description**: {param['description']}\n"
                                break

                    return response

        return "Parameter not found. Please specify the parameter name."

    def query_counter(self, query: str) -> str:
        """Explain a performance counter"""
        # Extract counter name
        counter_pattern = r'pm(\w+)'
        match = re.search(counter_pattern, query, re.IGNORECASE)

        if not match:
            return "Please specify a counter name (e.g., pmMimoSleepTime)"

        counter_name = f"pm{match.group(1).lower()}"

        # Find features using this counter
        if counter_name in self.counter_index:
            feature_ids = self.counter_index[counter_name]
            response = f"## Performance Counter: {counter_name}\n\n"
            response += f"Monitored in {len(feature_ids)} features:\n\n"

            for fid in feature_ids[:10]:
                feature = self.features[fid]
                response += f"- **{feature.name}** (FAJ {feature.id})\n"

            # Get counter description from first feature
            if feature_ids:
                feature = self.features[feature_ids[0]]
                for counter in feature.counters:
                    if counter_name.lower() in counter['name'].lower():
                        response += f"\n### Description\n{counter['description']}\n"
                        break

            return response

        return f"Counter {counter_name} not found"

    def get_activation_commands(self, query: str) -> str:
        """Provide exact activation/deactivation commands"""
        # Extract feature from query
        feature = self.extract_feature_from_query(query)

        if not feature:
            return "Feature not found in query"

        is_activation = "activate" in query.lower()

        response = f"## {feature.name}\n"
        response += f"FAJ {feature.id}"

        if feature.cxc_code:
            response += f" | CXC {feature.cxc_code}"

        response += "\n\n"

        if is_activation and feature.activation_step:
            response += "### Activation Procedure\n"
            response += feature.activation_step + "\n\n"

            # Check prerequisites
            deps = self.relationship_graph.get(feature.id, {})
            if deps.get('prerequisites'):
                response += "**Prerequisites**:\n"
                for prereq_id in deps['prerequisites'][:5]:
                    prereq = self.features.get(prereq_id)
                    if prereq:
                        response += f"- {prereq.name} (FAJ {prereq_id})"
                        if prereq.cxc_code:
                            response += f" - Activate CXC {prereq.cxc_code} first"
                        response += "\n"

        elif not is_activation and feature.deactivation_step:
            response += "### Deactivation Procedure\n"
            response += feature.deactivation_step + "\n\n"

            # Check conflicts
            deps = self.relationship_graph.get(feature.id, {})
            if deps.get('conflicts'):
                response += "**Note**: This feature conflicts with:\n"
                for conflict_id in deps['conflicts'][:5]:
                    conflict = self.features.get(conflict_id)
                    if conflict:
                        response += f"- {conflict.name} (FAJ {conflict_id})\n"

        else:
            response += "Activation/deactivation commands not available for this feature.\n"

        return response

    def extract_feature_from_query(self, query: str) -> Optional:
        """Extract feature from query by various identifiers"""
        # Try FAJ
        faj_match = re.search(r'FAJ\s*(\d+\s+\d+)', query, re.IGNORECASE)
        if faj_match:
            return self.features.get(faj_match.group(1))

        # Try CXC
        cxc_match = re.search(r'CXC(\d+)', query, re.IGNORECASE)
        if cxc_match:
            cxc_code = cxc_match.group(0)
            feature_id = self.cxc_index.get(cxc_code)
            if feature_id:
                return self.features[feature_id]

        # Try name matching
        query_words = query.lower().split()
        best_match = None
        best_score = 0

        for feature_id, feature in self.features.items():
            score = 0
            name_words = feature.name.lower().split()

            for qw in query_words:
                for nw in name_words:
                    if qw == nw:
                        score += 2
                    elif qw in nw or nw in qw:
                        score += 1

            if score > best_score and score >= 3:
                best_score = score
                best_match = feature

        return best_match

    def list_features_by_category(self, query: str) -> str:
        """List features by category (MIMO, CA, etc.)"""
        query_lower = query.lower()
        category_keywords = {
            'mimo': ['mimo', 'multiple input', 'multiple output'],
            'carrier': ['carrier', 'ca', 'aggregation'],
            'sleep': ['sleep', 'power save', 'energy'],
            'handover': ['handover', 'handoff', 'mobility'],
            'dual': ['dual', 'dual connectivity', 'dc']
        }

        matched_features = []

        for category, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                for feature in self.features.values():
                    content = (feature.name + ' ' + feature.content).lower()
                    if any(kw in content for kw in keywords):
                        matched_features.append(feature)
                break

        if not matched_features:
            return "No features found for the specified category"

        response = f"## Found {len(matched_features)} features\n\n"

        for feature in matched_features[:20]:  # Limit to 20
            response += f"- **{feature.name}** (FAJ {feature.id}"
            if feature.cxc_code:
                response += f", CXC {feature.cxc_code}"
            response += f")\n  - {feature.access_type} | {feature.value_package}\n"

        if len(matched_features) > 20:
            response += f"\n... and {len(matched_features) - 20} more features"

        return response

    def query_relationships(self, query: str) -> str:
        """Query feature relationships (dependencies, conflicts)"""
        feature = self.extract_feature_from_query(query)

        if not feature:
            return "Feature not found"

        relationships = self.relationship_graph.get(feature.id, {})
        response = f"## Feature Relationships: {feature.name}\n\n"

        if relationships.get('prerequisites'):
            response += "### Prerequisites\n"
            for prereq_id in relationships['prerequisites']:
                prereq = self.features.get(prereq_id)
                if prereq:
                    response += f"- **{prereq.name}** (FAJ {prereq_id}"
                    if prereq.cxc_code:
                        response += f", CXC {prereq.cxc_code}"
                    response += ")\n"

        if relationships.get('conflicts'):
            response += "\n### Conflicts\n"
            for conflict_id in relationships['conflicts']:
                conflict = self.features.get(conflict_id)
                if conflict:
                    response += f"- **{conflict.name}** (FAJ {conflict_id}"
                    if conflict.cxc_code:
                        response += f", CXC {conflict.cxc_code}"
                    response += ")\n"

        if relationships.get('related'):
            response += "\n### Related Features\n"
            for related_id in relationships['related']:
                related = self.features.get(related_id)
                if related:
                    response += f"- **{related.name}** (FAJ {related_id}"
                    if related.cxc_code:
                        response += f", CXC {related.cxc_code}"
                    response += ")\n"

        if not any(relationships.values()):
            response += "No relationships found for this feature."

        return response

    def general_search(self, query: str) -> str:
        """General text search across all features"""
        query_words = query.lower().split()
        matches = []

        for feature in self.features.values():
            score = 0
            searchable_text = (feature.name + ' ' + feature.content).lower()

            for word in query_words:
                if word in searchable_text:
                    score += searchable_text.count(word)

            if score > 0:
                matches.append((feature, score))

        # Sort by score
        matches.sort(key=lambda x: x[1], reverse=True)

        if not matches:
            return "No matches found. Try searching for specific features, parameters, or counters."

        response = f"## Found {len(matches)} matching features\n\n"

        for feature, score in matches[:10]:
            response += f"- **{feature.name}** (FAJ {feature.id}"
            if feature.cxc_code:
                response += f", CXC {feature.cxc_code}"
            response += f") - Relevance: {score}\n"

        return response


# Usage example
if __name__ == "__main__":
    from ericsson_processor import EricssonFeatureProcessor

    # Load processed data
    processor = EricssonFeatureProcessor()
    processor.load_processed_data("output/ericsson_data")

    # Create query engine
    engine = EricssonQueryEngine(processor)

    # Example queries
    queries = [
        "Tell me about feature FAJ 121 5201",
        "Which features use initPreschedulingEnable?",
        "Explain pmMimoSleepTime counter",
        "How do I activate CXC4011253?",
        "What conflicts with feature FAJ 121 5201?",
        "List all MIMO-related features"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        result = engine.query(query)
        print(result[:500] + "..." if len(result) > 500 else result)
```

### Phase 6: Advanced Features Implementation (Days 12-13)

#### 6.1 Feature Relationship Visualization
```python
def generate_dependency_diagram(self, feature_id):
    """Generate ASCII diagram of feature dependencies"""
    feature = self.get_feature(feature_id)
    diagram = f"{feature.name} (FAJ {feature.id})\n"

    # Show prerequisites
    for prereq in feature.prerequisites:
        diagram += f"  ← requires ← {prereq.name}\n"

    # Show related features
    for related in feature.related:
        diagram += f"  ↔ related to ↔ {related.name}\n"

    return diagram
```

#### 6.2 Configuration Validation
```python
def validate_configuration(self, features_to_activate):
    """Check for conflicts and missing prerequisites"""
    warnings = []
    errors = []

    for feature in features_to_activate:
        # Check prerequisites
        for prereq in feature.prerequisites:
            if prereq not in features_to_activate:
                errors.append(f"{feature.name} requires {prereq.name}")

        # Check conflicts
        for conflict in feature.conflicts:
            if conflict in features_to_activate:
                errors.append(f"{feature.name} conflicts with {conflict.name}")

    return {"warnings": warnings, "errors": errors}
```

### Phase 7: Integration and Packaging (Days 14-15)

#### 7.1 Main Integration Script
Create `ericsson_skill_builder.py`:
```python
#!/usr/bin/env python3
"""
Ericsson RAN Features Skill Builder
Integrates all components to generate Claude skill
"""

import os
import json
import zipfile
from pathlib import Path
from typing import Dict, List
from ericsson_processor import EricssonFeatureProcessor
from query_engine import EricssonQueryEngine


class EricssonSkillBuilder:
    """Builds Claude skill from processed Ericsson features"""

    def __init__(self, processor: EricssonFeatureProcessor):
        self.processor = processor
        self.skill_dir = processor.skill_dir
        self.features = processor.features

    def create_skill_structure(self):
        """Create skill directory structure"""
        print("📁 Creating skill structure...")

        # Create reference directories
        refs_dir = Path(self.skill_dir) / "references"
        (refs_dir / "features").mkdir(parents=True, exist_ok=True)
        (refs_dir / "parameters").mkdir(parents=True, exist_ok=True)
        (refs_dir / "counters").mkdir(parents=True, exist_ok=True)
        (refs_dir / "events").mkdir(parents=True, exist_ok=True)
        (refs_dir / "cxc_codes").mkdir(parents=True, exist_ok=True)
        (refs_dir / "relationships").mkdir(parents=True, exist_ok=True)
        (refs_dir / "engineering").mkdir(parents=True, exist_ok=True)

        print("✅ Skill structure created")

    def generate_references(self):
        """Generate reference files for all features"""
        print("📚 Generating reference files...")

        # Generate feature references
        self.generate_feature_references()

        # Generate parameter index
        self.generate_parameter_index()

        # Generate counter index
        self.generate_counter_index()

        # Generate CXC code index
        self.generate_cxc_index()

        # Generate relationship references
        self.generate_relationship_references()

        # Generate engineering guidelines
        self.generate_engineering_guidelines()

        print("✅ Reference files generated")

    def generate_feature_references(self):
        """Generate categorized feature references"""
        # Features by category
        features_by_category = self.categorize_features()

        refs_dir = Path(self.skill_dir) / "references" / "features"

        # Master index
        with open(refs_dir / "index.md", 'w') as f:
            f.write("# Ericsson Radio Features Index\n\n")
            f.write(f"**Total Features**: {len(self.features)}\n\n")
            f.write("## Categories\n\n")

            for category, features in features_by_category.items():
                f.write(f"### {category.replace('_', ' ').title()} ({len(features)})\n\n")
                for feature in features[:20]:  # Limit to 20 per category
                    f.write(f"- [{feature.name}](feature_{feature.id.replace(' ', '_')}.md) (FAJ {feature.id}")
                    if feature.cxc_code:
                        f.write(f", CXC {feature.cxc_code}")
                    f.write(")\n")

                if len(features) > 20:
                    f.write(f"- ... and {len(features) - 20} more features\n")
                f.write("\n")

        # Individual feature files
        for feature in list(self.features.values())[:100]:  # Limit to 100 for demo
            filename = f"feature_{feature.id.replace(' ', '_')}.md"
            with open(refs_dir / filename, 'w') as f:
                self.write_feature_file(f, feature)

    def write_feature_file(self, f, feature):
        """Write individual feature reference file"""
        f.write(f"# {feature.name}\n\n")
        f.write(f"**FAJ ID**: FAJ {feature.id}\n")
        if feature.cxc_code:
            f.write(f"**CXC Code**: {feature.cxc_code}\n")
        f.write(f"**Access Type**: {feature.access_type}\n")
        f.write(f"**Value Package**: {feature.value_package}\n")
        f.write(f"**Node Type**: {feature.node_type}\n\n")

        if feature.activation_step:
            f.write("## Activation\n\n")
            f.write("```bash\n")
            f.write(feature.activation_step)
            f.write("\n```\n\n")

        if feature.parameters:
            f.write(f"## Parameters ({len(feature.parameters)})\n\n")
            for param in feature.parameters:
                f.write(f"### {param['name']}\n")
                f.write(f"- **Type**: {param['type']}\n")
                f.write(f"- **MO Class**: {param['mo_class']}\n")
                f.write(f"- **Description**: {param['description']}\n\n")

        if feature.counters:
            f.write(f"## Performance Counters ({len(feature.counters)})\n\n")
            for counter in feature.counters:
                f.write(f"### {counter['name']}\n")
                f.write(f"- {counter['description']}\n\n")

        if feature.engineering_guidelines:
            f.write("## Engineering Guidelines\n\n")
            f.write(feature.engineering_guidelines)
            f.write("\n\n")

    def generate_parameter_index(self):
        """Generate parameter master index"""
        refs_dir = Path(self.skill_dir) / "references" / "parameters"

        with open(refs_dir / "index.md", 'w') as f:
            f.write("# Parameter Master Index\n\n")

            # Group by MO class
            mo_params = {}
            for param_name, feature_ids in self.processor.parameter_index.items():
                for fid in feature_ids:
                    feature = self.features[fid]
                    for param in feature.parameters:
                        if param_name in param['name'].lower():
                            mo_class = param['mo_class']
                            if mo_class not in mo_params:
                                mo_params[mo_class] = []
                            mo_params[mo_class].append((param, feature))
                            break

            for mo_class, params in sorted(mo_params.items()):
                f.write(f"## {mo_class}\n\n")
                for param, feature in params[:10]:  # Limit to 10 per MO
                    f.write(f"- **{param['name']}** - Used in {feature.name} (FAJ {feature.id})\n")
                f.write("\n")

    def generate_counter_index(self):
        """Generate counter master index"""
        refs_dir = Path(self.skill_dir) / "references" / "counters"

        with open(refs_dir / "index.md", 'w') as f:
            f.write("# Performance Counter Index\n\n")

            for counter_name, feature_ids in self.processor.counter_index.items():
                f.write(f"## {counter_name}\n\n")
                f.write(f"**Used in {len(feature_ids)} features**:\n\n")

                for fid in feature_ids[:5]:  # Limit to 5 features
                    feature = self.features[fid]
                    f.write(f"- {feature.name} (FAJ {feature.id})\n")
                f.write("\n")

    def generate_cxc_index(self):
        """Generate CXC code index"""
        refs_dir = Path(self.skill_dir) / "references" / "cxc_codes"

        with open(refs_dir / "index.md", 'w') as f:
            f.write("# CXC Feature Code Index\n\n")
            f.write("Quick reference for feature activation codes:\n\n")

            for cxc_code, feature_id in self.processor.cxc_index.items():
                feature = self.features[feature_id]
                f.write(f"## {cxc_code}\n\n")
                f.write(f"**Feature**: {feature.name}\n")
                f.write(f"**FAJ ID**: FAJ {feature.id}\n")
                f.write(f"**Access Type**: {feature.access_type}\n\n")

                if feature.activation_step:
                    f.write("**Activation**:\n```bash\n")
                    f.write(feature.activation_step)
                    f.write("\n```\n\n")

                if feature.deactivation_step:
                    f.write("**Deactivation**:\n```bash\n")
                    f.write(feature.deactivation_step)
                    f.write("\n```\n\n")

    def generate_relationship_references(self):
        """Generate feature relationship references"""
        refs_dir = Path(self.skill_dir) / "references" / "relationships"

        with open(refs_dir / "dependencies.md", 'w') as f:
            f.write("# Feature Dependencies\n\n")

            for feature_id, relationships in self.processor.relationship_graph.items():
                feature = self.features[feature_id]
                if any(relationships.values()):
                    f.write(f"## {feature.name} (FAJ {feature.id})\n\n")

                    if relationships.get('prerequisites'):
                        f.write("### Prerequisites\n")
                        for prereq_id in relationships['prerequisites']:
                            prereq = self.features.get(prereq_id)
                            if prereq:
                                f.write(f"- {prereq.name} (FAJ {prereq_id})\n")
                        f.write("\n")

                    if relationships.get('conflicts'):
                        f.write("### Conflicts\n")
                        for conflict_id in relationships['conflicts']:
                            conflict = self.features.get(conflict_id)
                            if conflict:
                                f.write(f"- {conflict.name} (FAJ {conflict_id})\n")
                        f.write("\n")

    def generate_engineering_guidelines(self):
        """Generate engineering guidelines compilation"""
        refs_dir = Path(self.skill_dir) / "references" / "engineering"

        with open(refs_dir / "guidelines.md", 'w') as f:
            f.write("# Engineering Guidelines\n\n")
            f.write("Collection of engineering guidelines from all features:\n\n")

            # Group by category
            guidelines_by_category = {
                'MIMO': [],
                'Carrier Aggregation': [],
                'Power Management': [],
                'Handover': [],
                'General': []
            }

            for feature in self.features.values():
                if feature.engineering_guidelines:
                    # Categorize based on feature name
                    category = 'General'
                    if 'mimo' in feature.name.lower():
                        category = 'MIMO'
                    elif 'carrier' in feature.name.lower() or 'aggregation' in feature.name.lower():
                        category = 'Carrier Aggregation'
                    elif 'sleep' in feature.name.lower() or 'power' in feature.name.lower():
                        category = 'Power Management'
                    elif 'handover' in feature.name.lower():
                        category = 'Handover'

                    guidelines_by_category[category].append((feature.name, feature.engineering_guidelines))

            for category, guidelines in guidelines_by_category.items():
                if guidelines:
                    f.write(f"## {category}\n\n")
                    for feature_name, guideline in guidelines[:5]:  # Limit to 5 per category
                        f.write(f"### {feature_name}\n\n")
                        f.write(f"{guideline[:500]}...\n\n")

    def categorize_features(self):
        """Categorize features by type"""
        categories = {
            'mimo_features': [],
            'carrier_aggregation': [],
            'power_management': [],
            'handover_mobility': [],
            'dual_connectivity': [],
            'other_features': []
        }

        for feature in self.features.values():
            name_lower = feature.name.lower()

            if 'mimo' in name_lower:
                categories['mimo_features'].append(feature)
            elif 'carrier' in name_lower or 'aggregation' in name_lower:
                categories['carrier_aggregation'].append(feature)
            elif 'sleep' in name_lower or 'power' in name_lower:
                categories['power_management'].append(feature)
            elif 'handover' in name_lower or 'mobility' in name_lower:
                categories['handover_mobility'].append(feature)
            elif 'dual' in name_lower or 'dc' in name_lower:
                categories['dual_connectivity'].append(feature)
            else:
                categories['other_features'].append(feature)

        return categories

    def create_skill_md(self):
        """Create main SKILL.md file"""
        print("📝 Creating SKILL.md...")

        skill_path = Path(self.skill_dir) / "SKILL.md"

        with open(skill_path, 'w') as f:
            f.write("""# Ericsson RAN Features Expert

## Overview
This skill provides comprehensive access to Ericsson LTE/NR radio features, including:
- """ + str(len(self.features)) + """ feature descriptions with technical details
- Complete parameter documentation with types and descriptions
- PM counters and KPI explanations
- Event definitions and triggers
- Feature dependencies and relationships
- Engineering guidelines and best practices
- CXC feature codes for activation/deactivation

## Capabilities

### Feature Information
- Get complete feature description: "Tell me about feature FAJ 121 5201"
- List features by category: "Show all MIMO-related features"
- Find features by parameter: "Which features use initPreschedulingEnable?"
- Find feature by CXC code: "What is feature CXC4011253?"

### Technical Details
- Parameter lookup: "What does ENodeBFunction.initPreschedulingEnable do?"
- Counter explanations: "Explain pmMimoSleepTime counter"
- Event details: "When is INTERNAL_PROC_UE_CTXT_RELEASE triggered?"

### Activation and Configuration
- Get activation commands: "How do I activate feature CXC4011072?"
- Get deactivation commands: "How do I deactivate TTI Bundling?"
- Exact MO configuration steps: "Show the exact command to activate FAJ 121 2051"

### Dependencies and Relationships
- Prerequisite checking: "What do I need for NR DL Carrier Aggregation?"
- Conflict detection: "Does feature X conflict with feature Y?"
- Feature grouping: "Show all features in the same value package"

### Engineering Support
- Configuration guidelines: "How do I configure MIMO Sleep Mode?"
- Troubleshooting: "What are common issues with CA configuration?"
- Best practices: "What are the recommended settings for feature X?"

## Quick Reference

### Common Feature Categories
- **MIMO Features**: Multiple Input Multiple Output optimizations
- **Carrier Aggregation**: Multi-carrier configurations
- **Power Management**: Energy saving features
- **Handover/Mobility**: Inter-cell handover optimization
- **Dual Connectivity**: Multi-RAT connectivity

### Access Patterns
- FAJ ID format: FAJ XXX XXXX (e.g., FAJ 121 5201)
- CXC Code format: CXC followed by numbers (e.g., CXC4011253)
- Parameter format: MOClass.parameterName
- Counter format: pmCounterName

## Reference Files
- `references/features/` - Complete feature documentation
- `references/parameters/` - Parameter master index
- `references/counters/` - Performance counter reference
- `references/cxc_codes/` - Activation code index
- `references/relationships/` - Feature dependencies
- `references/engineering/` - Engineering guidelines

## Usage Examples

### Example 1: Feature Lookup
User: "Tell me about MIMO Sleep Mode feature"
Response: Provides complete feature details including FAJ ID, CXC code, parameters, counters, and activation steps.

### Example 2: Parameter Search
User: "Which features affect scheduling?"
Response: Lists all features that use scheduling-related parameters with feature details.

### Example 3: Activation Help
User: "How to activate CXC4011253?"
Response: Provides exact activation command, prerequisites, and any conflicts.

## Notes
- This skill contains documentation for Ericsson Radio System features
- Always check prerequisites before activating features
- Verify compatibility with your specific node type and software version
- Consult engineering guidelines for optimal configuration
""")

        print("✅ SKILL.md created")

    def package_skill(self, output_filename="ericsson_ran_features.zip"):
        """Package skill into zip file"""
        print(f"📦 Packaging skill to {output_filename}...")

        with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all files in skill directory
            for root, dirs, files in os.walk(self.skill_dir):
                for file in files:
                    if not file.endswith('.backup'):  # Skip backup files
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, self.skill_dir)
                        zipf.write(file_path, arcname)

        print(f"✅ Skill packaged: {output_filename}")

        # Get file size
        size_mb = os.path.getsize(output_filename) / (1024 * 1024)
        print(f"📊 Package size: {size_mb:.2f} MB")


def main():
    """Main integration script"""
    print("🚀 Ericsson RAN Features Skill Builder\n")

    # Phase 1: Process all documentation files
    print("Phase 1: Processing documentation...")
    processor = EricssonFeatureProcessor()
    processor.process_directory("elex_features_only/")

    # Phase 2: Build indices and relationships
    print("\nPhase 2: Building indices and relationships...")
    processor.build_feature_relationships()

    # Phase 3: Generate skill files
    print("\nPhase 3: Generating skill...")
    skill_builder = EricssonSkillBuilder(processor)
    skill_builder.create_skill_structure()
    skill_builder.generate_references()
    skill_builder.create_skill_md()

    # Phase 4: Package skill
    print("\nPhase 4: Packaging skill...")
    skill_builder.package_skill("ericsson_ran_features_skill.zip")

    print("\n✅ Skill generation complete!")
    print("\nNext steps:")
    print("1. Upload ericsson_ran_features_skill.zip to Claude")
    print("2. Test with sample queries")
    print("3. Verify activation commands work in your environment")


if __name__ == "__main__":
    main()
```

#### 7.2 Performance Optimizations
- Implement caching for processed features
- Use lazy loading for large feature sets
- Create incremental processing for updates
- Optimize search with trie structures for parameter names

### Phase 8: Testing and Validation (Days 16-17)

#### 8.1 Test Scenarios
```python
test_cases = [
    ("Find features using initPreschedulingEnable", "Should return Prescheduling feature"),
    ("Explain pmMimoSleepTime", "Should return counter explanation and related features"),
    ("What conflicts with feature FAJ 121 5201?", "Should list conflicting features"),
    ("Show engineering guidelines for MIMO features", "Should return MIMO-related guidelines"),
    ("List all counters for EUtranCellFDD", "Should return comprehensive counter list")
]
```

#### 8.2 Validation Criteria
- All 2000+ features processed without errors
- Parameter extraction >95% accuracy
- Counter and event extraction complete
- Relationship graph correctly built
- Query response time <2 seconds
- Skill zip size <50MB

### Implementation Details

#### File Processing Pipeline
```python
def process_batch_directory(self, batch_path):
    """Process a single batch directory"""
    for file_path in glob.glob(f"{batch_path}/*.md"):
        feature = self.parse_feature_file(file_path)

        # Extract components
        feature.parameters = self.extract_parameters(feature.content)
        feature.counters = self.extract_counters(feature.content)
        feature.events = self.extract_events(feature.content)
        feature.dependencies = self.extract_dependencies(feature.content)

        # Store in master index
        self.features[feature.id] = feature
        self.update_indices(feature)
```

#### Error Handling Strategy
- Graceful handling of malformed markdown files
- Logging of processing errors for review
- Partial processing capability - skip problematic files
- Validation of extracted data against expected formats

#### Memory Management
- Process files in batches to manage memory usage
- Use generators for large file operations
- Implement LRU cache for frequently accessed features
- Serialize processed data to disk for persistence

### Success Metrics

1. **Processing Completeness**
   - 100% of markdown files processed
   - All parameters, counters, and events extracted
   - Complete relationship graph built

2. **Query Accuracy**
   - Feature lookup accuracy >98%
   - Parameter search accuracy >95%
   - Relationship queries 100% accurate

3. **Performance**
   - Initial processing time <4 hours
   - Query response time <2 seconds
   - Skill loading time <10 seconds

4. **Usability**
   - Intuitive query interface
   - Comprehensive coverage of all features
   - Clear, actionable responses

### Risk Mitigation

1. **Data Quality Issues**
   - Implement validation checks
   - Manual review of sample extractions
   - Fallback parsing for edge cases

2. **Performance Bottlenecks**
   - Use caching extensively
   - Implement pagination for large result sets
   - Optimize data structures

3. **Feature Complexity**
   - Handle complex features with special processing
   - Create custom extractors for edge cases
   - Maintain flexibility for future feature types

### Deliverables

1. **Core Processing Engine**
   - `ericsson_processor.py` - Main processing logic
   - `feature_extractor.py` - Specialized extraction methods
   - `query_engine.py` - Query processing and response

2. **Data Models**
   - `models.py` - Feature, Parameter, Counter, Event data classes
   - `indices.py` - Search index implementations
   - `relationships.py` - Relationship graph management

3. **Skill Generation**
   - `skill_builder.py` - Skill file generation
   - Templates for SKILL.md and reference files
   - `packager.py` - Zip creation utility

4. **Testing Suite**
   - Unit tests for all extraction methods
   - Integration tests for query processing
   - Performance benchmarks

5. **Documentation**
   - API documentation for all modules
   - User guide for the generated skill
   - Troubleshooting guide

### Future Enhancements

1. **Multi-language Support**
   - Process documentation in other languages
   - Translate extracted information
   - Support localized queries

2. **Real-time Updates**
   - Monitor documentation changes
   - Incremental processing of new features
   - Automatic skill updates

3. **Advanced Analytics**
   - Feature usage statistics
   - Parameter correlation analysis
   - Network impact modeling

4. **Integration Capabilities**
   - Export to other formats (JSON, XML)
   - API for external integrations
   - Database persistence option

This comprehensive plan ensures that all Ericsson feature documentation is processed into a fully operational Claude skill that can provide detailed, accurate information about features, parameters, counters, events, and engineering guidelines. The phased approach allows for iterative development and validation, ensuring a high-quality, scalable solution.

## Implementation Prerequisites

### Required Dependencies
```bash
pip3 install requests beautifulsoup4 markdown python-dataclasses
```

### Directory Structure
```
skills/
├── plan.md                           # This file
├── elex_features_only/               # Source Ericsson documentation
│   ├── batch_1/                      # 6 batches of feature files
│   ├── batch_2/
│   └── ...
├── ericsson_processor.py             # Core processing engine
├── query_engine.py                   # Query handling
├── ericsson_skill_builder.py         # Skill generation
└── output/                           # Generated output
    ├── ericsson_data/                # Processed feature data
    └── ericsson/                     # Claude skill
        ├── SKILL.md
        └── references/
```

## Quick Start Guide

### 1. Initial Setup
```bash
# Create project directory
mkdir ericsson_skill_project
cd ericsson_skill_project

# Copy or symlink plan.md and required modules
# Place elex_features_only/ directory with markdown files
```

### 2. Run Processing
```bash
# Execute the skill builder
python3 ericsson_skill_builder.py
```

### 3. Upload to Claude
1. Locate `ericsson_ran_features_skill.zip`
2. Upload to Claude.ai/skills
3. Test with sample queries

## Test Queries for Validation

### Basic Feature Queries
- "Tell me about FAJ 121 5201"
- "What is CXC4011253?"
- "List all MIMO features"

### Parameter and Counter Queries
- "Which features use initPreschedulingEnable?"
- "Explain pmMimoSleepTime"
- "Show parameters for EUtranCellFDD"

### Activation and Configuration
- "How do I activate CXC4011072?"
- "What conflicts with feature FAJ 121 5201?"
- "Show activation steps for TTI Bundling"

### Engineering Support
- "What are the engineering guidelines for MIMO?"
- "Configuration best practices for Carrier Aggregation"
- "Troubleshooting sleep mode features"

## Performance Benchmarks

### Expected Processing Times
- **Small batch (100 features)**: ~2 minutes
- **Medium batch (500 features)**: ~8 minutes
- **Full dataset (2000+ features)**: ~30 minutes

### Memory Requirements
- **Processing**: ~500MB RAM
- **Final skill**: ~50-100MB compressed
- **Query response**: <2 seconds

### Output Size Estimates
- **Raw JSON data**: ~200MB
- **Compressed skill**: ~50MB
- **Reference files**: ~1500 individual files

## Integration with Existing Systems

### API Integration (Optional Extension)
```python
# Example API endpoint for external systems
from flask import Flask, request, jsonify
from query_engine import EricssonQueryEngine

app = Flask(__name__)
engine = EricssonQueryEngine(processor)

@app.route('/query', methods=['POST'])
def api_query():
    query = request.json.get('query')
    result = engine.query(query)
    return jsonify({'response': result})
```

### Database Persistence (Optional)
```python
# Example for persistent storage
import sqlite3

def save_to_database(features):
    conn = sqlite3.connect('ericsson_features.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS features (
            id TEXT PRIMARY KEY,
            name TEXT,
            cxc_code TEXT,
            parameters TEXT,
            counters TEXT
        )
    ''')

    for feature in features.values():
        cursor.execute('''
            INSERT INTO features VALUES (?, ?, ?, ?, ?)
        ''', (
            feature.id,
            feature.name,
            feature.cxc_code,
            json.dumps(feature.parameters),
            json.dumps(feature.counters)
        ))

    conn.commit()
```

## Maintenance and Updates

### Adding New Features
```bash
# Add new markdown files to appropriate batch directory
# Re-run processor
python3 ericsson_skill_builder.py
```

### Updating Existing Features
1. Modify source markdown files
2. Delete specific feature JSON from output/ericsson_data/features/
3. Re-run processing - will only process modified/missing files

### Quality Assurance
1. Check `output/ericsson_data/summary.json` for processing statistics
2. Review error logs for skipped or malformed files
3. Validate extracted parameters/counters against sample

## Troubleshooting Guide

### Common Issues
1. **Memory errors**: Process in smaller batches
2. **Missing features**: Check markdown format and FAJ number extraction
3. **Empty parameters**: Verify table structure in source files
4. **Large zip size**: Limit feature files or increase compression

### Debug Mode
```python
# Add debug flag to processor
processor = EricssonFeatureProcessor(debug=True)
# Will log detailed extraction process
```

This plan provides a complete roadmap for transforming Ericsson's technical documentation into an AI-powered knowledge base, enabling efficient RAN optimization through natural language queries.