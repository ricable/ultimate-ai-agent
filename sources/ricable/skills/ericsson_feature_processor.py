#!/usr/bin/env python3
"""
Ericsson Feature Documentation Processor
Scalable processor for local markdown files - starts with 5 files, scales to all
Based on Skill_Seekers architecture adapted for local Ericsson documentation
"""

import os
import sys
import json
import re
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import markdown
from bs4 import BeautifulSoup


@dataclass
class EricssonFeature:
    """Complete data model for Ericsson feature documentation"""
    # Identity
    id: str = ""  # FAJ XXX XXXX
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
    parameters: List[Dict] = field(default_factory=list)
    counters: List[Dict] = field(default_factory=list)
    events: List[Dict] = field(default_factory=list)

    # Dependencies
    dependencies: Dict = field(default_factory=dict)

    # Operations
    activation_step: Optional[str] = None
    deactivation_step: Optional[str] = None

    # Guidelines
    engineering_guidelines: str = ""

    # Metadata
    source_file: str = ""
    file_hash: str = ""
    processed_at: str = ""

    # Performance and Network Impact
    network_impact: Dict = field(default_factory=dict)
    performance_impact: Dict = field(default_factory=dict)


class EricssonFeatureProcessor:
    """Scalable processor for Ericsson feature documentation"""

    def __init__(self, source_dir: str, output_dir: str = "output", batch_size: int = 50):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size

        # Data storage
        self.features: Dict[str, EricssonFeature] = {}
        self.processed_files: Set[str] = set()
        self.error_files: List[Tuple[str, str]] = []  # (file, error)

        # Search indices
        self.parameter_index = defaultdict(list)
        self.counter_index = defaultdict(list)
        self.cxc_index = {}
        self.name_index = {}

        # Create output directories
        self.setup_directories()

        # Statistics
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'errors': 0,
            'start_time': time.time(),
            'batches': []
        }

    def setup_directories(self):
        """Create necessary directory structure"""
        dirs = [
            self.output_dir,
            self.output_dir / "ericsson_data",
            self.output_dir / "ericsson_data" / "features",
            self.output_dir / "ericsson_data" / "indices",
            self.output_dir / "ericsson_data" / "cache",
            self.output_dir / "ericsson" / "references",
            self.output_dir / "ericsson" / "references" / "features",
            self.output_dir / "ericsson" / "references" / "parameters",
            self.output_dir / "ericsson" / "references" / "counters",
            self.output_dir / "ericsson" / "references" / "cxc_codes",
            self.output_dir / "ericsson" / "references" / "guidelines",
        ]

        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

    def discover_files(self) -> List[Path]:
        """Discover all markdown files in source directory"""
        print(f"🔍 Discovering markdown files in {self.source_dir}")

        md_files = list(self.source_dir.rglob("*.md"))
        self.stats['total_files'] = len(md_files)

        print(f"📊 Found {len(md_files)} markdown files")
        return md_files

    def process_all(self, limit: Optional[int] = None):
        """Process all files with batching for scalability"""
        md_files = self.discover_files()

        if limit:
            md_files = md_files[:limit]
            print(f"🎯 Processing limited to {limit} files")

        # Process in batches to manage memory
        for i in range(0, len(md_files), self.batch_size):
            batch = md_files[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(md_files) + self.batch_size - 1) // self.batch_size

            print(f"\n📦 Processing batch {batch_num}/{total_batches} ({len(batch)} files)")

            batch_start = time.time()
            batch_stats = self.process_batch(batch)
            batch_stats['batch_num'] = batch_num
            batch_stats['duration'] = time.time() - batch_start

            self.stats['batches'].append(batch_stats)

            # Save intermediate results
            if batch_num % 5 == 0:
                self.save_progress()
                print(f"💾 Saved progress after batch {batch_num}")

        # Final processing
        self.build_indices()
        self.save_all()

        # Print summary
        self.print_summary()

    def process_batch(self, files: List[Path]) -> Dict:
        """Process a batch of files"""
        batch_stats = {
            'files': len(files),
            'processed': 0,
            'errors': 0,
            'features': 0
        }

        for file_path in files:
            try:
                feature = self.process_file(file_path)
                if feature:
                    self.features[feature.id] = feature
                    self.processed_files.add(str(file_path))
                    batch_stats['features'] += 1
                    batch_stats['processed'] += 1
                else:
                    batch_stats['errors'] += 1
                    self.error_files.append((str(file_path), "No valid FAJ ID found"))
            except Exception as e:
                batch_stats['errors'] += 1
                self.error_files.append((str(file_path), str(e)))
                print(f"  ✗ Error processing {file_path.name}: {e}")

            self.stats['processed'] += 1

            # Progress indicator
            if batch_stats['processed'] % 10 == 0:
                print(f"  Processed {batch_stats['processed']} files...")

        return batch_stats

    def process_file(self, file_path: Path) -> Optional[EricssonFeature]:
        """Process a single markdown file"""
        # Check cache first
        cache_file = self.output_dir / "ericsson_data" / "cache" / f"{file_path.stem}.json"
        if cache_file.exists():
            file_hash = self.calculate_file_hash(file_path)
            cached_data = json.loads(cache_file.read_text())
            if cached_data.get('file_hash') == file_hash:
                # Return cached feature
                return EricssonFeature(**cached_data['feature'])

        # Read and parse file
        content = file_path.read_text(encoding='utf-8')

        # Convert to HTML for easier parsing
        html = markdown.markdown(
            content,
            extensions=['tables', 'fenced_code', 'toc']
        )
        soup = BeautifulSoup(html, 'html.parser')

        # Extract feature identity
        feature = self.extract_feature_identity(soup)
        if not feature:
            return None

        # Set metadata
        feature.source_file = str(file_path)
        feature.file_hash = self.calculate_file_hash(file_path)
        feature.processed_at = time.strftime('%Y-%m-%d %H:%M:%S')

        # Extract content sections
        feature.description = self.extract_section_content(soup, "Overview")
        feature.summary = self.extract_summary(soup)

        # Extract technical details
        feature.parameters = self.extract_parameters(soup)
        feature.counters = self.extract_counters(soup)
        feature.events = self.extract_events(soup)

        # Extract dependencies
        feature.dependencies = self.extract_dependencies(soup)

        # Extract activation/deactivation
        feature.activation_step = self.extract_activation_step(soup)
        feature.deactivation_step = self.extract_deactivation_step(soup)

        # Extract guidelines
        feature.engineering_guidelines = self.extract_engineering_guidelines(soup)

        # Extract impact information
        feature.network_impact = self.extract_network_impact(soup)
        feature.performance_impact = self.extract_performance_impact(soup)

        # Cache the result
        cache_data = {
            'file_hash': feature.file_hash,
            'feature': asdict(feature)
        }
        cache_file.write_text(json.dumps(cache_data, indent=2))

        return feature

    def extract_feature_identity(self, soup: BeautifulSoup) -> Optional[EricssonFeature]:
        """Extract feature identity from documentation"""
        # Look for feature identity table
        feature = EricssonFeature()

        # Try multiple patterns for FAJ ID
        faj_patterns = [
            r'FAJ\s*(\d+\s+\d+)',  # FAJ 121 3094
            r'FAJ\s*(\d{3}\s*\d{4})',  # FAJ 1213094
            r'Feature Identity\s*\|\s*FAJ\s*(\d+\s+\d+)',  # In table
        ]

        faj_id = None
        for pattern in faj_patterns:
            match = re.search(pattern, soup.get_text())
            if match:
                faj_id = match.group(1)
                break

        if not faj_id:
            return None

        feature.id = faj_id

        # Extract feature name (usually in first h1)
        h1 = soup.find('h1')
        if h1:
            feature.name = h1.get_text().strip()

        # Extract from feature table if present
        tables = soup.find_all('table')
        for table in tables:
            text = table.get_text()
            if 'Feature Name' in text:
                # Parse table rows
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        key = cells[0].get_text().strip()
                        value = cells[1].get_text().strip()

                        if 'Feature Name' in key:
                            feature.name = value
                        elif 'Value Package Name' in key:
                            feature.value_package = value
                        elif 'Value Package Identity' in key:
                            feature.value_package_id = value
                        elif 'Node Type' in key:
                            feature.node_type = value
                        elif 'Access Type' in key:
                            feature.access_type = value
                        elif 'Feature Identity' in key and 'FAJ' in value:
                            feature.id = re.search(r'FAJ\s*(\d+\s+\d+)', value).group(1)

        return feature

    def extract_section_content(self, soup: BeautifulSoup, section_name: str) -> str:
        """Extract content from a specific section"""
        # Look for h2 or h3 with the section name
        for tag in soup.find_all(['h2', 'h3']):
            if section_name.lower() in tag.get_text().lower():
                # Get content until next heading of same or higher level
                content = []
                current = tag.next_sibling

                while current:
                    if current.name in ['h1', 'h2', 'h3']:
                        if current.name <= tag.name:
                            break
                    if hasattr(current, 'get_text'):
                        text = current.get_text().strip()
                        if text:
                            content.append(text)
                    current = current.next_sibling

                return '\n'.join(content)

        return ""

    def extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract feature summary"""
        # Look for Summary section or first paragraph
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 50 and 'feature' in text.lower():
                return text

        # Look for explicit Summary heading
        for tag in soup.find_all(['h2', 'h3', 'h4']):
            if 'summary' in tag.get_text().lower():
                return self.extract_section_content(soup, tag.get_text())

        return ""

    def extract_parameters(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract parameters from documentation"""
        parameters = []

        # Look for parameter tables
        tables = soup.find_all('table')
        for table in tables:
            headers = [th.get_text().strip().lower() for th in table.find_all('th')]

            if any('parameter' in h for h in headers):
                rows = table.find_all('tr')[1:]  # Skip header

                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        param = {
                            'name': cells[0].get_text().strip(),
                            'type': cells[1].get_text().strip() if len(cells) > 1 else '',
                            'description': cells[2].get_text().strip() if len(cells) > 2 else '',
                            'mo_class': self.extract_mo_class(cells[0].get_text().strip())
                        }
                        if param['name']:
                            parameters.append(param)

        # Also look for parameter mentions in text
        text = soup.get_text()
        param_mentions = re.findall(r'([A-Z][a-zA-Z]*\.[a-zA-Z][a-zA-Z0-9]*)', text)

        for mention in param_mentions[:10]:  # Limit to avoid too many
            if not any(p['name'] == mention for p in parameters):
                parameters.append({
                    'name': mention,
                    'type': 'Unknown',
                    'description': 'Parameter mentioned in documentation',
                    'mo_class': mention.split('.')[0] if '.' in mention else 'Unknown'
                })

        return parameters

    def extract_mo_class(self, param_name: str) -> str:
        """Extract MO class from parameter name"""
        if '.' in param_name:
            return param_name.split('.')[0]
        return "Unknown"

    def extract_counters(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract performance counters"""
        counters = []
        text = soup.get_text()

        # Find PM counter patterns
        counter_patterns = [
            r'pm([A-Za-z0-9]+)',  # pmCounterName
            r'PM\s+([A-Za-z0-9]+)',  # PM CounterName
        ]

        found_counters = set()
        for pattern in counter_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_counters.update(matches)

        # Create counter entries
        for counter in sorted(found_counters):
            counters.append({
                'name': f'pm{counter}',
                'description': f'Performance counter pm{counter}',
                'category': self.guess_counter_category(counter)
            })

        return counters

    def guess_counter_category(self, counter: str) -> str:
        """Guess counter category based on name"""
        if 'mimo' in counter.lower():
            return 'MIMO'
        elif 'sleep' in counter.lower():
            return 'Energy'
        elif 'handover' in counter.lower():
            return 'Mobility'
        elif 'throughput' in counter.lower():
            return 'Throughput'
        else:
            return 'General'

    def extract_events(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract events from documentation"""
        events = []

        # Look for event tables
        tables = soup.find_all('table')
        for table in tables:
            headers = [th.get_text().strip().lower() for th in table.find_all('th')]

            if any('event' in h for h in headers):
                rows = table.find_all('tr')[1:]

                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        event = {
                            'name': cells[0].get_text().strip(),
                            'type': cells[1].get_text().strip() if len(cells) > 1 else '',
                            'description': cells[2].get_text().strip() if len(cells) > 2 else ''
                        }
                        if event['name']:
                            events.append(event)

        return events

    def extract_dependencies(self, soup: BeautifulSoup) -> Dict:
        """Extract feature dependencies"""
        dependencies = {
            'prerequisites': [],
            'related': [],
            'conflicts': []
        }

        # Look for Dependencies section
        dep_section = self.extract_section_content(soup, "Dependencies")

        # Extract FAJ references
        faj_refs = re.findall(r'FAJ\s*(\d+\s+\d+)', dep_section)

        # Categorize based on context
        for faj in faj_refs:
            if 'prerequisite' in dep_section.lower() or 'require' in dep_section.lower():
                dependencies['prerequisites'].append(faj)
            elif 'conflict' in dep_section.lower():
                dependencies['conflicts'].append(faj)
            else:
                dependencies['related'].append(faj)

        return dependencies

    def extract_activation_step(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract activation command"""
        # Look for activation section
        for tag in soup.find_all(['h2', 'h3']):
            if 'activate' in tag.get_text().lower():
                # Look for FeatureState in following content
                content = []
                current = tag.next_sibling

                while current and current.name not in ['h1', 'h2', 'h3']:
                    if hasattr(current, 'get_text'):
                        text = current.get_text()
                        if 'FeatureState' in text:
                            # Extract the exact command
                            lines = text.split('\n')
                            for line in lines:
                                if 'FeatureState' in line and 'ACTIVATED' in line:
                                    return line.strip()
                    current = current.next_sibling

        # Search entire document
        full_text = soup.get_text()
        match = re.search(r'1\.\s+Set the FeatureState\.featureState attribute to ACTIVATED in the (FeatureState=[^\s]+)', full_text)
        if match:
            return f"1. Set the FeatureState.featureState attribute to ACTIVATED in the {match.group(1)} MO instance."

        return None

    def extract_deactivation_step(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract deactivation command"""
        # Similar to activation but looking for DEACTIVATED
        full_text = soup.get_text()
        match = re.search(r'1\.\s+Set the FeatureState\.featureState attribute to DEACTIVATED in the (FeatureState=[^\s]+)', full_text)
        if match:
            return f"1. Set the FeatureState.featureState attribute to DEACTIVATED in the {match.group(1)} MO instance."

        return None

    def extract_engineering_guidelines(self, soup: BeautifulSoup) -> str:
        """Extract engineering guidelines"""
        guidelines = []

        # Look for Engineering Guidelines section
        section = self.extract_section_content(soup, "Engineering Guidelines")

        if section:
            # Clean up the content
            lines = section.split('\n')
            current_guideline = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith('Figure') and not line.startswith('Table'):
                    current_guideline.append(line)
                elif current_guideline:
                    guidelines.append(' '.join(current_guideline))
                    current_guideline = []

            if current_guideline:
                guidelines.append(' '.join(current_guideline))

        return '\n\n'.join(guidelines)

    def extract_network_impact(self, soup: BeautifulSoup) -> Dict:
        """Extract network impact information"""
        impact = {}

        # Look for Network Impact section
        section = self.extract_section_content(soup, "Network Impact")

        if section:
            # Extract key impacts
            if 'capacity' in section.lower():
                impact['capacity'] = 'Affects cell capacity temporarily'
            if 'coverage' in section.lower():
                impact['coverage'] = 'May temporarily affect cell coverage'
            if 'performance' in section.lower():
                impact['performance'] = 'Performance impact during reconfiguration'

        return impact

    def extract_performance_impact(self, soup: BeautifulSoup) -> Dict:
        """Extract performance impact information"""
        impact = {}

        # Look for Performance section
        section = self.extract_section_content(soup, "Performance")

        if section:
            # Extract key metrics
            if 'kpi' in section.lower():
                impact['kpi'] = 'Affects KPI monitoring'
            if 'counters' in section.lower():
                impact['counters'] = 'New counters introduced'

        return impact

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for caching"""
        content = file_path.read_bytes()
        return hashlib.md5(content).hexdigest()

    def build_indices(self):
        """Build search indices for fast lookup"""
        print("\n🔍 Building search indices...")

        for feature in self.features.values():
            # Parameter index
            for param in feature.parameters:
                self.parameter_index[param['name'].lower()].append(feature.id)

            # Counter index
            for counter in feature.counters:
                self.counter_index[counter['name'].lower()].append(feature.id)

            # CXC index
            if feature.cxc_code:
                self.cxc_index[feature.cxc_code] = feature.id

            # Name index
            name_words = feature.name.lower().split()
            for word in name_words:
                if len(word) > 3:
                    self.name_index[word] = feature.id

        print(f"✅ Built indices for {len(self.features)} features")

    def save_progress(self):
        """Save intermediate progress"""
        progress = {
            'stats': self.stats,
            'processed_files': list(self.processed_files),
            'error_files': self.error_files[:100]  # Limit error storage
        }

        progress_file = self.output_dir / "ericsson_data" / "progress.json"
        progress_file.write_text(json.dumps(progress, indent=2))

    def save_all(self):
        """Save all processed data"""
        print("\n💾 Saving processed data...")

        # Save features
        features_dir = self.output_dir / "ericsson_data" / "features"
        for feature_id, feature in self.features.items():
            filename = f"feature_{feature_id.replace(' ', '_')}.json"
            filepath = features_dir / filename
            filepath.write_text(json.dumps(asdict(feature), indent=2))

        # Save indices
        indices_dir = self.output_dir / "ericsson_data" / "indices"
        indices = {
            'parameters': dict(self.parameter_index),
            'counters': dict(self.counter_index),
            'cxc_codes': self.cxc_index,
            'names': self.name_index
        }

        for name, index in indices.items():
            index_file = indices_dir / f"{name}_index.json"
            index_file.write_text(json.dumps(index, indent=2))

        # Save summary
        summary = {
            'total_features': len(self.features),
            'total_parameters': sum(len(f.parameters) for f in self.features.values()),
            'total_counters': sum(len(f.counters) for f in self.features.values()),
            'processing_stats': self.stats,
            'feature_categories': self.categorize_features()
        }

        summary_file = self.output_dir / "ericsson_data" / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2))

        print(f"✅ Saved {len(self.features)} features")

    def categorize_features(self) -> Dict[str, int]:
        """Categorize features by type"""
        categories = defaultdict(int)

        for feature in self.features.values():
            name_lower = feature.name.lower()

            if 'mimo' in name_lower:
                categories['MIMO Features'] += 1
            elif 'sleep' in name_lower or 'energy' in name_lower:
                categories['Energy Efficiency'] += 1
            elif 'carrier' in name_lower or 'aggregation' in name_lower:
                categories['Carrier Aggregation'] += 1
            elif 'handover' in name_lower or 'mobility' in name_lower:
                categories['Mobility'] += 1
            elif 'dual' in name_lower:
                categories['Dual Connectivity'] += 1
            else:
                categories['Other'] += 1

        return dict(categories)

    def print_summary(self):
        """Print processing summary"""
        duration = time.time() - self.stats['start_time']

        print("\n" + "="*60)
        print("📊 PROCESSING SUMMARY")
        print("="*60)
        print(f"Total files found: {self.stats['total_files']}")
        print(f"Files processed: {self.stats['processed']}")
        print(f"Features extracted: {len(self.features)}")
        print(f"Processing errors: {len(self.error_files)}")
        print(f"Total time: {duration:.2f} seconds")
        print(f"Avg time per file: {duration/max(1, self.stats['processed']):.3f} seconds")

        print("\nFeature Categories:")
        categories = self.categorize_features()
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")

        print(f"\nData saved to: {self.output_dir}/ericsson_data/")

        if self.error_files:
            print(f"\n⚠ First 10 errors:")
            for file, error in self.error_files[:10]:
                print(f"  {file}: {error}")


# Main execution
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process Ericsson feature documentation')
    parser.add_argument('--source', default='elex_features_only', help='Source directory')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--limit', type=int, help='Limit number of files to process (for testing)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing')

    args = parser.parse_args()

    # Create processor
    processor = EricssonFeatureProcessor(
        source_dir=args.source,
        output_dir=args.output,
        batch_size=args.batch_size
    )

    # Process files
    print("🚀 Starting Ericsson Feature Processing")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")

    if args.limit:
        print(f"Limit: {args.limit} files")

    processor.process_all(limit=args.limit)

    print("\n✅ Processing complete!")
    print(f"Next step: Run ericsson_skill_generator.py to create Claude skill")