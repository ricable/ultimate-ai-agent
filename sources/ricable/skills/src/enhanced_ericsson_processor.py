#!/usr/bin/env python3
"""
Enhanced Ericsson Feature Processor with Scalable Batch Processing
Integrates the enhanced batch processing system with Ericsson-specific feature extraction
"""

import os
import sys
import json
import re
import logging
import hashlib
import time

logger = logging.getLogger(__name__)
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import markdown
from bs4 import BeautifulSoup

# Import the enhanced batch processor
from enhanced_batch_processor import BatchProcessor, BatchState, ProcessingStats

# Import Ericsson-specific components
from ericsson_feature_processor import EricssonFeature


class EnhancedEricssonProcessor(BatchProcessor):
    """
    Enhanced Ericsson feature processor with scalable batch processing
    Integrates Ericsson-specific feature extraction with memory-efficient batching
    """

    def __init__(self, source_dir: str, output_dir: str = "output", **kwargs):
        """
        Initialize enhanced Ericsson processor

        Args:
            source_dir: Directory containing Ericsson markdown files
            output_dir: Output directory for processed data
            **kwargs: Additional arguments passed to BatchProcessor
        """
        super().__init__(source_dir, output_dir, **kwargs)

        # Ericsson-specific indices
        self.parameter_index = defaultdict(list)
        self.counter_index = defaultdict(list)
        self.cxc_index = {}
        self.name_index = {}

        # Ericsson-specific statistics
        self.stats.ericsson_stats = {
            'faj_numbers_found': 0,
            'cxc_codes_extracted': 0,
            'parameters_extracted': 0,
            'counters_extracted': 0,
            'events_extracted': 0,
            'engineering_guidelines_found': 0
        }

    def process_file(self, file_path: Path) -> Optional[Dict]:
        """
        Process Ericsson markdown file with comprehensive feature extraction

        Args:
            file_path: Path to Ericsson markdown file

        Returns:
            Extracted feature data or None if processing failed
        """
        try:
            # Check cache first
            cache_file = self.output_dir / "ericsson_data" / "cache" / f"{file_path.stem}.json"
            if cache_file.exists():
                file_hash = self.calculate_file_hash(file_path)
                cached_data = json.loads(cache_file.read_text())
                if cached_data.get('file_hash') == file_hash:
                    self.stats.cached_files += 1
                    return cached_data['feature']

            # Read and parse file
            content = file_path.read_text(encoding='utf-8')

            # Convert to HTML for easier parsing
            html = markdown.markdown(
                content,
                extensions=['tables', 'fenced_code', 'toc']
            )
            soup = BeautifulSoup(html, 'html.parser')

            # Extract feature identity
            feature_data = self.extract_feature_identity(soup, file_path)
            if not feature_data:
                return None

            # Extract technical details
            self.extract_technical_details(soup, feature_data)

            # Extract operations information
            self.extract_operations_info(soup, feature_data)

            # Set metadata
            feature_data['source_file'] = str(file_path)
            feature_data['file_hash'] = self.calculate_file_hash(file_path)
            feature_data['processed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # Cache the result
            cache_file.parent.mkdir(exist_ok=True)
            cache_file.write_text(json.dumps({
                'file_hash': feature_data['file_hash'],
                'feature': feature_data
            }, indent=2))

            # Update Ericsson-specific statistics
            self.update_ericsson_stats(feature_data)

            return feature_data

        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            raise

    def calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for cache validation"""
        content = file_path.read_text(encoding='utf-8')
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def extract_feature_identity(self, soup: BeautifulSoup, file_path: Path) -> Optional[Dict]:
        """
        Extract feature identity information (FAJ number, name, etc.)

        Args:
            soup: Parsed HTML content
            file_path: Source file path

        Returns:
            Feature identity data or None if not found
        """
        # Look for FAJ number in various formats
        faj_patterns = [
            r'FAJ\s+(\d{3}\s+\d{4})',
            r'FAJ\s*-\s*(\d{3}\s*\d{4})',
            r'FAJ[:\s]*(\d{3}\s*\d{4})'
        ]

        faj_number = None
        for pattern in faj_patterns:
            match = re.search(pattern, soup.get_text(), re.IGNORECASE)
            if match:
                faj_number = match.group(1).replace(' ', ' ')
                break

        if not faj_number:
            # Try to extract from filename as fallback
            filename_match = re.search(r'(\d{3}[_\s]?\d{4})', file_path.stem)
            if filename_match:
                faj_number = filename_match.group(1).replace('_', ' ')

        if not faj_number:
            return None

        # Extract feature name from H1 or title
        feature_name = ""
        h1_tag = soup.find('h1')
        if h1_tag:
            feature_name = h1_tag.get_text().strip()
        else:
            title_tag = soup.find('title')
            if title_tag:
                feature_name = title_tag.get_text().strip()

        # Look for CXC code
        cxc_pattern = r'CXC\s+(\d{6})'
        cxc_match = re.search(cxc_pattern, soup.get_text(), re.IGNORECASE)
        cxc_code = cxc_match.group(1) if cxc_match else None

        return {
            'id': f"FAJ {faj_number}",
            'faj_number': faj_number,
            'name': feature_name,
            'cxc_code': cxc_code,
            'value_package': self.extract_value_package(soup),
            'node_type': self.extract_node_type(soup),
            'access_type': self.extract_access_type(soup),
            'description': self.extract_description(soup),
            'summary': self.extract_summary(soup)
        }

    def extract_value_package(self, soup: BeautifulSoup) -> str:
        """Extract value package information"""
        # Look for value package in tables or specific sections
        vp_patterns = [
            r'Value\s+Package[:\s]+([^\n\r]+)',
            r'VP[:\s]+([^\n\r]+)'
        ]

        for pattern in vp_patterns:
            match = re.search(pattern, soup.get_text(), re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def extract_node_type(self, soup: BeautifulSoup) -> str:
        """Extract node type information"""
        node_patterns = [
            r'Node\s+Type[:\s]+([^\n\r]+)',
            r'([A-Z]+Node)'
        ]

        for pattern in node_patterns:
            match = re.search(pattern, soup.get_text(), re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def extract_access_type(self, soup: BeautifulSoup) -> str:
        """Extract access type information"""
        access_patterns = [
            r'Access\s+Type[:\s]+([^\n\r]+)',
            r'(Local|Remote|Combined)\s+Access'
        ]

        for pattern in access_patterns:
            match = re.search(pattern, soup.get_text(), re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def extract_description(self, soup: BeautifulSoup) -> str:
        """Extract feature description"""
        # Look for description in first paragraph after title
        paragraphs = soup.find_all('p')
        for p in paragraphs[:3]:  # Check first 3 paragraphs
            text = p.get_text().strip()
            if len(text) > 50 and not text.startswith('FAJ'):
                return text

        return ""

    def extract_summary(self, soup: BeautifulSoup) -> str:
        """Extract feature summary"""
        # Look for summary section
        for header in soup.find_all(['h2', 'h3']):
            if 'summary' in header.get_text().lower():
                summary_text = ""
                next_sibling = header.find_next_sibling()
                while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3', 'h4']:
                    summary_text += next_sibling.get_text() + "\n"
                    next_sibling = next_sibling.find_next_sibling()
                return summary_text.strip()

        return ""

    def extract_technical_details(self, soup: BeautifulSoup, feature_data: Dict):
        """
        Extract technical details (parameters, counters, events)

        Args:
            soup: Parsed HTML content
            feature_data: Feature data to update
        """
        # Extract parameters from tables
        feature_data['parameters'] = self.extract_parameters(soup)

        # Extract PM counters
        feature_data['counters'] = self.extract_counters(soup)

        # Extract events
        feature_data['events'] = self.extract_events(soup)

        # Extract dependencies
        feature_data['dependencies'] = self.extract_dependencies(soup)

        # Extract engineering guidelines
        feature_data['engineering_guidelines'] = self.extract_engineering_guidelines(soup)

    def extract_parameters(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract parameters from markdown tables"""
        parameters = []

        tables = soup.find_all('table')
        for table in tables:
            # Check if table contains parameter information
            header_text = table.get_text().lower()
            if any(keyword in header_text for keyword in ['parameter', 'mo class', 'description']):
                rows = table.find_all('tr')
                if len(rows) > 1:  # Has header and data rows
                    headers = [th.get_text().strip() for th in rows[0].find_all(['th', 'td'])]

                    for row in rows[1:]:
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 3:  # Minimum expected columns
                            param_data = {}
                            for i, cell in enumerate(cells):
                                if i < len(headers):
                                    param_data[headers[i]] = cell.get_text().strip()

                            if 'parameter' in param_data or 'name' in param_data:
                                parameters.append(param_data)

        return parameters

    def extract_counters(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract PM counters"""
        counters = []

        # Look for counter sections
        for header in soup.find_all(['h2', 'h3', 'h4']):
            if 'counter' in header.get_text().lower():
                counter_text = ""
                next_sibling = header.find_next_sibling()
                while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3', 'h4']:
                    if next_sibling.name == 'table':
                        # Extract counter data from table
                        rows = next_sibling.find_all('tr')
                        for row in rows[1:]:  # Skip header
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 2:
                                counter_name = cells[0].get_text().strip()
                                counter_desc = cells[1].get_text().strip()
                                if counter_name:
                                    counters.append({
                                        'name': counter_name,
                                        'description': counter_desc,
                                        'category': self.determine_counter_category(counter_name)
                                    })
                    break

        return counters

    def determine_counter_category(self, counter_name: str) -> str:
        """Determine counter category based on naming convention"""
        counter_name = counter_name.lower()

        if 'quality' in counter_name or 'qos' in counter_name:
            return 'Quality'
        elif 'throughput' in counter_name or 'tp' in counter_name:
            return 'Throughput'
        elif 'delay' in counter_name or 'latency' in counter_name:
            return 'Latency'
        elif 'error' in counter_name or 'failure' in counter_name:
            return 'Error'
        elif 'utilization' in counter_name or 'usage' in counter_name:
            return 'Utilization'
        else:
            return 'General'

    def extract_events(self, soup: BeautifulSoup) -> List[Dict]:
        """Extract event definitions"""
        events = []

        # Look for event sections
        for header in soup.find_all(['h2', 'h3', 'h4']):
            if 'event' in header.get_text().lower():
                event_data = {
                    'name': header.get_text().strip(),
                    'description': '',
                    'triggers': [],
                    'parameters': []
                }

                # Extract description
                next_sibling = header.find_next_sibling()
                while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3', 'h4']:
                    event_data['description'] += next_sibling.get_text() + "\n"
                    next_sibling = next_sibling.find_next_sibling()

                events.append(event_data)

        return events

    def extract_dependencies(self, soup: BeautifulSoup) -> Dict:
        """Extract feature dependencies"""
        dependencies = {
            'prerequisites': [],
            'conflicts': [],
            'related_features': []
        }

        # Look for dependency sections
        text = soup.get_text().lower()

        # Prerequisites
        prereq_patterns = [
            r'prerequisite[:\s]+([^\n\r]+)',
            r'required[:\s]+([^\n\r]+)'
        ]
        for pattern in prereq_patterns:
            matches = re.findall(pattern, text)
            dependencies['prerequisites'].extend(matches)

        # Conflicts
        conflict_patterns = [
            r'conflict[:\s]+([^\n\r]+)',
            r'incompatible[:\s]+([^\n\r]+)'
        ]
        for pattern in conflict_patterns:
            matches = re.findall(pattern, text)
            dependencies['conflicts'].extend(matches)

        return dependencies

    def extract_engineering_guidelines(self, soup: BeautifulSoup) -> str:
        """Extract engineering guidelines"""
        guidelines = ""

        # Look for guidelines sections
        for header in soup.find_all(['h2', 'h3', 'h4']):
            if any(keyword in header.get_text().lower() for keyword in ['guideline', 'recommendation', 'best practice']):
                guidelines += f"\n## {header.get_text().strip()}\n"
                next_sibling = header.find_next_sibling()
                while next_sibling and next_sibling.name not in ['h1', 'h2', 'h3', 'h4']:
                    guidelines += next_sibling.get_text() + "\n"
                    next_sibling = next_sibling.find_next_sibling()

        return guidelines.strip()

    def extract_operations_info(self, soup: BeautifulSoup, feature_data: Dict):
        """Extract activation/deactivation operations"""
        feature_data['activation_step'] = self.extract_activation_step(soup)
        feature_data['deactivation_step'] = self.extract_deactivation_step(soup)

    def extract_activation_step(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract feature activation step"""
        # Look for activation commands
        activation_patterns = [
            r'activation[:\s\n]+([^.]*?featurestate[^.]*)',
            r'set\s+featurestate[^\n]*',
            r'activat[^\n]*feature'
        ]

        text = soup.get_text()
        for pattern in activation_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() if match.groups() else match.group(0).strip()

        return None

    def extract_deactivation_step(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract feature deactivation step"""
        # Look for deactivation commands
        deactivation_patterns = [
            r'deactivation[:\s\n]+([^.]*?featurestate[^.]*)',
            r'deactivat[^\n]*feature'
        ]

        text = soup.get_text()
        for pattern in deactivation_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip() if match.groups() else match.group(0).strip()

        return None

    def update_ericsson_stats(self, feature_data: Dict):
        """Update Ericsson-specific statistics"""
        if feature_data.get('faj_number'):
            self.stats.ericsson_stats['faj_numbers_found'] += 1

        if feature_data.get('cxc_code'):
            self.stats.ericsson_stats['cxc_codes_extracted'] += 1

        if feature_data.get('parameters'):
            self.stats.ericsson_stats['parameters_extracted'] += len(feature_data['parameters'])

        if feature_data.get('counters'):
            self.stats.ericsson_stats['counters_extracted'] += len(feature_data['counters'])

        if feature_data.get('events'):
            self.stats.ericsson_stats['events_extracted'] += len(feature_data['events'])

        if feature_data.get('engineering_guidelines'):
            self.stats.ericsson_stats['engineering_guidelines_found'] += 1

    def build_indices(self):
        """Build search indices for fast feature lookup"""
        logger.info("🔍 Building search indices...")

        # Load all processed features
        features_dir = self.output_dir / "ericsson_data" / "features"
        for feature_file in features_dir.glob("feature_*.json"):
            try:
                feature_data = json.loads(feature_file.read_text())
                feature_id = feature_data.get('id')

                if feature_id:
                    # Parameter index
                    for param in feature_data.get('parameters', []):
                        param_name = param.get('parameter', param.get('name', ''))
                        if param_name:
                            self.parameter_index[param_name.lower()].append(feature_id)

                    # Counter index
                    for counter in feature_data.get('counters', []):
                        counter_name = counter.get('name', '')
                        if counter_name:
                            self.counter_index[counter_name.lower()].append(feature_id)

                    # CXC index
                    cxc_code = feature_data.get('cxc_code')
                    if cxc_code:
                        self.cxc_index[cxc_code] = feature_id

                    # Name index (tokenized)
                    feature_name = feature_data.get('name', '')
                    if feature_name:
                        tokens = re.findall(r'\w+', feature_name.lower())
                        for token in tokens:
                            if len(token) > 2:  # Skip very short tokens
                                self.name_index.setdefault(token, []).append(feature_id)

            except Exception as e:
                logger.warning(f"Error indexing {feature_file.name}: {e}")

        # Save indices
        indices_dir = self.output_dir / "ericsson_data" / "indices"
        indices_dir.mkdir(exist_ok=True)

        indices = {
            'parameters': dict(self.parameter_index),
            'counters': dict(self.counter_index),
            'cxc_codes': self.cxc_index,
            'names': self.name_index
        }

        for index_name, index_data in indices.items():
            index_file = indices_dir / f"{index_name}_index.json"
            index_file.write_text(json.dumps(index_data, indent=2))

        logger.info(f"✅ Built indices: {len(self.cxc_index)} CXC codes, {len(self.parameter_index)} parameter types")

    def finalize_processing(self):
        """Finalize Ericsson-specific processing"""
        # Call parent finalization
        super().finalize_processing()

        # Build Ericsson-specific indices
        self.build_indices()

        # Create Ericsson-specific summary
        self.create_ericsson_summary()

    def create_ericsson_summary(self):
        """Create Ericsson-specific processing summary"""
        ericsson_summary = {
            'ericsson_statistics': self.stats.ericsson_stats,
            'feature_categories': self.analyze_feature_categories(),
            'cxc_distribution': self.analyze_cxc_distribution(),
            'parameter_coverage': self.analyze_parameter_coverage(),
            'processing_efficiency': {
                'features_per_file': self.stats.ericsson_stats['faj_numbers_found'] / max(self.stats.total_files, 1),
                'parameters_per_feature': self.stats.ericsson_stats['parameters_extracted'] / max(self.stats.ericsson_stats['faj_numbers_found'], 1),
                'cxc_extraction_rate': self.stats.ericsson_stats['cxc_codes_extracted'] / max(self.stats.ericsson_stats['faj_numbers_found'], 1)
            }
        }

        summary_file = self.output_dir / "ericsson_data" / "ericsson_summary.json"
        summary_file.write_text(json.dumps(ericsson_summary, indent=2, default=str))

        logger.info(f"📄 Ericsson summary saved to {summary_file}")

    def analyze_feature_categories(self) -> Dict[str, int]:
        """Analyze feature categories based on naming patterns"""
        categories = defaultdict(int)

        # This would analyze processed features to categorize them
        # Placeholder implementation
        features_dir = self.output_dir / "ericsson_data" / "features"
        for feature_file in features_dir.glob("feature_*.json"):
            try:
                feature_data = json.loads(feature_file.read_text())
                feature_name = feature_data.get('name', '').lower()

                if 'mobility' in feature_name:
                    categories['Mobility'] += 1
                elif 'capacity' in feature_name:
                    categories['Capacity'] += 1
                elif 'quality' in feature_name or 'qos' in feature_name:
                    categories['Quality'] += 1
                elif 'energy' in feature_name or 'power' in feature_name:
                    categories['Energy'] += 1
                else:
                    categories['Other'] += 1

            except Exception:
                pass

        return dict(categories)

    def analyze_cxc_distribution(self) -> Dict[str, int]:
        """Analyze distribution of CXC codes"""
        cxc_ranges = defaultdict(int)

        for cxc_code in self.cxc_index.keys():
            if cxc_code:
                # Categorize by first digit (rough category)
                first_digit = cxc_code[0]
                cxc_ranges[f'CXC{first_digit}XXXX'] += 1

        return dict(cxc_ranges)

    def analyze_parameter_coverage(self) -> Dict[str, int]:
        """Analyze parameter coverage across features"""
        param_frequency = defaultdict(int)

        for param_name, feature_ids in self.parameter_index.items():
            param_frequency[param_name] = len(feature_ids)

        # Return top 10 most common parameters
        return dict(sorted(param_frequency.items(), key=lambda x: x[1], reverse=True)[:10])


def main():
    """Example usage of the enhanced Ericsson processor"""
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced Ericsson Feature Processor")
    parser.add_argument("--source", required=True, help="Source directory with Ericsson markdown files")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--max-memory", type=float, default=1024, help="Max memory in MB")
    parser.add_argument("--limit", type=int, help="Limit number of files to process")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume capability")
    parser.add_argument("--no-gc", action="store_true", help="Disable automatic garbage collection")

    args = parser.parse_args()

    # Create enhanced Ericsson processor
    processor = EnhancedEricssonProcessor(
        source_dir=args.source,
        output_dir=args.output,
        batch_size=args.batch_size,
        max_memory_mb=args.max_memory,
        auto_gc=not args.no_gc,
        resume=not args.no_resume
    )

    # Start processing
    processor.process_all(limit=args.limit, pattern="*.md")


if __name__ == "__main__":
    main()