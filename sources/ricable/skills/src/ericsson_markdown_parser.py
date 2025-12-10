#!/usr/bin/env python3
"""
Ericsson Markdown Parser Module

This module provides comprehensive markdown parsing functionality for Ericsson feature documentation.
It uses BeautifulSoup4 for HTML parsing after markdown conversion and implements robust extraction
of FAJ numbers, feature names, parameters, counters, CXC codes, and activation sections.

Key features:
- FAJ number extraction (FAJ XXX XXXX pattern)
- Feature name extraction from H1 tags and tables
- Table parsing for parameters and counters with proper column mapping
- CXC code extraction from activation sections
- Error handling for malformed tables and missing data
- Returns structured data that matches the EricssonFeature model
"""

import re
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
from collections import defaultdict

import markdown
from bs4 import BeautifulSoup, Tag, NavigableString


@dataclass
class ParsedFeature:
    """Structured data model for parsed Ericsson feature documentation"""
    # Identity information
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

    # Technical details
    parameters: List[Dict] = None
    counters: List[Dict] = None
    events: List[Dict] = None

    # Dependencies
    dependencies: Dict = None

    # Operations
    activation_step: Optional[str] = None
    deactivation_step: Optional[str] = None

    # Guidelines
    engineering_guidelines: str = ""

    # Impact information
    network_impact: Dict = None
    performance_impact: Dict = None

    # Metadata
    source_file: str = ""
    file_hash: str = ""
    processed_at: str = ""

    def __post_init__(self):
        """Initialize default values for lists and dicts"""
        if self.parameters is None:
            self.parameters = []
        if self.counters is None:
            self.counters = []
        if self.events is None:
            self.events = []
        if self.dependencies is None:
            self.dependencies = {
                'prerequisites': [],
                'related': [],
                'conflicts': []
            }
        if self.network_impact is None:
            self.network_impact = {}
        if self.performance_impact is None:
            self.performance_impact = {}


class MarkdownParseError(Exception):
    """Custom exception for markdown parsing errors"""
    pass


class EricssonMarkdownParser:
    """
    Comprehensive markdown parser for Ericsson feature documentation.

    This parser handles the specific structure and format of Ericsson technical documentation,
    extracting all relevant information into structured data format.
    """

    def __init__(self):
        """Initialize the parser with default configuration"""
        self.faj_patterns = [
            r'FAJ\s*(\d{3}\s*\d{4})',  # FAJ 121 3094 or FAJ 1213094
            r'Feature\s+Identity\s*\|\s*FAJ\s*(\d{3}\s*\d{4})',  # In table format
            r'FAJ\s+(\d{3})\s+(\d{4})',  # FAJ 121 4219 with separate groups
        ]

        self.cxc_patterns = [
            r'CXC\s*(\d{6})',  # CXC 4011808
            r'FeatureState=(CXC\d+)',  # FeatureState=CXC4011808
            r'MO\s+instance\s+(\w*CXC\d+\w*)',  # MO instance containing CXC
        ]

        self.parameter_patterns = [
            r'([A-Z][a-zA-Z]*\.[a-zA-Z][a-zA-Z0-9]*)',  # MO.Parameter format
            r'([a-z][a-zA-Z]*[A-Z][a-zA-Z]*)',  # camelCase parameters
        ]

        self.counter_patterns = [
            r'pm([A-Za-z0-9]+)',  # pmCounterName
            r'PM\s+([A-Za-z0-9]+)',  # PM CounterName
            r'Pm([A-Za-z0-9]+)',  # PmCounterName
        ]

    def parse_markdown_file(self, file_path: Union[str, Path]) -> ParsedFeature:
        """
        Parse a markdown file and extract feature information.

        Args:
            file_path: Path to the markdown file to parse

        Returns:
            ParsedFeature: Structured feature data

        Raises:
            MarkdownParseError: If file cannot be parsed or no valid FAJ ID found
        """
        file_path = Path(file_path)

        try:
            # Read file content
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise MarkdownParseError(f"Failed to read file {file_path}: {e}")

        # Convert markdown to HTML
        try:
            html = markdown.markdown(
                content,
                extensions=['tables', 'fenced_code', 'toc']
            )
            soup = BeautifulSoup(html, 'html.parser')
        except Exception as e:
            raise MarkdownParseError(f"Failed to convert markdown to HTML: {e}")

        # Extract feature information
        feature = self._extract_feature_identity(soup)
        if not feature or not feature.id:
            raise MarkdownParseError(f"No valid FAJ ID found in {file_path}")

        # Set metadata
        feature.source_file = str(file_path)
        feature.file_hash = self._calculate_file_hash(file_path)
        feature.processed_at = time.strftime('%Y-%m-%d %H:%M:%S')

        # Extract content sections
        feature.description = self._extract_section_content(soup, "Overview")
        feature.summary = self._extract_summary(soup)

        # Extract technical details
        feature.parameters = self._extract_parameters(soup)
        feature.counters = self._extract_counters(soup)
        feature.events = self._extract_events(soup)

        # Extract dependencies
        feature.dependencies = self._extract_dependencies(soup)

        # Extract activation/deactivation steps
        feature.activation_step = self._extract_activation_step(soup)
        feature.deactivation_step = self._extract_deactivation_step(soup)

        # Extract guidelines
        feature.engineering_guidelines = self._extract_engineering_guidelines(soup)

        # Extract impact information
        feature.network_impact = self._extract_network_impact(soup)
        feature.performance_impact = self._extract_performance_impact(soup)

        return feature

    def _extract_feature_identity(self, soup: BeautifulSoup) -> Optional[ParsedFeature]:
        """
        Extract feature identity information from the documentation.

        Looks for:
        - FAJ ID in various formats
        - Feature name from H1 tags and tables
        - Value package information
        - Node type and access type
        - CXC codes

        Args:
            soup: BeautifulSoup object of the parsed markdown

        Returns:
            ParsedFeature with identity information, or None if no FAJ ID found
        """
        feature = ParsedFeature()

        # Extract FAJ ID
        faj_id = self._extract_faj_id(soup)
        if not faj_id:
            return None
        feature.id = faj_id

        # Extract feature name (try multiple sources)
        feature.name = self._extract_feature_name(soup)

        # Extract information from feature identity table
        identity_table = self._find_feature_identity_table(soup)
        if identity_table:
            self._parse_identity_table(identity_table, feature)

        # Extract CXC code
        feature.cxc_code = self._extract_cxc_code(soup)

        return feature

    def _extract_faj_id(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract FAJ ID from the document using multiple patterns.

        Args:
            soup: BeautifulSoup object

        Returns:
            FAJ ID string or None if not found
        """
        text = soup.get_text()

        for pattern in self.faj_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Handle different group structures
                if len(match.groups()) == 2:
                    # Pattern like r'FAJ\s+(\d{3})\s+(\d{4})'
                    return f"{match.group(1)} {match.group(2)}"
                else:
                    # Pattern with single group
                    faj_id = match.group(1)
                    # Ensure proper spacing
                    if re.match(r'\d{6}', faj_id):
                        return f"{faj_id[:3]} {faj_id[3:]}"
                    return faj_id

        return None

    def _extract_feature_name(self, soup: BeautifulSoup) -> str:
        """
        Extract feature name from H1 tags or other sources.

        Args:
            soup: BeautifulSoup object

        Returns:
            Feature name string
        """
        # Try H1 tag first
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags:
            text = h1.get_text().strip()
            if text and text != '#' and len(text) > 5:  # Avoid empty or generic headers
                # Clean up common prefixes
                text = re.sub(r'^#+\s*', '', text)
                if not text.startswith('Contents') and 'Table of Contents' not in text:
                    return text

        # Try to find in the first paragraph after header
        first_p = soup.find('p')
        if first_p:
            text = first_p.get_text().strip()
            if len(text) > 20 and not text.startswith('Summary'):
                # Try to extract feature name from first sentence
                sentences = re.split(r'[.!?]+', text)
                if sentences:
                    first_sentence = sentences[0].strip()
                    if 'feature' in first_sentence.lower():
                        # Extract the feature name
                        match = re.search(r'(.+?)\s+feature', first_sentence, re.IGNORECASE)
                        if match:
                            return match.group(1).strip()

        return "Unknown Feature"

    def _find_feature_identity_table(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Find the feature identity table in the document.

        Args:
            soup: BeautifulSoup object

        Returns:
            Table tag or None if not found
        """
        tables = soup.find_all('table')

        for table in tables:
            text = table.get_text().lower()
            if any(keyword in text for keyword in [
                'feature name', 'feature identity', 'value package', 'node type'
            ]):
                return table

        return None

    def _parse_identity_table(self, table: Tag, feature: ParsedFeature):
        """
        Parse the feature identity table and extract information.

        Args:
            table: BeautifulSoup table tag
            feature: ParsedFeature object to update
        """
        rows = table.find_all('tr')

        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                key = cells[0].get_text().strip().lower()
                value = cells[1].get_text().strip()

                if 'feature name' in key:
                    feature.name = value
                elif 'value package name' in key:
                    feature.value_package = value
                elif 'value package identity' in key and 'faj' in value.lower():
                    # Extract FAJ ID from value package identity
                    match = re.search(r'faj\s*(\d{3}\s*\d{4})', value, re.IGNORECASE)
                    if match:
                        feature.value_package_id = match.group(1)
                elif 'node type' in key:
                    feature.node_type = value
                elif 'access type' in key:
                    feature.access_type = value
                elif 'feature identity' in key and 'faj' in value.lower():
                    match = re.search(r'faj\s*(\d{3}\s*\d{4})', value, re.IGNORECASE)
                    if match:
                        feature.id = match.group(1)

    def _extract_cxc_code(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract CXC code from the document.

        Args:
            soup: BeautifulSoup object

        Returns:
            CXC code string or None if not found
        """
        text = soup.get_text()

        for pattern in self.cxc_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                cxc_code = match.group(1)
                # Ensure it starts with CXC
                if not cxc_code.upper().startswith('CXC'):
                    cxc_code = f"CXC{cxc_code}"
                return cxc_code.upper()

        return None

    def _extract_section_content(self, soup: BeautifulSoup, section_name: str) -> str:
        """
        Extract content from a specific section by heading.

        Args:
            soup: BeautifulSoup object
            section_name: Name of the section to extract

        Returns:
            Section content as string
        """
        # Look for headings containing the section name
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if section_name.lower() in tag.get_text().lower():
                return self._extract_content_until_next_heading(tag)

        return ""

    def _extract_content_until_next_heading(self, start_tag: Tag) -> str:
        """
        Extract all content from a starting tag until the next heading of same or higher level.

        Args:
            start_tag: The tag to start extraction from

        Returns:
            Extracted content as string
        """
        content_parts = []
        current = start_tag.next_sibling

        while current:
            if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Stop if we hit a heading of same or higher level
                if self._get_heading_level(current.name) <= self._get_heading_level(start_tag.name):
                    break

            if hasattr(current, 'get_text'):
                text = current.get_text().strip()
                if text and not text.startswith('Image'):
                    content_parts.append(text)

            current = current.next_sibling

        return '\n'.join(content_parts)

    def _get_heading_level(self, tag_name: str) -> int:
        """Get the numeric level of a heading tag"""
        if tag_name and tag_name.startswith('h') and len(tag_name) == 2:
            return int(tag_name[1])
        return 6  # Default to lowest level

    def _extract_summary(self, soup: BeautifulSoup) -> str:
        """
        Extract feature summary from the document.

        Args:
            soup: BeautifulSoup object

        Returns:
            Summary string
        """
        # Look for explicit Summary section
        summary = self._extract_section_content(soup, "Summary")
        if summary:
            return summary

        # Look for Overview section and extract first paragraph
        overview = self._extract_section_content(soup, "Overview")
        if overview:
            lines = overview.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 50 and 'feature' in line.lower():
                    return line

        # Fallback to first meaningful paragraph
        for p in soup.find_all('p'):
            text = p.get_text().strip()
            if len(text) > 50 and 'feature' in text.lower():
                return text

        return ""

    def _extract_parameters(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract parameters from documentation tables and text.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of parameter dictionaries
        """
        parameters = []

        # Extract from parameter tables
        parameters.extend(self._extract_parameters_from_tables(soup))

        # Extract from text mentions
        parameters.extend(self._extract_parameters_from_text(soup))

        # Remove duplicates
        seen_names = set()
        unique_parameters = []
        for param in parameters:
            if param['name'] not in seen_names:
                seen_names.add(param['name'])
                unique_parameters.append(param)

        return unique_parameters

    def _extract_parameters_from_tables(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract parameters from documentation tables.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of parameter dictionaries
        """
        parameters = []
        tables = soup.find_all('table')

        for table in tables:
            # Check if this is a parameter table
            headers = [th.get_text().strip().lower() for th in table.find_all('th')]

            if any('parameter' in h for h in headers) or \
               any('attribute' in h for h in headers) or \
               any('name' in h and 'description' in h for h in headers):

                rows = table.find_all('tr')[1:]  # Skip header row

                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        param = self._parse_parameter_row(cells, headers)
                        if param and param['name']:
                            parameters.append(param)

        return parameters

    def _parse_parameter_row(self, cells: List[Tag], headers: List[str]) -> Optional[Dict]:
        """
        Parse a single parameter table row.

        Args:
            cells: List of table cell tags
            headers: List of column headers

        Returns:
            Parameter dictionary or None if invalid
        """
        # Map cells to headers
        cell_texts = [cell.get_text().strip() for cell in cells]

        param = {
            'name': '',
            'type': 'Unknown',
            'description': '',
            'mo_class': 'Unknown',
            'default_value': '',
            'range': ''
        }

        # Find name column
        name_col = None
        for i, header in enumerate(headers):
            if 'parameter' in header or 'attribute' in header or 'name' in header:
                name_col = i
                break

        if name_col is not None and name_col < len(cell_texts):
            param['name'] = cell_texts[name_col]
            param['mo_class'] = self._extract_mo_class(param['name'])

        # Find description column
        desc_col = None
        for i, header in enumerate(headers):
            if 'description' in header or 'meaning' in header:
                desc_col = i
                break

        if desc_col is not None and desc_col < len(cell_texts):
            param['description'] = cell_texts[desc_col]

        # Find type column
        type_col = None
        for i, header in enumerate(headers):
            if 'type' in header:
                type_col = i
                break

        if type_col is not None and type_col < len(cell_texts):
            param['type'] = cell_texts[type_col]

        return param if param['name'] else None

    def _extract_parameters_from_text(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract parameters from text mentions in the document.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of parameter dictionaries
        """
        parameters = []
        text = soup.get_text()

        # Find parameter mentions using patterns
        for pattern in self.parameter_patterns:
            matches = re.findall(pattern, text)
            for match in matches[:10]:  # Limit to avoid too many
                if '.' in match:
                    param = {
                        'name': match,
                        'type': 'Unknown',
                        'description': 'Parameter mentioned in documentation',
                        'mo_class': self._extract_mo_class(match)
                    }
                    parameters.append(param)

        return parameters

    def _extract_mo_class(self, param_name: str) -> str:
        """
        Extract MO class from parameter name.

        Args:
            param_name: Parameter name string

        Returns:
            MO class string
        """
        if '.' in param_name:
            return param_name.split('.')[0]
        return "Unknown"

    def _extract_counters(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract performance counters from documentation.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of counter dictionaries
        """
        counters = []
        text = soup.get_text()

        # Find all counter mentions
        found_counters = set()
        for pattern in self.counter_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            found_counters.update([match.upper() for match in matches])

        # Create counter entries
        for counter_name in sorted(found_counters):
            counter = {
                'name': f'pm{counter_name}',
                'description': f'Performance counter pm{counter_name}',
                'category': self._guess_counter_category(counter_name)
            }
            counters.append(counter)

        return counters

    def _guess_counter_category(self, counter: str) -> str:
        """
        Guess counter category based on name patterns.

        Args:
            counter: Counter name string

        Returns:
            Category string
        """
        counter_lower = counter.lower()

        if 'mimo' in counter_lower:
            return 'MIMO'
        elif any(word in counter_lower for word in ['sleep', 'energy', 'power']):
            return 'Energy Efficiency'
        elif any(word in counter_lower for word in ['handover', 'mobility', 'handoff']):
            return 'Mobility'
        elif any(word in counter_lower for word in ['throughput', 'tput', 'data']):
            return 'Throughput'
        elif any(word in counter_lower for word in ['quality', 'qos', 'qci']):
            return 'Quality of Service'
        elif any(word in counter_lower for word in ['signal', 'rsrp', 'rsrq', 'sinr']):
            return 'Signal Quality'
        elif any(word in counter_lower for word in ['load', 'utilization']):
            return 'Load Management'
        else:
            return 'General'

    def _extract_events(self, soup: BeautifulSoup) -> List[Dict]:
        """
        Extract events from documentation tables.

        Args:
            soup: BeautifulSoup object

        Returns:
            List of event dictionaries
        """
        events = []
        tables = soup.find_all('table')

        for table in tables:
            headers = [th.get_text().strip().lower() for th in table.find_all('th')]

            if any('event' in h for h in headers):
                rows = table.find_all('tr')[1:]  # Skip header

                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 2:
                        event = {
                            'name': cells[0].get_text().strip(),
                            'type': cells[1].get_text().strip() if len(cells) > 1 else '',
                            'description': cells[2].get_text().strip() if len(cells) > 2 else '',
                            'trigger': cells[3].get_text().strip() if len(cells) > 3 else ''
                        }
                        if event['name']:
                            events.append(event)

        return events

    def _extract_dependencies(self, soup: BeautifulSoup) -> Dict:
        """
        Extract feature dependencies from documentation.

        Args:
            soup: BeautifulSoup object

        Returns:
            Dependencies dictionary with prerequisites, related, and conflicts
        """
        dependencies = {
            'prerequisites': [],
            'related': [],
            'conflicts': []
        }

        # Look for Dependencies section
        dep_section = self._extract_section_content(soup, "Dependencies")
        if not dep_section:
            # Try alternative section names
            dep_section = self._extract_section_content(soup, "Feature Dependencies")

        if dep_section:
            # Extract FAJ references
            faj_refs = re.findall(r'FAJ\s*(\d{3}\s*\d{4})', dep_section)

            # Look for dependency table
            dep_table = self._find_dependencies_table(soup)
            if dep_table:
                self._parse_dependencies_table(dep_table, dependencies)
            else:
                # Categorize based on context
                self._categorize_dependencies_by_context(dep_section, faj_refs, dependencies)

        return dependencies

    def _find_dependencies_table(self, soup: BeautifulSoup) -> Optional[Tag]:
        """
        Find the dependencies table in the document.

        Args:
            soup: BeautifulSoup object

        Returns:
            Table tag or None if not found
        """
        tables = soup.find_all('table')

        for table in tables:
            text = table.get_text().lower()
            if 'dependencies' in text or 'relationship' in text:
                return table

        return None

    def _parse_dependencies_table(self, table: Tag, dependencies: Dict):
        """
        Parse the dependencies table and extract relationships.

        Args:
            table: BeautifulSoup table tag
            dependencies: Dependencies dictionary to update
        """
        rows = table.find_all('tr')[1:]  # Skip header

        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                feature_text = cells[0].get_text()
                relationship = cells[1].get_text().lower() if len(cells) > 1 else ''

                # Extract FAJ ID from feature text
                faj_match = re.search(r'FAJ\s*(\d{3}\s*\d{4})', feature_text)
                if faj_match:
                    faj_id = faj_match.group(1)

                    if 'prerequisite' in relationship:
                        dependencies['prerequisites'].append(faj_id)
                    elif 'conflict' in relationship:
                        dependencies['conflicts'].append(faj_id)
                    else:
                        dependencies['related'].append(faj_id)

    def _categorize_dependencies_by_context(self, section: str, faj_refs: List[str], dependencies: Dict):
        """
        Categorize dependencies based on context in the text.

        Args:
            section: Dependencies section text
            faj_refs: List of FAJ references found
            dependencies: Dependencies dictionary to update
        """
        section_lower = section.lower()

        for faj in faj_refs:
            if 'prerequisite' in section_lower or 'require' in section_lower:
                dependencies['prerequisites'].append(faj)
            elif 'conflict' in section_lower:
                dependencies['conflicts'].append(faj)
            else:
                dependencies['related'].append(faj)

    def _extract_activation_step(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract activation command from documentation.

        Args:
            soup: BeautifulSoup object

        Returns:
            Activation step string or None if not found
        """
        # Look for Activation section
        activation_section = self._extract_section_content(soup, "Activating")
        if not activation_section:
            activation_section = self._extract_section_content(soup, "Activation")

        if activation_section:
            # Look for FeatureState command
            lines = activation_section.split('\n')
            for line in lines:
                if 'FeatureState' in line and 'ACTIVATED' in line:
                    return line.strip()

        # Search entire document for activation pattern
        full_text = soup.get_text()
        match = re.search(
            r'1\.\s+Set\s+the\s+FeatureState\.featureState\s+attribute\s+to\s+ACTIVATED\s+in\s+the\s+(\S+)',
            full_text
        )
        if match:
            return f"1. Set the FeatureState.featureState attribute to ACTIVATED in the {match.group(1)} MO instance."

        return None

    def _extract_deactivation_step(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract deactivation command from documentation.

        Args:
            soup: BeautifulSoup object

        Returns:
            Deactivation step string or None if not found
        """
        # Look for Deactivation section
        deactivation_section = self._extract_section_content(soup, "Deactivating")
        if not deactivation_section:
            deactivation_section = self._extract_section_content(soup, "Deactivation")

        if deactivation_section:
            # Look for FeatureState command
            lines = deactivation_section.split('\n')
            for line in lines:
                if 'FeatureState' in line and 'DEACTIVATED' in line:
                    return line.strip()

        # Search entire document for deactivation pattern
        full_text = soup.get_text()
        match = re.search(
            r'1\.\s+Set\s+the\s+FeatureState\.featureState\s+attribute\s+to\s+DEACTIVATED\s+in\s+the\s+(\S+)',
            full_text
        )
        if match:
            return f"1. Set the FeatureState.featureState attribute to DEACTIVATED in the {match.group(1)} MO instance."

        return None

    def _extract_engineering_guidelines(self, soup: BeautifulSoup) -> str:
        """
        Extract engineering guidelines from documentation.

        Args:
            soup: BeautifulSoup object

        Returns:
            Engineering guidelines string
        """
        guidelines = []

        # Look for Engineering Guidelines section
        section = self._extract_section_content(soup, "Engineering Guidelines")

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

    def _extract_network_impact(self, soup: BeautifulSoup) -> Dict:
        """
        Extract network impact information from documentation.

        Args:
            soup: BeautifulSoup object

        Returns:
            Network impact dictionary
        """
        impact = {}

        # Look for Network Impact section
        section = self._extract_section_content(soup, "Network Impact")

        if section:
            section_lower = section.lower()

            # Extract key impacts
            if 'capacity' in section_lower:
                impact['capacity'] = 'Affects cell capacity during operation'
            if 'coverage' in section_lower:
                impact['coverage'] = 'May temporarily affect cell coverage'
            if 'performance' in section_lower:
                impact['performance'] = 'Performance impact during reconfiguration'
            if 'interference' in section_lower:
                impact['interference'] = 'May affect interference levels'
            if 'handover' in section_lower:
                impact['handover'] = 'Affects handover performance'

        return impact

    def _extract_performance_impact(self, soup: BeautifulSoup) -> Dict:
        """
        Extract performance impact information from documentation.

        Args:
            soup: BeautifulSoup object

        Returns:
            Performance impact dictionary
        """
        impact = {}

        # Look for Performance section
        section = self._extract_section_content(soup, "Performance")

        if section:
            section_lower = section.lower()

            # Extract key metrics
            if 'kpi' in section_lower:
                impact['kpi'] = 'Affects KPI monitoring'
            if 'counters' in section_lower:
                impact['counters'] = 'New counters introduced'
            if 'throughput' in section_lower:
                impact['throughput'] = 'Affects throughput measurements'
            if 'latency' in section_lower:
                impact['latency'] = 'May affect latency measurements'

        return impact

    def _calculate_file_hash(self, file_path: Path) -> str:
        """
        Calculate MD5 hash of file for caching purposes.

        Args:
            file_path: Path to the file

        Returns:
            MD5 hash string
        """
        content = file_path.read_bytes()
        return hashlib.md5(content).hexdigest()


# Convenience functions for direct usage
def parse_ericsson_markdown(file_path: Union[str, Path]) -> ParsedFeature:
    """
    Convenience function to parse an Ericsson markdown file.

    Args:
        file_path: Path to the markdown file

    Returns:
        ParsedFeature object with extracted data

    Raises:
        MarkdownParseError: If parsing fails
    """
    parser = EricssonMarkdownParser()
    return parser.parse_markdown_file(file_path)


def batch_parse_markdown_files(file_paths: List[Union[str, Path]]) -> List[ParsedFeature]:
    """
    Parse multiple markdown files and return list of parsed features.

    Args:
        file_paths: List of file paths to parse

    Returns:
        List of ParsedFeature objects
    """
    parser = EricssonMarkdownParser()
    features = []

    for file_path in file_paths:
        try:
            feature = parser.parse_markdown_file(file_path)
            features.append(feature)
        except MarkdownParseError as e:
            print(f"Failed to parse {file_path}: {e}")

    return features


# Main execution for testing
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python ericsson_markdown_parser.py <markdown_file>")
        sys.exit(1)

    try:
        feature = parse_ericsson_markdown(sys.argv[1])

        print("Successfully parsed feature:")
        print(f"  ID: {feature.id}")
        print(f"  Name: {feature.name}")
        print(f"  CXC Code: {feature.cxc_code}")
        print(f"  Parameters: {len(feature.parameters)}")
        print(f"  Counters: {len(feature.counters)}")
        print(f"  Dependencies: {len(feature.dependencies['prerequisites'] + feature.dependencies['related'] + feature.dependencies['conflicts'])}")

    except MarkdownParseError as e:
        print(f"Error: {e}")
        sys.exit(1)