#!/usr/bin/env python3
"""
Ericsson Feature Search Index System
Advanced search index generation for fast feature lookups with partial matching,
fuzzy searching, and cross-references between features.
Handles 2000+ features with efficient lookup capabilities.
"""

import os
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import difflib
import hashlib


@dataclass
class SearchResult:
    """Individual search result with relevance score"""
    feature_id: str
    feature_name: str
    relevance_score: float
    match_type: str  # 'exact', 'partial', 'fuzzy', 'dependency'
    match_context: str = ""
    cxc_code: Optional[str] = None


@dataclass
class SearchIndex:
    """Complete search index structure"""
    parameter_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    counter_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    cxc_index: Dict[str, str] = field(default_factory=dict)
    name_index: Dict[str, str] = field(default_factory=dict)
    name_tokens_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    dependency_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    reverse_dependency_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    category_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    value_package_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    node_type_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))
    fuzzy_index: Dict[str, List[str]] = field(default_factory=lambda: defaultdict(list))

    # Metadata
    total_features: int = 0
    index_version: str = "1.0"
    built_at: str = ""
    build_time: float = 0.0


class EricssonSearchIndexBuilder:
    """Advanced search index builder for Ericsson features"""

    def __init__(self, features_dir: str, output_dir: str = "output"):
        self.features_dir = Path(features_dir)
        self.output_dir = Path(output_dir)
        self.features: Dict[str, Dict] = {}

        # Initialize search index
        self.search_index = SearchIndex()

        # Performance tracking
        self.stats = {
            'features_loaded': 0,
            'indexing_time': 0,
            'total_lookups': 0,
            'avg_lookup_time': 0
        }

        # Fuzzy matching threshold
        self.fuzzy_threshold = 0.7

        # Stop words for name indexing
        self.stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'feature', 'features'
        }

    def load_features(self) -> None:
        """Load all processed feature data from JSON files"""
        print("📁 Loading processed features...")

        if not self.features_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {self.features_dir}")

        start_time = time.time()

        for feature_file in self.features_dir.glob("*.json"):
            try:
                feature_data = json.loads(feature_file.read_text())
                feature_id = feature_data.get('id', '')
                if feature_id:
                    self.features[feature_id] = feature_data
                    self.stats['features_loaded'] += 1
            except Exception as e:
                print(f"⚠️  Error loading {feature_file}: {e}")

        load_time = time.time() - start_time
        print(f"✅ Loaded {len(self.features)} features in {load_time:.2f}s")

    def tokenize_name(self, name: str) -> List[str]:
        """Enhanced tokenization with better partial matching support"""
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', name.lower())

        # Filter out stop words and short words
        tokens = [word for word in words if word not in self.stop_words and len(word) > 2]

        # Add original name and normalized versions
        tokens.extend([
            name.lower(),
            re.sub(r'[^a-z0-9]', '', name.lower()),
            re.sub(r'[^a-z0-9]', ' ', name.lower())  # Replace separators with spaces
        ])

        # Enhanced partial token generation
        for word in tokens[:]:  # Copy to avoid modifying during iteration
            if len(word) > 3:
                # Add all meaningful prefixes (length 3+)
                for i in range(3, len(word)):
                    prefix = word[:i]
                    if len(prefix) >= 3:
                        tokens.append(prefix)

                # Add suffixes for common endings
                common_suffixes = ['ing', 'tion', 'ment', 'ness', 'ity', 'tion', 'ance', 'ence']
                for suffix in common_suffixes:
                    if word.endswith(suffix):
                        root = word[:-len(suffix)]
                        if len(root) >= 3:
                            tokens.append(root)

        # Add character n-grams for fuzzy matching (3-grams)
        clean_name = re.sub(r'[^a-z0-9]', '', name.lower())
        if len(clean_name) >= 3:
            for i in range(len(clean_name) - 2):
                ngram = clean_name[i:i+3]
                tokens.append(ngram)

        return list(set(tokens))  # Remove duplicates while preserving order

    def build_parameter_index(self) -> None:
        """Build parameter name to feature mapping"""
        print("🔧 Building parameter index...")

        for feature_id, feature in self.features.items():
            for param in feature.get('parameters', []):
                param_name = param.get('name', '').lower()
                if param_name:
                    # Exact match
                    self.search_index.parameter_index[param_name].append(feature_id)

                    # Partial matches - split parameter name
                    param_parts = re.findall(r'\b\w+\b', param_name)
                    for part in param_parts:
                        if len(part) > 2:
                            self.search_index.parameter_index[part].append(feature_id)

    def build_counter_index(self) -> None:
        """Build PM counter to feature mapping"""
        print("📊 Building counter index...")

        for feature_id, feature in self.features.items():
            for counter in feature.get('counters', []):
                counter_name = counter.get('name', '').lower()
                if counter_name:
                    # Exact match
                    self.search_index.counter_index[counter_name].append(feature_id)

                    # Partial matches
                    counter_parts = re.findall(r'\b\w+\b', counter_name)
                    for part in counter_parts:
                        if len(part) > 2:
                            self.search_index.counter_index[part].append(feature_id)

    def build_cxc_index(self) -> None:
        """Build CXC code to feature mapping"""
        print("🏷️  Building CXC index...")

        for feature_id, feature in self.features.items():
            cxc_code = feature.get('cxc_code')
            if cxc_code:
                self.search_index.cxc_index[cxc_code.upper()] = feature_id

    def build_name_indices(self) -> None:
        """Build name-based search indices with tokenization"""
        print("📝 Building name indices...")

        for feature_id, feature in self.features.items():
            name = feature.get('name', '').lower()
            if name:
                # Exact name match
                self.search_index.name_index[name] = feature_id

                # Tokenized name index
                tokens = self.tokenize_name(name)
                for token in tokens:
                    self.search_index.name_tokens_index[token].append(feature_id)

    def build_dependency_indices(self) -> None:
        """Build dependency relationship indices"""
        print("🔗 Building dependency indices...")

        for feature_id, feature in self.features.items():
            dependencies = feature.get('dependencies', {})

            # Prerequisites
            for prereq in dependencies.get('prerequisites', []):
                if isinstance(prereq, str):
                    self.search_index.dependency_index[feature_id].append(prereq)
                    self.search_index.reverse_dependency_index[prereq].append(feature_id)

            # Related features
            for related in dependencies.get('related_features', []):
                if isinstance(related, str):
                    self.search_index.dependency_index[feature_id].append(related)
                    self.search_index.reverse_dependency_index[related].append(feature_id)

    def build_category_indices(self) -> None:
        """Build category-based indices"""
        print("📂 Building category indices...")

        for feature_id, feature in self.features.items():
            # Value package
            value_package = feature.get('value_package', '').lower()
            if value_package:
                self.search_index.value_package_index[value_package].append(feature_id)

            # Node type
            node_type = feature.get('node_type', '').lower()
            if node_type:
                self.search_index.node_type_index[node_type].append(feature_id)

            # Auto-categorize based on description/content
            description = feature.get('description', '').lower()
            summary = feature.get('summary', '').lower()
            combined_text = f"{description} {summary}"

            # Common categories in Ericsson features
            categories = {
                'performance': ['performance', 'throughput', 'capacity', 'load', 'optimization'],
                'mobility': ['handover', 'mobility', 'handoff', 'cell', 'reselection'],
                'quality': ['qos', 'quality', 'service', 'experience', 'latency'],
                'security': ['security', 'encryption', 'authentication', 'authorization'],
                'power': ['power', 'energy', 'consumption', 'sleep', 'battery'],
                'coverage': ['coverage', 'signal', 'strength', 'range', 'extension'],
                'capacity': ['capacity', 'users', 'connections', 'sessions', 'throughput']
            }

            for category, keywords in categories.items():
                if any(keyword in combined_text for keyword in keywords):
                    self.search_index.category_index[category].append(feature_id)

    def build_fuzzy_index(self) -> None:
        """Enhanced fuzzy matching index with improved typo tolerance"""
        print("🔍 Building enhanced fuzzy search index...")

        # Collect all searchable terms with metadata
        all_terms = {}  # term -> {'features': set, 'type': str, 'original': str}

        # Feature names
        for feature_id, feature in self.features.items():
            name = feature.get('name', '')
            if name:
                name_lower = name.lower()
                all_terms[name_lower] = {
                    'features': {feature_id},
                    'type': 'feature_name',
                    'original': name
                }

                # Add tokenized versions
                tokens = self.tokenize_name(name)
                for token in tokens:
                    if token not in all_terms:
                        all_terms[token] = {
                            'features': set(),
                            'type': 'token',
                            'original': token
                        }
                    all_terms[token]['features'].add(feature_id)

        # Parameter names
        for feature_id, feature in self.features.items():
            for param in feature.get('parameters', []):
                param_name = param.get('name', '')
                if param_name:
                    param_lower = param_name.lower()
                    if param_lower not in all_terms:
                        all_terms[param_lower] = {
                            'features': set(),
                            'type': 'parameter',
                            'original': param_name
                        }
                    all_terms[param_lower]['features'].add(feature_id)

        # Counter names
        for feature_id, feature in self.features.items():
            for counter in feature.get('counters', []):
                counter_name = counter.get('name', '')
                if counter_name:
                    counter_lower = counter_name.lower()
                    if counter_lower not in all_terms:
                        all_terms[counter_lower] = {
                            'features': set(),
                            'type': 'counter',
                            'original': counter_name
                        }
                    all_terms[counter_lower]['features'].add(feature_id)

        # Build enhanced fuzzy index with multiple matching strategies
        all_terms_list = list(all_terms.keys())

        for term, term_data in all_terms.items():
            if len(term) > 3:  # Only for meaningful terms
                similar_matches = []

                # Strategy 1: Direct string similarity (difflib)
                direct_similar = difflib.get_close_matches(
                    term, all_terms_list, n=10, cutoff=0.6
                )
                for similar_term in direct_similar:
                    if similar_term != term:
                        similarity_score = difflib.SequenceMatcher(None, term, similar_term).ratio()
                        similar_matches.append({
                            'term': similar_term,
                            'score': similarity_score,
                            'method': 'direct'
                        })

                # Strategy 2: Edit distance based matching
                for other_term in all_terms_list:
                    if other_term != term and abs(len(term) - len(other_term)) <= 2:
                        # Use Levenshtein distance approximation
                        edit_distance = self._levenshtein_distance(term, other_term)
                        if edit_distance <= 2 and edit_distance > 0:
                            similarity = 1 - (edit_distance / max(len(term), len(other_term)))
                            if similarity > 0.7:
                                similar_matches.append({
                                    'term': other_term,
                                    'score': similarity,
                                    'method': 'edit_distance'
                                })

                # Strategy 3: Common typo patterns
                typo_matches = self._find_typo_variants(term, all_terms_list)
                for typo_match in typo_matches:
                    similar_matches.append({
                        'term': typo_match,
                        'score': 0.75,
                        'method': 'typo_pattern'
                    })

                # Deduplicate and sort by score
                unique_matches = {}
                for match in similar_matches:
                    term_key = match['term']
                    if term_key not in unique_matches or match['score'] > unique_matches[term_key]['score']:
                        unique_matches[term_key] = match

                # Add to fuzzy index with scores
                sorted_matches = sorted(unique_matches.values(), key=lambda x: x['score'], reverse=True)
                for match in sorted_matches[:5]:  # Top 5 matches
                    self.search_index.fuzzy_index[term].extend(
                        list(all_terms[match['term']]['features'])
                    )

        # Remove duplicates and convert to list
        for term in self.search_index.fuzzy_index:
            self.search_index.fuzzy_index[term] = list(set(self.search_index.fuzzy_index[term]))

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def _find_typo_variants(self, term: str, all_terms: List[str]) -> List[str]:
        """Find common typo variants of a term"""
        variants = []
        term_lower = term.lower()

        # Common typo patterns
        typo_patterns = [
            # Double letters
            lambda t: re.sub(r'(.)\1', r'\1\1', t),  # Add double letters
            # Missing letters
            lambda t: re.sub(r'([aeiou])', '', t),   # Remove vowels
            # Swapped adjacent letters
            self._generate_swapped_variants,
            # Phonetic substitutions
            lambda t: t.replace('c', 'k').replace('s', 'z').replace('ph', 'f')
        ]

        for pattern_func in typo_patterns:
            if callable(pattern_func):
                if pattern_func == self._generate_swapped_variants:
                    variant_list = pattern_func(term_lower)
                else:
                    variant = pattern_func(term_lower)
                    variant_list = [variant] if variant else []

                for variant in variant_list:
                    if variant in all_terms and variant != term_lower:
                        variants.append(variant)

        return variants

    def _generate_swapped_variants(self, term: str) -> List[str]:
        """Generate variants with adjacent letters swapped"""
        variants = []
        for i in range(len(term) - 1):
            swapped = term[:i] + term[i+1] + term[i] + term[i+2:]
            variants.append(swapped)
        return variants

    def _find_features_for_term(self, term: str) -> List[str]:
        """Find features that contain a specific term"""
        features = []
        term_lower = term.lower()

        for feature_id, feature in self.features.items():
            if (term_lower in feature.get('name', '').lower() or
                term_lower in feature.get('description', '').lower() or
                term_lower in feature.get('summary', '').lower() or
                any(term_lower in param.get('name', '').lower()
                    for param in feature.get('parameters', [])) or
                any(term_lower in counter.get('name', '').lower()
                    for counter in feature.get('counters', []))):
                features.append(feature_id)

        return features

    def build_all_indices(self) -> None:
        """Build all search indices"""
        print("🚀 Building comprehensive search indices...")
        start_time = time.time()

        self.build_parameter_index()
        self.build_counter_index()
        self.build_cxc_index()
        self.build_name_indices()
        self.build_dependency_indices()
        self.build_category_indices()
        self.build_fuzzy_index()

        # Update metadata
        self.search_index.total_features = len(self.features)
        self.search_index.built_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.search_index.build_time = time.time() - start_time

        print(f"✅ All indices built in {self.search_index.build_time:.2f}s")

    def save_indices(self, output_file: Optional[str] = None) -> None:
        """Save search indices to optimized file structure"""
        if output_file is None:
            output_file = self.output_dir / "ericsson_data" / "search_index.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create optimized index structure for fast loading
        index_data = {
            'metadata': {
                'version': self.search_index.index_version,
                'total_features': self.search_index.total_features,
                'built_at': self.search_index.built_at,
                'build_time': self.search_index.build_time,
                'feature_hashes': self._compute_feature_hashes()
            },
            'indices': {}
        }

        # Save each index separately for better compression and loading
        index_configs = {
            'parameter_index': self.search_index.parameter_index,
            'counter_index': self.search_index.counter_index,
            'cxc_index': self.search_index.cxc_index,
            'name_index': self.search_index.name_index,
            'name_tokens_index': self.search_index.name_tokens_index,
            'dependency_index': self.search_index.dependency_index,
            'reverse_dependency_index': self.search_index.reverse_dependency_index,
            'category_index': self.search_index.category_index,
            'value_package_index': self.search_index.value_package_index,
            'node_type_index': self.search_index.node_type_index,
            'fuzzy_index': self.search_index.fuzzy_index
        }

        # Convert defaultdicts and optimize structure
        for index_name, index_data_raw in index_configs.items():
            if isinstance(index_data_raw, defaultdict):
                index_data_raw = dict(index_data_raw)

            # Optimize by removing duplicates and sorting for compression
            if isinstance(index_data_raw, dict):
                optimized_index = {}
                for key, value in index_data_raw.items():
                    if isinstance(value, list):
                        # Remove duplicates and sort for better compression
                        optimized_value = sorted(list(set(value)))
                        optimized_index[key] = optimized_value
                    else:
                        optimized_index[key] = value
                index_data['indices'][index_name] = optimized_index
            else:
                index_data['indices'][index_name] = index_data_raw

        # Use compact JSON format for better performance
        json_str = json.dumps(index_data, separators=(',', ':'), ensure_ascii=False)

        # Compress if the index is large
        if len(json_str) > 1024 * 1024:  # 1MB threshold
            import gzip
            compressed_path = output_path.with_suffix('.json.gz')
            with gzip.open(compressed_path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
            print(f"💾 Compressed search indices saved to {compressed_path}")
        else:
            output_path.write_text(json_str)
            print(f"💾 Search indices saved to {output_path}")

        # Also save individual index files for incremental loading
        self._save_split_indices(index_data['indices'], output_path.parent)

    def _compute_feature_hashes(self) -> Dict[str, str]:
        """Compute hashes for all features to detect changes"""
        feature_hashes = {}
        for feature_id, feature in self.features.items():
            # Create hash from key feature fields
            hash_content = f"{feature.get('name', '')}{feature.get('cxc_code', '')}{feature.get('description', '')}"
            feature_hashes[feature_id] = hashlib.md5(hash_content.encode()).hexdigest()
        return feature_hashes

    def _save_split_indices(self, indices: Dict, output_dir: Path) -> None:
        """Save individual index files for incremental loading"""
        indices_dir = output_dir / "indices_split"
        indices_dir.mkdir(exist_ok=True)

        for index_name, index_data in indices.items():
            index_file = indices_dir / f"{index_name}.json"
            index_file.write_text(json.dumps(index_data, separators=(',', ':')))

        print(f"📁 Split indices saved to {indices_dir}")

    def load_indices(self, index_file: Optional[str] = None) -> bool:
        """Load search indices from optimized file structure"""
        if index_file is None:
            index_file = self.output_dir / "ericsson_data" / "search_index.json"

        index_path = Path(index_file)

        # Try compressed version first
        compressed_path = index_path.with_suffix('.json.gz')
        if compressed_path.exists():
            try:
                import gzip
                with gzip.open(compressed_path, 'rt', encoding='utf-8') as f:
                    index_data = json.load(f)
                print(f"✅ Compressed search indices loaded from {compressed_path}")
            except Exception as e:
                print(f"⚠️  Error loading compressed indices: {e}")
                return False
        elif index_path.exists():
            try:
                index_data = json.loads(index_path.read_text())
                print(f"✅ Search indices loaded from {index_path}")
            except Exception as e:
                print(f"⚠️  Error loading indices: {e}")
                return False
        else:
            return False

        # Reconstruct the search index object (with backward compatibility)
        metadata = index_data.get('metadata', {})
        self.search_index.total_features = metadata.get('total_features', 0)
        self.search_index.index_version = metadata.get('version', '1.0')
        self.search_index.built_at = metadata.get('built_at', 'unknown')
        self.search_index.build_time = metadata.get('build_time', 0.0)

        # Load individual indices with proper defaultdict conversion (backward compatibility)
        indices = index_data.get('indices', index_data)  # Handle both formats
        self.search_index.parameter_index = defaultdict(list, indices.get('parameter_index', {}))
        self.search_index.counter_index = defaultdict(list, indices.get('counter_index', {}))
        self.search_index.cxc_index = indices.get('cxc_index', {})
        self.search_index.name_index = indices.get('name_index', {})
        self.search_index.name_tokens_index = defaultdict(list, indices.get('name_tokens_index', {}))
        self.search_index.dependency_index = defaultdict(list, indices.get('dependency_index', {}))
        self.search_index.reverse_dependency_index = defaultdict(list, indices.get('reverse_dependency_index', {}))
        self.search_index.category_index = defaultdict(list, indices.get('category_index', {}))
        self.search_index.value_package_index = defaultdict(list, indices.get('value_package_index', {}))
        self.search_index.node_type_index = defaultdict(list, indices.get('node_type_index', {}))
        self.search_index.fuzzy_index = defaultdict(list, indices.get('fuzzy_index', {}))

        return True

    def incremental_update(self, modified_features: List[str], deleted_features: List[str] = None) -> None:
        """Incrementally update indices for modified and deleted features"""
        if deleted_features is None:
            deleted_features = []

        print(f"🔄 Performing incremental update for {len(modified_features)} modified, {len(deleted_features)} deleted features")

        # Remove deleted features from all indices
        for feature_id in deleted_features:
            self._remove_feature_from_indices(feature_id)

        # Update modified features
        for feature_id in modified_features:
            if feature_id in self.features:
                # Remove old entries first
                self._remove_feature_from_indices(feature_id)
                # Add new entries
                self._add_feature_to_indices(feature_id, self.features[feature_id])

        # Update metadata
        self.search_index.total_features = len(self.features)
        self.search_index.built_at = time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"✅ Incremental update completed. Total features: {self.search_index.total_features}")

    def _remove_feature_from_indices(self, feature_id: str) -> None:
        """Remove a feature from all indices"""
        feature = self.features.get(feature_id)
        if not feature:
            return

        # Remove from parameter index
        for param in feature.get('parameters', []):
            param_name = param.get('name', '').lower()
            if param_name and param_name in self.search_index.parameter_index:
                if feature_id in self.search_index.parameter_index[param_name]:
                    self.search_index.parameter_index[param_name].remove(feature_id)

        # Remove from counter index
        for counter in feature.get('counters', []):
            counter_name = counter.get('name', '').lower()
            if counter_name and counter_name in self.search_index.counter_index:
                if feature_id in self.search_index.counter_index[counter_name]:
                    self.search_index.counter_index[counter_name].remove(feature_id)

        # Remove from CXC index
        cxc_code = feature.get('cxc_code')
        if cxc_code and cxc_code.upper() in self.search_index.cxc_index:
            if self.search_index.cxc_index[cxc_code.upper()] == feature_id:
                del self.search_index.cxc_index[cxc_code.upper()]

        # Remove from name indices
        name = feature.get('name', '').lower()
        if name and name in self.search_index.name_index:
            del self.search_index.name_index[name]

        # Remove from tokenized name index
        tokens = self.tokenize_name(name)
        for token in tokens:
            if token in self.search_index.name_tokens_index:
                if feature_id in self.search_index.name_tokens_index[token]:
                    self.search_index.name_tokens_index[token].remove(feature_id)

        # Remove from dependency indices
        if feature_id in self.search_index.dependency_index:
            del self.search_index.dependency_index[feature_id]
        if feature_id in self.search_index.reverse_dependency_index:
            del self.search_index.reverse_dependency_index[feature_id]

        # Remove from category indices
        value_package = feature.get('value_package', '').lower()
        if value_package and value_package in self.search_index.value_package_index:
            if feature_id in self.search_index.value_package_index[value_package]:
                self.search_index.value_package_index[value_package].remove(feature_id)

        node_type = feature.get('node_type', '').lower()
        if node_type and node_type in self.search_index.node_type_index:
            if feature_id in self.search_index.node_type_index[node_type]:
                self.search_index.node_type_index[node_type].remove(feature_id)

    def _add_feature_to_indices(self, feature_id: str, feature: Dict) -> None:
        """Add a feature to all indices"""
        # Add to parameter index
        for param in feature.get('parameters', []):
            param_name = param.get('name', '').lower()
            if param_name:
                self.search_index.parameter_index[param_name].append(feature_id)
                param_parts = re.findall(r'\b\w+\b', param_name)
                for part in param_parts:
                    if len(part) > 2:
                        self.search_index.parameter_index[part].append(feature_id)

        # Add to counter index
        for counter in feature.get('counters', []):
            counter_name = counter.get('name', '').lower()
            if counter_name:
                self.search_index.counter_index[counter_name].append(feature_id)
                counter_parts = re.findall(r'\b\w+\b', counter_name)
                for part in counter_parts:
                    if len(part) > 2:
                        self.search_index.counter_index[part].append(feature_id)

        # Add to CXC index
        cxc_code = feature.get('cxc_code')
        if cxc_code:
            self.search_index.cxc_index[cxc_code.upper()] = feature_id

        # Add to name indices
        name = feature.get('name', '').lower()
        if name:
            self.search_index.name_index[name] = feature_id
            tokens = self.tokenize_name(name)
            for token in tokens:
                self.search_index.name_tokens_index[token].append(feature_id)

        # Add to dependency indices
        dependencies = feature.get('dependencies', {})
        for prereq in dependencies.get('prerequisites', []):
            if isinstance(prereq, str):
                self.search_index.dependency_index[feature_id].append(prereq)
                self.search_index.reverse_dependency_index[prereq].append(feature_id)

        # Add to category indices
        value_package = feature.get('value_package', '').lower()
        if value_package:
            self.search_index.value_package_index[value_package].append(feature_id)

        node_type = feature.get('node_type', '').lower()
        if node_type:
            self.search_index.node_type_index[node_type].append(feature_id)

    def check_index_consistency(self) -> Dict[str, List[str]]:
        """Check index consistency and return any issues found"""
        issues = {
            'orphaned_entries': [],
            'missing_features': [],
            'duplicate_entries': []
        }

        # Check for orphaned entries (entries pointing to non-existent features)
        all_feature_ids = set(self.features.keys())

        for index_name, index_data in [
            ('parameter_index', self.search_index.parameter_index),
            ('counter_index', self.search_index.counter_index),
            ('name_tokens_index', self.search_index.name_tokens_index),
            ('dependency_index', self.search_index.dependency_index),
            ('reverse_dependency_index', self.search_index.reverse_dependency_index),
            ('category_index', self.search_index.category_index),
            ('value_package_index', self.search_index.value_package_index),
            ('node_type_index', self.search_index.node_type_index),
            ('fuzzy_index', self.search_index.fuzzy_index)
        ]:
            for key, feature_list in index_data.items():
                for feature_id in feature_list:
                    if feature_id not in all_feature_ids:
                        issues['orphaned_entries'].append(f"{index_name}.{key} -> {feature_id}")

        # Check for missing features in indices
        indexed_features = set()
        for feature_id in self.search_index.cxc_index.values():
            indexed_features.add(feature_id)
        for feature_id in self.search_index.name_index.values():
            indexed_features.add(feature_id)

        missing_features = all_feature_ids - indexed_features
        issues['missing_features'] = list(missing_features)

        return issues

    def search_parameters(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for features by parameter name"""
        results = []
        query_lower = query.lower()

        # Exact match
        if query_lower in self.search_index.parameter_index:
            for feature_id in self.search_index.parameter_index[query_lower]:
                feature = self.features.get(feature_id)
                if feature:
                    results.append(SearchResult(
                        feature_id=feature_id,
                        feature_name=feature.get('name', ''),
                        relevance_score=1.0,
                        match_type='exact',
                        match_context=f"Parameter: {query}",
                        cxc_code=feature.get('cxc_code')
                    ))

        # Partial matches
        for term, features in self.search_index.parameter_index.items():
            if query_lower in term and term != query_lower:
                for feature_id in features[:max_results]:
                    feature = self.features.get(feature_id)
                    if feature:
                        results.append(SearchResult(
                            feature_id=feature_id,
                            feature_name=feature.get('name', ''),
                            relevance_score=0.8,
                            match_type='partial',
                            match_context=f"Parameter: {term}",
                            cxc_code=feature.get('cxc_code')
                        ))

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]

    def search_counters(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for features by PM counter name"""
        results = []
        query_lower = query.lower()

        # Exact match
        if query_lower in self.search_index.counter_index:
            for feature_id in self.search_index.counter_index[query_lower]:
                feature = self.features.get(feature_id)
                if feature:
                    results.append(SearchResult(
                        feature_id=feature_id,
                        feature_name=feature.get('name', ''),
                        relevance_score=1.0,
                        match_type='exact',
                        match_context=f"Counter: {query}",
                        cxc_code=feature.get('cxc_code')
                    ))

        # Partial matches
        for term, features in self.search_index.counter_index.items():
            if query_lower in term and term != query_lower:
                for feature_id in features[:max_results]:
                    feature = self.features.get(feature_id)
                    if feature:
                        results.append(SearchResult(
                            feature_id=feature_id,
                            feature_name=feature.get('name', ''),
                            relevance_score=0.8,
                            match_type='partial',
                            match_context=f"Counter: {term}",
                            cxc_code=feature.get('cxc_code')
                        ))

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]

    def search_cxc(self, cxc_code: str) -> Optional[SearchResult]:
        """Search for feature by CXC code"""
        cxc_upper = cxc_code.upper()
        feature_id = self.search_index.cxc_index.get(cxc_upper)

        if feature_id:
            feature = self.features.get(feature_id)
            if feature:
                return SearchResult(
                    feature_id=feature_id,
                    feature_name=feature.get('name', ''),
                    relevance_score=1.0,
                    match_type='exact',
                    match_context=f"CXC Code: {cxc_upper}",
                    cxc_code=cxc_upper
                )

        return None

    def search_names(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Search for features by name with tokenization"""
        results = []
        query_lower = query.lower()

        # Exact name match
        if query_lower in self.search_index.name_index:
            feature_id = self.search_index.name_index[query_lower]
            feature = self.features.get(feature_id)
            if feature:
                results.append(SearchResult(
                    feature_id=feature_id,
                    feature_name=feature.get('name', ''),
                    relevance_score=1.0,
                    match_type='exact',
                    match_context=f"Name: {query}",
                    cxc_code=feature.get('cxc_code')
                ))

        # Token matches
        query_tokens = self.tokenize_name(query)
        for token in query_tokens:
            if token in self.search_index.name_tokens_index:
                for feature_id in self.search_index.name_tokens_index[token]:
                    feature = self.features.get(feature_id)
                    if feature:
                        # Calculate relevance based on token match count
                        match_count = sum(1 for t in query_tokens if t in self.search_index.name_tokens_index.get(feature_id, []))
                        relevance = match_count / len(query_tokens)

                        if relevance > 0.3:  # Only include meaningful matches
                            results.append(SearchResult(
                                feature_id=feature_id,
                                feature_name=feature.get('name', ''),
                                relevance_score=relevance,
                                match_type='partial',
                                match_context=f"Name token: {token}",
                                cxc_code=feature.get('cxc_code')
                            ))

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]

    def search_dependencies(self, feature_id: str) -> List[SearchResult]:
        """Find features that depend on or are related to the given feature"""
        results = []

        # Features that depend on this one
        dependents = self.search_index.reverse_dependency_index.get(feature_id, [])
        for dependent_id in dependents:
            feature = self.features.get(dependent_id)
            if feature:
                results.append(SearchResult(
                    feature_id=dependent_id,
                    feature_name=feature.get('name', ''),
                    relevance_score=0.9,
                    match_type='dependency',
                    match_context=f"Dependent feature",
                    cxc_code=feature.get('cxc_code')
                ))

        # Features this one depends on
        dependencies = self.search_index.dependency_index.get(feature_id, [])
        for dep_id in dependencies:
            feature = self.features.get(dep_id)
            if feature:
                results.append(SearchResult(
                    feature_id=dep_id,
                    feature_name=feature.get('name', ''),
                    relevance_score=0.8,
                    match_type='dependency',
                    match_context=f"Prerequisite feature",
                    cxc_code=feature.get('cxc_code')
                ))

        return results

    def fuzzy_search(self, query: str, max_results: int = 10) -> List[SearchResult]:
        """Perform fuzzy search across all indices"""
        results = []
        query_lower = query.lower()

        # Check fuzzy index
        if query_lower in self.search_index.fuzzy_index:
            for feature_id in self.search_index.fuzzy_index[query_lower]:
                feature = self.features.get(feature_id)
                if feature:
                    results.append(SearchResult(
                        feature_id=feature_id,
                        feature_name=feature.get('name', ''),
                        relevance_score=0.7,
                        match_type='fuzzy',
                        match_context=f"Fuzzy match: {query}",
                        cxc_code=feature.get('cxc_code')
                    ))

        # If no fuzzy results, try finding similar terms
        if not results:
            all_terms = list(self.search_index.name_index.keys()) + \
                       list(self.search_index.parameter_index.keys()) + \
                       list(self.search_index.counter_index.keys())

            similar_terms = difflib.get_close_matches(
                query_lower, all_terms, n=5, cutoff=self.fuzzy_threshold
            )

            for similar_term in similar_terms:
                # Search in different indices with the similar term
                if similar_term in self.search_index.name_index:
                    feature_id = self.search_index.name_index[similar_term]
                    feature = self.features.get(feature_id)
                    if feature:
                        results.append(SearchResult(
                            feature_id=feature_id,
                            feature_name=feature.get('name', ''),
                            relevance_score=0.6,
                            match_type='fuzzy',
                            match_context=f"Similar to: {similar_term}",
                            cxc_code=feature.get('cxc_code')
                        ))

        return sorted(results, key=lambda x: x.relevance_score, reverse=True)[:max_results]

    def universal_search(self, query: str, max_results: int = 20) -> List[SearchResult]:
        """Universal search across all indices"""
        all_results = []

        # Try different search methods
        search_methods = [
            ("CXC", lambda q: [self.search_cxc(q)] if self.search_cxc(q) else []),
            ("Name", lambda q: self.search_names(q, max_results)),
            ("Parameter", lambda q: self.search_parameters(q, max_results)),
            ("Counter", lambda q: self.search_counters(q, max_results)),
            ("Fuzzy", lambda q: self.fuzzy_search(q, max_results))
        ]

        for method_name, search_func in search_methods:
            try:
                results = search_func(query)
                for result in results:
                    result.match_context = f"{method_name}: {result.match_context}"
                all_results.extend(results)
            except Exception as e:
                print(f"⚠️  Error in {method_name} search: {e}")

        # Remove duplicates and sort by relevance
        seen = set()
        unique_results = []
        for result in all_results:
            if result.feature_id not in seen:
                seen.add(result.feature_id)
                unique_results.append(result)

        return sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)[:max_results]

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the search indices"""
        stats = {
            'total_features': self.search_index.total_features,
            'index_version': self.search_index.index_version,
            'built_at': self.search_index.built_at,
            'build_time': self.search_index.build_time,
            'indices': {
                'parameter_index': len(self.search_index.parameter_index),
                'counter_index': len(self.search_index.counter_index),
                'cxc_index': len(self.search_index.cxc_index),
                'name_index': len(self.search_index.name_index),
                'name_tokens_index': len(self.search_index.name_tokens_index),
                'dependency_index': len(self.search_index.dependency_index),
                'reverse_dependency_index': len(self.search_index.reverse_dependency_index),
                'category_index': len(self.search_index.category_index),
                'value_package_index': len(self.search_index.value_package_index),
                'node_type_index': len(self.search_index.node_type_index),
                'fuzzy_index': len(self.search_index.fuzzy_index)
            },
            'categories': {
                category: len(features)
                for category, features in self.search_index.category_index.items()
            },
            'value_packages': {
                vp: len(features)
                for vp, features in self.search_index.value_package_index.items()
            },
            'node_types': {
                nt: len(features)
                for nt, features in self.search_index.node_type_index.items()
            }
        }

        return stats

    def export_index_summary(self, output_file: Optional[str] = None) -> None:
        """Export a human-readable summary of the search indices"""
        if output_file is None:
            output_file = self.output_dir / "ericsson_data" / "search_index_summary.md"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.get_index_statistics()

        summary = f"""# Ericsson Features Search Index Summary

## Index Overview
- **Total Features**: {stats['total_features']:,}
- **Index Version**: {stats['index_version']}
- **Built At**: {stats['built_at']}
- **Build Time**: {stats['build_time']:.2f} seconds

## Index Statistics
| Index Type | Entries |
|-----------|---------|
"""

        for index_name, count in stats['indices'].items():
            summary += f"| {index_name.replace('_', ' ').title()} | {count:,} |\n"

        summary += f"""
## Feature Categories
| Category | Feature Count |
|----------|---------------|
"""

        for category, count in stats['categories'].items():
            summary += f"| {category.title()} | {count:,} |\n"

        summary += f"""
## Value Packages
| Value Package | Feature Count |
|---------------|---------------|
"""

        for vp, count in sorted(stats['value_packages'].items()):
            summary += f"| {vp} | {count:,} |\n"

        summary += f"""
## Node Types
| Node Type | Feature Count |
|-----------|---------------|
"""

        for nt, count in sorted(stats['node_types'].items()):
            summary += f"| {nt} | {count:,} |\n"

        output_path.write_text(summary)
        print(f"📊 Index summary saved to {output_path}")


def main():
    """Main function for building search indices"""
    import argparse

    parser = argparse.ArgumentParser(description="Build search indices for Ericsson features")
    parser.add_argument("--features-dir", required=True, help="Directory containing processed feature JSON files")
    parser.add_argument("--output-dir", default="output", help="Output directory for indices")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild indices even if they exist")
    parser.add_argument("--test-search", help="Test search with a query after building indices")
    parser.add_argument("--export-summary", action="store_true", help="Export index summary to markdown")

    args = parser.parse_args()

    # Initialize index builder
    builder = EricssonSearchIndexBuilder(args.features_dir, args.output_dir)

    # Load features
    try:
        builder.load_features()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure you have processed features using ericsson_feature_processor.py first")
        return 1

    # Try to load existing indices
    if not args.force_rebuild and builder.load_indices():
        print("📚 Using existing search indices")
    else:
        # Build new indices
        builder.build_all_indices()
        builder.save_indices()

    # Export summary if requested
    if args.export_summary:
        builder.export_index_summary()

    # Print statistics
    stats = builder.get_index_statistics()
    print(f"\n📈 Search Index Statistics:")
    print(f"   Features indexed: {stats['total_features']:,}")
    print(f"   Parameter entries: {stats['indices']['parameter_index']:,}")
    print(f"   Counter entries: {stats['indices']['counter_index']:,}")
    print(f"   CXC codes: {stats['indices']['cxc_index']:,}")
    print(f"   Name tokens: {stats['indices']['name_tokens_index']:,}")
    print(f"   Categories: {len(stats['categories'])}")
    print(f"   Value packages: {len(stats['value_packages'])}")

    # Test search if requested
    if args.test_search:
        print(f"\n🔍 Testing search with query: '{args.test_search}'")
        results = builder.universal_search(args.test_search)

        if results:
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results[:10], 1):
                print(f"  {i}. {result.feature_name} ({result.feature_id}) "
                      f"[{result.match_type}] - {result.match_context}")
        else:
            print("No results found")

    return 0


if __name__ == "__main__":
    sys.exit(main())