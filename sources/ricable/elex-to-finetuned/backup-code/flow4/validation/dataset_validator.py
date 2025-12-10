"""
Advanced dataset validation for Flow4 generated datasets.

This module provides comprehensive validation for:
- MLX fine-tuning datasets
- Augmentoolkit-generated datasets
- RAG training datasets
- Dataset quality metrics and statistics
"""

import os
import json
import statistics
import re
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetValidationResult:
    """Result of dataset validation."""
    dataset_name: str
    dataset_type: str
    passed: bool
    score: float
    total_samples: int
    issues: List[str]
    statistics: Dict[str, Any]
    recommendations: List[str]


class DatasetValidator:
    """Comprehensive validator for Flow4 generated datasets."""

    def __init__(self, datasets_dir: str = "pipeline_output"):
        """Initialize dataset validator.
        
        Args:
            datasets_dir: Directory containing generated datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.validation_results = []
        
        # Quality thresholds
        self.min_samples = 50
        self.min_avg_length = 100
        self.max_avg_length = 8000
        self.max_duplicate_ratio = 0.05
        self.min_diversity_score = 0.7

    def find_datasets(self) -> Dict[str, List[Path]]:
        """Find all dataset files organized by type.
        
        Returns:
            Dictionary mapping dataset types to file paths
        """
        datasets = {
            'mlx': [],
            'augmentoolkit': [],
            'rag': [],
            'finetune': [],
            'other': []
        }
        
        # Search for dataset files
        for pattern in ['*.json', '*.jsonl']:
            for file_path in self.datasets_dir.glob(f"**/{pattern}"):
                file_name = file_path.name.lower()
                
                if 'mlx' in file_name:
                    datasets['mlx'].append(file_path)
                elif any(term in file_name for term in ['augmentoolkit', 'factual', 'rag_data']):
                    datasets['augmentoolkit'].append(file_path)
                elif 'rag' in file_name:
                    datasets['rag'].append(file_path)
                elif any(term in file_name for term in ['finetune', 'train', 'valid', 'test']):
                    datasets['finetune'].append(file_path)
                else:
                    datasets['other'].append(file_path)
        
        # Remove empty categories
        datasets = {k: v for k, v in datasets.items() if v}
        
        logger.info(f"Found datasets: {[(k, len(v)) for k, v in datasets.items()]}")
        return datasets

    def validate_dataset_format(self, file_path: Path) -> Tuple[bool, List[str], List[Dict]]:
        """Validate dataset file format and load data.
        
        Args:
            file_path: Path to dataset file
            
        Returns:
            Tuple of (is_valid, errors, data)
        """
        errors = []
        data = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # JSONL format
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            data.append(entry)
                        except json.JSONDecodeError as e:
                            errors.append(f"Line {line_num}: Invalid JSON - {str(e)}")
                else:
                    # JSON format
                    content = json.load(f)
                    if isinstance(content, list):
                        data = content
                    elif isinstance(content, dict):
                        if 'data' in content:
                            data = content['data']
                        elif 'training_data' in content:
                            data = content['training_data']
                        elif 'conversations' in content:
                            data = content['conversations']
                        else:
                            data = [content]
                    else:
                        errors.append("Invalid JSON structure: expected list or dict")
        
        except Exception as e:
            errors.append(f"File reading error: {str(e)}")
        
        is_valid = len(errors) == 0 and len(data) > 0
        return is_valid, errors, data

    def analyze_dataset_content(self, data: List[Dict], dataset_name: str) -> Dict[str, Any]:
        """Analyze dataset content for quality metrics.
        
        Args:
            data: List of dataset entries
            dataset_name: Name of the dataset
            
        Returns:
            Analysis statistics
        """
        stats = {
            'total_samples': len(data),
            'format_analysis': {},
            'content_quality': {},
            'diversity_metrics': {},
            'issues': []
        }
        
        if not data:
            stats['issues'].append("Dataset is empty")
            return stats
        
        # Analyze format types
        format_types = self._analyze_format_types(data)
        stats['format_analysis'] = format_types
        
        # Analyze content quality
        content_metrics = self._analyze_content_quality(data)
        stats['content_quality'] = content_metrics
        
        # Analyze diversity
        diversity_metrics = self._analyze_content_diversity(data)
        stats['diversity_metrics'] = diversity_metrics
        
        # Detect issues
        issues = self._detect_content_issues(data, stats)
        stats['issues'].extend(issues)
        
        return stats

    def _analyze_format_types(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze dataset format types."""
        format_counts = Counter()
        field_usage = Counter()
        
        for entry in data:
            # Determine format type
            if isinstance(entry, dict):
                if 'messages' in entry:
                    format_counts['chat_format'] += 1
                elif 'instruction' in entry and 'input' in entry and 'output' in entry:
                    format_counts['instruction_format'] += 1
                elif 'text' in entry:
                    format_counts['text_format'] += 1
                elif 'prompt' in entry and 'completion' in entry:
                    format_counts['prompt_completion'] += 1
                else:
                    format_counts['unknown'] += 1
                
                # Track field usage
                for field in entry.keys():
                    field_usage[field] += 1
        
        return {
            'format_distribution': dict(format_counts),
            'field_usage': dict(field_usage.most_common(10)),
            'dominant_format': format_counts.most_common(1)[0][0] if format_counts else 'unknown'
        }

    def _analyze_content_quality(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze content quality metrics."""
        text_lengths = []
        word_counts = []
        empty_count = 0
        very_short_count = 0
        very_long_count = 0
        
        for entry in data:
            text_content = self._extract_text_content(entry)
            
            if not text_content.strip():
                empty_count += 1
                continue
            
            length = len(text_content)
            word_count = len(text_content.split())
            
            text_lengths.append(length)
            word_counts.append(word_count)
            
            if length < 20:
                very_short_count += 1
            elif length > 8000:
                very_long_count += 1
        
        if not text_lengths:
            return {
                'empty_samples': empty_count,
                'avg_length': 0,
                'avg_words': 0,
                'quality_issues': ['All samples are empty']
            }
        
        return {
            'avg_length': statistics.mean(text_lengths),
            'median_length': statistics.median(text_lengths),
            'min_length': min(text_lengths),
            'max_length': max(text_lengths),
            'std_length': statistics.stdev(text_lengths) if len(text_lengths) > 1 else 0,
            'avg_words': statistics.mean(word_counts),
            'empty_samples': empty_count,
            'very_short_samples': very_short_count,
            'very_long_samples': very_long_count,
            'length_distribution': {
                'short (<100)': sum(1 for l in text_lengths if l < 100),
                'medium (100-1000)': sum(1 for l in text_lengths if 100 <= l <= 1000),
                'long (1000-4000)': sum(1 for l in text_lengths if 1000 < l <= 4000),
                'very_long (>4000)': sum(1 for l in text_lengths if l > 4000)
            }
        }

    def _analyze_content_diversity(self, data: List[Dict]) -> Dict[str, Any]:
        """Analyze content diversity and uniqueness."""
        text_hashes = set()
        duplicate_count = 0
        unique_trigrams = set()
        repeated_patterns = Counter()
        
        for entry in data:
            text_content = self._extract_text_content(entry)
            
            # Check for duplicates
            text_hash = hashlib.md5(text_content.encode()).hexdigest()
            if text_hash in text_hashes:
                duplicate_count += 1
            else:
                text_hashes.add(text_hash)
            
            # Analyze n-grams for diversity
            words = text_content.lower().split()
            if len(words) >= 3:
                for i in range(len(words) - 2):
                    trigram = ' '.join(words[i:i+3])
                    unique_trigrams.add(trigram)
                    repeated_patterns[trigram] += 1
        
        # Calculate diversity metrics
        total_samples = len(data)
        duplicate_ratio = duplicate_count / total_samples if total_samples > 0 else 0
        
        # Find highly repeated patterns
        highly_repeated = {pattern: count for pattern, count in repeated_patterns.items() 
                          if count > max(10, total_samples * 0.1)}
        
        return {
            'duplicate_count': duplicate_count,
            'duplicate_ratio': duplicate_ratio,
            'unique_trigrams': len(unique_trigrams),
            'diversity_score': 1 - duplicate_ratio,
            'highly_repeated_patterns': dict(list(highly_repeated.items())[:5]),
            'pattern_diversity': len(unique_trigrams) / max(total_samples, 1)
        }

    def _detect_content_issues(self, data: List[Dict], stats: Dict[str, Any]) -> List[str]:
        """Detect content issues and quality problems."""
        issues = []
        
        # Check sample count
        if stats['total_samples'] < self.min_samples:
            issues.append(f"Dataset too small: {stats['total_samples']} samples (minimum: {self.min_samples})")
        
        # Check content quality
        content_quality = stats.get('content_quality', {})
        avg_length = content_quality.get('avg_length', 0)
        
        if avg_length < self.min_avg_length:
            issues.append(f"Average text length too short: {avg_length:.0f} chars (minimum: {self.min_avg_length})")
        elif avg_length > self.max_avg_length:
            issues.append(f"Average text length too long: {avg_length:.0f} chars (maximum: {self.max_avg_length})")
        
        # Check for empty samples
        empty_samples = content_quality.get('empty_samples', 0)
        if empty_samples > 0:
            issues.append(f"Found {empty_samples} empty samples")
        
        # Check for duplicates
        diversity_metrics = stats.get('diversity_metrics', {})
        duplicate_ratio = diversity_metrics.get('duplicate_ratio', 0)
        
        if duplicate_ratio > self.max_duplicate_ratio:
            issues.append(f"High duplicate ratio: {duplicate_ratio:.1%} (maximum: {self.max_duplicate_ratio:.1%})")
        
        # Check diversity
        diversity_score = diversity_metrics.get('diversity_score', 1)
        if diversity_score < self.min_diversity_score:
            issues.append(f"Low content diversity: {diversity_score:.2f} (minimum: {self.min_diversity_score})")
        
        # Check format consistency
        format_analysis = stats.get('format_analysis', {})
        format_dist = format_analysis.get('format_distribution', {})
        
        if len(format_dist) > 1:
            total = sum(format_dist.values())
            dominant_ratio = max(format_dist.values()) / total if total > 0 else 0
            if dominant_ratio < 0.9:
                issues.append("Mixed format types detected - may cause training issues")
        
        # Check for training artifacts
        artifacts = self._detect_training_artifacts(data)
        if artifacts:
            issues.append(f"Training artifacts detected: {', '.join(artifacts)}")
        
        return issues

    def _detect_training_artifacts(self, data: List[Dict]) -> List[str]:
        """Detect potential training artifacts in the data."""
        artifacts = []
        
        # Common artifact patterns
        artifact_patterns = [
            r'document\.md',
            r'combined_document',
            r'radioequipmentclockreference',
            r'column \d+ =',
            r'--- =',
            r'chunk_\d+',
            r'source: .+\.html'
        ]
        
        artifact_counts = Counter()
        
        for entry in data:
            text_content = self._extract_text_content(entry)
            
            for pattern in artifact_patterns:
                matches = re.findall(pattern, text_content, re.IGNORECASE)
                if matches:
                    artifact_counts[pattern] += len(matches)
        
        # Report significant artifact presence
        total_samples = len(data)
        for pattern, count in artifact_counts.items():
            if count > max(5, total_samples * 0.1):
                artifacts.append(f"{pattern} ({count} instances)")
        
        return artifacts

    def _extract_text_content(self, entry: Dict) -> str:
        """Extract text content from dataset entry regardless of format."""
        if 'text' in entry:
            return str(entry['text'])
        elif 'messages' in entry:
            # Chat format
            messages = entry['messages']
            if isinstance(messages, list):
                return ' '.join(msg.get('content', '') for msg in messages if isinstance(msg, dict))
        elif 'instruction' in entry:
            # Instruction format
            parts = []
            for field in ['instruction', 'input', 'output']:
                if field in entry and entry[field]:
                    parts.append(str(entry[field]))
            return ' '.join(parts)
        elif 'prompt' in entry and 'completion' in entry:
            # Prompt-completion format
            return f"{entry.get('prompt', '')} {entry.get('completion', '')}"
        else:
            # Try to extract any text-like fields
            text_parts = []
            for key, value in entry.items():
                if isinstance(value, str) and len(value) > 10:
                    text_parts.append(value)
            return ' '.join(text_parts)
        
        return ""

    def validate_mlx_dataset(self, file_path: Path) -> DatasetValidationResult:
        """Validate MLX fine-tuning dataset."""
        logger.info(f"Validating MLX dataset: {file_path.name}")
        
        is_valid, errors, data = self.validate_dataset_format(file_path)
        
        if not is_valid:
            return DatasetValidationResult(
                dataset_name=file_path.name,
                dataset_type='mlx',
                passed=False,
                score=0.0,
                total_samples=0,
                issues=errors,
                statistics={},
                recommendations=["Fix format errors before proceeding"]
            )
        
        # Analyze content
        stats = self.analyze_dataset_content(data, file_path.name)
        
        # MLX-specific validations
        mlx_issues = self._validate_mlx_specific(data)
        stats['issues'].extend(mlx_issues)
        
        # Calculate quality score
        score = self._calculate_quality_score(stats)
        
        # Generate recommendations
        recommendations = self._generate_mlx_recommendations(stats)
        
        return DatasetValidationResult(
            dataset_name=file_path.name,
            dataset_type='mlx',
            passed=len(stats['issues']) == 0,
            score=score,
            total_samples=stats['total_samples'],
            issues=stats['issues'],
            statistics=stats,
            recommendations=recommendations
        )

    def _validate_mlx_specific(self, data: List[Dict]) -> List[str]:
        """MLX-specific validation checks."""
        issues = []
        
        # Check for required MLX format
        chat_format_count = 0
        text_format_count = 0
        
        for entry in data:
            if 'messages' in entry:
                chat_format_count += 1
                # Validate message structure
                messages = entry['messages']
                if not isinstance(messages, list):
                    issues.append("Invalid messages format: must be a list")
                    continue
                
                for msg in messages:
                    if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                        issues.append("Invalid message structure: missing role or content")
                        break
            elif 'text' in entry:
                text_format_count += 1
        
        # MLX prefers consistent format
        if chat_format_count > 0 and text_format_count > 0:
            issues.append("Mixed chat and text formats - MLX works best with consistent format")
        
        # Check for reasonable dataset size for MLX
        if len(data) < 100:
            issues.append("Dataset may be too small for effective MLX fine-tuning (recommended: 100+ samples)")
        elif len(data) > 10000:
            issues.append("Very large dataset - consider using a subset for initial MLX fine-tuning")
        
        return issues

    def validate_augmentoolkit_dataset(self, file_path: Path) -> DatasetValidationResult:
        """Validate Augmentoolkit-generated dataset."""
        logger.info(f"Validating Augmentoolkit dataset: {file_path.name}")
        
        is_valid, errors, data = self.validate_dataset_format(file_path)
        
        if not is_valid:
            return DatasetValidationResult(
                dataset_name=file_path.name,
                dataset_type='augmentoolkit',
                passed=False,
                score=0.0,
                total_samples=0,
                issues=errors,
                statistics={},
                recommendations=["Fix format errors before proceeding"]
            )
        
        # Analyze content
        stats = self.analyze_dataset_content(data, file_path.name)
        
        # Augmentoolkit-specific validations
        atk_issues = self._validate_augmentoolkit_specific(data, file_path.name)
        stats['issues'].extend(atk_issues)
        
        # Calculate quality score
        score = self._calculate_quality_score(stats)
        
        # Generate recommendations
        recommendations = self._generate_augmentoolkit_recommendations(stats)
        
        return DatasetValidationResult(
            dataset_name=file_path.name,
            dataset_type='augmentoolkit',
            passed=len(stats['issues']) == 0,
            score=score,
            total_samples=stats['total_samples'],
            issues=stats['issues'],
            statistics=stats,
            recommendations=recommendations
        )

    def _validate_augmentoolkit_specific(self, data: List[Dict], dataset_name: str) -> List[str]:
        """Augmentoolkit-specific validation checks."""
        issues = []
        
        # Check for Augmentoolkit format patterns
        if 'factual' in dataset_name.lower():
            # Validate factual QA format
            qa_pairs = 0
            for entry in data:
                if isinstance(entry, dict):
                    if 'question' in entry and 'answer' in entry:
                        qa_pairs += 1
                    elif 'messages' in entry:
                        messages = entry['messages']
                        if len(messages) >= 2:
                            qa_pairs += 1
            
            if qa_pairs < len(data) * 0.8:
                issues.append("Dataset doesn't appear to be proper factual QA format")
        
        elif 'rag' in dataset_name.lower():
            # Validate RAG training format
            rag_examples = 0
            for entry in data:
                text_content = self._extract_text_content(entry)
                if 'context' in text_content.lower() or 'retrieved' in text_content.lower():
                    rag_examples += 1
            
            if rag_examples < len(data) * 0.3:
                issues.append("Dataset doesn't appear to contain RAG training examples")
        
        # Check for Augmentoolkit quality markers
        high_quality_count = 0
        for entry in data:
            text_content = self._extract_text_content(entry)
            
            # Look for detailed, informative content
            if len(text_content) > 200 and any(marker in text_content.lower() for marker in [
                'specifically', 'according to', 'based on', 'the document', 'information'
            ]):
                high_quality_count += 1
        
        quality_ratio = high_quality_count / len(data) if data else 0
        if quality_ratio < 0.6:
            issues.append(f"Low proportion of high-quality generated content: {quality_ratio:.1%}")
        
        return issues

    def _calculate_quality_score(self, stats: Dict[str, Any]) -> float:
        """Calculate overall quality score for dataset."""
        score_components = []
        
        # Sample count score
        sample_count = stats['total_samples']
        if sample_count >= self.min_samples:
            count_score = min(1.0, sample_count / (self.min_samples * 2))
        else:
            count_score = sample_count / self.min_samples
        score_components.append(count_score)
        
        # Content quality score
        content_quality = stats.get('content_quality', {})
        avg_length = content_quality.get('avg_length', 0)
        
        if self.min_avg_length <= avg_length <= self.max_avg_length:
            length_score = 1.0
        elif avg_length < self.min_avg_length:
            length_score = avg_length / self.min_avg_length
        else:
            length_score = max(0.5, self.max_avg_length / avg_length)
        score_components.append(length_score)
        
        # Diversity score
        diversity_metrics = stats.get('diversity_metrics', {})
        diversity_score = diversity_metrics.get('diversity_score', 0)
        score_components.append(diversity_score)
        
        # Issue penalty
        issue_count = len(stats.get('issues', []))
        issue_penalty = max(0, 1 - (issue_count * 0.1))
        score_components.append(issue_penalty)
        
        # Calculate weighted average
        weights = [0.2, 0.3, 0.3, 0.2]  # count, length, diversity, issues
        final_score = sum(score * weight for score, weight in zip(score_components, weights))
        
        return min(1.0, max(0.0, final_score))

    def _generate_mlx_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate MLX-specific recommendations."""
        recommendations = []
        
        # Sample count recommendations
        total_samples = stats['total_samples']
        if total_samples < 100:
            recommendations.append("Increase dataset size to at least 100 samples for better MLX fine-tuning")
        elif total_samples < 500:
            recommendations.append("Consider generating more data for robust fine-tuning (500+ samples recommended)")
        
        # Format recommendations
        format_analysis = stats.get('format_analysis', {})
        dominant_format = format_analysis.get('dominant_format', 'unknown')
        
        if dominant_format == 'unknown':
            recommendations.append("Ensure consistent dataset format (chat or text) for MLX compatibility")
        
        # Content quality recommendations
        content_quality = stats.get('content_quality', {})
        avg_length = content_quality.get('avg_length', 0)
        
        if avg_length < 100:
            recommendations.append("Increase average sample length for more informative training")
        elif avg_length > 4000:
            recommendations.append("Consider chunking very long samples for better MLX performance")
        
        # Diversity recommendations
        diversity_metrics = stats.get('diversity_metrics', {})
        duplicate_ratio = diversity_metrics.get('duplicate_ratio', 0)
        
        if duplicate_ratio > 0.05:
            recommendations.append("Remove duplicate samples to improve training efficiency")
        
        return recommendations

    def _generate_augmentoolkit_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate Augmentoolkit-specific recommendations."""
        recommendations = []
        
        # Quality recommendations
        content_quality = stats.get('content_quality', {})
        avg_length = content_quality.get('avg_length', 0)
        
        if avg_length < 200:
            recommendations.append("Generated content appears short - consider regenerating with more detailed prompts")
        
        # Diversity recommendations
        diversity_metrics = stats.get('diversity_metrics', {})
        pattern_diversity = diversity_metrics.get('pattern_diversity', 0)
        
        if pattern_diversity < 0.5:
            recommendations.append("Low pattern diversity - consider using more varied generation templates")
        
        # Format recommendations
        format_analysis = stats.get('format_analysis', {})
        format_dist = format_analysis.get('format_distribution', {})
        
        if 'instruction_format' in format_dist:
            recommendations.append("Dataset suitable for instruction fine-tuning")
        elif 'chat_format' in format_dist:
            recommendations.append("Dataset suitable for conversational AI training")
        
        return recommendations

    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation on all found datasets.
        
        Returns:
            Comprehensive validation report
        """
        logger.info("Starting comprehensive dataset validation...")
        
        # Find all datasets
        datasets = self.find_datasets()
        
        if not any(datasets.values()):
            return {
                'status': 'error',
                'message': 'No datasets found for validation',
                'timestamp': datetime.now().isoformat()
            }
        
        # Validate each dataset
        validation_results = []
        
        for dataset_type, file_paths in datasets.items():
            for file_path in file_paths:
                if dataset_type == 'mlx':
                    result = self.validate_mlx_dataset(file_path)
                elif dataset_type == 'augmentoolkit':
                    result = self.validate_augmentoolkit_dataset(file_path)
                else:
                    # Generic validation
                    result = self.validate_generic_dataset(file_path, dataset_type)
                
                validation_results.append(result)
        
        # Generate summary
        summary = self._generate_validation_summary(validation_results)
        
        return {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'validation_results': [self._serialize_result(r) for r in validation_results],
            'summary': summary,
            'datasets_found': {k: len(v) for k, v in datasets.items()},
            'total_datasets': len(validation_results)
        }

    def validate_generic_dataset(self, file_path: Path, dataset_type: str) -> DatasetValidationResult:
        """Validate generic dataset file."""
        logger.info(f"Validating {dataset_type} dataset: {file_path.name}")
        
        is_valid, errors, data = self.validate_dataset_format(file_path)
        
        if not is_valid:
            return DatasetValidationResult(
                dataset_name=file_path.name,
                dataset_type=dataset_type,
                passed=False,
                score=0.0,
                total_samples=0,
                issues=errors,
                statistics={},
                recommendations=["Fix format errors before proceeding"]
            )
        
        # Analyze content
        stats = self.analyze_dataset_content(data, file_path.name)
        
        # Calculate quality score
        score = self._calculate_quality_score(stats)
        
        # Generate basic recommendations
        recommendations = self._generate_basic_recommendations(stats)
        
        return DatasetValidationResult(
            dataset_name=file_path.name,
            dataset_type=dataset_type,
            passed=len(stats['issues']) == 0,
            score=score,
            total_samples=stats['total_samples'],
            issues=stats['issues'],
            statistics=stats,
            recommendations=recommendations
        )

    def _generate_basic_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate basic recommendations for any dataset."""
        recommendations = []
        
        total_samples = stats['total_samples']
        if total_samples < 50:
            recommendations.append("Dataset is very small - consider generating more data")
        
        content_quality = stats.get('content_quality', {})
        empty_samples = content_quality.get('empty_samples', 0)
        if empty_samples > 0:
            recommendations.append(f"Remove {empty_samples} empty samples")
        
        diversity_metrics = stats.get('diversity_metrics', {})
        duplicate_ratio = diversity_metrics.get('duplicate_ratio', 0)
        if duplicate_ratio > 0.1:
            recommendations.append("High duplication detected - consider deduplication")
        
        return recommendations

    def _generate_validation_summary(self, results: List[DatasetValidationResult]) -> Dict[str, Any]:
        """Generate summary of all validation results."""
        if not results:
            return {}
        
        passed_count = sum(1 for r in results if r.passed)
        total_samples = sum(r.total_samples for r in results)
        avg_score = statistics.mean(r.score for r in results)
        
        # Group by type
        by_type = defaultdict(list)
        for result in results:
            by_type[result.dataset_type].append(result)
        
        type_summaries = {}
        for dataset_type, type_results in by_type.items():
            type_summaries[dataset_type] = {
                'count': len(type_results),
                'passed': sum(1 for r in type_results if r.passed),
                'avg_score': statistics.mean(r.score for r in type_results),
                'total_samples': sum(r.total_samples for r in type_results)
            }
        
        return {
            'total_datasets': len(results),
            'passed_validation': passed_count,
            'overall_pass_rate': passed_count / len(results),
            'total_samples': total_samples,
            'average_quality_score': avg_score,
            'by_type': type_summaries,
            'quality_grades': {
                'excellent (>0.9)': sum(1 for r in results if r.score > 0.9),
                'good (0.7-0.9)': sum(1 for r in results if 0.7 <= r.score <= 0.9),
                'fair (0.5-0.7)': sum(1 for r in results if 0.5 <= r.score < 0.7),
                'poor (<0.5)': sum(1 for r in results if r.score < 0.5)
            }
        }

    def _serialize_result(self, result: DatasetValidationResult) -> Dict[str, Any]:
        """Serialize validation result for JSON output."""
        return {
            'dataset_name': result.dataset_name,
            'dataset_type': result.dataset_type,
            'passed': result.passed,
            'score': result.score,
            'total_samples': result.total_samples,
            'issues': result.issues,
            'statistics': result.statistics,
            'recommendations': result.recommendations
        }

    def save_validation_report(self, report: Dict[str, Any], output_file: str):
        """Save validation report to file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset validation report saved to {output_file}")


def main():
    """Main function for running dataset validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Flow4 generated datasets")
    parser.add_argument("--datasets-dir", default="pipeline_output", 
                       help="Directory containing datasets to validate")
    parser.add_argument("--output", default="dataset_validation_report.json",
                       help="Output file for validation report")
    
    args = parser.parse_args()
    
    validator = DatasetValidator(args.datasets_dir)
    report = validator.run_comprehensive_validation()
    
    # Print summary to console
    if report['status'] == 'success':
        summary = report['summary']
        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        print(f"Datasets validated: {report['total_datasets']}")
        print(f"Passed validation: {summary['passed_validation']}/{summary['total_datasets']}")
        print(f"Overall pass rate: {summary['overall_pass_rate']:.1%}")
        print(f"Average quality score: {summary['average_quality_score']:.2f}")
        print(f"Total samples: {summary['total_samples']:,}")
        
        print(f"\nQuality Distribution:")
        for grade, count in summary['quality_grades'].items():
            print(f"  {grade}: {count} datasets")
        
        print(f"\nBy Dataset Type:")
        for dtype, stats in summary['by_type'].items():
            print(f"  {dtype}: {stats['count']} datasets, {stats['passed']} passed")
    else:
        print(f"Validation failed: {report['message']}")
    
    validator.save_validation_report(report, args.output)
    print(f"\nDetailed report saved to: {args.output}")


if __name__ == "__main__":
    main()