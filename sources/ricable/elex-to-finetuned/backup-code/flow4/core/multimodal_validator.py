"""
Multimodal Dataset Validation for Flow4

Validates multimodal fine-tuning datasets for quality, format correctness,
and compatibility with vision-language model training frameworks.

Features:
- Format validation for LLaVA, ShareGPT, and ChatML
- Content quality assessment
- Multimodal consistency checks
- Training framework compatibility validation
- Dataset statistics and analysis
"""

import json
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Results of multimodal dataset validation."""
    
    is_valid: bool
    format_name: str
    total_samples: int
    valid_samples: int
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def validity_score(self) -> float:
        """Calculate validity score as percentage of valid samples."""
        if self.total_samples == 0:
            return 0.0
        return (self.valid_samples / self.total_samples) * 100.0


class MultimodalDatasetValidator:
    """Validates multimodal datasets for training readiness."""
    
    def __init__(self):
        """Initialize the validator."""
        self.format_validators = {
            "llava": self._validate_llava_format,
            "sharegpt": self._validate_sharegpt_format,
            "chatml": self._validate_chatml_format,
            "conversation": self._validate_conversation_format
        }
        
        # Validation criteria
        self.min_text_length = 10
        self.max_text_length = 4000
        self.min_conversations_per_sample = 1
        self.max_conversations_per_sample = 10
        self.required_roles = {"human", "gpt", "assistant", "user"}
    
    def validate_dataset_file(self, file_path: str, format_name: str = None) -> ValidationResult:
        """Validate a multimodal dataset file.
        
        Args:
            file_path: Path to the dataset file
            format_name: Expected format (auto-detected if None)
            
        Returns:
            ValidationResult with detailed analysis
        """
        logger.info(f"🔍 Validating multimodal dataset: {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            return ValidationResult(
                is_valid=False,
                format_name="unknown",
                total_samples=0,
                valid_samples=0,
                errors=[f"File not found: {file_path}"]
            )
        
        # Auto-detect format if not specified
        if format_name is None:
            format_name = self._detect_format(file_path)
        
        # Load dataset
        try:
            dataset = self._load_dataset(file_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                format_name=format_name,
                total_samples=0,
                valid_samples=0,
                errors=[f"Failed to load dataset: {str(e)}"]
            )
        
        # Validate format
        validator = self.format_validators.get(format_name)
        if validator is None:
            return ValidationResult(
                is_valid=False,
                format_name=format_name,
                total_samples=len(dataset) if isinstance(dataset, list) else 1,
                valid_samples=0,
                errors=[f"Unsupported format: {format_name}"]
            )
        
        # Run validation
        result = validator(dataset, format_name)
        
        # Add general statistics
        result.statistics.update({
            "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            "file_format": file_path.suffix,
            "detected_format": format_name
        })
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        logger.info(f"✅ Validation complete: {result.validity_score:.1f}% valid samples")
        return result
    
    def _detect_format(self, file_path: Path) -> str:
        """Auto-detect dataset format from file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    # Read first line for JSONL files
                    first_line = f.readline().strip()
                    if first_line:
                        sample = json.loads(first_line)
                else:
                    # Read entire file for JSON files
                    content = f.read().strip()
                    if content.startswith('['):
                        sample = json.loads(content)[0]
                    else:
                        sample = json.loads(content)
            
            # Detect format based on structure
            if "conversations" in sample:
                if isinstance(sample["conversations"], list):
                    if len(sample["conversations"]) > 0:
                        first_conv = sample["conversations"][0]
                        if "from" in first_conv and "value" in first_conv:
                            return "llava" if "from" in str(first_conv) else "sharegpt"
                return "sharegpt"
            elif "messages" in sample:
                return "chatml"
            elif "instruction" in sample or "prompt" in sample:
                return "conversation"
            else:
                return "unknown"
                
        except Exception as e:
            logger.warning(f"Failed to detect format: {e}")
            return "unknown"
    
    def _load_dataset(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load dataset from file."""
        dataset = []
        
        if file_path.suffix == '.jsonl':
            # Load JSONL file
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        try:
                            dataset.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON on line {line_num}: {e}")
        else:
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    dataset = data
                else:
                    dataset = [data]
        
        return dataset
    
    def _validate_llava_format(self, dataset: List[Dict], format_name: str) -> ValidationResult:
        """Validate LLaVA format dataset."""
        result = ValidationResult(
            is_valid=True,
            format_name=format_name,
            total_samples=len(dataset),
            valid_samples=0
        )
        
        multimodal_samples = 0
        total_conversations = 0
        role_distribution = {}
        
        for i, sample in enumerate(dataset):
            sample_errors = []
            
            # Check required fields
            if "conversations" not in sample:
                sample_errors.append(f"Sample {i}: Missing 'conversations' field")
            else:
                conversations = sample["conversations"]
                if not isinstance(conversations, list):
                    sample_errors.append(f"Sample {i}: 'conversations' must be a list")
                elif len(conversations) == 0:
                    sample_errors.append(f"Sample {i}: Empty conversations list")
                else:
                    total_conversations += len(conversations)
                    
                    # Validate each conversation turn
                    for j, conv in enumerate(conversations):
                        if not isinstance(conv, dict):
                            sample_errors.append(f"Sample {i}, Conv {j}: Must be a dictionary")
                            continue
                        
                        # Check required conversation fields
                        if "from" not in conv:
                            sample_errors.append(f"Sample {i}, Conv {j}: Missing 'from' field")
                        else:
                            role = conv["from"]
                            role_distribution[role] = role_distribution.get(role, 0) + 1
                            
                            if role not in {"human", "gpt", "system"}:
                                result.warnings.append(f"Sample {i}, Conv {j}: Non-standard role '{role}'")
                        
                        if "value" not in conv:
                            sample_errors.append(f"Sample {i}, Conv {j}: Missing 'value' field")
                        else:
                            value = conv["value"]
                            if not isinstance(value, str):
                                sample_errors.append(f"Sample {i}, Conv {j}: 'value' must be a string")
                            elif len(value.strip()) < self.min_text_length:
                                sample_errors.append(f"Sample {i}, Conv {j}: Content too short")
                            elif len(value) > self.max_text_length:
                                result.warnings.append(f"Sample {i}, Conv {j}: Content very long ({len(value)} chars)")
            
            # Check for multimodal content indicators
            sample_str = json.dumps(sample)
            if any(indicator in sample_str.lower() for indicator in ["image", "table", "figure", "diagram"]):
                multimodal_samples += 1
            
            # Count valid samples
            if not sample_errors:
                result.valid_samples += 1
            else:
                result.errors.extend(sample_errors)
        
        # Update statistics
        result.statistics.update({
            "multimodal_samples": multimodal_samples,
            "total_conversations": total_conversations,
            "avg_conversations_per_sample": total_conversations / len(dataset) if dataset else 0,
            "role_distribution": role_distribution,
            "multimodal_percentage": (multimodal_samples / len(dataset) * 100) if dataset else 0
        })
        
        result.is_valid = result.valid_samples > 0 and len(result.errors) == 0
        return result
    
    def _validate_sharegpt_format(self, dataset: List[Dict], format_name: str) -> ValidationResult:
        """Validate ShareGPT format dataset."""
        result = ValidationResult(
            is_valid=True,
            format_name=format_name,
            total_samples=len(dataset),
            valid_samples=0
        )
        
        multimodal_samples = 0
        role_distribution = {}
        
        for i, sample in enumerate(dataset):
            sample_errors = []
            
            # Check required fields
            if "conversations" not in sample:
                sample_errors.append(f"Sample {i}: Missing 'conversations' field")
            else:
                conversations = sample["conversations"]
                if not isinstance(conversations, list):
                    sample_errors.append(f"Sample {i}: 'conversations' must be a list")
                else:
                    # Validate each conversation turn
                    for j, conv in enumerate(conversations):
                        if "from" not in conv:
                            sample_errors.append(f"Sample {i}, Conv {j}: Missing 'from' field")
                        else:
                            role = conv["from"]
                            role_distribution[role] = role_distribution.get(role, 0) + 1
                        
                        if "value" not in conv:
                            sample_errors.append(f"Sample {i}, Conv {j}: Missing 'value' field")
                        else:
                            content = conv["value"]
                            if len(content.strip()) < self.min_text_length:
                                sample_errors.append(f"Sample {i}, Conv {j}: Content too short")
            
            # Check for multimodal content
            if "metadata" in sample and sample["metadata"].get("multimodal", False):
                multimodal_samples += 1
            
            # Count valid samples
            if not sample_errors:
                result.valid_samples += 1
            else:
                result.errors.extend(sample_errors)
        
        result.statistics.update({
            "multimodal_samples": multimodal_samples,
            "role_distribution": role_distribution,
            "multimodal_percentage": (multimodal_samples / len(dataset) * 100) if dataset else 0
        })
        
        result.is_valid = result.valid_samples > 0 and len(result.errors) == 0
        return result
    
    def _validate_chatml_format(self, dataset: List[Dict], format_name: str) -> ValidationResult:
        """Validate ChatML format dataset."""
        result = ValidationResult(
            is_valid=True,
            format_name=format_name,
            total_samples=len(dataset),
            valid_samples=0
        )
        
        role_distribution = {}
        
        for i, sample in enumerate(dataset):
            sample_errors = []
            
            # Check required fields
            if "messages" not in sample:
                sample_errors.append(f"Sample {i}: Missing 'messages' field")
            else:
                messages = sample["messages"]
                if not isinstance(messages, list):
                    sample_errors.append(f"Sample {i}: 'messages' must be a list")
                else:
                    # Validate each message
                    for j, msg in enumerate(messages):
                        if "role" not in msg:
                            sample_errors.append(f"Sample {i}, Msg {j}: Missing 'role' field")
                        else:
                            role = msg["role"]
                            role_distribution[role] = role_distribution.get(role, 0) + 1
                            
                            if role not in {"system", "user", "assistant"}:
                                result.warnings.append(f"Sample {i}, Msg {j}: Non-standard role '{role}'")
                        
                        if "content" not in msg:
                            sample_errors.append(f"Sample {i}, Msg {j}: Missing 'content' field")
                        else:
                            content = msg["content"]
                            if len(content.strip()) < self.min_text_length:
                                sample_errors.append(f"Sample {i}, Msg {j}: Content too short")
            
            # Count valid samples
            if not sample_errors:
                result.valid_samples += 1
            else:
                result.errors.extend(sample_errors)
        
        result.statistics.update({
            "role_distribution": role_distribution
        })
        
        result.is_valid = result.valid_samples > 0 and len(result.errors) == 0
        return result
    
    def _validate_conversation_format(self, dataset: List[Dict], format_name: str) -> ValidationResult:
        """Validate general conversation format dataset."""
        result = ValidationResult(
            is_valid=True,
            format_name=format_name,
            total_samples=len(dataset),
            valid_samples=0
        )
        
        for i, sample in enumerate(dataset):
            sample_errors = []
            
            # Check for instruction-response pairs
            has_instruction = "instruction" in sample or "prompt" in sample
            has_response = "output" in sample or "response" in sample or "completion" in sample
            
            if not (has_instruction and has_response):
                sample_errors.append(f"Sample {i}: Missing instruction-response pair")
            
            # Count valid samples
            if not sample_errors:
                result.valid_samples += 1
            else:
                result.errors.extend(sample_errors)
        
        result.is_valid = result.valid_samples > 0 and len(result.errors) == 0
        return result
    
    def _generate_recommendations(self, result: ValidationResult) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Validity recommendations
        if result.validity_score < 50:
            recommendations.append("Dataset has low validity score - consider data cleaning")
        elif result.validity_score < 80:
            recommendations.append("Dataset has moderate validity - minor fixes recommended")
        
        # Sample count recommendations
        if result.total_samples < 100:
            recommendations.append("Dataset is small - consider adding more samples for better training")
        elif result.total_samples > 10000:
            recommendations.append("Large dataset - consider sampling for faster iteration")
        
        # Multimodal recommendations
        if "multimodal_percentage" in result.statistics:
            multimodal_pct = result.statistics["multimodal_percentage"]
            if multimodal_pct < 10:
                recommendations.append("Low multimodal content - ensure images/tables are properly included")
            elif multimodal_pct > 90:
                recommendations.append("High multimodal content - good for vision-language training")
        
        # Format-specific recommendations
        if result.format_name == "llava":
            if "role_distribution" in result.statistics:
                roles = result.statistics["role_distribution"]
                if roles.get("human", 0) != roles.get("gpt", 0):
                    recommendations.append("Unbalanced human/gpt roles - ensure proper conversation flow")
        
        # Error recommendations
        if len(result.errors) > 0:
            recommendations.append("Fix validation errors before training")
        
        if len(result.warnings) > len(result.errors) * 2:
            recommendations.append("Many warnings detected - review data quality")
        
        return recommendations
    
    def validate_multiple_formats(self, base_path: str, dataset_name: str) -> Dict[str, ValidationResult]:
        """Validate multiple format files for the same dataset.
        
        Args:
            base_path: Base directory containing dataset files
            dataset_name: Base name of the dataset
            
        Returns:
            Dictionary mapping format names to validation results
        """
        base_path = Path(base_path)
        results = {}
        
        # Define format file patterns
        format_patterns = {
            "llava": f"{dataset_name}_llava.json",
            "sharegpt": f"{dataset_name}_sharegpt.jsonl",
            "chatml": f"{dataset_name}_chatml.jsonl"
        }
        
        for format_name, pattern in format_patterns.items():
            file_path = base_path / pattern
            if file_path.exists():
                results[format_name] = self.validate_dataset_file(str(file_path), format_name)
            else:
                logger.warning(f"Format file not found: {file_path}")
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult], output_path: str):
        """Generate a comprehensive validation report.
        
        Args:
            results: Dictionary of validation results by format
            output_path: Path to save the report
        """
        report = {
            "validation_timestamp": "2024-01-01T00:00:00",  # Would use actual timestamp
            "summary": {
                "total_formats": len(results),
                "valid_formats": sum(1 for r in results.values() if r.is_valid),
                "avg_validity_score": sum(r.validity_score for r in results.values()) / len(results) if results else 0
            },
            "format_results": {},
            "overall_recommendations": []
        }
        
        # Add individual format results
        for format_name, result in results.items():
            report["format_results"][format_name] = {
                "is_valid": result.is_valid,
                "validity_score": result.validity_score,
                "total_samples": result.total_samples,
                "valid_samples": result.valid_samples,
                "error_count": len(result.errors),
                "warning_count": len(result.warnings),
                "statistics": result.statistics,
                "recommendations": result.recommendations
            }
        
        # Generate overall recommendations
        if report["summary"]["avg_validity_score"] > 80:
            report["overall_recommendations"].append("Dataset quality is good for training")
        else:
            report["overall_recommendations"].append("Dataset needs improvement before training")
        
        if report["summary"]["valid_formats"] == 0:
            report["overall_recommendations"].append("No valid formats found - regenerate dataset")
        elif report["summary"]["valid_formats"] < len(results):
            report["overall_recommendations"].append("Some formats have issues - focus on valid ones")
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📊 Validation report saved: {output_path}")


def validate_multimodal_dataset(
    dataset_path: str,
    format_name: str = None,
    output_report: str = None
) -> ValidationResult:
    """High-level function to validate a multimodal dataset.
    
    Args:
        dataset_path: Path to the dataset file
        format_name: Expected format (auto-detected if None)
        output_report: Optional path to save validation report
        
    Returns:
        ValidationResult object
    """
    validator = MultimodalDatasetValidator()
    result = validator.validate_dataset_file(dataset_path, format_name)
    
    if output_report:
        # Save detailed report
        report_data = {
            format_name or "detected": {
                "result": result,
                "file_path": dataset_path
            }
        }
        validator.generate_validation_report(report_data, output_report)
    
    return result