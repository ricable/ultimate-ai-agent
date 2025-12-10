"""Comprehensive model evaluation framework for fine-tuned MLX models."""

import json
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for model evaluation metrics."""
    
    # Performance metrics
    perplexity: Optional[float] = None
    bleu_score: Optional[float] = None
    rouge_l: Optional[float] = None
    
    # Quality metrics
    factual_accuracy: Optional[float] = None
    coherence_score: Optional[float] = None
    relevance_score: Optional[float] = None
    
    # Technical metrics
    response_time_avg: Optional[float] = None
    tokens_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    
    # Domain-specific metrics
    technical_terminology_usage: Optional[float] = None
    documentation_adherence: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "performance": {
                "perplexity": self.perplexity,
                "bleu_score": self.bleu_score,
                "rouge_l": self.rouge_l
            },
            "quality": {
                "factual_accuracy": self.factual_accuracy,
                "coherence_score": self.coherence_score,
                "relevance_score": self.relevance_score
            },
            "technical": {
                "response_time_avg": self.response_time_avg,
                "tokens_per_second": self.tokens_per_second,
                "memory_usage_mb": self.memory_usage_mb
            },
            "domain": {
                "technical_terminology_usage": self.technical_terminology_usage,
                "documentation_adherence": self.documentation_adherence
            }
        }


@dataclass
class EvaluationSample:
    """Single evaluation sample."""
    
    prompt: str
    expected_answer: Optional[str] = None
    context: Optional[str] = None
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    
    # Results
    generated_answer: Optional[str] = None
    response_time: Optional[float] = None
    evaluation_scores: Optional[Dict[str, float]] = None


class ModelEvaluator:
    """Comprehensive evaluation framework for fine-tuned models."""
    
    def __init__(self, model_path: str, adapter_path: Optional[str] = None):
        """Initialize evaluator.
        
        Args:
            model_path: Path to model (base model or fused model)
            adapter_path: Path to adapter (if using base model + adapter)
        """
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.technical_terms = self._load_technical_terms()
        
    def _load_technical_terms(self) -> List[str]:
        """Load domain-specific technical terms for evaluation."""
        # Technical terms relevant to the cellular/telecommunications domain
        return [
            "PTP", "sync", "delay", "announce", "RadioEquipmentClockReference",
            "BWP", "DCI", "RRC", "cell", "base station", "frequency", "power",
            "voltage", "current", "battery", "antenna", "signal", "modulation",
            "uplink", "downlink", "MIMO", "beamforming", "scheduling", "MAC",
            "PHY", "protocol", "frame", "slot", "symbol", "carrier", "bandwidth",
            "interference", "noise", "SNR", "BER", "throughput", "latency",
            "handover", "mobility", "coverage", "capacity", "optimization",
            "configuration", "parameter", "algorithm", "measurement", "KPI"
        ]
    
    def create_evaluation_dataset(self, chunks_dir: str) -> List[EvaluationSample]:
        """Create evaluation dataset from document chunks.
        
        Args:
            chunks_dir: Directory containing document chunks
            
        Returns:
            List of evaluation samples
        """
        logger.info(f"Creating evaluation dataset from {chunks_dir}")
        
        chunks_path = Path(chunks_dir)
        chunk_files = sorted(chunks_path.glob("chunk_*.json"))
        
        samples = []
        
        # Categories of evaluation questions
        question_templates = {
            "factual": [
                "What is {concept}?",
                "Explain {concept} in detail.",
                "How does {concept} work?",
                "What are the main features of {concept}?",
                "Describe the purpose of {concept}."
            ],
            "technical": [
                "What are the technical specifications of {concept}?",
                "How do you configure {concept}?",
                "What parameters control {concept}?",
                "What are the troubleshooting steps for {concept}?",
                "What are the performance metrics for {concept}?"
            ],
            "procedural": [
                "How do you implement {concept}?",
                "What are the steps to set up {concept}?",
                "How do you optimize {concept}?",
                "What is the procedure for {concept}?",
                "How do you monitor {concept}?"
            ]
        }
        
        for chunk_file in chunk_files[:50]:  # Limit for evaluation
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                text = chunk_data.get("text", "").strip()
                if len(text) < 100:  # Skip very short chunks
                    continue
                
                # Extract key concepts from the text
                concepts = self._extract_concepts(text)
                
                for concept in concepts[:2]:  # Max 2 concepts per chunk
                    for category, templates in question_templates.items():
                        for template in templates[:1]:  # One question per template
                            prompt = template.format(concept=concept)
                            
                            # Create sample
                            sample = EvaluationSample(
                                prompt=f"Document: combined_document.md\nQuestion: {prompt}\nAnswer:",
                                context=text,
                                category=category,
                                difficulty="medium"
                            )
                            samples.append(sample)
                            
                            if len(samples) >= 100:  # Limit total samples
                                break
                        
                        if len(samples) >= 100:
                            break
                    
                    if len(samples) >= 100:
                        break
                
                if len(samples) >= 100:
                    break
                    
            except Exception as e:
                logger.warning(f"Failed to process {chunk_file}: {e}")
        
        logger.info(f"Created {len(samples)} evaluation samples")
        return samples
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        concepts = []
        
        # Look for technical terms
        text_lower = text.lower()
        for term in self.technical_terms:
            if term.lower() in text_lower:
                concepts.append(term)
        
        # Extract capitalized terms (likely to be technical concepts)
        capitalized_terms = re.findall(r'\b[A-Z][A-Za-z]{2,}\b', text)
        concepts.extend(capitalized_terms[:3])  # Max 3 capitalized terms
        
        # Remove duplicates and return
        return list(set(concepts))[:5]  # Max 5 concepts
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> Tuple[str, float]:
        """Generate response using the model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response, response_time)
        """
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", self.model_path,
            "--prompt", prompt,
            "--max-tokens", str(max_tokens),
            "--temp", "0.7"
        ]
        
        if self.adapter_path:
            cmd.extend(["--adapter-path", self.adapter_path])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=120
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if result.returncode == 0:
                # Extract generated text from output
                output = result.stdout.strip()
                
                # Remove prompt from output if present
                if prompt in output:
                    response = output.replace(prompt, "").strip()
                else:
                    response = output
                
                # Clean up the response
                response = self._clean_response(response)
                
                return response, response_time
            else:
                logger.error(f"Generation failed: {result.stderr}")
                return f"Error: {result.stderr}", response_time
                
        except subprocess.TimeoutExpired:
            return "Error: Generation timed out", 120.0
        except Exception as e:
            return f"Error: {e}", 0.0
    
    def _clean_response(self, response: str) -> str:
        """Clean up generated response."""
        # Remove common artifacts
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip lines with generation artifacts
            if any(artifact in line.lower() for artifact in [
                'tokens-per-sec', 'peak memory', 'prompt:', 'calling', 'fetching'
            ]):
                continue
            if line.startswith('=========='):
                continue
            if line and not line.startswith('>'):
                cleaned_lines.append(line)
        
        return ' '.join(cleaned_lines).strip()
    
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationSample:
        """Evaluate a single sample.
        
        Args:
            sample: Sample to evaluate
            
        Returns:
            Sample with results filled in
        """
        # Generate response
        response, response_time = self.generate_response(sample.prompt)
        
        sample.generated_answer = response
        sample.response_time = response_time
        
        # Calculate evaluation scores
        scores = {}
        
        # Technical terminology usage
        scores['technical_terminology'] = self._calculate_technical_term_usage(response)
        
        # Response quality (simple heuristics)
        scores['response_length'] = min(len(response) / 200.0, 1.0)  # Normalized length
        scores['coherence'] = self._calculate_coherence_score(response)
        scores['relevance'] = self._calculate_relevance_score(sample.prompt, response)
        
        sample.evaluation_scores = scores
        
        return sample
    
    def _calculate_technical_term_usage(self, response: str) -> float:
        """Calculate percentage of technical terms used appropriately."""
        if not response:
            return 0.0
        
        response_lower = response.lower()
        term_count = 0
        
        for term in self.technical_terms:
            if term.lower() in response_lower:
                term_count += 1
        
        # Normalize by response length and term availability
        return min(term_count / max(len(response.split()) * 0.1, 1), 1.0)
    
    def _calculate_coherence_score(self, response: str) -> float:
        """Calculate coherence score based on simple heuristics."""
        if not response:
            return 0.0
        
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.5
        
        # Check for repeated words (sign of poor coherence)
        words = response.lower().split()
        unique_words = set(words)
        repetition_ratio = len(unique_words) / max(len(words), 1)
        
        # Check sentence structure
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
        length_score = min(avg_sentence_length / 15.0, 1.0)  # Ideal around 15 words
        
        return (repetition_ratio + length_score) / 2.0
    
    def _calculate_relevance_score(self, prompt: str, response: str) -> float:
        """Calculate relevance score based on keyword overlap."""
        if not response:
            return 0.0
        
        # Extract keywords from prompt
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower()))
        response_words = set(re.findall(r'\b\w+\b', response.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'what', 'how', 'is', 'are'}
        prompt_words -= stop_words
        response_words -= stop_words
        
        if not prompt_words:
            return 0.5
        
        # Calculate overlap
        overlap = len(prompt_words.intersection(response_words))
        return overlap / len(prompt_words)
    
    def run_comprehensive_evaluation(
        self, 
        samples: List[EvaluationSample],
        output_dir: str = "evaluation_results"
    ) -> EvaluationMetrics:
        """Run comprehensive evaluation on samples.
        
        Args:
            samples: List of evaluation samples
            output_dir: Directory to save results
            
        Returns:
            Aggregated evaluation metrics
        """
        logger.info(f"Starting comprehensive evaluation on {len(samples)} samples")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        evaluated_samples = []
        response_times = []
        scores_by_category = {}
        
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")
            
            evaluated_sample = self.evaluate_sample(sample)
            evaluated_samples.append(evaluated_sample)
            
            if evaluated_sample.response_time:
                response_times.append(evaluated_sample.response_time)
            
            # Group scores by category
            if sample.category not in scores_by_category:
                scores_by_category[sample.category] = []
            scores_by_category[sample.category].append(evaluated_sample.evaluation_scores)
        
        # Calculate aggregate metrics
        metrics = EvaluationMetrics()
        
        # Technical metrics
        if response_times:
            metrics.response_time_avg = np.mean(response_times)
        
        # Quality metrics from individual samples
        all_scores = [s.evaluation_scores for s in evaluated_samples if s.evaluation_scores]
        
        if all_scores:
            metrics.technical_terminology_usage = np.mean([
                s.get('technical_terminology', 0) for s in all_scores
            ])
            metrics.coherence_score = np.mean([
                s.get('coherence', 0) for s in all_scores
            ])
            metrics.relevance_score = np.mean([
                s.get('relevance', 0) for s in all_scores
            ])
        
        # Save detailed results
        self._save_evaluation_results(evaluated_samples, metrics, output_path)
        
        logger.info("Evaluation completed successfully")
        return metrics
    
    def _save_evaluation_results(
        self, 
        samples: List[EvaluationSample], 
        metrics: EvaluationMetrics,
        output_path: Path
    ) -> None:
        """Save evaluation results to files."""
        
        # Save individual sample results
        samples_data = []
        for sample in samples:
            sample_dict = {
                "prompt": sample.prompt,
                "generated_answer": sample.generated_answer,
                "response_time": sample.response_time,
                "category": sample.category,
                "difficulty": sample.difficulty,
                "evaluation_scores": sample.evaluation_scores
            }
            samples_data.append(sample_dict)
        
        with open(output_path / "evaluation_samples.json", 'w', encoding='utf-8') as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
        
        # Save aggregated metrics
        with open(output_path / "evaluation_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        # Generate evaluation report
        self._generate_evaluation_report(samples, metrics, output_path)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def _generate_evaluation_report(
        self, 
        samples: List[EvaluationSample], 
        metrics: EvaluationMetrics,
        output_path: Path
    ) -> None:
        """Generate human-readable evaluation report."""
        
        report_content = f"""# Model Evaluation Report

## Model Information
- Model Path: {self.model_path}
- Adapter Path: {self.adapter_path or 'N/A'}
- Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

## Summary Metrics

### Performance Metrics
- Average Response Time: {metrics.response_time_avg:.2f}s
- Technical Terminology Usage: {metrics.technical_terminology_usage:.2%}
- Coherence Score: {metrics.coherence_score:.2%}
- Relevance Score: {metrics.relevance_score:.2%}

### Sample Distribution
- Total Samples: {len(samples)}
- Categories: {len(set(s.category for s in samples))}
- Average Response Length: {np.mean([len(s.generated_answer or '') for s in samples]):.0f} characters

## Category Breakdown

"""
        
        # Category analysis
        categories = {}
        for sample in samples:
            if sample.category not in categories:
                categories[sample.category] = []
            categories[sample.category].append(sample)
        
        for category, cat_samples in categories.items():
            cat_scores = [s.evaluation_scores for s in cat_samples if s.evaluation_scores]
            if cat_scores:
                avg_coherence = np.mean([s.get('coherence', 0) for s in cat_scores])
                avg_relevance = np.mean([s.get('relevance', 0) for s in cat_scores])
                avg_tech_terms = np.mean([s.get('technical_terminology', 0) for s in cat_scores])
                
                report_content += f"""### {category.title()} Category
- Samples: {len(cat_samples)}
- Average Coherence: {avg_coherence:.2%}
- Average Relevance: {avg_relevance:.2%}
- Technical Terms Usage: {avg_tech_terms:.2%}

"""
        
        # Sample responses
        report_content += """## Sample Responses

"""
        
        for i, sample in enumerate(samples[:5]):  # Show first 5 samples
            report_content += f"""### Sample {i+1}
**Prompt:** {sample.prompt[:100]}...

**Response:** {sample.generated_answer[:200] if sample.generated_answer else 'No response'}...

**Response Time:** {sample.response_time:.2f}s

---

"""
        
        # Save report
        with open(output_path / "evaluation_report.md", 'w', encoding='utf-8') as f:
            f.write(report_content)
    
    def benchmark_model_performance(self) -> Dict[str, float]:
        """Benchmark model performance on standard tasks.
        
        Returns:
            Performance benchmark results
        """
        logger.info("Running model performance benchmark")
        
        benchmark_prompts = [
            "Document: combined_document.md\nQuestion: What is PTP synchronization?\nAnswer:",
            "Document: combined_document.md\nQuestion: How does power optimization work?\nAnswer:",
            "Document: combined_document.md\nQuestion: What are the main system components?\nAnswer:",
            "Document: combined_document.md\nQuestion: Explain the troubleshooting process.\nAnswer:",
            "Document: combined_document.md\nQuestion: What are the performance indicators?\nAnswer:"
        ]
        
        results = {
            "avg_response_time": 0.0,
            "avg_tokens_per_second": 0.0,
            "successful_generations": 0,
            "total_attempts": len(benchmark_prompts)
        }
        
        total_time = 0.0
        total_tokens = 0
        successful = 0
        
        for prompt in benchmark_prompts:
            response, response_time = self.generate_response(prompt, max_tokens=100)
            
            if not response.startswith("Error:"):
                successful += 1
                total_time += response_time
                total_tokens += len(response.split())
        
        if successful > 0:
            results["avg_response_time"] = total_time / successful
            results["avg_tokens_per_second"] = total_tokens / total_time if total_time > 0 else 0
        
        results["successful_generations"] = successful
        
        logger.info(f"Benchmark results: {results}")
        return results


def run_model_evaluation(
    model_path: str,
    adapter_path: Optional[str] = None,
    chunks_dir: str = "pipeline_output/chunks",
    output_dir: str = "evaluation_results"
) -> EvaluationMetrics:
    """Run complete model evaluation workflow.
    
    Args:
        model_path: Path to model
        adapter_path: Path to adapters (optional)
        chunks_dir: Directory with document chunks
        output_dir: Output directory for results
        
    Returns:
        Evaluation metrics
    """
    evaluator = ModelEvaluator(model_path, adapter_path)
    
    # Create evaluation dataset
    samples = evaluator.create_evaluation_dataset(chunks_dir)
    
    # Run comprehensive evaluation
    metrics = evaluator.run_comprehensive_evaluation(samples, output_dir)
    
    # Run performance benchmark
    benchmark_results = evaluator.benchmark_model_performance()
    
    # Update metrics with benchmark results
    metrics.response_time_avg = benchmark_results["avg_response_time"]
    metrics.tokens_per_second = benchmark_results["avg_tokens_per_second"]
    
    return metrics