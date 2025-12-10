"""
Advanced dataset generation for fine-tuning with quality optimization.

This module provides comprehensive dataset generation from Flow4 chunks with
deduplication, quality filtering, and intelligent response generation.
"""

import json
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

from ..utils.logging import get_logger
from ..utils.deduplication import DatasetDeduplicator
from .multimodal_dataset_generator import MultimodalDatasetGenerator, MultimodalDatasetConfig

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset generation."""
    chunks_dir: str = "output/chunks"
    output_dir: str = "output/fine_tuning_datasets"
    max_chunks: int = None
    enable_deduplication: bool = True
    dedupe_strategy: str = "progressive"
    context_length: int = 2048
    qa_pairs_per_chunk: int = 3
    include_reasoning: bool = True
    include_rag_format: bool = True
    include_chat_format: bool = True
    include_jsonl_format: bool = True  # Always create JSONL for MLX compatibility
    include_mlx_format: bool = True   # Create MLX-specific dataset structure
    dataset_context: str = "telecommunications technical documentation and 5G network specifications"
    
    # Multimodal dataset options
    enable_multimodal: bool = True
    multimodal_strategy: str = "enhanced"  # basic, enhanced, comprehensive
    include_image_analysis: bool = True
    include_table_analysis: bool = True
    multimodal_formats: List[str] = None  # LLaVA, ShareGPT, ChatML
    
    def __post_init__(self):
        if self.multimodal_formats is None:
            self.multimodal_formats = ["llava", "sharegpt", "chatml"]


class ChunkOptimizer:
    """Optimize chunk content for high-quality fine-tuning datasets."""
    
    def __init__(self):
        self.min_chunk_length = 100
        self.max_chunk_length = 2000
        
    def clean_chunk_content(self, text: str) -> str:
        """Deep clean chunk content."""
        if not text:
            return ""
            
        # Fix malformed table content (the biggest issue)
        # Pattern: "Feature Name, Feature Identity = FAJ 121 5443. Feature Name, Access Type = NR."
        text = re.sub(
            r'([A-Za-z\s]+?),\s*([A-Za-z\s]+?)\s*=\s*([^.]+?)\.\s*', 
            r'\1: \2 = \3. ', 
            text
        )
        
        # Fix broken lists and bullets
        text = re.sub(r'^\s*[-*+]\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n\s*[-*+]\s*([A-Z])', r'\n- \1', text)
        
        # Fix excessive punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\s*\.\s*\.\s*', '. ', text)
        
        # Fix broken sentences
        text = re.sub(r'(\w)\s*\.\s*([A-Z])', r'\1. \2', text)
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove standalone section headers that got separated
        standalone_headers = [
            r'^\s*\d+\s+[A-Z][A-Za-z\s]+\s*$',
            r'^\s*##?\s*[A-Z][A-Za-z\s]+\s*$',
            r'^\s*[A-Z][A-Za-z\s]+:\s*$'
        ]
        
        for pattern in standalone_headers:
            if re.match(pattern, text.strip()):
                return ""  # Skip header-only chunks
                
        # Clean up remaining artifacts
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
        
    def extract_meaningful_content(self, text: str) -> str:
        """Extract the most meaningful content from a chunk."""
        if not text:
            return ""
            
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        if not sentences:
            return text
            
        # Score sentences based on information content
        scored_sentences = []
        for sentence in sentences:
            score = 0
            
            # Favor sentences with technical terms
            technical_terms = [
                'feature', 'configuration', 'parameter', 'attribute', 'threshold',
                'optimization', 'performance', 'bandwidth', 'power', 'efficiency',
                'procedure', 'algorithm', 'implementation', 'system', 'network'
            ]
            score += sum(1 for term in technical_terms if term.lower() in sentence.lower())
            
            # Favor sentences with numbers/values
            if re.search(r'\d+', sentence):
                score += 0.5
                
            # Favor longer sentences (but not too long)
            length_score = min(len(sentence) / 100, 2.0)
            score += length_score
            
            # Penalize very short sentences
            if len(sentence) < 20:
                score -= 1.0
                
            scored_sentences.append((score, sentence))
            
        # Sort by score and take the best sentences
        scored_sentences.sort(reverse=True)
        
        # Reconstruct with best sentences
        if len(scored_sentences) <= 3:
            return '. '.join([s[1] for s in scored_sentences]) + '.'
        else:
            # Take top 60% of sentences
            keep_count = max(2, int(len(scored_sentences) * 0.6))
            kept_sentences = [s[1] for s in scored_sentences[:keep_count]]
            
            # Reorder to maintain original flow where possible
            original_order = []
            for sentence in sentences:
                if sentence in kept_sentences:
                    original_order.append(sentence)
                    
            return '. '.join(original_order) + '.'
            
    def is_high_quality_chunk(self, text: str) -> bool:
        """Check if chunk meets quality standards for training."""
        if not text or len(text.strip()) < self.min_chunk_length:
            return False
            
        if len(text) > self.max_chunk_length:
            return False
            
        # Must have at least 2 sentences
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        if sentence_count < 2:
            return False
            
        # Must contain technical content
        technical_indicators = [
            'feature', 'configuration', 'parameter', 'optimization', 'performance',
            'system', 'network', 'algorithm', 'implementation', 'procedure'
        ]
        
        if not any(indicator in text.lower() for indicator in technical_indicators):
            return False
            
        # Must not be mostly punctuation or malformed
        alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
        if alpha_ratio < 0.6:
            return False
            
        return True
        
    def optimize_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a single chunk."""
        original_text = chunk.get('text', '')
        
        # Clean the content
        cleaned_text = self.clean_chunk_content(original_text)
        
        if not cleaned_text:
            return None
            
        # Extract meaningful content
        optimized_text = self.extract_meaningful_content(cleaned_text)
        
        if not self.is_high_quality_chunk(optimized_text):
            return None
            
        # Update chunk
        optimized_chunk = chunk.copy()
        optimized_chunk['text'] = optimized_text
        optimized_chunk['metadata'] = chunk.get('metadata', {}).copy()
        optimized_chunk['metadata']['optimized'] = True
        optimized_chunk['metadata']['original_length'] = len(original_text)
        optimized_chunk['metadata']['optimized_length'] = len(optimized_text)
        
        return optimized_chunk
        
    def optimize_dataset(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize entire dataset."""
        optimized_chunks = []
        skipped_count = 0
        
        for chunk in chunks:
            optimized = self.optimize_chunk(chunk)
            if optimized:
                optimized_chunks.append(optimized)
            else:
                skipped_count += 1
                
        logger.info(f"Optimized {len(optimized_chunks)} chunks, skipped {skipped_count}")
        return optimized_chunks


class OptimizedFineTuneDatasetGenerator:
    """Generate high-quality fine-tuning datasets for technical documentation."""
    
    def __init__(self):
        """Initialize the generator with domain-specific templates."""
        self.technical_instruction_templates = [
            # Feature explanation templates
            "Explain how {feature} works in {domain}.",
            "What is {feature} and how does it improve {domain} performance?",
            "Describe the implementation of {feature} in {domain} systems.",
            "How does {feature} affect {domain} operations?",
            
            # Configuration templates  
            "How do you configure {parameter} for optimal {purpose}?",
            "What are the configuration options for {feature}?",
            "Explain the relationship between {param1} and {param2}.",
            
            # Troubleshooting templates
            "What are the limitations of {feature}?",
            "When should you avoid using {feature}?",
            "What dependencies does {feature} have?",
            "How does {feature} interact with {other_feature}?",
            
            # Process templates
            "Describe the {process_name} procedure.",
            "What are the steps involved in {process_name}?",
            "When is {process_name} triggered?",
            
            # Comparison templates
            "What's the difference between {option1} and {option2}?",
            "Compare {feature1} and {feature2} in terms of performance.",
            
            # Impact templates
            "What is the network impact of enabling {feature}?",
            "How does {feature} affect battery consumption?",
            "What performance benefits does {feature} provide?"
        ]
        
    def extract_technical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract technical entities from text for intelligent template filling."""
        entities = {
            'features': [],
            'parameters': [],
            'procedures': [],
            'technologies': [],
            'components': []
        }
        
        # Extract NR/5G features
        feature_patterns = [
            r'NR\s+([A-Z][A-Za-z\s]+?)(?=\s+feature|\s+Feature|\s*\||\s*,|\s*\.|\s*$)',
            r'(BWP\s*\w*)',
            r'(DCI-based\s+\w+)',
            r'(Power\s+Optimizer?)',
            r'(Carrier\s+Aggregation)',
            r'(MIMO\s+\w*)',
            r'(Beamforming)'
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['features'].extend([m.strip() for m in matches if len(m.strip()) > 2])
        
        # Extract parameters
        param_patterns = [
            r'(\w+\.\w+\.[a-zA-Z]+)',  # MOM attributes
            r'([a-zA-Z]+Threshold)',
            r'([a-zA-Z]+Timer)',
            r'([a-zA-Z]+Config)'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, text)
            entities['parameters'].extend([m for m in matches if len(m) > 3])
        
        # Extract procedures
        procedure_patterns = [
            r'(\w+switch\s+Procedure)',
            r'(Handover)',
            r'(Setup)',
            r'(Configuration)'
        ]
        
        for pattern in procedure_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['procedures'].extend([m.strip() for m in matches])
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = list(set([e for e in entities[key] if len(e) > 2]))[:5]  # Limit to 5 per type
            
        return entities
    
    def generate_intelligent_response(self, chunk_text: str, instruction: str) -> str:
        """Generate intelligent response based on instruction type and content."""
        
        # Clean the chunk text first
        cleaned_text = self.clean_chunk_text(chunk_text)
        
        if "explain how" in instruction.lower() or "how does" in instruction.lower():
            return self.generate_explanation_response(cleaned_text)
        elif "what is" in instruction.lower() or "describe" in instruction.lower():
            return self.generate_description_response(cleaned_text)
        elif "configure" in instruction.lower() or "configuration" in instruction.lower():
            return self.generate_configuration_response(cleaned_text)
        elif "procedure" in instruction.lower() or "steps" in instruction.lower():
            return self.generate_procedure_response(cleaned_text)
        elif "limitations" in instruction.lower() or "dependencies" in instruction.lower():
            return self.generate_constraints_response(cleaned_text)
        else:
            return self.generate_summary_response(cleaned_text)
    
    def clean_chunk_text(self, text: str) -> str:
        """Clean malformed chunk text."""
        # Fix broken table formatting
        text = re.sub(r'(\w+),\s*([A-Z][a-z\s]+)\s*=\s*([^.]+)\.', r'\1: \2 = \3.', text)
        
        # Fix repeated periods
        text = re.sub(r'\.{2,}', '.', text)
        
        # Fix broken sentences
        text = re.sub(r'(\w)\s*\.\s*(\w)', r'\1. \2', text)
        
        # Remove malformed list markers
        text = re.sub(r'^\s*[-*+]\s*$', '', text, flags=re.MULTILINE)
        
        # Fix spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def generate_explanation_response(self, text: str) -> str:
        """Generate explanation-style response."""
        # Extract key concepts and explain them
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if not sentences:
            return text
            
        # Start with the main concept
        intro = sentences[0] + '.'
        
        # Add explanation of how it works
        working_sentences = [s for s in sentences if any(word in s.lower() for word in ['works', 'enables', 'allows', 'provides', 'supports'])]
        
        explanation = intro
        if working_sentences:
            explanation += f" {working_sentences[0]}."
            
        # Add benefits or purpose
        benefit_sentences = [s for s in sentences if any(word in s.lower() for word in ['improve', 'reduce', 'increase', 'optimize', 'enhance'])]
        if benefit_sentences:
            explanation += f" This {benefit_sentences[0]}."
            
        return explanation
    
    def generate_description_response(self, text: str) -> str:
        """Generate descriptive response."""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) <= 2:
            return text
            
        # Create structured description
        description = f"{sentences[0]}. "
        
        # Add key characteristics
        if len(sentences) > 1:
            description += f"{sentences[1]}. "
            
        # Add context or additional info
        context_sentences = [s for s in sentences[2:] if len(s) > 20]
        if context_sentences:
            description += f"Additionally, {context_sentences[0].lower()}."
            
        return description
    
    def generate_configuration_response(self, text: str) -> str:
        """Generate configuration-focused response."""
        # Extract configuration-related information
        config_keywords = ['attribute', 'parameter', 'threshold', 'timer', 'setting', 'configure']
        config_sentences = [s for s in text.split('.') if any(kw in s.lower() for kw in config_keywords)]
        
        if config_sentences:
            response = "Configuration details: "
            response += '. '.join(config_sentences[:3]) + '.'
            return response
        
        return self.generate_summary_response(text)
    
    def generate_procedure_response(self, text: str) -> str:
        """Generate procedure-focused response."""
        # Look for procedural language
        procedure_keywords = ['step', 'procedure', 'process', 'performed', 'triggered', 'executed']
        proc_sentences = [s for s in text.split('.') if any(kw in s.lower() for kw in procedure_keywords)]
        
        if proc_sentences:
            response = "Process overview: "
            response += '. '.join(proc_sentences[:3]) + '.'
            return response
            
        return self.generate_summary_response(text)
    
    def generate_constraints_response(self, text: str) -> str:
        """Generate constraints/limitations response."""
        constraint_keywords = ['limitation', 'requirement', 'dependency', 'constraint', 'not support', 'cannot', 'must']
        constraint_sentences = [s for s in text.split('.') if any(kw in s.lower() for kw in constraint_keywords)]
        
        if constraint_sentences:
            response = "Key constraints and requirements: "
            response += '. '.join(constraint_sentences[:3]) + '.'
            return response
            
        return self.generate_summary_response(text)
    
    def generate_summary_response(self, text: str) -> str:
        """Generate summary response."""
        sentences = [s.strip() for s in text.split('.') if s.strip() and len(s.strip()) > 10]
        
        if len(sentences) <= 3:
            return text
            
        # Take first sentence and most informative sentence
        summary = sentences[0] + '. '
        
        # Find the most informative sentence (longest with technical terms)
        technical_terms = ['feature', 'system', 'configuration', 'performance', 'optimization']
        scored_sentences = []
        
        for sentence in sentences[1:]:
            score = sum(1 for term in technical_terms if term in sentence.lower())
            score += len(sentence) / 100  # Favor longer sentences
            scored_sentences.append((score, sentence))
            
        if scored_sentences:
            scored_sentences.sort(reverse=True)
            summary += scored_sentences[0][1] + '.'
            
        return summary
    
    def create_quality_instruction_pairs(self, chunks: List[Dict]) -> List[Dict[str, Any]]:
        """Create high-quality instruction-response pairs."""
        training_data = []
        
        for chunk in chunks:
            text = chunk.get('text', '')
            if not text or len(text.strip()) < 50:  # Skip very short chunks
                continue
                
            # Extract technical entities for intelligent template filling
            entities = self.extract_technical_entities(text)
            
            # Generate domain-specific instructions
            instructions = self.generate_domain_instructions(text, entities)
            
            # Limit to 2-3 high-quality instructions per chunk
            for instruction in instructions[:3]:
                # Generate intelligent response
                response = self.generate_intelligent_response(text, instruction)
                
                # Quality check
                if self.passes_quality_check(instruction, response, text):
                    training_data.append({
                        "instruction": instruction,
                        "input": "",  # No input to avoid data leakage
                        "output": response,
                        "metadata": {
                            "source": chunk.get('source', 'unknown'),
                            "chunk_id": chunk.get('id', 'unknown'),
                            "type": "optimized_instruction_response",
                            "entities": entities,
                            "quality_score": self.calculate_quality_score(instruction, response)
                        }
                    })
                    
        return training_data
    
    def generate_domain_instructions(self, text: str, entities: Dict[str, List[str]]) -> List[str]:
        """Generate domain-specific instructions using extracted entities."""
        instructions = []
        
        # Use entities to fill templates
        for template in self.technical_instruction_templates[:8]:  # Limit templates
            try:
                if '{feature}' in template and entities['features']:
                    feature = random.choice(entities['features'])
                    instruction = template.replace('{feature}', feature)
                    instruction = instruction.replace('{domain}', 'NR/5G')
                    instruction = instruction.replace('{purpose}', 'power optimization')
                    instructions.append(instruction)
                    
                elif '{parameter}' in template and entities['parameters']:
                    param = random.choice(entities['parameters'])
                    instruction = template.replace('{parameter}', param)
                    instruction = instruction.replace('{purpose}', 'optimal performance')
                    instructions.append(instruction)
                    
                elif '{process_name}' in template and entities['procedures']:
                    proc = random.choice(entities['procedures'])
                    instruction = template.replace('{process_name}', proc)
                    instructions.append(instruction)
                    
            except (IndexError, KeyError):
                continue
                
        # Add generic but good instructions
        if not instructions:
            instructions = [
                f"Explain the key technical concepts discussed in this documentation.",
                f"What are the main features and their benefits described here?",
                f"Summarize the important configuration and operational details."
            ]
            
        return instructions[:3]  # Limit to 3 per chunk
    
    def passes_quality_check(self, instruction: str, response: str, original_text: str) -> bool:
        """Check if instruction-response pair meets quality standards."""
        
        # Basic length checks
        if len(instruction) < 10 or len(response) < 30:
            return False
            
        # Check for data leakage (response shouldn't be identical to input)
        if response.strip() == original_text.strip():
            return False
            
        # Check for meaningful content
        if response.count('.') < 2:  # Should have at least 2 sentences
            return False
            
        # Check for technical content
        technical_indicators = ['feature', 'configuration', 'parameter', 'optimization', 'performance']
        if not any(word in response.lower() for word in technical_indicators):
            return False
            
        # Check instruction quality
        if instruction.count('?') == 0 and not any(word in instruction.lower() for word in ['explain', 'describe', 'what', 'how']):
            return False
            
        return True
    
    def calculate_quality_score(self, instruction: str, response: str) -> float:
        """Calculate quality score for prioritization."""
        score = 0.0
        
        # Instruction quality
        if any(word in instruction.lower() for word in ['explain', 'how', 'what', 'describe']):
            score += 1.0
            
        # Response quality
        sentence_count = response.count('.')
        score += min(sentence_count / 5.0, 1.0)  # Favor 3-5 sentences
        
        # Technical content
        technical_terms = ['feature', 'configuration', 'optimization', 'performance', 'system']
        tech_score = sum(1 for term in technical_terms if term in response.lower())
        score += min(tech_score / 3.0, 1.0)
        
        # Length appropriateness
        if 100 <= len(response) <= 300:
            score += 0.5
            
        return min(score, 5.0)


@dataclass
class LargeMLXDatasetGeneratorConfig:
    """Configuration for large MLX dataset generation."""
    chunks_dir: str = "./output/chunks"
    dataset_context: str = "Ericsson telecommunications equipment and LTE network documentation"
    max_chunks: Optional[int] = None
    qa_pairs_per_chunk: int = 3
    include_rag_format: bool = True

class LargeMLXDatasetGenerator:
    """Generate large-scale MLX datasets efficiently from cached chunks."""
    
    def __init__(self, config: Optional[LargeMLXDatasetGeneratorConfig] = None):
        """Initialize the generator with domain-specific templates."""
        self.config = config or LargeMLXDatasetGeneratorConfig()
        self.min_chunk_length = 50
        self.max_chunk_length = 2000
        
    def load_cached_chunks(self, cache_file: str) -> List[Dict[str, Any]]:
        """Load chunks from the cached JSON file."""
        with open(cache_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        logger.info(f"✅ Loaded {len(chunks)} chunks from cache")
        return chunks
    
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from the Flow4 output directory or cached file."""
        chunks_path = Path(self.config.chunks_dir)
        
        # Try to load from cached file first (more efficient)
        cache_files = list(chunks_path.glob("chunk_cache_*.json"))
        if cache_files:
            logger.info(f"📋 Found cached chunks file: {cache_files[0]}")
            try:
                with open(cache_files[0], 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                if self.config.max_chunks:
                    chunks = chunks[:self.config.max_chunks]
                    
                logger.info(f"✅ Loaded {len(chunks)} chunks from cache")
                return chunks
            except Exception as e:
                logger.warning(f"Failed to load cached chunks: {e}, falling back to individual files")
        
        # Fallback to individual chunk files
        chunk_files = sorted(chunks_path.glob("chunk_*.json"))
        
        if self.config.max_chunks:
            chunk_files = chunk_files[:self.config.max_chunks]
        
        chunks = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to load chunk {chunk_file}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(chunks)} chunks from individual files")
        return chunks
        
    def create_qa_pairs(self, chunks: List[Dict[str, Any]], num_pairs: int = 800) -> List[Dict[str, Any]]:
        """Create question-answer pairs from chunks efficiently."""
        qa_pairs = []
        
        # Sample chunks if we have more than needed
        if len(chunks) > num_pairs:
            selected_chunks = random.sample(chunks, num_pairs)
        else:
            selected_chunks = chunks
        
        logger.info(f"📝 Creating QA pairs from {len(selected_chunks)} chunks...")
        
        for i, chunk in enumerate(selected_chunks):
            text = chunk.get('text', '')
            
            # Extract key information from the chunk
            lines = text.split('\n')
            content_lines = [line.strip() for line in lines if line.strip() and not line.startswith('metadata')]
            
            if len(content_lines) < 2:
                continue
                
            # Create different types of questions based on content
            questions = []
            
            # Type 1: Direct content questions
            if 'KPI' in text or 'Performance' in text:
                questions.append({
                    "question": f"What key performance indicators are mentioned in this Ericsson documentation section?",
                    "answer": f"This section covers: {' '.join(content_lines[:3])}"
                })
            
            # Type 2: Technical definition questions
            if any(term in text for term in ['E-RAB', 'LTE', 'Throughput', 'Success Rate']):
                questions.append({
                    "question": f"Explain the technical concepts described in this telecommunications documentation.",
                    "answer": f"The documentation explains: {' '.join(content_lines[:4])}"
                })
            
            # Type 3: Network monitoring questions
            if any(term in text for term in ['monitoring', 'measurement', 'network', 'cell']):
                questions.append({
                    "question": f"How does this relate to LTE network monitoring and performance measurement?",
                    "answer": f"For network monitoring: {' '.join(content_lines[:3])}"
                })
            
            # Type 4: Troubleshooting questions
            if any(term in text for term in ['troubleshooting', 'performance', 'issue', 'error']):
                questions.append({
                    "question": f"What troubleshooting guidance does this Ericsson documentation provide?",
                    "answer": f"The troubleshooting guidance includes: {' '.join(content_lines[:3])}"
                })
            
            # Add a comprehensive question for each chunk
            questions.append({
                "question": f"Summarize the key information from this Ericsson LTE documentation section.",
                "answer": f"This section covers: {' '.join(content_lines[:5])}"
            })
            
            # Add the best questions for this chunk
            for q in questions[:2]:  # Take best 2 questions per chunk
                qa_pairs.append({
                    "question": q["question"],
                    "answer": q["answer"],
                    "source": f"chunk_{i}",
                    "category": "ericsson_lte_technical",
                    "metadata": chunk.get('metadata', {})
                })
        
        logger.info(f"✅ Created {len(qa_pairs)} QA pairs")
        return qa_pairs
        
    def format_for_mlx(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format QA pairs for MLX training (ChatML format)."""
        mlx_data = []
        
        for qa in qa_pairs:
            # ChatML format for instruction tuning
            text = f"Question: {qa['question']} Answer: {qa['answer']}"
            
            mlx_data.append({
                "text": text,
                "metadata": qa.get('metadata', {})
            })
        
        logger.info(f"✅ Formatted {len(mlx_data)} examples for MLX")
        return mlx_data
        
    def split_dataset(self, data: List[Dict[str, Any]], train_ratio: float = 0.7, valid_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train/validation/test sets."""
        total = len(data)
        
        # Shuffle data
        random.shuffle(data)
        
        # Calculate splits
        train_size = int(total * train_ratio)
        valid_size = int(total * valid_ratio)
        
        train_data = data[:train_size]
        valid_data = data[train_size:train_size + valid_size]
        test_data = data[train_size + valid_size:]
        
        logger.info(f"📊 Dataset split: {len(train_data)} train, {len(valid_data)} valid, {len(test_data)} test")
        return train_data, valid_data, test_data
    
    def deduplicate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple deduplication based on answer content similarity."""
        if not dataset:
            return dataset
        
        logger.info(f"🔍 Deduplicating {len(dataset)} entries...")
        
        seen = set()
        unique_data = []
        
        for item in dataset:
            # Create a more specific hash based on answer content
            answer = item.get('answer', '').strip().lower()
            question = item.get('question', '').strip().lower()
            
            # Use a combination of question type and unique answer content
            # Extract meaningful content words (skip common prefixes)
            answer_words = answer.replace('this section covers:', '').replace('the documentation explains:', '').replace('for network monitoring:', '').strip()
            question_type = question.split('?')[0][-20:] if '?' in question else question[-20:]
            
            # Create hash from question type + answer content (skip first 50 chars to avoid common prefixes)
            if len(answer_words) > 50:
                key = f"{question_type}_{answer_words[50:150]}"
            else:
                key = f"{question_type}_{answer_words}"
            
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        
        logger.info(f"📉 Deduplication: {len(dataset)} → {len(unique_data)} entries")
        return unique_data
        
    def save_jsonl(self, data: List[Dict[str, Any]], filepath: Path):
        """Save data to JSONL format."""
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        logger.info(f"💾 Saved {len(data)} examples to {filepath}")
        
    def generate_large_mlx_dataset(self, cache_file: str, output_dir: str, num_pairs: int = 800) -> Dict[str, Any]:
        """Generate large MLX dataset from cached chunks."""
        logger.info("🚀 Creating Large MLX Dataset from Cached Chunks")
        
        # Paths
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set random seed for reproducibility
        random.seed(42)
        
        try:
            # Load chunks
            chunks = self.load_cached_chunks(cache_file)
            
            # Create QA pairs (targeting num_pairs examples)
            qa_pairs = self.create_qa_pairs(chunks, num_pairs=num_pairs)
            
            # Format for MLX
            mlx_data = self.format_for_mlx(qa_pairs)
            
            # Split dataset
            train_data, valid_data, test_data = self.split_dataset(mlx_data)
            
            # Save splits
            self.save_jsonl(train_data, output_path / "train.jsonl")
            self.save_jsonl(valid_data, output_path / "valid.jsonl") 
            self.save_jsonl(test_data, output_path / "test.jsonl")
            
            # Save summary
            summary = {
                "total_examples": len(mlx_data),
                "train_examples": len(train_data),
                "valid_examples": len(valid_data),
                "test_examples": len(test_data),
                "source_chunks": len(chunks),
                "domain": "ericsson_lte_telecommunications",
                "format": "chatML_instruction_tuning",
                "generation_method": "large_mlx_generator"
            }
            
            with open(output_path / "dataset_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"\n🎉 Large MLX Dataset Creation Complete!")
            logger.info(f"📁 Output directory: {output_path}")
            logger.info(f"📊 Total examples: {len(mlx_data)}")
            logger.info(f"🎯 Ready for MLX fine-tuning!")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating large MLX dataset: {e}")
            raise
    
    def generate_large_mlx_dataset_from_chunks(self, chunks: List[Dict[str, Any]], output_dir: str, num_pairs: int = 800) -> Dict[str, Any]:
        """Generate large MLX dataset from loaded chunks."""
        logger.info(f"🚀 Starting efficient MLX dataset generation from {len(chunks)} chunks...")
        
        try:
            # Use the loaded chunks directly
            dataset = self.create_qa_pairs(chunks)
            
            if not dataset:
                raise ValueError("No valid Q&A pairs could be created from chunks")
            
            # Deduplicate
            unique_dataset = self.deduplicate_dataset(dataset)
            logger.info(f"🔄 Deduplication: {len(dataset)} → {len(unique_dataset)} pairs")
            
            # Apply size limit
            if len(unique_dataset) > num_pairs:
                logger.info(f"📊 Limiting dataset to {num_pairs} highest quality pairs")
                # Sort by quality score and take top examples
                unique_dataset.sort(key=lambda x: x.get('metadata', {}).get('quality_score', 0.5), reverse=True)
                unique_dataset = unique_dataset[:num_pairs]
            
            # Convert to MLX format
            mlx_dataset = self.format_for_mlx(unique_dataset)
            
            # Split into train/val/test
            train_data, val_data, test_data = self.split_dataset(mlx_dataset)
            
            # Save datasets
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save training data
            train_file = output_path / "mlx_train.jsonl"
            with open(train_file, 'w', encoding='utf-8') as f:
                for item in train_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Save validation data
            val_file = output_path / "mlx_valid.jsonl"
            with open(val_file, 'w', encoding='utf-8') as f:
                for item in val_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Save test data
            test_file = output_path / "mlx_test.jsonl"
            with open(test_file, 'w', encoding='utf-8') as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Create summary report
            summary = {
                "total_examples": len(mlx_dataset),
                "train_examples": len(train_data),
                "valid_examples": len(val_data),
                "test_examples": len(test_data),
                "source_chunks": len(chunks),
                "utilization_rate": f"{len(chunks)/len(chunks)*100:.1f}%",
                "domain": self.config.dataset_context,
                "format": "ChatML for MLX",
                "output_dir": str(output_path),
                "files_created": [str(train_file), str(val_file), str(test_file)]
            }
            
            logger.info(f"✅ Dataset generation completed: {len(mlx_dataset)} examples from {len(chunks)} chunks")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error generating MLX dataset from chunks: {e}")
            raise


class FineTuningDatasetGenerator:
    """Generate comprehensive fine-tuning datasets from Flow4 chunks."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        try:
            self.deduplicator = DatasetDeduplicator() 
        except ImportError:
            logger.warning("DatasetDeduplicator not available")
            self.deduplicator = None
        
        # Initialize multimodal generator if enabled
        self.multimodal_generator = None
        if self.config.enable_multimodal:
            multimodal_config = self._create_multimodal_config()
            self.multimodal_generator = MultimodalDatasetGenerator(multimodal_config)
        
    def load_chunks(self) -> List[Dict[str, Any]]:
        """Load chunks from the Flow4 output directory or cached file."""
        chunks_path = Path(self.config.chunks_dir)
        
        # Try to load from cached file first (more efficient)
        cache_files = list(chunks_path.glob("chunk_cache_*.json"))
        if cache_files:
            logger.info(f"📋 Found cached chunks file: {cache_files[0]}")
            try:
                with open(cache_files[0], 'r', encoding='utf-8') as f:
                    chunks = json.load(f)
                
                if self.config.max_chunks:
                    chunks = chunks[:self.config.max_chunks]
                    
                logger.info(f"✅ Loaded {len(chunks)} chunks from cache")
                return chunks
            except Exception as e:
                logger.warning(f"Failed to load cached chunks: {e}, falling back to individual files")
        
        # Fallback to individual chunk files
        chunk_files = sorted(chunks_path.glob("chunk_*.json"))
        
        if self.config.max_chunks:
            chunk_files = chunk_files[:self.config.max_chunks]
        
        chunks = []
        for chunk_file in chunk_files:
            try:
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                    chunks.append(chunk_data)
            except Exception as e:
                logger.warning(f"Failed to load {chunk_file}: {e}")
        
        logger.info(f"✅ Loaded {len(chunks)} chunks from {self.config.chunks_dir}")
        return chunks
    
    def extract_technical_terms(self, text: str) -> List[str]:
        """Extract technical terms and acronyms from text."""
        # Simple extraction of likely technical terms
        
        # Find acronyms (2-6 uppercase letters)
        acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
        
        # Find technical terms with numbers
        tech_terms = re.findall(r'\b[A-Za-z]+\d+[A-Za-z]*\b', text)
        
        # Find terms with hyphens (often technical)
        hyphenated = re.findall(r'\b[A-Za-z]+-[A-Za-z]+\b', text)
        
        return list(set(acronyms + tech_terms + hyphenated))
    
    def generate_qa_pairs_from_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple Q&A pairs from a single chunk."""
        text = chunk.get("text", "").strip()
        enriched_text = chunk.get("metadata", {}).get("enriched_text", text)
        headings = chunk.get("metadata", {}).get("docling_headings", [])
        chunk_id = chunk.get("id", "unknown")
        
        if len(text) < 50:  # Skip very short chunks
            return []
        
        qa_pairs = []
        
        # Extract technical terms for focused questions
        tech_terms = self.extract_technical_terms(text)
        
        # 1. Section Overview Question (if we have headings)
        if headings:
            heading = headings[0]
            qa_pairs.append({
                "instruction": f"What does the section '{heading}' cover?",
                "input": "",
                "output": text[:400] + ("..." if len(text) > 400 else ""),
                "metadata": {
                    "type": "section_overview",
                    "chunk_id": chunk_id,
                    "heading": heading,
                    "context": self.config.dataset_context
                }
            })
        
        # 2. Technical Definition Question
        if tech_terms:
            term = tech_terms[0]  # Pick the first technical term
            qa_pairs.append({
                "instruction": f"Explain what {term} means in this context.",
                "input": f"Context: {enriched_text[:300]}...",
                "output": self._extract_definition_or_explanation(text, term),
                "metadata": {
                    "type": "technical_definition", 
                    "chunk_id": chunk_id,
                    "term": term,
                    "context": self.config.dataset_context
                }
            })
        
        # 3. Factual Question about content
        key_sentence = self._extract_key_sentence(text)
        if key_sentence:
            question = self._generate_factual_question(key_sentence)
            qa_pairs.append({
                "instruction": question,
                "input": "",
                "output": key_sentence,
                "metadata": {
                    "type": "factual_qa",
                    "chunk_id": chunk_id,
                    "context": self.config.dataset_context
                }
            })
        
        # 4. RAG-style question with context
        if self.config.include_rag_format:
            qa_pairs.append({
                "instruction": "Based on the provided technical documentation, answer the question.",
                "input": f"Documentation: {enriched_text[:500]}...\n\nQuestion: What are the key technical details mentioned?",
                "output": self._summarize_key_points(text),
                "metadata": {
                    "type": "rag_qa",
                    "chunk_id": chunk_id,
                    "context": self.config.dataset_context
                }
            })
        
        return qa_pairs[:self.config.qa_pairs_per_chunk]
    
    def _extract_definition_or_explanation(self, text: str, term: str) -> str:
        """Extract definition or explanation of a term from text."""
        sentences = text.split('. ')
        
        # Find sentences containing the term
        relevant_sentences = [s for s in sentences if term.lower() in s.lower()]
        
        if relevant_sentences:
            return relevant_sentences[0].strip() + "."
        else:
            # Fallback to first sentence
            return sentences[0].strip() + "." if sentences else "Definition not available in this context."
    
    def _extract_key_sentence(self, text: str) -> str:
        """Extract a key sentence from the text."""
        sentences = text.split('. ')
        
        # Prefer sentences with technical indicators
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['feature', 'enables', 'provides', 'supports', 'configuration']):
                return sentence.strip() + "."
        
        # Fallback to first substantial sentence
        for sentence in sentences:
            if len(sentence.strip()) > 30:
                return sentence.strip() + "."
        
        return ""
    
    def _generate_factual_question(self, sentence: str) -> str:
        """Generate a factual question about a sentence."""
        sentence_lower = sentence.lower()
        
        if "feature" in sentence_lower:
            return "What feature is described here?"
        elif "configuration" in sentence_lower or "setting" in sentence_lower:
            return "What configuration details are mentioned?"
        elif "protocol" in sentence_lower or "procedure" in sentence_lower:
            return "What protocol or procedure is being described?"
        elif "parameter" in sentence_lower:
            return "What parameters are specified?"
        else:
            return "What technical information is provided?"
    
    def _summarize_key_points(self, text: str) -> str:
        """Create a summary of key technical points."""
        sentences = text.split('. ')
        
        # Select up to 3 most informative sentences
        key_sentences = []
        for sentence in sentences[:5]:  # Look at first 5 sentences
            if len(sentence.strip()) > 20 and any(word in sentence.lower() for word in 
                ['feature', 'configuration', 'parameter', 'protocol', 'specification', 'enables', 'provides']):
                key_sentences.append(sentence.strip())
                if len(key_sentences) >= 3:
                    break
        
        if key_sentences:
            return '. '.join(key_sentences) + "."
        else:
            return text[:200] + "..." if len(text) > 200 else text
    
    def create_chat_format_dataset(self, qa_pairs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert Q&A pairs to chat format suitable for fine-tuning."""
        chat_data = []
        
        for qa in qa_pairs:
            # Standard instruction-following format
            chat_entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a helpful AI assistant specializing in {self.config.dataset_context}. Provide clear, accurate, and technical answers based on your knowledge."
                    },
                    {
                        "role": "user", 
                        "content": qa["instruction"] + (f"\n\n{qa['input']}" if qa.get("input") else "")
                    },
                    {
                        "role": "assistant",
                        "content": qa["output"]
                    }
                ],
                "metadata": qa.get("metadata", {})
            }
            chat_data.append(chat_entry)
        
        return chat_data
    
    def deduplicate_dataset(self, dataset: List[Dict[str, Any]], keys: List[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Deduplicate the dataset using Flow4's deduplication utilities."""
        if not self.deduplicator:
            logger.warning("⚠️  Deduplication not available, returning original dataset")
            return dataset, {"status": "skipped", "reason": "deduplication not available"}
        
        logger.info(f"🔍 Deduplicating {len(dataset)} entries...")
        
        # Default keys to check for duplication
        if keys is None:
            keys = ["instruction", "output"]
        
        # Run deduplication
        deduplicated, report = self.deduplicator.deduplicate_comprehensive(
            dataset, 
            keys=keys, 
            strategy=self.config.dedupe_strategy
        )
        
        logger.info(f"📉 Deduplication: {len(dataset)} → {len(deduplicated)} entries")
        return deduplicated, report
    
    def generate_complete_dataset(self) -> Dict[str, Any]:
        """Generate the complete fine-tuning dataset."""
        logger.info(f"🚀 Generating fine-tuning dataset from chunks...")
        
        # Load chunks
        chunks = self.load_chunks()
        if not chunks:
            raise ValueError("No chunks found!")
        
        # Generate Q&A pairs
        all_qa_pairs = []
        for i, chunk in enumerate(chunks):
            qa_pairs = self.generate_qa_pairs_from_chunk(chunk)
            all_qa_pairs.extend(qa_pairs)
            
            if i % 100 == 0:
                logger.info(f"  Processed {i+1}/{len(chunks)} chunks...")
        
        logger.info(f"✅ Generated {len(all_qa_pairs)} Q&A pairs from {len(chunks)} chunks")
        
        # Deduplicate if enabled
        if self.config.enable_deduplication:
            all_qa_pairs, dedup_report = self.deduplicate_dataset(all_qa_pairs)
        else:
            dedup_report = {"status": "disabled"}
        
        # Create different dataset formats
        results = {
            "instruction_dataset": all_qa_pairs,
            "deduplication_report": dedup_report,
            "statistics": {
                "total_chunks": len(chunks),
                "total_qa_pairs": len(all_qa_pairs),
                "dataset_context": self.config.dataset_context
            }
        }
        
        # Add chat format if requested
        if self.config.include_chat_format:
            chat_data = self.create_chat_format_dataset(all_qa_pairs)
            results["chat_dataset"] = chat_data
        
        # Generate multimodal datasets if enabled
        if self.config.enable_multimodal and self.multimodal_generator:
            logger.info("🎨 Generating multimodal datasets...")
            try:
                multimodal_results = self.multimodal_generator.generate_multimodal_dataset(
                    chunks, 
                    str(output_path / "multimodal"),
                    "multimodal_dataset"
                )
                results["multimodal_datasets"] = multimodal_results
                logger.info(f"✅ Generated multimodal datasets: {multimodal_results.get('datasets', {})}")
            except Exception as e:
                logger.error(f"❌ Failed to generate multimodal datasets: {e}")
                results["multimodal_error"] = str(e)
        
        return results
    
    def clean_text_for_jsonl(self, text: str) -> str:
        """Clean text to prevent JSON parsing issues and improve quality."""
        if not text:
            return ""
        
        # Remove or replace problematic characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)  # Remove control characters
        text = re.sub(r'\\', '\\\\', text)  # Escape backslashes
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        # Limit length to prevent overly long sequences
        if len(text) > 2000:
            text = text[:2000] + "..."
        
        return text

    def validate_jsonl_entry(self, prompt: str, completion: str) -> bool:
        """Validate that prompt and completion are suitable for JSONL output."""
        # Skip if either field is too short
        if len(prompt) < 10 or len(completion) < 20:
            return False
            
        # Check for reasonable content
        if not prompt.strip() or not completion.strip():
            return False
            
        return True

    def save_jsonl_format(self, dataset: List[Dict[str, Any]], output_file: Path):
        """Save dataset in JSONL format compatible with MLX fine-tuning."""
        cleaned_data = []
        skipped = 0
        
        for item in dataset:
            # Convert to MLX format: prompt + completion
            prompt = item.get("instruction", "")
            if item.get("input"):
                prompt += f"\n\n{item['input']}"
            
            completion = item.get("output", "")
            
            # Clean the fields
            prompt = self.clean_text_for_jsonl(prompt)
            completion = self.clean_text_for_jsonl(completion)
            
            # Validate entry
            if not self.validate_jsonl_entry(prompt, completion):
                skipped += 1
                continue
            
            mlx_entry = {
                "prompt": prompt,
                "completion": completion
            }
            
            # Test that it can be serialized properly
            try:
                json.dumps(mlx_entry, ensure_ascii=False)
                cleaned_data.append(mlx_entry)
            except (TypeError, ValueError) as e:
                logger.warning(f"Skipping entry due to serialization error: {e}")
                skipped += 1
                continue
        
        # Write cleaned data
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in cleaned_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        if skipped > 0:
            logger.info(f"⚠️  Cleaned and skipped {skipped} problematic entries during JSONL conversion")
        
        logger.info(f"✅ Saved {len(cleaned_data)} clean JSONL entries to {output_file}")

    def create_mlx_dataset_format(self, dataset: List[Dict[str, Any]], output_dir: Path) -> str:
        """Create MLX-compatible dataset with train/valid/test splits."""
        # Convert to MLX format (simple text completion)
        mlx_data = []
        skipped = 0
        
        for item in dataset:
            prompt = item.get("instruction", "")
            if item.get("input"):
                prompt += f"\n\n{item['input']}"
            
            completion = item.get("output", "")
            
            # Clean the fields
            prompt = self.clean_text_for_jsonl(prompt)
            completion = self.clean_text_for_jsonl(completion)
            
            # Validate entry
            if not self.validate_jsonl_entry(prompt, completion):
                skipped += 1
                continue
            
            # Format: prompt + completion as single text for MLX
            text = f"{prompt}\n\n{completion}"
            
            mlx_entry = {"text": text}
            
            # Test serialization
            try:
                json.dumps(mlx_entry, ensure_ascii=False)
                mlx_data.append(mlx_entry)
            except (TypeError, ValueError):
                skipped += 1
                continue
        
        if not mlx_data:
            raise ValueError("No valid data for MLX format conversion")
        
        # Shuffle and create splits
        random.seed(42)
        random.shuffle(mlx_data)
        
        # Split ratios
        val_split = 0.1
        test_split = 0.1
        
        total_len = len(mlx_data)
        test_idx = int(total_len * (1 - test_split))
        valid_idx = int(test_idx * (1 - val_split / (1 - test_split)))
        
        train_data = mlx_data[:valid_idx]
        valid_data = mlx_data[valid_idx:test_idx]
        test_data = mlx_data[test_idx:]
        
        # Create MLX dataset directory
        mlx_dir = output_dir / "mlx_dataset"
        mlx_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in MLX format
        datasets = [
            ("train.jsonl", train_data),
            ("valid.jsonl", valid_data), 
            ("test.jsonl", test_data)
        ]
        
        for filename, data_split in datasets:
            file_path = mlx_dir / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                for item in data_split:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            logger.info(f"📁 Created {filename}: {len(data_split)} examples")
        
        if skipped > 0:
            logger.info(f"⚠️  Skipped {skipped} problematic entries during MLX conversion")
        
        logger.info(f"🚀 Created MLX dataset: {mlx_dir} (Total: {len(mlx_data)} examples)")
        return str(mlx_dir)
    
    def save_sharegpt_jsonl_format(self, chat_dataset: List[Dict[str, Any]], output_file: Path):
        """Save chat dataset in ShareGPT JSONL format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            for chat_entry in chat_dataset:
                # Convert messages format to ShareGPT conversations format
                messages = chat_entry.get("messages", [])
                conversations = []
                
                for msg in messages:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    # Map roles to ShareGPT format
                    if role == "system":
                        from_role = "system"
                    elif role == "user":
                        from_role = "human"
                    elif role == "assistant":
                        from_role = "gpt"
                    else:
                        from_role = role
                    
                    conversations.append({
                        "from": from_role,
                        "value": content
                    })
                
                sharegpt_entry = {
                    "conversations": conversations
                }
                
                # Add metadata if available
                if "metadata" in chat_entry:
                    sharegpt_entry["metadata"] = chat_entry["metadata"]
                
                f.write(json.dumps(sharegpt_entry, ensure_ascii=False) + '\n')
    
    def save_datasets(self, datasets: Dict[str, Any]):
        """Save the generated datasets to files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save instruction dataset (JSON)
        instruction_file = output_dir / "instruction_dataset.json"
        with open(instruction_file, 'w', encoding='utf-8') as f:
            json.dump(datasets["instruction_dataset"], f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved instruction dataset: {instruction_file}")
        
        # Save instruction dataset (JSONL for MLX compatibility)
        if self.config.include_jsonl_format:
            instruction_jsonl_file = output_dir / "instruction_dataset.jsonl"
            self.save_jsonl_format(datasets["instruction_dataset"], instruction_jsonl_file)
            logger.info(f"💾 Saved instruction dataset (JSONL): {instruction_jsonl_file}")
        
        # Save chat dataset if available
        if "chat_dataset" in datasets:
            chat_file = output_dir / "chat_dataset.json"
            with open(chat_file, 'w', encoding='utf-8') as f:
                json.dump(datasets["chat_dataset"], f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved chat dataset: {chat_file}")
            
            # Save chat dataset in JSONL ShareGPT format
            chat_jsonl_file = output_dir / "chat_dataset.jsonl"
            self.save_sharegpt_jsonl_format(datasets["chat_dataset"], chat_jsonl_file)
            logger.info(f"💾 Saved chat dataset (ShareGPT JSONL): {chat_jsonl_file}")
        
        # Save statistics and reports
        stats_file = output_dir / "dataset_statistics.json"
        stats = {
            "statistics": datasets["statistics"],
            "deduplication_report": datasets["deduplication_report"],
            "config": {
                "chunks_dir": self.config.chunks_dir,
                "max_chunks": self.config.max_chunks,
                "enable_deduplication": self.config.enable_deduplication,
                "dedupe_strategy": self.config.dedupe_strategy,
                "qa_pairs_per_chunk": self.config.qa_pairs_per_chunk,
                "dataset_context": self.config.dataset_context
            }
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"📊 Saved statistics: {stats_file}")
        
        # Create train/validation splits
        self.create_train_val_splits(datasets["instruction_dataset"], output_dir)
        
        # Create MLX-compatible dataset format
        if self.config.include_mlx_format:
            try:
                mlx_dir = self.create_mlx_dataset_format(datasets["instruction_dataset"], output_dir)
                logger.info(f"🚀 Created MLX-ready dataset: {mlx_dir}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to create MLX dataset format: {e}")
        
        # Try to generate large MLX dataset if cached chunks are available and current dataset is small
        chunks_path = Path(self.config.chunks_dir)
        cache_files = list(chunks_path.glob("chunk_cache_*.json"))
        current_size = len(datasets["instruction_dataset"])
        
        if cache_files and current_size < 500:  # Only if current dataset is small
            logger.info(f"🔍 Small dataset detected ({current_size} examples), attempting large MLX dataset generation...")
            try:
                large_mlx_generator = LargeMLXDatasetGenerator()
                large_mlx_dir = output_dir / "large_mlx_dataset"
                
                # Calculate target size based on available chunks
                with open(cache_files[0], 'r') as f:
                    cached_chunks = json.load(f)
                target_pairs = min(1000, len(cached_chunks) * 2)  # Target more examples
                
                large_summary = large_mlx_generator.generate_large_mlx_dataset(
                    str(cache_files[0]), 
                    str(large_mlx_dir),
                    num_pairs=target_pairs
                )
                datasets["large_mlx_summary"] = large_summary
                logger.info(f"🎉 Generated large MLX dataset with {large_summary['total_examples']} examples (improvement: {large_summary['total_examples']/current_size:.1f}x)")
            except Exception as e:
                logger.warning(f"⚠️  Failed to generate large MLX dataset: {e}")
        elif current_size >= 500:
            logger.info(f"✅ Current dataset size ({current_size}) is already substantial, skipping large MLX generation")
    
    def create_train_val_splits(self, dataset: List[Dict[str, Any]], output_dir: Path):
        """Create train/validation splits for the dataset."""
        # Shuffle the dataset
        random.seed(42)
        shuffled = dataset.copy()
        random.shuffle(shuffled)
        
        # Create splits (80% train, 20% validation)
        split_idx = int(len(shuffled) * 0.8)
        train_data = shuffled[:split_idx]
        val_data = shuffled[split_idx:]
        
        # Save JSON splits
        train_file = output_dir / "train_dataset.json"
        val_file = output_dir / "validation_dataset.json"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📚 Created train split: {len(train_data)} samples → {train_file}")
        logger.info(f"📚 Created validation split: {len(val_data)} samples → {val_file}")
        
        # Save JSONL splits for MLX compatibility
        if self.config.include_jsonl_format:
            train_jsonl_file = output_dir / "train_dataset.jsonl"
            val_jsonl_file = output_dir / "validation_dataset.jsonl"
            
            self.save_jsonl_format(train_data, train_jsonl_file)
            self.save_jsonl_format(val_data, val_jsonl_file)
            
            logger.info(f"📚 Created train split (JSONL): {len(train_data)} samples → {train_jsonl_file}")
            logger.info(f"📚 Created validation split (JSONL): {len(val_data)} samples → {val_jsonl_file}")
    
    def _create_multimodal_config(self) -> MultimodalDatasetConfig:
        """Create multimodal dataset configuration from main config."""
        return MultimodalDatasetConfig(
            include_llava_format="llava" in self.config.multimodal_formats,
            include_sharegpt_format="sharegpt" in self.config.multimodal_formats,
            include_chatml_format="chatml" in self.config.multimodal_formats,
            image_text_strategy="interleaved" if self.config.multimodal_strategy == "enhanced" else "inline",
            table_text_strategy="inline",
            conversations_per_chunk=min(self.config.qa_pairs_per_chunk, 2),
            dataset_context=self.config.dataset_context,
            domain_expertise="telecommunications and 5G networks",
            generate_image_descriptions=self.config.include_image_analysis,
            preserve_table_structure=self.config.include_table_analysis,
            require_multimodal_content=self.config.multimodal_strategy == "comprehensive"
        )


class DatasetComparator:
    """Compare and analyze dataset quality."""
    
    def analyze_dataset_quality(self, data: dict, dataset_name: str):
        """Analyze dataset quality metrics."""
        training_data = data.get('training_data', [])
        
        if not training_data:
            logger.info(f"{dataset_name}: No training data found")
            return
            
        logger.info(f"\n=== {dataset_name} Analysis ===")
        logger.info(f"Total examples: {len(training_data)}")
        
        # Analyze instructions
        instructions = [item.get('instruction', '') for item in training_data]
        unique_instructions = len(set(instructions))
        logger.info(f"Unique instructions: {unique_instructions} ({unique_instructions/len(instructions)*100:.1f}% unique)")
        
        # Analyze outputs
        outputs = [item.get('output', '') for item in training_data]
        avg_output_length = sum(len(output) for output in outputs) / len(outputs)
        logger.info(f"Average output length: {avg_output_length:.0f} characters")
        
        # Check for data leakage (output == input)
        leakage_count = 0
        for item in training_data:
            inp = item.get('input', '')
            out = item.get('output', '')
            if inp and inp.strip() == out.strip():
                leakage_count += 1
        
        logger.info(f"Data leakage instances: {leakage_count} ({leakage_count/len(training_data)*100:.1f}%)")
        
        # Quality scores (if available)
        if 'quality_stats' in data:
            stats = data['quality_stats']
            logger.info(f"Average quality score: {stats.get('avg_quality_score', 'N/A')}")
            logger.info(f"High quality examples (≥3.0): {stats.get('high_quality_count', 'N/A')}")
        
        # Analyze instruction diversity
        instruction_types = {
            'explain': sum(1 for i in instructions if 'explain' in i.lower()),
            'describe': sum(1 for i in instructions if 'describe' in i.lower()),
            'what': sum(1 for i in instructions if i.lower().startswith('what')),
            'how': sum(1 for i in instructions if 'how' in i.lower()),
            'summarize': sum(1 for i in instructions if 'summarize' in i.lower())
        }
        
        logger.info(f"Instruction type distribution: {instruction_types}")
        
        # Sample quality examples
        if training_data:
            logger.info(f"\nSample instruction: {instructions[0]}")
            logger.info(f"Sample output: {outputs[0][:100]}...")
    
    def compare_datasets(self, dataset_paths: List[Path]):
        """Compare multiple datasets."""
        logger.info("=" * 60)
        logger.info("FINE-TUNING DATASET OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        
        for path in dataset_paths:
            if path.exists():
                with open(path, 'r') as f:
                    data = json.load(f)
                self.analyze_dataset_quality(data, path.stem.upper())