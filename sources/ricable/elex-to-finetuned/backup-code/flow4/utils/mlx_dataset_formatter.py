"""
MLX Dataset Formatter for Flow4

This module addresses all the MLX dataset formatting issues encountered during development
and provides robust JSONL generation for MLX fine-tuning.

Issues addressed:
1. JSON encoding errors with special characters
2. MLX batch size requirements (minimum 4 examples per split)
3. Proper JSONL line-by-line format validation
4. MLX text format requirements
5. Dataset split ratios and validation
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import unicodedata

logger = logging.getLogger(__name__)


@dataclass
class MLXDatasetConfig:
    """Configuration for MLX dataset generation"""
    min_examples_per_split: int = 4  # MLX requirement
    train_ratio: float = 0.7
    valid_ratio: float = 0.2
    test_ratio: float = 0.1
    max_text_length: int = 2048
    min_text_length: int = 10


class MLXDatasetFormatter:
    """
    Robust JSONL dataset formatter for MLX fine-tuning.
    
    Handles all formatting issues encountered during development:
    - JSON encoding problems
    - MLX batch size requirements
    - Text cleaning and validation
    - Proper JSONL format generation
    """
    
    def __init__(self, config: Optional[MLXDatasetConfig] = None):
        self.config = config or MLXDatasetConfig()
        
    def clean_text_for_jsonl(self, text: str) -> str:
        """
        Clean text for JSONL format to prevent JSON encoding errors.
        
        Issues addressed:
        - Control characters that break JSON parsing
        - Unicode normalization
        - Proper escaping of special characters
        - Length validation
        """
        if not text:
            return ""
            
        # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)
        
        # Remove control characters that break JSON
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
        
        # Escape backslashes and quotes properly
        text = text.replace('\\', '\\\\')
        text = text.replace('"', '\\"')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\r')
        text = text.replace('\t', '\\t')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Validate length
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length - 3] + "..."
            logger.warning(f"Text truncated to {self.config.max_text_length} characters")
            
        if len(text) < self.config.min_text_length:
            logger.warning(f"Text too short ({len(text)} chars), may affect training quality")
            
        return text
    
    def validate_jsonl_line(self, line: str) -> bool:
        """
        Validate that a JSONL line is properly formatted.
        
        Issues addressed:
        - JSON parsing errors
        - Missing required fields
        - Invalid JSON structure
        """
        try:
            data = json.loads(line.strip())
            
            # Check for required MLX format
            if "text" not in data:
                logger.error("Missing 'text' field in JSONL line")
                return False
                
            if not isinstance(data["text"], str):
                logger.error("'text' field must be a string")
                return False
                
            if len(data["text"].strip()) == 0:
                logger.error("Empty 'text' field in JSONL line")
                return False
                
            return True
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return False
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def create_mlx_text_entry(self, prompt: str, completion: str) -> Dict[str, str]:
        """
        Create a single MLX text entry from prompt and completion.
        
        MLX format requires:
        {"text": "prompt_text + completion_text"}
        """
        # Clean both prompt and completion
        clean_prompt = self.clean_text_for_jsonl(prompt)
        clean_completion = self.clean_text_for_jsonl(completion)
        
        # Combine with proper formatting
        combined_text = f"{clean_prompt}{clean_completion}"
        
        return {"text": combined_text}
    
    def create_mlx_chat_entry(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Convert chat messages to MLX text format.
        
        Handles OpenAI-style messages and converts to MLX format.
        """
        text_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "user":
                text_parts.append(f"Question: {content}")
            elif role == "assistant":
                text_parts.append(f"Answer: {content}")
            elif role == "system":
                text_parts.append(f"System: {content}")
                
        combined_text = "\n".join(text_parts)
        clean_text = self.clean_text_for_jsonl(combined_text)
        
        return {"text": clean_text}
    
    def split_dataset(self, examples: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Split dataset into train/valid/test ensuring MLX requirements.
        
        Issues addressed:
        - Minimum 4 examples per split for MLX batch requirements
        - Proper ratio handling with small datasets
        - Warning for insufficient data
        """
        total_examples = len(examples)
        
        if total_examples < 12:  # Minimum for 4 examples per split
            logger.warning(f"Dataset has only {total_examples} examples. MLX requires minimum 4 per split.")
            logger.warning("Duplicating examples to meet minimum requirements.")
            
            # Duplicate examples to meet minimum requirements
            while len(examples) < 12:
                examples.extend(examples[:min(len(examples), 12 - len(examples))])
            
            total_examples = len(examples)
        
        # Calculate split sizes
        train_size = max(4, int(total_examples * self.config.train_ratio))
        valid_size = max(4, int(total_examples * self.config.valid_ratio))
        test_size = max(4, total_examples - train_size - valid_size)
        
        # Adjust if total exceeds available examples
        if train_size + valid_size + test_size > total_examples:
            train_size = total_examples - 8  # Leave 8 for valid and test
            valid_size = 4
            test_size = 4
        
        # Split the data
        train_data = examples[:train_size]
        valid_data = examples[train_size:train_size + valid_size]
        test_data = examples[train_size + valid_size:train_size + valid_size + test_size]
        
        logger.info(f"Dataset split: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")
        
        return train_data, valid_data, test_data
    
    def write_jsonl_file(self, data: List[Dict[str, str]], filepath: Path) -> bool:
        """
        Write data to JSONL file with validation.
        
        Issues addressed:
        - File encoding problems
        - Line-by-line validation
        - Atomic writing to prevent corruption
        """
        try:
            # Write to temporary file first
            temp_filepath = filepath.with_suffix('.tmp')
            
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                for i, example in enumerate(data):
                    json_line = json.dumps(example, ensure_ascii=False)
                    
                    # Validate the line before writing
                    if not self.validate_jsonl_line(json_line):
                        logger.error(f"Invalid JSONL line {i}: {json_line[:100]}...")
                        return False
                    
                    f.write(json_line + '\n')
            
            # Move temp file to final location
            temp_filepath.rename(filepath)
            
            logger.info(f"Successfully wrote {len(data)} examples to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing JSONL file {filepath}: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            return False
    
    def validate_jsonl_file(self, filepath: Path) -> bool:
        """
        Validate existing JSONL file for MLX compatibility.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                line_count = 0
                for line_num, line in enumerate(f, 1):
                    if line.strip():  # Skip empty lines
                        if not self.validate_jsonl_line(line):
                            logger.error(f"Invalid line {line_num} in {filepath}")
                            return False
                        line_count += 1
                
                if line_count < self.config.min_examples_per_split:
                    logger.error(f"File {filepath} has only {line_count} examples, need minimum {self.config.min_examples_per_split}")
                    return False
                    
            logger.info(f"JSONL file {filepath} is valid with {line_count} examples")
            return True
            
        except Exception as e:
            logger.error(f"Error validating JSONL file {filepath}: {e}")
            return False
    
    def create_mlx_dataset_from_chunks(self, chunks: List[Dict[str, Any]], output_dir: Path) -> bool:
        """
        Create complete MLX dataset from Flow4 chunks.
        
        This is the main integration point that handles all formatting issues.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate Q&A pairs from chunks
            examples = []
            
            for chunk in chunks:
                # Try multiple possible content fields
                content = chunk.get('content', '') or chunk.get('text', '') or chunk.get('enriched_text', '')
                if isinstance(content, dict):
                    content = content.get('text', '')
                content = content.strip()
                if not content:
                    continue
                
                # Create multiple Q&A pairs per chunk
                qa_pairs = self.generate_qa_pairs_from_chunk(content)
                
                for prompt, completion in qa_pairs:
                    mlx_entry = self.create_mlx_text_entry(prompt, completion)
                    examples.append(mlx_entry)
            
            if not examples:
                logger.error("No valid examples generated from chunks")
                return False
            
            # Split dataset
            train_data, valid_data, test_data = self.split_dataset(examples)
            
            # Write files
            success = True
            success &= self.write_jsonl_file(train_data, output_dir / "train.jsonl")
            success &= self.write_jsonl_file(valid_data, output_dir / "valid.jsonl")
            success &= self.write_jsonl_file(test_data, output_dir / "test.jsonl")
            
            # Validate all files
            success &= self.validate_jsonl_file(output_dir / "train.jsonl")
            success &= self.validate_jsonl_file(output_dir / "valid.jsonl")
            success &= self.validate_jsonl_file(output_dir / "test.jsonl")
            
            if success:
                logger.info(f"Successfully created MLX dataset in {output_dir}")
                return True
            else:
                logger.error("Failed to create valid MLX dataset")
                return False
                
        except Exception as e:
            logger.error(f"Error creating MLX dataset: {e}")
            return False
    
    def generate_qa_pairs_from_chunk(self, content: str) -> List[Tuple[str, str]]:
        """
        Generate Q&A pairs from chunk content.
        
        This is a robust rule-based approach that extracts meaningful
        information from the content for training.
        """
        qa_pairs = []
        
        if len(content) < self.config.min_text_length:
            return qa_pairs
        
        # Clean and extract title/heading if present
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        if not lines:
            return qa_pairs
            
        title = lines[0]
        # Remove markdown formatting and numbering
        title = re.sub(r'^#+\s*', '', title)
        title = re.sub(r'^\d+\.\s*', '', title)
        title = re.sub(r'^\d+\s+', '', title)
        
        # Extract key information
        main_content = '\n'.join(lines[1:]) if len(lines) > 1 else content
        
        # Create more targeted questions based on content
        if "overview" in title.lower() or "introduction" in title.lower():
            questions = [
                f"What is {title}?",
                f"What is the purpose of {title}?",
                f"How does {title} work?",
            ]
        elif "feature" in title.lower() or "optimization" in title.lower():
            questions = [
                f"What does {title} do?",
                f"What are the benefits of {title}?",
                f"How is {title} implemented?",
            ]
        else:
            # Generic questions
            questions = [
                f"What is {title}?",
                f"How does {title} work?",
                f"What are the key aspects of {title}?",
            ]
        
        # Generate answers with appropriate length
        for question in questions[:3]:  # Limit to 3 questions
            # Create more informative answers
            if len(main_content) > 100:
                # Use the most relevant part of the content
                answer = main_content[:800] if len(main_content) > 800 else main_content
            else:
                # Use the full content if it's short
                answer = content[:800] if len(content) > 800 else content
            
            # Clean up the answer
            answer = re.sub(r'\s+', ' ', answer).strip()
            if not answer.endswith('.'):
                answer += "."
            
            prompt = f"Question: {question}\nAnswer:"
            completion = f" {answer}"
            qa_pairs.append((prompt, completion))
        
        # Add a summary question if content is long enough
        if len(content) > 300:
            summary_prompt = f"Question: Summarize the key points about {title}.\nAnswer:"
            # Create a concise summary from the first part
            summary = main_content[:400] if main_content else content[:400]
            summary = re.sub(r'\s+', ' ', summary).strip()
            if not summary.endswith('.'):
                summary += "."
            summary_completion = f" {summary}"
            qa_pairs.append((summary_prompt, summary_completion))
        
        return qa_pairs


def convert_existing_dataset_to_mlx_format(input_file: Path, output_dir: Path) -> bool:
    """
    Convert existing datasets to proper MLX format.
    
    Handles common format conversion issues.
    """
    formatter = MLXDatasetFormatter()
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = []
            
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    entry = json.loads(line)
                    
                    # Handle different input formats
                    if "messages" in entry:
                        # OpenAI chat format
                        mlx_entry = formatter.create_mlx_chat_entry(entry["messages"])
                    elif "prompt" in entry and "completion" in entry:
                        # Prompt-completion format
                        mlx_entry = formatter.create_mlx_text_entry(entry["prompt"], entry["completion"])
                    elif "text" in entry:
                        # Already in MLX format, just clean
                        clean_text = formatter.clean_text_for_jsonl(entry["text"])
                        mlx_entry = {"text": clean_text}
                    else:
                        logger.warning(f"Unknown format in line {line_num}, skipping")
                        continue
                    
                    data.append(mlx_entry)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error in line {line_num}: {e}")
                    continue
        
        if not data:
            logger.error("No valid data found in input file")
            return False
        
        # Split and write
        train_data, valid_data, test_data = formatter.split_dataset(data)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        success = True
        success &= formatter.write_jsonl_file(train_data, output_dir / "train.jsonl")
        success &= formatter.write_jsonl_file(valid_data, output_dir / "valid.jsonl")
        success &= formatter.write_jsonl_file(test_data, output_dir / "test.jsonl")
        
        return success
        
    except Exception as e:
        logger.error(f"Error converting dataset: {e}")
        return False