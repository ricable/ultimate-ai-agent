"""Main document processing pipeline orchestration."""

import os
import time
import json
import glob
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from .converter import DocumentConverter
from .chunker import DocumentChunker  
from .cleaner import HTMLCleaner, MarkdownCleaner
from .document_cleaner import DocumentCleaner
from ..utils.config import PipelineConfig, create_output_structure
from ..utils.logging import get_logger

# Optional imports with graceful fallback
try:
    from .mlx_finetuner import MLXFineTuner
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    MLXFineTuner = None

logger = get_logger(__name__)


class DocumentPipeline:
    """Main document processing pipeline that orchestrates all components."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the document pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self.converter = DocumentConverter(self.config.docling, self.config.disable_filtering)
        self.chunker = DocumentChunker(self.config.docling)
        self.cleaner = HTMLCleaner()
        self.markdown_cleaner = MarkdownCleaner()
        self.document_cleaner = DocumentCleaner(self.config)
        
        # Initialize optional components
        self.mlx_finetuner = None
        
        if HAS_MLX:
            try:
                from ..utils.config import MLXConfig
                mlx_config = MLXConfig()
                self.mlx_finetuner = MLXFineTuner(mlx_config)
            except Exception as e:
                logger.warning(f"Failed to initialize MLXFineTuner: {e}")
        
        # Track processing statistics
        self.stats = {
            "start_time": None,
            "end_time": None,
            "total_files": 0,
            "processed_files": 0,
            "skipped_files": 0,
            "markdown_files": 0,
            "chunks_created": 0,
            "errors": []
        }
    
    def extract_zip(self, zip_path: str, extract_dir: str) -> List[str]:
        """Extract files from ZIP archive.
        
        Args:
            zip_path: Path to ZIP file
            extract_dir: Directory to extract files to
            
        Returns:
            List of extracted file paths
        """
        logger.info(f"Extracting {zip_path} to {extract_dir}")
        
        try:
            os.makedirs(extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Filter files if exclusion pattern is provided
                if self.config.pattern_exclude:
                    file_list = [f for f in file_list if self.config.pattern_exclude not in f]
                
                # Extract files
                for file in file_list:
                    zip_ref.extract(file, extract_dir)
            
            # Get list of extracted files
            extracted_files = glob.glob(os.path.join(extract_dir, "**", "*"), recursive=True)
            extracted_files = [f for f in extracted_files if os.path.isfile(f)]
            
            logger.info(f"Extracted {len(extracted_files)} files")
            return extracted_files
        
        except Exception as e:
            error_msg = f"Error extracting ZIP: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return []
    
    def find_documents(self, directory: str) -> Tuple[List[str], List[str]]:
        """Find HTML and PDF files in directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            Tuple of (html_files, pdf_files)
        """
        print(f"\n[SEARCH] Searching for documents in: {directory}")
        logger.info(f"Finding documents in {directory}")
        
        # Find HTML files
        html_files = glob.glob(
            os.path.join(directory, "**", self.config.html_pattern), 
            recursive=True
        )
        
        # Find PDF files
        pdf_files = glob.glob(
            os.path.join(directory, "**", self.config.pdf_pattern), 
            recursive=True
        )
        
        # Apply exclusion pattern if provided
        if self.config.pattern_exclude:
            html_files = [f for f in html_files if self.config.pattern_exclude not in f]
            pdf_files = [f for f in pdf_files if self.config.pattern_exclude not in f]
        
        # Apply file limit if specified
        all_files = html_files + pdf_files
        if self.config.max_files and self.config.max_files > 0:
            original_count = len(all_files)
            all_files = all_files[:self.config.max_files]
            
            # Redistribute between html and pdf
            html_files = [f for f in all_files if f.endswith('.html')]
            pdf_files = [f for f in all_files if f.endswith('.pdf')]
            
            logger.info(f"Limited to {len(all_files)} files (found {original_count} total)")
        
        print(f"[SEARCH] ✓ Found {len(html_files)} HTML files and {len(pdf_files)} PDF files")
        if html_files:
            print(f"[SEARCH] HTML files sample: {[os.path.basename(f) for f in html_files[:3]]}{'...' if len(html_files) > 3 else ''}")
        if pdf_files:
            print(f"[SEARCH] PDF files sample: {[os.path.basename(f) for f in pdf_files[:3]]}{'...' if len(pdf_files) > 3 else ''}")
        logger.info(f"Found {len(html_files)} HTML files and {len(pdf_files)} PDF files")
        return html_files, pdf_files
    
    def _strip_yaml_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content.
        
        Args:
            content: Markdown content that may contain frontmatter
            
        Returns:
            Content with frontmatter removed
        """
        lines = content.split('\n')
        if lines and lines[0].strip() == '---':
            # Find the closing ---
            for i in range(1, len(lines)):
                if lines[i].strip() == '---':
                    # Return content after the closing ---
                    return '\n'.join(lines[i+1:]).lstrip()
        return content
    
    def _clean_markdown_content(self, content: str, source_filename: str) -> str:
        """Clean individual markdown content for concatenation.
        
        Args:
            content: Raw markdown content
            source_filename: Original filename for context
            
        Returns:
            Cleaned content ready for concatenation
        """
        # Remove YAML frontmatter
        content = self._strip_yaml_frontmatter(content)
        
        # Remove the duplicate title that matches the filename
        lines = content.split('\n')
        cleaned_lines = []
        
        file_stem = os.path.splitext(source_filename)[0]
        skip_next_empty = False
        
        for line in lines:
            # Skip title lines that match the filename
            if line.strip() == f"# {file_stem}" or line.strip() == f"# {file_stem.replace('_', ' ')}":
                skip_next_empty = True
                continue
            
            # Skip empty lines immediately after removed titles
            if skip_next_empty and not line.strip():
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def concatenate_markdown_files(self, markdown_files: List[str], output_path: str) -> str:
        """Concatenate multiple Markdown files into one clean document.
        
        Args:
            markdown_files: List of markdown file paths
            output_path: Path for combined markdown file
            
        Returns:
            Path to combined markdown file
        """
        print(f"[COMBINE] Concatenating {len(markdown_files)} Markdown files")
        logger.info(f"Concatenating {len(markdown_files)} Markdown files")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        all_content = []
        processed_count = 0
        
        # Process files in sorted order
        for markdown_path in sorted(markdown_files):
            try:
                with open(markdown_path, 'r', encoding='utf-8') as infile:
                    raw_content = infile.read()
                
                filename = os.path.basename(markdown_path)
                print(f"[COMBINE] [{processed_count + 1}/{len(markdown_files)}] Processing: {filename}")
                
                # Clean the individual file content
                cleaned_content = self._clean_markdown_content(raw_content, filename)
                
                if cleaned_content.strip():
                    all_content.append(cleaned_content)
                    print(f"[COMBINE] ✓ Added {len(cleaned_content)} characters from {filename}")
                else:
                    print(f"[COMBINE] ⚠ Skipped empty content from {filename}")
                
                processed_count += 1
                
            except Exception as e:
                error_msg = f"Error reading {markdown_path}: {str(e)}"
                print(f"[COMBINE] ✗ Error processing {os.path.basename(markdown_path)}: {str(e)}")
                logger.error(error_msg)
                self.stats["errors"].append(error_msg)
        
        # Combine all content with clean separators
        print(f"[COMBINE] Joining {len(all_content)} content sections...")
        
        # Join content with proper spacing for optimal chunking
        combined_content = '\n\n\n'.join(all_content)
        
        # Ensure clean start and end
        combined_content = combined_content.strip()
        
        # Write the raw combined content  
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.write(combined_content)
        
        print(f"[COMBINE] ✓ Raw combination complete: {len(combined_content)} characters")
        
        # Apply comprehensive document cleaning for optimal chunking
        print(f"[COMBINE] Applying comprehensive document cleaning...")
        logger.info("Applying document cleaning for LLM optimization")
        try:
            print(f"[COMBINE] Pre-cleaning: {os.path.getsize(output_path)} bytes")
            
            # Use the new document cleaner for better results
            cleaned_path = self.document_cleaner.clean_document(
                input_path=output_path,
                output_path=output_path  # Clean in place
            )
            
            print(f"[COMBINE] Post-cleaning: {os.path.getsize(cleaned_path)} bytes")
            print(f"[COMBINE] ✓ Comprehensive document cleaning completed")
            logger.info("Document cleaning completed")
        except Exception as e:
            print(f"[COMBINE] ✗ Error during document cleaning: {e}")
            logger.error(f"Error during document cleaning: {e}")
            # Fallback to original markdown cleaner
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    combined_content = f.read()
                
                cleaned_content = self.markdown_cleaner.clean_markdown_comprehensive(
                    combined_content,
                    remove_excessive_newlines=True,
                    optimize_for_llm=True
                )
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                
                print(f"[COMBINE] ✓ Fallback markdown cleaning completed")
                logger.info("Fallback markdown cleaning completed")
            except Exception as fallback_error:
                print(f"[COMBINE] ✗ Fallback cleaning also failed: {fallback_error}")
                logger.error(f"Fallback cleaning failed: {fallback_error}")
                # Continue with uncleaned version
        
        print(f"[COMBINE] ✓ Clean combined document ready for chunking: {os.path.basename(output_path)}")
        logger.info(f"Combined document saved to {output_path}")
        return output_path
    
    def create_rag_datasets(self, chunks: List, output_dir: str) -> Tuple[str, str]:
        """Create RAG and fine-tuning datasets from chunks.
        
        Args:
            chunks: List of document chunks
            output_dir: Directory to save datasets
            
        Returns:
            Tuple of (rag_dataset_path, finetune_dataset_path)
        """
        logger.info(f"Creating enhanced RAG datasets from {len(chunks)} chunks")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Create enhanced RAG datasets with multiple formats
            enhanced_datasets = self.chunker.create_enhanced_rag_dataset(chunks, output_dir)
            rag_dataset_path = enhanced_datasets.get("json")
            
            # Create enhanced fine-tuning datasets
            finetune_datasets = self.create_enhanced_finetune_datasets(chunks, output_dir)
            finetune_dataset_path = finetune_datasets.get("instruction_response")
            
            # Create dataset validation report
            validation_report = self.validate_datasets(chunks, output_dir)
            
            logger.info(f"Enhanced RAG datasets created: {list(enhanced_datasets.keys())}")
            logger.info(f"Enhanced fine-tuning datasets created: {list(finetune_datasets.keys())}")
            logger.info(f"Dataset validation report saved")
            
            return rag_dataset_path, finetune_dataset_path
        
        except Exception as e:
            error_msg = f"Error creating RAG datasets: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return None, None
    
    def create_enhanced_finetune_datasets(self, chunks: List, output_dir: str) -> Dict[str, str]:
        """Create enhanced fine-tuning datasets with proper formatting.
        
        Args:
            chunks: List of document chunks
            output_dir: Output directory
            
        Returns:
            Dictionary mapping format names to file paths
        """
        results = {}
        
        # 1. Instruction-Response format (OpenAI/Anthropic style)
        instruction_data = self._create_instruction_response_dataset(chunks)
        instruction_path = os.path.join(output_dir, "finetune_instruction_response.json")
        with open(instruction_path, 'w', encoding='utf-8') as f:
            json.dump(instruction_data, f, indent=2, ensure_ascii=False)
        results["instruction_response"] = instruction_path
        
        # 2. JSONL format for streaming training
        jsonl_data = self._create_finetune_jsonl(chunks)
        jsonl_path = os.path.join(output_dir, "finetune_dataset.jsonl")
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for item in jsonl_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        results["jsonl"] = jsonl_path
        
        # 3. Simple format (for compatibility)
        simple_data = {
            "data": [
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": {
                        "source": chunk.source,
                        "tokens": chunk.metadata.get("tokens", len(chunk.text.split()))
                    }
                }
                for chunk in chunks
            ],
            "total_chunks": len(chunks),
            "config": {
                "chunk_size": self.config.docling.chunk_size,
                "chunk_overlap": self.config.docling.chunk_overlap
            }
        }
        
        simple_path = os.path.join(output_dir, "finetune_dataset.json")
        with open(simple_path, 'w', encoding='utf-8') as f:
            json.dump(simple_data, f, indent=2, ensure_ascii=False)
        results["simple"] = simple_path
        
        return results
    
    def _create_instruction_response_dataset(self, chunks: List) -> Dict[str, Any]:
        """Create instruction-response format dataset for LLM fine-tuning.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Instruction-response dataset
        """
        import random
        training_data = []
        
        # Diverse instruction templates
        instruction_templates = [
            "Explain the content about the following topic.",
            "Provide information about what is described in this section.",
            "Summarize the key points from this documentation.",
            "What does this section cover?",
            "Describe the information contained in this excerpt.",
            "What are the main concepts discussed in this section?",
            "What technical details are provided in this excerpt?",
            "Explain the implementation described in this section.",
        ]
        
        for chunk in chunks:
            # Skip very short chunks that don't add value
            if not chunk.text or len(chunk.text.strip()) < 50:
                continue
                
            # Create 1-2 high-quality examples per chunk
            selected_templates = random.sample(instruction_templates, min(2, len(instruction_templates)))
            
            for template in selected_templates:
                # Create meaningful input/output pairs
                if len(chunk.text) > 200:
                    # Use a meaningful excerpt as input, full text as output
                    input_text = self._create_meaningful_input(chunk.text)
                    output_text = chunk.text
                else:
                    # For short chunks, use topic as input
                    input_text = self._extract_topic_from_chunk(chunk.text)
                    output_text = chunk.text
                
                training_data.append({
                    "instruction": template,
                    "input": input_text,
                    "output": output_text,
                    "metadata": {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id,
                        "type": "instruction_response"
                    }
                })
        
        return {
            "training_data": training_data,
            "total_examples": len(training_data),
            "format": "instruction_response",
            "description": "Instruction-response pairs for LLM fine-tuning",
            "version": "2.0.0"
        }
    
    def _create_meaningful_input(self, text: str) -> str:
        """Create meaningful input that doesn't leak the full output."""
        # Extract first sentence or key topic
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if sentences:
            return sentences[0] + "."
        return text[:100] + "..." if len(text) > 100 else text
    
    def _extract_topic_from_chunk(self, text: str) -> str:
        """Extract the main topic/title from chunk text."""
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            # Look for headings or first meaningful line
            if line and not line.startswith('#'):
                if len(line) < 100:  # Likely a title/topic
                    return line
        
        # Fallback to first sentence
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        return sentences[0] + "." if sentences else text[:50] + "..."
    
    def _create_finetune_jsonl(self, chunks: List) -> List[Dict[str, Any]]:
        """Create JSONL format data for streaming fine-tuning.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of JSONL-ready dictionaries
        """
        jsonl_data = []
        
        for chunk in chunks:
            # Create prompt-completion pairs
            prompts = [
                f"Question: What is this section about?\nAnswer:",
                f"Content summary:",
                f"Explain:"
            ]
            
            for prompt in prompts:
                jsonl_data.append({
                    "prompt": prompt,
                    "completion": chunk.text,
                    "metadata": {
                        "source": chunk.source,
                        "chunk_id": chunk.chunk_id
                    }
                })
        
        return jsonl_data
    
    def validate_datasets(self, chunks: List, output_dir: str) -> Dict[str, Any]:
        """Validate created datasets and generate quality report.
        
        Args:
            chunks: List of document chunks
            output_dir: Output directory
            
        Returns:
            Validation report
        """
        report = {
            "validation_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "total_chunks": len(chunks),
            "validation_results": {}
        }
        
        # Basic statistics
        chunk_sizes = [len(chunk.text) for chunk in chunks]
        word_counts = [len(chunk.text.split()) for chunk in chunks]
        
        report["validation_results"]["statistics"] = {
            "total_characters": sum(chunk_sizes),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(chunk_sizes) / len(chunks) if chunks else 0,
            "min_chunk_size": min(chunk_sizes) if chunks else 0,
            "max_chunk_size": max(chunk_sizes) if chunks else 0,
            "avg_word_count": sum(word_counts) / len(chunks) if chunks else 0
        }
        
        # Quality checks
        empty_chunks = [chunk for chunk in chunks if not chunk.text.strip()]
        
        report["validation_results"]["quality_checks"] = {
            "empty_chunks": len(empty_chunks),
            "quality_score": (len(chunks) - len(empty_chunks)) / len(chunks) * 100 if chunks else 100
        }
        
        # Source distribution
        source_counts = {}
        for chunk in chunks:
            source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1
        
        report["validation_results"]["source_distribution"] = source_counts
        
        # Save validation report
        report_path = os.path.join(output_dir, "dataset_validation_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return report
    
    def run_mlx_finetuning(
        self, 
        dataset_path: str, 
        output_dir: str,
        chat_mode: bool = False
    ) -> Optional[str]:
        """Run MLX fine-tuning on generated datasets.
        
        Args:
            dataset_path: Path to training dataset
            output_dir: Output directory for fine-tuned model
            chat_mode: Whether to enable interactive chat after training
            
        Returns:
            Path to fine-tuned model or None if failed
        """
        if not HAS_MLX or not self.mlx_finetuner:
            logger.error("MLX not available. Install with: pip install mlx>=0.12.0 mlx-lm>=0.8.0")
            return None
        
        try:
            logger.info(f"Starting MLX fine-tuning on dataset: {dataset_path}")
            
            # Run fine-tuning
            model_path = self.mlx_finetuner.finetune(
                dataset_path=dataset_path,
                output_dir=output_dir
            )
            
            if model_path:
                logger.info(f"MLX fine-tuning completed: {model_path}")
                
                if chat_mode:
                    logger.info("Starting interactive chat mode...")
                    self.mlx_finetuner.interactive_chat(model_path)
            else:
                logger.error("MLX fine-tuning failed")
            
            return model_path
        
        except Exception as e:
            error_msg = f"Error during MLX fine-tuning: {str(e)}"
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            return None
    
    def generate_summary(self, output_paths: Dict[str, Path]) -> Dict[str, Any]:
        """Generate pipeline execution summary.
        
        Args:
            output_paths: Dictionary of output paths
            
        Returns:
            Summary dictionary
        """
        processing_time = (self.stats["end_time"] - self.stats["start_time"]) if self.stats["end_time"] and self.stats["start_time"] else 0
        
        summary = {
            "pipeline_version": "Flow4-v0.2.0",
            "execution_time": time.strftime('%Y-%m-%d %H:%M:%S'),
            "processing_time_seconds": processing_time,
            "statistics": {
                "total_files_found": self.stats["total_files"],
                "files_processed": self.stats["processed_files"],
                "files_skipped": self.stats["skipped_files"],
                "markdown_files_created": self.stats["markdown_files"],
                "chunks_created": self.stats["chunks_created"],
                "success_rate": self.stats["processed_files"] / max(self.stats["total_files"], 1) * 100
            },
            "configuration": {
                "docling_features": {
                    "accelerator": self.config.docling.with_accelerator,
                    "extract_tables": self.config.docling.extract_tables,
                    "extract_figures": self.config.docling.extract_figures,
                    "multimodal": self.config.docling.multimodal,
                    "custom_convert": self.config.docling.custom_convert
                },
                "chunking": {
                    "chunk_size": self.config.docling.chunk_size,
                    "chunk_overlap": self.config.docling.chunk_overlap,
                    "split_on_headings": self.config.docling.split_on_headings,
                    "tokenizer": self.config.docling.tokenizer
                },
                "pipeline": {
                    "num_workers": self.config.num_workers,
                    "max_files": self.config.max_files,
                    "pattern_exclude": self.config.pattern_exclude
                }
            },
            "output_structure": {str(k): str(v) for k, v in output_paths.items()},
            "errors": self.stats["errors"]
        }
        
        return summary
    
    def run(
        self, 
        input_path: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete document processing pipeline.
        
        Args:
            input_path: Path to input ZIP file or directory (overrides config)
            output_dir: Output directory (overrides config)
            
        Returns:
            Pipeline execution summary
        """
        self.stats["start_time"] = time.time()
        
        # Use provided paths or fall back to config
        input_path = input_path or self.config.input_path
        output_dir = output_dir or self.config.output_dir
        
        if not input_path:
            raise ValueError("Input path must be provided either in config or as parameter")
        
        print(f"\n{'='*60}")
        print(f"[PIPELINE] Starting Flow4 Document Processing Pipeline")
        print(f"[PIPELINE] Input: {input_path}")
        print(f"[PIPELINE] Output: {output_dir}")
        print(f"{'='*60}")
        logger.info(f"Starting document processing pipeline")
        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_dir}")
        
        # Create output directory structure
        config_dict = self.config.__dict__.copy()
        config_dict['output_dir'] = output_dir
        output_paths = create_output_structure(
            PipelineConfig(**config_dict)
        )
        
        try:
            # Phase 1: Extract or find files
            print(f"\n[PHASE 1] File Discovery and Extraction")
            if input_path.lower().endswith('.zip'):
                print(f"[PHASE 1] Extracting ZIP archive...")
                extract_dir = self.config.extract_dir or output_paths["extracted"]
                extracted_files = self.extract_zip(input_path, str(extract_dir))
                search_dir = str(extract_dir)
                print(f"[PHASE 1] ✓ Extracted {len(extracted_files) if extracted_files else 0} files")
            else:
                print(f"[PHASE 1] Using directory input: {input_path}")
                search_dir = input_path
            
            # Phase 2: Find documents to process
            print(f"\n[PHASE 2] Document Discovery")
            html_files, pdf_files = self.find_documents(search_dir)
            all_files = html_files + pdf_files
            self.stats["total_files"] = len(all_files)
            
            if not all_files:
                print(f"[PHASE 2] ✗ No documents found to process")
                logger.warning("No documents found to process")
                return self.generate_summary(output_paths)
            
            print(f"[PHASE 2] ✓ Total files to process: {len(all_files)}")
            
            # Phase 3: Convert documents to Markdown
            print(f"\n[PHASE 3] Document Conversion")
            print(f"[PHASE 3] Converting {len(all_files)} documents to Markdown...")
            logger.info("Converting documents to Markdown...")
            
            if not self.converter.is_available():
                logger.error("Document converter not available. Install docling with: pip install docling")
                print(f"[PHASE 3] ✗ Document converter not available")
                return self.generate_summary(output_paths)
            
            markdown_files, skipped_files = self.converter.batch_convert(
                all_files, 
                str(output_paths["markdown"]), 
                self.config.num_workers
            )
            
            self.stats["processed_files"] = len(markdown_files)
            self.stats["skipped_files"] = len(skipped_files)
            self.stats["markdown_files"] = len(markdown_files)
            
            if not markdown_files:
                print(f"[PHASE 3] ✗ No files were successfully converted")
                logger.warning("No files were successfully converted")
                return self.generate_summary(output_paths)
            
            print(f"[PHASE 3] ✓ Successfully converted {len(markdown_files)} files, skipped {len(skipped_files)} files")
            
            # Phase 4: Concatenate Markdown files
            print(f"\n[PHASE 4] Document Concatenation")
            print(f"[PHASE 4] Concatenating {len(markdown_files)} Markdown files...")
            logger.info("Concatenating Markdown files...")
            combined_path = str(output_paths["combined"] / "combined_document.md")
            combined_path = self.concatenate_markdown_files(markdown_files, combined_path)
            print(f"[PHASE 4] ✓ Combined document created: {os.path.basename(combined_path)}")
            
            # Phase 5: Chunk the combined document
            print(f"\n[PHASE 5] Document Chunking")
            print(f"[PHASE 5] Chunking combined document...")
            logger.info("Chunking documents...")
            chunks = self.chunker.chunk_file(combined_path, use_docling=True)
            self.stats["chunks_created"] = len(chunks)
            print(f"[PHASE 5] ✓ Created {len(chunks)} chunks")
            
            # Save chunks
            print(f"[PHASE 5] Saving chunks...")
            self.chunker.save_chunks(chunks, str(output_paths["chunks"]))
            print(f"[PHASE 5] ✓ Chunks saved to: {output_paths['chunks']}")
            
            # Phase 6: Create RAG datasets
            print(f"\n[PHASE 6] RAG Dataset Creation")
            print(f"[PHASE 6] Creating RAG datasets from {len(chunks)} chunks...")
            logger.info("Creating RAG datasets...")
            rag_dataset_path, finetune_dataset_path = self.create_rag_datasets(
                chunks, 
                str(output_paths["rag"])
            )
            if rag_dataset_path:
                print(f"[PHASE 6] ✓ RAG dataset: {os.path.basename(rag_dataset_path)}")
            if finetune_dataset_path:
                print(f"[PHASE 6] ✓ Fine-tuning dataset: {os.path.basename(finetune_dataset_path)}")
            
            # Generate and save summary
            self.stats["end_time"] = time.time()
            summary = self.generate_summary(output_paths)
            
            summary_path = output_paths["base"] / "pipeline_summary.json"
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            processing_time = self.stats["end_time"] - self.stats["start_time"]
            print(f"\n{'='*60}")
            print(f"[PIPELINE] ✓ COMPLETED SUCCESSFULLY")
            print(f"[PIPELINE] Processing time: {processing_time:.2f} seconds")
            print(f"[PIPELINE] Files processed: {self.stats['processed_files']}/{self.stats['total_files']}")
            print(f"[PIPELINE] Chunks created: {self.stats['chunks_created']}")
            print(f"[PIPELINE] Summary: {summary_path}")
            print(f"{'='*60}")
            logger.info(f"Pipeline completed successfully in {processing_time:.2f} seconds")
            logger.info(f"Summary saved to {summary_path}")
            
            return summary
        
        except Exception as e:
            error_msg = f"Pipeline execution failed: {str(e)}"
            print(f"\n[PIPELINE] ✗ PIPELINE FAILED: {error_msg}")
            logger.error(error_msg)
            self.stats["errors"].append(error_msg)
            self.stats["end_time"] = time.time()
            return self.generate_summary(output_paths)