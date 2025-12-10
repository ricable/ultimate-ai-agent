"""Intelligent document chunking with semantic understanding."""

import json
import os
import re
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..utils.logging import get_logger
from ..utils.config import DoclingConfig

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata."""
    
    chunk_id: str
    text: str
    source: str
    position: int
    tokens: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "position": self.position,
            "tokens": self.tokens,
            "metadata": self.metadata
        }


class DocumentChunker:
    """Intelligent document chunker with semantic understanding."""
    
    def __init__(self, config: DoclingConfig):
        """Initialize the document chunker.
        
        Args:
            config: Docling configuration
        """
        self.config = config
        self._init_tokenizer()
        
    def _init_tokenizer(self):
        """Initialize tokenizer for token counting."""
        try:
            if self.config.tokenizer == "cl100k_base":
                import tiktoken
                self.tokenizer = tiktoken.get_encoding("cl100k_base")
                self.token_counter = self._tiktoken_count
            else:
                # Fallback to simple word-based counting
                self.tokenizer = None
                self.token_counter = self._simple_count
                
        except ImportError:
            logger.warning("tiktoken not available, using simple word counting")
            self.tokenizer = None
            self.token_counter = self._simple_count
    
    def _tiktoken_count(self, text: str) -> int:
        """Count tokens using tiktoken."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return self._simple_count(text)
    
    def _simple_count(self, text: str) -> int:
        """Simple word-based token counting."""
        return len(text.split())
    
    def chunk_file(self, file_path: str, use_docling: bool = True) -> List[DocumentChunk]:
        """Chunk a markdown file into semantic chunks.
        
        Args:
            file_path: Path to markdown file
            use_docling: Whether to use Docling-aware chunking
            
        Returns:
            List of document chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            source = os.path.basename(file_path)
            logger.info(f"Chunking file: {source}")
            
            if use_docling and self.config.enable_semantic_chunking:
                chunks = self._semantic_chunk(content, source)
            else:
                chunks = self._simple_chunk(content, source)
            
            logger.info(f"Created {len(chunks)} chunks from {source}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking file {file_path}: {e}")
            return []
    
    def _semantic_chunk(self, content: str, source: str) -> List[DocumentChunk]:
        """Perform semantic-aware chunking.
        
        Args:
            content: Document content
            source: Source filename
            
        Returns:
            List of semantic chunks
        """
        # Split content into sections based on headers
        sections = self._split_by_headers(content)
        
        chunks = []
        position = 0
        
        for section in sections:
            # Check if section fits in one chunk
            section_tokens = self.token_counter(section)
            
            if section_tokens <= self.config.max_tokens:
                # Section fits in one chunk
                chunk = self._create_chunk(section, source, position)
                chunks.append(chunk)
                position += 1
            else:
                # Section needs to be split further
                section_chunks = self._split_large_section(section, source, position)
                chunks.extend(section_chunks)
                position += len(section_chunks)
        
        # Merge small adjacent chunks if enabled
        if self.config.enable_merge_peers:
            chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _split_by_headers(self, content: str) -> List[str]:
        """Split content by markdown headers while preserving structure.
        
        Args:
            content: Document content
            
        Returns:
            List of sections
        """
        # First, deduplicate content
        content = self._deduplicate_content(content)
        
        sections = []
        current_section = []
        
        lines = content.split('\n')
        
        for line in lines:
            # Skip empty lines at section boundaries
            if not line.strip() and not current_section:
                continue
                
            # Check if line is a header
            if re.match(r'^#{1,6}\s+', line) and self.config.split_on_headings:
                # Save current section if it exists
                if current_section:
                    section_text = '\n'.join(current_section).strip()
                    if section_text:  # Only add non-empty sections
                        sections.append(section_text)
                    current_section = []
                
                # Start new section with header
                if self.config.keep_headings:
                    current_section.append(line)
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append(section_text)
        
        return sections
    
    def _split_large_section(self, section: str, source: str, start_position: int) -> List[DocumentChunk]:
        """Split a large section into smaller chunks.
        
        Args:
            section: Section content
            source: Source filename
            start_position: Starting position number
            
        Returns:
            List of chunks from the section
        """
        chunks = []
        position = start_position
        
        # Try to split by paragraphs first
        paragraphs = self._split_by_paragraphs(section)
        
        current_chunk_text = ""
        
        for paragraph in paragraphs:
            # Check if adding this paragraph would exceed the limit
            potential_text = current_chunk_text + "\n\n" + paragraph if current_chunk_text else paragraph
            potential_tokens = self.token_counter(potential_text)
            
            if potential_tokens <= self.config.max_tokens:
                # Add paragraph to current chunk
                current_chunk_text = potential_text
            else:
                # Save current chunk if it has content
                if current_chunk_text.strip():
                    chunk = self._create_chunk(current_chunk_text, source, position)
                    chunks.append(chunk)
                    position += 1
                
                # Start new chunk with current paragraph
                if self.token_counter(paragraph) <= self.config.max_tokens:
                    current_chunk_text = paragraph
                else:
                    # Paragraph is too large, split by sentences
                    para_chunks = self._split_by_sentences(paragraph, source, position)
                    chunks.extend(para_chunks)
                    position += len(para_chunks)
                    current_chunk_text = ""
        
        # Add final chunk
        if current_chunk_text.strip():
            chunk = self._create_chunk(current_chunk_text, source, position)
            chunks.append(chunk)
        
        return chunks
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraphs.
        
        Args:
            content: Content to split
            
        Returns:
            List of paragraphs
        """
        # Split by double newlines (paragraph breaks)
        paragraphs = re.split(r'\n\s*\n', content)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_by_sentences(self, content: str, source: str, start_position: int) -> List[DocumentChunk]:
        """Split content by sentences for very large paragraphs.
        
        Args:
            content: Content to split
            source: Source filename
            start_position: Starting position number
            
        Returns:
            List of sentence-based chunks
        """
        chunks = []
        position = start_position
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+(?=\s|$)', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        current_chunk_text = ""
        
        for sentence in sentences:
            # Add punctuation back
            if not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            potential_text = current_chunk_text + " " + sentence if current_chunk_text else sentence
            potential_tokens = self.token_counter(potential_text)
            
            if potential_tokens <= self.config.max_tokens:
                current_chunk_text = potential_text
            else:
                # Save current chunk
                if current_chunk_text.strip():
                    chunk = self._create_chunk(current_chunk_text, source, position)
                    chunks.append(chunk)
                    position += 1
                
                # Start new chunk
                current_chunk_text = sentence
        
        # Add final chunk
        if current_chunk_text.strip():
            chunk = self._create_chunk(current_chunk_text, source, position)
            chunks.append(chunk)
        
        return chunks
    
    def _simple_chunk(self, content: str, source: str) -> List[DocumentChunk]:
        """Perform simple sliding window chunking.
        
        Args:
            content: Document content
            source: Source filename
            
        Returns:
            List of simple chunks
        """
        chunks = []
        words = content.split()
        position = 0
        
        i = 0
        while i < len(words):
            # Take chunk_size words
            chunk_words = words[i:i + self.config.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Create chunk
            chunk = self._create_chunk(chunk_text, source, position)
            chunks.append(chunk)
            
            # Move by (chunk_size - overlap) words
            i += max(1, self.config.chunk_size - self.config.chunk_overlap)
            position += 1
        
        return chunks
    
    def _create_chunk(self, text: str, source: str, position: int) -> DocumentChunk:
        """Create a document chunk with metadata.
        
        Args:
            text: Chunk text
            source: Source filename
            position: Position in document
            
        Returns:
            Document chunk
        """
        # Clean and normalize the text before creating chunk
        text = self._clean_chunk_text(text)
        
        chunk_id = str(uuid.uuid4())
        tokens = self.token_counter(text)
        
        # Analyze content type
        content_type = self._analyze_content_type(text)
        
        metadata = {
            "content_type": content_type,
            "quality_score": self._calculate_quality_score(text),
            "has_code": self._has_code_content(text),
            "has_tables": self._has_table_content(text),
            "language": "en",  # Default to English
            "created_at": "2024-01-01",  # Placeholder
        }
        
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text.strip(),
            source=source,
            position=position,
            tokens=tokens,
            metadata=metadata
        )
    
    def _analyze_content_type(self, text: str) -> str:
        """Analyze the type of content in the chunk.
        
        Args:
            text: Chunk text
            
        Returns:
            Content type classification
        """
        text_lower = text.lower()
        
        # Check for specific content types based on keywords and patterns
        if re.search(r'class\s+\w+\s*:', text) or 'class definition' in text_lower:
            return 'class_definition'
        elif re.search(r'enum\s+\w+', text) or 'enumeration' in text_lower:
            return 'enum_definition'
        elif any(kpi in text_lower for kpi in ['kpi', 'indicator', 'measurement', 'metric']):
            if any(calc in text_lower for calc in ['calculate', 'formula', 'equation']):
                return 'kpi_calculation'
            else:
                return 'kpi_definition'
        elif any(proc in text_lower for proc in ['procedure', 'step', 'process', 'algorithm']):
            return 'procedure'
        elif any(config in text_lower for config in ['configuration', 'setup', 'settings', 'parameters']):
            return 'configuration'
        elif any(feat in text_lower for feat in ['feature', 'functionality', 'capability']):
            return 'feature_description'
        else:
            return 'general_content'
    
    def _calculate_quality_score(self, text: str) -> float:
        """Calculate a quality score for the chunk.
        
        Args:
            text: Chunk text
            
        Returns:
            Quality score between 0 and 1
        """
        score = 0.0
        
        # Length factor (prefer medium-length chunks)
        length = len(text)
        if 100 <= length <= 1000:
            score += 0.3
        elif 50 <= length <= 1500:
            score += 0.2
        
        # Structure factor (prefer chunks with headers or lists)
        if re.search(r'^#{1,6}\s+', text, re.MULTILINE):
            score += 0.2
        if re.search(r'^[*-+]\s+', text, re.MULTILINE):
            score += 0.1
        
        # Content density factor
        words = text.split()
        if len(words) > 10:
            score += 0.2
        
        # Technical content factor
        if self.config.technical_content_boost:
            technical_terms = ['system', 'network', 'protocol', 'interface', 'configuration']
            if any(term in text.lower() for term in technical_terms):
                score += 0.2
        
        return min(score, 1.0)
    
    def _deduplicate_content(self, content: str) -> str:
        """Remove duplicate consecutive lines and sections.
        
        Args:
            content: Raw content with potential duplicates
            
        Returns:
            Deduplicated content
        """
        lines = content.split('\n')
        deduplicated_lines = []
        seen_lines = set()
        
        for line in lines:
            line_clean = line.strip()
            # Skip empty lines or very short lines
            if not line_clean or len(line_clean) < 3:
                deduplicated_lines.append(line)
                continue
                
            # For longer lines, check for exact duplicates
            if line_clean not in seen_lines:
                deduplicated_lines.append(line)
                seen_lines.add(line_clean)
            elif line_clean.startswith('#'):
                # Always keep headers even if duplicated
                deduplicated_lines.append(line)
        
        return '\n'.join(deduplicated_lines)
    
    def _clean_chunk_text(self, text: str) -> str:
        """Clean and normalize chunk text for better quality.
        
        Args:
            text: Raw chunk text
            
        Returns:
            Cleaned chunk text
        """
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Fix broken table formatting
        text = self._fix_table_formatting(text)
        
        # Remove orphaned table headers
        text = self._remove_orphaned_headers(text)
        
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        
        return text.strip()
    
    def _fix_table_formatting(self, text: str) -> str:
        """Fix broken table formatting in text.
        
        Args:
            text: Text with potential table fragments
            
        Returns:
            Text with improved table formatting
        """
        lines = text.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check if this looks like a table row
            if '|' in line and line.count('|') >= 2:
                # Look for continuation of table
                table_lines = [line]
                j = i + 1
                
                while j < len(lines) and ('|' in lines[j] or lines[j].strip() == ''):
                    if lines[j].strip():  # Skip empty lines
                        table_lines.append(lines[j])
                    j += 1
                
                # Only keep if we have a proper table (at least 2 rows)
                if len(table_lines) >= 2:
                    fixed_lines.extend(table_lines)
                else:
                    # Convert broken table to regular text
                    for table_line in table_lines:
                        cleaned = table_line.replace('|', '').strip()
                        if cleaned:
                            fixed_lines.append(cleaned)
                
                i = j
            else:
                fixed_lines.append(line)
                i += 1
        
        return '\n'.join(fixed_lines)
    
    def _remove_orphaned_headers(self, text: str) -> str:
        """Remove table headers without corresponding data.
        
        Args:
            text: Text with potential orphaned headers
            
        Returns:
            Text with orphaned headers removed
        """
        lines = text.split('\n')
        filtered_lines = []
        
        orphaned_headers = [
            'Managed Object',
            'Additional Text', 
            'On-site Activities',
            'Link to Remedy Actions',
            'Feature Name',
            'Feature Identity',
            'Value Package Name'
        ]
        
        for line in lines:
            line_clean = line.strip()
            
            # Skip standalone orphaned headers
            if line_clean in orphaned_headers:
                continue
                
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _has_code_content(self, text: str) -> bool:
        """Check if chunk contains code content.
        
        Args:
            text: Chunk text
            
        Returns:
            True if chunk contains code
        """
        # Look for code blocks or code-like patterns
        return bool(re.search(r'```|`[^`]+`|^\s*[a-zA-Z_]\w*\s*=', text, re.MULTILINE))
    
    def _has_table_content(self, text: str) -> bool:
        """Check if chunk contains table content.
        
        Args:
            text: Chunk text
            
        Returns:
            True if chunk contains tables
        """
        # Look for markdown table patterns
        return bool(re.search(r'^\|.*\|\s*$', text, re.MULTILINE))
    
    def _merge_small_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Merge small adjacent chunks to improve quality.
        
        Args:
            chunks: List of chunks to merge
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return chunks
        
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # Check if we should merge
            current_tokens = current_chunk.tokens
            next_tokens = next_chunk.tokens
            combined_tokens = self.token_counter(current_chunk.text + "\n\n" + next_chunk.text)
            
            should_merge = (
                current_tokens < self.config.min_tokens or
                next_tokens < self.config.min_tokens
            ) and combined_tokens <= self.config.max_tokens
            
            if should_merge:
                # Merge chunks
                merged_text = current_chunk.text + "\n\n" + next_chunk.text
                merged_metadata = current_chunk.metadata.copy()
                merged_metadata["merged_from"] = [current_chunk.chunk_id, next_chunk.chunk_id]
                
                current_chunk = DocumentChunk(
                    chunk_id=str(uuid.uuid4()),
                    text=merged_text,
                    source=current_chunk.source,
                    position=current_chunk.position,
                    tokens=combined_tokens,
                    metadata=merged_metadata
                )
            else:
                # Keep current chunk and move to next
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        # Add final chunk
        merged.append(current_chunk)
        
        return merged
    
    def save_chunks(self, chunks: List[DocumentChunk], output_dir: str) -> None:
        """Save chunks to individual JSON files.
        
        Args:
            chunks: List of chunks to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for chunk in chunks:
            chunk_file = output_path / f"chunk_{chunk.position:04d}_{chunk.chunk_id[:8]}.json"
            with open(chunk_file, 'w', encoding='utf-8') as f:
                json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save chunks summary
        summary = {
            "total_chunks": len(chunks),
            "total_tokens": sum(chunk.tokens for chunk in chunks),
            "avg_tokens_per_chunk": sum(chunk.tokens for chunk in chunks) / len(chunks) if chunks else 0,
            "content_types": self._analyze_content_distribution(chunks),
            "quality_stats": self._analyze_quality_distribution(chunks)
        }
        
        summary_file = output_path / "chunks_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_dir}")
    
    def _analyze_content_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, int]:
        """Analyze content type distribution across chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Distribution of content types
        """
        distribution = {}
        for chunk in chunks:
            content_type = chunk.metadata.get("content_type", "unknown")
            distribution[content_type] = distribution.get(content_type, 0) + 1
        return distribution
    
    def _analyze_quality_distribution(self, chunks: List[DocumentChunk]) -> Dict[str, float]:
        """Analyze quality score distribution across chunks.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Quality statistics
        """
        if not chunks:
            return {}
        
        scores = [chunk.metadata.get("quality_score", 0.0) for chunk in chunks]
        return {
            "avg_quality": sum(scores) / len(scores),
            "min_quality": min(scores),
            "max_quality": max(scores),
            "high_quality_chunks": len([s for s in scores if s >= 0.7])
        }
    
    def create_enhanced_rag_dataset(self, chunks: List[DocumentChunk], output_dir: str) -> Dict[str, str]:
        """Create enhanced RAG datasets in multiple formats.
        
        Args:
            chunks: List of document chunks
            output_dir: Output directory
            
        Returns:
            Dictionary mapping format names to file paths
        """
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Standard JSON format
        json_data = {
            "chunks": [chunk.to_dict() for chunk in chunks],
            "metadata": {
                "total_chunks": len(chunks),
                "total_tokens": sum(chunk.tokens for chunk in chunks),
                "content_distribution": self._analyze_content_distribution(chunks),
                "quality_stats": self._analyze_quality_distribution(chunks)
            }
        }
        
        json_file = output_path / "rag_dataset.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        results["json"] = str(json_file)
        
        # 2. JSONL format for streaming
        jsonl_file = output_path / "rag_dataset.jsonl"
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + '\n')
        results["jsonl"] = str(jsonl_file)
        
        # 3. CSV format for analysis
        csv_file = output_path / "rag_dataset.csv"
        with open(csv_file, 'w', encoding='utf-8') as f:
            f.write("chunk_id,text,source,position,tokens,content_type,quality_score\n")
            for chunk in chunks:
                text_escaped = chunk.text.replace('"', '""').replace('\n', ' ')
                content_type = chunk.metadata.get("content_type", "unknown")
                quality_score = chunk.metadata.get("quality_score", 0.0)
                f.write(f'"{chunk.chunk_id}","{text_escaped}","{chunk.source}",{chunk.position},{chunk.tokens},"{content_type}",{quality_score}\n')
        results["csv"] = str(csv_file)
        
        logger.info(f"Created RAG datasets: {list(results.keys())}")
        return results