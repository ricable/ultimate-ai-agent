"""Document cleaner for removing duplicates and improving formatting."""

import re
from pathlib import Path
from typing import Optional

from ..utils.logging import get_logger
from ..utils.config import PipelineConfig

logger = get_logger(__name__)


class DocumentCleaner:
    """Clean combined documents to remove duplicates and improve formatting."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """Initialize the document cleaner.
        
        Args:
            config: Pipeline configuration (optional)
        """
        self.config = config or PipelineConfig()
    
    def clean_document(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Clean a combined document.
        
        Args:
            input_path: Path to input document
            output_path: Path to output document (optional, defaults to input_path)
            
        Returns:
            Path to cleaned document
        """
        if output_path is None:
            output_path = input_path
        
        logger.info(f"Cleaning document: {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply cleaning steps
            content = self._remove_duplicate_sections(content)
            content = self._fix_table_formatting(content)
            content = self._remove_orphaned_headers(content)
            content = self._clean_whitespace(content)
            content = self._fix_markdown_structure(content)
            
            # Save cleaned content
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Cleaned document saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error cleaning document {input_path}: {e}")
            raise
    
    def _remove_duplicate_sections(self, content: str) -> str:
        """Remove duplicate consecutive sections.
        
        Args:
            content: Document content
            
        Returns:
            Content with duplicates removed
        """
        lines = content.split('\n')
        deduplicated = []
        seen_blocks = set()
        current_block = []
        
        for line in lines:
            if line.strip() == '':
                if current_block:
                    block_text = '\n'.join(current_block)
                    block_hash = hash(block_text.strip().lower())
                    
                    if block_hash not in seen_blocks:
                        deduplicated.extend(current_block)
                        seen_blocks.add(block_hash)
                    
                    deduplicated.append('')
                    current_block = []
                else:
                    deduplicated.append('')
            else:
                current_block.append(line)
        
        # Handle final block
        if current_block:
            block_text = '\n'.join(current_block)
            block_hash = hash(block_text.strip().lower())
            if block_hash not in seen_blocks:
                deduplicated.extend(current_block)
        
        return '\n'.join(deduplicated)
    
    def _fix_table_formatting(self, content: str) -> str:
        """Fix broken table formatting.
        
        Args:
            content: Document content
            
        Returns:
            Content with improved table formatting
        """
        lines = content.split('\n')
        fixed_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip broken table content before proper tables
            if 'Table 1 Alarm Summary' in line and '|' not in line:
                # Skip until we find the actual table
                while i < len(lines) and not (lines[i].startswith('|') and lines[i].count('|') >= 3):
                    i += 1
                continue
            
            # Process proper markdown tables
            if line.startswith('|') and line.count('|') >= 3:
                table_lines = [line]
                i += 1
                
                # Collect all table rows
                while i < len(lines):
                    if lines[i].startswith('|') or lines[i].strip() == '':
                        if lines[i].strip():
                            table_lines.append(lines[i])
                        i += 1
                    else:
                        break
                
                # Only keep tables with proper structure
                if len(table_lines) >= 2:
                    fixed_lines.extend(table_lines)
                
                i -= 1  # Adjust for the outer loop increment
            else:
                fixed_lines.append(line)
            
            i += 1
        
        return '\n'.join(fixed_lines)
    
    def _remove_orphaned_headers(self, content: str) -> str:
        """Remove standalone orphaned headers.
        
        Args:
            content: Document content
            
        Returns:
            Content with orphaned headers removed
        """
        lines = content.split('\n')
        filtered_lines = []
        
        orphaned_headers = {
            'Managed Object',
            'Additional Text', 
            'On-site Activities',
            'Link to Remedy Actions',
            'Link to Remedy  Actions',
            'Feature Name',
            'Feature Identity',
            'Value Package Name',
            'Contents'
        }
        
        for i, line in enumerate(lines):
            line_clean = line.strip()
            
            # Skip standalone orphaned headers (not part of tables or proper sections)
            if (line_clean in orphaned_headers and 
                (i == 0 or lines[i-1].strip() == '') and 
                (i == len(lines)-1 or lines[i+1].strip() == '' or not lines[i+1].startswith('|'))):
                continue
            
            # Skip repetitive raw text blocks
            if 'RadioEquipmentClockReference' in line and '|' not in line and line.count('.') > 3:
                continue
                
            filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _clean_whitespace(self, content: str) -> str:
        """Clean up excessive whitespace.
        
        Args:
            content: Document content
            
        Returns:
            Content with cleaned whitespace
        """
        # Remove excessive blank lines
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        # Remove trailing whitespace
        lines = content.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        return '\n'.join(cleaned_lines)
    
    def _fix_markdown_structure(self, content: str) -> str:
        """Fix markdown structure and hierarchy.
        
        Args:
            content: Document content
            
        Returns:
            Content with improved markdown structure
        """
        lines = content.split('\n')
        fixed_lines = []
        
        for line in lines:
            # Fix numbered headers (convert "1 Title" to "## 1. Title")
            if re.match(r'^\d+\s+[A-Z]', line) and not line.startswith('#'):
                number = re.match(r'^(\d+)', line).group(1)
                title = line[len(number):].strip()
                fixed_lines.append(f"## {number}. {title}")
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)