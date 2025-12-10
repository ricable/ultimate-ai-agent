"""Simple HTML to Markdown converter fallback when docling is not available."""

import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
from bs4 import BeautifulSoup

from ..utils.logging import get_logger
from ..utils.config import DoclingConfig
from .cleaner import HTMLCleaner

logger = get_logger(__name__)

# Check for BeautifulSoup
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None


class SimpleHTMLConverter:
    """Simple HTML to Markdown converter using BeautifulSoup."""
    
    def __init__(self, config: DoclingConfig, disable_filtering: bool = False):
        """Initialize the simple converter.
        
        Args:
            config: Docling configuration
            disable_filtering: Whether to disable HTML content filtering
        """
        self.config = config
        self.disable_filtering = disable_filtering
        self.html_cleaner = HTMLCleaner()
        
        if not HAS_BS4:
            logger.error("BeautifulSoup not available. Install with: pip install beautifulsoup4")
    
    def convert_file(self, file_path: str, output_dir: str) -> Optional[str]:
        """Convert a single HTML file to Markdown.
        
        Args:
            file_path: Path to input file
            output_dir: Output directory for converted file
            
        Returns:
            Path to converted Markdown file or None if failed
        """
        if not HAS_BS4:
            logger.error("BeautifulSoup not available")
            return None
        
        # Check if file should be skipped based on content
        if not self.disable_filtering:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    html_content = f.read()
                
                if self.html_cleaner.should_skip_file(html_content):
                    logger.info(f"Skipping file based on content filter: {os.path.basename(file_path)}")
                    return None
            except Exception as e:
                logger.warning(f"Could not perform content filtering on {file_path}: {e}")
                # Continue with conversion if filtering fails
        
        try:
            input_path = Path(file_path)
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate output filename
            output_file = output_path / f"{input_path.stem}.md"
            
            logger.info(f"Converting {input_path.name} to Markdown...")
            
            # Read HTML file
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Convert to markdown
            markdown_content = self._html_to_markdown(html_content)
            
            # Save markdown file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            logger.info(f"Successfully converted {input_path.name}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error converting {file_path}: {e}")
            return None
    
    def _html_to_markdown(self, html_content: str) -> str:
        """Convert HTML to Markdown using BeautifulSoup.
        
        Args:
            html_content: HTML content
            
        Returns:
            Markdown content
        """
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        markdown_lines = []
        
        # Process elements
        for element in soup.find_all():
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(element.name[1])
                heading = '#' * level + ' ' + element.get_text().strip()
                markdown_lines.append(heading)
                markdown_lines.append('')
            
            elif element.name == 'p':
                text = element.get_text().strip()
                if text:
                    markdown_lines.append(text)
                    markdown_lines.append('')
            
            elif element.name in ['ul', 'ol']:
                for li in element.find_all('li'):
                    text = li.get_text().strip()
                    if text:
                        prefix = '- ' if element.name == 'ul' else '1. '
                        markdown_lines.append(prefix + text)
                markdown_lines.append('')
            
            elif element.name == 'table':
                table_md = self._convert_table_to_markdown(element)
                if table_md:
                    markdown_lines.append(table_md)
                    markdown_lines.append('')
            
            elif element.name in ['div', 'section', 'article']:
                text = element.get_text().strip()
                if text and len(text) > 50:  # Only include substantial content
                    # Clean up whitespace
                    text = re.sub(r'\s+', ' ', text)
                    markdown_lines.append(text)
                    markdown_lines.append('')
        
        # If no structured content found, get all text
        if not markdown_lines:
            text = soup.get_text()
            # Clean up whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r'[ \t]+', ' ', text)
            markdown_lines = text.split('\n')
        
        # Join and clean up
        markdown = '\n'.join(markdown_lines)
        
        # Clean up excessive newlines
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()
    
    def _convert_table_to_markdown(self, table_element) -> str:
        """Convert HTML table to Markdown table.
        
        Args:
            table_element: BeautifulSoup table element
            
        Returns:
            Markdown table string
        """
        rows = []
        
        # Get header row
        header_row = table_element.find('tr')
        if header_row:
            headers = []
            for th in header_row.find_all(['th', 'td']):
                headers.append(th.get_text().strip())
            
            if headers:
                rows.append('| ' + ' | '.join(headers) + ' |')
                rows.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        # Get data rows
        for tr in table_element.find_all('tr')[1:]:  # Skip header row
            cells = []
            for td in tr.find_all(['td', 'th']):
                cells.append(td.get_text().strip())
            
            if cells:
                rows.append('| ' + ' | '.join(cells) + ' |')
        
        return '\n'.join(rows) if rows else ''
    
    def batch_convert(
        self, 
        file_paths: List[str], 
        output_dir: str, 
        num_workers: int = 4
    ) -> Tuple[List[str], List[str]]:
        """Convert multiple files.
        
        Args:
            file_paths: List of input file paths
            output_dir: Output directory
            num_workers: Number of parallel workers (ignored in simple version)
            
        Returns:
            Tuple of (successful_files, failed_files)
        """
        if not HAS_BS4:
            logger.error("BeautifulSoup not available")
            return [], file_paths
        
        successful_files = []
        failed_files = []
        
        logger.info(f"Converting {len(file_paths)} files using simple converter...")
        
        for file_path in file_paths:
            try:
                result = self.convert_file(file_path, output_dir)
                if result:
                    successful_files.append(result)
                    logger.info(f"✓ Converted: {os.path.basename(file_path)}")
                else:
                    failed_files.append(file_path)
                    logger.warning(f"✗ Failed: {os.path.basename(file_path)}")
            except Exception as e:
                failed_files.append(file_path)
                logger.error(f"✗ Error converting {os.path.basename(file_path)}: {e}")
        
        logger.info(f"Simple conversion complete: {len(successful_files)} successful, {len(failed_files)} failed")
        return successful_files, failed_files
    
    def is_available(self) -> bool:
        """Check if the simple converter is available.
        
        Returns:
            True if BeautifulSoup is available, False otherwise
        """
        return HAS_BS4