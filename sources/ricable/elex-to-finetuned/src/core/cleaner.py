"""Document cleaning utilities for HTML and Markdown content."""

import re
from typing import List, Optional
from ..utils.logging import get_logger

logger = get_logger(__name__)


class HTMLCleaner:
    """Cleaner for HTML content before conversion."""
    
    def __init__(self):
        """Initialize the HTML cleaner."""
        self.common_headers = [
            r'<header.*?>.*?</header>',
            r'<nav.*?>.*?</nav>',
            r'<div[^>]*class="[^"]*header[^"]*".*?>.*?</div>',
            r'<div[^>]*id="[^"]*header[^"]*".*?>.*?</div>',
        ]
        
        self.common_footers = [
            r'<footer.*?>.*?</footer>',
            r'<div[^>]*class="[^"]*footer[^"]*".*?>.*?</div>',
            r'<div[^>]*id="[^"]*footer[^"]*".*?>.*?</div>',
        ]
        
        self.common_navigation = [
            r'<nav.*?>.*?</nav>',
            r'<div[^>]*class="[^"]*nav[^"]*".*?>.*?</div>',
            r'<div[^>]*class="[^"]*menu[^"]*".*?>.*?</div>',
            r'<ul[^>]*class="[^"]*nav[^"]*".*?>.*?</ul>',
        ]
        
        self.script_and_style = [
            r'<script.*?>.*?</script>',
            r'<style.*?>.*?</style>',
            r'<link[^>]*rel="stylesheet"[^>]*>',
        ]
        
        # Skip patterns for code documentation
        self.skip_patterns = ["class", "enum", "struct", "derivedDataType", "deriveddatatype", "module", "interface"]
        
        # Legal content patterns to remove
        self.legal_patterns = [
            r"Legal\s*\|.*?(?=\n\n|\Z)",
            r"Copyright\s*©.*?(?=\n\n|\Z)",
            r"©.*?All rights reserved\..*?(?=\n\n|\Z)",
            r"Disclaimer\s*The contents.*?(?=\n\n|\Z)",
            r"This document contains proprietary.*?(?=\n\n|\Z)",
            r"Confidential.*?(?=\n\n|\Z)",
        ]
        
    def should_skip_file(self, html_content: str) -> bool:
        """Determine if an HTML file should be skipped based on content.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            True if file should be skipped, False otherwise
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("BeautifulSoup not available, cannot perform advanced HTML filtering")
            return False
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Check title for skip patterns
        if soup.title and soup.title.string:
            title_text = soup.title.string.strip().lower()
            for pattern in self.skip_patterns:
                if title_text.startswith(pattern.lower()):
                    logger.info(f"Skipping code documentation with title: {title_text}")
                    return True
        
        # Check for title anchors with skip patterns
        title_anchors = soup.find_all('a', attrs={'name': 'TITLE'})
        for anchor in title_anchors:
            if anchor.get_text():
                anchor_text = anchor.get_text().strip().lower()
                for pattern in self.skip_patterns:
                    if pattern.lower() in anchor_text:
                        logger.info(f"Skipping HTML file with '{pattern}' title anchor: {anchor.get_text().strip()}")
                        return True
        
        # Check meta tags with name="TITLE"
        title_meta = soup.find('meta', attrs={'name': 'TITLE'})
        if title_meta and title_meta.get('content'):
            content_text = title_meta.get('content').strip().lower()
            for pattern in self.skip_patterns:
                if pattern.lower() in content_text:
                    logger.info(f"Skipping HTML file with '{pattern}' meta title: {title_meta.get('content')}")
                    return True
        
        return False
    
    def remove_legal_content(self, content: str) -> str:
        """Remove legal disclaimers and copyright notices.
        
        Args:
            content: Input content
            
        Returns:
            Content with legal patterns removed
        """
        for pattern in self.legal_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        return content
    
    def clean_html(self, html_content: str) -> str:
        """Clean HTML content by removing common unwanted elements.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned HTML content
        """
        content = html_content
        
        # Remove scripts and styles
        for pattern in self.script_and_style:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove headers
        for pattern in self.common_headers:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove footers
        for pattern in self.common_footers:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove navigation
        for pattern in self.common_navigation:
            content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove comments
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        
        # Remove legal content
        content = self.remove_legal_content(content)
        
        # Clean up whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        content = content.strip()
        
        return content


class MarkdownCleaner:
    """Cleaner for Markdown content after conversion."""
    
    def __init__(self):
        """Initialize the Markdown cleaner."""
        self.disclaimer_patterns = [
            r'(?i)copyright\s+©.*?all\s+rights\s+reserved',
            r'(?i)disclaimer:.*?(?=\n\n|\n#|\Z)',
            r'(?i)legal\s+notice:.*?(?=\n\n|\n#|\Z)',
            r'(?i)this\s+document.*?proprietary.*?(?=\n\n|\n#|\Z)',
            r'(?i)confidential.*?information.*?(?=\n\n|\n#|\Z)',
        ]
        
        self.footer_patterns = [
            r'(?i)^.*?page\s+\d+\s+of\s+\d+.*?$',
            r'(?i)^.*?document\s+id:.*?$',
            r'(?i)^.*?revision:.*?$',
            r'(?i)^.*?last\s+updated:.*?$',
        ]
        
    def clean_markdown_comprehensive(
        self, 
        content: str, 
        remove_excessive_newlines: bool = True,
        optimize_for_llm: bool = True
    ) -> str:
        """Apply comprehensive cleaning to markdown content.
        
        Args:
            content: Raw markdown content
            remove_excessive_newlines: Whether to clean up excessive newlines
            optimize_for_llm: Whether to optimize for LLM consumption
            
        Returns:
            Cleaned markdown content
        """
        logger.debug("Applying comprehensive markdown cleaning...")
        
        # Remove YAML frontmatter
        content = self._remove_yaml_frontmatter(content)
        
        # Remove disclaimers and legal notices
        for pattern in self.disclaimer_patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE | re.DOTALL)
        
        # Remove footer patterns
        for pattern in self.footer_patterns:
            content = re.sub(pattern, '', content, flags=re.MULTILINE)
        
        # Clean up excessive newlines
        if remove_excessive_newlines:
            content = self._normalize_whitespace(content)
        
        # Optimize for LLM if requested
        if optimize_for_llm:
            content = self._optimize_for_llm(content)
        
        return content.strip()
    
    def _remove_yaml_frontmatter(self, content: str) -> str:
        """Remove YAML frontmatter from markdown content."""
        lines = content.split('\n')
        if lines and lines[0].strip() == '---':
            # Find the closing ---
            for i in range(1, len(lines)):
                if lines[i].strip() == '---':
                    # Return content after the closing ---
                    return '\n'.join(lines[i+1:]).lstrip()
        return content
    
    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace in content."""
        # Replace multiple consecutive newlines with double newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove trailing whitespace from lines
        lines = content.split('\n')
        lines = [line.rstrip() for line in lines]
        content = '\n'.join(lines)
        
        # Remove empty lines at the beginning and end
        content = content.strip()
        
        return content
    
    def _optimize_for_llm(self, content: str) -> str:
        """Optimize content for LLM consumption."""
        # Ensure proper spacing around headers
        content = re.sub(r'\n(#{1,6}\s+)', r'\n\n\1', content)
        content = re.sub(r'(#{1,6}.*?)\n([^\n#])', r'\1\n\n\2', content)
        
        # Ensure proper spacing around lists
        content = re.sub(r'\n([*-+]\s+)', r'\n\n\1', content)
        content = re.sub(r'([*-+].*?)\n([^\n*-+\s])', r'\1\n\n\2', content)
        
        # Ensure proper spacing around tables
        content = re.sub(r'\n(\|.*?\|)', r'\n\n\1', content)
        
        # Clean up any excessive spacing we might have created
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    
    def remove_duplicates(self, content: str, source_filename: str) -> str:
        """Remove duplicate titles and redundant content.
        
        Args:
            content: Markdown content
            source_filename: Original filename for context
            
        Returns:
            Content with duplicates removed
        """
        lines = content.split('\n')
        cleaned_lines = []
        
        file_stem = source_filename.replace('.html', '').replace('.pdf', '')
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
    
    def clean_for_chunking(self, content: str) -> str:
        """Clean content specifically for optimal chunking.
        
        Args:
            content: Markdown content
            
        Returns:
            Content optimized for chunking
        """
        # Ensure sections are properly separated
        content = re.sub(r'\n(#{1,6}\s+)', r'\n\n\1', content)
        
        # Ensure list items are properly grouped
        content = re.sub(r'\n([*-+]\s+)', r'\n\1', content)
        
        # Ensure code blocks are preserved
        content = re.sub(r'\n(```)', r'\n\n\1', content)
        content = re.sub(r'(```)\n', r'\1\n\n', content)
        
        # Clean up excessive whitespace but maintain structure
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        
        return content.strip()