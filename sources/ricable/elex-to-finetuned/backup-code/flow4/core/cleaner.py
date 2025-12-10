"""HTML and Markdown cleaning functionality for document processing optimized for LLM training and RAG."""

import re
from typing import List, Optional, Dict, Tuple
from bs4 import BeautifulSoup

from ..utils.logging import get_logger

logger = get_logger(__name__)


class HTMLCleaner:
    """HTML cleaner for removing unwanted elements and content."""
    
    def __init__(self):
        """Initialize the HTML cleaner."""
        self.legal_patterns = [
            r"Legal\s*\|.*?(?=\n\n|\Z)",
            r"Copyright\s*©.*?(?=\n\n|\Z)",
            r"©.*?All rights reserved\..*?(?=\n\n|\Z)",
            r"Disclaimer\s*The contents.*?(?=\n\n|\Z)",
            r"This document contains proprietary.*?(?=\n\n|\Z)",
            r"Confidential.*?(?=\n\n|\Z)",
        ]
    
    def identify_removal_elements(self, soup: BeautifulSoup) -> List:
        """Identify elements to be removed from HTML.
        
        Args:
            soup: BeautifulSoup parsed HTML document
            
        Returns:
            List of elements to remove
        """
        elements_to_remove = []
        
        # Find header elements
        header_candidates = [
            soup.find('div', id='header-container'),
            soup.find('div', id='header'),
            soup.find('header'),
            soup.find('div', class_=lambda c: c and ('header' in c.lower())),
        ]
        elements_to_remove.extend([elem for elem in header_candidates if elem])
        
        # Find footer elements
        footer_candidates = [
            soup.find('div', id='footer'),
            soup.find('footer'),
            soup.find('div', class_=lambda c: c and ('footer' in c.lower())),
        ]
        elements_to_remove.extend([elem for elem in footer_candidates if elem])
        
        # Find copyright and disclaimer elements
        copyright_candidates = [
            soup.find('div', class_=lambda c: c and ('copyright' in c.lower())),
            soup.find('div', id=lambda i: i and ('copyright' in i.lower())),
            soup.find(lambda tag: tag.name and tag.string and 'copyright' in tag.get_text().lower()),
        ]
        elements_to_remove.extend([elem for elem in copyright_candidates if elem])
        
        # Find disclaimer elements
        disclaimer_candidates = [
            soup.find('div', class_=lambda c: c and ('disclaimer' in c.lower())),
            soup.find('div', id=lambda i: i and ('disclaimer' in i.lower())),
        ]
        elements_to_remove.extend([elem for elem in disclaimer_candidates if elem])
        
        return [elem for elem in elements_to_remove if elem is not None]
    
    def update_local_references(self, soup: BeautifulSoup, base_dir: str = "") -> BeautifulSoup:
        """Update href and img src references to point to local files.
        
        Args:
            soup: BeautifulSoup parsed HTML document
            base_dir: Base directory for local file references
            
        Returns:
            Modified BeautifulSoup object
        """
        # Update href attributes in <a> tags
        for link in soup.find_all('a', href=True):
            href = link['href']
            if not href.startswith(('http://', 'https://', 'mailto:', '#')):
                # This is a local reference, update it
                if base_dir:
                    link['href'] = f"{base_dir}/{href}"
                logger.debug(f"Updated href: {href} -> {link['href']}")
        
        # Update src attributes in <img> tags
        for img in soup.find_all('img', src=True):
            src = img['src']
            if not src.startswith(('http://', 'https://', 'data:')):
                # This is a local reference, update it
                if base_dir:
                    img['src'] = f"{base_dir}/{src}"
                logger.debug(f"Updated img src: {src} -> {img['src']}")
        
        # Update link tags (CSS, etc.)
        for link in soup.find_all('link', href=True):
            href = link['href']
            if not href.startswith(('http://', 'https://')):
                if base_dir:
                    link['href'] = f"{base_dir}/{href}"
                logger.debug(f"Updated link href: {href} -> {link['href']}")
        
        return soup
    
    def clean_html_structure(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Clean HTML structure by removing unwanted elements.
        
        Args:
            soup: BeautifulSoup parsed HTML document
            
        Returns:
            Cleaned BeautifulSoup object
        """
        # Remove elements identified for removal
        elements_to_remove = self.identify_removal_elements(soup)
        for element in elements_to_remove:
            logger.debug(f"Removing element: {element.name} with id/class {element.get('id', element.get('class'))}")
            element.decompose()
        
        # Remove script and style tags
        for script in soup.find_all(['script', 'style']):
            script.decompose()
        
        # Remove comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remove empty paragraphs and divs
        for tag in soup.find_all(['p', 'div']):
            if not tag.get_text(strip=True) and not tag.find_all(['img', 'video', 'audio']):
                tag.decompose()
        
        return soup
    
    def remove_legal_content(self, text: str) -> str:
        """Remove legal disclaimers and copyright notices from text.
        
        Args:
            text: Input text content
            
        Returns:
            Text with legal content removed
        """
        for pattern in self.legal_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up extra whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    
    def clean_html_file(
        self, 
        html_content: str, 
        base_dir: str = "",
        remove_legal: bool = True
    ) -> str:
        """Clean HTML content comprehensively.
        
        Args:
            html_content: Raw HTML content
            base_dir: Base directory for local file references
            remove_legal: Whether to remove legal content
            
        Returns:
            Cleaned HTML content as string
        """
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Clean HTML structure
        soup = self.clean_html_structure(soup)
        
        # Update local references
        if base_dir:
            soup = self.update_local_references(soup, base_dir)
        
        # Convert back to string
        cleaned_html = str(soup)
        
        # Remove legal content if requested
        if remove_legal:
            cleaned_html = self.remove_legal_content(cleaned_html)
        
        return cleaned_html
    
    def should_skip_file(self, html_content: str) -> bool:
        """Determine if a file should be skipped based on content analysis.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            True if file should be skipped, False otherwise
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        print(f"[CLEANER] Analyzing HTML file for skip conditions...")
        
        # Check if title exists and indicates code documentation
        if soup.title and soup.title.string:
            title_text = soup.title.string.strip().lower()
            print(f"[CLEANER] Found title: '{soup.title.string.strip()}'")
            skip_patterns = ["class", "enum", "struct", "derivedDataType", "deriveddatatype", "module", "interface"]
            for pattern in skip_patterns:
                if title_text.startswith(pattern):
                    print(f"[CLEANER] Skipping code documentation with title: {title_text}")
                    logger.info(f"Skipping code documentation with title: {title_text}")
                    return True
        
        # Check for specific anchor patterns (same as in converter)
        # Looking for <a name="TITLE">derivedDataType&#160;RuleDataType</a> or similar
        title_anchors = soup.find_all('a', attrs={'name': 'TITLE'})
        for anchor in title_anchors:
            if anchor.get_text():
                anchor_text = anchor.get_text().strip().lower()
                print(f"[CLEANER] Found title anchor: '{anchor.get_text().strip()}'")
                
                # Check for all skip patterns
                for pattern in skip_patterns:
                    if pattern.lower() in anchor_text:
                        print(f"[CLEANER] Skipping HTML file with '{pattern}' title anchor: {anchor.get_text().strip()}")
                        logger.info(f"Skipping HTML file with '{pattern}' title anchor: {anchor.get_text().strip()}")
                        return True
        
        # Also check meta tags with name="TITLE"
        title_meta = soup.find('meta', attrs={'name': 'TITLE'})
        if title_meta and title_meta.get('content'):
            content_text = title_meta.get('content').strip()
            content_text_lower = content_text.lower()
            print(f"[CLEANER] Found meta title: '{content_text}'")
            
            # Check for all skip patterns
            for pattern in skip_patterns:
                if pattern.lower() in content_text_lower:
                    print(f"[CLEANER] Skipping HTML file with '{pattern}' meta title: {content_text}")
                    logger.info(f"Skipping HTML file with '{pattern}' meta title: {content_text}")
                    return True
        
        # Check for other indicators of technical documentation that should be skipped
        technical_indicators = soup.find_all(class_=lambda c: c and any(
            indicator in c.lower() for indicator in ['api-doc', 'code-doc', 'reference-doc']
        ))
        
        if technical_indicators:
            print(f"[CLEANER] Skipping technical documentation based on CSS classes")
            logger.info("Skipping technical documentation based on CSS classes")
            return True
        
        print(f"[CLEANER] File passed skip checks - will be processed")
        return False


class MarkdownCleaner:
    """Advanced markdown cleaner optimized for LLM fine-tuning and RAG datasets."""
    
    def __init__(self):
        """Initialize the markdown cleaner with patterns and rules."""
        
        # Patterns for removing unwanted content
        self.removal_patterns = [
            # Navigation and UI elements
            r'(?i)^\s*(?:home|back|next|previous|contents?|index|search)\s*$',
            r'(?i)^\s*(?:skip to|jump to|go to)\s+.*$',
            r'(?i)^\s*(?:table of contents?|toc)\s*$',
            
            # Document metadata lines - ENHANCED
            r'^\s*##\s+Document:\s+.*?\.md\s*$',  # Remove document section headers
            r'^\s*---\s*$',  # Remove YAML frontmatter delimiters
            r'^\s*#\s+Combined Documentation\s*$',  # Remove combined doc title
            r'^\s*#\s+\d+_\d+-.*?\.\w+\s*$',  # Remove filename-based titles like "# 138_22104-LZA7016017_1Uen.BV"
            r'^\s*#\s*$',  # Remove empty headings
            r'^\s*This document contains content from \d+ source files\.\s*$',  # Combined doc description
            
            # Technical documentation artifacts - ENHANCED FOR TECHNICAL DOCS
            r'^\s*NR Dynamic Power Optimizer - Overview\s*$',  # Remove overview title remnants
            r'^\s*- Different Amounts of Data in the Buffer\s*$',  # Remove TOC items
            r'^\s*- Switch Threshold Configuration for Different Maximum Channel Capacities\s*$',
            r'^\s*- Dependencies\s*$',  # Remove standalone section markers
            r'^\s*- Feature Operation\s*$',
            r'^\s*- Network Impact\s*$',
            r'^\s*- Parameters\s*$',
            r'^\s*- Performance\s*$',
            r'^\s*- Activate the Feature\s*$',
            r'^\s*- Deactivate the Feature\s*$',
            r'^\s*- Engineering Guidelines\s*$',
            r'^\s*- Appendix:.*?$',  # Remove appendix markers
            r'^\s*- Downswitch Procedure\s*$',  # Specific procedure TOC items
            r'^\s*- Upswitch Procedure\s*$',
            r'^\s*- Impacted Alarms\s*$',
            r'^\s*- Capacity and Performance\s*$',
            r'^\s*- Interfaces\s*$',
            r'^\s*- Other Network Elements\s*$',
            r'^\s*- KPIs\s*$',
            r'^\s*- Counters\s*$',
            r'^\s*- Events\s*$',
            
            # Table of contents artifacts - EXPANDED
            r'^\s*-\s+.*?(?:Configuration|Support|Feature|Operation|Guidelines)\s*$',  # TOC items
            r'^\s*-\s+.*?(?:BWP|NR|DCI|UE).*?$',  # Technical TOC items
            r'^\s*-\s+.*?(?:Monitoring|Performance|Duration|Improved).*?$',  # Performance TOC
            
            # Malformed table headers and cells - ENHANCED
            r'^\s*\|\s*Feature Name\s*\|\s*NR Dynamic Power Optimizer\s*\|\s*$',  # Broken table rows
            r'^\s*\|[-\s\|]*\|\s*$',  # Empty table separators
            r'^\s*\|\s*\|\s*$',  # Completely empty table rows
            r'^\s*Summary\s*$',  # Standalone summary headers
            r'^\s*Additional Information\s*$',  # Standalone info headers
            r'^\s*Hardware\s*$',  # Standalone hardware headers
            r'^\s*Limitations\s*$',  # Standalone limitations headers
            r'^\s*Network Requirements\s*$',  # Standalone requirements headers
            r'^\s*Summary and Benefits\s*$',  # Standalone summary sections
            r'^\s*Operation\s*$',  # Standalone operation headers
            
            # Page elements
            r'(?i)^\s*page\s+\d+(?:\s+of\s+\d+)?\s*$',
            r'(?i)^\s*\d+\s*/\s*\d+\s*$',
            r'(?i)^\s*(?:continued|cont\.?)\s*$',
            
            # Legal and metadata
            r'(?i)^\s*(?:copyright|©).*?(?:all rights reserved|reserved).*?$',
            r'(?i)^\s*(?:confidential|proprietary|internal use).*?$',
            r'(?i)^\s*(?:draft|preliminary|work in progress).*?$',
            
            # Form elements and interactive content
            r'(?i)^\s*(?:click|select|choose|enter|input|submit|button|link).*?$',
            r'(?i)^\s*\[.*?(?:button|link|field|dropdown|checkbox)\].*?$',
            
            # Empty or meaningless content
            r'^\s*[•·\-\*\+]\s*$',  # Lone bullets
            r'^\s*[\[\(\{\<].*?[\]\)\}\>]\s*$',  # Empty brackets/parens
            r'^\s*(?:\.{3,}|_{3,}|\-{3,})\s*$',  # Excessive punctuation
            
            # File paths and conversion metadata - ENHANCED
            r'.*?_\d{4,}-LZA\d+.*?\.html.*?$',  # File path references
            r'^\s*=\s+.*?\.html.*?$',  # HTML file assignments
            r'.*?WARN,?\s+.*?(?:not expanded|not found|missing).*?$',  # Conversion warnings
            r'.*?ERROR,?\s+.*?$',  # Error messages
            r'^\s*\d+\s+Â\s*$',  # Malformed numbering
            r'^\s*\d+\.\d+\s+Â\s*$',  # Malformed section numbers
            r'.*?Production Subtask:?\s+\w+.*?$',  # Production metadata
            r'^\s*\d+\s*Â\s*\n?\s*\d+\.\d+\s*Â.*?$',  # Malformed section headers
            r'.*?converted_with:\s+Flow4-Docling.*?$',  # Conversion metadata
            r'.*?features:\s+accelerator:\s+True.*?$',  # Feature metadata
            r'.*?source:\s+.*?\.html.*?$',  # Source file references
            r'.*?title:\s+.*?_\d{4,}-.*?$',  # Generated titles
            
            # Technical documentation specific artifacts
            r'^\s*is equal to or smaller than the uplink\s*$',  # Broken sentence fragments
            r'^\s*is equal to or smaller than the uplink threshold\.\s*$',
            r'^\s*is higher than the uplink threshold\.\s*$',
            r'^\s*bandwidth, the gNodeB configures the F-BWP.*?$',  # Incomplete list items
            r'^\s*both P-BWP and F-BWP.*?$',
            r'^\s*it stays configured to the P-BWP\.\s*$',
            r'^\s*PCell\.\s*$',  # Orphaned technical terms
            r'^\s*F-BWP\.\s*$',
            r'^\s*counters are used for time tracking:\s*$',  # Orphaned section intros
            
            # Combined document metadata
            r'^\s*This document contains content from \d+ source files\.\s*$',  # Combined doc description
            r'^\s*# Combined Documentation\s*$',  # Combined doc title
            r'^\s*date:\s+\d{4}-\d{2}-\d{2}\s*$',  # Date metadata
            r'^\s*source_files:\s+\d+\s*$',  # Source files count
            r'^\s*pipeline_version:\s+Flow4-.*?$',  # Pipeline version
        ]
        
        # Additional patterns for cleaning conversion artifacts
        self.artifact_patterns = [
            # HTML encoding artifacts
            (r'\bÂ\b', ''),  # Remove standalone Â characters
            (r'Â\s*\n', '\n'),  # Remove Â at line endings
            (r'\s*Â\s*', ' '),  # Replace Â with single space
            (r'&#160;', ' '),  # Replace non-breaking space entity
            (r'&nbsp;', ' '),  # Replace HTML non-breaking space
            
            # File references and metadata
            (r'\b\d+_\d+-LZA\d+.*?\.html\b', ''),  # Remove file references
            (r'\bDocument:\s+.*?\.md\b', ''),  # Remove document metadata
            (r'\btitle:\s+.*?source:', ''),  # Remove title/source metadata
            (r'\bconverted_with:.*?(?=\n|$)', ''),  # Remove conversion metadata
            (r'\bfeatures:.*?custom_rules:\s+\w+', ''),  # Remove feature flags
            
            # YAML frontmatter remnants - COMPREHENSIVE
            (r'^---\s*$', ''),  # Remove standalone YAML delimiters
            (r'^\s*title:\s+.*?$', ''),  # Remove title metadata lines
            (r'^\s*source:\s+.*?$', ''),  # Remove source metadata lines
            (r'^\s*converted_with:\s+.*?$', ''),  # Remove conversion metadata
            (r'^\s*features:\s*$', ''),  # Remove features section start
            (r'^\s*accelerator:\s+.*?$', ''),  # Remove accelerator metadata
            (r'^\s*tables:\s+.*?$', ''),  # Remove tables metadata
            (r'^\s*figures:\s+.*?$', ''),  # Remove figures metadata
            (r'^\s*multimodal:\s+.*?$', ''),  # Remove multimodal metadata
            (r'^\s*custom_rules:\s+.*?$', ''),  # Remove custom_rules metadata
            (r'^\s*date:\s+\d{4}-\d{2}-\d{2}\s*$', ''),  # Remove date metadata
            (r'^\s*source_files:\s+\d+\s*$', ''),  # Remove source files count
            (r'^\s*pipeline_version:\s+.*?$', ''),  # Remove pipeline version
            
            # Warning and error messages
            (r'\bWARN,?\s+[^.]*?\.(?:\s|$)', ''),  # Remove warning messages
            (r'\bERROR,?\s+[^.]*?\.(?:\s|$)', ''),  # Remove error messages
            (r'\bNo matching MOM reference found for[^.]*\.', ''),  # Specific warning
            (r'\bAbbreviations not expanded[^.]*\.', ''),  # Specific warning
            
            # Production and processing metadata
            (r'\bProduction Subtask:\s+\w+', ''),  # Remove subtask references
            (r'\b\d{2}/\d{4,}-L[^.]*\.', ''),  # Remove processing references
            
            # Malformed section numbering
            (r'^\s*\d+\s*\n\s*\d+\.\d+\s*', ''),  # Fix broken section numbers
            (r'^\s*\d+\s*Â\s*\n\s*\d+\.\d+\s*Â\s*', ''),  # Fix Â in section numbers
        ]
        
        # Patterns for content normalization
        self.normalization_patterns = [
            # Fix heading spacing
            (r'(#{1,6})\s*\n+', r'\1 '),  # Remove newlines in headings
            (r'\n+(#{1,6})', r'\n\n\1'),  # Ensure blank line before headings
            (r'(#{1,6}[^\n]*)\n{3,}', r'\1\n\n'),  # Limit lines after headings
            
            # Fix list formatting
            (r'\n+(\s*[\*\-\+])', r'\n\1'),  # Fix list spacing
            (r'(\s*[\*\-\+][^\n]*)\n{3,}', r'\1\n'),  # Limit lines after list items
            
            # Clean up excessive whitespace
            (r'\n{4,}', '\n\n\n'),  # Max 3 consecutive newlines for section breaks
            (r'[ \t]{3,}', '  '),  # Max 2 consecutive spaces
            (r'[ \t]+\n', '\n'),  # Remove trailing whitespace
            
            # Fix punctuation spacing - ensure proper sentence separation
            (r'([.!?])\s{2,}', r'\1 '),  # Single space after sentence punctuation
            (r'([,:;])\s{2,}', r'\1 '),  # Single space after clause punctuation
            (r'([.!?])\s*(#{1,6})', r'\1\n\n\2'),  # Ensure proper separation between sentences and headings
            (r'([^\n])\s*(#{1,6})', r'\1\n\n\2'),  # Ensure paragraphs are separated from headings
        ]
        
        # Table cleaning patterns - ENHANCED for technical documentation
        self.table_patterns = [
            # Remove empty table cells
            (r'\|\s*\|\s*\|', '| |'),
            # Clean up table alignment
            (r'\|\s*\:\-+\s*\|', '|:--|'),
            (r'\|\s*\-+\:\s*\|', '|--:|'),
            (r'\|\s*\:\-+\:\s*\|', '|:--:|'),
            # Remove broken table rows
            (r'\|\s*\n\s*\|', '|\n|'),
            
            # Technical documentation table fixes - NEW
            # Fix malformed table headers with mixed separators
            (r'\|[-\|\s]*\|[-\|\s]*\|[-\|\s]*\|', '|---|---|---|'),
            # Remove table rows that are just separators
            (r'^\s*\|\s*[-\|]+\s*\|\s*[-\|]+\s*\|\s*$', ''),
            # Fix broken table cells with escaped pipes
            (r'\\\|', '|'),
            # Clean up extra spaces in table cells
            (r'\|\s{3,}', '| '),
            (r'\s{3,}\|', ' |'),
            # Fix tables with inconsistent column counts
            (r'\|\s*\|\s*\|\s*\|\s*\|\s*\|\s*\|', '| | | | | |'),
            # Remove table formatting artifacts
            (r'\|\s*\*\*\s*\|\s*\*\*\s*\|', '| | |'),  # Empty bold cells
            (r'\|\s*`\s*`\s*\|', '| |'),  # Empty code cells
        ]
    
    def remove_unwanted_content(self, text: str) -> str:
        """Remove unwanted content using predefined patterns.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text with unwanted content removed
        """
        lines = text.split('\n')
        cleaned_lines = []
        in_yaml_block = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Handle YAML frontmatter blocks
            if line_stripped == '---':
                if not in_yaml_block:
                    in_yaml_block = True
                    continue  # Skip opening ---
                else:
                    in_yaml_block = False
                    continue  # Skip closing ---
            
            # Skip all lines inside YAML blocks
            if in_yaml_block:
                continue
            
            # Skip empty lines (handle separately)
            if not line_stripped:
                cleaned_lines.append(line)
                continue
            
            # Check against removal patterns
            should_remove = False
            for pattern in self.removal_patterns:
                if re.match(pattern, line_stripped):
                    logger.debug(f"Removing line: {line_stripped[:50]}...")
                    should_remove = True
                    break
            
            if not should_remove:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and formatting for better chunking.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text with normalized whitespace
        """
        # Apply normalization patterns
        for pattern, replacement in self.normalization_patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
        
        # Handle the specific case of excessive newlines
        # Replace 4+ consecutive newlines with exactly 2 (paragraph break)
        text = re.sub(r'\n{4,}', '\n\n', text)
        
        # Ensure document starts and ends cleanly
        text = text.strip()
        
        return text
    
    def clean_conversion_artifacts(self, text: str) -> str:
        """Remove conversion artifacts, warnings, and metadata pollution.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text with conversion artifacts removed
        """
        # Apply artifact cleaning patterns
        for pattern, replacement in self.artifact_patterns:
            text = re.sub(pattern, replacement, text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove lines that are just whitespace after artifact removal
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.strip()
            # Keep the line if it has actual content after cleaning
            if cleaned_line:
                cleaned_lines.append(line)
            elif not cleaned_line and cleaned_lines and cleaned_lines[-1].strip():
                # Keep empty lines that separate content
                cleaned_lines.append('')
        
        # Remove excessive empty lines introduced by artifact removal
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text
    
    def clean_tables(self, text: str) -> str:
        """Clean and normalize table formatting.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text with cleaned tables
        """
        # Apply table cleaning patterns
        for pattern, replacement in self.table_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Fix specific table formatting issues from the technical document
        lines = text.split('\n')
        cleaned_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Fix broken table rows that start mid-sentence
            if ('|' in line and 
                not line.strip().startswith('|') and
                '|' in line and
                line.count('|') >= 2):
                
                # Split text before first pipe and make it a paragraph
                parts = line.split('|', 1)
                if parts[0].strip():
                    cleaned_lines.append(parts[0].strip())
                    # Reconstruct the table row
                    table_part = '|' + parts[1]
                    if table_part.strip():
                        cleaned_lines.append(table_part)
                else:
                    cleaned_lines.append(line)
            
            # Fix tables that have missing headers by adding proper structure
            elif (line.strip().startswith('|') and 
                  i > 0 and 
                  not lines[i-1].strip().startswith('|') and
                  '|' in line):
                
                # Count columns to add proper header
                cols = line.count('|') - 1
                if cols > 0:
                    # Check if this looks like a data row without header
                    if not any(word in line.lower() for word in ['feature', 'name', 'identity', 'type']):
                        # Add a generic header
                        header_cols = ['Column ' + str(j+1) for j in range(cols)]
                        header = '| ' + ' | '.join(header_cols) + ' |'
                        separator = '|' + '---|' * cols
                        cleaned_lines.append(header)
                        cleaned_lines.append(separator)
                
                cleaned_lines.append(line)
            
            else:
                cleaned_lines.append(line)
            
            i += 1
        
        text = '\n'.join(cleaned_lines)
        
        # Remove completely empty tables
        text = re.sub(r'\n\s*\|\s*\|\s*\n\s*\|\s*[\-:]+\s*\|\s*\n(?:\s*\|\s*\|\s*\n)*', '\n', text)
        
        # Fix tables with inconsistent formatting - ensure proper spacing
        text = re.sub(r'\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|', r'| \1 | \2 |', text)
        
        return text
    
    def fix_technical_content_structure(self, text: str) -> str:
        """Fix technical documentation-specific content issues.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text with improved structure for technical documentation
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Fix broken feature identity tables
            if '| Feature Identity |' in line and 'FAJ' in line:
                # Ensure proper table structure for feature descriptions
                if not (i > 0 and '|' in lines[i-1]):
                    cleaned_lines.append('| Property | Value |')
                    cleaned_lines.append('|---|---|')
            
            # Fix lines that are table content but misformatted  
            if line.strip() == 'DCI-based BWP switch.' and i > 0 and '|' in lines[i-1]:
                # This should be part of the previous table cell
                if cleaned_lines and '|' in cleaned_lines[-1]:
                    cleaned_lines[-1] = cleaned_lines[-1].rstrip() + ' DCI-based BWP switch. |'
                    i += 1
                    continue
            
            # Fix broken list items that got concatenated
            if line.strip() and not line.startswith(('- ', '* ', '+ ', '#')):
                # Check if line contains multiple sentences that should be list items
                if 'Note:' in line or 'increases.' in line or 'decreases.' in line:
                    # Split on common sentence boundaries in technical docs
                    parts = re.split(r'(?<=\.) (?=[A-Z])', line)
                    if len(parts) > 1:
                        for part in parts:
                            if part.strip():
                                if part.strip().startswith('Note:'):
                                    cleaned_lines.append('')
                                    cleaned_lines.append('**Note:** ' + part.strip()[5:].strip())
                                else:
                                    cleaned_lines.append(part.strip())
                        i += 1
                        continue
            
            # Fix improperly formatted notes and important callouts
            if line.strip().startswith('Note:'):
                cleaned_lines.append('')
                cleaned_lines.append('**Note:** ' + line.strip()[5:].strip())
            elif 'Note:' in line and not line.strip().startswith('**Note:**'):
                # Extract and format inline notes
                before_note, note_content = line.split('Note:', 1)
                if before_note.strip():
                    cleaned_lines.append(before_note.strip())
                cleaned_lines.append('')
                cleaned_lines.append('**Note:** ' + note_content.strip())
            else:
                cleaned_lines.append(line)
            
            i += 1
        
        text = '\n'.join(cleaned_lines)
        
        # Fix specific technical documentation patterns
        # Fix attribute references that are incomplete
        text = re.sub(r'The value of these\s*attributes', 'The value of these attributes', text)
        text = re.sub(r'attributes\. The', 'attributes. The', text)
        
        # Fix broken procedure listings
        text = re.sub(r'procedures: - ', 'procedures:\n- ', text)
        text = re.sub(r'met: - ', 'met:\n- ', text)
        text = re.sub(r'apply: - ', 'apply:\n- ', text)
        
        # Fix KPI and counter references
        text = re.sub(r'following\s*:\s*-\s*', 'following:\n- ', text)
        text = re.sub(r'counters and\s*might', 'counters and KPIs might', text)
        
        # Fix orphaned list continuation
        text = re.sub(r'\n(is equal to or smaller than the uplink)\n', ' is equal to or smaller than the uplink\n', text)
        text = re.sub(r'\n(is higher than the uplink threshold\.)\n', ' is higher than the uplink threshold.\n', text)
        
        # Fix duplicate table headers
        text = re.sub(r'\| Column 1 \| Column 2 \| Column 3 \|\s*\| Column 1 \| Column 2 \| Column 3 \|', '| Column 1 | Column 2 | Column 3 |', text)
        
        # Fix broken feature descriptions
        text = re.sub(r'buffer\. The feature can configure the following types of BWPs to a UE:\n', 'buffer. The feature can configure the following types of BWPs to a UE:\n\n', text)
        
        # Fix broken list items in technical content
        text = re.sub(r'met:\s*-\s*The number', 'met:\n- The number', text)
        text = re.sub(r'factors:\s*-\s*The maximum', 'factors:\n- The maximum', text)
        text = re.sub(r'supported:\s*-\s*Only', 'supported:\n- Only', text)
        text = re.sub(r'elements:\s*-\s*It is', 'elements:\n- It is', text)
        
        # Fix malformed code formatting
        text = re.sub(r'format 1\\_1', 'format 1_1', text)
        
        return text
    
    def fix_heading_hierarchy(self, text: str) -> str:
        """Fix heading hierarchy for better document structure.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text with normalized heading hierarchy
        """
        lines = text.split('\n')
        current_level = 0
        fixed_lines = []
        
        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
            if heading_match:
                hashes, title = heading_match.groups()
                level = len(hashes)
                
                # Normalize heading levels (no skipping levels)
                if level > current_level + 1:
                    level = current_level + 1
                    hashes = '#' * level
                    logger.debug(f"Normalized heading level: {title}")
                
                current_level = level
                fixed_lines.append(f"{hashes} {title}")
            else:
                fixed_lines.append(line)
        
        return '\n'.join(fixed_lines)
    
    def remove_html_artifacts(self, text: str) -> str:
        """Remove HTML artifacts that survived markdown conversion.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text with HTML artifacts removed
        """
        # Remove HTML comments
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        
        # Remove HTML tags that might have survived
        text = re.sub(r'</?(?:div|span|p|br|hr)[^>]*>', '', text, flags=re.IGNORECASE)
        
        # Remove HTML entities
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&mdash;': '—',
            '&ndash;': '–',
            '&hellip;': '…',
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # Remove remaining numeric HTML entities
        text = re.sub(r'&#\d+;', '', text)
        
        return text
    
    def optimize_for_chunking(self, text: str) -> str:
        """Optimize text structure for better semantic chunking.
        
        Args:
            text: Input markdown text
            
        Returns:
            Text optimized for chunking
        """
        # Ensure sections are properly separated
        text = re.sub(r'(\n#{1,6}[^\n]*)\n([^\n#])', r'\1\n\n\2', text)
        
        # Ensure list items are grouped
        text = re.sub(r'(\n\s*[\*\-\+][^\n]*)\n\n(\s*[\*\-\+])', r'\1\n\2', text)
        
        # Group related sentences (avoid single-sentence chunks)
        lines = text.split('\n')
        optimized_lines = []
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # If it's a short line (< 50 chars) and not a heading/list
            if (len(line.strip()) < 50 and 
                not re.match(r'^\s*#{1,6}', line) and 
                not re.match(r'^\s*[\*\-\+]', line) and
                line.strip()):
                
                # Try to combine with next line if it's also short
                if (i + 1 < len(lines) and 
                    len(lines[i + 1].strip()) < 100 and
                    lines[i + 1].strip() and
                    not re.match(r'^\s*#{1,6}', lines[i + 1])):
                    
                    combined = f"{line.strip()} {lines[i + 1].strip()}"
                    optimized_lines.append(combined)
                    i += 2
                    continue
            
            optimized_lines.append(line)
            i += 1
        
        return '\n'.join(optimized_lines)
    
    def clean_markdown_comprehensive(
        self, 
        text: str, 
        remove_excessive_newlines: bool = True,
        optimize_for_llm: bool = True
    ) -> str:
        """Comprehensive markdown cleaning for LLM training and RAG.
        
        Args:
            text: Input markdown text
            remove_excessive_newlines: Whether to normalize newlines
            optimize_for_llm: Whether to apply LLM-specific optimizations
            
        Returns:
            Cleaned markdown text
        """
        logger.info("Starting comprehensive markdown cleaning")
        
        # Step 1: Remove HTML artifacts
        text = self.remove_html_artifacts(text)
        logger.debug("Removed HTML artifacts")
        
        # Step 2: Clean conversion artifacts and metadata pollution
        text = self.clean_conversion_artifacts(text)
        logger.debug("Cleaned conversion artifacts")
        
        # Step 3: Remove unwanted content
        text = self.remove_unwanted_content(text)
        logger.debug("Removed unwanted content")
        
        # Step 4: Clean tables
        text = self.clean_tables(text)
        logger.debug("Cleaned tables")
        
        # Step 5: Fix technical content structure
        text = self.fix_technical_content_structure(text)
        logger.debug("Fixed technical content structure")
        
        # Step 6: Fix heading hierarchy
        text = self.fix_heading_hierarchy(text)
        logger.debug("Fixed heading hierarchy")
        
        # Step 7: Normalize whitespace
        if remove_excessive_newlines:
            text = self.normalize_whitespace(text)
            logger.debug("Normalized whitespace")
        
        # Step 8: Optimize for chunking and LLM training
        if optimize_for_llm:
            text = self.optimize_for_chunking(text)
            logger.debug("Optimized for chunking")
        
        # Final cleanup
        text = text.strip()
        
        # Log statistics
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        logger.info(f"Cleaning complete: {len(lines)} total lines, {len(non_empty_lines)} non-empty lines")
        
        return text
    
    def clean_combined_document_file(self, file_path: str, output_path: str = None) -> str:
        """Clean a combined markdown document file for optimal chunking.
        
        Args:
            file_path: Path to the combined markdown file
            output_path: Optional path to save cleaned content
            
        Returns:
            Cleaned markdown content
        """
        logger.info(f"Cleaning combined document: {file_path}")
        
        try:
            # Read the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Apply comprehensive cleaning
            cleaned_content = self.clean_markdown_comprehensive(
                content, 
                remove_excessive_newlines=True,
                optimize_for_llm=True
            )
            
            # Save to output path if specified
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_content)
                logger.info(f"Cleaned content saved to: {output_path}")
            
            return cleaned_content
            
        except Exception as e:
            logger.error(f"Error cleaning combined document: {e}")
            raise


class EnhancedCleaner:
    """Combined HTML and Markdown cleaner for optimal document processing."""
    
    def __init__(self):
        """Initialize combined cleaner."""
        self.html_cleaner = HTMLCleaner()
        self.markdown_cleaner = MarkdownCleaner()
    
    def clean_document_pipeline(
        self, 
        html_content: str,
        base_dir: str = "",
        remove_legal: bool = True,
        clean_markdown: bool = True,
        optimize_for_llm: bool = True
    ) -> str:
        """Complete document cleaning pipeline from HTML to clean markdown.
        
        Args:
            html_content: Raw HTML content
            base_dir: Base directory for local file references
            remove_legal: Whether to remove legal content
            clean_markdown: Whether to apply markdown cleaning
            optimize_for_llm: Whether to optimize for LLM training
            
        Returns:
            Fully cleaned content ready for conversion and chunking
        """
        logger.info("Starting complete document cleaning pipeline")
        
        # Step 1: Clean HTML
        cleaned_html = self.html_cleaner.clean_html_file(
            html_content, 
            base_dir=base_dir, 
            remove_legal=remove_legal
        )
        logger.debug("HTML cleaning completed")
        
        # If markdown cleaning is requested, assume this will be converted to markdown
        # and apply post-processing (this would typically be called after Docling conversion)
        if clean_markdown:
            # Note: In practice, you'd call this after markdown conversion
            # For now, return the cleaned HTML for conversion
            logger.info("HTML cleaned, ready for markdown conversion and post-processing")
        
        return cleaned_html
    
    def clean_markdown_post_conversion(
        self, 
        markdown_content: str,
        optimize_for_llm: bool = True
    ) -> str:
        """Clean markdown content after conversion from HTML.
        
        Args:
            markdown_content: Markdown content from document converter
            optimize_for_llm: Whether to optimize for LLM training
            
        Returns:
            Clean markdown optimized for chunking and LLM training
        """
        return self.markdown_cleaner.clean_markdown_comprehensive(
            markdown_content,
            remove_excessive_newlines=True,
            optimize_for_llm=optimize_for_llm
        )