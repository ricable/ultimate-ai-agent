"""Markdown formatting for auto-browser output."""

from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import re
from urllib.parse import urlparse

class MarkdownFormatter:
    """Format analyzed content as markdown."""
    
    def format_content(self, analyzed_data: Dict[str, Any]) -> str:
        """Format analyzed data into markdown."""
        sections = []
        
        # Add title
        sections.append(f"# {analyzed_data.get('title', 'Analysis Result')}")
        
        # Add metadata if present
        metadata = analyzed_data.get('metadata', {})
        if metadata:
            sections.append("## Metadata")
            for key, value in metadata.items():
                if key == 'timestamp':
                    try:
                        dt = datetime.fromisoformat(value)
                        value = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        pass
                sections.append(f"- **{key.title()}**: {value}")
        
        # Add main content
        content = analyzed_data.get('content')
        if content:
            sections.append("## Content")
            if isinstance(content, dict):
                for key, value in content.items():
                    sections.append(f"### {key}")
                    sections.append(str(value))
            else:
                sections.append(str(content))
        
        return "\n\n".join(sections)
    
    def _create_filename(self, url: str) -> str:
        """Create a safe filename from URL with timestamp."""
        # Parse URL to get domain and path
        parsed = urlparse(url)
        domain = parsed.netloc.replace("www.", "")
        
        # Clean domain and path
        domain = re.sub(r'[^\w\-_]', '_', domain)
        path = re.sub(r'[^\w\-_]', '_', parsed.path)
        if path and path != '/':
            base = f"{domain}_{path.strip('_')}"
        else:
            base = domain
            
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create final filename
        return f"{base}_{timestamp}.md"
    
    def save_markdown(self, content: str, url: str, output_dir: str = "output") -> Path:
        """Save markdown content to file with unique name."""
        # Create unique filename
        filename = self._create_filename(url)
            
        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save content
        file_path = output_path / filename
        file_path.write_text(content)
        return file_path
