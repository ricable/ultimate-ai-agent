"""Generic content processor for auto-browser."""

import json
from pathlib import Path
from typing import Dict, Any, Union, List
from dataclasses import dataclass

@dataclass
class PageElement:
    """Represents an interactive element on the page."""
    selector: str
    element_type: str  # button, input, link, text, etc.
    name: str
    description: str = ""
    is_interactive: bool = False
    possible_actions: List[str] = None

    def __post_init__(self):
        if self.possible_actions is None:
            self.possible_actions = []
            if self.element_type == "input":
                self.possible_actions.extend(["type", "clear"])
            elif self.element_type in ["button", "link"]:
                self.possible_actions.append("click")
            elif self.element_type == "select":
                self.possible_actions.extend(["select", "get_options"])

class ContentProcessor:
    """Process and format content from any webpage."""
    
    def __init__(self):
        self.page_elements = {}
        self.content_structure = {}
    
    def analyze_page(self, page_content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze page content and determine structure and possible actions."""
        analyzed = {
            "content": {},
            "interactive_elements": [],
            "forms": [],
            "navigation": []
        }
        
        for key, value in page_content.items():
            element = self._analyze_element(key, value)
            if element.is_interactive:
                analyzed["interactive_elements"].append(element)
                if element.element_type == "input":
                    self._find_related_form(element, analyzed["forms"])
            else:
                analyzed["content"][key] = value
                
        return analyzed
    
    def _analyze_element(self, key: str, value: Any) -> PageElement:
        """Analyze a single element and determine its type and possible actions."""
        # Common form-related keywords
        input_keywords = ["search", "login", "username", "password", "email", "input"]
        button_keywords = ["submit", "button", "send", "login", "sign"]
        
        element_type = "text"
        is_interactive = False
        possible_actions = []
        
        # Determine element type from key name and value
        key_lower = key.lower()
        if any(keyword in key_lower for keyword in input_keywords):
            element_type = "input"
            is_interactive = True
        elif any(keyword in key_lower for keyword in button_keywords):
            element_type = "button"
            is_interactive = True
        elif "link" in key_lower or (isinstance(value, str) and value.startswith("http")):
            element_type = "link"
            is_interactive = True
            
        return PageElement(
            selector=f"#{key}",  # Default selector, should be updated with actual selector
            element_type=element_type,
            name=key,
            is_interactive=is_interactive,
            possible_actions=possible_actions
        )
    
    def _find_related_form(self, element: PageElement, forms: List[Dict]) -> None:
        """Find or create a form group for related input elements."""
        # Group related form elements based on naming patterns and proximity
        form_keywords = {
            "login": ["username", "password", "email"],
            "search": ["query", "term", "keyword"],
            "contact": ["name", "email", "message"]
        }
        
        element_name = element.name.lower()
        for form_type, fields in form_keywords.items():
            if any(field in element_name for field in fields):
                # Find existing form or create new one
                form = next((f for f in forms if f["type"] == form_type), None)
                if not form:
                    form = {"type": form_type, "fields": []}
                    forms.append(form)
                form["fields"].append(element)
    
    def format_content(self, analyzed_data: Dict[str, Any]) -> str:
        """Format analyzed content as markdown."""
        sections = [
            "# Page Content Summary\n",
            "## Main Content",
            self._format_content_section(analyzed_data["content"]),
            "\n## Interactive Elements",
            self._format_interactive_section(analyzed_data["interactive_elements"]),
            "\n## Forms",
            self._format_forms_section(analyzed_data["forms"])
        ]
        
        return "\n\n".join(sections)
    
    def _format_content_section(self, content: Dict[str, Any]) -> str:
        """Format the main content section."""
        formatted = []
        for key, value in content.items():
            if isinstance(value, (list, tuple)):
                formatted.append(f"### {key}")
                formatted.extend(f"- {item}" for item in value)
            else:
                formatted.append(f"### {key}")
                formatted.append(str(value))
        return "\n\n".join(formatted)
    
    def _format_interactive_section(self, elements: List[PageElement]) -> str:
        """Format the interactive elements section."""
        if not elements:
            return "No interactive elements found"
            
        formatted = []
        for element in elements:
            formatted.append(f"- **{element.name}** ({element.element_type})")
            if element.possible_actions:
                formatted.append(f"  - Possible actions: {', '.join(element.possible_actions)}")
        return "\n".join(formatted)
    
    def _format_forms_section(self, forms: List[Dict]) -> str:
        """Format the forms section."""
        if not forms:
            return "No forms found"
            
        formatted = []
        for form in forms:
            formatted.append(f"### {form['type'].title()} Form")
            for field in form["fields"]:
                formatted.append(f"- {field.name} ({field.element_type})")
        return "\n\n".join(formatted)
    
    def save_markdown(self, content: str, url: str, output_dir: str = "output") -> Path:
        """Save markdown content to file."""
        output_path = Path(output_dir) / f"{url.split('/')[-1]}.md"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        return output_path
