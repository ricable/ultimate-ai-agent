"""Content processing module for auto-browser."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class PageElement:
    """Represents an element on the page."""
    selector: str
    element_type: str
    name: str
    description: str = ""
    is_interactive: bool = False
    actions: List[str] = None

    def __post_init__(self):
        if self.actions is None:
            self.actions = self._determine_actions()

    def _determine_actions(self) -> List[str]:
        """Determine possible actions based on element type."""
        actions = []
        if self.element_type == "input":
            actions.extend(["type", "clear"])
        elif self.element_type in ["button", "link"]:
            actions.append("click")
        elif self.element_type == "select":
            actions.extend(["select", "get_options"])
        return actions
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'selector': self.selector,
            'element_type': self.element_type,
            'name': self.name,
            'description': self.description,
            'is_interactive': self.is_interactive,
            'actions': self.actions or []
        }

class ContentProcessor:
    """Process and analyze webpage content."""
    
    def analyze_page(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze page content and structure."""
        elements = self._identify_elements(content)
        return {
            "content": self._process_content(content),
            "elements": [el.to_dict() for el in elements],
            "forms": self._find_forms(content)
        }
    
    def _process_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw content into structured format."""
        processed = {}
        for key, value in content.items():
            if isinstance(value, (str, int, float)):
                processed[key] = str(value)
            elif isinstance(value, (list, tuple)):
                processed[key] = [str(item) for item in value]
        return processed
    
    def _identify_elements(self, content: Dict[str, Any]) -> List[PageElement]:
        """Identify and categorize page elements."""
        elements = []
        for key, value in content.items():
            element_type = self._determine_element_type(key, value)
            elements.append(PageElement(
                selector=f"#{key}",
                element_type=element_type,
                name=key,
                is_interactive=element_type in ["input", "button", "link", "select"]
            ))
        return elements
    
    def _determine_element_type(self, key: str, value: Any) -> str:
        """Determine element type from key and value."""
        key_lower = key.lower()
        if any(k in key_lower for k in ["input", "search", "login", "password"]):
            return "input"
        elif any(k in key_lower for k in ["button", "submit", "send"]):
            return "button"
        elif "link" in key_lower or (isinstance(value, str) and value.startswith("http")):
            return "link"
        return "text"
    
    def _find_forms(self, content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify form groups in the content."""
        forms = []
        form_groups = {
            "login": ["username", "password", "email"],
            "search": ["query", "term", "keyword"],
            "contact": ["name", "email", "message"]
        }
        
        current_form = None
        for key in content.keys():
            key_lower = key.lower()
            for form_type, fields in form_groups.items():
                if any(field in key_lower for field in fields):
                    if not current_form or current_form["type"] != form_type:
                        current_form = {"type": form_type, "fields": []}
                        forms.append(current_form)
                    current_form["fields"].append(key)
                    break
        
        return forms
