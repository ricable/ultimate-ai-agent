"""Interactive browser actions processor for auto-browser."""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class BrowserAction:
    """Represents a browser action to perform."""
    action_type: str  # click, type, select, etc.
    selector: str
    value: Optional[str] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert action to dictionary format."""
        return {
            'action_type': self.action_type,
            'selector': self.selector,
            'value': self.value,
            'description': self.description
        }

class InteractiveProcessor:
    """Process and execute interactive browser actions."""
    
    def analyze_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        """Analyze user prompt to determine required actions."""
        actions = []
        prompt_lower = prompt.lower()
        
        # Search actions
        if any(word in prompt_lower for word in ["search", "find", "look for"]):
            actions.extend([action.to_dict() for action in self._create_search_actions(prompt)])
            
        # Form filling actions
        if any(word in prompt_lower for word in ["fill", "input", "enter"]):
            actions.extend([action.to_dict() for action in self._create_form_actions(prompt)])
            
        # Navigation actions
        if any(word in prompt_lower for word in ["click", "navigate", "go to"]):
            actions.extend([action.to_dict() for action in self._create_navigation_actions(prompt)])
            
        return actions
    
    def _create_search_actions(self, prompt: str) -> List[BrowserAction]:
        """Create actions for search operations."""
        return [
            BrowserAction(
                action_type="type",
                selector="input[type='search']",
                value=self._extract_search_term(prompt),
                description="Enter search term"
            ),
            BrowserAction(
                action_type="click",
                selector="button[type='submit']",
                description="Submit search"
            )
        ]
    
    def _create_form_actions(self, prompt: str) -> List[BrowserAction]:
        """Create actions for form filling."""
        actions = []
        if "username" in prompt.lower():
            actions.append(BrowserAction(
                action_type="type",
                selector="input[type='text'], input[name='username']",
                value=self._extract_value(prompt, "username"),
                description="Enter username"
            ))
        if "password" in prompt.lower():
            actions.append(BrowserAction(
                action_type="type",
                selector="input[type='password']",
                value=self._extract_value(prompt, "password"),
                description="Enter password"
            ))
        return actions
    
    def _create_navigation_actions(self, prompt: str) -> List[BrowserAction]:
        """Create actions for navigation."""
        return [BrowserAction(
            action_type="click",
            selector=self._extract_link_selector(prompt),
            description="Navigate to target"
        )]
    
    def _extract_search_term(self, prompt: str) -> str:
        """Extract search term from prompt."""
        # Simple extraction - could be enhanced with NLP
        keywords = ["search for", "find", "look for"]
        for keyword in keywords:
            if keyword in prompt.lower():
                return prompt.lower().split(keyword)[1].strip().strip('"\'')
        return ""
    
    def _extract_value(self, prompt: str, field: str) -> str:
        """Extract value for a field from prompt."""
        # Simple extraction - could be enhanced with NLP
        if f"{field} " in prompt.lower():
            parts = prompt.lower().split(f"{field} ")
            if len(parts) > 1:
                return parts[1].split()[0].strip('"\'')
        return ""
    
    def _extract_link_selector(self, prompt: str) -> str:
        """Extract link selector from prompt."""
        # Simple extraction - could be enhanced with NLP
        if "click" in prompt.lower():
            text = prompt.lower().split("click")[1].strip().strip('"\'')
            return f"a:contains('{text}')"
        return "a"
