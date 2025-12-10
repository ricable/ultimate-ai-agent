"""Tool for enhancing LLM interactions with PolarisOne's token weighting capabilities."""

from typing import List, Dict, Any
from langchain_core.messages import HumanMessage
from ..llm import initialize_llm
from ..polaris import create_polaris_model

class PolarisTool:
    """Tool that adds PolarisOne's token weighting capabilities to LLM interactions."""
    
    def __init__(self):
        """Initialize the tool with a base LLM."""
        base_model = initialize_llm("openai", "gpt-3.5-turbo")
        self.polaris_model = create_polaris_model(base_model)
    
    def __call__(self, text: str) -> Dict[str, Any]:
        """Process text using PolarisOne's token weighting.
        
        Args:
            text: Input text to process
            
        Returns:
            Dict containing:
                response: Generated response text
                token_weights: List of (token, weight) tuples
        """
        messages = [HumanMessage(content=text)]
        response, token_weights = self.polaris_model.generate_with_weights(messages)
        
        return {
            "response": response,
            "token_weights": token_weights
        }

# Create tool instance
tool = PolarisTool()
