from typing import List, Tuple, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
import re

class TokenWeightOutputParser:
    """Parser for extracting token weights from model output."""
    
    def __call__(self, text: str) -> List[Tuple[str, float]]:
        """Parse the model's output to extract token weights.
        
        Args:
            text: Raw text output from the model
            
        Returns:
            List of (token, weight) tuples
        """
        # Extract token-weight pairs from the format "token: weight"
        pattern = r'([^:]+):\s*([\d.]+)'
        matches = re.findall(pattern, text)
        return [(token.strip(), float(weight)) for token, weight in matches]

class PolarisWrapper:
    """Wrapper that adds PolarisOne capabilities to any LangChain chat model."""
    
    def __init__(self, base_model: BaseChatModel):
        """Initialize the wrapper with a base LangChain chat model.
        
        Args:
            base_model: The underlying LangChain chat model to wrap
        """
        self.base_model = base_model
        self.weight_parser = TokenWeightOutputParser()
        
        # Prompt for token weighting
        self.weight_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Analyze the following text and assign importance weights (0-1) to each significant word or phrase. 
Focus on key terms that contribute to the meaning. Format each weight as "term: weight".

Text: {text}

Weights:"""
        )
    
    def _get_token_weights(self, text: str) -> List[Tuple[str, float]]:
        """Get token weights for input text using the language model.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of (token, weight) tuples
        """
        # Create messages for token weighting
        prompt = self.weight_prompt.format(text=text)
        messages = [HumanMessage(content=prompt)]
        
        # Get weights from model
        weight_response = self.base_model.invoke(messages)
        return self.weight_parser(weight_response.content)
    
    def generate_with_weights(self, messages: List[BaseMessage]) -> Tuple[str, List[Tuple[str, float]]]:
        """Generate a response and return token weights for the input.
        
        Args:
            messages: List of chat messages
            
        Returns:
            Tuple of (response text, list of (token, weight) tuples for the last message)
        """
        # Get weights for the last message
        last_message = messages[-1].content
        token_weights = self._get_token_weights(last_message)
        
        # Generate response using base model
        # Filter out high-weight tokens for focused response
        high_weight_tokens = [token for token, weight in token_weights if weight > 0.5]
        focused_prompt = f"Focus on these key terms: {', '.join(high_weight_tokens)}.\n\nOriginal message: {last_message}"
        
        focused_messages = messages[:-1] + [HumanMessage(content=focused_prompt)]
        response = self.base_model.invoke(focused_messages)
        
        return response.content, token_weights

def create_polaris_model(base_model: BaseChatModel) -> PolarisWrapper:
    """Create a PolarisOne-enhanced model from a base LangChain chat model.
    
    Args:
        base_model: Base LangChain chat model to enhance
        
    Returns:
        PolarisWrapper: Enhanced model with PolarisOne capabilities
    """
    return PolarisWrapper(base_model)
