"""
Main agent definition for the RAG AI agent.
"""
import os
import sys
from typing import List, Dict, Any, Optional, TypedDict
from pydantic_ai import Agent
from pydantic_ai.tools import Tool
from dotenv import load_dotenv
from pathlib import Path

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.tools import KnowledgeBaseSearch, KnowledgeBaseSearchParams, KnowledgeBaseSearchResult
from agent.prompts import RAG_SYSTEM_PROMPT

# Load environment variables from the project root .env file
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / '.env'

# Force override of existing environment variables
load_dotenv(dotenv_path, override=True)

class AgentDeps(TypedDict, total=False):
    """
    Dependencies for the RAG agent.
    """
    kb_search: KnowledgeBaseSearch


class RAGAgent:
    """
    RAG AI agent with knowledge base search capabilities.
    
    Args:
        model: OpenAI model to use. Defaults to OPENAI_MODEL env var.
        api_key: OpenAI API key. Defaults to OPENAI_API_KEY env var.
        kb_search: KnowledgeBaseSearch instance for searching the knowledge base.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        kb_search: Optional[KnowledgeBaseSearch] = None
    ):
        """
        Initialize the RAG agent.
        """
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either as an argument or environment variable.")
        
        # Initialize the knowledge base search tool
        self.kb_search = kb_search or KnowledgeBaseSearch()
        
        # Create the search tool
        self.search_tool = Tool(self.kb_search.search)
        
        # Initialize the Pydantic AI agent
        self.agent = Agent(
            f"openai:{self.model}",
            system_prompt=RAG_SYSTEM_PROMPT,
            tools=[self.search_tool]
        )
    
    async def query(
        self, 
        question: str, 
        max_results: int = 5,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG agent with a question.
        
        Args:
            question: The question to ask
            max_results: Maximum number of knowledge base results to retrieve
            source_filter: Optional filter to search only within a specific source
            
        Returns:
            Dictionary containing the agent's response and the knowledge base search results
        """
        # Create dependencies for the agent
        deps = AgentDeps(kb_search=self.kb_search)
        
        # Run the agent with the question
        result = await self.agent.run(
            question,
            deps=deps
        )
        
        # Get the agent's response
        response = result.output
        
        # Get the knowledge base search results from the tool calls
        kb_results = []
        for tool_call in result.tool_calls:
            if tool_call.tool.name == "search":
                kb_results = tool_call.result
        
        return {
            "response": response,
            "kb_results": kb_results
        }
    
    async def get_available_sources(self) -> List[str]:
        """
        Get a list of all available sources in the knowledge base.
        
        Returns:
            List of source identifiers
        """
        return await self.kb_search.get_available_sources()


# Create a singleton instance for easy import
agent = RAGAgent()
