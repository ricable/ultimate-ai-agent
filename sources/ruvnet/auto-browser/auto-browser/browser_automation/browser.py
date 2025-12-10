"""Browser automation module for auto-browser."""

import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dotenv import load_dotenv
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from langchain_openai import ChatOpenAI
from rich.console import Console

console = Console()

class BrowserAutomation:
    """Handle browser automation and content extraction."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "output"):
        """Initialize browser automation.
        
        Args:
            config: Site configuration dictionary
            output_dir: Directory for output files
        """
        load_dotenv()
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def process_url(self, url: str, output_name: Optional[str] = None) -> Dict[str, Any]:
        """Process a URL and extract content.
        
        Args:
            url: URL to process
            output_name: Optional name for output file
            
        Returns:
            Dictionary of extracted content
        """
        # Always use headless mode in non-GUI environment
        browser = Browser(config=BrowserConfig(
            headless=True  # Force headless mode
        ))
        
        try:
            # Create task description
            task = self._create_task(url)
            
            # Get API key and model from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set")

            model = os.getenv('LLM_MODEL', 'gpt-4o-mini')

            # Create agent with explicit API key and model
            agent = Agent(
                task=task,
                llm=ChatOpenAI(
                    api_key=api_key,
                    model=model
                ),
                browser=browser,
                use_vision=True
            )
            
            # Run agent and get result
            result = await agent.run()
            
            # If there are interactive actions, create a new task for them
            if self.config.get('actions'):
                actions_desc = "\n".join([
                    f"- {action['action_type']}: {action.get('description', '')}"
                    for action in self.config['actions']
                ])
                
                interactive_task = f"""
                1. Navigate to '{url}'
                2. Perform these actions in sequence:
                {actions_desc}
                3. Extract and return the results
                """
                
                # Create new agent for interactive task
                interactive_agent = Agent(
                    task=interactive_task,
                    llm=ChatOpenAI(
                        api_key=api_key,
                        model=model
                    ),
                    browser=browser,
                    use_vision=True
                )
                
                # Run interactive task
                interactive_result = await interactive_agent.run()
                result = f"{result}\n\nInteractive Results:\n{interactive_result}"
                    
            # Structure the content
            structured_content = {
                'title': 'Analysis Result',
                'content': str(result),  # Convert result to string
                'metadata': {
                    'url': url,
                    'timestamp': datetime.now().isoformat(),
                    'description': self.config.get('description', '')
                }
            }
            
            return structured_content
            
        finally:
            await browser.close()
            
    def _create_task(self, url: str) -> str:
        """Create task description for the agent."""
        description = self.config.get('description', 'Extract content from the page')
        selectors = self.config.get('selectors', {})
        
        task_parts = [
            f"1. Navigate to '{url}'",
            "2. Extract the following content:",
            f"   {description}"  # Include the user's description in the task
        ]
        
        if selectors:
            for name, selector in selectors.items():
                task_parts.append(f"- Extract {name} using selector '{selector}'")
        else:
            # If no selectors, use AI to find relevant content
            task_parts.extend([
                "- Find and extract the main content",
                "- Extract any relevant data mentioned in the description",
                "- Include any important context or metadata"
            ])
            
        task_parts.append("3. Return the content in a JSON format with descriptive keys.")
        
        return "\n".join(task_parts)
            
    async def _extract_content(self, browser: Browser) -> str:
        """Extract content from the page."""
        try:
            # Get the page content directly from the agent's result
            return str(await browser.get_content())
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Failed to extract content: {e}")
            return "Failed to extract content"
