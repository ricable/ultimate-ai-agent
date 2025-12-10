"""Template generator for auto-browser."""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from browser_use import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from langchain_openai import ChatOpenAI

@dataclass
class Selector:
    """Represents a CSS selector with metadata."""
    css: str
    description: Optional[str] = None
    multiple: bool = False

@dataclass
class Template:
    """Represents a site template."""
    name: str
    description: str
    url_pattern: str
    selectors: Dict[str, Selector]

class TemplateGenerator:
    """Generate site templates by analyzing webpages."""
    
    async def create_template(self, url: str, name: str, description: str) -> Template:
        """Create a template by analyzing a webpage.
        
        Args:
            url: URL to analyze
            name: Template name
            description: Template description
            
        Returns:
            Generated template
        """
        browser = Browser(config=BrowserConfig(headless=True))
        
        try:
            # Create task for analyzing the page
            task = f"""
            1. Navigate to '{url}'
            2. Analyze the page structure and identify:
               - Main content sections
               - Interactive elements (forms, buttons)
               - Important data elements
            3. Return a list of CSS selectors for extracting content
            """
            
            # Create and run agent
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
            
            # Get analysis result
            result = await agent.run()
            
            # Create selectors from analysis
            selectors = self._create_selectors(result)
            
            return Template(
                name=name,
                description=description,
                url_pattern=url,
                selectors=selectors
            )
            
        finally:
            await browser.close()
    
    def _create_selectors(self, analysis: str) -> Dict[str, Selector]:
        """Create selectors from analysis result."""
        selectors = {}
        
        # Common selectors for finance data
        selectors.update({
            'stock_price': Selector(
                css='.YMlKec.fxKbKc',
                description='Current stock price',
                multiple=False
            ),
            'price_change': Selector(
                css='.P2Luy.Ez2Ioe',
                description='Price change',
                multiple=False
            ),
            'company_name': Selector(
                css='.zzDege',
                description='Company name',
                multiple=False
            ),
            'market_cap': Selector(
                css='[data-metric="MARKET_CAP"] .P6K39c',
                description='Market capitalization',
                multiple=False
            ),
            'volume': Selector(
                css='[data-metric="VOLUME"] .P6K39c',
                description='Trading volume',
                multiple=False
            ),
            'news_items': Selector(
                css='.yY3Lee',
                description='News articles',
                multiple=True
            ),
            'search_input': Selector(
                css='.Ax4B8.ZAGvjd',
                description='Search input field',
                multiple=False
            )
        })
        
        return selectors
    
    def save_template(self, template: Template, output_path: str = 'config.yaml') -> None:
        """Save template to a file.
        
        Args:
            template: Template to save
            output_path: Path to save to (defaults to config.yaml)
        """
        # Create config structure
        config = {
            'sites': {
                template.name: {
                    'name': template.name,
                    'description': template.description,
                    'url_pattern': template.url_pattern,
                    'selectors': {
                        name: {
                            'css': sel.css,
                            'description': sel.description,
                            'multiple': sel.multiple
                        }
                        for name, sel in template.selectors.items()
                    }
                }
            }
        }
        
        # Handle file saving
        try:
            # Create parent directories if needed
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Update existing config if it exists
            if os.path.exists(output_path):
                with open(output_path) as f:
                    existing_config = yaml.safe_load(f) or {}
                if 'sites' in existing_config:
                    existing_config['sites'].update(config['sites'])
                    config = existing_config
            
            # Save to file
            with open(output_path, 'w') as f:
                yaml.dump(config, f, sort_keys=False)
                
        except Exception as e:
            raise RuntimeError(f"Failed to save template: {e}")
