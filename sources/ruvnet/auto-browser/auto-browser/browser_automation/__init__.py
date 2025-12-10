"""Browser automation package for web scraping and interaction."""

import logging

def configure_logging(verbose: bool = False):
    """Configure logging based on verbosity."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(message)s',  # Only show the message, no timestamp or level
        force=True  # Override any existing configuration
    )

# Set default logging to warnings
configure_logging(False)

# Import main components
from .browser import BrowserAutomation
from .template_generator import TemplateGenerator, Template, Selector
from .processors.content import ContentProcessor, PageElement
from .processors.interactive import InteractiveProcessor, BrowserAction
from .formatters.markdown import MarkdownFormatter

__all__ = [
    'BrowserAutomation',
    'TemplateGenerator',
    'Template',
    'Selector',
    'ContentProcessor',
    'PageElement',
    'InteractiveProcessor',
    'BrowserAction',
    'MarkdownFormatter'
]
