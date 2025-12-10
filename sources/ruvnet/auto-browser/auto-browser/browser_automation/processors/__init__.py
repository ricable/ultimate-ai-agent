"""Processors package for auto-browser."""

from .content import ContentProcessor, PageElement
from .interactive import InteractiveProcessor, BrowserAction

__all__ = ['ContentProcessor', 'PageElement', 'InteractiveProcessor', 'BrowserAction']
