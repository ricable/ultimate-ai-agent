"""Flow4: Document Processing Pipeline

A comprehensive document processing pipeline that converts HTML and PDF documents 
to Markdown, cleans them, and chunks them for RAG applications using IBM Docling.
"""

__version__ = "0.2.0"
__author__ = "Flow4 Team"

from .core.pipeline import DocumentPipeline
from .core.converter import DocumentConverter
from .core.chunker import DocumentChunker
from .core.cleaner import HTMLCleaner

__all__ = [
    "DocumentPipeline",
    "DocumentConverter", 
    "DocumentChunker",
    "HTMLCleaner",
]