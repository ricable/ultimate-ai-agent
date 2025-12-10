"""
Document processors for extracting text from various file types.
"""
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import PyPDF2

# Set up logging
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Base class for document processors.
    """
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text content from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content as a string
        """
        raise NotImplementedError("Subclasses must implement extract_text method")
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary containing document metadata
        """
        # Basic metadata common to all file types
        path = Path(file_path)
        return {
            "filename": path.name,
            "file_extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "created_at": path.stat().st_ctime,
            "modified_at": path.stat().st_mtime
        }


class TxtProcessor(DocumentProcessor):
    """
    Processor for plain text files with robust error handling.
    """
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a TXT file with encoding fallbacks.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Extracted text content
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Try different encodings if UTF-8 fails
        encodings = ["utf-8", "latin-1", "cp1252", "ascii"]
        content = ""
        
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    content = file.read()
                logger.info(f"Successfully read text file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                logger.warning(f"Failed to decode with {encoding}, trying next encoding")
            except Exception as e:
                logger.error(f"Error reading file with {encoding}: {str(e)}")
                raise
        
        if not content:
            raise ValueError(f"Could not decode file with any of the attempted encodings")
            
        return content
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a TXT file.
        
        Args:
            file_path: Path to the TXT file
            
        Returns:
            Dictionary containing document metadata
        """
        metadata = super().get_metadata(file_path)
        metadata["content_type"] = "text/plain"
        metadata["processor"] = "TxtProcessor"
        
        # Count lines and words
        try:
            text = self.extract_text(file_path)
            metadata["line_count"] = len(text.splitlines())
            metadata["word_count"] = len(text.split())
        except Exception:
            # Don't fail metadata collection if text extraction fails
            pass
            
        return metadata


class PdfProcessor(DocumentProcessor):
    """
    Processor for PDF files with improved text extraction.
    """
    
    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file with page tracking.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract text from all pages with page numbers
                content = []
                total_pages = len(reader.pages)
                
                for page_num in range(total_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        
                        # Add page marker and text
                        if page_text and page_text.strip():
                            content.append(f"[Page {page_num + 1} of {total_pages}]\n{page_text}\n")
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        content.append(f"[Page {page_num + 1} of {total_pages} - Text extraction failed]\n")
                
                return "\n".join(content)
                
        except Exception as e:
            logger.error(f"Error processing PDF file: {str(e)}")
            raise
    
    def get_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Get metadata for a PDF file including PDF-specific properties.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary containing document metadata
        """
        metadata = super().get_metadata(file_path)
        metadata["content_type"] = "application/pdf"
        metadata["processor"] = "PdfProcessor"
        
        # Extract PDF-specific metadata
        try:
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                # Basic PDF properties
                metadata["page_count"] = len(reader.pages)
                
                # PDF document info if available
                if reader.metadata:
                    pdf_info = reader.metadata
                    if pdf_info.title:
                        metadata["title"] = pdf_info.title
                    if pdf_info.author:
                        metadata["author"] = pdf_info.author
                    if pdf_info.subject:
                        metadata["subject"] = pdf_info.subject
                    if pdf_info.creator:
                        metadata["creator"] = pdf_info.creator
                    if pdf_info.producer:
                        metadata["producer"] = pdf_info.producer
        except Exception as e:
            logger.warning(f"Error extracting PDF metadata: {str(e)}")
            
        return metadata


def get_document_processor(file_path: str) -> Optional[DocumentProcessor]:
    """
    Get the appropriate processor for a file based on its extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        DocumentProcessor instance for the file type or None if unsupported
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    
    processors = {
        ".txt": TxtProcessor(),
        ".pdf": PdfProcessor(),
        # Add more processors here as needed
    }
    
    processor = processors.get(extension)
    
    if processor:
        logger.info(f"Using {processor.__class__.__name__} for {path.name}")
        return processor
    else:
        logger.warning(f"Unsupported file type: {extension}")
        return None


def get_supported_extensions() -> List[str]:
    """
    Get a list of supported file extensions.
    
    Returns:
        List of supported file extensions
    """
    return [".txt", ".pdf"]
