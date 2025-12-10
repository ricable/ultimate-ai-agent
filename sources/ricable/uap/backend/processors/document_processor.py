# File: backend/processors/document_processor.py
"""
Document Processing Core Module
Handles document processing using Docling and provides a unified interface
for document analysis, extraction, and processing capabilities.
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import logging
from dataclasses import dataclass, asdict

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.document import ConversionResult
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    ConversionResult = Any  # Fallback type for when docling is not available
    logging.warning("Docling not available. Document processing will use fallback mode.")

@dataclass
class DocumentMetadata:
    """Metadata structure for processed documents"""
    id: str
    filename: str
    original_size: int
    processed_size: Optional[int]
    content_type: str
    upload_timestamp: datetime
    processing_timestamp: Optional[datetime]
    status: str  # 'uploaded', 'processing', 'completed', 'error'
    error_message: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = None

@dataclass 
class DocumentContent:
    """Processed document content structure"""
    text: str
    markdown: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None
    structure: Optional[Dict[str, Any]] = None

class DocumentProcessor:
    """
    Core document processor using Docling for document analysis and extraction.
    Provides unified interface for processing various document formats.
    """
    
    def __init__(self, storage_path: str = "documents", max_file_size: int = 50 * 1024 * 1024):
        """
        Initialize the document processor.
        
        Args:
            storage_path: Path to store processed documents
            max_file_size: Maximum file size in bytes (default: 50MB)
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.max_file_size = max_file_size
        self.logger = logging.getLogger(__name__)
        
        # Initialize Docling converter if available
        if DOCLING_AVAILABLE:
            self.converter = DocumentConverter()
            self.pdf_options = PdfPipelineOptions()
            self.pdf_options.do_ocr = True
            self.pdf_options.do_table_structure = True
        else:
            self.converter = None
            self.logger.warning("Docling not available - using fallback text extraction")
        
        # Supported file formats
        self.supported_formats = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.html': 'text/html',
            '.htm': 'text/html',
            '.rtf': 'application/rtf',
            '.odt': 'application/vnd.oasis.opendocument.text'
        }
    
    def validate_file(self, filename: str, file_size: int) -> Dict[str, Any]:
        """
        Validate uploaded file format and size.
        
        Args:
            filename: Name of the file
            file_size: Size of the file in bytes
            
        Returns:
            Dict with validation results
        """
        file_ext = Path(filename).suffix.lower()
        
        # Check file extension
        if file_ext not in self.supported_formats:
            return {
                "valid": False,
                "error": f"Unsupported file format: {file_ext}",
                "supported_formats": list(self.supported_formats.keys())
            }
        
        # Check file size
        if file_size > self.max_file_size:
            return {
                "valid": False,
                "error": f"File size ({file_size} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)"
            }
        
        return {
            "valid": True,
            "content_type": self.supported_formats[file_ext]
        }
    
    async def process_document(self, file_path: Union[str, Path], original_filename: str) -> DocumentMetadata:
        """
        Process a document file and extract content.
        
        Args:
            file_path: Path to the document file
            original_filename: Original filename of the document
            
        Returns:
            DocumentMetadata object with processing results
        """
        file_path = Path(file_path)
        doc_id = str(uuid.uuid4())
        
        # Initialize metadata
        metadata = DocumentMetadata(
            id=doc_id,
            filename=original_filename,
            original_size=file_path.stat().st_size,
            processed_size=None,
            content_type=self.supported_formats.get(file_path.suffix.lower(), 'application/octet-stream'),
            upload_timestamp=datetime.utcnow(),
            processing_timestamp=None,
            status='processing'
        )
        
        try:
            self.logger.info(f"Starting document processing for {original_filename}")
            
            # Process document based on availability of Docling
            if DOCLING_AVAILABLE and self.converter:
                content = await self._process_with_docling(file_path, metadata)
            else:
                content = await self._process_fallback(file_path, metadata)
            
            # Save processed content
            content_path = self.storage_path / f"{doc_id}.json"
            with open(content_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(content), f, indent=2, default=str)
            
            # Update metadata
            metadata.processed_size = content_path.stat().st_size
            metadata.processing_timestamp = datetime.utcnow()
            metadata.status = 'completed'
            metadata.word_count = len(content.text.split()) if content.text else 0
            
            # Save metadata
            metadata_path = self.storage_path / f"{doc_id}_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metadata), f, indent=2, default=str)
            
            self.logger.info(f"Document processing completed for {original_filename}")
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error processing document {original_filename}: {str(e)}")
            metadata.status = 'error'
            metadata.error_message = str(e)
            metadata.processing_timestamp = datetime.utcnow()
            return metadata
    
    async def _process_with_docling(self, file_path: Path, metadata: DocumentMetadata) -> DocumentContent:
        """Process document using Docling library."""
        try:
            # Run Docling conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._convert_with_docling, 
                str(file_path)
            )
            
            # Extract content from Docling result
            text_content = result.document.export_to_text()
            markdown_content = result.document.export_to_markdown()
            
            # Extract metadata
            doc_metadata = {
                "title": getattr(result.document, 'title', None),
                "author": getattr(result.document, 'author', None),
                "creation_date": getattr(result.document, 'creation_date', None),
                "modification_date": getattr(result.document, 'modification_date', None),
                "subject": getattr(result.document, 'subject', None)
            }
            
            # Extract tables if available
            tables = []
            if hasattr(result.document, 'tables'):
                for table in result.document.tables:
                    tables.append({
                        "id": getattr(table, 'id', None),
                        "data": getattr(table, 'data', None),
                        "bbox": getattr(table, 'bbox', None)
                    })
            
            # Extract images if available
            images = []
            if hasattr(result.document, 'images'):
                for image in result.document.images:
                    images.append({
                        "id": getattr(image, 'id', None),
                        "bbox": getattr(image, 'bbox', None),
                        "alt_text": getattr(image, 'alt_text', None)
                    })
            
            # Update metadata with page count
            if hasattr(result.document, 'pages'):
                metadata.page_count = len(result.document.pages)
            
            return DocumentContent(
                text=text_content,
                markdown=markdown_content,
                metadata=doc_metadata,
                tables=tables if tables else None,
                images=images if images else None,
                structure={"docling_processed": True}
            )
            
        except Exception as e:
            self.logger.error(f"Docling processing failed: {str(e)}")
            # Fall back to basic text extraction
            return await self._process_fallback(file_path, metadata)
    
    def _convert_with_docling(self, file_path: str) -> ConversionResult:
        """Synchronous Docling conversion (runs in thread pool)."""
        return self.converter.convert(file_path)
    
    async def _process_fallback(self, file_path: Path, metadata: DocumentMetadata) -> DocumentContent:
        """Fallback processing for when Docling is not available."""
        try:
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            elif file_path.suffix.lower() == '.md':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
            else:
                # For other formats, return a placeholder
                text_content = f"[Document content extraction not available for {file_path.suffix} files without Docling]"
            
            return DocumentContent(
                text=text_content,
                metadata={"fallback_processed": True},
                structure={"fallback_mode": True}
            )
            
        except Exception as e:
            self.logger.error(f"Fallback processing failed: {str(e)}")
            return DocumentContent(
                text="[Error: Could not extract document content]",
                metadata={"error": str(e)},
                structure={"error": True}
            )
    
    async def get_document_content(self, doc_id: str) -> Optional[DocumentContent]:
        """
        Retrieve processed document content by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentContent object or None if not found
        """
        try:
            content_path = self.storage_path / f"{doc_id}.json"
            if not content_path.exists():
                return None
            
            with open(content_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            
            return DocumentContent(**content_data)
            
        except Exception as e:
            self.logger.error(f"Error retrieving document content {doc_id}: {str(e)}")
            return None
    
    async def get_document_metadata(self, doc_id: str) -> Optional[DocumentMetadata]:
        """
        Retrieve document metadata by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            DocumentMetadata object or None if not found
        """
        try:
            metadata_path = self.storage_path / f"{doc_id}_metadata.json"
            if not metadata_path.exists():
                return None
            
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata_data = json.load(f)
            
            # Convert timestamp strings back to datetime objects
            for field in ['upload_timestamp', 'processing_timestamp']:
                if metadata_data.get(field):
                    metadata_data[field] = datetime.fromisoformat(metadata_data[field].replace('Z', '+00:00'))
            
            return DocumentMetadata(**metadata_data)
            
        except Exception as e:
            self.logger.error(f"Error retrieving document metadata {doc_id}: {str(e)}")
            return None
    
    async def list_documents(self, limit: int = 50, offset: int = 0) -> List[DocumentMetadata]:
        """
        List all processed documents with pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            
        Returns:
            List of DocumentMetadata objects
        """
        try:
            metadata_files = list(self.storage_path.glob("*_metadata.json"))
            metadata_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            documents = []
            for metadata_file in metadata_files[offset:offset + limit]:
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata_data = json.load(f)
                    
                    # Convert timestamp strings back to datetime objects
                    for field in ['upload_timestamp', 'processing_timestamp']:
                        if metadata_data.get(field):
                            metadata_data[field] = datetime.fromisoformat(metadata_data[field].replace('Z', '+00:00'))
                    
                    documents.append(DocumentMetadata(**metadata_data))
                except Exception as e:
                    self.logger.error(f"Error loading metadata from {metadata_file}: {str(e)}")
                    continue
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error listing documents: {str(e)}")
            return []
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a processed document and its metadata.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            content_path = self.storage_path / f"{doc_id}.json"
            metadata_path = self.storage_path / f"{doc_id}_metadata.json"
            
            deleted = False
            if content_path.exists():
                content_path.unlink()
                deleted = True
            
            if metadata_path.exists():
                metadata_path.unlink()
                deleted = True
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False