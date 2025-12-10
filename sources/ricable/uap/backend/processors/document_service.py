# File: backend/processors/document_service.py
"""
Document Service
High-level service that orchestrates document processing operations.
Combines document storage and processing capabilities with integration
to the Agno framework for document analysis.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime

from .document_processor import DocumentProcessor, DocumentMetadata, DocumentContent
from .document_storage import DocumentStorageManager, UploadResult

class DocumentService:
    """
    High-level document service that orchestrates the complete document processing pipeline.
    Handles uploads, processing, storage, and retrieval of documents.
    """
    
    def __init__(self, 
                 upload_path: str = "uploads",
                 storage_path: str = "documents",
                 max_file_size: int = 50 * 1024 * 1024):
        """
        Initialize the document service.
        
        Args:
            upload_path: Path for temporary file uploads
            storage_path: Path for processed documents
            max_file_size: Maximum file size in bytes
        """
        self.storage_manager = DocumentStorageManager(
            upload_path=upload_path,
            processed_path=storage_path,
            max_file_size=max_file_size
        )
        self.processor = DocumentProcessor(
            storage_path=storage_path,
            max_file_size=max_file_size
        )
        self.logger = logging.getLogger(__name__)
        
        # Task tracking for async processing
        self.processing_tasks: Dict[str, asyncio.Task] = {}
    
    async def initialize(self):
        """Initialize the document service."""
        await self.storage_manager.start_cleanup_scheduler()
        self.logger.info("Document service initialized")
    
    async def upload_document(self, 
                            file_data: bytes, 
                            filename: str, 
                            content_type: Optional[str] = None,
                            process_immediately: bool = True) -> Dict[str, Any]:
        """
        Upload and optionally process a document.
        
        Args:
            file_data: Raw file data
            filename: Original filename
            content_type: MIME type of the file
            process_immediately: Whether to start processing immediately
            
        Returns:
            Dict with upload and processing results
        """
        try:
            # Upload file to temporary storage
            upload_result = await self.storage_manager.save_uploaded_file(
                file_data, filename, content_type
            )
            
            if not upload_result.success:
                return {
                    "success": False,
                    "error": upload_result.error_message,
                    "file_id": upload_result.file_id
                }
            
            result = {
                "success": True,
                "file_id": upload_result.file_id,
                "filename": upload_result.original_filename,
                "file_size": upload_result.file_size,
                "content_type": upload_result.content_type,
                "checksum": upload_result.checksum,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "processing_status": "pending"
            }
            
            # Start processing if requested
            if process_immediately:
                processing_task = asyncio.create_task(
                    self._process_document_async(upload_result.file_id, upload_result.file_path, filename)
                )
                self.processing_tasks[upload_result.file_id] = processing_task
                result["processing_status"] = "started"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error uploading document {filename}: {str(e)}")
            return {
                "success": False,
                "error": f"Upload failed: {str(e)}",
                "filename": filename
            }
    
    async def _process_document_async(self, file_id: str, file_path: str, original_filename: str):
        """
        Asynchronously process a document.
        
        Args:
            file_id: File ID
            file_path: Path to the uploaded file
            original_filename: Original filename
        """
        try:
            self.logger.info(f"Starting async processing for {original_filename}")
            
            # Process the document
            metadata = await self.processor.process_document(file_path, original_filename)
            
            # Clean up uploaded file after processing
            await self.storage_manager.delete_file(file_id)
            
            self.logger.info(f"Async processing completed for {original_filename}")
            
        except Exception as e:
            self.logger.error(f"Error in async processing for {original_filename}: {str(e)}")
        finally:
            # Remove from tracking
            if file_id in self.processing_tasks:
                del self.processing_tasks[file_id]
    
    async def process_document_sync(self, 
                                  file_data: bytes, 
                                  filename: str, 
                                  content_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload and process a document synchronously.
        
        Args:
            file_data: Raw file data
            filename: Original filename
            content_type: MIME type of the file
            
        Returns:
            Dict with complete processing results
        """
        try:
            # Upload file
            upload_result = await self.storage_manager.save_uploaded_file(
                file_data, filename, content_type
            )
            
            if not upload_result.success:
                return {
                    "success": False,
                    "error": upload_result.error_message,
                    "file_id": upload_result.file_id
                }
            
            # Process immediately
            metadata = await self.processor.process_document(
                upload_result.file_path, upload_result.original_filename
            )
            
            # Clean up uploaded file
            await self.storage_manager.delete_file(upload_result.file_id)
            
            # Get processed content
            content = None
            if metadata.status == 'completed':
                content = await self.processor.get_document_content(metadata.id)
            
            return {
                "success": True,
                "document_id": metadata.id,
                "filename": metadata.filename,
                "file_size": metadata.original_size,
                "processed_size": metadata.processed_size,
                "content_type": metadata.content_type,
                "upload_timestamp": metadata.upload_timestamp.isoformat(),
                "processing_timestamp": metadata.processing_timestamp.isoformat() if metadata.processing_timestamp else None,
                "status": metadata.status,
                "error_message": metadata.error_message,
                "page_count": metadata.page_count,
                "word_count": metadata.word_count,
                "language": metadata.language,
                "content_preview": content.text[:500] + "..." if content and content.text and len(content.text) > 500 else content.text if content else None,
                "has_tables": bool(content.tables) if content else False,
                "has_images": bool(content.images) if content else False
            }
            
        except Exception as e:
            self.logger.error(f"Error in sync processing for {filename}: {str(e)}")
            return {
                "success": False,
                "error": f"Processing failed: {str(e)}",
                "filename": filename
            }
    
    async def get_document(self, doc_id: str, include_content: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document ID
            include_content: Whether to include full content
            
        Returns:
            Document data or None if not found
        """
        try:
            metadata = await self.processor.get_document_metadata(doc_id)
            if not metadata:
                return None
            
            result = {
                "document_id": metadata.id,
                "filename": metadata.filename,
                "file_size": metadata.original_size,
                "processed_size": metadata.processed_size,
                "content_type": metadata.content_type,
                "upload_timestamp": metadata.upload_timestamp.isoformat(),
                "processing_timestamp": metadata.processing_timestamp.isoformat() if metadata.processing_timestamp else None,
                "status": metadata.status,
                "error_message": metadata.error_message,
                "page_count": metadata.page_count,
                "word_count": metadata.word_count,
                "language": metadata.language
            }
            
            if include_content:
                content = await self.processor.get_document_content(doc_id)
                if content:
                    result.update({
                        "text": content.text,
                        "markdown": content.markdown,
                        "metadata": content.metadata,
                        "tables": content.tables,
                        "images": content.images,
                        "structure": content.structure
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error retrieving document {doc_id}: {str(e)}")
            return None
    
    async def list_documents(self, 
                           limit: int = 50, 
                           offset: int = 0,
                           status_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        List documents with pagination and filtering.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            status_filter: Filter by status ('completed', 'processing', 'error')
            
        Returns:
            Dict with documents list and pagination info
        """
        try:
            documents = await self.processor.list_documents(limit, offset)
            
            # Apply status filter if specified
            if status_filter:
                documents = [doc for doc in documents if doc.status == status_filter]
            
            # Convert to dict format
            document_list = []
            for metadata in documents:
                document_list.append({
                    "document_id": metadata.id,
                    "filename": metadata.filename,
                    "file_size": metadata.original_size,
                    "processed_size": metadata.processed_size,
                    "content_type": metadata.content_type,
                    "upload_timestamp": metadata.upload_timestamp.isoformat(),
                    "processing_timestamp": metadata.processing_timestamp.isoformat() if metadata.processing_timestamp else None,
                    "status": metadata.status,
                    "error_message": metadata.error_message,
                    "page_count": metadata.page_count,
                    "word_count": metadata.word_count,
                    "language": metadata.language
                })
            
            return {
                "documents": document_list,
                "total": len(document_list),
                "limit": limit,
                "offset": offset,
                "has_more": len(document_list) == limit
            }
            
        except Exception as e:
            self.logger.error(f"Error listing documents: {str(e)}")
            return {
                "documents": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
                "error": str(e)
            }
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Delete a document and its associated data.
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Dict with deletion results
        """
        try:
            # Cancel processing task if running
            if doc_id in self.processing_tasks:
                self.processing_tasks[doc_id].cancel()
                del self.processing_tasks[doc_id]
            
            # Delete processed document
            deleted = await self.processor.delete_document(doc_id)
            
            if deleted:
                return {
                    "success": True,
                    "message": f"Document {doc_id} deleted successfully"
                }
            else:
                return {
                    "success": False,
                    "error": f"Document {doc_id} not found"
                }
                
        except Exception as e:
            self.logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to delete document: {str(e)}"
            }
    
    async def get_processing_status(self, file_id: str) -> Dict[str, Any]:
        """
        Get the processing status of a document.
        
        Args:
            file_id: File ID or document ID
            
        Returns:
            Dict with processing status
        """
        try:
            # Check if currently processing
            if file_id in self.processing_tasks:
                task = self.processing_tasks[file_id]
                if task.done():
                    return {"status": "completed", "processing": False}
                else:
                    return {"status": "processing", "processing": True}
            
            # Check processed documents
            metadata = await self.processor.get_document_metadata(file_id)
            if metadata:
                return {
                    "status": metadata.status,
                    "processing": False,
                    "error_message": metadata.error_message
                }
            
            return {"status": "not_found", "processing": False}
            
        except Exception as e:
            self.logger.error(f"Error getting processing status for {file_id}: {str(e)}")
            return {
                "status": "error",
                "processing": False,
                "error_message": str(e)
            }
    
    async def analyze_document_with_agno(self, doc_id: str, analysis_type: str = "general") -> Dict[str, Any]:
        """
        Analyze a document using the Agno framework.
        
        Args:
            doc_id: Document ID
            analysis_type: Type of analysis to perform
            
        Returns:
            Dict with analysis results
        """
        try:
            # Get document content and metadata
            content = await self.processor.get_document_content(doc_id)
            metadata = await self.processor.get_document_metadata(doc_id)
            
            if not content or not metadata:
                return {
                    "success": False,
                    "error": f"Document {doc_id} not found"
                }
            
            # Import AgnoAgentManager here to avoid circular imports
            from ..frameworks.agno.agent import AgnoAgentManager
            
            # Initialize Agno agent if not already done
            agno_manager = AgnoAgentManager()
            if not agno_manager.is_initialized:
                await agno_manager.initialize()
            
            # Prepare context for Agno framework
            document_context = {
                "document_id": doc_id,
                "document_type": self._determine_document_type(metadata.content_type, metadata.filename),
                "filename": metadata.filename,
                "file_size": metadata.original_size,
                "page_count": metadata.page_count,
                "word_count": metadata.word_count,
                "upload_timestamp": metadata.upload_timestamp.isoformat() if metadata.upload_timestamp else None,
                "processing_timestamp": metadata.processing_timestamp.isoformat() if metadata.processing_timestamp else None
            }
            
            # Use the specialized document processing method
            agno_response = await agno_manager.process_document(
                document_content=content.text or "",
                document_type=document_context["document_type"],
                analysis_type=analysis_type
            )
            
            # Process and enhance the Agno response
            enhanced_results = self._enhance_agno_results(
                agno_response, 
                content, 
                metadata, 
                analysis_type
            )
            
            return {
                "success": True,
                "document_id": doc_id,
                "analysis_type": analysis_type,
                "analysis_results": enhanced_results,
                "document_metadata": document_context,
                "agno_metadata": agno_response.get("metadata", {}),
                "processed_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {doc_id} with Agno: {str(e)}")
            return {
                "success": False,
                "error": f"Analysis failed: {str(e)}",
                "document_id": doc_id,
                "analysis_type": analysis_type
            }
    
    def _determine_document_type(self, content_type: str, filename: str) -> str:
        """
        Determine the document type for Agno processing.
        
        Args:
            content_type: MIME type of the document
            filename: Original filename
            
        Returns:
            Document type string for Agno
        """
        if 'pdf' in content_type.lower():
            return 'pdf'
        elif 'word' in content_type.lower() or filename.lower().endswith(('.doc', '.docx')):
            return 'document'
        elif 'text' in content_type.lower() or filename.lower().endswith('.txt'):
            return 'text'
        elif 'markdown' in content_type.lower() or filename.lower().endswith('.md'):
            return 'markdown'
        elif 'html' in content_type.lower() or filename.lower().endswith(('.html', '.htm')):
            return 'html'
        else:
            return 'document'
    
    def _enhance_agno_results(self, agno_response: Dict[str, Any], content: 'DocumentContent', 
                             metadata: 'DocumentMetadata', analysis_type: str) -> Dict[str, Any]:
        """
        Enhance Agno analysis results with additional document insights.
        
        Args:
            agno_response: Response from Agno framework
            content: Document content
            metadata: Document metadata
            analysis_type: Type of analysis performed
            
        Returns:
            Enhanced analysis results
        """
        try:
            # Base results from Agno
            results = {
                "agno_analysis": agno_response.get("content", ""),
                "framework_used": "agno",
                "analysis_type": analysis_type,
                "processing_mode": agno_response.get("metadata", {}).get("processing_mode", "unknown")
            }
            
            # Add document statistics
            if content.text:
                text = content.text
                results.update({
                    "document_statistics": {
                        "character_count": len(text),
                        "word_count": len(text.split()),
                        "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
                        "line_count": len(text.split('\n'))
                    }
                })
            
            # Add structure information
            if content.structure:
                results["document_structure"] = content.structure
            
            # Add table information
            if content.tables:
                results["tables_detected"] = {
                    "count": len(content.tables),
                    "tables": content.tables
                }
            
            # Add image information
            if content.images:
                results["images_detected"] = {
                    "count": len(content.images),
                    "images": content.images
                }
            
            # Add metadata insights
            results["document_metadata"] = {
                "filename": metadata.filename,
                "file_size": metadata.original_size,
                "content_type": metadata.content_type,
                "page_count": metadata.page_count,
                "processing_status": metadata.status,
                "upload_date": metadata.upload_timestamp.isoformat() if metadata.upload_timestamp else None,
                "processing_date": metadata.processing_timestamp.isoformat() if metadata.processing_timestamp else None
            }
            
            # Add Docling-specific information if available
            if content.metadata:
                results["docling_metadata"] = content.metadata
            
            # Analysis type specific enhancements
            if analysis_type == "summary" and content.text:
                # Add quick statistics for summary
                results["summary_statistics"] = {
                    "original_length": len(content.text),
                    "compression_ratio": len(agno_response.get("content", "")) / len(content.text) if content.text else 0
                }
            
            elif analysis_type == "extraction" and (content.tables or content.structure):
                # Add extraction insights
                results["extraction_summary"] = {
                    "structured_data_found": bool(content.tables or content.structure),
                    "table_count": len(content.tables) if content.tables else 0,
                    "has_document_structure": bool(content.structure)
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error enhancing Agno results: {str(e)}")
            # Return basic results if enhancement fails
            return {
                "agno_analysis": agno_response.get("content", ""),
                "framework_used": "agno",
                "analysis_type": analysis_type,
                "enhancement_error": str(e)
            }
    
    async def get_service_status(self) -> Dict[str, Any]:
        """
        Get the current status of the document service.
        
        Returns:
            Dict with service status information
        """
        try:
            active_tasks = len(self.processing_tasks)
            
            # Get basic statistics
            documents = await self.processor.list_documents(limit=1000)
            total_documents = len(documents)
            completed_documents = len([d for d in documents if d.status == 'completed'])
            error_documents = len([d for d in documents if d.status == 'error'])
            
            return {
                "service": "document_processing",
                "status": "active",
                "active_processing_tasks": active_tasks,
                "statistics": {
                    "total_documents": total_documents,
                    "completed_documents": completed_documents,
                    "error_documents": error_documents,
                    "success_rate": (completed_documents / total_documents * 100) if total_documents > 0 else 0
                },
                "capabilities": {
                    "docling_available": hasattr(self.processor, 'converter') and self.processor.converter is not None,
                    "supported_formats": list(self.processor.supported_formats.keys()),
                    "max_file_size": self.processor.max_file_size
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting service status: {str(e)}")
            return {
                "service": "document_processing",
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }