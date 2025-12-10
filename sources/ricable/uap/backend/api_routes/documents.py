# File: backend/api_routes/documents.py
"""
Document Processing API Routes
Provides endpoints for document upload, processing, retrieval, and analysis
using the Docling integration and document processing services.
"""

import os
import uuid
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Path, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import document processing services
try:
    from ..processors.document_service import DocumentService
    from ..processors.document_processor import DocumentMetadata, DocumentContent
    DOCUMENT_SERVICE_AVAILABLE = True
except ImportError:
    DOCUMENT_SERVICE_AVAILABLE = False

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Global document service instance
document_service = None
if DOCUMENT_SERVICE_AVAILABLE:
    document_service = DocumentService(
        upload_path="uploads",
        storage_path="documents",
        max_file_size=50 * 1024 * 1024  # 50MB
    )

# Models
class DocumentAnalysisRequest(BaseModel):
    analysis_type: str = "general"  # "general", "summary", "extraction", "classification"
    
class DocumentDeleteResponse(BaseModel):
    success: bool
    message: str

# Initialize document service
@router.on_event("startup")
async def startup_event():
    if document_service:
        await document_service.initialize()
        print("Document processing service initialized")

# Document upload endpoint
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    process_immediately: bool = Form(True)
):
    """
    Upload a document file for processing.
    
    Args:
        file: The document file to upload
        process_immediately: Whether to start processing immediately
        
    Returns:
        Upload result with document ID and processing status
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        # Read file data
        file_data = await file.read()
        
        # Validate file
        if len(file_data) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # Upload and process document
        if process_immediately:
            result = await document_service.process_document_sync(
                file_data=file_data,
                filename=file.filename or "unknown",
                content_type=file.content_type
            )
        else:
            result = await document_service.upload_document(
                file_data=file_data,
                filename=file.filename or "unknown", 
                content_type=file.content_type,
                process_immediately=False
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# List documents endpoint
@router.get("/")
async def list_documents(
    limit: int = Query(50, description="Maximum number of documents to return"),
    offset: int = Query(0, description="Number of documents to skip"),
    status: Optional[str] = Query(None, description="Filter by status: completed, processing, error")
):
    """
    List uploaded documents with pagination and optional status filtering.
    
    Args:
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        status: Optional status filter
        
    Returns:
        List of documents with metadata
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        result = await document_service.list_documents(
            limit=limit,
            offset=offset,
            status_filter=status
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

# Get document details endpoint
@router.get("/{document_id}")
async def get_document(
    document_id: str = Path(..., description="Document ID"),
    include_content: bool = Query(False, description="Include full document content")
):
    """
    Get detailed information about a specific document.
    
    Args:
        document_id: The document ID to retrieve
        include_content: Whether to include full content
        
    Returns:
        Document metadata and optionally content
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        document = await document_service.get_document(document_id, include_content)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")

# Get document content endpoint
@router.get("/{document_id}/content")
async def get_document_content(document_id: str = Path(..., description="Document ID")):
    """
    Get the full processed content of a document.
    
    Args:
        document_id: The document ID
        
    Returns:
        Full document content including text, markdown, tables, etc.
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        document = await document_service.get_document(document_id, include_content=True)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        
        # Return only the content fields
        content_fields = ['text', 'markdown', 'metadata', 'tables', 'images', 'structure']
        content = {field: document.get(field) for field in content_fields if field in document}
        
        return {
            "document_id": document_id,
            "content": content,
            "status": document.get('status'),
            "processing_timestamp": document.get('processing_timestamp')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document content: {str(e)}")

# Get document processing status
@router.get("/{document_id}/status")
async def get_document_status(document_id: str = Path(..., description="Document ID")):
    """
    Get the processing status of a document.
    
    Args:
        document_id: The document ID
        
    Returns:
        Processing status and progress information
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        status = await document_service.get_processing_status(document_id)
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get processing status: {str(e)}")

# Analyze document with Agno endpoint
@router.post("/{document_id}/analyze")
async def analyze_document(
    document_id: str = Path(..., description="Document ID"),
    request: DocumentAnalysisRequest = DocumentAnalysisRequest()
):
    """
    Analyze a document using the Agno framework.
    
    Args:
        document_id: The document ID to analyze
        request: Analysis configuration
        
    Returns:
        Analysis results from Agno framework
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        # Check if document exists
        document = await document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")
        
        # Ensure document is processed
        if document.get('status') != 'completed':
            raise HTTPException(
                status_code=400, 
                detail=f"Document processing not completed. Current status: {document.get('status')}"
            )
        
        # Analyze with Agno
        analysis_result = await document_service.analyze_document_with_agno(
            doc_id=document_id,
            analysis_type=request.analysis_type
        )
        
        return analysis_result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document analysis failed: {str(e)}")

# Delete document endpoint
@router.delete("/{document_id}")
async def delete_document(document_id: str = Path(..., description="Document ID")):
    """
    Delete a document and its associated data.
    
    Args:
        document_id: The document ID to delete
        
    Returns:
        Deletion confirmation
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        result = await document_service.delete_document(document_id)
        
        if not result.get('success'):
            raise HTTPException(status_code=404, detail=result.get('error', 'Document not found'))
        
        return DocumentDeleteResponse(
            success=True,
            message=result.get('message', f"Document '{document_id}' deleted successfully")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

# Batch upload endpoint
@router.post("/batch-upload")
async def batch_upload_documents(
    files: List[UploadFile] = File(...),
    process_immediately: bool = Form(True)
):
    """
    Upload multiple documents at once.
    
    Args:
        files: List of document files to upload
        process_immediately: Whether to start processing immediately
        
    Returns:
        List of upload results
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 files allowed per batch")
    
    results = []
    
    for file in files:
        try:
            file_data = await file.read()
            
            if process_immediately:
                result = await document_service.process_document_sync(
                    file_data=file_data,
                    filename=file.filename or "unknown",
                    content_type=file.content_type
                )
            else:
                result = await document_service.upload_document(
                    file_data=file_data,
                    filename=file.filename or "unknown",
                    content_type=file.content_type,
                    process_immediately=False
                )
            
            results.append(result)
            
        except Exception as e:
            results.append({
                "success": False,
                "filename": file.filename or "unknown",
                "error": str(e)
            })
    
    return {
        "batch_results": results,
        "total_files": len(files),
        "successful_uploads": len([r for r in results if r.get('success')]),
        "failed_uploads": len([r for r in results if not r.get('success')]),
        "timestamp": datetime.utcnow().isoformat()
    }

# Document statistics endpoint
@router.get("/stats/overview")
async def get_document_statistics():
    """
    Get comprehensive document processing statistics.
    
    Returns:
        Statistics about document processing
    """
    if not document_service:
        raise HTTPException(status_code=503, detail="Document processing service not available")
    
    try:
        # Get all documents for statistics
        all_docs = await document_service.list_documents(limit=1000)
        documents = all_docs.get('documents', [])
        
        # Calculate statistics
        total_documents = len(documents)
        completed_docs = len([d for d in documents if d['status'] == 'completed'])
        processing_docs = len([d for d in documents if d['status'] == 'processing'])
        error_docs = len([d for d in documents if d['status'] == 'error'])
        
        total_size = sum(d.get('file_size', 0) for d in documents)
        total_pages = sum(d.get('page_count', 0) for d in documents if d.get('page_count'))
        total_words = sum(d.get('word_count', 0) for d in documents if d.get('word_count'))
        
        # Format breakdown
        format_stats = {}
        for doc in documents:
            content_type = doc.get('content_type', 'unknown')
            format_stats[content_type] = format_stats.get(content_type, 0) + 1
        
        return {
            "total_documents": total_documents,
            "status_breakdown": {
                "completed": completed_docs,
                "processing": processing_docs,
                "error": error_docs
            },
            "success_rate": (completed_docs / total_documents * 100) if total_documents > 0 else 0,
            "total_size_bytes": total_size,
            "total_size_formatted": _format_file_size(total_size),
            "total_pages": total_pages,
            "total_words": total_words,
            "format_breakdown": format_stats,
            "service_status": await document_service.get_service_status() if hasattr(document_service, 'get_service_status') else {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# Helper function
def _format_file_size(bytes_size: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    if bytes_size == 0:
        return "0 B"
    
    size_units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(bytes_size)
    
    while size >= 1024 and unit_index < len(size_units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {size_units[unit_index]}"

# Fallback endpoints for when document service is not available
if not DOCUMENT_SERVICE_AVAILABLE:
    @router.post("/upload")
    async def upload_document_fallback():
        raise HTTPException(
            status_code=503, 
            detail="Document processing service not available. Please ensure Docling and document processors are installed."
        )
    
    @router.get("/")
    async def list_documents_fallback():
        return {
            "documents": [],
            "total": 0,
            "message": "Document processing service not available"
        }
