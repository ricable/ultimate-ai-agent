# File: backend/processors/document_storage.py
"""
Document Storage Manager
Handles file uploads, temporary storage, and document lifecycle management
for the UAP document processing pipeline.
"""

import os
import uuid
import aiofiles
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, BinaryIO
from datetime import datetime, timedelta
import asyncio
import logging
from dataclasses import dataclass
import mimetypes

@dataclass
class UploadResult:
    """Result of a file upload operation"""
    success: bool
    file_id: str
    file_path: Optional[str] = None
    original_filename: str = ""
    file_size: int = 0
    content_type: str = ""
    error_message: Optional[str] = None
    checksum: Optional[str] = None

class DocumentStorageManager:
    """
    Manages file uploads, temporary storage, and document lifecycle.
    Provides secure file handling with validation and cleanup capabilities.
    """
    
    def __init__(self, 
                 upload_path: str = "uploads",
                 processed_path: str = "documents", 
                 max_file_size: int = 50 * 1024 * 1024,
                 cleanup_interval_hours: int = 24):
        """
        Initialize the document storage manager.
        
        Args:
            upload_path: Path for temporary file uploads
            processed_path: Path for processed documents
            max_file_size: Maximum file size in bytes
            cleanup_interval_hours: Hours after which to cleanup temporary files
        """
        self.upload_path = Path(upload_path)
        self.processed_path = Path(processed_path)
        self.max_file_size = max_file_size
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        # Allowed MIME types
        self.allowed_mime_types = {
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/msword',
            'text/plain',
            'text/markdown',
            'text/html',
            'application/rtf',
            'application/vnd.oasis.opendocument.text'
        }
        
        # File extension to MIME type mapping
        self.extension_mime_map = {
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
    
    async def save_uploaded_file(self, 
                                file_data: bytes, 
                                filename: str, 
                                content_type: Optional[str] = None) -> UploadResult:
        """
        Save an uploaded file to temporary storage.
        
        Args:
            file_data: Raw file data
            filename: Original filename
            content_type: MIME type of the file
            
        Returns:
            UploadResult with operation details
        """
        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())
            file_extension = Path(filename).suffix.lower()
            
            # Validate file
            validation_result = self._validate_file(file_data, filename, content_type)
            if not validation_result["valid"]:
                return UploadResult(
                    success=False,
                    file_id=file_id,
                    original_filename=filename,
                    error_message=validation_result["error"]
                )
            
            # Determine content type
            final_content_type = content_type or self.extension_mime_map.get(file_extension, 'application/octet-stream')
            
            # Create file path
            safe_filename = f"{file_id}_{self._sanitize_filename(filename)}"
            file_path = self.upload_path / safe_filename
            
            # Calculate checksum
            checksum = hashlib.sha256(file_data).hexdigest()
            
            # Save file
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_data)
            
            self.logger.info(f"File uploaded successfully: {filename} -> {safe_filename}")
            
            return UploadResult(
                success=True,
                file_id=file_id,
                file_path=str(file_path),
                original_filename=filename,
                file_size=len(file_data),
                content_type=final_content_type,
                checksum=checksum
            )
            
        except Exception as e:
            self.logger.error(f"Error saving uploaded file {filename}: {str(e)}")
            return UploadResult(
                success=False,
                file_id=file_id if 'file_id' in locals() else "",
                original_filename=filename,
                error_message=f"Failed to save file: {str(e)}"
            )
    
    def _validate_file(self, file_data: bytes, filename: str, content_type: Optional[str]) -> Dict[str, Any]:
        """
        Validate uploaded file data.
        
        Args:
            file_data: Raw file data
            filename: Original filename
            content_type: MIME type of the file
            
        Returns:
            Dict with validation results
        """
        # Check file size
        if len(file_data) > self.max_file_size:
            return {
                "valid": False,
                "error": f"File size ({len(file_data)} bytes) exceeds maximum allowed size ({self.max_file_size} bytes)"
            }
        
        # Check if file is empty
        if len(file_data) == 0:
            return {
                "valid": False,
                "error": "File is empty"
            }
        
        # Check file extension
        file_extension = Path(filename).suffix.lower()
        if file_extension not in self.extension_mime_map:
            return {
                "valid": False,
                "error": f"Unsupported file extension: {file_extension}",
                "supported_extensions": list(self.extension_mime_map.keys())
            }
        
        # Validate MIME type if provided
        if content_type and content_type not in self.allowed_mime_types:
            return {
                "valid": False,
                "error": f"Unsupported MIME type: {content_type}",
                "supported_types": list(self.allowed_mime_types)
            }
        
        # Basic file content validation
        validation_error = self._validate_file_content(file_data, file_extension)
        if validation_error:
            return {
                "valid": False,
                "error": validation_error
            }
        
        return {"valid": True}
    
    def _validate_file_content(self, file_data: bytes, file_extension: str) -> Optional[str]:
        """
        Perform basic content validation based on file extension.
        
        Args:
            file_data: Raw file data
            file_extension: File extension
            
        Returns:
            Error message if validation fails, None otherwise
        """
        try:
            # PDF validation
            if file_extension == '.pdf':
                if not file_data.startswith(b'%PDF-'):
                    return "Invalid PDF file format"
            
            # ZIP-based formats (DOCX, ODT)
            elif file_extension in ['.docx', '.odt']:
                if not file_data.startswith(b'PK'):
                    return f"Invalid {file_extension.upper()} file format"
            
            # Text-based formats
            elif file_extension in ['.txt', '.md', '.html', '.htm']:
                try:
                    file_data.decode('utf-8')
                except UnicodeDecodeError:
                    return f"Invalid {file_extension.upper()} file: not valid UTF-8 text"
            
            # RTF validation
            elif file_extension == '.rtf':
                if not file_data.startswith(b'{\\rtf'):
                    return "Invalid RTF file format"
            
            return None
            
        except Exception as e:
            return f"File content validation error: {str(e)}"
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename for safe storage.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove or replace dangerous characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_"
        sanitized = ''.join(c if c in safe_chars else '_' for c in filename)
        
        # Limit length
        if len(sanitized) > 100:
            name_part = Path(sanitized).stem[:80]
            ext_part = Path(sanitized).suffix
            sanitized = f"{name_part}{ext_part}"
        
        return sanitized
    
    async def get_file_path(self, file_id: str) -> Optional[str]:
        """
        Get the file path for a given file ID.
        
        Args:
            file_id: File ID
            
        Returns:
            File path or None if not found
        """
        try:
            # Search in upload directory
            for file_path in self.upload_path.glob(f"{file_id}_*"):
                if file_path.is_file():
                    return str(file_path)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting file path for {file_id}: {str(e)}")
            return None
    
    async def delete_file(self, file_id: str) -> bool:
        """
        Delete a file by ID.
        
        Args:
            file_id: File ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Search and delete from upload directory
            deleted = False
            for file_path in self.upload_path.glob(f"{file_id}_*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted = True
                    self.logger.info(f"Deleted file: {file_path}")
            
            return deleted
            
        except Exception as e:
            self.logger.error(f"Error deleting file {file_id}: {str(e)}")
            return False
    
    async def cleanup_old_files(self) -> int:
        """
        Clean up old temporary files.
        
        Returns:
            Number of files cleaned up
        """
        try:
            cleaned_count = 0
            cutoff_time = datetime.now() - self.cleanup_interval
            
            for file_path in self.upload_path.iterdir():
                if file_path.is_file():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        self.logger.info(f"Cleaned up old file: {file_path}")
            
            if cleaned_count > 0:
                self.logger.info(f"Cleaned up {cleaned_count} old files")
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
            return 0
    
    async def get_file_info(self, file_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a file.
        
        Args:
            file_id: File ID
            
        Returns:
            File information dict or None if not found
        """
        try:
            file_path = await self.get_file_path(file_id)
            if not file_path:
                return None
            
            path_obj = Path(file_path)
            if not path_obj.exists():
                return None
            
            stat = path_obj.stat()
            
            return {
                "file_id": file_id,
                "file_path": str(path_obj),
                "filename": path_obj.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "extension": path_obj.suffix.lower(),
                "mime_type": self.extension_mime_map.get(path_obj.suffix.lower(), 'application/octet-stream')
            }
            
        except Exception as e:
            self.logger.error(f"Error getting file info for {file_id}: {str(e)}")
            return None
    
    async def start_cleanup_scheduler(self):
        """Start the automatic cleanup scheduler."""
        async def cleanup_task():
            while True:
                try:
                    await asyncio.sleep(3600)  # Run every hour
                    await self.cleanup_old_files()
                except Exception as e:
                    self.logger.error(f"Error in cleanup scheduler: {str(e)}")
        
        asyncio.create_task(cleanup_task())
        self.logger.info("Cleanup scheduler started")