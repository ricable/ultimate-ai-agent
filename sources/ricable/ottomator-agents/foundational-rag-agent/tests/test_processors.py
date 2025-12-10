"""
Unit tests for the document processors module.
"""
import os
import sys
import pytest
from pathlib import Path
import tempfile

# Add parent directory to path to allow relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_processing.processors import DocumentProcessor, TxtProcessor, PdfProcessor, get_document_processor


class TestDocumentProcessor:
    """
    Test cases for the DocumentProcessor base class.
    """
    
    def test_extract_text_not_implemented(self):
        """
        Test that the base DocumentProcessor raises NotImplementedError for extract_text.
        """
        processor = DocumentProcessor()
        with pytest.raises(NotImplementedError):
            processor.extract_text("dummy_path")
            
    def test_get_metadata_basic(self):
        """
        Test that the base DocumentProcessor provides basic metadata.
        """
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"Test content")
            temp_file_path = temp_file.name
            
        try:
            processor = DocumentProcessor()
            metadata = processor.get_metadata(temp_file_path)
            
            # Check basic metadata fields
            assert "filename" in metadata
            assert "file_extension" in metadata
            assert "file_size_bytes" in metadata
            assert metadata["file_extension"] == ".txt"
        finally:
            # Clean up
            os.unlink(temp_file_path)


class TestTxtProcessor:
    """
    Test cases for the TxtProcessor class.
    """
    
    def test_extract_text_nonexistent_file(self):
        """
        Test that TxtProcessor raises FileNotFoundError for nonexistent files.
        """
        processor = TxtProcessor()
        with pytest.raises(FileNotFoundError):
            processor.extract_text("nonexistent_file.txt")
    
    def test_extract_text_valid_txt_file(self):
        """
        Test that TxtProcessor correctly extracts text from a valid TXT file.
        """
        # Create a temporary TXT file
        content = "This is a test document.\nIt has multiple lines.\nAnd some content to process."
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(content.encode("utf-8"))
            temp_file_path = temp_file.name
        
        try:
            processor = TxtProcessor()
            extracted_text = processor.extract_text(temp_file_path)
            
            # Check the extracted text
            assert extracted_text == content
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
    
    def test_get_metadata_txt_file(self):
        """
        Test that TxtProcessor correctly extracts metadata from a TXT file.
        """
        # Create a temporary TXT file
        content = "This is a test document.\nIt has multiple lines.\nAnd some content to process."
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(content.encode("utf-8"))
            temp_file_path = temp_file.name
        
        try:
            processor = TxtProcessor()
            metadata = processor.get_metadata(temp_file_path)
            
            # Check metadata fields
            assert "filename" in metadata
            assert "file_extension" in metadata
            assert "file_size_bytes" in metadata
            assert "content_type" in metadata
            assert metadata["file_extension"] == ".txt"
            assert metadata["content_type"] == "text/plain"
            assert "processor" in metadata
            assert metadata["processor"] == "TxtProcessor"
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)


class TestPdfProcessor:
    """
    Test cases for the PdfProcessor class.
    """
    
    def test_extract_text_nonexistent_file(self):
        """
        Test that PdfProcessor raises FileNotFoundError for nonexistent files.
        """
        processor = PdfProcessor()
        with pytest.raises(FileNotFoundError):
            processor.extract_text("nonexistent_file.pdf")
    
    def test_get_metadata_pdf_file(self):
        """
        Test that PdfProcessor correctly extracts metadata.
        Note: This test doesn't use a real PDF file to avoid dependencies,
        but just tests the error handling.
        """
        processor = PdfProcessor()
        
        # Since we can't easily create a valid PDF file in a test,
        # we'll just test that the method handles errors gracefully
        with pytest.raises(FileNotFoundError):
            processor.get_metadata("nonexistent_file.pdf")


class TestGetDocumentProcessor:
    """
    Test cases for the get_document_processor function.
    """
    
    def test_get_document_processor_txt(self):
        """
        Test that get_document_processor returns a TxtProcessor for .txt files.
        """
        processor = get_document_processor("test.txt")
        assert isinstance(processor, TxtProcessor)
    
    def test_get_document_processor_pdf(self):
        """
        Test that get_document_processor returns a PdfProcessor for .pdf files.
        """
        processor = get_document_processor("test.pdf")
        assert isinstance(processor, PdfProcessor)
    
    def test_get_document_processor_unsupported(self):
        """
        Test that get_document_processor returns None for unsupported file types.
        """
        assert get_document_processor("test.docx") is None
        assert get_document_processor("test.csv") is None
