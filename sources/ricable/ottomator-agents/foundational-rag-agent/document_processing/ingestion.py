"""
Document ingestion pipeline for processing documents and generating embeddings.
"""
import os
import uuid
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime

from document_processing.chunker import TextChunker
from document_processing.embeddings import EmbeddingGenerator
from document_processing.processors import get_document_processor
from database.setup import SupabaseClient

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    """
    Simplified document ingestion pipeline with robust error handling.
    """
    
    def __init__(self, supabase_client: Optional[SupabaseClient] = None):
        """
        Initialize the document ingestion pipeline with default components.
        
        Args:
            supabase_client: Optional SupabaseClient for database operations
        """
        self.chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
        self.embedding_generator = EmbeddingGenerator()
        self.max_file_size_mb = 10  # Maximum file size in MB
        self.supabase_client = supabase_client or SupabaseClient()
        
        logger.info("Initialized DocumentIngestionPipeline with default components")
    
    def _check_file(self, file_path: str) -> bool:
        """
        Validate file exists and is within size limits.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            True if file is valid, False otherwise
        """
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
            
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            logger.error(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({self.max_file_size_mb} MB)")
            return False
            
        return True
    
    def process_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a document file, extract text, generate chunks and embeddings.
        
        Args:
            file_path: Path to the document file
            metadata: Optional metadata to associate with the document
            
        Returns:
            List of document chunks with embeddings
        """
        # Validate file
        if not self._check_file(file_path):
            return []
        
        # Get appropriate document processor
        try:
            processor = get_document_processor(file_path)
            if not processor:
                logger.error(f"Unsupported file type: {file_path}")
                return []
        except Exception as e:
            logger.error(f"Error getting document processor: {str(e)}")
            return []
        
        # Extract text from document
        try:
            text = processor.extract_text(file_path)
            logger.info(f"Extracted {len(text)} characters from {os.path.basename(file_path)}")
            
            if not text or not text.strip():
                logger.warning(f"No text content extracted from {os.path.basename(file_path)}")
                return []
                
        except Exception as e:
            logger.error(f"Failed to extract text from {os.path.basename(file_path)}: {str(e)}")
            return []
        
        # Generate chunks
        try:
            chunks = self.chunker.chunk_text(text)
            
            # Filter out empty chunks
            chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
            
            if not chunks:
                logger.warning("No valid chunks generated from document")
                return []
                
            logger.info(f"Generated {len(chunks)} valid chunks from document")
            
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}")
            return []
        
        # Generate embeddings for chunks
        try:
            embeddings = self.embedding_generator.embed_batch(chunks, batch_size=5)
            
            if len(embeddings) != len(chunks):
                logger.warning(f"Mismatch between chunks ({len(chunks)}) and embeddings ({len(embeddings)})")
                # Ensure we only process chunks that have embeddings
                chunks = chunks[:len(embeddings)]
                
            logger.info(f"Generated {len(embeddings)} embeddings")
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
        
        # Create document records
        try:
            # Generate a unique document ID
            document_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Add file info to metadata
            metadata.update({
                "filename": os.path.basename(file_path),
                "file_path": file_path,
                "file_size_bytes": os.path.getsize(file_path),
                "processed_at": timestamp,
                "chunk_count": len(chunks)
            })
            
            # Create records and store in database
            records = []
            stored_records = []
            
            # Create a URL/identifier for the document
            url = f"file://{os.path.basename(file_path)}"
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create record for return value
                record = {
                    "id": f"{document_id}_{i}",
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": metadata.copy()
                }
                records.append(record)
                
                # Store in Supabase
                try:
                    stored_record = self.supabase_client.store_document_chunk(
                        url=url,
                        chunk_number=i,
                        content=chunk,
                        embedding=embedding,
                        metadata=metadata.copy()
                    )
                    stored_records.append(stored_record)
                except Exception as e:
                    logger.error(f"Error storing chunk {i} in database: {str(e)}")
            
            logger.info(f"Created {len(records)} document records with ID {document_id}")
            logger.info(f"Stored {len(stored_records)} chunks in database")
            return records
            
        except Exception as e:
            logger.error(f"Error creating document records: {str(e)}")
            return []
    
    def process_text(self, text: str, source_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Process a text string through the ingestion pipeline.
        
        Args:
            text: Text content to process
            source_id: Identifier for the source of the text
            metadata: Optional metadata about the text
            
        Returns:
            List of dictionaries containing the processed chunks with their IDs
        """
        if not text or not text.strip():
            logger.warning("Empty text provided to process_text")
            return []
            
        if metadata is None:
            metadata = {}
        
        # Add source information to metadata
        metadata.update({
            "source_type": "text",
            "source_id": source_id,
            "processed_at": datetime.now().isoformat()
        })
        
        try:
            # Generate chunks
            chunks = self.chunker.chunk_text(text)
            chunks = [chunk for chunk in chunks if chunk and chunk.strip()]
            
            if not chunks:
                logger.warning("No valid chunks generated from text")
                return []
                
            logger.info(f"Generated {len(chunks)} chunks from text")
            
            # Generate embeddings
            embeddings = self.embedding_generator.embed_batch(chunks)
            
            # Create document records
            document_id = str(uuid.uuid4())
            
            # Create records and store in database
            records = []
            stored_records = []
            
            # Create a URL/identifier for the text
            url = f"text://{source_id}"
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Create record for return value
                record = {
                    "id": f"{document_id}_{i}",
                    "document_id": document_id,
                    "chunk_index": i,
                    "text": chunk,
                    "embedding": embedding,
                    "metadata": metadata.copy()
                }
                records.append(record)
                
                # Store in Supabase
                try:
                    stored_record = self.supabase_client.store_document_chunk(
                        url=url,
                        chunk_number=i,
                        content=chunk,
                        embedding=embedding,
                        metadata=metadata.copy()
                    )
                    stored_records.append(stored_record)
                except Exception as e:
                    logger.error(f"Error storing text chunk {i} in database: {str(e)}")
            
            logger.info(f"Created {len(records)} records from text input")
            logger.info(f"Stored {len(stored_records)} text chunks in database")
            return records
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            return []
    
    def process_batch(self, file_paths: List[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process a batch of files through the ingestion pipeline.
        
        Args:
            file_paths: List of paths to document files
            metadata: Optional shared metadata for all files
            
        Returns:
            Dictionary mapping file paths to their processed chunks
        """
        results = {}
        
        for file_path in file_paths:
            try:
                # Create file-specific metadata
                file_metadata = metadata.copy() if metadata else {}
                file_metadata["batch_processed"] = True
                
                # Process the file
                file_results = self.process_file(file_path, file_metadata)
                results[file_path] = file_results
                
                logger.info(f"Processed {file_path} with {len(file_results)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                results[file_path] = []
        
        return results
