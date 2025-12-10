"""Advanced content serialization framework for Flow4's chunker.py.

This module implements custom serialization providers and strategies based on Docling
examples, providing flexible content handling for tables, pictures, and multimodal content.
"""

import re
import json
from typing import Any, Dict, List, Optional, Union, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..utils.logging import get_logger

logger = get_logger(__name__)

# Try to import docling serialization components
try:
    from docling_core.transforms.chunker.hierarchical_chunker import (
        ChunkingDocSerializer,
        ChunkingSerializerProvider,
    )
    from docling_core.transforms.serializer.markdown import (
        MarkdownTableSerializer as DoclingMarkdownTableSerializer,
        MarkdownPictureSerializer,
    )
    from docling_core.transforms.serializer.base import (
        BaseDocSerializer,
        SerializationResult,
    )
    from docling_core.types.doc.document import (
        DoclingDocument,
        PictureClassificationData,
        PictureDescriptionData,
        PictureItem,
        TableItem,
    )
    HAS_DOCLING_SERIALIZATION = True
    logger.info("Docling serialization components available")
except ImportError:
    logger.warning("Docling serialization components not available. Using fallback implementations.")
    HAS_DOCLING_SERIALIZATION = False
    # Define fallback types
    class ChunkingSerializerProvider:
        pass
    class ChunkingDocSerializer:
        pass
    class SerializationResult:
        def __init__(self, text: str):
            self.text = text
    class BaseDocSerializer:
        pass


@dataclass
class SerializationConfig:
    """Configuration for serialization providers."""
    
    # Table serialization options
    table_format: str = "markdown"  # markdown, csv, json, html
    include_table_captions: bool = True
    table_placeholder_format: str = "[TABLE_{index}: {caption}]"
    preserve_table_styling: bool = False
    
    # Picture serialization options
    picture_format: str = "annotation"  # annotation, placeholder, description
    include_picture_descriptions: bool = True
    picture_placeholder_format: str = "[FIGURE_{index}: {description}]"
    extract_picture_metadata: bool = True
    
    # General serialization options
    preserve_markup: bool = True
    include_source_metadata: bool = True
    custom_placeholder_patterns: Dict[str, str] = None
    
    def __post_init__(self):
        if self.custom_placeholder_patterns is None:
            self.custom_placeholder_patterns = {}


class SerializationStrategy(Protocol):
    """Protocol for serialization strategies."""
    
    def serialize(self, content: Any, metadata: Dict[str, Any] = None) -> str:
        """Serialize content to string representation."""
        ...


class BaseContentSerializer(ABC):
    """Base class for content serializers."""
    
    def __init__(self, config: SerializationConfig = None):
        self.config = config or SerializationConfig()
    
    @abstractmethod
    def serialize(self, content: Any, metadata: Dict[str, Any] = None) -> str:
        """Serialize content to string representation."""
        pass
    
    def validate_content(self, content: Any) -> bool:
        """Validate that content can be serialized."""
        return content is not None


class MarkdownTableSerializer(BaseContentSerializer):
    """Advanced Markdown table serializer with enhanced formatting."""
    
    def serialize(self, content: Any, metadata: Dict[str, Any] = None) -> str:
        """Serialize table content to Markdown format.
        
        Args:
            content: Table content (dict, list, or string)
            metadata: Additional metadata about the table
            
        Returns:
            Formatted Markdown table string
        """
        if not self.validate_content(content):
            return ""
        
        metadata = metadata or {}
        
        try:
            # Handle different content types
            if isinstance(content, str):
                # Already a markdown table or raw content
                return self._format_existing_table(content, metadata)
            elif isinstance(content, dict):
                return self._format_dict_table(content, metadata)
            elif isinstance(content, list):
                return self._format_list_table(content, metadata)
            else:
                # Try to convert to string representation
                return self._format_generic_table(content, metadata)
                
        except Exception as e:
            logger.warning(f"Error serializing table: {e}")
            return self._create_table_placeholder(metadata)
    
    def _format_existing_table(self, content: str, metadata: Dict[str, Any]) -> str:
        """Format an existing table string."""
        # Clean up malformed table syntax
        content = re.sub(r',\s*=\s*\.\s*,\s*LTE\s*=\s*\.', '', content)
        content = re.sub(r'Level,\s*=\s*\.\s*Level,\s*LTE\s*=\s*[A-Za-z]+\.', '', content)
        
        # Add caption if available
        if self.config.include_table_captions and metadata.get('caption'):
            caption = f"**Table: {metadata['caption']}**\n\n"
            return caption + content
        
        return content
    
    def _format_dict_table(self, content: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Format a dictionary as a table."""
        if not content:
            return self._create_table_placeholder(metadata)
        
        # Create table from dict
        headers = ["Property", "Value"]
        rows = []
        
        for key, value in content.items():
            # Handle nested structures
            if isinstance(value, (dict, list)):
                value = json.dumps(value, indent=2)
            rows.append([str(key), str(value)])
        
        return self._create_markdown_table(headers, rows, metadata)
    
    def _format_list_table(self, content: List[Any], metadata: Dict[str, Any]) -> str:
        """Format a list as a table."""
        if not content:
            return self._create_table_placeholder(metadata)
        
        # Handle list of dicts (most common case)
        if all(isinstance(item, dict) for item in content):
            # Get all unique keys for headers
            headers = list(set().union(*(item.keys() for item in content if isinstance(item, dict))))
            rows = []
            
            for item in content:
                if isinstance(item, dict):
                    row = [str(item.get(header, '')) for header in headers]
                    rows.append(row)
            
            return self._create_markdown_table(headers, rows, metadata)
        
        # Handle list of lists
        elif all(isinstance(item, list) for item in content):
            if content:
                headers = [f"Column {i+1}" for i in range(len(content[0]))]
                rows = [[str(cell) for cell in row] for row in content]
                return self._create_markdown_table(headers, rows, metadata)
        
        # Handle simple list
        else:
            headers = ["Index", "Value"]
            rows = [[str(i), str(item)] for i, item in enumerate(content)]
            return self._create_markdown_table(headers, rows, metadata)
    
    def _format_generic_table(self, content: Any, metadata: Dict[str, Any]) -> str:
        """Format generic content as a table."""
        # Try to extract table-like data from object
        if hasattr(content, '__dict__'):
            return self._format_dict_table(content.__dict__, metadata)
        else:
            return self._create_table_placeholder(metadata)
    
    def _create_markdown_table(self, headers: List[str], rows: List[List[str]], metadata: Dict[str, Any]) -> str:
        """Create a properly formatted Markdown table."""
        if not headers or not rows:
            return self._create_table_placeholder(metadata)
        
        # Create table header
        table_lines = []
        
        # Add caption if available
        if self.config.include_table_captions and metadata.get('caption'):
            table_lines.append(f"**Table: {metadata['caption']}**")
            table_lines.append("")
        
        # Header row
        header_row = "| " + " | ".join(headers) + " |"
        table_lines.append(header_row)
        
        # Separator row
        separator = "| " + " | ".join(["---"] * len(headers)) + " |"
        table_lines.append(separator)
        
        # Data rows
        for row in rows:
            # Ensure row has same length as headers
            padded_row = (row + [""] * len(headers))[:len(headers)]
            row_str = "| " + " | ".join(str(cell).replace('|', '\\|') for cell in padded_row) + " |"
            table_lines.append(row_str)
        
        return "\n".join(table_lines)
    
    def _create_table_placeholder(self, metadata: Dict[str, Any]) -> str:
        """Create a table placeholder."""
        caption = metadata.get('caption', 'Table')
        index = metadata.get('index', 0)
        return self.config.table_placeholder_format.format(index=index, caption=caption)


class AnnotationPictureSerializer(BaseContentSerializer):
    """Picture serializer with annotation and description support."""
    
    def serialize(self, content: Any, metadata: Dict[str, Any] = None) -> str:
        """Serialize picture content with annotations.
        
        Args:
            content: Picture content or path
            metadata: Picture metadata including annotations
            
        Returns:
            Formatted picture representation
        """
        if not self.validate_content(content):
            return ""
        
        metadata = metadata or {}
        
        try:
            if self.config.picture_format == "annotation":
                return self._create_annotation_representation(content, metadata)
            elif self.config.picture_format == "description":
                return self._create_description_representation(content, metadata)
            else:  # placeholder
                return self._create_picture_placeholder(metadata)
                
        except Exception as e:
            logger.warning(f"Error serializing picture: {e}")
            return self._create_picture_placeholder(metadata)
    
    def _create_annotation_representation(self, content: Any, metadata: Dict[str, Any]) -> str:
        """Create detailed annotation representation."""
        parts = []
        
        # Basic picture info
        index = metadata.get('index', 0)
        description = metadata.get('description', 'Image')
        
        parts.append(f"[FIGURE {index}: {description}]")
        
        # Add classification if available
        if 'classification' in metadata:
            classification = metadata['classification']
            if isinstance(classification, dict):
                predicted_class = classification.get('predicted_class')
                confidence = classification.get('confidence')
                if predicted_class:
                    conf_str = f" (confidence: {confidence:.2f})" if confidence else ""
                    parts.append(f"Picture type: {predicted_class}{conf_str}")
        
        # Add description if available
        if self.config.include_picture_descriptions and 'detailed_description' in metadata:
            parts.append(f"Description: {metadata['detailed_description']}")
        
        # Add extracted text if available
        if 'extracted_text' in metadata and metadata['extracted_text']:
            parts.append(f"Text content: {metadata['extracted_text']}")
        
        # Add technical metadata if requested
        if self.config.extract_picture_metadata:
            tech_metadata = []
            for key in ['format', 'dimensions', 'size', 'resolution']:
                if key in metadata:
                    tech_metadata.append(f"{key}: {metadata[key]}")
            if tech_metadata:
                parts.append(f"Technical details: {', '.join(tech_metadata)}")
        
        return "\n".join(parts)
    
    def _create_description_representation(self, content: Any, metadata: Dict[str, Any]) -> str:
        """Create description-focused representation."""
        description = metadata.get('description', 'Image')
        detailed_desc = metadata.get('detailed_description', '')
        
        if detailed_desc:
            return f"{description}: {detailed_desc}"
        else:
            return description
    
    def _create_picture_placeholder(self, metadata: Dict[str, Any]) -> str:
        """Create a picture placeholder."""
        description = metadata.get('description', 'Image')
        index = metadata.get('index', 0)
        return self.config.picture_placeholder_format.format(index=index, description=description)


class CustomListSerializer(BaseContentSerializer):
    """Serializer for list content with enhanced formatting."""
    
    def serialize(self, content: Any, metadata: Dict[str, Any] = None) -> str:
        """Serialize list content."""
        if not isinstance(content, list):
            return str(content)
        
        metadata = metadata or {}
        list_type = metadata.get('list_type', 'bullet')
        
        if list_type == 'numbered':
            return self._create_numbered_list(content)
        else:
            return self._create_bullet_list(content)
    
    def _create_numbered_list(self, items: List[Any]) -> str:
        """Create a numbered list."""
        lines = []
        for i, item in enumerate(items, 1):
            lines.append(f"{i}. {str(item)}")
        return "\n".join(lines)
    
    def _create_bullet_list(self, items: List[Any]) -> str:
        """Create a bullet list."""
        lines = []
        for item in items:
            lines.append(f"- {str(item)}")
        return "\n".join(lines)


class Flow4SerializerProvider:
    """Custom serializer provider for Flow4 chunking."""
    
    def __init__(self, config: SerializationConfig = None):
        self.config = config or SerializationConfig()
        self.table_serializer = MarkdownTableSerializer(self.config)
        self.picture_serializer = AnnotationPictureSerializer(self.config)
        self.list_serializer = CustomListSerializer(self.config)
    
    def get_serializer(self, doc: Any = None) -> 'Flow4DocSerializer':
        """Get a configured document serializer."""
        return Flow4DocSerializer(
            doc=doc,
            config=self.config,
            table_serializer=self.table_serializer,
            picture_serializer=self.picture_serializer,
            list_serializer=self.list_serializer
        )


class Flow4DocSerializer:
    """Custom document serializer for Flow4."""
    
    def __init__(
        self,
        doc: Any = None,
        config: SerializationConfig = None,
        table_serializer: MarkdownTableSerializer = None,
        picture_serializer: AnnotationPictureSerializer = None,
        list_serializer: CustomListSerializer = None
    ):
        self.doc = doc
        self.config = config or SerializationConfig()
        self.table_serializer = table_serializer or MarkdownTableSerializer(self.config)
        self.picture_serializer = picture_serializer or AnnotationPictureSerializer(self.config)
        self.list_serializer = list_serializer or CustomListSerializer(self.config)
    
    def serialize_content(self, content: Any, content_type: str, metadata: Dict[str, Any] = None) -> str:
        """Serialize content based on type."""
        metadata = metadata or {}
        
        try:
            if content_type == "table":
                return self.table_serializer.serialize(content, metadata)
            elif content_type == "picture" or content_type == "figure":
                return self.picture_serializer.serialize(content, metadata)
            elif content_type == "list":
                return self.list_serializer.serialize(content, metadata)
            else:
                # Default text serialization
                return self._serialize_text(content, metadata)
        except Exception as e:
            logger.warning(f"Error serializing {content_type}: {e}")
            return str(content) if content else ""
    
    def _serialize_text(self, content: Any, metadata: Dict[str, Any]) -> str:
        """Default text serialization."""
        if isinstance(content, str):
            return content
        else:
            return str(content)


# Docling integration classes (only available if Docling is installed)
if HAS_DOCLING_SERIALIZATION:
    
    class Flow4ChunkingSerializerProvider(ChunkingSerializerProvider):
        """Flow4-specific chunking serializer provider."""
        
        def __init__(self, config: SerializationConfig = None):
            self.config = config or SerializationConfig()
        
        def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
            """Get configured chunking serializer."""
            return ChunkingDocSerializer(
                doc=doc,
                table_serializer=Flow4MarkdownTableSerializer(self.config),
                picture_serializer=Flow4AnnotationPictureSerializer(self.config),
            )
    
    
    class Flow4MarkdownTableSerializer(DoclingMarkdownTableSerializer):
        """Enhanced Markdown table serializer for Flow4."""
        
        def __init__(self, config: SerializationConfig = None):
            super().__init__()
            self.config = config or SerializationConfig()
            self.fallback_serializer = MarkdownTableSerializer(self.config)
        
        def serialize(
            self, 
            *, 
            item: TableItem, 
            doc_serializer: BaseDocSerializer, 
            doc: DoclingDocument, 
            **kwargs: Any
        ) -> SerializationResult:
            """Serialize table with enhanced formatting."""
            try:
                # Try original Docling serialization first
                result = super().serialize(
                    item=item, 
                    doc_serializer=doc_serializer, 
                    doc=doc, 
                    **kwargs
                )
                
                # Enhance with Flow4 specific formatting
                enhanced_text = self._enhance_table_formatting(result.text, item)
                return SerializationResult(enhanced_text)
                
            except Exception as e:
                logger.warning(f"Docling table serialization failed, using fallback: {e}")
                
                # Use fallback serializer
                metadata = self._extract_table_metadata(item)
                fallback_text = self.fallback_serializer.serialize(item, metadata)
                return SerializationResult(fallback_text)
        
        def _enhance_table_formatting(self, original_text: str, item: TableItem) -> str:
            """Enhance table formatting with Flow4 specific improvements."""
            # Clean up malformed table syntax
            text = re.sub(r',\s*=\s*\.\s*,\s*LTE\s*=\s*\.', '', original_text)
            text = re.sub(r'Level,\s*=\s*\.\s*Level,\s*LTE\s*=\s*[A-Za-z]+\.', '', text)
            
            # Add table metadata if configured
            if self.config.include_table_captions and hasattr(item, 'caption') and item.caption:
                text = f"**Table: {item.caption}**\n\n{text}"
            
            return text
        
        def _extract_table_metadata(self, item: TableItem) -> Dict[str, Any]:
            """Extract metadata from table item."""
            metadata = {}
            
            if hasattr(item, 'caption') and item.caption:
                metadata['caption'] = item.caption
            
            if hasattr(item, 'id'):
                metadata['index'] = item.id
            
            return metadata
    
    
    class Flow4AnnotationPictureSerializer(MarkdownPictureSerializer):
        """Enhanced picture serializer with annotations for Flow4."""
        
        def __init__(self, config: SerializationConfig = None):
            super().__init__()
            self.config = config or SerializationConfig()
            self.fallback_serializer = AnnotationPictureSerializer(self.config)
        
        def serialize(
            self, 
            *, 
            item: PictureItem, 
            doc_serializer: BaseDocSerializer, 
            doc: DoclingDocument, 
            **kwargs: Any
        ) -> SerializationResult:
            """Serialize picture with enhanced annotations."""
            try:
                text_parts = []
                
                # Extract basic picture info
                picture_id = getattr(item, 'id', 0)
                
                # Process annotations
                if hasattr(item, 'annotations') and item.annotations:
                    for annotation in item.annotations:
                        if isinstance(annotation, PictureClassificationData):
                            predicted_class = (
                                annotation.predicted_classes[0].class_name 
                                if annotation.predicted_classes else None
                            )
                            confidence = (
                                annotation.predicted_classes[0].confidence 
                                if annotation.predicted_classes else None
                            )
                            
                            if predicted_class:
                                conf_str = f" (confidence: {confidence:.2f})" if confidence else ""
                                text_parts.append(f"Picture type: {predicted_class}{conf_str}")
                        
                        elif isinstance(annotation, PictureDescriptionData):
                            if self.config.include_picture_descriptions:
                                text_parts.append(f"Description: {annotation.description}")
                
                # Create picture placeholder
                description = text_parts[0] if text_parts else "Image"
                placeholder = self.config.picture_placeholder_format.format(
                    index=picture_id, 
                    description=description
                )
                
                # Combine placeholder with annotations
                if text_parts:
                    final_text = placeholder + "\n" + "\n".join(text_parts)
                else:
                    final_text = placeholder
                
                return SerializationResult(final_text)
                
            except Exception as e:
                logger.warning(f"Docling picture serialization failed, using fallback: {e}")
                
                # Use fallback serializer
                metadata = self._extract_picture_metadata(item)
                fallback_text = self.fallback_serializer.serialize(item, metadata)
                return SerializationResult(fallback_text)
        
        def _extract_picture_metadata(self, item: PictureItem) -> Dict[str, Any]:
            """Extract metadata from picture item."""
            metadata = {}
            
            if hasattr(item, 'id'):
                metadata['index'] = item.id
            
            if hasattr(item, 'annotations') and item.annotations:
                for annotation in item.annotations:
                    if isinstance(annotation, PictureClassificationData):
                        if annotation.predicted_classes:
                            metadata['classification'] = {
                                'predicted_class': annotation.predicted_classes[0].class_name,
                                'confidence': annotation.predicted_classes[0].confidence
                            }
                    elif isinstance(annotation, PictureDescriptionData):
                        metadata['detailed_description'] = annotation.description
            
            return metadata

else:
    # Fallback classes when Docling is not available
    class Flow4ChunkingSerializerProvider:
        """Fallback chunking serializer provider."""
        
        def __init__(self, config: SerializationConfig = None):
            self.config = config or SerializationConfig()
            self.provider = Flow4SerializerProvider(self.config)
        
        def get_serializer(self, doc: Any = None):
            """Get fallback serializer."""
            return self.provider.get_serializer(doc)


class SerializationFramework:
    """Main serialization framework for Flow4."""
    
    def __init__(self, config: SerializationConfig = None):
        self.config = config or SerializationConfig()
        self.provider = Flow4ChunkingSerializerProvider(self.config)
        logger.info(f"Initialized serialization framework with Docling support: {HAS_DOCLING_SERIALIZATION}")
    
    def create_serializer_provider(self) -> Flow4ChunkingSerializerProvider:
        """Create a serializer provider."""
        return self.provider
    
    def serialize_multimodal_content(
        self, 
        content_items: List[Dict[str, Any]], 
        context: str = ""
    ) -> str:
        """Serialize multimodal content with context.
        
        Args:
            content_items: List of content items with type and data
            context: Additional context for serialization
            
        Returns:
            Serialized content string
        """
        serializer = self.provider.get_serializer()
        parts = []
        
        if context:
            parts.append(f"Context: {context}\n")
        
        for item in content_items:
            content_type = item.get('type', 'text')
            content_data = item.get('data')
            metadata = item.get('metadata', {})
            
            serialized = serializer.serialize_content(content_data, content_type, metadata)
            if serialized:
                parts.append(serialized)
        
        return "\n\n".join(parts)
    
    def configure_for_chunking(self, preserve_structure: bool = True) -> 'SerializationFramework':
        """Configure serialization specifically for chunking operations.
        
        Args:
            preserve_structure: Whether to preserve document structure
            
        Returns:
            Configured framework instance
        """
        chunking_config = SerializationConfig(
            table_format="markdown",
            picture_format="annotation",
            preserve_markup=preserve_structure,
            include_source_metadata=True,
            include_table_captions=True,
            include_picture_descriptions=True
        )
        
        return SerializationFramework(chunking_config)
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported serialization formats."""
        return {
            "tables": ["markdown", "csv", "json", "html"],
            "pictures": ["annotation", "placeholder", "description"],
            "lists": ["bullet", "numbered"],
            "text": ["markdown", "plain"]
        }