"""
Multimodal Dataset Generator for Flow4

Generates high-quality multimodal fine-tuning datasets that combine text, tables, 
and images in formats suitable for vision-language model training.

Features:
- LLaVA-style conversation formats
- ShareGPT multimodal conversation structures
- Image-text interleaved formats
- Table-text integration
- Vision-language instruction tuning datasets
- Metadata preservation for multimodal relationships
"""

import json
import base64
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from urllib.parse import urlparse
import re

from ..utils.logging import get_logger
from ..utils.config import DoclingConfig

logger = get_logger(__name__)


@dataclass
class MultimodalDatasetConfig:
    """Configuration for multimodal dataset generation."""
    
    # Output formats
    include_llava_format: bool = True
    include_sharegpt_format: bool = True
    include_chatml_format: bool = True
    include_conversation_format: bool = True
    
    # Content integration strategies
    image_text_strategy: str = "interleaved"  # interleaved, separate, inline
    table_text_strategy: str = "inline"  # inline, separate, structured
    
    # Image handling
    image_base_path: str = "images"
    save_images_locally: bool = True
    generate_image_descriptions: bool = True
    image_description_length: str = "detailed"  # brief, detailed, comprehensive
    preserve_image_metadata: bool = True
    
    # Table handling
    table_format: str = "markdown"  # markdown, csv, json, html
    include_table_captions: bool = True
    preserve_table_structure: bool = True
    
    # Conversation generation
    conversations_per_chunk: int = 2
    turns_per_conversation: int = 3
    include_system_messages: bool = True
    
    # Quality control
    min_content_length: int = 50
    max_content_length: int = 4000
    require_multimodal_content: bool = False
    
    # Dataset context
    dataset_context: str = "technical documentation with tables and diagrams"
    domain_expertise: str = "telecommunications and 5G networks"


@dataclass
class MultimodalContent:
    """Represents multimodal content with text, images, and tables."""
    
    text: str
    images: List[Dict[str, Any]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_images(self) -> bool:
        return len(self.images) > 0
    
    def has_tables(self) -> bool:
        return len(self.tables) > 0
    
    def is_multimodal(self) -> bool:
        return self.has_images() or self.has_tables()


class MultimodalDatasetGenerator:
    """Generate multimodal fine-tuning datasets from Flow4 chunks."""
    
    def __init__(self, config: MultimodalDatasetConfig):
        """Initialize the multimodal dataset generator.
        
        Args:
            config: Configuration for multimodal dataset generation
        """
        self.config = config
        self.conversation_templates = self._load_conversation_templates()
        self.image_description_templates = self._load_image_description_templates()
        self.table_question_templates = self._load_table_question_templates()
    
    def _load_conversation_templates(self) -> Dict[str, List[str]]:
        """Load conversation templates for different content types."""
        return {
            "image_analysis": [
                "What can you see in this image?",
                "Describe the key components shown in this diagram.",
                "What technical information is presented in this figure?",
                "Explain the relationships shown in this visual representation.",
                "What are the main elements depicted in this image?"
            ],
            "table_analysis": [
                "What information is presented in this table?",
                "Summarize the key data from this table.",
                "What patterns or trends can you identify in this table?",
                "Explain the significance of the data in this table.",
                "What are the main categories or columns in this table?"
            ],
            "text_comprehension": [
                "What are the main points discussed in this section?",
                "Explain the key concepts presented in this text.",
                "Summarize the technical details provided.",
                "What procedures or processes are described here?",
                "What are the important specifications mentioned?"
            ],
            "multimodal_integration": [
                "How does the image relate to the text content?",
                "What additional information does the table provide to the text?",
                "Explain how the visual elements support the written content.",
                "What connections can you make between the text, images, and tables?",
                "How do all these elements work together to convey information?"
            ]
        }
    
    def _load_image_description_templates(self) -> Dict[str, str]:
        """Load templates for generating image descriptions."""
        return {
            "brief": "This image shows {main_content}.",
            "detailed": "This {image_type} displays {main_content}. Key elements include {key_elements}. {technical_context}",
            "comprehensive": "This {image_type} provides a {detailed_view} of {main_content}. The image contains {key_elements} and illustrates {technical_concepts}. {domain_context} {relationships}"
        }
    
    def _load_table_question_templates(self) -> List[str]:
        """Load question templates for tables."""
        return [
            "What are the key parameters shown in this table?",
            "How many {entity_type} are listed in this table?",
            "What is the range of values for {parameter_name}?",
            "Which {entity_type} has the highest {metric}?",
            "What configuration options are available according to this table?",
            "How are the {entities} categorized in this table?",
            "What relationships exist between the columns in this table?",
            "What technical specifications are defined in this table?"
        ]
    
    def extract_multimodal_content(self, chunk: Dict[str, Any]) -> MultimodalContent:
        """Extract multimodal content from a Flow4 chunk.
        
        Args:
            chunk: Flow4 document chunk with text and metadata
            
        Returns:
            MultimodalContent object with text, images, and tables
        """
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})
        
        # Extract images from metadata
        images = []
        if "images" in metadata:
            for img_data in metadata["images"]:
                images.append({
                    "type": "image",
                    "path": img_data.get("path", ""),
                    "caption": img_data.get("caption", ""),
                    "alt_text": img_data.get("alt_text", ""),
                    "description": img_data.get("description", ""),
                    "metadata": img_data.get("metadata", {})
                })
        
        # Extract tables from metadata or text
        tables = []
        if "tables" in metadata:
            for table_data in metadata["tables"]:
                tables.append({
                    "type": "table",
                    "content": table_data.get("content", ""),
                    "caption": table_data.get("caption", ""),
                    "headers": table_data.get("headers", []),
                    "rows": table_data.get("rows", []),
                    "format": table_data.get("format", "markdown"),
                    "metadata": table_data.get("metadata", {})
                })
        
        # Also check for Docling-specific metadata
        if "docling_figures" in metadata:
            for figure in metadata["docling_figures"]:
                images.append({
                    "type": "figure",
                    "path": figure.get("image_path", ""),
                    "caption": figure.get("caption", ""),
                    "description": self._generate_figure_description(figure),
                    "metadata": figure
                })
        
        if "docling_tables" in metadata:
            for table in metadata["docling_tables"]:
                tables.append({
                    "type": "docling_table",
                    "content": table.get("table_content", ""),
                    "caption": table.get("caption", ""),
                    "format": "markdown",
                    "metadata": table
                })
        
        return MultimodalContent(
            text=text,
            images=images,
            tables=tables,
            metadata=metadata
        )
    
    def _generate_figure_description(self, figure_data: Dict[str, Any]) -> str:
        """Generate a description for a figure based on its metadata.
        
        Args:
            figure_data: Figure metadata from Docling
            
        Returns:
            Generated description string
        """
        caption = figure_data.get("caption", "")
        if caption:
            return f"A technical diagram showing {caption.lower()}"
        
        # Fallback description based on context
        return f"A technical diagram from {self.config.domain_expertise} documentation"
    
    def generate_image_description(self, image_data: Dict[str, Any]) -> str:
        """Generate an AI-style description for an image.
        
        Args:
            image_data: Image metadata
            
        Returns:
            Generated description string
        """
        if not self.config.generate_image_descriptions:
            return image_data.get("caption", image_data.get("alt_text", ""))
        
        template_key = self.config.image_description_length
        template = self.image_description_templates.get(template_key, self.image_description_templates["detailed"])
        
        # Extract information for template
        main_content = image_data.get("caption", "technical components")
        image_type = "diagram" if "diagram" in str(image_data).lower() else "figure"
        key_elements = "technical specifications and system components"
        technical_context = f"This relates to {self.config.domain_expertise}."
        
        # Format the template
        description = template.format(
            main_content=main_content,
            image_type=image_type,
            key_elements=key_elements,
            technical_context=technical_context,
            detailed_view="comprehensive overview",
            technical_concepts="system architecture and configuration options",
            domain_context=f"In the context of {self.config.domain_expertise},",
            relationships="The visual elements demonstrate interconnections between system components."
        )
        
        return description
    
    def format_table_content(self, table_data: Dict[str, Any]) -> str:
        """Format table content according to configuration.
        
        Args:
            table_data: Table metadata and content
            
        Returns:
            Formatted table string
        """
        content = table_data.get("content", "")
        caption = table_data.get("caption", "")
        
        if self.config.table_format == "markdown":
            formatted = content
            if caption and self.config.include_table_captions:
                formatted = f"**{caption}**\n\n{content}"
        elif self.config.table_format == "csv":
            # Convert to CSV if needed
            formatted = content  # Assume already in correct format
        elif self.config.table_format == "json":
            # Convert to JSON structure
            headers = table_data.get("headers", [])
            rows = table_data.get("rows", [])
            if headers and rows:
                json_data = [dict(zip(headers, row)) for row in rows]
                formatted = json.dumps(json_data, indent=2)
            else:
                formatted = content
        else:
            formatted = content
        
        return formatted
    
    def create_llava_conversation(self, content: MultimodalContent) -> Dict[str, Any]:
        """Create a conversation in LLaVA format.
        
        Args:
            content: Multimodal content to convert
            
        Returns:
            LLaVA-formatted conversation
        """
        conversations = []
        
        # System message
        if self.config.include_system_messages:
            conversations.append({
                "from": "system",
                "value": f"You are a helpful AI assistant specializing in {self.config.dataset_context}. "
                        f"You can analyze text, images, and tables to provide comprehensive answers."
            })
        
        # Create multimodal conversation turns
        for turn in range(self.config.turns_per_conversation):
            if turn == 0:
                # First turn: Introduce the content
                user_message = self._create_multimodal_user_message(content)
                assistant_message = self._create_multimodal_assistant_response(content)
            else:
                # Follow-up turns: Ask specific questions
                user_message = self._create_followup_question(content, turn)
                assistant_message = self._create_followup_response(content, turn)
            
            conversations.extend([
                {"from": "human", "value": user_message},
                {"from": "gpt", "value": assistant_message}
            ])
        
        return {
            "id": self._generate_conversation_id(content),
            "conversations": conversations,
            "metadata": {
                "has_images": content.has_images(),
                "has_tables": content.has_tables(),
                "source": content.metadata.get("source", "unknown"),
                "format": "llava"
            }
        }
    
    def create_sharegpt_conversation(self, content: MultimodalContent) -> Dict[str, Any]:
        """Create a conversation in ShareGPT format.
        
        Args:
            content: Multimodal content to convert
            
        Returns:
            ShareGPT-formatted conversation
        """
        conversations = []
        
        # System message
        if self.config.include_system_messages:
            conversations.append({
                "from": "system",
                "value": f"You are an expert assistant in {self.config.domain_expertise}. "
                        f"Provide detailed analysis of technical documents, images, and data tables."
            })
        
        # Human message with multimodal content
        human_message = self._create_comprehensive_user_message(content)
        conversations.append({
            "from": "human", 
            "value": human_message
        })
        
        # Assistant response
        assistant_message = self._create_comprehensive_assistant_response(content)
        conversations.append({
            "from": "gpt",
            "value": assistant_message
        })
        
        return {
            "conversations": conversations,
            "metadata": {
                "multimodal": content.is_multimodal(),
                "content_types": self._get_content_types(content),
                "source": content.metadata.get("source", "unknown"),
                "format": "sharegpt"
            }
        }
    
    def create_chatml_conversation(self, content: MultimodalContent) -> Dict[str, Any]:
        """Create a conversation in ChatML format.
        
        Args:
            content: Multimodal content to convert
            
        Returns:
            ChatML-formatted conversation
        """
        messages = []
        
        # System message
        if self.config.include_system_messages:
            messages.append({
                "role": "system",
                "content": f"You are a technical expert in {self.config.domain_expertise}. "
                          f"Analyze and explain content from technical documentation including text, images, and tables."
            })
        
        # User message
        user_content = self._create_structured_user_content(content)
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Assistant message
        assistant_content = self._create_structured_assistant_content(content)
        messages.append({
            "role": "assistant", 
            "content": assistant_content
        })
        
        return {
            "messages": messages,
            "metadata": {
                "multimodal_elements": len(content.images) + len(content.tables),
                "source": content.metadata.get("source", "unknown"),
                "format": "chatml"
            }
        }
    
    def _create_multimodal_user_message(self, content: MultimodalContent) -> str:
        """Create a user message that references multimodal content."""
        message_parts = []
        
        # Add text reference
        if content.text:
            message_parts.append("Please analyze the following technical documentation:")
            message_parts.append(f"Text content: {content.text[:200]}...")
        
        # Add image references
        if content.has_images():
            for i, image in enumerate(content.images):
                if self.config.image_text_strategy == "interleaved":
                    description = self.generate_image_description(image)
                    message_parts.append(f"Image {i+1}: {description}")
                elif self.config.image_text_strategy == "inline":
                    message_parts.append(f"[IMAGE: {image.get('caption', 'Technical diagram')}]")
        
        # Add table references
        if content.has_tables():
            for i, table in enumerate(content.tables):
                if self.config.table_text_strategy == "inline":
                    formatted_table = self.format_table_content(table)
                    message_parts.append(f"Table {i+1}:\n{formatted_table}")
                elif self.config.table_text_strategy == "separate":
                    message_parts.append(f"[TABLE: {table.get('caption', 'Data table')}]")
        
        message_parts.append("What can you tell me about this content?")
        
        return "\n\n".join(message_parts)
    
    def _create_multimodal_assistant_response(self, content: MultimodalContent) -> str:
        """Create an assistant response that analyzes multimodal content."""
        response_parts = []
        
        # Analyze text content
        if content.text:
            response_parts.append("Based on the technical documentation provided:")
            response_parts.append(f"The text describes {self._extract_main_topic(content.text)}")
        
        # Analyze images
        if content.has_images():
            response_parts.append("Regarding the visual elements:")
            for i, image in enumerate(content.images):
                description = self.generate_image_description(image)
                response_parts.append(f"- Image {i+1}: {description}")
        
        # Analyze tables
        if content.has_tables():
            response_parts.append("The tabular data shows:")
            for i, table in enumerate(content.tables):
                analysis = self._analyze_table_content(table)
                response_parts.append(f"- Table {i+1}: {analysis}")
        
        # Provide synthesis
        if content.is_multimodal():
            response_parts.append(
                "These elements work together to provide a comprehensive view of "
                f"{self.config.domain_expertise} concepts and specifications."
            )
        
        return "\n\n".join(response_parts)
    
    def _extract_main_topic(self, text: str) -> str:
        """Extract the main topic from text content."""
        # Simple extraction - could be enhanced with NLP
        words = text.split()
        if len(words) > 10:
            return " ".join(words[:10]) + "..."
        return text
    
    def _analyze_table_content(self, table_data: Dict[str, Any]) -> str:
        """Analyze table content and provide summary."""
        caption = table_data.get("caption", "")
        if caption:
            return f"data related to {caption.lower()}"
        
        content = table_data.get("content", "")
        if "parameter" in content.lower():
            return "configuration parameters and their values"
        elif "feature" in content.lower():
            return "feature specifications and capabilities"
        else:
            return "structured technical information"
    
    def _create_comprehensive_user_message(self, content: MultimodalContent) -> str:
        """Create a comprehensive user message for ShareGPT format."""
        message = f"I have technical documentation from {self.config.domain_expertise} that includes:\n\n"
        
        if content.text:
            message += f"Text content discussing: {self._extract_main_topic(content.text)}\n\n"
        
        if content.has_images():
            message += f"Visual diagrams and figures ({len(content.images)} items)\n\n"
        
        if content.has_tables():
            message += f"Data tables with specifications ({len(content.tables)} items)\n\n"
        
        message += "Please provide a detailed analysis covering all aspects of this technical content."
        
        return message
    
    def _create_comprehensive_assistant_response(self, content: MultimodalContent) -> str:
        """Create a comprehensive assistant response."""
        response = f"This technical documentation provides valuable information about {self.config.domain_expertise}. Let me analyze each component:\n\n"
        
        # Analyze text
        if content.text:
            response += "**Text Analysis:**\n"
            response += f"The documentation covers {self._extract_main_topic(content.text)}\n\n"
        
        # Analyze images
        if content.has_images():
            response += "**Visual Analysis:**\n"
            for i, image in enumerate(content.images):
                description = self.generate_image_description(image)
                response += f"- Figure {i+1}: {description}\n"
            response += "\n"
        
        # Analyze tables
        if content.has_tables():
            response += "**Data Analysis:**\n"
            for i, table in enumerate(content.tables):
                analysis = self._analyze_table_content(table)
                response += f"- Table {i+1}: {analysis}\n"
            response += "\n"
        
        response += "**Summary:**\n"
        response += f"This comprehensive documentation demonstrates {self.config.domain_expertise} concepts through multiple modalities, providing both theoretical background and practical specifications."
        
        return response
    
    def _create_structured_user_content(self, content: MultimodalContent) -> str:
        """Create structured user content for ChatML format."""
        sections = []
        
        if content.text:
            sections.append(f"<text>\n{content.text[:500]}{'...' if len(content.text) > 500 else ''}\n</text>")
        
        if content.has_images():
            for i, image in enumerate(content.images):
                description = self.generate_image_description(image)
                sections.append(f"<image id='{i+1}'>\n{description}\n</image>")
        
        if content.has_tables():
            for i, table in enumerate(content.tables):
                formatted = self.format_table_content(table)
                sections.append(f"<table id='{i+1}'>\n{formatted}\n</table>")
        
        sections.append("<question>Explain the technical concepts and relationships shown in this multimodal content.</question>")
        
        return "\n\n".join(sections)
    
    def _create_structured_assistant_content(self, content: MultimodalContent) -> str:
        """Create structured assistant content for ChatML format."""
        response = "<analysis>\n"
        
        if content.text:
            response += f"The text content describes {self._extract_main_topic(content.text)}\n\n"
        
        if content.has_images():
            response += "Visual elements include:\n"
            for i, image in enumerate(content.images):
                description = self.generate_image_description(image)
                response += f"- Image {i+1}: {description}\n"
            response += "\n"
        
        if content.has_tables():
            response += "Tabular data provides:\n"
            for i, table in enumerate(content.tables):
                analysis = self._analyze_table_content(table)
                response += f"- Table {i+1}: {analysis}\n"
            response += "\n"
        
        response += "</analysis>\n\n"
        response += f"<synthesis>\nThese multimodal elements collectively illustrate {self.config.domain_expertise} principles and provide comprehensive technical specifications.\n</synthesis>"
        
        return response
    
    def _create_followup_question(self, content: MultimodalContent, turn: int) -> str:
        """Create a follow-up question for continued conversation."""
        if content.has_images() and turn == 1:
            return "Can you explain the technical diagrams in more detail?"
        elif content.has_tables() and turn == 2:
            return "What specific parameters or configurations are shown in the tables?"
        else:
            return "How do all these elements relate to practical implementation?"
    
    def _create_followup_response(self, content: MultimodalContent, turn: int) -> str:
        """Create a follow-up response for continued conversation."""
        if content.has_images() and turn == 1:
            response = "Looking at the technical diagrams in detail:\n\n"
            for i, image in enumerate(content.images):
                description = self.generate_image_description(image)
                response += f"Diagram {i+1} shows {description.lower()}\n"
            return response
        elif content.has_tables() and turn == 2:
            response = "The tables provide specific technical parameters:\n\n"
            for i, table in enumerate(content.tables):
                analysis = self._analyze_table_content(table)
                response += f"Table {i+1} contains {analysis}\n"
            return response
        else:
            return f"For practical implementation in {self.config.domain_expertise}, these specifications provide the necessary technical foundation for system configuration and optimization."
    
    def _get_content_types(self, content: MultimodalContent) -> List[str]:
        """Get list of content types present in the multimodal content."""
        types = ["text"] if content.text else []
        if content.has_images():
            types.append("images")
        if content.has_tables():
            types.append("tables")
        return types
    
    def _generate_conversation_id(self, content: MultimodalContent) -> str:
        """Generate a unique conversation ID."""
        source = content.metadata.get("source", "unknown")
        content_hash = hashlib.md5(content.text.encode()).hexdigest()[:8]
        return f"multimodal_{source}_{content_hash}"
    
    def generate_multimodal_dataset(
        self, 
        chunks: List[Dict[str, Any]], 
        output_dir: str,
        dataset_name: str = "multimodal_dataset"
    ) -> Dict[str, Any]:
        """Generate a complete multimodal dataset from chunks.
        
        Args:
            chunks: List of Flow4 document chunks
            output_dir: Directory to save the dataset
            dataset_name: Name for the dataset
            
        Returns:
            Generation results and statistics
        """
        logger.info(f"🎨 Generating multimodal dataset from {len(chunks)} chunks")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract multimodal content
        multimodal_contents = []
        for chunk in chunks:
            content = self.extract_multimodal_content(chunk)
            if len(content.text) >= self.config.min_content_length:
                if not self.config.require_multimodal_content or content.is_multimodal():
                    multimodal_contents.append(content)
        
        logger.info(f"📊 Found {len(multimodal_contents)} valid multimodal contents")
        
        # Generate different format datasets
        results = {
            "total_chunks": len(chunks),
            "valid_contents": len(multimodal_contents),
            "multimodal_contents": sum(1 for c in multimodal_contents if c.is_multimodal()),
            "datasets": {}
        }
        
        # Generate LLaVA format
        if self.config.include_llava_format:
            llava_data = []
            for content in multimodal_contents:
                conversation = self.create_llava_conversation(content)
                llava_data.append(conversation)
            
            llava_file = output_path / f"{dataset_name}_llava.json"
            with open(llava_file, 'w', encoding='utf-8') as f:
                json.dump(llava_data, f, indent=2, ensure_ascii=False)
            
            results["datasets"]["llava"] = {
                "file": str(llava_file),
                "conversations": len(llava_data)
            }
            logger.info(f"💾 Saved LLaVA format: {llava_file}")
        
        # Generate ShareGPT format
        if self.config.include_sharegpt_format:
            sharegpt_data = []
            for content in multimodal_contents:
                conversation = self.create_sharegpt_conversation(content)
                sharegpt_data.append(conversation)
            
            sharegpt_file = output_path / f"{dataset_name}_sharegpt.jsonl"
            with open(sharegpt_file, 'w', encoding='utf-8') as f:
                for conversation in sharegpt_data:
                    f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
            
            results["datasets"]["sharegpt"] = {
                "file": str(sharegpt_file),
                "conversations": len(sharegpt_data)
            }
            logger.info(f"💾 Saved ShareGPT format: {sharegpt_file}")
        
        # Generate ChatML format
        if self.config.include_chatml_format:
            chatml_data = []
            for content in multimodal_contents:
                conversation = self.create_chatml_conversation(content)
                chatml_data.append(conversation)
            
            chatml_file = output_path / f"{dataset_name}_chatml.jsonl"
            with open(chatml_file, 'w', encoding='utf-8') as f:
                for conversation in chatml_data:
                    f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
            
            results["datasets"]["chatml"] = {
                "file": str(chatml_file),
                "conversations": len(chatml_data)
            }
            logger.info(f"💾 Saved ChatML format: {chatml_file}")
        
        # Save generation summary
        summary_file = output_path / f"{dataset_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🎉 Multimodal dataset generation complete: {output_path}")
        return results


def create_multimodal_config(
    domain_expertise: str = "telecommunications and 5G networks",
    dataset_context: str = "technical documentation with tables and diagrams",
    **kwargs
) -> MultimodalDatasetConfig:
    """Create a multimodal dataset configuration with sensible defaults.
    
    Args:
        domain_expertise: Domain expertise for the dataset
        dataset_context: Context description for the dataset
        **kwargs: Additional configuration options
        
    Returns:
        Configured MultimodalDatasetConfig instance
    """
    config = MultimodalDatasetConfig(
        domain_expertise=domain_expertise,
        dataset_context=dataset_context
    )
    
    # Update with any additional kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


async def generate_multimodal_dataset(
    chunks_path: str,
    output_path: str,
    dataset_name: str = "multimodal_dataset",
    domain_expertise: str = "telecommunications and 5G networks",
    **config_kwargs
) -> Dict[str, Any]:
    """High-level function to generate multimodal datasets.
    
    Args:
        chunks_path: Path to directory containing chunks
        output_path: Path to save the generated dataset
        dataset_name: Name for the dataset
        domain_expertise: Domain expertise context
        **config_kwargs: Additional configuration options
        
    Returns:
        Generation results dictionary
    """
    # Load chunks
    chunks_dir = Path(chunks_path)
    chunk_files = sorted(chunks_dir.glob("chunk_*.json"))
    
    chunks = []
    for chunk_file in chunk_files:
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                chunks.append(chunk_data)
        except Exception as e:
            logger.warning(f"Failed to load chunk {chunk_file}: {e}")
    
    if not chunks:
        raise ValueError(f"No valid chunks found in {chunks_path}")
    
    # Create configuration
    config = create_multimodal_config(
        domain_expertise=domain_expertise,
        **config_kwargs
    )
    
    # Generate dataset
    generator = MultimodalDatasetGenerator(config)
    results = generator.generate_multimodal_dataset(chunks, output_path, dataset_name)
    
    return results