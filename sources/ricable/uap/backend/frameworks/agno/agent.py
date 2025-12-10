# File: backend/frameworks/agno/agent.py
# Agno Agent Manager - Real Implementation
# Provides document processing and analysis capabilities using the Agno framework

import asyncio
import logging
from typing import Dict, Any, Optional, List
import os

# Conditional import for Agno framework
try:
    from agno.agent import Agent
    from agno.models.anthropic import Claude
    from agno.models.openai import OpenAIChat
    from agno.tools.reasoning import ReasoningTools
    AGNO_AVAILABLE = True
except ImportError:
    AGNO_AVAILABLE = False
    logging.warning("Agno framework not installed. Using fallback implementation.")

class AgnoAgentManager:
    """Real implementation of Agno agent framework for document processing.
    
    This class provides document analysis, processing, and reasoning capabilities
    using the Agno framework. It specializes in:
    - Document analysis and understanding
    - Multi-modal content processing (text, images, PDFs)
    - Structured data extraction
    - Content summarization and insights
    """
    
    def __init__(self):
        self.framework_name = "Agno"
        self.agent: Optional[Agent] = None
        self.is_initialized = False
        self.fallback_mode = not AGNO_AVAILABLE
        
        logging.info(f"{self.framework_name} manager initialized.")
    
    async def initialize(self) -> None:
        """Initialize the Agno framework resources.
        
        Sets up the Agno agent with document processing capabilities.
        """
        if self.fallback_mode:
            logging.warning(f"{self.framework_name} running in fallback mode - install 'agno' package for full functionality.")
            self.is_initialized = True
            return
        
        try:
            # Initialize Agno agent with document processing focus
            model = self._get_configured_model()
            
            self.agent = Agent(
                model=model,
                tools=[ReasoningTools(add_instructions=True)],
                instructions="""You are a specialized document processing and analysis agent. Your capabilities include:
                
                1. Document Analysis: Analyze and extract insights from various document types
                2. Content Summarization: Provide concise summaries of lengthy documents
                3. Data Extraction: Extract structured information from unstructured text
                4. Multi-modal Processing: Handle text, images, and mixed content
                5. Reasoning: Apply logical reasoning to document content
                
                Always provide clear, structured responses with relevant metadata.
                Use tables and formatted output when displaying extracted data.
                Be thorough in your analysis while maintaining clarity.""",
                markdown=True,
                show_tool_calls=False,
                debug_mode=False
            )
            
            self.is_initialized = True
            logging.info(f"{self.framework_name} agent initialized successfully.")
            
        except Exception as e:
            logging.error(f"Failed to initialize {self.framework_name} agent: {e}")
            self.fallback_mode = True
            self.is_initialized = True
    
    def _get_configured_model(self):
        """Get the configured language model for Agno agent."""
        # Try to use Claude first (Anthropic), then fallback to OpenAI
        claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if claude_api_key:
            return Claude(
                id="claude-3-5-sonnet-20241022",
                api_key=claude_api_key
            )
        elif openai_api_key:
            return OpenAIChat(
                id="gpt-4o",
                api_key=openai_api_key
            )
        else:
            # Fallback to default (will use environment variables)
            return Claude(id="claude-3-5-sonnet-20241022")
    
    async def process_message(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a user message and return a response.
        
        Args:
            message: The user's input message
            context: Additional context including file paths, document type, etc.
            
        Returns:
            Dict containing 'content' and 'metadata' keys
        """
        if not self.is_initialized:
            await self.initialize()
        
        # Check if this is a document-related query
        is_document_query = self._is_document_related(message, context)
        
        if self.fallback_mode:
            return await self._fallback_process_message(message, context, is_document_query)
        
        try:
            # Enhanced document processing with Agno
            enhanced_message = self._enhance_message_for_documents(message, context)
            
            # Use asyncio to run the potentially blocking agent.run() method
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.agent.run, enhanced_message
            )
            
            # Process and format the response
            processed_response = self._process_agno_response(response, context)
            
            return {
                "content": processed_response["content"],
                "metadata": {
                    "source": self.framework_name,
                    "context_received": bool(context),
                    "specialization": "document_processing",
                    "document_related": is_document_query,
                    "processing_mode": "agno_native",
                    **processed_response.get("metadata", {})
                }
            }
            
        except Exception as e:
            logging.error(f"Error processing message with {self.framework_name}: {e}")
            # Fallback to basic processing
            return await self._fallback_process_message(message, context, is_document_query)
    
    def _is_document_related(self, message: str, context: Dict[str, Any]) -> bool:
        """Determine if the message is document-related."""
        document_keywords = [
            'document', 'analyze', 'pdf', 'text', 'file', 'extract', 
            'summarize', 'parse', 'read', 'review', 'content', 'paper',
            'report', 'article', 'manuscript', 'data', 'table', 'chart'
        ]
        
        message_lower = message.lower()
        has_document_keywords = any(keyword in message_lower for keyword in document_keywords)
        has_file_context = context.get('file_path') or context.get('document_type')
        
        return has_document_keywords or has_file_context
    
    def _enhance_message_for_documents(self, message: str, context: Dict[str, Any]) -> str:
        """Enhance the message with document processing context."""
        enhanced_parts = [message]
        
        if context.get('file_path'):
            enhanced_parts.append(f"\nDocument file: {context['file_path']}")
        
        if context.get('document_type'):
            enhanced_parts.append(f"Document type: {context['document_type']}")
        
        if context.get('document_content'):
            enhanced_parts.append(f"\nDocument content:\n{context['document_content']}")
        
        return "\n".join(enhanced_parts)
    
    def _process_agno_response(self, response, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process the raw Agno response into structured format."""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
        
        # Extract metadata from response if available
        metadata = {}
        if hasattr(response, 'metadata'):
            metadata = response.metadata
        elif hasattr(response, 'tool_calls'):
            metadata['tool_calls_used'] = len(response.tool_calls) if response.tool_calls else 0
        
        return {
            "content": content,
            "metadata": metadata
        }
    
    async def _fallback_process_message(self, message: str, context: Dict[str, Any], is_document_query: bool) -> Dict[str, Any]:
        """Fallback processing when Agno is not available."""
        if is_document_query:
            if context.get('document_content'):
                response_content = f"""Document Analysis (Fallback Mode):

Message: {message}

Document Processing Status: Agno framework not fully available. For complete document processing capabilities, please install the 'agno' package.

Basic Analysis:
- Document content detected: {len(context['document_content'])} characters
- Request type: Document analysis
- Recommended action: Install Agno framework for advanced document processing

To enable full document processing:
```bash
pip install -U agno
```

Current limitations:
- No multimodal processing
- No structured data extraction
- No advanced reasoning capabilities"""
            else:
                response_content = f"""Document Processing Request (Fallback Mode):

Your request: "{message}"

This appears to be a document-related query. The Agno framework provides specialized document processing capabilities including:
- Multi-modal document analysis
- Structured data extraction
- Content summarization
- Advanced reasoning over documents

To enable these features, please install the Agno framework:
```bash
pip install -U agno
```

Current fallback response: I understand you want to work with documents, but I need the full Agno framework to provide comprehensive document processing capabilities."""
        else:
            response_content = f"Agno response (Fallback Mode): {message}\n\nFor enhanced AI capabilities, please install the Agno framework: pip install -U agno"
        
        return {
            "content": response_content,
            "metadata": {
                "source": self.framework_name,
                "context_received": bool(context),
                "specialization": "document_processing",
                "document_related": is_document_query,
                "processing_mode": "fallback",
                "agno_available": False,
                "recommendation": "Install agno package for full functionality"
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the Agno manager.
        
        Returns:
            Dict containing status information
        """
        return {
            "status": "active" if self.is_initialized else "initializing",
            "agents": 1 if self.agent else 0,
            "framework": self.framework_name,
            "specialization": "document_processing",
            "agno_available": AGNO_AVAILABLE and not self.fallback_mode,
            "fallback_mode": self.fallback_mode,
            "capabilities": [
                "document_analysis",
                "content_extraction", 
                "multi_modal_processing",
                "structured_output",
                "reasoning_tools"
            ] if not self.fallback_mode else ["basic_text_processing"],
            "initialized": self.is_initialized
        }
    
    async def process_document(self, document_content: str, document_type: str = "text", analysis_type: str = "general") -> Dict[str, Any]:
        """Specialized method for direct document processing.
        
        Args:
            document_content: The content of the document to process
            document_type: Type of document (text, pdf, image, etc.)
            analysis_type: Type of analysis to perform (summary, extraction, reasoning)
        
        Returns:
            Dict containing processed document information
        """
        context = {
            "document_content": document_content,
            "document_type": document_type,
            "analysis_type": analysis_type
        }
        
        analysis_prompts = {
            "summary": f"Please provide a comprehensive summary of this {document_type} document.",
            "extraction": f"Extract key information and structured data from this {document_type} document.",
            "reasoning": f"Analyze this {document_type} document and provide insights and reasoning about its content.",
            "general": f"Analyze this {document_type} document and provide relevant insights."
        }
        
        message = analysis_prompts.get(analysis_type, analysis_prompts["general"])
        return await self.process_message(message, context)

logging.info("Agno agent module loaded.")