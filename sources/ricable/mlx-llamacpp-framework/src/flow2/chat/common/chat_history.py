#!/usr/bin/env python3
"""
Chat History Management Utilities

This module provides utilities for managing chat history and context across 
different LLM frameworks. It handles storing, retrieving, and formatting 
conversation history for use with both llama.cpp and MLX frameworks.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict


@dataclass
class Message:
    """A single message in a chat history."""
    role: str  # 'system', 'user', or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


@dataclass
class ChatHistory:
    """Class to manage a complete chat history."""
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: str, content: str) -> None:
        """Add a new message to the history."""
        self.messages.append(Message(role=role, content=content))
    
    def add_system_message(self, content: str) -> None:
        """Add a system message to the history."""
        self.add_message("system", content)
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the history."""
        self.add_message("assistant", content)
    
    def get_last_message(self) -> Optional[Message]:
        """Get the last message in the history."""
        if not self.messages:
            return None
        return self.messages[-1]
    
    def clear(self) -> None:
        """Clear all messages from the history."""
        self.messages.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatHistory":
        """Create a ChatHistory from dictionary data."""
        history = cls(metadata=data.get("metadata", {}))
        for msg_data in data.get("messages", []):
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                timestamp=msg_data.get("timestamp", time.time())
            )
            history.messages.append(msg)
        return history
    
    def save(self, file_path: str) -> None:
        """Save the chat history to a file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, file_path: str) -> "ChatHistory":
        """Load a chat history from a file."""
        if not os.path.exists(file_path):
            return cls()
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def format_for_llama_cpp(self, template_type: str = "chatml") -> str:
        """
        Format chat history for llama.cpp models.
        
        Args:
            template_type: The type of template to use ('chatml', 'llama2', 'alpaca', etc.)
            
        Returns:
            Formatted chat history as a string
        """
        if template_type == "chatml":
            formatted = ""
            for msg in self.messages:
                if msg.role == "system":
                    formatted += f"<|im_start|>system\n{msg.content}<|im_end|>\n"
                elif msg.role == "user":
                    formatted += f"<|im_start|>user\n{msg.content}<|im_end|>\n"
                elif msg.role == "assistant":
                    formatted += f"<|im_start|>assistant\n{msg.content}<|im_end|>\n"
            # Add the final assistant prompt
            formatted += "<|im_start|>assistant\n"
            return formatted
            
        elif template_type == "llama2":
            formatted = ""
            # Add system message if it exists
            system_msg = next((msg for msg in self.messages if msg.role == "system"), None)
            if system_msg:
                formatted = f"<s>[INST] <<SYS>>\n{system_msg.content}\n<</SYS>>\n\n"
            else:
                formatted = "<s>[INST] "
            
            # Add conversation history
            user_assistant_pairs = []
            for i in range(len(self.messages)):
                if self.messages[i].role == "system":
                    continue
                if self.messages[i].role == "user" and i+1 < len(self.messages) and self.messages[i+1].role == "assistant":
                    user_assistant_pairs.append((self.messages[i].content, self.messages[i+1].content))
            
            # Format pairs
            for i, (user_msg, assistant_msg) in enumerate(user_assistant_pairs):
                if i == 0 and system_msg:
                    formatted += f"{user_msg} [/INST] {assistant_msg} </s><s>[INST] "
                else:
                    formatted += f"{user_msg} [/INST] {assistant_msg} </s><s>[INST] "
            
            # Add the final user message if it exists
            if self.messages and self.messages[-1].role == "user":
                formatted += f"{self.messages[-1].content} [/INST] "
                
            return formatted
        
        elif template_type == "alpaca":
            formatted = ""
            for msg in self.messages:
                if msg.role == "system":
                    formatted += f"### Instruction:\n{msg.content}\n\n"
                elif msg.role == "user":
                    formatted += f"### Input:\n{msg.content}\n\n"
                elif msg.role == "assistant":
                    formatted += f"### Response:\n{msg.content}\n\n"
            # Add the final response prompt
            formatted += "### Response:\n"
            return formatted
            
        else:
            # Simple default format
            formatted = ""
            for msg in self.messages:
                formatted += f"{msg.role.capitalize()}: {msg.content}\n\n"
            formatted += "Assistant: "
            return formatted
    
    def format_for_mlx(self) -> List[Dict[str, str]]:
        """
        Format chat history for MLX models.
        
        Returns:
            List of message dictionaries
        """
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]


class ChatContextManager:
    """Manager for chat context including history, model parameters, and session state."""
    
    def __init__(
        self, 
        system_message: str = "You are a helpful assistant.",
        max_context_length: int = 2048,
        history_file: Optional[str] = None
    ):
        self.max_context_length = max_context_length
        self.history_file = history_file
        
        # Load history or create new
        if history_file and os.path.exists(history_file):
            self.history = ChatHistory.load(history_file)
        else:
            self.history = ChatHistory()
            # Add system message if it doesn't exist
            if not any(msg.role == "system" for msg in self.history.messages):
                self.history.add_system_message(system_message)
    
    def add_user_message(self, content: str) -> None:
        """Add a user message to the context."""
        self.history.add_user_message(content)
        self._save_history()
    
    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the context."""
        self.history.add_assistant_message(content)
        self._save_history()
    
    def _save_history(self) -> None:
        """Save history if a file path is specified."""
        if self.history_file:
            self.history.save(self.history_file)
    
    def get_formatted_context(self, format_type: str = "chatml") -> Union[str, List[Dict[str, str]]]:
        """
        Get the formatted context for the specified model type.
        
        Args:
            format_type: The type of formatting to use ('chatml', 'llama2', 'alpaca', 'mlx')
            
        Returns:
            Formatted context as a string or list depending on the format type
        """
        if format_type == "mlx":
            return self.history.format_for_mlx()
        else:
            return self.history.format_for_llama_cpp(format_type)
    
    def truncate_context_if_needed(self) -> None:
        """
        Truncate the context if it exceeds the maximum allowed tokens.
        This is a simplified implementation - a more accurate one would count tokens.
        """
        # Simple approximation - each message takes roughly content_length/4 tokens
        while len(self.history.messages) > 2:  # Keep at least system + last message
            total_chars = sum(len(msg.content) for msg in self.history.messages)
            approximate_tokens = total_chars / 4
            
            if approximate_tokens <= self.max_context_length:
                break
                
            # Remove the oldest non-system message
            for i, msg in enumerate(self.history.messages):
                if msg.role != "system":
                    self.history.messages.pop(i)
                    break
                    
        self._save_history()
    
    def clear_history(self, keep_system_message: bool = True) -> None:
        """
        Clear the chat history.
        
        Args:
            keep_system_message: Whether to keep the system message
        """
        system_message = None
        if keep_system_message:
            system_message = next((msg for msg in self.history.messages if msg.role == "system"), None)
            
        self.history.clear()
        
        if system_message:
            self.history.messages.append(system_message)
            
        self._save_history()


def create_chat_session(
    system_message: str = "You are a helpful assistant.",
    history_file: Optional[str] = None,
    max_context_length: int = 2048
) -> ChatContextManager:
    """
    Create a new chat session with the specified parameters.
    
    Args:
        system_message: The system message to use
        history_file: Path to save/load chat history
        max_context_length: Maximum context length in tokens
        
    Returns:
        A new ChatContextManager instance
    """
    return ChatContextManager(
        system_message=system_message,
        history_file=history_file,
        max_context_length=max_context_length
    )